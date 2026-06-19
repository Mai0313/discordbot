"""Optional spoken-reply synthesis: turn a chosen reply segment into an attached voice clip.

The answer model decides per reply which part lands better said out loud (a roast, a punch
line, an excited outburst) by wrapping that segment in `<voice>...</voice>` (see `markers.py`).
`ResponseStreamer` extracts the wrapped segment, keeps it in the visible text, and asks
`VoiceSynthesizer` to render just that segment to audio, editing it onto the already-sent
message. The decision lives inside the answer model on purpose so the written reply and the
spoken clip stay coherent (the model knows it is speaking); a separate post-hoc classifier
would not. Synthesis is best-effort: any failure leaves a normal text reply.

The spoken delivery rides in `TTS_STYLE_DIRECTIVE` (it fixes the voice age/gender and lets
the tone follow the reply's own wording), prepended to the input text because the proxy's
`instructions` parameter is silently ignored for this TTS model. `response_format` is
intentionally not sent (the proxy 500s on it); the model returns WAV, hence `reply.wav`.
"""

import re
from enum import StrEnum
from typing import Protocol

from openai import AsyncOpenAI, APITimeoutError
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation

# Tunable voice config (edit here). The style directive fixes the voice age/gender and lets
# the spoken tone follow the reply's own wording (a heavy fixed tone sounds forced and
# distorts); it is prepended to the input (English on purpose: Gemini TTS style prompting is
# documented in English and is read as style, not spoken aloud).
TTS_MODEL_NAME = "gemini-3.1-flash-tts-preview"
TTS_VOICE = "Despina"
TTS_STYLE_DIRECTIVE = "Using a natural 18-year-old woman's voice that fits the following text:"
TTS_SPEED = 1.5

# Filename of the attached voice clip. Shared so input rendering can recognise and skip the
# bot's own clip when it later appears in history, instead of re-uploading it as self-input.
VOICE_REPLY_FILENAME = "reply.wav"

# Bound: a request timeout so a slow/hung clip cannot keep this message's own pipeline (its final
# status reaction + memory scheduling) waiting. The synthesis is per-message and runs after the text is
# already on screen, so the wait only delays its own message, never others; it is generous so a
# longer spoken reply has room to render. There is deliberately no spoken-length cap: the answer
# model decides how much to say. The upload-size guard lives at the attach site (`streaming.py`),
# where the guild's real `filesize_limit` is known, not as a hardcoded byte ceiling here.
VOICE_TIMEOUT_SECONDS = 300.0


# Discord-specific markup the answer model may embed in a reply: user/role/channel mentions
# (`<@id>` / `<@!id>` / `<@&id>` / `<#id>`), custom emoji (`<:name:id>` / `<a:name:id>`),
# timestamps (`<t:unix:style>`), and slash-command references (`</cmd:id>`). Read aloud
# verbatim these are a bare snowflake or a colon-wrapped name, so they are rewritten to a
# spoken form before synthesis; the visible text reply keeps the real markup so Discord still
# renders it.
_MENTION_RE = re.compile(r"<(?:@[!&]?|#)(\d+)>")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:\w+:\d+>")
_TIMESTAMP_RE = re.compile(r"<t:-?\d+(?::[tTdDfFR])?>")
_SLASH_COMMAND_RE = re.compile(r"</([\w ]+):\d+>")
_COLLAPSE_SPACES_RE = re.compile(r"[ \t]{2,}")


class MentionNameResolver(Protocol):
    """Resolves a Discord snowflake (member/role/channel) to a name for the spoken reply."""

    def __call__(self, *, target_id: int) -> str | None: ...


def speechify_discord_markup(*, text: str, resolve_name: MentionNameResolver) -> str:
    """Rewrites Discord markup into plain spoken text before TTS synthesis.

    A mention becomes the resolved member/role/channel name (or is dropped when it cannot be
    resolved), custom emoji and timestamp tags are dropped, and a slash-command reference keeps
    only its command words. Only the spoken-clip input is cleaned, so the model's `<@id>`-style
    markup is never read aloud as a raw snowflake while the visible reply stays untouched.
    """

    def _named(match: re.Match[str]) -> str:
        return resolve_name(target_id=int(match.group(1))) or ""

    cleaned = _MENTION_RE.sub(_named, text)
    cleaned = _CUSTOM_EMOJI_RE.sub("", cleaned)
    cleaned = _TIMESTAMP_RE.sub("", cleaned)
    cleaned = _SLASH_COMMAND_RE.sub(r"\1", cleaned)
    cleaned = _COLLAPSE_SPACES_RE.sub(" ", cleaned)
    return cleaned.strip()


class VoiceOutcome(StrEnum):
    """Why a spoken-clip synthesis attempt ended, so the caller can hint appropriately."""

    OK = "ok"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"


class VoiceClip(BaseModel):
    """Result of one synthesis attempt: the audio (when produced) plus why it ended.

    A failed attempt carries `audio=None` and a non-OK `outcome`; the caller always degrades
    to a plain text reply and uses `outcome` to decide its best-effort failure hint.
    """

    audio: bytes | None = Field(
        default=None, description="Rendered WAV bytes, or None when no clip was produced."
    )
    outcome: VoiceOutcome = Field(
        ..., description="Why synthesis ended; drives the caller's best-effort failure hint."
    )


class VoiceSynthesizer(BaseModel):
    """Best-effort text-to-speech for spoken replies through the LiteLLM proxy.

    Holds the shared async client plus the fixed voice / style / speed config; `synthesize`
    renders one reply to a `VoiceClip` carrying the WAV bytes (when produced) plus an outcome
    (OK / EMPTY / TIMEOUT / ERROR), so the caller both degrades to a text reply and can hint
    why the clip is missing (a timeout vs. any other provider error, e.g. a policy refusal).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client used for the audio.speech call."
    )
    model_name: str = Field(
        default=TTS_MODEL_NAME, description="TTS model string dispatched on the proxy."
    )
    voice: str = Field(
        default=TTS_VOICE, description="Fixed voice timbre name for spoken replies."
    )
    style_directive: str = Field(
        default=TTS_STYLE_DIRECTIVE,
        description="Style directive prepended to the spoken text (fixes voice age/gender).",
    )
    speed: float = Field(default=TTS_SPEED, description="Playback speed passed to the TTS model.")

    async def synthesize(self, *, text: str, end_user_id: str) -> VoiceClip:
        """Renders reply text to a VoiceClip, reporting why it ended for best-effort hinting."""
        spoken = text.strip()
        if not spoken:
            logfire.info(
                "Voice synthesis skipped: reply text was empty after stripping",
                end_user_id=end_user_id,
            )
            return VoiceClip(outcome=VoiceOutcome.EMPTY)
        try:
            responses = await self.client.audio.speech.create(
                input=f"{self.style_directive}\n\n{spoken}",
                model=self.model_name,
                voice=self.voice,
                speed=self.speed,
                extra_headers={"x-litellm-end-user-id": end_user_id},
                timeout=VOICE_TIMEOUT_SECONDS,
            )
            audio = await responses.aread()
            logfire.debug(
                "Voice synthesis succeeded",
                model=self.model_name,
                speed=self.speed,
                end_user_id=end_user_id,
                text_chars=len(spoken),
                audio_bytes=len(audio),
            )
            return VoiceClip(audio=audio, outcome=VoiceOutcome.OK)
        except APITimeoutError:
            # The clip took longer than VOICE_TIMEOUT_SECONDS to render. The caller marks the
            # message with a timeout hint and still leaves a plain text reply.
            logfire.warn(
                "Voice synthesis timed out; replying without audio",
                model=self.model_name,
                end_user_id=end_user_id,
                text_chars=len(spoken),
                _exc_info=True,
            )
            return VoiceClip(outcome=VoiceOutcome.TIMEOUT)
        except Exception:
            # Any other provider error (most often the clip was refused, e.g. policy). The
            # caller marks the message with a warning hint and degrades to a text reply.
            logfire.warn(
                "Voice synthesis failed; replying without audio",
                model=self.model_name,
                end_user_id=end_user_id,
                text_chars=len(spoken),
                _exc_info=True,
            )
            return VoiceClip(outcome=VoiceOutcome.ERROR)
