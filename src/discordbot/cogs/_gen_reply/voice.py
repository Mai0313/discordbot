"""Optional spoken-reply synthesis: turn a fierce QA reply into an attached voice clip.

The answer model decides per reply whether the message lands better said out loud
(a roast, a scolding, an excited outburst) by appending the `VOICE_MARKER` tag as the
final line of its reply. `ResponseStreamer` strips the tag from the visible text and,
when present, asks `VoiceSynthesizer` to render the same reply text to audio and edits it
onto the already-sent message. The decision lives inside the answer model on purpose so
the written reply and the spoken clip stay coherent (the model knows it is speaking); a
separate post-hoc classifier would not. Synthesis is best-effort: any failure leaves a
normal text reply.

The spoken delivery rides in `TTS_STYLE_DIRECTIVE` (it only fixes the voice gender and lets
the tone follow the reply's own wording), prepended to the input text because the proxy's
`instructions` parameter is silently ignored for this TTS model. `response_format` is
intentionally not sent (the proxy 500s on it); the model returns WAV, hence `reply.wav`.
"""

import re

from openai import AsyncOpenAI
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation

# Control token the answer model appends to request a spoken reply. Distinctive and
# tag-shaped so it never collides with natural reply text; stripped before display.
VOICE_MARKER = "</need-voice>"
# The bare tag body, tolerant of stray whitespace, an optional leading/trailing slash, and a
# hyphen the model may split with spaces, so a near-miss tag never leaks into the reply.
_MARKER_BODY = r"<\s*/?\s*need\s*-?\s*voice\s*/?\s*>"
# Strip the marker only at the very end of the reply (its intended position), absorbing the
# whitespace/backticks it sits on so a clean reply remains.
_TRAILING_VOICE_MARKER_RE = re.compile(rf"[\s`]*{_MARKER_BODY}[\s`]*\Z", re.IGNORECASE)
# Detect / scrub a marker anywhere, optionally backtick-wrapped. Used to flag voice and to
# remove a stray mid-reply marker WITHOUT eating the surrounding text (so words never join).
_ANY_VOICE_MARKER_RE = re.compile(rf"`?{_MARKER_BODY}`?", re.IGNORECASE)

# Tunable voice config (edit here). The style directive only fixes the voice gender and lets
# the spoken tone follow the reply's own wording (a heavy fixed tone sounds forced and
# distorts); it is prepended to the input (English on purpose: Gemini TTS style prompting is
# documented in English and is read as style, not spoken aloud).
TTS_MODEL_NAME = "gemini-3.1-flash-tts-preview"
TTS_VOICE = "Zephyr"
TTS_STYLE_DIRECTIVE = "Say the following in a male voice:"
TTS_SPEED = 1.3

# Bounds: cap spoken text so a long reply cannot balloon the WAV past Discord's upload limit.
# No explicit request timeout on purpose: synthesis runs after the text reply is already on
# screen and is per-message, so a slow clip only delays its own message, never others; the
# SDK's own timeout backstops a genuine hang.
VOICE_MAX_INPUT_CHARS = 400
VOICE_MAX_AUDIO_BYTES = 8 * 1024 * 1024


def strip_voice_marker(*, text: str) -> tuple[str, bool]:
    """Removes the voice marker from a finished reply, reporting whether it was present.

    Returns the text untouched when no marker is found so non-voice replies keep their exact
    content. The intended trailing marker is stripped with the whitespace it sat on; a stray
    marker elsewhere (the model misplaced it) is scrubbed in place so it never leaks, without
    collapsing the surrounding words into each other.
    """
    if not _ANY_VOICE_MARKER_RE.search(text):
        return text, False
    cleaned = _TRAILING_VOICE_MARKER_RE.sub("", text)
    cleaned = _ANY_VOICE_MARKER_RE.sub("", cleaned)
    return cleaned.rstrip(), True


def strip_partial_voice_marker(*, text: str) -> str:
    """Hides the marker (complete or still streaming in) from a live preview snapshot.

    A complete marker anywhere is removed; a tail that is a prefix of the marker (e.g.
    `</need-voi` mid-stream) is trimmed so the control token never flickers into the
    preview before the final strip.
    """
    cleaned = _ANY_VOICE_MARKER_RE.sub("", text)
    stripped = cleaned.rstrip()
    lowered = stripped.lower()
    for cut in range(len(VOICE_MARKER) - 1, 1, -1):
        fragment = VOICE_MARKER[:cut].lower()
        if lowered.endswith(fragment):
            return stripped[: -len(fragment)].rstrip()
    return stripped


class VoiceSynthesizer(BaseModel):
    """Best-effort text-to-speech for spoken replies through the LiteLLM proxy.

    Holds the shared async client plus the fixed voice / style / speed config; `synthesize`
    renders one reply to WAV bytes or returns None on over-length input, an oversized clip,
    or any provider error, so the caller always degrades to a plain text reply.
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
        description="Style directive prepended to the spoken text (fixes voice gender).",
    )
    speed: float = Field(default=TTS_SPEED, description="Playback speed passed to the TTS model.")

    async def synthesize(self, *, text: str, end_user_id: str) -> bytes | None:
        """Renders reply text to WAV bytes, or None when it should be skipped or failed."""
        spoken = text.strip()
        if not spoken or len(spoken) > VOICE_MAX_INPUT_CHARS:
            return None
        try:
            responses = await self.client.audio.speech.create(
                input=f"{self.style_directive}\n\n{spoken}",
                model=self.model_name,
                voice=self.voice,
                speed=self.speed,
                extra_headers={"x-litellm-end-user-id": end_user_id},
            )
            audio = await responses.aread()
        except Exception:
            # Best-effort: any provider error degrades to a plain text reply.
            logfire.warn("Voice synthesis failed; replying without audio")
            return None
        if len(audio) > VOICE_MAX_AUDIO_BYTES:
            logfire.warn(
                "Synthesized voice exceeds the Discord upload bound; dropping audio",
                audio_bytes=len(audio),
            )
            return None
        return audio
