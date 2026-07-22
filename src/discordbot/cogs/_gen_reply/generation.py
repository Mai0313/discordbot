"""Media-generation services: the image, voice, video, and music render calls behind one shape.

All runtime media generators are BaseModel services held as cog `cached_property`s, so every media
render goes through the same calling convention instead of a half-free-function / half-class mix:

- `PromptGenerator` is the upstream prompt director shared by the router IMAGE and VIDEO routes:
  `refine` expands a thin user request into one rich, self-contained generation prompt with the
  grounding tools (so a vague "draw the heroine of some anime" is looked up first), best-effort and
  gated per-route by `IMAGE_REFINE_PROMPT_ENABLED` / `VIDEO_REFINE_PROMPT_ENABLED`. It runs on the
  proxy like the answer model. The QA-route inline
  `<generate-image>` marker does NOT refine (its description is already written by the answer model).
- `ImageGenerator` runs the downstream image model on the LiteLLM proxy (`AsyncOpenAI`). `render`
  is the raising primitive shared by the router IMAGE route (which also edits source pixels) and
  the best-effort inline path; `generate` is the QA-route `<generate-image>` marker's best-effort wrapper
  (generation-only, timeout, None on any failure) so a slow inline render never blocks anything but
  its own reply.
- `VoiceGenerator` runs the text-to-speech model on the same LiteLLM proxy (`AsyncOpenAI`) as the
  image generator. Kept on the proxy on purpose: TTS has many interchangeable providers, so the
  one-SDK proxy path stays the most portable, unlike Veo / Lyria below which can only go direct.
  `generate` is best-effort but returns a `VoiceClip` carrying a `VoiceOutcome`
  (OK / EMPTY / TIMEOUT / ERROR) rather than a bare None, so the caller can hint a timeout (⏱️)
  apart from any other failure (⚠️). The `speechify_discord_markup` helper that prepares its spoken
  input lives alongside it.
- `VideoGenerator` runs the native-omni render behind the VIDEO route DIRECT to Google
  (`genai.Client`, the Interactions API is Gemini-only, not reachable via the proxy). One model
  (`interactions.create`) backs text / reference-image / source-video generation: a `source_video`
  is pinned to `task="edit"` while everything else omits the task so omni infers the mode
  (image_to_video / reference_to_video / text_to_video). `render` is the raising primitive for the
  VIDEO route; `generate` is its best-effort twin for the QA-route `<generate-video>` marker (None on
  any failure), mirroring `ImageGenerator`.
- `MusicGenerator` runs the native-Lyria render behind the QA-route `<generate-music>` marker via the
  Gemini Interactions API, also DIRECT to Google. Like `ImageGenerator.generate` it is best-effort
  only (`generate`, None on any failure), since music is inline-only.

Keeping them here means a future provider swap (or a move of a render off the proxy) changes
one place.
"""

from io import BytesIO
import re
from enum import StrEnum
import time
import base64
from typing import TYPE_CHECKING, Protocol, cast
import asyncio
from collections.abc import AsyncIterator

from google import genai
from openai import AsyncOpenAI, APITimeoutError
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from google.genai.types import FileState
from google.genai.interactions import (
    TextContentParam,
    VideoConfigParam,
    ImageContentParam,
    VideoContentParam,
    GenerationConfigParam,
    VideoResponseFormatParam,
)
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.llm import output_text_or_empty
from discordbot.utils.images import convert_base64_to_data_uri
from discordbot.typings.models import ModelSettings

if TYPE_CHECKING:
    from google.genai.interactions import ImageContentMimeType
    from openai.types.responses.response_input_file_param import ResponseInputFileParam

# Bound for the inline-image best-effort path: the render runs after the text reply is already
# on screen, so the wait only delays this message's own image, never others. Generous (mirrors
# VOICE_TIMEOUT_SECONDS) so a slower render still has room to land.
INLINE_IMAGE_TIMEOUT_SECONDS = 300.0

# Bound for the prompt-refinement call: it sits SERIALLY before the image/video render on the
# IMAGE/VIDEO critical path, so a hung director must not keep the route waiting forever. On
# timeout the refine falls back to the raw user prompt like any other failure.
PROMPT_REFINE_TIMEOUT_SECONDS = 120.0

# Hard ceiling on the whole omni video render (a single blocking interactions.create) so a hung
# provider job cannot leave the message handler waiting forever. Co-located with the image timeout
# since it is a property of the render, not of the route that calls it.
VIDEO_RENDER_TIMEOUT_SECONDS = 600.0

# Bound for waiting on a Files API entry to become usable: the source video uploaded for an omni
# edit (polled to ACTIVE) and the URI-delivered generated clip (download retried until it lands).
# Generous because a large clip can sit in PROCESSING a while; the render hard-fails past it, since
# video is the primary deliverable.
FILES_READY_TIMEOUT_SECONDS = 180.0

# omni accepts a handful of subject reference images. Shared so the VIDEO route caps the frames it
# grounds the prompt director on to exactly the set render will send, rather than letting the
# director describe references omni never receives (and uploading those unused bytes on the path).
MAX_VIDEO_REFERENCE_IMAGES = 3

# Bound for the inline-music best-effort path, mirroring the inline-image timeout: the render
# runs after the text reply is on screen, so the wait only delays this message's own clip.
MUSIC_RENDER_TIMEOUT_SECONDS = 300.0

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

# Fixed musical-style directive sent as the Lyria `system_instruction`. English on purpose (the
# Lyria prompt surface is documented in English). Lyria picks the lyric language from the prompt
# (docs: "generates lyrics in the language of your prompt"), so the language is defaulted here too,
# not just the genre. The QA `<generate-music>` marker prompt already steers the answer model to default to
# this style/language and write it into the description, so this is a backstop that still honors a
# description asking for a different genre or language. A 2026-06 Interactions spike produced
# Japanese vocals with this Japanese steer, but the load-bearing path stays the prompt-side default.
MUSIC_STYLE_DIRECTIVE = (
    "Compose in a Japanese anime / J-pop style with Japanese-language vocals by default; if the "
    "description clearly asks for a different genre, style, or lyric language, or for an "
    "instrumental, follow the description instead."
)

# Map a returned audio mime type to a Discord-playable file extension. Discord's inline audio
# player keys off the extension, and `AudioContent.mime_type` can be a non-obvious value
# (`audio/mpeg`, `audio/l16`) or None, so a naive `split("/")[-1]` would yield an unplayable
# name; fall back to `.mp3` for anything unmapped.
_AUDIO_MIME_EXTENSIONS = {
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/l16": ".wav",
    "audio/m4a": ".m4a",
    "audio/aac": ".m4a",
    "audio/ogg": ".ogg",
    "audio/opus": ".ogg",
    "audio/flac": ".flac",
    "audio/aiff": ".aiff",
}


def music_filename(*, mime_type: str | None) -> str:
    """The Discord attachment filename for a generated music clip, by its audio mime type."""
    extension = _AUDIO_MIME_EXTENSIONS.get((mime_type or "").lower(), ".mp3")
    return f"music{extension}"


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


class ImageGenerator(BaseModel):
    """Image render shared by the router IMAGE route and the QA-route `<generate-image>` marker.

    Holds the shared client and the image model. `render` is the raising primitive (edits when
    source bytes are present, else generates); `generate` is the best-effort inline wrapper that
    returns None on any failure or timeout, mirroring how `VoiceGenerator` is gated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client used for the image render."
    )
    image_model: ModelSettings = Field(
        ..., description="Model settings for image generation and editing."
    )

    async def render(
        self, *, prompt: str, end_user_id: str, image_bytes_list: list[bytes] | None = None
    ) -> bytes:
        """Renders one image to PNG bytes, editing source bytes when present, else generating fresh.

        Retries once on an empty payload (a transient hiccup occasionally returns no image) before
        raising, so a flaky empty result does not surface as a user-facing error; a genuine safety
        block returns empty on both attempts and still raises so callers can degrade. The image
        model is dispatched on the proxy with the same parameters for the edit and generate paths.
        """

        async def _dispatch() -> str | None:
            if image_bytes_list:
                result = await self.client.images.edit(
                    image=image_bytes_list,
                    prompt=prompt or "請依照附件內容進行編輯或優化。",
                    model=self.image_model.name,
                    n=1,
                    response_format="b64_json",
                    quality="auto",
                    size="auto",
                    extra_headers={"x-litellm-end-user-id": end_user_id},
                )
            else:
                result = await self.client.images.generate(
                    prompt=prompt,
                    model=self.image_model.name,
                    n=1,
                    response_format="b64_json",
                    quality="auto",
                    size="auto",
                    extra_headers={"x-litellm-end-user-id": end_user_id},
                )
            return result.data[0].b64_json if result.data else None

        for attempt in range(2):
            image_b64 = await _dispatch()
            if image_b64 is not None:
                return base64.b64decode(image_b64)
            if attempt == 0:
                logfire.info(
                    "Image operation returned an empty result; retrying once",
                    model=self.image_model.name,
                    edit=bool(image_bytes_list),
                )
        raise ValueError("Image operation returned no image data after one retry")

    async def generate(
        self, *, user_prompt: str, end_user_id: str, image_bytes_list: list[bytes] | None = None
    ) -> bytes | None:
        """Renders one image from the description; None on any failure or timeout.

        Best-effort wrapper around `render` for the QA-route `<generate-image>` marker, inside a generous
        timeout, returning None to disable the inline path for a reply rather than raising into the
        streamer's path. When `image_bytes_list` is supplied (the user uploaded image(s) the answer
        model is illustrating), it rides through to `render` as edit source pixels, so an inline
        `<generate-image>` over an attached photo edits it instead of generating a fresh one.
        """
        started = time.monotonic()
        try:
            async with asyncio.timeout(delay=INLINE_IMAGE_TIMEOUT_SECONDS):
                image = await self.render(
                    prompt=user_prompt, end_user_id=end_user_id, image_bytes_list=image_bytes_list
                )
        except Exception as exc:
            # Broad on purpose: the inline-marker boundary must degrade to "reply without an
            # image" whether the render timed out, came back empty, or the provider errored.
            logfire.warn(
                "Inline image generation failed; replying without an image",
                error_type=type(exc).__name__,
                end_user_id=end_user_id,
                edit=bool(image_bytes_list),
                _exc_info=exc,
            )
            return None
        logfire.info(
            "gen_reply inline image generated",
            elapsed_seconds=time.monotonic() - started,
            image_bytes=len(image),
        )
        return image


class PromptGenerator(BaseModel):
    """Prompt director for the router IMAGE and VIDEO routes, running on the LiteLLM proxy.

    Holds the shared proxy client and the director model; each `refine` call is gated per-route
    by the caller's own flag (`IMAGE_REFINE_PROMPT_ENABLED` / `VIDEO_REFINE_PROMPT_ENABLED`).
    `refine` expands a thin user request ("draw the heroine of some anime") into one rich,
    self-contained generation prompt, looking subjects up first with the grounding tools, so the
    downstream image/video model renders a far stronger result than from the raw request. Any
    already-loaded source bytes ride along as input images so an edit prompt is grounded in the
    actual picture without a re-download.

    Best-effort by construction: a disabled flag, an empty draft, a timeout, or ANY error all
    fall back to the raw `user_prompt`, so a director failure never aborts generation and callers
    can treat `refine` as a pure prompt-in / prompt-out step. The QA-route inline `<generate-image>` marker
    does NOT use this: that description is already authored by the answer model with grounding.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client used for the refinement call."
    )
    prompt_model: ModelSettings = Field(
        ..., description="Model settings for the prompt director (flash + high + grounding)."
    )

    async def refine(
        self,
        *,
        user_prompt: str,
        instructions: str,
        end_user_id: str,
        enabled: bool,
        image_bytes_list: list[bytes] | None = None,
    ) -> str:
        """Expands a thin IMAGE/VIDEO request into a rich, self-contained generation prompt.

        Runs `prompt_model` with the grounding tools so a vague request is looked up and resolved
        before the image/video model renders it. With `enabled=False` the director is skipped and
        the raw `user_prompt` is returned; an empty draft, a timeout, or any error fall back the
        same way, so an exception never escapes here. The caller passes its own per-route flag
        (`image_refine_prompt_enabled` / `video_refine_prompt_enabled`).
        """
        if not enabled:
            return user_prompt
        director_content: list[
            ResponseInputTextParam | ResponseInputImageParam | ResponseInputFileParam
        ] = [
            ResponseInputTextParam(
                text=f"User generation request:\n{user_prompt}", type="input_text"
            )
        ]
        for image_bytes in image_bytes_list or []:
            director_content.append(
                ResponseInputImageParam(
                    image_url=convert_base64_to_data_uri(
                        base64_image=base64.b64encode(image_bytes).decode()
                    ),
                    detail="auto",
                    type="input_image",
                )
            )
        director_input: list[EasyInputMessageParam] = [
            EasyInputMessageParam(role="user", content=director_content)
        ]
        started = time.monotonic()
        try:
            async with asyncio.timeout(delay=PROMPT_REFINE_TIMEOUT_SECONDS):
                with logfire.span("gen_reply prompt refine", model=self.prompt_model.name):
                    responses = await self.client.responses.create(
                        model=self.prompt_model.name,
                        instructions=instructions,
                        input=cast("ResponseInputParam", director_input),
                        reasoning=self.prompt_model.reasoning,
                        tools=list(self.prompt_model.tools),
                        service_tier="auto",
                        extra_headers={"x-litellm-end-user-id": end_user_id},
                        extra_body={"mock_testing_fallbacks": False},
                    )
            refined = output_text_or_empty(responses=responses).strip()
        except Exception as exc:
            logfire.warn(
                "Prompt refinement failed; using raw user prompt",
                error_type=type(exc).__name__,
                end_user_id=end_user_id,
                _exc_info=exc,
            )
            return user_prompt
        logfire.info(
            "gen_reply prompt refine done",
            elapsed_seconds=time.monotonic() - started,
            refined=bool(refined),
        )
        return refined or user_prompt


class VoiceGenerator(BaseModel):
    """Best-effort text-to-speech for spoken replies through the LiteLLM proxy.

    Holds the shared async client plus the fixed voice / style / speed config; `generate`
    renders one reply to a `VoiceClip` carrying the WAV bytes (when produced) plus an outcome
    (OK / EMPTY / TIMEOUT / ERROR), so the caller both degrades to a text reply and can hint
    why the clip is missing (a timeout vs. any other provider error, e.g. a policy refusal).

    The spoken delivery rides in `TTS_STYLE_DIRECTIVE` (it fixes the voice age/gender and lets
    the tone follow the reply's own wording), prepended to the input text because the proxy's
    `instructions` parameter is silently ignored for this TTS model. `response_format` is
    intentionally not sent (the proxy 500s on it); the model returns WAV, hence `reply.wav`.
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

    async def generate(self, *, text: str, end_user_id: str) -> VoiceClip:
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
        except Exception as exc:
            # Any other provider error (most often the clip was refused, e.g. policy), and broad
            # because the proxy can fail in any shape. The caller marks the message with a warning
            # hint and degrades to a text reply; error_type separates a refusal from a proxy 500.
            logfire.warn(
                "Voice synthesis failed; replying without audio",
                model=self.model_name,
                end_user_id=end_user_id,
                text_chars=len(spoken),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return VoiceClip(outcome=VoiceOutcome.ERROR)


class VideoGenerator(BaseModel):
    """Native Gemini (omni) video render behind the VIDEO route and the QA-route `<generate-video>` marker.

    Holds the direct-to-Google client and the video model. `render` (raising) backs the VIDEO
    route, where video is always the primary deliverable; `generate` is its best-effort twin
    (None on any failure) for the inline `<generate-video>` marker, mirroring `ImageGenerator`'s render /
    generate split so a slow or refused inline clip never blocks anything but its own reply. Runs
    on the Gemini Interactions API (`interactions.create`), which unifies text / reference-image /
    source-video generation on one model, so the same call backs plain generation and true
    source-video editing (`task="edit"`).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[genai.Client] = Field(
        ...,
        description="Direct-to-Google Gemini client (the Interactions API is Gemini-only, no proxy).",
    )
    video_model: ModelSettings = Field(
        ..., description="Model settings for native Gemini (omni) video generation."
    )

    async def render(
        self,
        *,
        prompt: str,
        reference_image_sources: list[tuple[bytes, str]],
        source_video: tuple[bytes, str] | None = None,
    ) -> bytes:
        """Renders one video to MP4 bytes via the native Gemini (omni) Interactions API; raises.

        The single downstream video render, so the VIDEO route shares one implementation and a
        future provider swap changes one place. Goes DIRECT to Google (the Interactions API is
        Gemini-only). A `source_video` is uploaded to the Files API and edited in place with an
        explicit `task="edit"`; otherwise the task is omitted and omni infers image_to_video /
        reference_to_video / text_to_video from the prompt plus any input images (up to
        `MAX_VIDEO_REFERENCE_IMAGES`, each carrying its real mime type — omni 400s an image content
        block whose mime is empty, "Unsupported MIME type: "). 16:9 is sent only for pure text (an
        edit keeps the source clip's ratio, and an image request may become image_to_video, which
        follows the source frame's ratio, so no aspect ratio is sent there); `delivery="uri"` so the
        clip comes back as a Files URI (no base64 bloat) and is downloaded with a bounded retry
        (`_download_output_video`):
        the file can still be finalizing when the interaction reports `completed`, so a larger clip's
        first download may fail and is retried until it lands. Duration is
        left to omni's default. Raises `RuntimeError` when the interaction is not `completed` or
        carries no video, folding in `status` + `output_text` (omni signals a soft refusal /
        incomplete / budget_exceeded that way; the `Interaction` has no `error` / `rai_*` field).
        """
        text = prompt or "請依照訊息內容生成一段影片。"
        content: list[TextContentParam | ImageContentParam | VideoContentParam]
        # A source-video edit is the one task we still pin: with task omitted, omni infers the mode
        # (image_to_video vs reference_to_video vs text_to_video) from the prompt + input media,
        # which follows the #317 "hand the raw request over and let the model decide" direction and
        # covers image_to_video without a brittle image-count heuristic. If a future deployment finds
        # the inferred mode underwhelming (e.g. a lone image not animated in place), pin it back here.
        generation_config: GenerationConfigParam | None = None
        task_label: str
        # 16:9 is the generation default we keep for pure text; it is omitted when images are present
        # (omni may pick image_to_video, which follows the source frame's ratio and can reject an
        # aspect_ratio the way an edit does) and on an edit (omni 400s it, "cannot be set in response
        # format for edit task", since an edit keeps the source clip's ratio).
        response_format = VideoResponseFormatParam(type="video", delivery="uri")
        if source_video is not None:
            file_uri = await self._upload_source_video(source_video=source_video)
            content = [
                VideoContentParam(type="video", uri=file_uri),
                TextContentParam(type="text", text=text),
            ]
            generation_config = GenerationConfigParam(video_config=VideoConfigParam(task="edit"))
            task_label = "edit"
        elif reference_image_sources:
            content = [TextContentParam(type="text", text=text)]
            # Each image MUST carry its real mime type: omni rejects an image content block with an
            # empty mime ("Unsupported MIME type: "). The mime already rides in the source tuple.
            content.extend(
                ImageContentParam(
                    type="image",
                    data=base64.b64encode(raw).decode(),
                    mime_type=cast("ImageContentMimeType", mime),
                )
                for raw, mime in reference_image_sources[:MAX_VIDEO_REFERENCE_IMAGES]
            )
            task_label = "infer_image"
        else:
            content = [TextContentParam(type="text", text=text)]
            task_label = "infer_text"
            response_format["aspect_ratio"] = "16:9"

        started = time.monotonic()
        async with asyncio.timeout(delay=VIDEO_RENDER_TIMEOUT_SECONDS):
            interaction = await self.client.aio.interactions.create(
                model=self.video_model.name,
                input=content,
                response_format=response_format,
                generation_config=generation_config,
                timeout=VIDEO_RENDER_TIMEOUT_SECONDS,
            )
        # No `stream=True`, so this is the interaction rather than an event stream. Narrowed by
        # excluding the stream instead of casting to `google.genai.interactions.Interaction`:
        # that module star-imports the name from both `triggers` (an alias for the union of the
        # two *request* models, which carry no `output_video` / `status` / `output_text`) and
        # `types.interactions` (the response class), and only the runtime import order makes the
        # response class win, so the cast named the request union and typed the wrong object.
        if isinstance(interaction, AsyncIterator):
            raise RuntimeError("Video generation returned an event stream, not an interaction")
        video = interaction.output_video
        if interaction.status != "completed" or video is None or video.uri is None:
            logfire.warn(
                "gen_reply video generation failed",
                status=str(interaction.status),
                task=task_label,
                note=interaction.output_text,
            )
            raise RuntimeError(
                f"Video generation failed: status={interaction.status} note={interaction.output_text!r}"
            )
        logfire.debug(
            "gen_reply video job done", task=task_label, render_seconds=time.monotonic() - started
        )
        return await self._download_output_video(uri=video.uri)

    async def generate(
        self, *, user_prompt: str, reference_image_sources: list[tuple[bytes, str]] | None = None
    ) -> bytes | None:
        """Renders one inline `<generate-video>` clip to MP4 bytes; None on any failure or timeout.

        Best-effort twin of `render` for the QA-route `<generate-video>` marker: it returns None (disabling
        the inline video for this reply) instead of raising into the streamer's single media-attach
        gather, so a slow / refused clip never aborts the ready voice / music / images alongside it.
        When the user attached image(s) they ride along as `(bytes, mime)` pairs and omni infers the
        task (image_to_video / reference_to_video); otherwise it is plain text. `render` already
        bounds itself with `VIDEO_RENDER_TIMEOUT_SECONDS`, so no extra timeout is needed here.
        """
        try:
            return await self.render(
                prompt=user_prompt, reference_image_sources=reference_image_sources or []
            )
        except Exception:
            logfire.warn("Inline video generation failed; replying without video", _exc_info=True)
            return None

    async def _download_output_video(self, *, uri: str) -> bytes:
        """Downloads a URI-delivered clip, retrying while its Files entry finalizes; raises past the bound.

        With `delivery="uri"` the interaction reports `completed` while `files.get` can still show
        PROCESSING for a moment (observed even when the download already succeeds), so rather than
        poll the lagging state we retry the download itself — it returns the whole clip as soon as
        the file is servable and adds no wait when the first attempt already works. Bounded by
        `FILES_READY_TIMEOUT_SECONDS` so a stuck file hard-fails diagnosably instead of hanging,
        since video is the primary deliverable.
        """
        deadline = time.monotonic() + FILES_READY_TIMEOUT_SECONDS
        attempt = 0
        while True:
            try:
                return await self.client.aio.files.download(file=uri)
            except Exception as exc:
                if time.monotonic() >= deadline:
                    raise
                attempt += 1
                logfire.debug(
                    "gen_reply video download not ready; retrying",
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
                await asyncio.sleep(2.0)

    async def _upload_source_video(self, *, source_video: tuple[bytes, str]) -> str:
        """Uploads a source clip to the Files API and returns its ACTIVE uri; raises on failure.

        The edit path feeds the actual clip (not a poster frame), so the video IS the primary
        deliverable and a failed upload must hard-fail with a diagnosable error rather than pass a
        None uri into `VideoContentParam`. The activation bound is generous
        (`FILES_READY_TIMEOUT_SECONDS`) because a raw clip is far larger than an image
        and can sit in PROCESSING longer than the reply-upload's window.
        """
        data, mime = source_video
        uploaded = await self.client.aio.files.upload(
            file=BytesIO(data), config={"mime_type": mime, "display_name": "source.mp4"}
        )
        file_name = uploaded.name
        if file_name is None:
            raise RuntimeError("Source video upload returned no file name")
        deadline = time.monotonic() + FILES_READY_TIMEOUT_SECONDS
        while uploaded.state == FileState.PROCESSING:
            if time.monotonic() >= deadline:
                raise RuntimeError("Source video did not become ACTIVE before the deadline")
            await asyncio.sleep(1.0)
            uploaded = await self.client.aio.files.get(name=file_name)
        if uploaded.state != FileState.ACTIVE or uploaded.uri is None:
            raise RuntimeError(f"Source video upload failed: state={uploaded.state}")
        return uploaded.uri


class MusicClip(BaseModel):
    """A generated music clip: the audio bytes plus the mime type used to pick a file extension."""

    audio: bytes = Field(..., description="Rendered audio bytes for the music clip.")
    mime_type: str = Field(
        ..., description="Audio mime type Lyria reported, e.g. audio/mp3, used to name the file."
    )


class MusicGenerator(BaseModel):
    """Native-Lyria music render behind the QA-route `<generate-music>` marker.

    Holds the direct-to-Google client and the music model. Best-effort only (`generate`, None on
    any failure), mirroring `ImageGenerator.generate`: music is inline-only, so a slow or refused
    render never blocks anything but its own reply. Goes DIRECT to Google via the Gemini
    Interactions API (the music model is dispatched there, not via the proxy).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[genai.Client] = Field(
        ..., description="Direct-to-Google Gemini client for the Lyria Interactions call."
    )
    music_model: ModelSettings = Field(
        ..., description="Model settings for native Gemini (Lyria) music generation."
    )

    async def generate(self, *, user_prompt: str) -> MusicClip | None:
        """Renders one music clip from the description; None on any failure or timeout.

        Best-effort wrapper for the QA-route `<generate-music>` marker: one non-streaming Interactions
        call inside a generous timeout, returning None to disable the inline path for this reply
        rather than raising into the streamer. The fixed anime/J-pop style rides in
        `system_instruction`; the description is passed as the plain-string `input`. The returned
        audio mime type is carried back so the caller can pick a Discord-playable extension.
        """
        started = time.monotonic()
        # Output extraction and base64 decode stay inside the guard: an unexpected audio shape
        # (a raising `output_audio`, non-base64 `data`) must drop the clip to None like any other
        # failure, never escape into the streamer's single media-attach gather and abort the
        # already-ready voice / images alongside it.
        try:
            async with asyncio.timeout(delay=MUSIC_RENDER_TIMEOUT_SECONDS):
                interaction = await self.client.aio.interactions.create(
                    model=self.music_model.name,
                    input=user_prompt,
                    system_instruction=MUSIC_STYLE_DIRECTIVE,
                )
            audio = interaction.output_audio
            if audio is None or not audio.data:
                logfire.warn("Inline music generation returned no audio; replying without music")
                return None
            clip = base64.b64decode(audio.data)
        except Exception as exc:
            logfire.warn(
                "Inline music generation failed; replying without music",
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        logfire.info(
            "gen_reply inline music generated",
            elapsed_seconds=time.monotonic() - started,
            music_bytes=len(clip),
            mime_type=audio.mime_type,
        )
        return MusicClip(audio=clip, mime_type=audio.mime_type or "audio/mp3")
