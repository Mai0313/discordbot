"""Media-generation services: the prompt director and the image, voice, video and music renders.

Each is a BaseModel service held as a `ReplyToolkit` `cached_property`, so every media render
goes through one calling convention, and a provider swap (or a move of a render off the proxy)
changes one place.
"""

import re
import time
import base64
from typing import TYPE_CHECKING, Literal, Protocol, cast
import asyncio

from google import genai
from openai import AsyncOpenAI, APITimeoutError
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
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
from discordbot.utils.images import to_data_uri
from discordbot.typings.media import LoadedMedia
from discordbot.typings.models import ModelSettings
from discordbot.typings.timeouts import (
    VOICE_TIMEOUT_SECONDS,
    FILES_READY_TIMEOUT_SECONDS,
    INLINE_IMAGE_TIMEOUT_SECONDS,
    MUSIC_RENDER_TIMEOUT_SECONDS,
    VIDEO_RENDER_TIMEOUT_SECONDS,
    PROMPT_REFINE_TIMEOUT_SECONDS,
)
from discordbot.typings.context_budgets import MAX_VIDEO_REFERENCE_IMAGES
from discordbot.cogs.gen_reply.files_api import upload_until_active

if TYPE_CHECKING:
    from google.genai.interactions import ImageContentMimeType
    from openai.types.responses.response_input_file_param import ResponseInputFileParam

# Tunable voice config (edit here). The style directive fixes the voice age/gender and lets
# the spoken tone follow the reply's own wording (a heavy fixed tone sounds forced and
# distorts); it is prepended to the input (English on purpose: Gemini TTS style prompting is
# documented in English and is read as style, not spoken aloud).
TTS_VOICE = "Despina"
TTS_STYLE_DIRECTIVE = "Using a natural 18-year-old woman's voice that fits the following text:"
TTS_SPEED = 1.5

# Filename of the attached voice clip. Shared so input rendering can recognise and skip the
# bot's own clip when it later appears in history, instead of re-uploading it as self-input.
VOICE_REPLY_FILENAME = "reply.wav"

# Default filename for a generated image; multiple images fall back to `generated_<n>.png`.
INLINE_IMAGE_FILENAME = "generated.png"

# Filename for a generated video attachment; omni returns MP4, which Discord inline-plays.
INLINE_VIDEO_FILENAME = "generated.mp4"

# Fixed musical-style directive sent as the Lyria `system_instruction` (in English per Lyria's
# prompt surface), defaulting both genre and vocal language to Japanese anime / J-pop unless the
# description asks for a different genre, style, lyric language, or instrumental.
MUSIC_STYLE_DIRECTIVE = (
    "Compose in a Japanese anime / J-pop style with Japanese-language vocals by default; if the "
    "description clearly asks for a different genre, style, or lyric language, or for an "
    "instrumental, follow the description instead."
)


def music_filename(mime_type: str | None) -> str:
    """Returns the Discord attachment filename for a generated music clip from `mime_type`.

    Args:
        mime_type (str | None): Audio MIME type reported by the model, or None.
    """
    # Discord's inline audio player keys off the extension, and `AudioContent.mime_type` can be a
    # non-obvious value (`audio/mpeg`, `audio/l16`) or None, so a naive `split("/")[-1]` would
    # yield an unplayable name; anything unmapped falls back to `.mp3`.
    audio_mime_extensions = {
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
    extension = audio_mime_extensions.get((mime_type or "").lower(), ".mp3")
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

    def __call__(self, target_id: int) -> str | None:
        """Returns the spoken name for `target_id`, or None when unresolved.

        Args:
            target_id (int): Discord snowflake id to resolve.
        """
        ...


def speechify_discord_markup(text: str, resolve_name: MentionNameResolver) -> str:
    """Returns `text` with Discord markup rewritten into plain spoken text for TTS synthesis.

    Replaces mentions with `resolve_name` (or drops unresolved ones), strips custom emoji and
    timestamp tags, and keeps only the command words of slash-command references.

    Args:
        text (str): Reply text that may contain Discord markup.
        resolve_name (MentionNameResolver): Resolver for member, role, and channel ids.
    """

    def _named(match: re.Match[str]) -> str:
        return resolve_name(target_id=int(match.group(1))) or ""

    cleaned = _MENTION_RE.sub(_named, text)
    cleaned = _CUSTOM_EMOJI_RE.sub("", cleaned)
    cleaned = _TIMESTAMP_RE.sub("", cleaned)
    cleaned = _SLASH_COMMAND_RE.sub(r"\1", cleaned)
    cleaned = _COLLAPSE_SPACES_RE.sub(" ", cleaned)
    return cleaned.strip()


# Why a spoken-clip synthesis attempt ended, so the caller can hint appropriately.
VoiceOutcome = Literal["ok", "empty", "timeout", "error"]


class VoiceClip(BaseModel):
    """Result of one synthesis attempt: the audio (when produced) plus why it ended.

    A failed attempt carries `audio=None` and a non-`"ok"` `outcome`; the caller always degrades
    to a plain text reply and uses `outcome` to decide its best-effort failure hint.
    """

    audio: bytes | None = Field(
        default=None, description="Rendered WAV bytes, or None when no clip was produced."
    )
    outcome: VoiceOutcome = Field(
        ..., description="Why synthesis ended; drives the caller's best-effort failure hint."
    )


class ImageGenerator(BaseModel):
    """Image renderer through the LiteLLM proxy, in raising (`render`) and best-effort (`generate`) forms."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client used for the image render."
    )
    image_model: ModelSettings = Field(
        ..., description="Model settings for image generation and editing."
    )

    async def render(
        self, prompt: str, end_user_id: str, image_bytes_list: list[bytes] | None = None
    ) -> bytes:
        """Returns PNG bytes rendered from `prompt`, editing `image_bytes_list` when provided.

        Retries once on an empty response payload before raising `ValueError`.

        Args:
            prompt (str): Generation or editing prompt.
            end_user_id (str): Discord username sent as the LiteLLM end-user header.
            image_bytes_list (list[bytes] | None): Source image bytes to edit, or None to
                generate a fresh image.

        Raises:
            ValueError: Both attempts returned no image data.
        """

        async def _dispatch() -> str | None:
            if image_bytes_list:
                responses = await self.client.images.edit(
                    image=image_bytes_list,
                    prompt=prompt or "Edit or refine according to the attached content.",
                    model=self.image_model.name,
                    n=1,
                    response_format="b64_json",
                    quality="auto",
                    size="auto",
                    extra_headers={"x-litellm-end-user-id": end_user_id},
                )
            else:
                responses = await self.client.images.generate(
                    prompt=prompt,
                    model=self.image_model.name,
                    n=1,
                    response_format="b64_json",
                    quality="auto",
                    size="auto",
                    extra_headers={"x-litellm-end-user-id": end_user_id},
                )
            return responses.data[0].b64_json if responses.data else None

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
        self, user_prompt: str, end_user_id: str, image_bytes_list: list[bytes] | None = None
    ) -> bytes | None:
        """Returns PNG bytes rendered from `user_prompt`, or None on any failure or timeout.

        Args:
            user_prompt (str): Image description or edit instructions.
            end_user_id (str): Discord username sent as the LiteLLM end-user header.
            image_bytes_list (list[bytes] | None): Optional source image bytes to edit in place.
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
                model=self.image_model.name,
                error_type=type(exc).__name__,
                end_user_id=end_user_id,
                edit=bool(image_bytes_list),
                _exc_info=exc,
            )
            return None
        logfire.info(
            "gen_reply inline image generated",
            elapsed_seconds=time.monotonic() - started,
            model=self.image_model.name,
            image_bytes=len(image),
        )
        return image


class PromptGenerator(BaseModel):
    """Prompt director that expands a user request into a grounded generation prompt on the proxy."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client used for the refinement call."
    )
    prompt_model: ModelSettings = Field(
        ..., description="Model settings for the prompt director, which needs the grounding tools."
    )

    async def refine(
        self,
        user_prompt: str,
        instructions: str,
        end_user_id: str,
        enabled: bool,
        image_bytes_list: list[bytes] | None = None,
    ) -> str:
        """Returns a refined generation prompt, falling back to `user_prompt` when disabled or on failure.

        Args:
            user_prompt (str): Raw user generation request.
            instructions (str): Developer instructions for the refinement pass.
            end_user_id (str): Discord username sent as the LiteLLM end-user header.
            enabled (bool): Whether prompt refinement is enabled for this call.
            image_bytes_list (list[bytes] | None): Optional source images passed as visual context.
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
                    image_url=to_data_uri(data=image_bytes), detail="auto", type="input_image"
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
                    )
            refined = output_text_or_empty(responses=responses).strip()
        except Exception as exc:
            logfire.warn(
                "Prompt refinement failed; using raw user prompt",
                model=self.prompt_model.name,
                error_type=type(exc).__name__,
                end_user_id=end_user_id,
                _exc_info=exc,
            )
            return user_prompt
        logfire.info(
            "gen_reply prompt refine done",
            elapsed_seconds=time.monotonic() - started,
            model=self.prompt_model.name,
            refined=bool(refined),
        )
        return refined or user_prompt


class VoiceGenerator(BaseModel):
    """Best-effort text-to-speech renderer through the LiteLLM proxy.

    Prepends `TTS_STYLE_DIRECTIVE` to the input text because the proxy ignores `instructions`
    for this TTS model, and omits `response_format` because the proxy 500s on it (the model
    returns WAV by default).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client used for the audio.speech call."
    )
    # A bare string rather than a `ModelSettings`, since TTS dispatches no effort and offers
    # no tools; the caller reads the name off the catalog's `tts_model`.
    model_name: str = Field(..., description="TTS model string dispatched on the proxy.")

    async def generate(self, text: str, end_user_id: str) -> VoiceClip:
        """Returns a `VoiceClip` synthesized from `text`, capturing any timeout or error in `outcome`.

        Args:
            text (str): Spoken text to synthesize.
            end_user_id (str): Discord username sent as the LiteLLM end-user header.
        """
        spoken = text.strip()
        if not spoken:
            logfire.info(
                "Voice synthesis skipped: reply text was empty after stripping",
                end_user_id=end_user_id,
            )
            return VoiceClip(outcome="empty")
        try:
            async with asyncio.timeout(delay=VOICE_TIMEOUT_SECONDS):
                responses = await self.client.audio.speech.create(
                    input=f"{TTS_STYLE_DIRECTIVE}\n\n{spoken}",
                    model=self.model_name,
                    voice=TTS_VOICE,
                    speed=TTS_SPEED,
                    extra_headers={"x-litellm-end-user-id": end_user_id},
                )
                audio = await responses.aread()
            logfire.debug(
                "Voice synthesis succeeded",
                model=self.model_name,
                speed=TTS_SPEED,
                end_user_id=end_user_id,
                text_chars=len(spoken),
                audio_bytes=len(audio),
            )
            return VoiceClip(audio=audio, outcome="ok")
        except (TimeoutError, APITimeoutError):
            # The clip did not arrive within VOICE_TIMEOUT_SECONDS, or the SDK gave up on its own
            # transport timeout first. The caller marks the message with a timeout hint and still
            # leaves a plain text reply.
            logfire.warn(
                "Voice synthesis timed out; replying without audio",
                model=self.model_name,
                end_user_id=end_user_id,
                text_chars=len(spoken),
                _exc_info=True,
            )
            return VoiceClip(outcome="timeout")
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
            return VoiceClip(outcome="error")


class _RenderedVideo(Protocol):
    """The attributes read off a completed omni interaction's `output_video`."""

    @property
    def uri(self) -> str | None: ...


class _RenderedAudio(Protocol):
    """The attributes read off a completed Lyria interaction's `output_audio`."""

    @property
    def data(self) -> str | None: ...

    @property
    def mime_type(self) -> str | None: ...


class _InteractionResult(Protocol):
    """Structural view of a non-streaming `interactions.create` result.

    Excluding the stream with `isinstance` does not narrow the call's return type, since
    `AsyncStream` is only structurally an `AsyncIterator` and so stays in the union; the read
    attributes are declared here and cast to instead. Naming the response class is its own
    trap: `google.genai.interactions` star-imports `Interaction` from both the request-union
    alias (which carries none of these attributes) and the response module, and only the
    runtime import order makes the response class win. `ty` resolves it to the response class,
    so a nominal `cast("Interaction", ...)` is not wrong there, merely fragile: its meaning rests
    on that import order, while this Protocol pins the attributes actually read. The collision
    is real rather than theoretical, since a checker can bind the request union instead, and no
    checker in the tree will show it.
    """

    @property
    def status(self) -> str: ...

    @property
    def output_text(self) -> str | None: ...

    @property
    def output_video(self) -> _RenderedVideo | None: ...

    @property
    def output_audio(self) -> _RenderedAudio | None: ...


class VideoGenerator(BaseModel):
    """Native Gemini (omni) video renderer on the Interactions API, in raising and best-effort forms."""

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
        prompt: str,
        reference_image_sources: list[LoadedMedia],
        source_video: LoadedMedia | None = None,
    ) -> bytes:
        """Returns MP4 bytes rendered via the native Gemini (omni) Interactions API.

        Uploads `source_video` to the Files API for `task="edit"` when present; otherwise lets
        omni infer the task from `prompt` and up to `MAX_VIDEO_REFERENCE_IMAGES` reference
        images, requesting `delivery="uri"` and downloading the resulting clip.

        Args:
            prompt (str): Video generation or editing prompt.
            reference_image_sources (list[LoadedMedia]): Reference images with MIME types.
            source_video (LoadedMedia | None): Source video clip to edit, or None.

        Raises:
            RuntimeError: The source video upload, generation, or output download failed.
        """
        text = prompt or "Generate a video from the message content."
        content: list[TextContentParam | ImageContentParam | VideoContentParam]
        # A source-video edit is the one task pinned: with task omitted, omni infers the mode
        # (image_to_video vs reference_to_video vs text_to_video) from the prompt + input media,
        # which covers image_to_video without a brittle image-count heuristic. If the inferred
        # mode underwhelms (e.g. a lone image not animated in place), pin it here.
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
            # Each image must carry its real mime type: omni rejects an image content block with
            # an empty mime ("Unsupported MIME type: ").
            content.extend(
                ImageContentParam(
                    type="image",
                    data=base64.b64encode(loaded.data).decode(),
                    mime_type=cast("ImageContentMimeType", loaded.mime_type),
                )
                for loaded in reference_image_sources[:MAX_VIDEO_REFERENCE_IMAGES]
            )
            task_label = "infer_image"
        else:
            content = [TextContentParam(type="text", text=text)]
            task_label = "infer_text"
            response_format["aspect_ratio"] = "16:9"

        started = time.monotonic()
        async with asyncio.timeout(delay=VIDEO_RENDER_TIMEOUT_SECONDS):
            # No `stream=True`, so this is the interaction rather than an event stream, read
            # through the structural `_InteractionResult` view (its docstring has why).
            responses = cast(
                "_InteractionResult",
                await self.client.aio.interactions.create(
                    model=self.video_model.name,
                    input=content,
                    response_format=response_format,
                    generation_config=generation_config,
                ),
            )
        video = responses.output_video
        if responses.status != "completed" or video is None or video.uri is None:
            logfire.warn(
                "gen_reply video generation failed",
                model=self.video_model.name,
                status=str(responses.status),
                task=task_label,
                note=responses.output_text,
            )
            raise RuntimeError(
                f"Video generation failed: status={responses.status} "
                f"note={responses.output_text!r}"
            )
        logfire.debug(
            "gen_reply video job done",
            model=self.video_model.name,
            task=task_label,
            render_seconds=time.monotonic() - started,
        )
        return await self._download_output_video(uri=video.uri)

    async def generate(
        self, user_prompt: str, reference_image_sources: list[LoadedMedia] | None = None
    ) -> bytes | None:
        """Returns MP4 bytes rendered from `user_prompt`, or None on any failure or timeout.

        Args:
            user_prompt (str): Video description.
            reference_image_sources (list[LoadedMedia] | None): Optional reference images.
        """
        started = time.monotonic()
        try:
            video = await self.render(
                prompt=user_prompt, reference_image_sources=reference_image_sources or []
            )
        except Exception as exc:
            # Broad on purpose: the inline-marker boundary must degrade to "reply without a
            # video" whether the render timed out, was refused, or the download failed.
            logfire.warn(
                "Inline video generation failed; replying without video",
                model=self.video_model.name,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        logfire.info(
            "gen_reply inline video generated",
            elapsed_seconds=time.monotonic() - started,
            model=self.video_model.name,
            video_bytes=len(video),
        )
        return video

    async def _download_output_video(self, uri: str) -> bytes:
        """Downloads a URI-delivered clip within `FILES_READY_TIMEOUT_SECONDS`, retrying while it finalizes."""
        attempt = 0
        failure: Exception | None = None
        try:
            # Covers an attempt that never returns as well as the retries: google-genai sets no
            # transport timeout of its own.
            async with asyncio.timeout(delay=FILES_READY_TIMEOUT_SECONDS):
                while True:
                    try:
                        return await self.client.aio.files.download(file=uri)
                    except Exception as exc:
                        failure = exc
                        attempt += 1
                        logfire.debug(
                            "gen_reply video download not ready; retrying",
                            attempt=attempt,
                            error_type=type(exc).__name__,
                            _exc_info=exc,
                        )
                        await asyncio.sleep(2.0)
        except TimeoutError:
            # The last failed attempt says why the file never became servable; only a download
            # that never answered at all is left as the bare timeout.
            if failure is None:
                raise
            raise failure from None

    async def _upload_source_video(self, source_video: LoadedMedia) -> str:
        """Uploads `source_video` to the Files API and returns its `FileState.ACTIVE` URI.

        Bounds both the upload and activation poll with `FILES_READY_TIMEOUT_SECONDS`.
        """
        try:
            async with asyncio.timeout(delay=FILES_READY_TIMEOUT_SECONDS):
                return await upload_until_active(
                    client=self.client,
                    source=source_video.data,
                    mime_type=source_video.mime_type,
                    display_name="source.mp4",
                )
        except TimeoutError as exc:
            raise RuntimeError("Source video did not become ACTIVE before the deadline") from exc


class MusicClip(BaseModel):
    """A generated music clip: the audio bytes plus the mime type used to pick a file extension."""

    audio: bytes = Field(..., description="Rendered audio bytes for the music clip.")
    mime_type: str = Field(
        ..., description="Audio mime type Lyria reported, e.g. audio/mp3, used to name the file."
    )


class MusicGenerator(BaseModel):
    """Best-effort native Lyria music renderer on the Gemini Interactions API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[genai.Client] = Field(
        ..., description="Direct-to-Google Gemini client for the Lyria Interactions call."
    )
    music_model: ModelSettings = Field(
        ..., description="Model settings for native Gemini (Lyria) music generation."
    )

    async def generate(self, user_prompt: str) -> MusicClip | None:
        """Returns a `MusicClip` rendered from `user_prompt`, or None on any failure or timeout.

        Args:
            user_prompt (str): Music description passed as the plain-string `input`.
        """
        started = time.monotonic()
        # Output extraction and base64 decode stay inside the guard: an unexpected audio shape
        # (a raising `output_audio`, non-base64 `data`) must drop the clip to None like any other
        # failure, never escape into the streamer's single media-attach gather and abort the
        # already-ready voice / images alongside it.
        try:
            async with asyncio.timeout(delay=MUSIC_RENDER_TIMEOUT_SECONDS):
                responses = await self.client.aio.interactions.create(
                    model=self.music_model.name,
                    input=user_prompt,
                    system_instruction=MUSIC_STYLE_DIRECTIVE,
                )
            # No `stream=True`, so this is the interaction rather than an event stream (read via
            # `_InteractionResult`, see its docstring).
            audio = cast("_InteractionResult", responses).output_audio
            if audio is None or not audio.data:
                logfire.warn(
                    "Inline music generation returned no audio; replying without music",
                    model=self.music_model.name,
                )
                return None
            clip = base64.b64decode(audio.data)
        except Exception as exc:
            logfire.warn(
                "Inline music generation failed; replying without music",
                model=self.music_model.name,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        logfire.info(
            "gen_reply inline music generated",
            elapsed_seconds=time.monotonic() - started,
            model=self.music_model.name,
            music_bytes=len(clip),
            mime_type=audio.mime_type,
        )
        return MusicClip(audio=clip, mime_type=audio.mime_type or "audio/mp3")
