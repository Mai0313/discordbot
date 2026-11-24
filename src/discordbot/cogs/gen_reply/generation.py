"""Media-generation services: prompt refinement and image, voice, video, and music renderers."""

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
from discordbot.utils.llm_transcript import defuse_separator_fences
from discordbot.cogs.gen_reply.images import to_data_uri
from discordbot.typings.context_budgets import MAX_VIDEO_REFERENCE_IMAGES
from discordbot.cogs.gen_reply.files_api import upload_until_active

if TYPE_CHECKING:
    from google.genai.interactions import ImageContentMimeType
    from openai.types.responses.response_input_file_param import ResponseInputFileParam

# Gemini TTS style prompting is documented in English and read as style rather than spoken aloud.
TTS_VOICE = "Despina"
TTS_STYLE_DIRECTIVE = "Using a natural 18-year-old woman's voice that fits the following text:"
TTS_SPEED = 1.5

# Sent as Lyria's system_instruction (in English per Lyria's prompt surface).
MUSIC_STYLE_DIRECTIVE = (
    "Compose in a Japanese anime / J-pop style with Japanese-language vocals by default; if the "
    "description clearly asks for a different genre, style, or lyric language, or for an "
    "instrumental, follow the description instead."
)

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
            # Broad: inline-marker boundary degrades to no image on timeout, empty data, or error.
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
        image_bytes_list: list[bytes] | None = None,
        conversation_context: str | None = None,
    ) -> str:
        """Returns a refined generation prompt, falling back to `user_prompt` on failure.

        Args:
            user_prompt (str): Raw user generation request.
            instructions (str): Developer instructions for the refinement pass.
            end_user_id (str): Discord username sent as the LiteLLM end-user header.
            image_bytes_list (list[bytes] | None): Optional source images passed as visual context.
            conversation_context (str | None): Optional recent text-only conversation lines used
                only to resolve references, pronouns, or follow-up edits in `user_prompt`.
        """
        if conversation_context is not None and conversation_context.strip():
            safe_context = defuse_separator_fences(text=conversation_context.strip())
            safe_prompt = defuse_separator_fences(text=user_prompt)
            request_text = (
                f"==== Recent conversation (reference only) ====\n{safe_context}\n"
                f"==== Current generation request ====\n{safe_prompt}"
            )
        else:
            request_text = f"User generation request:\n{user_prompt}"
        director_content: list[
            ResponseInputTextParam | ResponseInputImageParam | ResponseInputFileParam
        ] = [ResponseInputTextParam(text=request_text, type="input_text")]
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

    Prepends `TTS_STYLE_DIRECTIVE` to `input` and omits `response_format` (model defaults to WAV).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client used for the audio.speech call."
    )
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
                # Proxy ignores `instructions` for this TTS model and 500s on `response_format`.
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
            logfire.warn(
                "Voice synthesis timed out; replying without audio",
                model=self.model_name,
                end_user_id=end_user_id,
                text_chars=len(spoken),
                _exc_info=True,
            )
            return VoiceClip(outcome="timeout")
        except Exception as exc:
            # Broad: proxy/provider failures can take any shape; degrade to text reply with hint.
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

    `interactions.create` returns `Interaction | AsyncStream[...]` and `isinstance` against
    `AsyncIterator` does not narrow `AsyncStream`, so callers cast to this Protocol instead.
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
        # Pin task="edit" only for source video; otherwise omni infers the mode from input media.
        generation_config: GenerationConfigParam | None = None
        task_label: str
        # Set 16:9 only for pure text; omni rejects aspect_ratio on edit and may reject it on
        # image_to_video.
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
            # Omni rejects image blocks with an empty mime_type.
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
            # Broad: inline-marker boundary degrades to no video on timeout, refusal, or error.
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
            # Covers hung reads as well as retries: google-genai sets no default transport timeout.
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
            # Surface the last download error; re-raise TimeoutError only if no attempt failed.
            if failure is None:
                raise
            raise failure from None

    async def _upload_source_video(self, source_video: LoadedMedia) -> str:
        """Uploads `source_video` within `FILES_READY_TIMEOUT_SECONDS`, returning its URI."""
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
        # Broad: keep extraction and base64 decode inside the guard so malformed audio drops to
        # None instead of escaping into the streamer's media gather.
        try:
            async with asyncio.timeout(delay=MUSIC_RENDER_TIMEOUT_SECONDS):
                responses = await self.client.aio.interactions.create(
                    model=self.music_model.name,
                    input=user_prompt,
                    system_instruction=MUSIC_STYLE_DIRECTIVE,
                )
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
