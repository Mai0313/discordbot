"""Media-generation services: the image, video, and music render calls behind one uniform shape.

All runtime media generators are BaseModel services held as cog `cached_property`s (mirroring
`VoiceSynthesizer`), so every media render goes through the same calling convention instead of a
half-free-function / half-class mix:

- `ImageGenerator` runs the downstream image model. `render` is the raising primitive shared by
  the router IMAGE route (which also edits source pixels) and the best-effort inline path;
  `generate` is the QA-route `<image>` marker's best-effort wrapper (generation-only, timeout,
  None on any failure) so a slow inline render never blocks anything but its own reply.
- `VideoGenerator` runs the single native-Veo render behind the VIDEO route. It has only `render`
  (raising): video has no inline marker and is always the primary deliverable, so there is no
  best-effort twin by design.
- `MusicGenerator` runs the native-Lyria render behind the QA-route `<music>` marker via the
  Gemini Interactions API. Like `ImageGenerator.generate` it is best-effort only (`generate`,
  None on any failure), since music is inline-only; it goes DIRECT to Google like `VideoGenerator`.

Keeping them here means a future provider swap (or a move of a render off the proxy) changes
one place.
"""

import time
import base64
import asyncio

from google import genai
from openai import AsyncOpenAI
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from google.genai.types import (
    Image,
    GenerateVideosConfig,
    VideoGenerationReferenceType,
    VideoGenerationReferenceImage,
)

from discordbot.typings.models import ModelSettings

# Bound for the inline-image best-effort path: the render runs after the text reply is already
# on screen, so the wait only delays this message's own image, never others. Generous (mirrors
# VOICE_TIMEOUT_SECONDS) so a slower render still has room to land.
INLINE_IMAGE_TIMEOUT_SECONDS = 300.0

# Hard ceiling on the video-generation polling loop so a hung provider job cannot leave the
# message handler waiting forever. Co-located with the image timeout since it is a property of
# the render, not of the route that calls it.
VIDEO_RENDER_TIMEOUT_SECONDS = 600.0

# Bound for the inline-music best-effort path, mirroring the inline-image timeout: the render
# runs after the text reply is on screen, so the wait only delays this message's own clip.
MUSIC_RENDER_TIMEOUT_SECONDS = 300.0

# Fixed musical-style directive sent as the Lyria `system_instruction`. English on purpose (the
# Lyria prompt surface is documented in English). Lyria picks the lyric language from the prompt
# (docs: "generates lyrics in the language of your prompt"), so the language is defaulted here too,
# not just the genre. The QA `<music>` marker prompt already steers the answer model to default to
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


class ImageGenerator(BaseModel):
    """Image render shared by the router IMAGE route and the QA-route `<image>` marker.

    Holds the shared client and the image model. `render` is the raising primitive (edits when
    source bytes are present, else generates); `generate` is the best-effort inline wrapper that
    returns None on any failure or timeout, mirroring how `VoiceSynthesizer` is gated.
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
                    prompt=prompt or "請生成一張圖片。",
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
                logfire.warn("Image operation returned an empty result; retrying once")
        raise ValueError("Image operation returned no image data after one retry")

    async def generate(self, *, user_prompt: str, end_user_id: str) -> bytes | None:
        """Renders one image from the description; None on any failure or timeout.

        Best-effort wrapper around `render` for the QA-route `<image>` marker: generation-only
        (no editing) inside a generous timeout, returning None to disable the inline path for a
        reply rather than raising into the streamer's path.
        """
        started = time.monotonic()
        try:
            async with asyncio.timeout(delay=INLINE_IMAGE_TIMEOUT_SECONDS):
                image = await self.render(prompt=user_prompt, end_user_id=end_user_id)
        except Exception:
            logfire.warn(
                "Inline image generation failed; replying without an image", _exc_info=True
            )
            return None
        logfire.info(
            "gen_reply inline image generated",
            elapsed_seconds=time.monotonic() - started,
            image_bytes=len(image),
        )
        return image


class VideoGenerator(BaseModel):
    """Native-Veo video render behind the VIDEO route.

    Holds the direct-to-Google client and the video model. Only `render` (raising) exists: video
    has no inline marker and is always the primary deliverable, so unlike `ImageGenerator` there
    is no best-effort twin.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[genai.Client] = Field(
        ..., description="Direct-to-Google Gemini client (Veo is unreachable via the proxy)."
    )
    video_model: ModelSettings = Field(
        ..., description="Model settings for native Gemini (Veo) video generation."
    )

    async def render(
        self, *, prompt: str, reference_image_sources: list[tuple[bytes, str]]
    ) -> bytes:
        """Renders one video to MP4 bytes with the native Gemini (Veo) SDK; raises on failure.

        The video twin of `ImageGenerator.render` as the single downstream video render, so the
        VIDEO route shares one implementation and a future provider swap changes one place. Goes
        DIRECT to Google (the LiteLLM proxy cannot reach Veo); up to three source images ride as
        ASSET reference images. Raises `RuntimeError` on an operation error or an empty result,
        surfacing the RAI safety-filter reasons when the operation finished cleanly but filtered
        every candidate, so a blocked render is diagnosable instead of a bare `None`.
        """
        reference_images = [
            VideoGenerationReferenceImage(
                image=Image(image_bytes=raw, mime_type=mime),
                reference_type=VideoGenerationReferenceType.ASSET,
            )
            for raw, mime in reference_image_sources[:3]
        ]
        # Veo 3.1 requires duration_seconds=8 at 1080p and with reference images, so it is pinned
        # (4/6/8 are only selectable at 720p); audio rides on by default. Only fields this model
        # accepts are set: enhance_prompt 400s on veo-3.1-generate-preview, and fps / seed /
        # generate_audio / compression_quality are Vertex-only and 400 here too.
        video_config = GenerateVideosConfig(
            number_of_videos=1,
            aspect_ratio="16:9",
            resolution="1080p",
            duration_seconds=8,
            reference_images=reference_images or None,
        )
        started = time.monotonic()
        operation = await self.client.aio.models.generate_videos(
            model=self.video_model.name,
            prompt=prompt or "請依照訊息內容生成一段影片。",
            config=video_config,
        )
        logfire.debug(
            "gen_reply video job created",
            operation=operation.name,
            reference_images=len(reference_images),
        )
        async with asyncio.timeout(delay=VIDEO_RENDER_TIMEOUT_SECONDS):
            while not operation.done:
                await asyncio.sleep(5)
                operation = await self.client.aio.operations.get(operation=operation)
                logfire.debug(
                    "gen_reply video poll",
                    operation=operation.name,
                    done=operation.done,
                    poll_seconds=time.monotonic() - started,
                )
        response = operation.response
        if operation.error or not (response and response.generated_videos):
            # A clean finish with no videos means every candidate was safety-filtered; surface the
            # RAI reasons so the empty case is distinguishable from a true operation error instead
            # of the misleading bare `None` that operation.error carries here.
            rai_reasons = response.rai_media_filtered_reasons if response else None
            logfire.warn(
                "gen_reply video generation failed",
                operation=operation.name,
                error=str(operation.error),
                rai_media_filtered_count=response.rai_media_filtered_count if response else None,
                rai_media_filtered_reasons=rai_reasons,
            )
            raise RuntimeError(f"Video generation failed: {operation.error or rai_reasons}")
        generated = response.generated_videos[0]
        if generated.video is None or generated.video.uri is None:
            raise RuntimeError(f"Video generation returned no video for {operation.name}")
        return await self.client.aio.files.download(file=generated.video.uri)


class MusicClip(BaseModel):
    """A generated music clip: the audio bytes plus the mime type used to pick a file extension."""

    audio: bytes = Field(..., description="Rendered audio bytes for the music clip.")
    mime_type: str = Field(
        ..., description="Audio mime type Lyria reported, e.g. audio/mp3, used to name the file."
    )


class MusicGenerator(BaseModel):
    """Native-Lyria music render behind the QA-route `<music>` marker.

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

        Best-effort wrapper for the QA-route `<music>` marker: one non-streaming Interactions
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
        except Exception:
            logfire.warn("Inline music generation failed; replying without music", _exc_info=True)
            return None
        logfire.info(
            "gen_reply inline music generated",
            elapsed_seconds=time.monotonic() - started,
            music_bytes=len(clip),
            mime_type=audio.mime_type,
        )
        return MusicClip(audio=clip, mime_type=audio.mime_type or "audio/mp3")
