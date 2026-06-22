"""Shared media-generation helpers: the image and video render calls.

`generate_image_bytes` runs the downstream image model, so the router IMAGE route and the
QA-route inline `<image>` marker share one implementation. `generate_video_bytes` is its video
twin: the single native-Veo render call behind the VIDEO route, kept here so a future provider
swap (or a move of either render off the proxy) changes one place. `ImageReplyGenerator` wraps
`generate_image_bytes` for the streamer's best-effort inline path: the answer model's own
`<image>` description is rendered directly into a generation-only image (no editing; editing
stays the router IMAGE route's job), best-effort with a generous timeout so a slow render never
blocks anything but its own reply.
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

from discordbot.typings.models import ModelSettings, RuntimeModelCatalog

# Bound for the inline-image best-effort path: the render runs after the text reply is already
# on screen, so the wait only delays this message's own image, never others. Generous (mirrors
# VOICE_TIMEOUT_SECONDS) so a slower render still has room to land.
INLINE_IMAGE_TIMEOUT_SECONDS = 300.0


async def generate_image_bytes(
    *,
    client: AsyncOpenAI,
    image_model: ModelSettings,
    prompt: str,
    end_user_id: str,
    image_bytes_list: list[bytes] | None = None,
) -> bytes:
    """Renders one image to PNG bytes, editing source bytes when present, else generating fresh.

    Retries once on an empty payload (a transient hiccup occasionally returns no image) before
    raising, so a flaky empty result does not surface as a user-facing error; a genuine safety
    block returns empty on both attempts and still raises so callers can degrade. The image
    model is dispatched on the proxy with the same parameters for the edit and generate paths.
    """

    async def _dispatch() -> str | None:
        if image_bytes_list:
            result = await client.images.edit(
                image=image_bytes_list,
                prompt=prompt or "請依照附件內容進行編輯或優化。",
                model=image_model.name,
                n=1,
                response_format="b64_json",
                quality="auto",
                size="auto",
                extra_headers={"x-litellm-end-user-id": end_user_id},
            )
        else:
            result = await client.images.generate(
                prompt=prompt or "請生成一張圖片。",
                model=image_model.name,
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


async def generate_video_bytes(
    *,
    client: genai.Client,
    video_model: ModelSettings,
    prompt: str,
    reference_image_sources: list[tuple[bytes, str]],
    timeout_seconds: float,
) -> bytes:
    """Renders one video to MP4 bytes with the native Gemini (Veo) SDK; raises on failure.

    Mirrors `generate_image_bytes` as the single downstream video-render call, so the VIDEO
    route shares one implementation and a future provider swap changes one place. Goes DIRECT to
    Google (the LiteLLM proxy cannot reach Veo); up to three source images ride as ASSET
    reference images. Raises `RuntimeError` on an operation error or an empty result, surfacing
    the RAI safety-filter reasons when the operation finished cleanly but filtered every
    candidate, so a blocked render is diagnosable instead of a bare `None`.
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
    operation = await client.aio.models.generate_videos(
        model=video_model.name,
        prompt=prompt or "請依照訊息內容生成一段影片。",
        config=video_config,
    )
    logfire.debug(
        "gen_reply video job created",
        operation=operation.name,
        reference_images=len(reference_images),
    )
    async with asyncio.timeout(delay=timeout_seconds):
        while not operation.done:
            await asyncio.sleep(5)
            operation = await client.aio.operations.get(operation=operation)
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
    return await client.aio.files.download(file=generated.video.uri)


class ImageReplyGenerator(BaseModel):
    """Best-effort inline-image render for the QA-route `<image>` marker.

    Holds the shared client and model catalog; `generate` renders one generation-only image
    from the answer model's own `<image>` description, returning the PNG bytes or None on any
    failure or timeout. None disables the inline path for a reply (the kill-switch / non-QA
    routes), mirroring how `VoiceSynthesizer` is gated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client used for the image render."
    )
    runtime_models: RuntimeModelCatalog = Field(
        ..., description="Model catalog supplying the image model."
    )

    async def generate(self, *, user_prompt: str, end_user_id: str) -> bytes | None:
        """Renders one image from the description; None on any failure or timeout."""
        started = time.monotonic()
        try:
            async with asyncio.timeout(delay=INLINE_IMAGE_TIMEOUT_SECONDS):
                image = await generate_image_bytes(
                    client=self.client,
                    image_model=self.runtime_models.image_model,
                    prompt=user_prompt,
                    end_user_id=end_user_id,
                )
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
