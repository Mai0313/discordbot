"""Shared image-generation helper: the image render call.

`generate_image_bytes` runs the downstream image model, so the router IMAGE route and the
QA-route inline `<image>` marker share one implementation. `ImageReplyGenerator` wraps it for
the streamer's best-effort inline path: the answer model's own `<image>` description is rendered
directly into a generation-only image (no editing; editing stays the router IMAGE route's job),
best-effort with a generous timeout so a slow render never blocks anything but its own reply.
"""

import time
import base64
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation

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

    Raises on an empty result or a missing payload so callers can degrade; the image model
    itself is dispatched on the proxy with the same parameters for the edit and generate paths.
    """
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
    if not result.data:
        raise ValueError("Image operation returned no results")
    image_b64 = result.data[0].b64_json
    if image_b64 is None:
        raise ValueError("Image operation returned no b64_json")
    return base64.b64decode(image_b64)


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
