"""Shared image-generation helpers: the prompt director plus the image render call.

`refine_generation_prompt` expands a thin IMAGE/VIDEO request into one rich, self-contained
generation prompt (with grounding tools) and `generate_image_bytes` runs the downstream image
model, so the router IMAGE/VIDEO routes and the QA-route inline `<image>` marker all share one
implementation. `ImageReplyGenerator` wraps both for the streamer's best-effort inline path:
the answer model only needs a rough description, this refines it and renders a generation-only
image (no editing; editing stays the router IMAGE route's job), best-effort with a generous
timeout so a slow render never blocks anything but its own reply.
"""

import time
import base64
from typing import TYPE_CHECKING, cast
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.images import convert_base64_to_data_uri
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.cogs._gen_reply.prompts import IMAGE_PROMPT

if TYPE_CHECKING:
    from openai.types.responses.response_input_file_param import ResponseInputFileParam

# Bound for the inline-image best-effort path: refine + render run serially after the text reply
# is already on screen, so the wait only delays this message's own image, never others. Generous
# (mirrors VOICE_TIMEOUT_SECONDS) so a slower render still has room to land.
INLINE_IMAGE_TIMEOUT_SECONDS = 300.0


async def refine_generation_prompt(  # noqa: PLR0913 -- shared director inputs: enabled kill-switch + client + prompt model + user prompt + instructions + end-user id + optional source images
    *,
    enabled: bool = True,
    client: AsyncOpenAI,
    prompt_model: ModelSettings,
    user_prompt: str,
    instructions: str,
    end_user_id: str,
    image_bytes_list: list[bytes] | None = None,
) -> str:
    """Expands a thin IMAGE/VIDEO request into a rich, self-contained generation prompt.

    Runs `prompt_model` with grounding tools so a vague request ("draw the heroine of some
    anime") is looked up and resolved before the image/video model renders it. Any already-loaded
    source bytes ride along as input images so the draft is grounded in them without a re-download.
    Best-effort: an empty draft or ANY error falls back to the raw `user_prompt`, so a director
    failure never aborts generation — callers wrap generation in an error path and must not see an
    exception escape here. With `enabled=False` (the `REFINE_PROMPT_ENABLED` kill-switch) the
    director is skipped entirely and the raw `user_prompt` is returned, same as an empty draft.
    """
    if not enabled:
        return user_prompt
    director_content: list[
        ResponseInputTextParam | ResponseInputImageParam | ResponseInputFileParam
    ] = [
        ResponseInputTextParam(text=f"User generation request:\n{user_prompt}", type="input_text")
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
        with logfire.span("gen_reply prompt refine", model=prompt_model.name):
            responses = await client.responses.create(
                model=prompt_model.name,
                instructions=instructions,
                input=cast("ResponseInputParam", director_input),
                reasoning=prompt_model.reasoning,
                tools=list(prompt_model.tools),
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": end_user_id},
                extra_body={"mock_testing_fallbacks": False},
            )
        refined = (responses.output_text or "").strip()
    except Exception:
        logfire.warn("Prompt refinement failed; using raw user prompt", _exc_info=True)
        return user_prompt
    logfire.info(
        "gen_reply prompt refine done",
        elapsed_seconds=time.monotonic() - started,
        refined=bool(refined),
    )
    return refined or user_prompt


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

    Holds the shared client and model catalog; `generate` refines the answer model's rough
    description and renders one generation-only image, returning the PNG bytes or None on any
    failure or timeout. None disables the inline path for a reply (the kill-switch / non-QA
    routes), mirroring how `VoiceSynthesizer` is gated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client used for refinement and the image render."
    )
    runtime_models: RuntimeModelCatalog = Field(
        ..., description="Model catalog supplying the prompt-director and image models."
    )
    refine_enabled: bool = Field(
        default=True,
        description="Whether the prompt director refines the rough description before rendering.",
    )

    async def generate(self, *, user_prompt: str, end_user_id: str) -> bytes | None:
        """Refines the rough description then renders one image; None on any failure or timeout."""
        started = time.monotonic()
        try:
            async with asyncio.timeout(delay=INLINE_IMAGE_TIMEOUT_SECONDS):
                refined = await refine_generation_prompt(
                    enabled=self.refine_enabled,
                    client=self.client,
                    prompt_model=self.runtime_models.prompt_model,
                    user_prompt=user_prompt,
                    instructions=IMAGE_PROMPT,
                    end_user_id=end_user_id,
                )
                image = await generate_image_bytes(
                    client=self.client,
                    image_model=self.runtime_models.image_model,
                    prompt=refined,
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
