"""Image generation for the reply pipeline: prompt refinement plus best-effort drawing.

`refine_generation_prompt` is the shared prompt director: it expands a thin user request into
one rich, self-contained generation prompt using grounding tools, and is reused by the
IMAGE / VIDEO routes (`_handle_image_reply` / `_handle_video_reply`) and the inline `<image>`
tag path, so the refine logic has a single definition. `ImageGenerator` wraps refine plus
`images.generate` into a best-effort service (mirroring `VoiceSynthesizer`) so the streamer can
draw an image the answer model asked for inline via `<image>...</image>` and attach it onto the
reply afterward; any failure or timeout leaves a text-only reply.
"""

import time
import base64
import asyncio
from enum import StrEnum
from typing import cast

from openai import AsyncOpenAI, APITimeoutError
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.images import convert_base64_to_data_uri
from discordbot.typings.models import ModelSettings

# Hard ceiling on the inline-image refine + draw so a hung provider job cannot keep the
# message's own pipeline (its 🆗 reaction + memory scheduling) waiting. The text reply is
# already on screen, so this only delays the attached image of this one message, never others;
# it is generous (matching the voice timeout) so a slower draw still lands.
IMAGE_GENERATION_TIMEOUT_SECONDS = 300.0

# Filename of the inline-generated image attached onto the reply.
GENERATED_IMAGE_FILENAME = "generated.png"


async def refine_generation_prompt(  # noqa: PLR0913 -- director needs the request, instructions, identity, and optional source images
    *,
    client: AsyncOpenAI,
    prompt_model: ModelSettings,
    user_prompt: str,
    instructions: str,
    end_user_id: str,
    image_bytes_list: list[bytes] | None = None,
) -> str:
    """Expands a thin IMAGE/VIDEO request into a rich, self-contained generation prompt.

    Runs `prompt_model` with grounding tools so a vague request ("draw the heroine of some
    anime") is looked up and resolved before the image/video model renders it. Any
    already-loaded source bytes ride along as input images so the draft is grounded in them
    without a re-download. Best-effort: an empty draft or ANY error falls back to the raw
    `user_prompt`, so a director failure never aborts generation — callers wrap this in an
    error path and must not see an exception escape here.
    """
    director_content: list[
        ResponseInputTextParam | ResponseInputImageParam | ResponseInputFileParam
    ] = [ResponseInputTextParam(text=f"User generation request:\n{user_prompt}", type="input_text")]
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


class ImageGenerationOutcome(StrEnum):
    """Why an inline image generation attempt ended, so the caller can hint appropriately."""

    OK = "ok"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"


class GeneratedImage(BaseModel):
    """Result of one inline generation attempt: the image (when produced) plus why it ended.

    A failed attempt carries `image_b64=None` and a non-OK `outcome`; the caller always keeps
    the text reply and uses `outcome` to decide its best-effort failure hint.
    """

    image_b64: str | None = Field(
        default=None, description="Base64-encoded PNG, or None when no image was produced."
    )
    outcome: ImageGenerationOutcome = Field(
        ..., description="Why generation ended; drives the caller's best-effort failure hint."
    )


class ImageGenerator(BaseModel):
    """Best-effort inline image generation through the LiteLLM proxy.

    Holds the shared async client, the image model name, the director's `prompt_model`, and the
    refine `instructions` (set by the cog to `IMAGE_PROMPT`, kept as a field so this module need
    not import the prompt text). `generate` refines the answer model's rough `<image>` request
    then draws it (text-to-image), returning a `GeneratedImage` with the PNG (when produced)
    plus an outcome (OK / EMPTY / TIMEOUT / ERROR) so the caller degrades to a text reply.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client used for refine and images.generate."
    )
    image_model_name: str = Field(
        ..., description="Image model string dispatched on the images.generate endpoint."
    )
    prompt_model: ModelSettings = Field(
        ..., description="Director model that refines the rough request before drawing."
    )
    instructions: str = Field(
        ..., description="Refine instructions handed to the director (the IMAGE_PROMPT text)."
    )

    async def generate(self, *, rough_prompt: str, end_user_id: str) -> GeneratedImage:
        """Refines the rough request then draws it, reporting why it ended for hinting."""
        draft = rough_prompt.strip()
        if not draft:
            logfire.info(
                "Inline image skipped: the image span was empty", end_user_id=end_user_id
            )
            return GeneratedImage(outcome=ImageGenerationOutcome.EMPTY)
        started = time.monotonic()
        try:
            async with asyncio.timeout(delay=IMAGE_GENERATION_TIMEOUT_SECONDS):
                refined = await refine_generation_prompt(
                    client=self.client,
                    prompt_model=self.prompt_model,
                    user_prompt=draft,
                    instructions=self.instructions,
                    end_user_id=end_user_id,
                )
                with logfire.span("gen_reply inline image", model=self.image_model_name):
                    result = await self.client.images.generate(
                        prompt=refined or draft,
                        model=self.image_model_name,
                        n=1,
                        response_format="b64_json",
                        quality="auto",
                        size="auto",
                        extra_headers={"x-litellm-end-user-id": end_user_id},
                    )
        except (TimeoutError, APITimeoutError):
            logfire.warn(
                "Inline image generation timed out; replying without image",
                end_user_id=end_user_id,
                _exc_info=True,
            )
            return GeneratedImage(outcome=ImageGenerationOutcome.TIMEOUT)
        except Exception:
            logfire.warn(
                "Inline image generation failed; replying without image",
                end_user_id=end_user_id,
                _exc_info=True,
            )
            return GeneratedImage(outcome=ImageGenerationOutcome.ERROR)
        image_b64 = result.data[0].b64_json if result.data else None
        if image_b64 is None:
            logfire.warn(
                "Inline image generation returned no image; replying without it",
                end_user_id=end_user_id,
            )
            return GeneratedImage(outcome=ImageGenerationOutcome.ERROR)
        logfire.info(
            "Inline image generated",
            elapsed_seconds=time.monotonic() - started,
            end_user_id=end_user_id,
        )
        return GeneratedImage(image_b64=image_b64, outcome=ImageGenerationOutcome.OK)
