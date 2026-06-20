"""Shared best-effort Responses API call surfaces for one-shot LLM calls.

Each helper owns the proxy call surface, the per-call timeout, and the failure handling so a
caller only maps a None result to its own fallback. Client construction lives at the call
sites as inline `AsyncOpenAI(...)` / `genai.Client(...)` cached_properties, not here.
"""

from typing import cast
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import BaseModel, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.models import ModelSettings


async def parse_responses_or_none[StructuredT: BaseModel](  # noqa: PLR0913 -- shared best-effort call surface; all params are per-call inputs
    *,
    client: AsyncOpenAI,
    model: ModelSettings,
    instructions: str,
    user_text: str,
    end_user_id: str,
    text_format: type[StructuredT],
    timeout_seconds: float,
) -> StructuredT | None:
    """Runs one best-effort structured Responses.parse call, returning None on any failure.

    Owns the shared proxy call surface, the timeout, and the failure handling so each caller
    only maps None to its own fallback: a timeout, an empty or refused output
    (`ValidationError` from parsing no text), an incomplete (truncated) response, or any
    other error all degrade to None.
    """
    try:
        async with asyncio.timeout(delay=timeout_seconds):
            responses = await client.responses.parse(
                model=model.name,
                instructions=instructions,
                input=cast(
                    "ResponseInputParam", [EasyInputMessageParam(role="user", content=user_text)]
                ),
                text_format=text_format,
                reasoning=model.reasoning,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": end_user_id},
                extra_body={"mock_testing_fallbacks": False},
            )
    except TimeoutError:
        logfire.warn(
            "Structured LLM request timed out; skipping",
            end_user_id=end_user_id,
            timeout_seconds=timeout_seconds,
        )
        return None
    except ValidationError:
        logfire.warn("Structured LLM parse returned no text; skipping", end_user_id=end_user_id)
        return None
    except Exception:
        logfire.warn(
            "Structured LLM request failed; skipping", end_user_id=end_user_id, _exc_info=True
        )
        return None
    if responses.status == "incomplete":
        logfire.warn(
            "Structured LLM response incomplete; skipping",
            end_user_id=end_user_id,
            incomplete_details=str(responses.incomplete_details),
        )
        return None
    return responses.output_parsed


async def create_text_or_none(  # noqa: PLR0913 -- shared best-effort call surface; all params are per-call inputs
    *,
    client: AsyncOpenAI,
    model: ModelSettings,
    instructions: str,
    user_text: str,
    end_user_id: str,
    timeout_seconds: float,
) -> str | None:
    """Runs one best-effort text Responses.create call, returning None on any failure.

    Mirrors `parse_responses_or_none` for the non-structured callers: owns the shared proxy
    call surface, the timeout, and the failure handling, and returns the trimmed output text
    (or None on timeout / any error) so each caller maps None to its own fallback line.
    """
    try:
        async with asyncio.timeout(delay=timeout_seconds):
            responses = await client.responses.create(
                model=model.name,
                instructions=instructions,
                input=cast(
                    "ResponseInputParam", [EasyInputMessageParam(role="user", content=user_text)]
                ),
                reasoning=model.reasoning,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": end_user_id},
                extra_body={"mock_testing_fallbacks": False},
            )
    except TimeoutError:
        logfire.warn(
            "Text LLM request timed out; using fallback",
            end_user_id=end_user_id,
            timeout_seconds=timeout_seconds,
        )
        return None
    except Exception:
        logfire.warn(
            "Text LLM request failed; using fallback", end_user_id=end_user_id, _exc_info=True
        )
        return None
    return (responses.output_text or "").strip()
