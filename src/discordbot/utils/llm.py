"""Shared best-effort Responses API call surfaces for one-shot LLM calls.

Each helper owns the proxy call surface and the failure handling so a caller only maps a None
result to its own fallback. Client construction lives at the call sites as inline
`AsyncOpenAI(...)` / `genai.Client(...)` cached_properties, not here.

Neither helper takes a deadline: the SDK's own (connect 5s / read 600s) is the bound, and the
`APITimeoutError` it raises lands in the same broad `except` that already absorbs a proxy
`ServiceUnavailableError`, so every caller's fallback is reached either way. See
`typings/timeouts.py` for why no LLM call in this tree carries one of ours.
"""

from typing import cast

from openai import AsyncOpenAI
import logfire
from pydantic import BaseModel, ValidationError
from openai.types.responses import Response
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.models import ModelSettings


def output_text_or_empty(*, responses: Response) -> str:
    """Aggregates the response's text, tolerating output_text parts whose `text` is None.

    The SDK's `Response.output_text` does a bare `"".join(...)` over every output_text part's
    text; a single part with `text=None` — a Gemini-via-proxy quirk seen on some grounded /
    refused turns — makes that join raise `TypeError`, defeating the usual
    `(responses.output_text or "")` guard. This mirrors the SDK aggregation but keeps only the
    non-empty text parts, so a stray None yields "" instead of raising.
    """
    texts = [
        content.text
        for output in responses.output
        if output.type == "message"
        for content in output.content
        if content.type == "output_text" and content.text
    ]
    return "".join(texts)


async def parse_responses_or_none[StructuredT: BaseModel](  # noqa: PLR0913 -- shared best-effort call surface; all params are per-call inputs
    *,
    client: AsyncOpenAI,
    model: ModelSettings,
    instructions: str,
    user_text: str,
    end_user_id: str,
    text_format: type[StructuredT],
) -> StructuredT | None:
    """Runs one best-effort structured Responses.parse call, returning None on any failure.

    Owns the shared proxy call surface and the failure handling so each caller only maps None
    to its own fallback: a transport timeout, an empty output or a payload that does not match
    `text_format` (both surface as `ValidationError`), a refusal (which simply leaves
    `output_parsed` None), an incomplete (truncated) response, or any other error all degrade
    to None.
    """
    try:
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
        )
    except ValidationError as exc:
        logfire.warn(
            "Structured LLM parse returned no text or an off-schema payload; skipping",
            end_user_id=end_user_id,
            model=model.name,
            _exc_info=exc,
        )
        return None
    # Broad on purpose: the LiteLLM proxy surfaces openai.APIError, httpx transport errors and
    # proxy-side 5xx bodies as unrelated types, and every failure here degrades to None.
    except Exception as exc:
        logfire.warn(
            "Structured LLM request failed; skipping",
            end_user_id=end_user_id,
            model=model.name,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
        return None
    if responses.status == "incomplete":
        logfire.warn(
            "Structured LLM response incomplete; skipping",
            end_user_id=end_user_id,
            model=model.name,
            incomplete_details=str(responses.incomplete_details),
        )
        return None
    return responses.output_parsed


async def create_text_or_none(
    *,
    client: AsyncOpenAI,
    model: ModelSettings,
    instructions: str,
    user_text: str,
    end_user_id: str,
) -> str | None:
    """Runs one best-effort text Responses.create call, returning None on any failure.

    Mirrors `parse_responses_or_none` for the non-structured callers: owns the shared proxy
    call surface and the failure handling, and returns the trimmed output text (or None on any
    error) so each caller maps None to its own fallback line.
    """
    try:
        responses = await client.responses.create(
            model=model.name,
            instructions=instructions,
            input=cast(
                "ResponseInputParam", [EasyInputMessageParam(role="user", content=user_text)]
            ),
            reasoning=model.reasoning,
            service_tier="auto",
            extra_headers={"x-litellm-end-user-id": end_user_id},
        )
    # Broad on purpose: this shared surface owns failure handling so every caller only maps
    # None to its own fallback line; proxy, transport and SDK errors share no base class.
    except Exception as exc:
        logfire.warn(
            "Text LLM request failed; using fallback",
            end_user_id=end_user_id,
            model=model.name,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
        return None
    return output_text_or_empty(responses=responses).strip()
