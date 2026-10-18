"""Shared best-effort Responses API call surfaces for one-shot LLM calls.

Each helper owns the proxy call surface, the optional per-call deadline, and the failure
handling so a caller only maps a None result to its own fallback. Client construction stays at
its call site: this is the call surface, not a client factory.

`timeout_seconds` is optional because most callers should not set one: it is for a caller with
a deadline of its own, not a guard against a hung provider. The client already bounds itself
(`typings/timeouts.py` carries the measured figures), so a wrapper on background work would
restate that bound for nothing.
"""

import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import BaseModel, ValidationError
from openai.types.responses import Response
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.models import ModelSettings


def output_text_or_empty(responses: Response) -> str:
    """Returns the concatenated `output_text` from `responses`, tolerating `text=None` parts.

    The SDK's `Response.output_text` does a bare `"".join(...)` over every `output_text` part's
    text; a part with `text=None` makes that join raise `TypeError`. This keeps only non-empty
    text parts so a stray None yields `""` instead of raising.

    Args:
        responses (Response): Responses API result to read text from.
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
    client: AsyncOpenAI,
    model: ModelSettings,
    instructions: str,
    user_text: str,
    end_user_id: str,
    text_format: type[StructuredT],
    timeout_seconds: float | None = None,
) -> StructuredT | None:
    """Returns the parsed `StructuredT` from one best-effort `responses.parse` call, or None on failure.

    A timeout, an empty output or off-schema payload (`ValidationError`), a refusal
    (`output_parsed` is None), an incomplete response, or any other error all degrade to None.

    Args:
        client (AsyncOpenAI): OpenAI client pointed at the LiteLLM proxy.
        model (ModelSettings): Model name and reasoning configuration.
        instructions (str): System instructions for the request.
        user_text (str): User message content.
        end_user_id (str): End-user identifier sent in `x-litellm-end-user-id`.
        text_format (type[StructuredT]): Pydantic model class to parse the output into.
        timeout_seconds (float | None): Optional caller-side wall-clock deadline in seconds.
    """
    try:
        async with asyncio.timeout(delay=timeout_seconds):
            responses = await client.responses.parse(
                model=model.name,
                instructions=instructions,
                input=[EasyInputMessageParam(role="user", content=user_text)],
                text_format=text_format,
                reasoning=model.reasoning,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": end_user_id},
            )
    except TimeoutError as exc:
        logfire.warn(
            "Structured LLM request timed out; skipping",
            end_user_id=end_user_id,
            model=model.name,
            timeout_seconds=timeout_seconds,
            _exc_info=exc,
        )
        return None
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


async def create_text_or_none(  # noqa: PLR0913 -- shared best-effort call surface; all params are per-call inputs
    client: AsyncOpenAI,
    model: ModelSettings,
    instructions: str,
    user_text: str,
    end_user_id: str,
    timeout_seconds: float | None = None,
) -> str | None:
    """Returns the trimmed output text from one best-effort `responses.create` call, or None on failure.

    Args:
        client (AsyncOpenAI): OpenAI client pointed at the LiteLLM proxy.
        model (ModelSettings): Model name and reasoning configuration.
        instructions (str): System instructions for the request.
        user_text (str): User message content.
        end_user_id (str): End-user identifier sent in `x-litellm-end-user-id`.
        timeout_seconds (float | None): Optional caller-side wall-clock deadline in seconds.
    """
    try:
        async with asyncio.timeout(delay=timeout_seconds):
            responses = await client.responses.create(
                model=model.name,
                instructions=instructions,
                input=[EasyInputMessageParam(role="user", content=user_text)],
                reasoning=model.reasoning,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": end_user_id},
            )
    except TimeoutError as exc:
        logfire.warn(
            "Text LLM request timed out; using fallback",
            end_user_id=end_user_id,
            model=model.name,
            timeout_seconds=timeout_seconds,
            _exc_info=exc,
        )
        return None
    # Broad on purpose: proxy, transport and SDK errors share no base class.
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
