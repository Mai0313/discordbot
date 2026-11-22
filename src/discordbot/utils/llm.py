"""Shared best-effort Responses API helpers for one-shot LLM calls.

Each helper degrades errors to None; callers map None to their own fallback and construct their
own `AsyncOpenAI` clients. On `create_text_or_none`, leave `timeout_seconds=None` unless the
caller owns a tighter deadline than the client's own timeout (`typings/timeouts.py`).
"""

import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import BaseModel, ValidationError
from openai.types.responses import Response
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.models import ModelSettings


def output_text_or_empty(responses: Response) -> str:
    """Returns concatenated non-empty `output_text` parts from `responses`, or `""` if none.

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
) -> StructuredT | None:
    """Returns the parsed `StructuredT` from `responses.parse`, or None on failure.

    An empty output or off-schema payload (`ValidationError`), a refusal (`output_parsed` is None),
    an incomplete response, or any other error all degrade to None.

    Args:
        client (AsyncOpenAI): OpenAI client pointed at the LiteLLM proxy.
        model (ModelSettings): Model name and reasoning configuration.
        instructions (str): System instructions for the request.
        user_text (str): User message content.
        end_user_id (str): End-user identifier sent in `x-litellm-end-user-id`.
        text_format (type[StructuredT]): Pydantic model class to parse the output into.
    """
    try:
        responses = await client.responses.parse(
            model=model.name,
            instructions=instructions,
            input=[EasyInputMessageParam(role="user", content=user_text)],
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
    # Broad on purpose: best-effort structured call where every `APIError` subclass (status,
    # transport, timeout) and any unexpected runtime error degrades to None.
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
    """Returns trimmed output text from `responses.create`, or None on failure.

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
    # Broad on purpose: best-effort text call where every `APIError` subclass (status, transport,
    # timeout) and any unexpected runtime error degrades to None.
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
