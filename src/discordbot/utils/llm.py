"""Shared best-effort Responses API call surfaces for one-shot LLM calls.

Three helpers live here. `parse_responses_or_none` runs one structured `responses.parse` and
`create_text_or_none` one plain-text `responses.create`; `output_text_or_empty` aggregates a
finished response's text around an SDK quirk, and is borrowed by the one caller that owns its own
call surface but hits the same quirk (`gen_reply/generation.py`'s prompt director, which needs
grounding tools and image parts).

Both call helpers promise the same thing: each owns the proxy call surface (the model string and
its reasoning options, `service_tier`, the LiteLLM end-user header and the `mock_testing_fallbacks`
opt-out), the per-call timeout, and the failure handling, and returns None instead of raising.
That is what the callers come here for — each maps None to its own deterministic outcome (a
template headline, the brief's first line, keeping the previous memory state, or simply not posting
the auto-unmute gripe), so a failed one-shot call degrades there rather than reaching the feature's
error path.

What these deliberately do not do: build the client (the caller supplies the `AsyncOpenAI`, so
this module builds none itself and reads no configuration), retry, stream, or carry anything
richer than one `instructions` plus one user-role text message. A path that needs history,
attachments, tools or deltas owns its own call surface instead — `gen_reply/streaming.py` for the
answer turn, `gen_reply/generation.py` for image / voice / music / video, `cogs/research/` for the
direct-to-Google Interactions API.

It sits below the cogs because its callers span layers that may not import one another: the
`stock`, `research` and `auto_unmute` cogs, `services/memory/extraction.py`, and `gen_reply`.
"""

from typing import cast
import asyncio

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

    Output items and content parts are discriminated on their `.type` literal rather than by
    isinstance against the SDK classes, the same way `gen_reply/streaming.py::_consume`
    discriminates its stream events. Both runtime callers hand over a real `Response`, so here it
    buys only the `SimpleNamespace` stand-in `tests/test_llm.py` casts into one.

    Args:
        responses (Response): The finished response whose output items are aggregated.

    Returns:
        Every non-empty output_text part joined in order, or "" when the turn carried none.
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
    timeout_seconds: float,
) -> StructuredT | None:
    """Runs one best-effort structured Responses.parse call, returning None on any failure.

    Owns the shared proxy call surface, the timeout, and the failure handling so each caller only
    maps None to its own fallback. Every outcome short of a parsed payload degrades the same way:
    a timeout; an empty or off-schema body, both of which arrive as `ValidationError` because the
    SDK validates every output_text part against `text_format`; a response reported `incomplete`,
    rejected after the parse already succeeded; and any transport or proxy-side error.

    Args:
        client (AsyncOpenAI): Proxy-backed client this call is dispatched on.
        model (ModelSettings): Tier supplying the model string and its reasoning options.
        instructions (str): Developer-authority prompt for this turn.
        user_text (str): Body of the single user-role message.
        end_user_id (str): Sent as the LiteLLM `x-litellm-end-user-id` header and logged on
            failure; callers pass a per-feature label rather than a Discord identity.
        text_format (type[StructuredT]): Schema the reply is parsed into.
        timeout_seconds (float): Wall-clock budget for the whole call.

    Returns:
        The parsed `text_format` instance, or None on timeout, failure, an incomplete response, or
        a turn that carried no parsed output_text part at all (where a bare refusal lands).
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
    *,
    client: AsyncOpenAI,
    model: ModelSettings,
    instructions: str,
    user_text: str,
    end_user_id: str,
    timeout_seconds: float,
) -> str | None:
    """Runs one best-effort text Responses.create call, returning None on any failure.

    Mirrors `parse_responses_or_none` for the non-structured callers: owns the shared proxy call
    surface, the timeout, and the failure handling, so each caller maps None to its own fallback
    line. The text is aggregated through `output_text_or_empty` rather than read off
    `responses.output_text`, so a part carrying `text=None` cannot turn a delivered answer into a
    raised `TypeError`.

    Args:
        client (AsyncOpenAI): Proxy-backed client this call is dispatched on.
        model (ModelSettings): Tier supplying the model string and its reasoning options.
        instructions (str): Developer-authority prompt for this turn.
        user_text (str): Body of the single user-role message.
        end_user_id (str): Sent as the LiteLLM `x-litellm-end-user-id` header and logged on
            failure; callers pass a per-feature label rather than a Discord identity.
        timeout_seconds (float): Wall-clock budget for the whole call.

    Returns:
        The stripped output text, "" when the turn produced none, or None on timeout or any other
        failure.
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
    except TimeoutError as exc:
        logfire.warn(
            "Text LLM request timed out; using fallback",
            end_user_id=end_user_id,
            model=model.name,
            timeout_seconds=timeout_seconds,
            _exc_info=exc,
        )
        return None
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
