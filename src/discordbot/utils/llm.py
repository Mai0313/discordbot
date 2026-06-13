"""Factories for the runtime LiteLLM-proxy OpenAI client and the Gemini upload client."""

from typing import Any, cast
import asyncio

from google import genai
from openai import AsyncOpenAI
import logfire
from pydantic import BaseModel, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings


def litellm_call_kwargs(end_user_id: str) -> dict[str, Any]:
    """Returns the shared kwargs every runtime Responses call passes to the LiteLLM proxy.

    Centralizes the auto service tier, the per-end-user header LiteLLM keys spend and
    rate-limit tracking on, and the flag that disables the proxy's mock fallbacks, so a
    proxy-wide change lives here instead of being re-typed at every `responses.*` call site.
    Spread it as `**litellm_call_kwargs(end_user_id=...)` next to the per-call model,
    instructions, input, and reasoning. Returns a fresh dict each call so a caller can never
    mutate shared state.
    """
    return {
        "service_tier": "auto",
        "extra_headers": {"x-litellm-end-user-id": end_user_id},
        "extra_body": {"mock_testing_fallbacks": False},
    }


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

    Owns the shared call surface (`litellm_call_kwargs`), the timeout, and the failure
    handling so each caller only maps None to its own fallback: a timeout, an empty or
    refused output (`ValidationError` from parsing no text), an incomplete (truncated)
    response, or any other error all degrade to None.
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
                **litellm_call_kwargs(end_user_id=end_user_id),
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

    Mirrors `parse_responses_or_none` for the non-structured callers: owns the shared call
    surface, the timeout, and the failure handling, and returns the trimmed output text (or
    None on timeout / any error) so each caller maps None to its own fallback line.
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
                **litellm_call_kwargs(end_user_id=end_user_id),
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


def create_litellm_client(config: LLMConfig) -> AsyncOpenAI:
    """Returns a fresh AsyncOpenAI client pointed at the LiteLLM proxy.

    Each cog keeps its own client instance; this factory only centralizes the
    base-url / api-key wiring so proxy configuration lives in one place.

    Args:
        config: Runtime LLM configuration holding the proxy base URL and API key.

    Returns:
        A configured OpenAI-compatible client for the LiteLLM proxy.
    """
    return AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)


def create_gemini_client(config: LLMConfig) -> genai.Client:
    """Returns a Gemini client for direct Files API uploads.

    Attachment ingestion uploads through this client (not the LiteLLM proxy) so
    a fresh upload can be polled to an ACTIVE `state` before it is referenced;
    the proxy's file resource cannot report that readiness. The answer request
    still references the uploaded file by its URI through the proxy.

    Args:
        config: Runtime LLM configuration holding the Google AI Studio key.

    Returns:
        A Gemini client authenticated with the configured Files API credential.
    """
    return genai.Client(api_key=config.gemini_api_key)
