"""Factories for the runtime LiteLLM-proxy OpenAI client and the Gemini upload client."""

from typing import Any

from google import genai
from openai import AsyncOpenAI

from discordbot.typings.llm import LLMConfig


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
