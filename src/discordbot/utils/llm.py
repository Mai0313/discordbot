"""Factory for the runtime LiteLLM-proxy OpenAI client."""

from openai import AsyncOpenAI

from discordbot.typings.llm import LLMConfig


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
