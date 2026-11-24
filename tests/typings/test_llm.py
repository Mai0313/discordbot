"""Tests for the LLM settings read from the environment."""

from discordbot.typings.llm import LLMConfig


def test_gemini_key_configured_rejects_blank_key() -> None:
    """`gemini_key_configured` is true only when `gemini_api_key` contains non-whitespace."""
    config = LLMConfig.model_construct(gemini_api_key="AIza-key")
    assert config.gemini_key_configured is True
    config.gemini_api_key = "   "
    assert config.gemini_key_configured is False
    config.gemini_api_key = ""
    assert config.gemini_key_configured is False
