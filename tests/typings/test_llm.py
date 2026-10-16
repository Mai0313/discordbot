"""Tests for the LLM settings read from the environment."""

from discordbot.typings.llm import LLMConfig


def test_deep_research_available_requires_enabled_and_key() -> None:
    config = LLMConfig.model_construct(deep_research_enabled=True, gemini_api_key="AIza-key")
    assert config.deep_research_available is True
    config.gemini_api_key = "   "
    assert config.deep_research_available is False
    config.gemini_api_key = "AIza-key"
    config.deep_research_enabled = False
    assert config.deep_research_available is False
