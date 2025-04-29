from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class OpenAIConfig(BaseSettings):
    api_key: str = Field(
        ...,
        description="The api key from openai for calling models.",
        examples=["sk-proj-...", "141698ac..."],
        alias="OPENAI_API_KEY",
        frozen=False,
        deprecated=False,
    )
    api_type: str = Field(
        ...,
        description="The api type from openai for calling models.",
        examples=["openai", "azure"],
        alias="OPENAI_API_TYPE",
        frozen=False,
        deprecated=False,
    )
    api_version: str = Field(
        default="2024-12-01-preview",
        description="The api version from openai for calling models.",
        examples=["2024-12-01-preview"],
        alias="OPENAI_API_VERSION",
        frozen=False,
        deprecated=False,
    )
    api_endpoint: str = Field(
        default="https://api.openai.com/v1",
        description="The base url from openai for calling models.",
        examples=["https://api.openai.com/v1", "https://xxxx.openai.azure.com"],
        alias="OPENAI_API_ENDPOINT",
        frozen=False,
        deprecated=False,
    )

    def get_llm_config(self, model: str) -> dict[str, Any]:
        llm_config = {
            "timeout": 60,
            "temperature": 0,
            "cache_seed": None,
            "config_list": [
                {
                    "model": model,
                    "api_key": self.api_key,
                    "base_url": self.api_endpoint,
                    "api_type": self.api_type,
                    "api_version": self.api_version,
                }
            ],
        }
        return llm_config


class PerplexityConfig(BaseSettings):
    pplx_api_key: str = Field(
        ...,
        description="The api key from perplexity for calling models.",
        examples=["pplx-..."],
        alias="PERPLEXITY_API_KEY",
        frozen=False,
        deprecated=False,
    )


class DiscordConfig(BaseSettings):
    discord_bot_token: str = Field(
        ...,
        description="The token from discord for calling models.",
        examples=["MTEz-..."],
        alias="DISCORD_BOT_TOKEN",
        frozen=False,
        deprecated=False,
    )
    discord_test_server_id: Optional[str] = Field(
        default=None,
        description="The id of the test server for testing the bot.",
        examples=["1143289646042853487", "981592566745149522"],
        alias="DISCORD_TEST_SERVER_ID",
        frozen=False,
        deprecated=False,
    )


class Config(OpenAIConfig, PerplexityConfig, DiscordConfig):
    pass


__all__ = ["Config"]
