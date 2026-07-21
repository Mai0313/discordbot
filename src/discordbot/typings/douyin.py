"""Runtime configuration for the Douyin link features."""

import dotenv
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv()


class DouyinConfig(BaseSettings):
    """Configuration for Douyin link auto-expansion, read from environment variables.

    Kept apart from `LLMConfig` because auto-expansion is not an LLM feature: it downloads a
    post and posts it back. The reply pipeline's own Douyin switch lives with the other model
    kill-switches instead.

    Attributes:
        auto_expand_enabled: Kill-switch for expanding a pasted Douyin link into the channel.
            The one lever that stops the bot talking to Douyin at all if its WAF starts
            blocking, since expansion turns every pasted link into a request.
    """

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    auto_expand_enabled: bool = Field(
        default=True,
        description="Whether a Douyin link pasted in chat is expanded into the channel.",
        validation_alias=AliasChoices("DOUYIN_AUTO_EXPAND_ENABLED"),
    )
