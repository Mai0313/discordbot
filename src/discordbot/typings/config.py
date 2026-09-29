"""Environment-backed configuration for the pieces that are not a feature of their own."""

from typing import Literal

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings


class DiscordConfig(BaseSettings):
    """Configuration settings for the Discord bot, reading from environment variables."""

    discord_bot_token: str = Field(
        ...,
        description="The token from Discord the bot authenticates its gateway session with.",
        examples=["MTEz-..."],
        validation_alias=AliasChoices("DISCORD_BOT_TOKEN"),
    )


class LoggingConfig(BaseSettings):
    """Console and log-file verbosity, loaded from environment variables."""

    log_level: Literal["trace", "debug", "info", "notice", "warn", "warning", "error", "fatal"] = (
        Field(
            "debug",
            description="Lowest severity written to the console and to ./data/logs. Defaults to debug so the log file keeps the full trace; raise it to info on a deployment that only wants outcomes.",
            examples=["debug", "info"],
            validation_alias=AliasChoices("LOG_LEVEL"),
        )
    )


__all__ = ["DiscordConfig", "LoggingConfig"]
