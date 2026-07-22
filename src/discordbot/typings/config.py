from typing import Literal

import dotenv
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class DiscordConfig(BaseSettings):
    """Configuration settings for the Discord bot, reading from environment variables.

    Attributes:
        discord_bot_token: The authentication token for the Discord bot.
    """

    discord_bot_token: str = Field(
        ...,
        description="The token from discord for calling models.",
        examples=["MTEz-..."],
        validation_alias=AliasChoices("DISCORD_BOT_TOKEN"),
    )


class EconomyConfig(BaseSettings):
    """Economy feature settings loaded from environment variables."""

    allow_central_bank_self_approval: bool = Field(
        False,
        description="Allow central-bank borrowers to approve their own loan requests for local testing.",
        examples=[False],
        validation_alias=AliasChoices("ECONOMY_ALLOW_CENTRAL_BANK_SELF_APPROVAL"),
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


__all__ = ["DiscordConfig", "EconomyConfig", "LoggingConfig"]
