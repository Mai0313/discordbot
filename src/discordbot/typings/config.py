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


__all__ = ["DiscordConfig", "EconomyConfig"]
