import dotenv
from pydantic import Field, AliasChoices, field_validator
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
        frozen=False,
        deprecated=False,
    )


_MEMORY_ENABLED_DEFAULT = True
_MEMORY_CLEAR_ENABLED_DEFAULT = False


class MemoryConfig(BaseSettings):
    """Per-user long-term memory settings loaded from environment variables."""

    enabled: bool = Field(
        _MEMORY_ENABLED_DEFAULT,
        description="Master switch for per-user memory injection and background extraction.",
        examples=[True],
        validation_alias=AliasChoices("MEMORY_ENABLED"),
        frozen=False,
        deprecated=False,
    )
    clear_enabled: bool = Field(
        _MEMORY_CLEAR_ENABLED_DEFAULT,
        description="Whether `/memory clear` actually deletes memory; off keeps the command visible but paused.",
        examples=[False],
        validation_alias=AliasChoices("MEMORY_CLEAR_ENABLED"),
        frozen=False,
        deprecated=False,
    )

    @field_validator("enabled", mode="before")
    @classmethod
    def _blank_env_means_default(cls, value: object) -> object:
        """Treats a blank `MEMORY_ENABLED=` env line as the default instead of failing cog load."""
        if isinstance(value, str) and not value.strip():
            return _MEMORY_ENABLED_DEFAULT
        return value

    @field_validator("clear_enabled", mode="before")
    @classmethod
    def _blank_clear_env_means_default(cls, value: object) -> object:
        """Treats a blank `MEMORY_CLEAR_ENABLED=` env line as the default instead of failing cog load."""
        if isinstance(value, str) and not value.strip():
            return _MEMORY_CLEAR_ENABLED_DEFAULT
        return value


class EconomyConfig(BaseSettings):
    """Economy feature settings loaded from environment variables."""

    allow_central_bank_self_approval: bool = Field(
        False,
        description="Allow central-bank borrowers to approve their own loan requests for local testing.",
        examples=[False],
        validation_alias=AliasChoices("ECONOMY_ALLOW_CENTRAL_BANK_SELF_APPROVAL"),
        frozen=False,
        deprecated=False,
    )


__all__ = ["DiscordConfig", "EconomyConfig", "MemoryConfig"]
