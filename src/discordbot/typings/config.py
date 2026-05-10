import dotenv
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class DiscordConfig(BaseSettings):
    """Configuration settings for the Discord bot, reading from environment variables.

    Attributes:
        discord_bot_token: The authentication token for the Discord bot.
        discord_test_server_id: Optional ID of a test server for guild-specific commands.
    """

    discord_bot_token: str = Field(
        ...,
        description="The token from discord for calling models.",
        examples=["MTEz-..."],
        validation_alias=AliasChoices("DISCORD_BOT_TOKEN"),
        frozen=False,
        deprecated=False,
    )
    discord_test_server_id: int | None = Field(
        default=None,
        description="The id of the test server for testing the bot.",
        examples=[1143289646042853487, 981592566745149522],
        validation_alias=AliasChoices("DISCORD_TEST_SERVER_ID"),
        frozen=False,
        deprecated=False,
    )


# Guilds where slash commands are registered for instant rollout.
#
# Global slash-command sync (``guild_ids=None``) takes up to an hour to
# propagate; pinning a command to these guilds makes it available immediately.
# Module-level (not on ``DiscordConfig``) on purpose — ``@nextcord.slash_command(...)``
# is a decorator evaluated at import time, so the value has to be a plain
# constant rather than something that needs a ``BaseSettings`` instance.
# Edit this list directly to change the target guilds.
FAST_SYNC_GUILD_IDS: list[int] = [981592566208282634, 1143289646042853487]


__all__ = ["FAST_SYNC_GUILD_IDS", "DiscordConfig"]
