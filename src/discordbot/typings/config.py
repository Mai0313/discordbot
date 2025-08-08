import dotenv
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class DiscordConfig(BaseSettings):
    discord_bot_token: str = Field(
        ...,
        description="The token from discord for calling models.",
        examples=["MTEz-..."],
        validation_alias=AliasChoices("DISCORD_BOT_TOKEN"),
        frozen=False,
        deprecated=False,
    )
    discord_test_server_id: str | None = Field(
        default=None,
        description="The id of the test server for testing the bot.",
        examples=["1143289646042853487", "981592566745149522"],
        validation_alias=AliasChoices("DISCORD_TEST_SERVER_ID"),
        frozen=False,
        deprecated=False,
    )


__all__ = ["DiscordConfig"]
