import re
import asyncio

import pandas as pd
import logfire
from nextcord import Message, DMChannel
from pydantic import BaseModel, ConfigDict, computed_field
from sqlalchemy import Engine, create_engine
from nextcord.ext import commands

CONTROL_CHARS_RE = re.compile(r"\x00")

# Single shared engine — putting create_engine() on a per-message
# cached_property leaked the connection pool, dialect cache and inspector
# cache for every Discord message.
_sql_engine: Engine = create_engine(url="sqlite:///data/messages.db")


class MessageLogger(BaseModel):
    """Persists a Discord message and its metadata to SQLite.

    Attributes:
        message: The Discord message being logged.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    message: Message

    @staticmethod
    def sanitize_text(s: str | None) -> str:
        """Sanitizes text by removing control characters (null bytes).

        Args:
            s: The string to sanitize.

        Returns:
            The sanitized string, or an empty string if input was None.
        """
        if s is None:
            return ""
        return CONTROL_CHARS_RE.sub("", s)

    @computed_field
    @property
    def table_name(self) -> str:
        """The database table name for this message.

        Returns:
            A DM-specific table name for direct messages, otherwise a
            channel-specific table name.
        """
        if isinstance(self.message.channel, DMChannel):
            return f"DM_{self.message.author.id}"
        return f"channel_{self.message.channel.id}"

    @computed_field
    @property
    def channel_name_or_author_name(self) -> str:
        """The channel name or DM author label for this message.

        Returns:
            A label containing the DM author display name and ID for direct
            messages, otherwise the channel name and ID.
        """
        if isinstance(self.message.channel, DMChannel):
            author_name = self.message.author.display_name
            return f"DM_{author_name}_{self.message.author.id}"
        return f"channel_{self.message.channel.name}_{self.message.channel.id}"

    @computed_field
    @property
    def channel_id_or_author_id(self) -> str:
        """The channel ID or DM author ID for this message.

        Returns:
            The author ID for direct messages, otherwise the channel ID.
        """
        if isinstance(self.message.channel, DMChannel):
            return f"{self.message.author.id}"
        return f"{self.message.channel.id}"

    async def _save_attachments(self) -> list[str]:
        """Extracts attachment URLs from the message."""
        attachment_urls = []
        for attachment in self.message.attachments:
            attachment_urls.append(attachment.url)
        return attachment_urls

    async def _save_stickers(self) -> list[str]:
        """Extracts sticker URLs from the message."""
        sticker_urls = []
        for sticker in self.message.stickers:
            sticker_urls.append(sticker.url)
        return sticker_urls

    async def _save_messages(self) -> None:
        """Saves the message data to the SQLite database."""
        attachment_paths = await self._save_attachments()
        sticker_paths = await self._save_stickers()
        data_dict = {
            "author": self.sanitize_text(self.message.author.name),
            "author_id": self.channel_id_or_author_id,
            "content": self.sanitize_text(self.message.content),
            "created_at": self.message.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "channel_name": self.channel_name_or_author_name,
            "channel_id": self.channel_id_or_author_id,
            "attachments": ";".join(attachment_paths),
            "stickers": ";".join(sticker_paths),
        }
        messages = pd.DataFrame([data_dict]).astype(str)

        messages.to_sql(
            name=f"{self.table_name}",
            con=_sql_engine,
            if_exists="append",
            index=False,
            chunksize=10_000,
            method="multi",
        )

    async def log(self) -> None:
        """Logs the message if it's not from a bot."""
        try:
            if self.message.author.bot:
                return
            await self._save_messages()
        except Exception:
            logfire.error("Failed to log message", _exc_info=True)


class LogMessageCog(commands.Cog):
    """Logs Discord messages and completed command messages.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the LogMessageCog instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and logs them asynchronously.

        Args:
            message: The message that was sent.
        """
        if message.author.bot:
            return
        asyncio.create_task(MessageLogger(message=message).log())  # noqa: RUF006

    @commands.Cog.listener()
    async def on_command_completion(self, context: commands.Context) -> None:
        """Listens for command completions and logs the message that triggered it.

        Args:
            context: The context of the command.
        """
        asyncio.create_task(MessageLogger(message=context.message).log())  # noqa: RUF006


async def setup(bot: commands.Bot) -> None:
    """Adds the LogMessageCog to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(LogMessageCog(bot))
