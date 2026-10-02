"""Message logging cog backed by the local SQLite message store."""

from typing import TYPE_CHECKING, Final

import logfire
from nextcord import Message, DMChannel
from pydantic import Field, BaseModel, ConfigDict
from sqlalchemy import MetaData, text
from nextcord.ext import commands
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncConnection, create_async_engine

from discordbot.utils.asyncio_locks import spawn_tracked
from discordbot.utils.sqlite_config import SqliteBootstrap

if TYPE_CHECKING:
    import asyncio

# Single shared engine, never a per-message `cached_property`: that leaks the
# connection pool, dialect cache and inspector cache once per Discord message.
_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/messages.db")

_CREATE_MESSAGES_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discord_message_id TEXT,
    source_type TEXT NOT NULL,
    author TEXT,
    author_id TEXT,
    content TEXT,
    created_at TEXT,
    channel_name TEXT,
    channel_id TEXT,
    attachments TEXT,
    stickers TEXT
)
"""

_CREATE_MESSAGES_INDEX_SQL: Final[tuple[str, ...]] = (
    "CREATE INDEX IF NOT EXISTS ix_messages_created_at ON messages(created_at)",
    "CREATE INDEX IF NOT EXISTS ix_messages_channel_id_created_at "
    "ON messages(channel_id, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_messages_author_id_created_at "
    "ON messages(author_id, created_at)",
    # Partial unique index gives the UPSERT below a conflict target while
    # leaving legacy NULL-id rows untouched.
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_messages_discord_message_id "
    "ON messages(discord_message_id) WHERE discord_message_id IS NOT NULL",
)

# UPSERT: streaming bot replies edit themselves several times after the initial
# `reply()`, so each `on_message_edit` re-fires this INSERT with the same
# `discord_message_id`. The conflict on the partial unique index turns the
# repeat write into an UPDATE so messages.db converges to the final on-Discord
# state. `created_at` is intentionally NOT touched — the original send-time stays
# pinned even as content / attachments mutate.
_INSERT_MESSAGE_SQL: Final[str] = """
INSERT INTO messages
    (
        discord_message_id,
        source_type,
        author,
        author_id,
        content,
        created_at,
        channel_name,
        channel_id,
        attachments,
        stickers
    )
VALUES
    (
        :discord_message_id,
        :source_type,
        :author,
        :author_id,
        :content,
        :created_at,
        :channel_name,
        :channel_id,
        :attachments,
        :stickers
    )
ON CONFLICT (discord_message_id) WHERE discord_message_id IS NOT NULL DO UPDATE SET
    content = excluded.content,
    attachments = excluded.attachments,
    stickers = excluded.stickers
"""


async def _create_messages_table(conn: AsyncConnection) -> None:
    """Creates the messages table and its indexes, each only where it does not exist yet.

    A fresh file gets the columns and indexes the deployed `messages.db` has; an existing table
    is never altered. Raw DDL rather than a declared model: `create_all` never alters one either,
    the deployed file is in the gigabyte range with legacy NULL-id rows, and the DDL a model
    compiles to differs from this text, so a model would be a second definition of the table
    that nothing checks against the first.
    """
    await conn.execute(statement=text(text=_CREATE_MESSAGES_TABLE_SQL))
    for statement in _CREATE_MESSAGES_INDEX_SQL:
        await conn.execute(statement=text(text=statement))


# No declared tables: the one table and its indexes come from `_create_messages_table`.
_database = SqliteBootstrap(metadata=MetaData(), after_create=_create_messages_table)


async def _write_row(row: dict[str, str]) -> None:
    """Inserts one row, or folds a repeat of the same Discord message into its row.

    Args:
        row: Mapping matching the schema declared in `_CREATE_MESSAGES_TABLE_SQL`.
    """
    async with _database.open_session(engine=_engine) as session, session.begin():
        await session.execute(statement=text(text=_INSERT_MESSAGE_SQL), params=row)


class MessageLogger(BaseModel):
    """Persists a Discord message and its metadata to SQLite."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    message: Message = Field(..., description="The Discord message being logged.")

    @staticmethod
    def sanitize_text(s: str) -> str:
        """Sanitizes text by removing null bytes.

        Args:
            s: The string to sanitize.

        Returns:
            The sanitized string.
        """
        return s.replace("\x00", "")

    @property
    def source_type(self) -> str:
        """The storage source type for this message.

        Returns:
            `"dm"` for direct messages, otherwise `"guild"`.
        """
        if isinstance(self.message.channel, DMChannel):
            return "dm"
        return "guild"

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
        channel = self.message.channel
        channel_name = getattr(channel, "name", None) or channel.id
        return f"channel_{channel_name}_{channel.id}"

    @property
    def channel_id_or_author_id(self) -> str:
        """The channel ID or DM author ID for this message.

        Returns:
            The author ID for direct messages, otherwise the channel ID.
        """
        if isinstance(self.message.channel, DMChannel):
            return f"{self.message.author.id}"
        return f"{self.message.channel.id}"

    async def log(self) -> None:
        """Persists the message row.

        Author filtering (human or this bot's own reply) lives in
        `LogMessageCog` so this method stays generic and is safe to call from
        anywhere that already knows the message is loggable.
        """
        try:
            attachment_paths = [attachment.url for attachment in self.message.attachments]
            sticker_paths = [sticker.url for sticker in self.message.stickers]
            row: dict[str, str] = {
                "discord_message_id": str(self.message.id),
                "source_type": self.source_type,
                "author": self.sanitize_text(s=self.message.author.name),
                "author_id": str(self.message.author.id),
                "content": self.sanitize_text(s=self.message.content),
                "created_at": self.message.created_at.strftime(format="%Y-%m-%d %H:%M:%S"),
                "channel_name": self.channel_name_or_author_name,
                "channel_id": self.channel_id_or_author_id,
                "attachments": ";".join(attachment_paths),
                "stickers": ";".join(sticker_paths),
            }
            await _write_row(row=row)
        except Exception as exc:
            # Stays broad: this runs as a detached task, and this log carries the message's
            # ids, which the spawner's generic failure line does not.
            logfire.error(
                "Failed to log message",
                discord_message_id=self.message.id,
                channel_id=self.channel_id_or_author_id,
                source_type=self.source_type,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )


class LogMessageCog(commands.Cog):
    """Logs Discord messages and their later edits.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the LogMessageCog instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self._tasks: set[asyncio.Task[None]] = set()

    def _should_log(self, message: Message) -> bool:
        """Returns True for human messages or this bot's own replies.

        Third-party bots (e.g. other Discord apps sharing the guild) are
        deliberately skipped so messages.db tracks only the conversation
        participants this bot actually engages with — its users and itself.
        """
        if not message.author.bot:
            return True
        return bool(self.bot.user and message.author.id == self.bot.user.id)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and logs them asynchronously.

        Args:
            message: The message that was sent.
        """
        if not self._should_log(message=message):
            return
        spawn_tracked(
            coro=MessageLogger(message=message).log(), tasks=self._tasks, name="log-message"
        )

    @commands.Cog.listener()
    async def on_message_edit(self, _before: Message, after: Message) -> None:
        """Re-logs message edits so streaming bot replies converge to their final state.

        `on_message` only fires on the initial send, so a reply that streams
        its content in by editing itself would otherwise be logged as whatever
        it happened to hold first. Every subsequent edit fires here, and the
        UPSERT on `discord_message_id` collapses them into a single row whose
        content matches what is actually on Discord.

        Args:
            _before: The pre-edit message snapshot (unused; only `after.id`
                matters for the UPSERT key).
            after: The current message state.
        """
        if not self._should_log(message=after):
            return
        spawn_tracked(
            coro=MessageLogger(message=after).log(), tasks=self._tasks, name="log-message"
        )


def setup(bot: commands.Bot) -> None:
    """Adds the LogMessageCog to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(LogMessageCog(bot), override=True)
