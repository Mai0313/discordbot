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
    # Partial unique index gives the upsert below a conflict target while
    # leaving `NULL`-id rows untouched.
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_messages_discord_message_id "
    "ON messages(discord_message_id) WHERE discord_message_id IS NOT NULL",
)

# Streaming bot replies edit themselves after the initial send, so each `on_message_edit`
# re-fires this `INSERT` with the same `discord_message_id`. The conflict on the partial
# unique index updates content, attachments, and stickers in place while keeping the
# original `created_at`.
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
    """Creates the `messages` table and its indexes if they do not exist.

    Uses raw DDL rather than a declarative model so the schema definition stays in one place
    and existing tables with `NULL`-id rows are never altered.
    """
    await conn.execute(statement=text(text=_CREATE_MESSAGES_TABLE_SQL))
    for statement in _CREATE_MESSAGES_INDEX_SQL:
        await conn.execute(statement=text(text=statement))


# No declared tables: the one table and its indexes come from `_create_messages_table`.
_database = SqliteBootstrap(metadata=MetaData(), after_create=_create_messages_table)


async def _write_row(row: dict[str, str]) -> None:
    """Upserts one message row matching `_CREATE_MESSAGES_TABLE_SQL`."""
    async with _database.open_session(engine=_engine) as session, session.begin():
        await session.execute(statement=text(text=_INSERT_MESSAGE_SQL), params=row)


class MessageLogger(BaseModel):
    """Persists a Discord message and its metadata to SQLite."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    message: Message = Field(..., description="The Discord message being logged.")

    @staticmethod
    def sanitize_text(s: str) -> str:
        """Returns `s` with null bytes removed.

        Args:
            s (str): Text to sanitize.
        """
        return s.replace("\x00", "")

    @property
    def source_type(self) -> str:
        """Returns `"dm"` for direct messages, or `"guild"` otherwise."""
        if isinstance(self.message.channel, DMChannel):
            return "dm"
        return "guild"

    @property
    def channel_name_or_author_name(self) -> str:
        """Returns the DM author label (`DM_<name>_<id>`) or channel label (`channel_<name>_<id>`)."""
        if isinstance(self.message.channel, DMChannel):
            author_name = self.message.author.display_name
            return f"DM_{author_name}_{self.message.author.id}"
        channel = self.message.channel
        channel_name = getattr(channel, "name", None) or channel.id
        return f"channel_{channel_name}_{channel.id}"

    @property
    def channel_id_or_author_id(self) -> str:
        """Returns the author ID for direct messages, or the channel ID otherwise."""
        if isinstance(self.message.channel, DMChannel):
            return f"{self.message.author.id}"
        return f"{self.message.channel.id}"

    async def log(self) -> None:
        """Persists the message row without filtering by author."""
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
        bot (commands.Bot): The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the cog.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot
        self._tasks: set[asyncio.Task[None]] = set()

    def _should_log(self, message: Message) -> bool:
        """Returns whether `message` is from a human or from this bot itself."""
        if not message.author.bot:
            return True
        return bool(self.bot.user and message.author.id == self.bot.user.id)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Logs eligible new messages in the background.

        Args:
            message (Message): The message that was sent.
        """
        if not self._should_log(message=message):
            return
        spawn_tracked(
            coro=MessageLogger(message=message).log(), tasks=self._tasks, name="log-message"
        )

    @commands.Cog.listener()
    async def on_message_edit(self, _before: Message, after: Message) -> None:
        """Re-logs message edits so streaming bot replies converge to their final state.

        Args:
            _before (Message): Pre-edit message snapshot (unused).
            after (Message): Updated message state to upsert by `discord_message_id`.
        """
        if not self._should_log(message=after):
            return
        spawn_tracked(
            coro=MessageLogger(message=after).log(), tasks=self._tasks, name="log-message"
        )


def setup(bot: commands.Bot) -> None:
    """Adds `LogMessageCog` to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(LogMessageCog(bot), override=True)
