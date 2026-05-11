import re
from typing import Any, Final
import asyncio
import threading

import logfire
from nextcord import Message, DMChannel
from pydantic import BaseModel, ConfigDict, computed_field
from sqlalchemy import Engine, text, event, create_engine
from nextcord.ext import commands

CONTROL_CHARS_RE = re.compile(pattern=r"\x00")

# Single shared engine — putting create_engine() on a per-message
# cached_property leaked the connection pool, dialect cache and inspector
# cache for every Discord message.
_sql_engine: Engine = create_engine(url="sqlite:///data/messages.db")


@event.listens_for(_sql_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Sets WAL mode + a tolerant busy_timeout on every new connection.

    Default rollback-journal mode serializes reads against writes; with this
    DB already in the gigabyte range, any concurrent reader (e.g. analytics)
    would wedge the live logging path. WAL flips that around so reads never
    block on writes. `synchronous=NORMAL` is the right durability trade-off
    in WAL: every commit fsyncs the WAL frame; the main file is fsynced on
    checkpoint, not on every write.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


_MESSAGES_TABLE_LOCK = threading.Lock()
_MESSAGES_TABLE_READY_FOR: Engine | None = None

_CREATE_MESSAGES_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
)

_INSERT_MESSAGE_SQL: Final[str] = """
INSERT INTO messages
    (
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
"""


def _write_row_sync(row: dict[str, str]) -> None:
    """Ensures the canonical messages table exists and inserts one row.

    SQLite writes run off the event loop via `asyncio.to_thread`; the table
    readiness marker is therefore guarded with a thread lock. The marker tracks
    the current engine object so tests can swap `_sql_engine` without leaking
    readiness from a previous temp DB.

    Args:
        row: Mapping matching the schema declared in `_CREATE_MESSAGES_TABLE_SQL`.
    """
    global _MESSAGES_TABLE_READY_FOR  # noqa: PLW0603 -- module-level cache by engine identity

    needs_create = _MESSAGES_TABLE_READY_FOR is not _sql_engine
    if needs_create:
        with _MESSAGES_TABLE_LOCK:
            needs_create = _MESSAGES_TABLE_READY_FOR is not _sql_engine
    with _sql_engine.begin() as conn:
        if needs_create:
            conn.execute(statement=text(text=_CREATE_MESSAGES_TABLE_SQL))
            for statement in _CREATE_MESSAGES_INDEX_SQL:
                conn.execute(statement=text(text=statement))
        conn.execute(statement=text(text=_INSERT_MESSAGE_SQL), parameters=row)

    if needs_create:
        with _MESSAGES_TABLE_LOCK:
            _MESSAGES_TABLE_READY_FOR = _sql_engine


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
    def source_type(self) -> str:
        """The storage source type for this message.

        Returns:
            `"dm"` for direct messages, otherwise `"guild"`.
        """
        if isinstance(self.message.channel, DMChannel):
            return "dm"
        return "guild"

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
        """Persists the message row off the event loop.

        SQLite I/O is synchronous; running it from the coroutine directly
        would block the entire event loop while the WAL frame is fsynced.
        Offloading via `asyncio.to_thread` lets Discord events, LLM streams
        and game settlements keep ticking while the row lands on disk.
        SQLite serializes the threads via its file-level write lock plus
        the connection's `busy_timeout`.
        """
        attachment_paths = await self._save_attachments()
        sticker_paths = await self._save_stickers()
        row: dict[str, str] = {
            "source_type": self.source_type,
            "author": self.sanitize_text(s=self.message.author.name),
            "author_id": str(self.message.author.id),
            "content": self.sanitize_text(s=self.message.content),
            "created_at": self.message.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "channel_name": self.channel_name_or_author_name,
            "channel_id": self.channel_id_or_author_id,
            "attachments": ";".join(attachment_paths),
            "stickers": ";".join(sticker_paths),
        }
        await asyncio.to_thread(_write_row_sync, row=row)

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


def setup(bot: commands.Bot) -> None:
    """Adds the LogMessageCog to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(LogMessageCog(bot))
