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


# Per-channel tables are created on first sight. The previous pandas-backed
# write path did this implicitly via reflection on every call, paying ~1-3 ms
# per message rebuilding a DataFrame and re-introspecting the schema. The
# cache below replaces that with a one-shot CREATE TABLE IF NOT EXISTS the
# first time we see a table name; subsequent writes go straight to INSERT.
#
# The set is mutated from worker threads (writes are offloaded via
# `asyncio.to_thread`), so guard it with a thread-safe Lock — `asyncio.Lock`
# would bind to a single event loop and break under cross-thread access.
_TABLE_INIT_LOCK = threading.Lock()
_INITIALIZED_TABLES: set[str] = set()

# Schema mirrors what pandas used to produce for `.astype(str)` data —
# 8 TEXT columns — so the 1.3 GB of existing rows continues to round-trip
# without surprises. SQLite's dynamic typing means even the older tables
# whose pandas-inferred schema had BIGINT columns happily accept TEXT
# inserts via type affinity.
_CREATE_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS "{table}" (
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

_INSERT_SQL: Final[str] = """
INSERT INTO "{table}"
    (author, author_id, content, created_at, channel_name, channel_id, attachments, stickers)
VALUES
    (:author, :author_id, :content, :created_at, :channel_name, :channel_id, :attachments, :stickers)
"""


def _write_row_sync(table_name: str, row: dict[str, str]) -> None:
    """Ensures the table exists and inserts one row.

    Both statements run inside one transaction so a freshly-created table
    cannot disappear (e.g. via concurrent VACUUM) before the INSERT. The
    table-init cache makes the CREATE statement effectively free after the
    first write per table.

    Args:
        table_name: Destination table; safe because callers derive it from
            integer channel/author IDs only.
        row: 8-key mapping matching the schema declared in `_CREATE_TABLE_SQL`.
    """
    needs_create = table_name not in _INITIALIZED_TABLES
    if needs_create:
        with _TABLE_INIT_LOCK:
            needs_create = table_name not in _INITIALIZED_TABLES

    with _sql_engine.begin() as conn:
        if needs_create:
            conn.execute(text(_CREATE_TABLE_SQL.format(table=table_name)))
        conn.execute(text(_INSERT_SQL.format(table=table_name)), row)

    if needs_create:
        with _TABLE_INIT_LOCK:
            _INITIALIZED_TABLES.add(table_name)


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
            "author": self.sanitize_text(s=self.message.author.name),
            "author_id": self.channel_id_or_author_id,
            "content": self.sanitize_text(s=self.message.content),
            "created_at": self.message.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "channel_name": self.channel_name_or_author_name,
            "channel_id": self.channel_id_or_author_id,
            "attachments": ";".join(attachment_paths),
            "stickers": ";".join(sticker_paths),
        }
        await asyncio.to_thread(_write_row_sync, self.table_name, row)

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
