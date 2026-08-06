"""Archives the bot's own conversation into `data/database/messages.db`, and owns that store.

The Discord surface is three listeners and nothing else: no command, no kill-switch, no
permission gate, and nothing written back to the channel, so a user never sees this cog work or
fail. `on_message` and `on_message_edit` both filter through `_should_log` — human authors plus
this bot's own messages, every third-party app sharing the guild dropped — while
`on_command_completion` covers the prefix-command path and fires for nothing today. Each of the
three schedules a detached `MessageLogger.log()` task, which is why every one of them carries a
`noqa: RUF006`: the write must not hold up the gateway's dispatch, and nothing is left to await
its result.

Below that surface this file owns the whole store: the module-level `Engine`, the `messages`
table, its indexes and the one INSERT. A single flat table holds every source, with
`source_type` separating a guild row from a DM row and `channel_id` carrying the peer's user id
on a DM so those rows group by person rather than by channel. `on_message_edit` is load-bearing
rather than tidy: a streamed reply is created near-empty and grows through repeated
`reply.edit(...)`, so the write is an UPSERT on `discord_message_id` that converges the row on
what finally stands on Discord while pinning `created_at` to the original send time.

Nothing in the process reads the table back — the reply pipeline reads channel history from
Discord itself — so this is a durable archive for offline use, not a cache any runtime path
depends on. That is also why a failed write is absorbed into a `logfire.error` here instead of
degrading any feature.
"""

import re
from typing import Any, Final
import asyncio
import threading

import logfire
from nextcord import Message, DMChannel
from pydantic import Field, BaseModel, ConfigDict, computed_field
from sqlalchemy import Engine, text, event, create_engine
from nextcord.ext import commands

from discordbot.utils.sqlite_config import configure_sqlite_connection

CONTROL_CHARS_RE = re.compile(pattern=r"\x00")

# Single shared engine — putting create_engine() on a per-message
# cached_property leaked the connection pool, dialect cache and inspector
# cache for every Discord message.
_sql_engine: Engine = create_engine(url="sqlite:///data/database/messages.db")


@event.listens_for(_sql_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Applies the project's SQLite PRAGMAs to every connection this engine opens.

    WAL is what makes the archive usable while the bot runs: the default rollback journal
    serializes reads against writes, and with this DB already in the gigabyte range any
    concurrent reader (an analytics query, a manual `sqlite3`) would wedge the live logging
    path. The `StoredInteger` UDFs are skipped because no column here holds one.

    `@event.listens_for` binds to the engine that exists at import, so an engine a test swaps
    onto `_sql_engine` gets none of this and runs on SQLite's defaults.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
        _connection_record (Any): SQLAlchemy's pool record for it, unused.
    """
    configure_sqlite_connection(dbapi_connection=dbapi_connection, register_stored_integer=False)


_MESSAGES_TABLE_LOCK = threading.Lock()
_MESSAGES_TABLE_READY_FOR: Engine | None = None

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
    # leaving the legacy rows that carry no discord_message_id untouched.
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


def _write_row_sync(row: dict[str, str]) -> None:
    """Creates the schema if this engine has not been seen yet, then writes one row.

    Runs on a worker thread (`asyncio.to_thread` in `_save_messages`), so the readiness marker
    is read and written under `_MESSAGES_TABLE_LOCK`. The lock guards the marker only, not the
    DDL: two threads racing the first write both run it, which is harmless because every
    statement is `IF NOT EXISTS` and rides the same transaction as the insert. The marker holds
    the engine object rather than a flag, so a test that swaps `_sql_engine` onto a temp file
    gets the schema created there instead of inheriting readiness from the previous database.

    Args:
        row (dict[str, str]): One row keyed by the columns `_CREATE_MESSAGES_TABLE_SQL` declares.
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
    """One message's archive row: the columns derived from it, and the write that lands them.

    Carries no filtering of its own — `LogMessageCog` decides what belongs in the archive — so
    it is safe to construct anywhere a message is already known to be loggable.

    Attributes:
        message: The Discord message being logged.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    message: Message = Field(..., description="The Discord message being logged.")

    @staticmethod
    def sanitize_text(s: str | None) -> str:
        """Drops NUL bytes from one message field and maps a missing one to the empty string.

        Every column of the row binds as `str`, so a None has to be flattened here rather than
        at the call site. `CONTROL_CHARS_RE` matches NUL alone, despite the name.

        Args:
            s (str | None): The message field to clean.

        Returns:
            The cleaned text, empty when `s` was None.
        """
        if s is None:
            return ""
        return CONTROL_CHARS_RE.sub("", s)

    @computed_field
    @property
    def source_type(self) -> str:
        """The `source_type` column: which side of Discord the message came from.

        Returns:
            `"dm"` for a `DMChannel`, otherwise `"guild"` — a thread or a group DM included,
            since only the 1:1 DM case is told apart here.
        """
        if isinstance(self.message.channel, DMChannel):
            return "dm"
        return "guild"

    @computed_field
    @property
    def channel_name_or_author_name(self) -> str:
        """The `channel_name` column: a human-readable label for where the message was sent.

        Returns:
            `DM_<display name>_<user id>` for a direct message, otherwise
            `channel_<name>_<id>`, with the id standing in for the name on a channel type that
            carries none.
        """
        if isinstance(self.message.channel, DMChannel):
            author_name = self.message.author.display_name
            return f"DM_{author_name}_{self.message.author.id}"
        channel = self.message.channel
        channel_name = getattr(channel, "name", None) or channel.id
        return f"channel_{channel_name}_{channel.id}"

    @computed_field
    @property
    def channel_id_or_author_id(self) -> str:
        """The `channel_id` column, as text.

        Returns:
            The author's user id for a direct message, otherwise the channel id — so DM rows
            group by the person on the other side and guild rows by the channel.
        """
        if isinstance(self.message.channel, DMChannel):
            return f"{self.message.author.id}"
        return f"{self.message.channel.id}"

    async def _save_messages(self) -> None:
        """Builds this message's row and writes it from a worker thread.

        SQLite I/O is synchronous and a WAL commit fsyncs, so writing inline would block the
        whole event loop — Discord events, LLM streams and game settlement included — until the
        row landed. `asyncio.to_thread` moves it off; SQLite serializes the resulting threads
        itself through its file-level write lock and the connection's `busy_timeout`.

        Attachments and stickers are stored as `;`-joined CDN urls, which is why an edit that
        attaches files has to rewrite those columns and not only `content`.
        """
        attachment_paths = [attachment.url for attachment in self.message.attachments]
        sticker_paths = [sticker.url for sticker in self.message.stickers]
        row: dict[str, str] = {
            "discord_message_id": str(self.message.id),
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
        """Writes the row, absorbing any failure into one log line.

        Raises nothing on purpose: every caller schedules this as a detached task, so an
        escaping exception would reach nobody and surface only as "Task exception was never
        retrieved" once the task is collected.
        """
        try:
            await self._save_messages()
        except Exception as exc:
            # Stays broad: this runs as a detached create_task, so anything not caught
            # here surfaces only as "Task exception was never retrieved".
            logfire.error(
                "Failed to log message",
                discord_message_id=self.message.id,
                channel_id=self.channel_id_or_author_id,
                source_type=self.source_type,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )


class LogMessageCog(commands.Cog):
    """The archive's Discord surface: three listeners, no command, nothing sent back.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Keeps the bot handle the author filter reads its own user id off.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot

    def _should_log(self, message: Message) -> bool:
        """Whether this message belongs in the archive.

        Third-party bots (other Discord apps sharing the guild) are deliberately skipped so
        messages.db tracks only the conversation participants this bot actually engages with —
        its users and itself.

        Args:
            message (Message): The message to judge.

        Returns:
            True for a human author or this bot's own message, False for every other bot and
            for a bot-authored message arriving while `bot.user` is still unset.
        """
        if not message.author.bot:
            return True
        return bool(self.bot.user and message.author.id == self.bot.user.id)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Archives a message the first time it is seen.

        Args:
            message (Message): The message that was sent.
        """
        if not self._should_log(message=message):
            return
        asyncio.create_task(MessageLogger(message=message).log())  # noqa: RUF006

    @commands.Cog.listener()
    async def on_message_edit(self, _before: Message, after: Message) -> None:
        """Re-archives an edited message so a streaming reply converges on its final state.

        `on_message` sees the streaming text path in `cogs/gen_reply/streaming.py` at its
        initial `reply()`, when the message carries at most a reasoning-summary preview.
        Everything after that — the answer as it grows, the usage footer, an attached voice clip
        or image — arrives as a `reply.edit(...)` here, and the UPSERT on `discord_message_id`
        folds the lot into the one row.

        Args:
            _before (Message): The pre-edit snapshot, unused: the row is rebuilt from `after`.
            after (Message): The current message state.
        """
        if not self._should_log(message=after):
            return
        asyncio.create_task(MessageLogger(message=after).log())  # noqa: RUF006

    @commands.Cog.listener()
    async def on_command_completion(self, context: commands.Context[commands.Bot]) -> None:
        """Archives the message that invoked a completed prefix command.

        Dormant in this deployment: nothing registers a `commands.command`, and a slash command
        dispatches `on_application_command_completion` instead. It is also the one listener that
        does not consult `_should_log`, which costs nothing while `on_message` has already
        written the same message and the UPSERT collapses the repeat.

        Args:
            context (commands.Context[commands.Bot]): The context of the command.
        """
        asyncio.create_task(MessageLogger(message=context.message).log())  # noqa: RUF006


def setup(bot: commands.Bot) -> None:
    """Registers the cog, sync because nextcord fires an async `setup` without awaiting it.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(LogMessageCog(bot), override=True)
