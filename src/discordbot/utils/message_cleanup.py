"""Timed deletion of public Discord messages, with a record that outlives the process.

A public response with a TTL (a settled Blackjack table, a Dragon Gate round, an economy embed,
a stock or fishing panel) is removed from the channel once it stops being useful.
In-process that is either `schedule_public_message_delete`, a fire-and-forget task that sleeps
then deletes, or a view that owns its message and calls `delete_public_message` off its own idle
timeout. The second half exists because neither survives a restart, so every tracked message is
also written to the `pending_game_message` table in `games.db` and
`delete_tracked_public_messages` sweeps the leftovers at the next startup (the games cog owns
that one call, from its `on_ready`). A row is dropped by whichever cleanup path deletes the
message; one removed out of band (a moderator, its author, another bot) keeps its row until the
next sweep finds the message gone and clears it.

Bookkeeping here is best-effort by design: a DB failure is logged and swallowed, because the
worst it can cost is one stale message, while raising would break the command that produced it.
`list_pending_public_messages` is the one call logged as an error, since its empty-list degrade
disables the whole sweep for that process rather than losing a single row.

Lives in `utils/` because economy, games, stock and `owned_message_views` all schedule cleanup
and none of them may import a peer cog to reach it. It keeps its own engine on the shared
`games.db` (the games and fishing engines are separate ones, deliberately) and creates its table
on every access, since nothing bootstraps that schema. It deliberately covers neither
ephemeral responses (Discord expires those itself) nor anything about the message's content.
"""

from typing import Any, Final
import asyncio
from pathlib import Path

import logfire
from nextcord import Message, NotFound, Forbidden, HTTPException
from pydantic import Field, BaseModel
from sqlalchemy import Engine, text, event, create_engine
from nextcord.abc import Messageable
from nextcord.ext import commands
from sqlalchemy.engine import Connection

from discordbot.utils.sqlite_config import configure_sqlite_connection

PUBLIC_MESSAGE_TTL_SECONDS = 180
_PENDING_PUBLIC_MESSAGE_DB_PATH = Path("data/database/games.db")
_pending_engine: Engine | None = None
_pending_engine_path: Path | None = None
_CREATE_PENDING_PUBLIC_MESSAGES_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS pending_game_message (
    message_id INTEGER PRIMARY KEY,
    channel_id INTEGER NOT NULL,
    guild_name TEXT,
    channel_name TEXT,
    user_name TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
_UPSERT_PENDING_PUBLIC_MESSAGE_SQL: Final[str] = """
INSERT INTO pending_game_message (message_id, channel_id, guild_name, channel_name, user_name)
VALUES (:message_id, :channel_id, :guild_name, :channel_name, :user_name)
ON CONFLICT(message_id) DO UPDATE SET
    channel_id = excluded.channel_id,
    guild_name = excluded.guild_name,
    channel_name = excluded.channel_name,
    user_name = COALESCE(excluded.user_name, pending_game_message.user_name)
"""
_DELETE_PENDING_PUBLIC_MESSAGE_SQL: Final[str] = """
DELETE FROM pending_game_message WHERE message_id = :message_id
"""
_LIST_PENDING_PUBLIC_MESSAGES_SQL: Final[str] = """
SELECT channel_id, message_id, guild_name, channel_name, user_name
FROM pending_game_message
ORDER BY created_at ASC, message_id ASC
"""


class PendingPublicMessage(BaseModel):
    """A public response that still needs Discord-side cleanup."""

    channel_id: int = Field(..., description="Channel holding the tracked public message.")
    message_id: int = Field(..., description="Discord id of the tracked public message.")
    guild_name: str | None = Field(default=None, description="Guild name for cleanup logs.")
    channel_name: str | None = Field(default=None, description="Channel name for cleanup logs.")
    user_name: str | None = Field(default=None, description="Triggering user, for cleanup logs.")


def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Applies the project's shared SQLite PRAGMA setup to a new cleanup connection.

    Registered as SQLAlchemy's `connect` listener, so both parameters are positional and the
    second is never read. The `StoredInteger` UDFs are skipped: this table holds plain Discord
    ids, not the decimal text the economy tables store.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
        _connection_record (Any): SQLAlchemy's pool record, unused here.
    """
    configure_sqlite_connection(dbapi_connection=dbapi_connection, register_stored_integer=False)


def _pending_db_engine() -> Engine:
    """Returns the cleanup engine, rebuilding it when the configured DB path has changed.

    The path is re-read on every call and the previous engine disposed on a change, which is what
    lets a test point `_PENDING_PUBLIC_MESSAGE_DB_PATH` at a `tmp_path` without leaking the old
    pool. Creates the parent directory, so a fresh checkout needs no setup step.

    Returns:
        The process-wide engine for the current `_PENDING_PUBLIC_MESSAGE_DB_PATH`.
    """
    global _pending_engine, _pending_engine_path  # noqa: PLW0603 -- testable singleton by DB path

    db_path = Path(_PENDING_PUBLIC_MESSAGE_DB_PATH)
    if _pending_engine is not None and _pending_engine_path == db_path:
        return _pending_engine

    if _pending_engine is not None:
        _pending_engine.dispose()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _pending_engine = create_engine(url=f"sqlite:///{db_path}")
    event.listen(_pending_engine, "connect", _configure_sqlite)
    _pending_engine_path = db_path
    return _pending_engine


def _ensure_pending_table(conn: Connection) -> None:
    """Creates the pending-cleanup table when it is not there yet.

    Run before every read and write rather than once at import: nothing bootstraps this schema at
    import or startup, and `_pending_db_engine` can be rebuilt onto a different path mid-process
    (a test pointing the DB at a `tmp_path`), so the table has to be created against whichever
    file the engine of the moment opened.

    Args:
        conn (Connection): Open connection inside the caller's transaction.
    """
    conn.execute(statement=text(text=_CREATE_PENDING_PUBLIC_MESSAGES_SQL))


def _message_record(message: Message, user_name: str | None = None) -> PendingPublicMessage | None:
    """Extracts the persistent cleanup identity from a Discord message.

    Reads through `getattr` because callers hand over whatever their send returned, and returns
    None on a missing channel/message id pair instead of raising: those two are the only fields
    the sweep can act on, and failing to track must never break the command that just posted.
    The guild and channel names are decoration for the cleanup logs, taken where available.

    Args:
        message (Message): The public message just posted.
        user_name (str | None): Discord account name of the triggering user, for cleanup logs.

    Returns:
        The record to persist, or None when the message carries no usable ids.
    """
    channel = getattr(message, "channel", None)
    channel_id = getattr(channel, "id", None)
    message_id = getattr(message, "id", None)
    if not isinstance(channel_id, int) or not isinstance(message_id, int):
        return None
    guild = getattr(message, "guild", None) or getattr(channel, "guild", None)
    guild_name = getattr(guild, "name", None)
    channel_name = getattr(channel, "name", None)
    return PendingPublicMessage(
        channel_id=channel_id,
        message_id=message_id,
        guild_name=guild_name if isinstance(guild_name, str) else None,
        channel_name=channel_name if isinstance(channel_name, str) else None,
        user_name=user_name,
    )


def _track_public_message_sync(record: PendingPublicMessage) -> None:
    """Upserts one pending-cleanup row.

    Blocking; `track_public_message` runs it off the loop. Re-tracking the same message
    `COALESCE`s `user_name`, so a later call without a name keeps the one already stored.

    Args:
        record (PendingPublicMessage): Identity of the message awaiting deletion.
    """
    with _pending_db_engine().begin() as conn:
        _ensure_pending_table(conn=conn)
        conn.execute(
            statement=text(text=_UPSERT_PENDING_PUBLIC_MESSAGE_SQL),
            parameters={
                "message_id": record.message_id,
                "channel_id": record.channel_id,
                "guild_name": record.guild_name,
                "channel_name": record.channel_name,
                "user_name": record.user_name,
            },
        )


def _forget_public_message_sync(message_id: int) -> None:
    """Deletes one pending-cleanup row.

    Blocking; `forget_public_message` runs it off the loop. An id with no row is a no-op, which
    is what lets the TTL path and the restart sweep both forget the same message.

    Args:
        message_id (int): Discord id of the message no longer needing cleanup.
    """
    with _pending_db_engine().begin() as conn:
        _ensure_pending_table(conn=conn)
        conn.execute(
            statement=text(text=_DELETE_PENDING_PUBLIC_MESSAGE_SQL),
            parameters={"message_id": message_id},
        )


def _list_pending_public_messages_sync() -> list[PendingPublicMessage]:
    """Reads every row still waiting for cleanup, oldest first.

    Blocking; `list_pending_public_messages` runs it off the loop. Ordered by `created_at` with
    the message id as tiebreak, so the restart sweep deletes in the order things were posted.

    Returns:
        Every pending record, oldest first.
    """
    with _pending_db_engine().begin() as conn:
        _ensure_pending_table(conn=conn)
        rows = conn.execute(statement=text(text=_LIST_PENDING_PUBLIC_MESSAGES_SQL)).fetchall()
        return [
            PendingPublicMessage(
                channel_id=int(row[0]),
                message_id=int(row[1]),
                guild_name=str(row[2]) if row[2] is not None else None,
                channel_name=str(row[3]) if row[3] is not None else None,
                user_name=str(row[4]) if row[4] is not None else None,
            )
            for row in rows
        ]


async def track_public_message(
    message: Message, user_name: str | None = None
) -> PendingPublicMessage | None:
    """Records a public response so a restart can delete it later.

    The write runs in a worker thread and is best-effort: on failure the record still comes back,
    so the caller's in-process deletion is unaffected and only restart survival is lost.

    Args:
        message (Message): Discord message created for an expiring public response.
        user_name (str | None): Optional Discord account name of the user who triggered the
            response.

    Returns:
        The record, or `None` when the message object has no usable `channel.id` / `id` pair.
    """
    record = _message_record(message=message, user_name=user_name)
    if record is None:
        return None
    try:
        await asyncio.to_thread(_track_public_message_sync, record=record)
    # Stays broad: a narrowed handler would let a RuntimeError from a closing executor escape
    # into a fire-and-forget task and skip the in-process deletion entirely.
    except Exception as exc:
        logfire.warn(
            "Failed to track pending public response",
            message_id=record.message_id,
            channel_id=record.channel_id,
            error_type=type(exc).__name__,
            _exc_info=True,
        )
    return record


async def forget_public_message(message_id: int) -> None:
    """Stops tracking a message that no longer needs cleaning up.

    Best-effort, in a worker thread; the caller is always a path that has just deleted the
    message or found it already gone, so a failure costs only a stale row.

    Args:
        message_id (int): Discord id of the message to stop tracking.
    """
    try:
        await asyncio.to_thread(_forget_public_message_sync, message_id=message_id)
    # Stays broad for the same reason as tracking; the stale row self-heals on the next
    # delete_tracked_public_messages sweep via its NotFound branch.
    except Exception as exc:
        logfire.warn(
            "Failed to forget pending public response",
            message_id=message_id,
            error_type=type(exc).__name__,
            _exc_info=True,
        )


async def list_pending_public_messages() -> list[PendingPublicMessage]:
    """Returns public messages left over from a previous process.

    Reads in a worker thread and degrades to an empty list rather than raising into `on_ready`.

    Returns:
        Every message still tracked, oldest first, or an empty list when the read failed.
    """
    try:
        return await asyncio.to_thread(_list_pending_public_messages_sync)
    # The one error of the trio: unlike a single lost bookkeeping row, an empty list disables
    # the whole restart sweep for this process, so every stale message stays on screen.
    except Exception as exc:
        logfire.error(
            "Failed to list pending public responses",
            error_type=type(exc).__name__,
            _exc_info=True,
        )
        return []


async def _fetch_tracked_message(bot: commands.Bot, record: PendingPublicMessage) -> Message:
    """Resolves a tracked message, via the channel cache when it can serve one.

    A cached channel that is not `Messageable` is re-fetched rather than trusted, since only a
    concrete channel can fetch a message by id. Discord errors from the two fetches propagate;
    `delete_tracked_public_messages` decides per kind whether the record survives.

    Args:
        bot (commands.Bot): Bot whose channel cache and REST client resolve the channel.
        record (PendingPublicMessage): The tracked message to resolve.

    Returns:
        The live message, ready to delete.

    Raises:
        TypeError: The channel resolved but still cannot fetch messages.
    """
    channel = bot.get_channel(record.channel_id)
    if channel is None or not isinstance(channel, Messageable):
        channel = await bot.fetch_channel(record.channel_id)
    if not isinstance(channel, Messageable):
        msg = f"Channel {record.channel_id} cannot fetch messages"
        raise TypeError(msg)
    return await channel.fetch_message(record.message_id)


async def delete_public_message(message: Message, message_id: int | None = None) -> bool:
    """Deletes a public message and drops its persisted cleanup record.

    A message Discord no longer has counts as success: someone removed it first, and the record
    goes with it. A permission or transport failure keeps the record instead, so the next restart
    sweep gets another try. `message_id` is for callers that already know it (the sweep) or hold
    a message object that may not expose one; otherwise it comes off the message.

    Args:
        message (Message): The message to delete.
        message_id (int | None): Id to forget, when the caller already knows it.

    Returns:
        True when the message is gone from the channel, False when the delete failed.
    """
    resolved_message_id = message_id if message_id is not None else getattr(message, "id", None)
    try:
        await message.delete()
    except NotFound:
        pass
    except (Forbidden, HTTPException):
        logfire.warn(
            "Failed to delete public response",
            message_id=resolved_message_id,
            channel_id=getattr(getattr(message, "channel", None), "id", None),
            _exc_info=True,
        )
        return False
    if isinstance(resolved_message_id, int):
        await forget_public_message(message_id=resolved_message_id)
    return True


async def delete_tracked_public_messages(bot: commands.Bot) -> None:
    """Deletes persisted public responses left by an earlier bot process.

    The restart half of the TTL, called once per process from the games cog's `on_ready`. A
    `NotFound` is forgotten and counted as cleaned up, which covers both a message Discord no
    longer has and a recorded channel id that `fetch_channel` itself 404s on. The record is KEPT
    on a `TypeError` (the channel resolved but cannot fetch messages), on a `Forbidden`, and on
    every other `HTTPException`, so a transient outage or a temporarily missing permission does
    not throw away the only trace of a message still sitting in the channel.

    Args:
        bot (commands.Bot): Bot used to resolve the recorded channels.
    """
    records = await list_pending_public_messages()
    deleted_count = 0
    for record in records:
        try:
            message = await _fetch_tracked_message(bot=bot, record=record)
        except NotFound:
            await forget_public_message(message_id=record.message_id)
            deleted_count += 1
            continue
        except TypeError:
            logfire.warn(
                "Failed to resolve stale public response channel",
                channel_id=record.channel_id,
                message_id=record.message_id,
                _exc_info=True,
            )
            continue
        except (Forbidden, HTTPException):
            logfire.warn(
                "Failed to fetch stale public response",
                channel_id=record.channel_id,
                message_id=record.message_id,
                _exc_info=True,
            )
            continue
        if await delete_public_message(message=message, message_id=record.message_id):
            deleted_count += 1
    if records:
        logfire.info(
            "Deleted stale public responses",
            deleted_count=deleted_count,
            pending_count=len(records),
        )


async def delete_public_message_after(
    message: Message, delay: float = PUBLIC_MESSAGE_TTL_SECONDS, user_name: str | None = None
) -> None:
    """Deletes a public response after a delay.

    Tracking happens before the sleep, not after it, so a process that dies mid-TTL still leaves
    a record for the next startup sweep.

    Args:
        message (Message): Discord message to delete.
        delay (float): Seconds to wait before deletion.
        user_name (str | None): Optional Discord account name of the user who triggered the
            response.
    """
    await track_public_message(message=message, user_name=user_name)
    await asyncio.sleep(delay=delay)
    await delete_public_message(message=message)


def schedule_public_message_delete(
    message: Message, delay: float = PUBLIC_MESSAGE_TTL_SECONDS, user_name: str | None = None
) -> None:
    """Schedules delayed deletion for a public response without awaiting it.

    Creates the task without awaiting it, so the command that sent the message returns
    immediately. Nothing keeps a reference to it either, which is the risk the RUF006 noqa
    suppresses: a task garbage-collected before it fires costs the same as one the process does
    not live to finish, and both are covered by the record `delete_public_message_after` writes
    before it sleeps.

    Args:
        message (Message): Discord message to delete once the delay elapses.
        delay (float): Seconds to wait before deletion.
        user_name (str | None): Optional Discord account name of the user who triggered the
            response.
    """
    asyncio.create_task(  # noqa: RUF006 -- fire-and-forget cleanup cannot block commands.
        coro=delete_public_message_after(message=message, delay=delay, user_name=user_name),
        name="delete-public-response",
    )
