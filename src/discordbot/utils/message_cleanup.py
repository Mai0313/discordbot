"""Timed cleanup helpers for public Discord messages."""

from typing import Any, Final
import asyncio
from pathlib import Path

import logfire
import nextcord
from nextcord import Message
from pydantic import Field, BaseModel
from sqlalchemy import Engine, text, event, create_engine
from nextcord.ext import commands
from sqlalchemy.engine import Connection

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

    channel_id: int = Field(description="Channel holding the tracked public message.")
    message_id: int = Field(description="Discord id of the tracked public message.")
    guild_name: str | None = Field(default=None, description="Guild name for cleanup logs.")
    channel_name: str | None = Field(default=None, description="Channel name for cleanup logs.")
    user_name: str | None = Field(default=None, description="Triggering user, for cleanup logs.")


def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Sets WAL mode and a tolerant busy timeout for cleanup persistence."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def _pending_db_engine() -> Engine:
    """Returns the cleanup DB engine for the current DB path."""
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
    """Ensures the cleanup table exists before a read or write."""
    conn.execute(statement=text(text=_CREATE_PENDING_PUBLIC_MESSAGES_SQL))


def _message_record(message: Message, user_name: str | None = None) -> PendingPublicMessage | None:
    """Extracts the persistent cleanup identity from a Discord message."""
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
    """Persists a pending cleanup record."""
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
    """Removes a pending cleanup record."""
    with _pending_db_engine().begin() as conn:
        _ensure_pending_table(conn=conn)
        conn.execute(
            statement=text(text=_DELETE_PENDING_PUBLIC_MESSAGE_SQL),
            parameters={"message_id": message_id},
        )


def _list_pending_public_messages_sync() -> list[PendingPublicMessage]:
    """Lists all messages still waiting for cleanup."""
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

    Args:
        message: Discord message created for an expiring public response.
        user_name: Optional Discord account name of the user who triggered the response.

    Returns:
        The persisted record, or `None` when the message object has no usable
        `channel.id` / `id` pair.
    """
    record = _message_record(message=message, user_name=user_name)
    if record is None:
        return None
    try:
        await asyncio.to_thread(_track_public_message_sync, record=record)
    except Exception:
        logfire.error("Failed to track pending public response", _exc_info=True)
    return record


async def forget_public_message(message_id: int) -> None:
    """Deletes a public message cleanup record."""
    try:
        await asyncio.to_thread(_forget_public_message_sync, message_id=message_id)
    except Exception:
        logfire.error("Failed to forget pending public response", _exc_info=True)


async def list_pending_public_messages() -> list[PendingPublicMessage]:
    """Returns public messages left over from a previous process."""
    try:
        return await asyncio.to_thread(_list_pending_public_messages_sync)
    except Exception:
        logfire.error("Failed to list pending public responses", _exc_info=True)
        return []


async def _fetch_tracked_message(bot: commands.Bot, record: PendingPublicMessage) -> Message:
    """Fetches a tracked message from a concrete Discord channel."""
    channel = bot.get_channel(record.channel_id)
    if channel is None or not hasattr(channel, "fetch_message"):
        channel = await bot.fetch_channel(record.channel_id)
    if not hasattr(channel, "fetch_message"):
        msg = f"Channel {record.channel_id} cannot fetch messages"
        raise TypeError(msg)
    return await channel.fetch_message(record.message_id)


async def delete_public_message(message: Message, message_id: int | None = None) -> bool:
    """Deletes a public message and removes its persisted cleanup record."""
    resolved_message_id = message_id if message_id is not None else getattr(message, "id", None)
    try:
        await message.delete()
    except nextcord.NotFound:
        pass
    except (nextcord.Forbidden, nextcord.HTTPException):
        logfire.warn("Failed to delete public response", _exc_info=True)
        return False
    if isinstance(resolved_message_id, int):
        await forget_public_message(message_id=resolved_message_id)
    return True


async def delete_tracked_public_messages(bot: commands.Bot) -> None:
    """Deletes persisted public responses left by an earlier bot process."""
    records = await list_pending_public_messages()
    deleted_count = 0
    for record in records:
        try:
            message = await _fetch_tracked_message(bot=bot, record=record)
        except nextcord.NotFound:
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
        except (nextcord.Forbidden, nextcord.HTTPException):
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

    Args:
        message: Discord message to delete.
        delay: Seconds to wait before deletion.
        user_name: Optional Discord account name of the user who triggered the response.
    """
    await track_public_message(message=message, user_name=user_name)
    await asyncio.sleep(delay=delay)
    await delete_public_message(message=message)


def schedule_public_message_delete(
    message: Message, delay: float = PUBLIC_MESSAGE_TTL_SECONDS, user_name: str | None = None
) -> None:
    """Schedules delayed deletion for a public response."""
    asyncio.create_task(  # noqa: RUF006 -- fire-and-forget cleanup cannot block commands.
        coro=delete_public_message_after(message=message, delay=delay, user_name=user_name),
        name="delete-public-response",
    )
