"""Timed cleanup helpers for casino game messages."""

from typing import Any, Final
import asyncio
from pathlib import Path

import logfire
import nextcord
from nextcord import Message
from pydantic import BaseModel
from sqlalchemy import Engine, text, event, create_engine
from nextcord.ext import commands
from sqlalchemy.engine import Connection

GAME_RESPONSE_TTL_SECONDS = 180
_PENDING_GAME_MESSAGE_DB_PATH = Path("data/game_cleanup.db")
_pending_engine: Engine | None = None
_pending_engine_path: Path | None = None
_CREATE_PENDING_GAME_MESSAGES_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS pending_game_message (
    message_id INTEGER PRIMARY KEY,
    channel_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
_UPSERT_PENDING_GAME_MESSAGE_SQL: Final[str] = """
INSERT INTO pending_game_message (message_id, channel_id)
VALUES (:message_id, :channel_id)
ON CONFLICT(message_id) DO UPDATE SET
    channel_id = excluded.channel_id
"""
_DELETE_PENDING_GAME_MESSAGE_SQL: Final[str] = """
DELETE FROM pending_game_message WHERE message_id = :message_id
"""
_LIST_PENDING_GAME_MESSAGES_SQL: Final[str] = """
SELECT channel_id, message_id
FROM pending_game_message
ORDER BY created_at ASC, message_id ASC
"""


class PendingGameMessage(BaseModel):
    """A game response that still needs Discord-side cleanup."""

    channel_id: int
    message_id: int


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

    db_path = Path(_PENDING_GAME_MESSAGE_DB_PATH)
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
    conn.execute(statement=text(text=_CREATE_PENDING_GAME_MESSAGES_SQL))


def _message_record(message: Message) -> PendingGameMessage | None:
    """Extracts the persistent cleanup identity from a Discord message."""
    channel = getattr(message, "channel", None)
    channel_id = getattr(channel, "id", None)
    message_id = getattr(message, "id", None)
    if not isinstance(channel_id, int) or not isinstance(message_id, int):
        return None
    return PendingGameMessage(channel_id=channel_id, message_id=message_id)


def _track_game_message_sync(record: PendingGameMessage) -> None:
    """Persists a pending cleanup record."""
    with _pending_db_engine().begin() as conn:
        _ensure_pending_table(conn=conn)
        conn.execute(
            statement=text(text=_UPSERT_PENDING_GAME_MESSAGE_SQL),
            parameters={"message_id": record.message_id, "channel_id": record.channel_id},
        )


def _forget_game_message_sync(message_id: int) -> None:
    """Removes a pending cleanup record."""
    with _pending_db_engine().begin() as conn:
        _ensure_pending_table(conn=conn)
        conn.execute(
            statement=text(text=_DELETE_PENDING_GAME_MESSAGE_SQL),
            parameters={"message_id": message_id},
        )


def _list_pending_game_messages_sync() -> list[PendingGameMessage]:
    """Lists all messages still waiting for cleanup."""
    with _pending_db_engine().begin() as conn:
        _ensure_pending_table(conn=conn)
        rows = conn.execute(statement=text(text=_LIST_PENDING_GAME_MESSAGES_SQL)).fetchall()
        return [PendingGameMessage(channel_id=int(row[0]), message_id=int(row[1])) for row in rows]


async def track_game_message(message: Message) -> PendingGameMessage | None:
    """Records a game message so a restart can delete it later.

    Args:
        message: Discord message created for a game round or related expiring response.

    Returns:
        The persisted record, or `None` when the message object has no usable
        ``channel.id`` / ``id`` pair.
    """
    record = _message_record(message=message)
    if record is None:
        return None
    try:
        await asyncio.to_thread(_track_game_message_sync, record=record)
    except Exception:
        logfire.error("Failed to track pending game response", _exc_info=True)
    return record


async def forget_game_message(message_id: int) -> None:
    """Deletes a game message cleanup record."""
    try:
        await asyncio.to_thread(_forget_game_message_sync, message_id=message_id)
    except Exception:
        logfire.error("Failed to forget pending game response", _exc_info=True)


async def list_pending_game_messages() -> list[PendingGameMessage]:
    """Returns game messages left over from a previous process."""
    try:
        return await asyncio.to_thread(_list_pending_game_messages_sync)
    except Exception:
        logfire.error("Failed to list pending game responses", _exc_info=True)
        return []


async def _fetch_tracked_message(bot: commands.Bot, record: PendingGameMessage) -> Message:
    """Fetches a tracked message from a concrete Discord channel."""
    channel = bot.get_channel(record.channel_id)
    if channel is None or not hasattr(channel, "fetch_message"):
        channel = await bot.fetch_channel(record.channel_id)
    if not hasattr(channel, "fetch_message"):
        msg = f"Channel {record.channel_id} cannot fetch messages"
        raise TypeError(msg)
    return await channel.fetch_message(record.message_id)


async def delete_tracked_game_messages(bot: commands.Bot) -> None:
    """Deletes persisted game responses left by an earlier bot process."""
    records = await list_pending_game_messages()
    deleted_count = 0
    for record in records:
        deleted = False
        try:
            message = await _fetch_tracked_message(bot=bot, record=record)
            await message.delete()
            deleted = True
        except nextcord.NotFound:
            deleted = True
        except TypeError:
            logfire.warn(
                "Failed to resolve stale game response channel",
                channel_id=record.channel_id,
                message_id=record.message_id,
                _exc_info=True,
            )
            continue
        except (nextcord.Forbidden, nextcord.HTTPException):
            logfire.warn(
                "Failed to delete stale game response",
                channel_id=record.channel_id,
                message_id=record.message_id,
                _exc_info=True,
            )
            continue
        await forget_game_message(message_id=record.message_id)
        if deleted:
            deleted_count += 1
    if records:
        logfire.info(
            "Deleted stale game responses", deleted_count=deleted_count, pending_count=len(records)
        )


async def delete_game_message_after(
    message: Message, delay: float = GAME_RESPONSE_TTL_SECONDS
) -> None:
    """Deletes a game response after a delay.

    Args:
        message: Discord message to delete.
        delay: Seconds to wait before deletion.
    """
    record = await track_game_message(message=message)
    await asyncio.sleep(delay=delay)
    try:
        await message.delete()
    except nextcord.NotFound:
        pass
    except (nextcord.Forbidden, nextcord.HTTPException):
        logfire.warn("Failed to delete expired game response", _exc_info=True)
        return
    if record is not None:
        await forget_game_message(message_id=record.message_id)


def schedule_game_message_delete(
    message: Message, delay: float = GAME_RESPONSE_TTL_SECONDS
) -> None:
    """Schedules delayed deletion for a casino game response."""
    asyncio.create_task(  # noqa: RUF006 -- fire-and-forget cleanup cannot block commands.
        coro=delete_game_message_after(message=message, delay=delay), name="delete-game-response"
    )
