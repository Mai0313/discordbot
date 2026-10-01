"""Timed cleanup helpers for public Discord messages."""

from typing import Final
import asyncio

import logfire
from nextcord import Message, NotFound, Forbidden, Interaction, HTTPException
from pydantic import Field, BaseModel
from sqlalchemy import MetaData, text
from nextcord.abc import Messageable
from nextcord.ext import commands
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncConnection, create_async_engine

from discordbot.utils.asyncio_locks import spawn_tracked
from discordbot.utils.sqlite_config import SqliteBootstrap

PUBLIC_MESSAGE_TTL_SECONDS = 180
# The scheduled deletions still waiting out their TTL.
_delete_tasks: set[asyncio.Task[None]] = set()
_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/games.db")
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


async def _create_pending_table(*, conn: AsyncConnection) -> None:
    """Creates the cleanup table where it does not exist yet.

    A fresh file gets the columns the deployed `games.db` table has; an existing table is never
    altered. Raw DDL rather than a declared model: `create_all` never alters one either, and the
    DDL a model compiles to differs from this text, so a model would be a second definition of
    the table that nothing checks against the first.
    """
    await conn.execute(statement=text(text=_CREATE_PENDING_PUBLIC_MESSAGES_SQL))


# No declared tables: the one table comes from `_create_pending_table`.
_database = SqliteBootstrap(metadata=MetaData(), after_create=_create_pending_table)


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


async def _track_public_message(record: PendingPublicMessage) -> None:
    """Persists a pending cleanup record."""
    async with _database.open_session(engine=_engine) as session, session.begin():
        await session.execute(
            statement=text(text=_UPSERT_PENDING_PUBLIC_MESSAGE_SQL),
            params={
                "message_id": record.message_id,
                "channel_id": record.channel_id,
                "guild_name": record.guild_name,
                "channel_name": record.channel_name,
                "user_name": record.user_name,
            },
        )


async def _forget_public_message(message_id: int) -> None:
    """Removes a pending cleanup record."""
    async with _database.open_session(engine=_engine) as session, session.begin():
        await session.execute(
            statement=text(text=_DELETE_PENDING_PUBLIC_MESSAGE_SQL),
            params={"message_id": message_id},
        )


async def _list_pending_public_messages() -> list[PendingPublicMessage]:
    """Lists all messages still waiting for cleanup."""
    async with _database.open_session(engine=_engine) as session:
        result = await session.execute(statement=text(text=_LIST_PENDING_PUBLIC_MESSAGES_SQL))
        return [PendingPublicMessage.model_validate(obj=dict(row)) for row in result.mappings()]


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
        await _track_public_message(record=record)
    # Stays broad: anything escaping here would end the fire-and-forget task before the
    # in-process deletion ran.
    except Exception as exc:
        logfire.warn(
            "Failed to track pending public response",
            message_id=record.message_id,
            channel_id=record.channel_id,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
    return record


async def forget_public_message(message_id: int) -> None:
    """Deletes a public message cleanup record."""
    try:
        await _forget_public_message(message_id=message_id)
    # Stays broad for the same reason as tracking; the stale row self-heals on the next
    # delete_tracked_public_messages sweep via its NotFound branch.
    except Exception as exc:
        logfire.warn(
            "Failed to forget pending public response",
            message_id=message_id,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )


async def list_pending_public_messages() -> list[PendingPublicMessage]:
    """Returns public messages left over from a previous process."""
    try:
        return await _list_pending_public_messages()
    # Unlike a single lost bookkeeping row, an empty list disables the whole restart sweep for
    # this process, so every stale message stays on screen.
    except Exception as exc:
        logfire.error(
            "Failed to list pending public responses", error_type=type(exc).__name__, _exc_info=exc
        )
        return []


async def _fetch_tracked_message(bot: commands.Bot, record: PendingPublicMessage) -> Message:
    """Fetches a tracked message from a concrete Discord channel."""
    channel = bot.get_channel(record.channel_id)
    if channel is None or not isinstance(channel, Messageable):
        channel = await bot.fetch_channel(record.channel_id)
    if not isinstance(channel, Messageable):
        msg = f"Channel {record.channel_id} cannot fetch messages"
        raise TypeError(msg)
    return await channel.fetch_message(record.message_id)


def report_press_failure(error: HTTPException, message: Message, action: str) -> None:
    """Logs a press's token failing to `action` a public message, before the channel is tried.

    A 404 is a message already gone or one the token cannot reach, which the channel attempt
    tells apart and reports; a 403 is a refusal whose code is the whole finding. Anything else
    keeps its traceback.
    """
    text = f"A press could not {action} a public message; trying the channel"
    message_id = getattr(message, "id", None)
    channel_id = getattr(getattr(message, "channel", None), "id", None)
    if isinstance(error, NotFound):
        logfire.info(text, message_id=message_id, channel_id=channel_id, code=error.code)
    elif isinstance(error, Forbidden):
        logfire.warn(text, message_id=message_id, channel_id=channel_id, code=error.code)
    else:
        logfire.warn(
            text, message_id=message_id, channel_id=channel_id, code=error.code, _exc_info=error
        )


async def _deleted_through_press(
    interaction: Interaction[commands.Bot] | None, message: Message
) -> bool:
    """Deletes through a live press on the message; False leaves the delete to the channel.

    A press's token reaches the message its control sits on whatever the channel allows, where
    the channel endpoint answers 403 once the server shuts the bot out. That the token can
    delete that message, not only edit it, is read off Discord's docs rather than measured, so
    any failure hands the delete to the channel.
    """
    if interaction is None or interaction.is_expired():
        return False
    try:
        await interaction.delete_original_message()
    except HTTPException as error:
        report_press_failure(error=error, message=message, action="delete")
        return False
    return True


async def delete_public_message(
    message: Message,
    message_id: int | None = None,
    interaction: Interaction[commands.Bot] | None = None,
) -> bool:
    """Deletes a public message and removes its persisted cleanup record.

    Args:
        message: Discord message to delete.
        message_id: Record key when the message object carries none.
        interaction: A press on the message whose token, while it lives, deletes it in place of
            the channel. Never persisted, so a restart's sweep has only the channel.

    Returns:
        True when the message is gone, an already-deleted one included; False when Discord
        refused the delete, which keeps the record for the next process's startup sweep.
    """
    resolved_message_id = message_id if message_id is not None else getattr(message, "id", None)
    try:
        if not await _deleted_through_press(interaction=interaction, message=message):
            await message.delete()
    except NotFound:
        pass
    except Forbidden:
        # Expected rather than diagnosable, like the sweep's fetch below: the server's overwrites
        # shut the bot out of the channel, so the ids are the whole finding.
        logfire.warn(
            "Discord refused to delete a public response",
            message_id=resolved_message_id,
            channel_id=getattr(getattr(message, "channel", None), "id", None),
        )
        return False
    except HTTPException:
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
    """Deletes persisted public responses left by an earlier bot process."""
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
        except Forbidden:
            # Expected rather than diagnosable, and the one `warn` here that attaches nothing:
            # the channel's overwrites are the server's to set, so the bot's own identity can
            # be shut out of a channel an interaction still reaches. The type and the two ids
            # are the whole story, while the traceback is sixteen identical lines per record
            # per boot. Split from the branch below so a 5xx or a rate limit — which IS worth
            # a traceback — does not lose one by sharing this handler.
            logfire.warn(
                "Stale public response sits in a channel the bot cannot read",
                channel_id=record.channel_id,
                message_id=record.message_id,
            )
            continue
        except HTTPException:
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
    message: Message,
    delay: float = PUBLIC_MESSAGE_TTL_SECONDS,
    user_name: str | None = None,
    interaction: Interaction[commands.Bot] | None = None,
) -> None:
    """Deletes a public response after a delay.

    Args:
        message: Discord message to delete.
        delay: Seconds to wait before deletion.
        user_name: Optional Discord account name of the user who triggered the response.
        interaction: Optional press on the message to delete through (`delete_public_message`).
    """
    await track_public_message(message=message, user_name=user_name)
    await asyncio.sleep(delay=delay)
    await delete_public_message(message=message, interaction=interaction)


def schedule_public_message_delete(
    message: Message,
    delay: float = PUBLIC_MESSAGE_TTL_SECONDS,
    user_name: str | None = None,
    interaction: Interaction[commands.Bot] | None = None,
) -> None:
    """Schedules delayed deletion for a public response, never blocking the command."""
    spawn_tracked(
        coro=delete_public_message_after(
            message=message, delay=delay, user_name=user_name, interaction=interaction
        ),
        tasks=_delete_tasks,
        name="delete-public-response",
    )
