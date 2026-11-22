"""Timed cleanup helpers for public Discord messages."""

from typing import Any, Final, Protocol, runtime_checkable
import asyncio
from datetime import UTC, datetime

import logfire
from nextcord import Message, NotFound, Forbidden, Interaction, HTTPException
from pydantic import Field, BaseModel
from sqlalchemy import Text, Table, Column, Integer, MetaData, text
from nextcord.ext import commands
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from discordbot.utils.logged_ui import LoggedView
from discordbot.typings.timeouts import GAME_FINAL_EDIT_TIMEOUT_SECONDS
from discordbot.utils.asyncio_locks import LoopLocalRegistry, spawn_tracked
from discordbot.utils.sqlite_config import SqliteBootstrap, sqlite_url
from discordbot.utils.discord_errors import report_discord_failure

PUBLIC_MESSAGE_TTL_SECONDS = 180
_delete_tasks: LoopLocalRegistry[str, set[asyncio.Task[None]]] = LoopLocalRegistry()
_engine: AsyncEngine = create_async_engine(url=sqlite_url(filename="cleanup.db"))
metadata = MetaData()
Table(
    "pending_public_message",
    metadata,
    Column("message_id", Integer, primary_key=True),
    Column("channel_id", Integer, nullable=False),
    Column("guild_name", Text),
    Column("channel_name", Text),
    Column("user_name", Text),
    Column("created_at", Text, nullable=False, server_default=text(text="CURRENT_TIMESTAMP")),
)
_UPSERT_PENDING_PUBLIC_MESSAGE_SQL: Final[str] = """
INSERT INTO pending_public_message (message_id, channel_id, guild_name, channel_name, user_name)
VALUES (:message_id, :channel_id, :guild_name, :channel_name, :user_name)
ON CONFLICT(message_id) DO UPDATE SET
    channel_id = excluded.channel_id,
    guild_name = excluded.guild_name,
    channel_name = excluded.channel_name,
    user_name = COALESCE(excluded.user_name, pending_public_message.user_name)
"""
_DELETE_PENDING_PUBLIC_MESSAGE_SQL: Final[str] = """
DELETE FROM pending_public_message WHERE message_id = :message_id
"""
_LIST_PENDING_PUBLIC_MESSAGES_SQL: Final[str] = """
SELECT channel_id, message_id, guild_name, channel_name, user_name
FROM pending_public_message
WHERE :tracked_before IS NULL OR created_at < :tracked_before
ORDER BY created_at ASC, message_id ASC
"""


class PendingPublicMessage(BaseModel):
    """A public response that still needs Discord-side cleanup."""

    channel_id: int = Field(..., description="Channel holding the tracked public message.")
    message_id: int = Field(..., description="Discord id of the tracked public message.")
    guild_name: str | None = Field(default=None, description="Guild name for cleanup logs.")
    channel_name: str | None = Field(default=None, description="Channel name for cleanup logs.")
    user_name: str | None = Field(default=None, description="Triggering user, for cleanup logs.")


_database = SqliteBootstrap()


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


async def _list_pending_public_messages(
    tracked_before: datetime | None,
) -> list[PendingPublicMessage]:
    """Lists the messages still waiting for cleanup, recorded before `tracked_before` if given."""
    # `created_at` is SQLite's own CURRENT_TIMESTAMP: UTC, to the whole second.
    cutoff = (
        tracked_before.astimezone(tz=UTC).strftime(format="%Y-%m-%d %H:%M:%S")
        if tracked_before is not None
        else None
    )
    async with _database.open_session(engine=_engine) as session:
        result = await session.execute(
            statement=text(text=_LIST_PENDING_PUBLIC_MESSAGES_SQL),
            params={"tracked_before": cutoff},
        )
        return [PendingPublicMessage.model_validate(obj=dict(row)) for row in result.mappings()]


async def track_public_message(
    message: Message, user_name: str | None = None
) -> PendingPublicMessage | None:
    """Returns the persisted cleanup record for `message`, or None when it has no usable ids.

    Args:
        message (Message): Discord message created for an expiring public response.
        user_name (str | None): Optional Discord account name of the user who triggered the
            response.
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
    """Deletes a public message cleanup record.

    Args:
        message_id (int): Discord message id of the cleanup record to remove.
    """
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


async def list_pending_public_messages(
    tracked_before: datetime | None = None,
) -> list[PendingPublicMessage]:
    """Returns the public messages still waiting for cleanup.

    Args:
        tracked_before (datetime | None): Only rows recorded before this moment; every row when
            omitted.
    """
    try:
        return await _list_pending_public_messages(tracked_before=tracked_before)
    # Unlike a single lost bookkeeping row, an empty list disables the whole restart sweep for
    # this process, so every stale message stays on screen.
    except Exception as exc:
        logfire.error(
            "Failed to list pending public responses", error_type=type(exc).__name__, _exc_info=exc
        )
        return []


@runtime_checkable
class MessageFetcher(Protocol):
    """Channel protocol supporting `fetch_message`, excluding non-messageable channels."""

    async def fetch_message(self, id: int, /) -> Message:  # noqa: A002 -- nextcord's own name
        """Returns one message read back off the channel.

        Args:
            id (int): Discord message id to fetch.
        """
        ...


async def resolve_message_channel(bot: commands.Bot, channel_id: int) -> MessageFetcher:
    """Resolves `channel_id` (cache first, then REST) into a channel that can fetch messages.

    Raises `TypeError` when the resolved channel cannot fetch messages, and lets Discord
    `HTTPException` subclasses (`NotFound`, `Forbidden`, 5xx) propagate.

    Args:
        bot (commands.Bot): Bot used to look up or fetch the channel.
        channel_id (int): Stored Discord channel id.
    """
    channel = bot.get_channel(channel_id)
    if not isinstance(channel, MessageFetcher):
        channel = await bot.fetch_channel(channel_id)
    if not isinstance(channel, MessageFetcher):
        msg = f"Channel {channel_id} cannot fetch messages"
        raise TypeError(msg)
    return channel


async def fetch_stored_message(bot: commands.Bot, channel_id: int, message_id: int) -> Message:
    """Resolves `channel_id` (cache first, then REST) and fetches `message_id`.

    Raises `TypeError` when the resolved channel cannot fetch messages, and lets Discord
    `HTTPException` subclasses (`NotFound`, `Forbidden`, 5xx) propagate.

    Args:
        bot (commands.Bot): Bot used to look up or fetch the channel.
        channel_id (int): Stored Discord channel id.
        message_id (int): Stored Discord message id.
    """
    channel = await resolve_message_channel(bot=bot, channel_id=channel_id)
    return await channel.fetch_message(message_id)


def report_press_failure(error: HTTPException, message: Message, action: str) -> None:
    """Logs an interaction-token `action` failure before falling back to the channel endpoint.

    Logs 404 and 401 at `info` (code only), 403 at `warn` (code only), and other HTTP errors at
    `warn` with traceback.

    Args:
        error (HTTPException): Exception raised when acting through the press's token.
        message (Message): Target public message.
        action (str): Verb naming the attempted action (`"edit"` or `"delete"`).
    """
    text = f"A press could not {action} a public message; trying the channel"
    message_id = getattr(message, "id", None)
    channel_id = getattr(getattr(message, "channel", None), "id", None)
    if error.status == 401:
        logfire.info(
            text,
            message_id=message_id,
            channel_id=channel_id,
            code=error.code,
            error_type=type(error).__name__,
        )
        return
    report_discord_failure(
        error=error,
        not_found_message=text,
        forbidden_message=text,
        other_message=text,
        message_id=message_id,
        channel_id=channel_id,
    )


async def edit_public_message(
    message: Message, interaction: Interaction[commands.Bot] | None, payload: dict[str, Any]
) -> None:
    """Edits `message` via `interaction`'s token while unexpired, falling back to `message.edit`.

    `interaction` must have been acknowledged on `message` itself (not a separate ephemeral
    notice). Any `HTTPException` from the interaction token falls back to the channel endpoint.

    Args:
        message (Message): Public message to edit.
        interaction (Interaction[commands.Bot] | None): Press acknowledged on `message`, or None
            to edit directly via the channel.
        payload (dict[str, Any]): Keyword arguments forwarded to the edit call.
    """
    if interaction is not None and not interaction.is_expired():
        try:
            await interaction.edit_original_message(**payload)
        except HTTPException as error:
            report_press_failure(error=error, message=message, action="edit")
        else:
            return
    await message.edit(**payload)


async def edit_or_report(  # noqa: PLR0913 -- wraps edit_public_message with timeout and per-outcome failure logs
    message: Message,
    interaction: Interaction[commands.Bot] | None,
    payload: dict[str, Any],
    not_found_message: str,
    forbidden_message: str,
    other_message: str,
    timeout_seconds: float = GAME_FINAL_EDIT_TIMEOUT_SECONDS,
    **failure_fields: object,
) -> bool:
    """Edits `message` via `edit_public_message` under `timeout_seconds`, returning success.

    Never raises: `NotFound`, `Forbidden`, timeout, and other errors are logged through
    `report_discord_failure` with `failure_fields` so callers can still schedule cleanup.

    Args:
        message (Message): Public message to edit.
        interaction (Interaction[commands.Bot] | None): Press acknowledged on `message`, or None
            to edit directly via the channel.
        payload (dict[str, Any]): Keyword arguments forwarded to `edit_public_message`.
        not_found_message (str): Log message when `NotFound` is raised.
        forbidden_message (str): Log message when `Forbidden` is raised.
        other_message (str): Log message for timeouts and all other failures.
        timeout_seconds (float): Wall-clock bound in seconds on the edit call.
        **failure_fields (object): Structured logfire fields attached to failure records.
    """
    try:
        await asyncio.wait_for(
            edit_public_message(message=message, interaction=interaction, payload=payload),
            timeout=timeout_seconds,
        )
    # Broad on purpose: callers have already settled or stopped the view, so a failed edit must
    # log and return False instead of skipping settlement or cleanup scheduling.
    except Exception as exc:
        report_discord_failure(
            error=exc,
            not_found_message=not_found_message,
            forbidden_message=forbidden_message,
            other_message=other_message,
            not_found_level="info",
            not_found_exc_info=False,
            other_level="warn",
            **failure_fields,
        )
        return False
    return True


class PressBoundView(LoggedView):
    """A view bound to a public message whose timeouts and cleanup ride its newest press.

    `nextcord` restarts a view's timeout on every press (refused ones included), so keeping the
    newest acknowledged press lets timeout edits and delayed deletion succeed even after the
    opening command's token expires or channel permissions change.
    """

    message: Message | None = None
    last_press: Interaction[commands.Bot] | None = None

    def keep_press(self, interaction: Interaction[commands.Bot]) -> None:
        """Records `interaction` as the press driving timeout edits and delayed deletion.

        Args:
            interaction (Interaction[commands.Bot]): Most recent acknowledged press on the view.
        """
        self.message = interaction.message or self.message
        self.last_press = interaction

    def schedule_delete(
        self,
        user_name: str | None = None,
        message: Message | None = None,
        interaction: Interaction[commands.Bot] | None = None,
    ) -> None:
        """Schedules delayed deletion of the view's public message through its tracked press.

        Args:
            user_name (str | None): Account name recorded for cleanup logs; falls back to
                `interaction.user.name` when `interaction` is provided.
            message (Message | None): Explicit message override; falls back to `self.message`,
                then `interaction.message`.
            interaction (Interaction[commands.Bot] | None): Press override whose token and user
                identity drive cleanup; falls back to `self.last_press` for the token.
        """
        press = interaction if interaction is not None else self.last_press
        target_message = message or self.message or getattr(interaction, "message", None)
        if target_message is None:
            return
        if user_name is None and interaction is not None and interaction.user is not None:
            user_name = interaction.user.name
        schedule_public_message_delete(
            message=target_message, user_name=user_name, interaction=press
        )


async def _deleted_through_press(
    interaction: Interaction[commands.Bot] | None, message: Message
) -> bool:
    """Returns True after deleting `message` via an unexpired `interaction` token, else False."""
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
    """Returns whether `message` was deleted (or already gone) and its cleanup record removed.

    False when Discord refused the delete, which keeps the record for the next process's startup
    sweep.

    Args:
        message (Message): Discord message to delete.
        message_id (int | None): Record key when the message object carries none.
        interaction (Interaction[commands.Bot] | None): Press on the message whose token, while
            it lives, deletes it in place of the channel.
    """
    resolved_message_id = message_id if message_id is not None else getattr(message, "id", None)
    try:
        if not await _deleted_through_press(interaction=interaction, message=message):
            await message.delete()
    except NotFound:
        pass
    except HTTPException as exc:
        report_discord_failure(
            error=exc,
            forbidden_message="Discord refused to delete a public response",
            other_message="Failed to delete public response",
            message_id=resolved_message_id,
            channel_id=getattr(getattr(message, "channel", None), "id", None),
        )
        return False
    if isinstance(resolved_message_id, int):
        await forget_public_message(message_id=resolved_message_id)
    return True


async def delete_tracked_public_messages(bot: commands.Bot, tracked_before: datetime) -> None:
    """Deletes persisted public responses left by an earlier bot process.

    Args:
        bot (commands.Bot): The bot whose channels hold the messages.
        tracked_before (datetime): When this process started. Interactions run before `on_ready`
            does, so a row recorded since is a message this process still owns, a live view
            included.
    """
    records = await list_pending_public_messages(tracked_before=tracked_before)
    deleted_count = 0
    for record in records:
        try:
            message = await fetch_stored_message(
                bot=bot, channel_id=record.channel_id, message_id=record.message_id
            )
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
            await forget_public_message(message_id=record.message_id)
            continue
        except HTTPException as exc:
            report_discord_failure(
                error=exc,
                forbidden_message="Stale public response sits in a channel the bot cannot read",
                other_message="Failed to fetch stale public response",
                channel_id=record.channel_id,
                message_id=record.message_id,
            )
            if isinstance(exc, Forbidden):
                await forget_public_message(message_id=record.message_id)
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
        message (Message): Discord message to delete.
        delay (float): Seconds to wait before deletion.
        user_name (str | None): Optional Discord account name of the user who triggered the
            response.
        interaction (Interaction[commands.Bot] | None): Optional press on the message to delete
            through (`delete_public_message`).
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
    """Schedules delayed deletion for a public response, never blocking the command.

    Args:
        message (Message): Discord message to delete.
        delay (float): Seconds to wait before deletion.
        user_name (str | None): Optional Discord account name of the user who triggered the
            response.
        interaction (Interaction[commands.Bot] | None): Optional press on the message to delete
            through (`delete_public_message`).
    """
    spawn_tracked(
        coro=delete_public_message_after(
            message=message, delay=delay, user_name=user_name, interaction=interaction
        ),
        tasks=_delete_tasks.setdefault(key="delete", default=set()),
        name="delete-public-response",
    )
