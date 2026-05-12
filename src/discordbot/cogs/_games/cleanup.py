"""Timed cleanup helpers for casino game messages."""

import asyncio

import logfire
import nextcord
from nextcord import Message

GAME_RESPONSE_TTL_SECONDS = 180


async def delete_game_message_after(
    *, message: Message, delay: float = GAME_RESPONSE_TTL_SECONDS
) -> None:
    """Deletes a game response after a delay.

    Args:
        message: Discord message to delete.
        delay: Seconds to wait before deletion.
    """
    await asyncio.sleep(delay=delay)
    try:
        await message.delete()
    except nextcord.NotFound:
        return
    except (nextcord.Forbidden, nextcord.HTTPException):
        logfire.warn("Failed to delete expired game response", _exc_info=True)


def schedule_game_message_delete(
    *, message: Message, delay: float = GAME_RESPONSE_TTL_SECONDS
) -> None:
    """Schedules delayed deletion for a casino game response."""
    asyncio.create_task(  # noqa: RUF006 -- fire-and-forget cleanup cannot block commands.
        coro=delete_game_message_after(message=message, delay=delay), name="delete-game-response"
    )
