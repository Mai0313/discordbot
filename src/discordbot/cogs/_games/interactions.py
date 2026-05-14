"""Shared helpers for game view interactions."""

from typing import Any
import asyncio
from collections.abc import Iterable

import logfire
from nextcord import Message, Interaction, ui
from nextcord.errors import DiscordServerError


async def send_ephemeral_notice(interaction: Interaction, content: str, log_message: str) -> None:
    """Sends an ephemeral interaction notice with response/followup fallback."""
    try:
        if interaction.response.is_done():
            await interaction.followup.send(content=content, ephemeral=True)
            return
        await interaction.response.send_message(content=content, ephemeral=True)
    except Exception:
        logfire.warn(log_message, _exc_info=True)


def disable_view_components(
    children: Iterable[ui.Item], component_types: tuple[type[object], ...]
) -> None:
    """Disables view children matching any supplied component type."""
    for child in children:
        if isinstance(child, component_types):
            child.disabled = True


async def edit_message_with_retry(
    message: Message,
    attempts: int = 3,
    **kwargs: Any,  # noqa: ANN401 -- transparent forwarder to Message.edit's heterogeneous kwargs
) -> Message:
    """Edits ``message`` retrying transient Discord 5xx errors with backoff.

    Cloudflare in front of discord.com occasionally returns 502/503/504 for a
    couple of seconds; the game-start edits must succeed or the lobby is left
    stopped with antes already charged. Backoff grows 0.5s, 1.0s, ... so the
    final attempt covers ~1.5s of upstream flakiness before propagating.
    """
    for attempt in range(attempts - 1):
        try:
            return await message.edit(**kwargs)
        except DiscordServerError as error:
            logfire.warn(
                "Discord 5xx on message.edit, retrying",
                attempt=attempt + 1,
                status=error.status,
                message_id=message.id,
            )
            await asyncio.sleep(0.5 * (attempt + 1))
    return await message.edit(**kwargs)
