"""Shared helpers for game view interactions."""

from typing import Any
import asyncio
from collections.abc import Callable, Iterable

import logfire
from nextcord import Message
from nextcord.ui import Item, View
from nextcord.errors import DiscordServerError


def disable_view_components(
    children: Iterable[Item], component_types: tuple[type[Item], ...]
) -> None:
    """Disables view children matching any supplied component type."""
    for child in children:
        if isinstance(child, component_types):
            child.disabled = True


def set_view_item_visible(view: View, item: Item, visible: bool) -> None:
    """Adds or removes one view item without recreating the component."""
    if visible and item not in view.children:
        view.add_item(item=item)
    elif not visible and item in view.children:
        view.remove_item(item=item)


async def edit_message_with_retry(
    message: Message,
    attempts: int = 3,
    kwargs_factory: Callable[[], dict[str, Any]] | None = None,
    **kwargs: Any,  # noqa: ANN401 -- transparent forwarder to Message.edit's heterogeneous kwargs
) -> Message:
    """Edits `message` retrying transient Discord 5xx errors with backoff.

    Cloudflare in front of discord.com occasionally returns 502/503/504 for a
    couple of seconds; the game-start edits must succeed or the lobby is left
    stopped with antes already charged. Backoff grows 0.5s, 1.0s, ... so the
    final attempt covers ~1.5s of upstream flakiness before propagating.
    """

    def edit_kwargs() -> dict[str, Any]:
        return kwargs_factory() if kwargs_factory is not None else kwargs

    for attempt in range(attempts - 1):
        try:
            return await message.edit(**edit_kwargs())
        except DiscordServerError as error:
            logfire.warn(
                "Discord 5xx on message.edit, retrying",
                attempt=attempt + 1,
                status=error.status,
                message_id=message.id,
                _exc_info=error,
            )
            await asyncio.sleep(0.5 * (attempt + 1))
    return await message.edit(**edit_kwargs())
