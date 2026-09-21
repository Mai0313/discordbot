"""Shared helpers for game view interactions."""

from typing import Any, Final
import asyncio
from collections.abc import Callable, Iterable

import logfire
from nextcord import Embed, Message
from nextcord.ui import Item, View, Button
from nextcord.errors import DiscordServerError

from discordbot.utils.discord_embeds import embed_spacer_payload

_EDIT_ATTEMPTS: Final[int] = 3


def table_edit_kwargs(
    *, embeds: list[Embed], view: View | None, target: object | None = None
) -> dict[str, Any]:
    """Builds the shared edit payload for a game table render."""
    return {
        "embeds": embeds,
        "view": view,
        **embed_spacer_payload(embeds=embeds, is_edit=True, target=target),
    }


def disable_view_components(
    children: Iterable[Item[View]], component_types: tuple[type[Button[View]], ...]
) -> None:
    """Disables view children matching any supplied component type."""
    for child in children:
        if isinstance(child, component_types) and isinstance(child, Button):
            child.disabled = True


def set_view_item_visible(view: View, item: Item[View], visible: bool) -> None:
    """Adds or removes one view item without recreating the component."""
    if visible and item not in view.children:
        view.add_item(item=item)
    elif not visible and item in view.children:
        view.remove_item(item=item)


async def edit_message_with_retry(
    message: Message, kwargs_factory: Callable[[], dict[str, Any]]
) -> Message:
    """Edits `message`, retrying transient Discord 5xx errors with backoff.

    Cloudflare in front of discord.com returns 502/503/504 for a couple of seconds at a time,
    and a game-start edit that never lands leaves the lobby stopped with antes already charged,
    so the backoff spends ~1.5s on that window before the error propagates.

    The payload is rebuilt per attempt because a failed one has already consumed any upload
    streams it carries.
    """
    for attempt in range(_EDIT_ATTEMPTS - 1):
        try:
            return await message.edit(**kwargs_factory())
        except DiscordServerError as error:
            logfire.warn(
                "Discord 5xx on message.edit, retrying",
                attempt=attempt + 1,
                status=error.status,
                message_id=message.id,
                _exc_info=error,
            )
            await asyncio.sleep(0.5 * (attempt + 1))
    return await message.edit(**kwargs_factory())
