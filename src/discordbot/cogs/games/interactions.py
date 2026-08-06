"""Shared nextcord view plumbing for the casino tables, carrying no game rules.

Three helpers the cog's views all need:

- `disable_view_components` greys a control set out while leaving it on screen, for the
  moments a view is stopping or is mid-animation: the lobby timeout, Blackjack's
  `_finalize_locked` before any money moves, and the two-stage dealer peek.
- `set_view_item_visible` is the other half of that pair and the one used far more often. It
  is what makes the table controls presence-based: `BlackjackView.sync_buttons` and
  `DragonGateView.sync_controls` drop every control and add back only the legal ones, so a
  table never renders a dead button.
- `edit_message_with_retry` covers the one edit that must not be lost, the lobby-to-table
  handover that runs after the antes are already charged.

These sit here rather than in `utils/` because every caller is one of this cog's own views
(`lobby.py`, `blackjack_views.py`, `dragon_gate_views.py`). A second cog needing one would
have to move it down instead of importing it, since nothing under `cogs/<a>/` may import
from `cogs/<b>/`.
"""

from typing import Any
import asyncio
from collections.abc import Callable, Iterable

import logfire
from nextcord import Message
from nextcord.ui import Item, View, Button
from nextcord.errors import DiscordServerError


def disable_view_components(
    children: Iterable[Item[View]], component_types: tuple[type[Button[View]], ...]
) -> None:
    """Disables every child that is an instance of one of `component_types`.

    Args:
        children (Iterable[Item[View]]): The view's children; matching ones are mutated in place.
        component_types (tuple[type[Button[View]], ...]): Component classes to disable.
    """
    for child in children:
        # The second check narrows for the type checker: `disabled` lives on the concrete
        # components, not on `Item`.
        if isinstance(child, component_types) and isinstance(child, Button):
            child.disabled = True


def set_view_item_visible(view: View, item: Item[View], visible: bool) -> None:
    """Adds or removes one view item without recreating the component.

    Keeping the same object keeps its callback binding and its label / option state, so a
    control that comes back is the one that left. A re-added item is appended, so the rendered
    component order follows the order of the calls that added it back rather than the order the
    view was built in; that is why the table views hide their whole control set and re-add it in
    one deterministic pass instead of toggling items individually.

    Args:
        view (View): The view whose children are adjusted.
        item (Item[View]): The component to show or hide.
        visible (bool): Whether the item should be present on the view.
    """
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
    """Edits `message`, retrying transient Discord 5xx errors with backoff.

    Cloudflare in front of discord.com occasionally returns 502/503/504 for a couple of seconds;
    the game-start edits must succeed or the lobby is left stopped with antes already charged.
    Backoff grows 0.5s, 1.0s, ... so the default three attempts cover ~1.5s of upstream flakiness
    before the last failure propagates. Only `DiscordServerError` is retried; anything else, a
    permission or validation error included, propagates from the attempt that raised it.

    Pass `kwargs_factory` whenever the payload carries files. A `File` is consumed by the attempt
    that failed, so a retry reusing the same object would upload an exhausted stream; the factory
    is called once per attempt, after the previous failure has been awaited.

    Args:
        message (Message): The message to edit.
        attempts (int): Total edits to try. Anything below 2 still makes the one uncaught attempt.
        kwargs_factory (Callable[[], dict[str, Any]] | None): Builds a fresh payload per attempt.
            When None the fixed `**kwargs` are reused on every attempt.
        **kwargs (Any): Forwarded to `Message.edit`, ignored when `kwargs_factory` is given.

    Returns:
        The edited message, as returned by `Message.edit`.
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
