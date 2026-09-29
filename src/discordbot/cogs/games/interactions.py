"""Shared helpers for game view interactions."""

from typing import Any, Self, Final, Unpack, ClassVar, TypedDict
import asyncio
from collections.abc import Callable

import logfire
from nextcord import Embed, Message, NotFound, Interaction
from nextcord.ui import Item, View, Button
from nextcord.ext import commands
from nextcord.errors import DiscordServerError

from discordbot.typings.timeouts import GAME_FINAL_EDIT_TIMEOUT_SECONDS
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import schedule_public_message_delete
from discordbot.utils.interaction_responses import send_ephemeral_notice

_EDIT_ATTEMPTS: Final[int] = 3


class _FinalRenderFailureFields(TypedDict, total=False):
    """What a table adds to the warning its failed final render logs.

    Spread into a logfire call, so the keys stay statically known and never collide with
    logfire's `_tags` / `_exc_info` keyword-only parameters.
    """

    channel_id: int
    message_id: int
    players: int
    reason: str


class GameView(View):
    """Failure logging, private notices and button disabling, shared by every game view.

    A subclass names its two log lines: `interaction_failure_log` for a control whose callback
    raised, `notice_failure_log` for a private notice Discord refused.
    """

    interaction_failure_log: ClassVar[str]
    notice_failure_log: ClassVar[str]

    async def on_error(
        self, error: Exception, item: Item[Self], interaction: Interaction[commands.Bot]
    ) -> None:
        """Logs a control's failure instead of letting nextcord only print it to stderr."""
        logfire.error(
            self.interaction_failure_log,
            item_label=getattr(item, "label", None),
            user_id=getattr(interaction.user, "id", None),
            _exc_info=(type(error), error, error.__traceback__),
        )

    async def _send_notice(self, interaction: Interaction[commands.Bot], content: str) -> None:
        """Sends a private notice to the interacting user; a refusal is logged, never raised."""
        await send_ephemeral_notice(
            interaction=interaction, content=content, log_message=self.notice_failure_log
        )

    def _disable_buttons(self) -> None:
        """Disables every button currently attached to the view."""
        for child in self.children:
            if isinstance(child, Button):
                child.disabled = True


def table_edit_kwargs(
    *, embeds: list[Embed], view: View | None, target: object | None = None
) -> dict[str, Any]:
    """Builds the shared edit payload for a game table render."""
    return {
        "embeds": embeds,
        "view": view,
        **embed_spacer_payload(embeds=embeds, is_edit=True, target=target),
    }


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
    so the backoff spends ~1.5s on that window before the error propagates. What rides on it is
    a game start: an edit that never lands leaves the lobby on screen marked started, refusing
    every press and skipping its timeout cleanup, while the table never appears and a jackpot
    game's antes stay charged.

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


async def publish_final_table(
    message: Message,
    embeds: list[Embed],
    user_name: str,
    game_name: str,
    **failure_fields: Unpack[_FinalRenderFailureFields],
) -> bool:
    """Shows a settled table's final embeds with no controls, then schedules its deletion.

    Never raises: settlement is already committed when this runs, so a render that fails is
    logged (`failure_fields` ride the warning) and the deletion is scheduled regardless.

    Returns:
        Whether the final render reached the message.
    """
    try:
        await asyncio.wait_for(
            message.edit(**table_edit_kwargs(embeds=embeds, view=None, target=message)),
            timeout=GAME_FINAL_EDIT_TIMEOUT_SECONDS,
        )
    except NotFound:
        # Opener deleted the public table before the round finished; nothing to render.
        logfire.info(f"{game_name} table message gone before final edit", message_id=message.id)
        landed = False
    # Broad on purpose: settlement is already committed, so this render must never raise back
    # into the round and skip the cleanup scheduling below.
    except Exception as exc:
        logfire.warn(
            f"{game_name} final table edit failed; settled round never rendered",
            **failure_fields,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
        landed = False
    else:
        landed = True
    schedule_public_message_delete(message=message, user_name=user_name)
    return landed
