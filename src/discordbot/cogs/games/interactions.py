"""Shared helpers for game view interactions."""

from typing import Any, Self, Unpack, ClassVar, TypedDict
import asyncio

import logfire
from nextcord import Embed, Message, NotFound, Interaction
from nextcord.ui import Item, View, Button
from nextcord.ext import commands

from discordbot.typings.timeouts import GAME_FINAL_EDIT_TIMEOUT_SECONDS
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import schedule_public_message_delete
from discordbot.utils.interaction_responses import send_ephemeral_notice


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


async def edit_game_message(
    message: Message, interaction: Interaction[commands.Bot] | None, payload: dict[str, Any]
) -> None:
    """Edits a game message through the press it answers, or through the channel without one.

    A press's own token edits the message its control sits on whatever the channel allows, where
    the channel endpoint answers 403 once the server shuts the bot out. An edit no press
    triggered has only the channel.
    """
    if interaction is None:
        await message.edit(**payload)
    else:
        await interaction.edit_original_message(**payload)


async def publish_final_table(
    message: Message,
    embeds: list[Embed],
    user_name: str,
    game_name: str,
    interaction: Interaction[commands.Bot] | None,
    **failure_fields: Unpack[_FinalRenderFailureFields],
) -> bool:
    """Shows a settled table's final embeds with no controls, then schedules its deletion.

    `interaction` is the press that settled the table, if one did (`edit_game_message`). Never
    raises: settlement is already committed when this runs, so a render that fails is logged
    (`failure_fields` ride the warning) and the deletion is scheduled regardless.

    Returns:
        Whether the final render reached the message.
    """
    try:
        await asyncio.wait_for(
            edit_game_message(
                message=message,
                interaction=interaction,
                payload=table_edit_kwargs(embeds=embeds, view=None, target=message),
            ),
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
