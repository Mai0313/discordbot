"""Shared helpers for game view interactions."""

from typing import Any, Unpack, ClassVar, TypedDict
import asyncio

import logfire
from nextcord import Embed, Message, NotFound, Forbidden, Interaction
from nextcord.ui import Item, View, Button
from nextcord.ext import commands

from discordbot.utils.logged_ui import LoggedView
from discordbot.typings.timeouts import GAME_FINAL_EDIT_TIMEOUT_SECONDS
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import edit_public_message, schedule_public_message_delete
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


class GameView(LoggedView):
    """Failure logging, private notices and button disabling, shared by every game view.

    A subclass names its two log lines: `interaction_failure_log` for a control whose callback
    raised, `notice_failure_log` for a private notice Discord refused.
    """

    interaction_failure_log: ClassVar[str]
    notice_failure_log: ClassVar[str]
    # The game's public message and the last press acknowledged on it; a timeout's edit and
    # delete ride that press's token while it lives.
    message: Message | None
    last_press: Interaction[commands.Bot] | None

    def _keep_press(self, interaction: Interaction[commands.Bot]) -> None:
        """Makes a press acknowledged on the game message the one a timeout closes it through.

        nextcord restarts the view's timer on every press, a refused one included, so only the
        newest press holds a token sure to outlive that timer and the cleanup after it.
        """
        self.message = interaction.message or self.message
        self.last_press = interaction

    async def _send_notice(self, interaction: Interaction[commands.Bot], content: str) -> None:
        """Acknowledges the press on the game message, then sends the user a private notice.

        Acknowledging first makes the notice a followup, so the press's own token still reaches
        the game message (`_keep_press`). A refused notice is logged, never raised; a failed
        acknowledgement raises like any press's defer.
        """
        if not interaction.response.is_done():
            await interaction.response.defer()
        self._keep_press(interaction=interaction)
        await send_ephemeral_notice(
            interaction=interaction, content=content, log_message=self.notice_failure_log
        )

    def _disable_buttons(self) -> None:
        """Disables every button currently attached to the view."""
        for child in self.children:
            if isinstance(child, Button):
                child.disabled = True


def table_edit_kwargs(
    embeds: list[Embed], view: View | None, target: object | None = None
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


async def publish_final_table(
    message: Message,
    embeds: list[Embed],
    user_name: str,
    game_name: str,
    interaction: Interaction[commands.Bot] | None,
    **failure_fields: Unpack[_FinalRenderFailureFields],
) -> bool:
    """Shows a settled table's final embeds with no controls, then schedules its deletion.

    `interaction` is the press that settled the table or, on a timeout, the last press
    acknowledged on it; the render and the deletion both go through it while its token lives
    (`edit_public_message`). Never raises: settlement is already committed when this runs, so a
    render that fails is logged (`failure_fields` ride the warning) and the deletion is
    scheduled regardless.

    Returns:
        Whether the final render reached the message.
    """
    try:
        await asyncio.wait_for(
            edit_public_message(
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
    except Forbidden:
        # Only a render with no working press behind it goes through the channel, which the
        # server can shut the bot out of; the ids are the whole finding.
        logfire.warn(
            f"Discord refused the {game_name} final table edit; settled round never rendered",
            **failure_fields,
        )
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
    schedule_public_message_delete(message=message, user_name=user_name, interaction=interaction)
    return landed
