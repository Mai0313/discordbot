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
    """Extra logfire fields attached when a table's final render fails."""

    channel_id: int
    message_id: int
    players: int
    reason: str


class GameView(LoggedView):
    """Failure logging, private notices, and button disabling shared by every game view.

    Attributes:
        interaction_failure_log (ClassVar[str]): Log message when a control callback raises.
        notice_failure_log (ClassVar[str]): Log message when Discord refuses a private notice.
        message (Message | None): The game's public message.
        last_press (Interaction[commands.Bot] | None): Most recent press acknowledged on the
            message, whose token drives timeout edits and deletion while alive.
    """

    interaction_failure_log: ClassVar[str]
    notice_failure_log: ClassVar[str]
    message: Message | None
    last_press: Interaction[commands.Bot] | None

    def _keep_press(self, interaction: Interaction[commands.Bot]) -> None:
        """Records `interaction` as the press a timeout will edit and delete the table through."""
        self.message = interaction.message or self.message
        self.last_press = interaction

    async def _send_notice(self, interaction: Interaction[commands.Bot], content: str) -> None:
        """Defers `interaction` on the game message if needed, records it, and sends an ephemeral notice."""
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
    """Returns the shared edit payload for a game table render.

    Args:
        embeds (list[Embed]): Table embeds to render.
        view (View | None): Active component view, or None once settled.
        target (object | None): Message or interaction used to size the spacer attachment.
    """
    return {
        "embeds": embeds,
        "view": view,
        **embed_spacer_payload(embeds=embeds, is_edit=True, target=target),
    }


def set_view_item_visible(view: View, item: Item[View], visible: bool) -> None:
    """Adds or removes `item` on `view` without recreating the component.

    Args:
        view (View): Target view to mutate.
        item (Item[View]): Control to show or hide.
        visible (bool): Whether `item` should be present in `view.children`.
    """
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
    """Returns whether the settled table's final embeds rendered, and schedules message deletion.

    `interaction` is the press that settled the table or, on a timeout, the last press
    acknowledged on it; the render and the deletion both go through it while its token lives
    (`edit_public_message`). Never raises: settlement is already committed when this runs, so a
    render that fails is logged (`failure_fields` ride the warning) and the deletion is
    scheduled regardless.

    Args:
        message (Message): Public table message to edit and schedule for deletion.
        embeds (list[Embed]): Final settled table embeds.
        user_name (str): Account name of the table owner for cleanup tracking.
        game_name (str): Game label used in failure logs.
        interaction (Interaction[commands.Bot] | None): Press whose token drives the edit and
            deletion while alive.
        **failure_fields (Unpack[_FinalRenderFailureFields]): Structured fields attached to
            render-failure warnings.
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
