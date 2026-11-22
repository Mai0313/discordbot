"""Shared helpers for game view interactions."""

from typing import Any, Unpack, ClassVar, TypedDict

from nextcord import Embed, Message, Interaction
from nextcord.ui import Item, View, Button
from nextcord.ext import commands

from discordbot.typings.timeouts import GAME_FINAL_EDIT_TIMEOUT_SECONDS
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import (
    PressBoundView,
    edit_or_report,
    schedule_public_message_delete,
)
from discordbot.utils.interaction_responses import send_ephemeral_notice


class _FinalRenderFailureFields(TypedDict, total=False):
    """Extra logfire fields attached when a table's final render fails."""

    channel_id: int
    message_id: int
    players: int
    reason: str


class GameView(PressBoundView):
    """Failure logging, private notices, and button disabling shared by every game view.

    Attributes:
        interaction_failure_log (ClassVar[str]): Log message when a control callback raises.
        notice_failure_log (ClassVar[str]): Log message when Discord refuses a private notice.
    """

    interaction_failure_log: ClassVar[str]
    notice_failure_log: ClassVar[str]

    async def _send_notice(self, interaction: Interaction[commands.Bot], content: str) -> None:
        """Defers `interaction` if needed, records it, and sends an ephemeral notice."""
        if not interaction.response.is_done():
            await interaction.response.defer()
        self.keep_press(interaction=interaction)
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
    (`edit_or_report`). Never raises: render failures are logged with `failure_fields` and
    deletion is scheduled regardless.

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
    fields: dict[str, object] = {"message_id": message.id, **failure_fields}
    landed = await edit_or_report(
        message=message,
        interaction=interaction,
        payload=table_edit_kwargs(embeds=embeds, view=None, target=message),
        not_found_message=f"{game_name} table message gone before final edit",
        forbidden_message=(
            f"Discord refused the {game_name} final table edit; settled round never rendered"
        ),
        other_message=f"{game_name} final table edit failed; settled round never rendered",
        timeout_seconds=GAME_FINAL_EDIT_TIMEOUT_SECONDS,
        **fields,
    )
    schedule_public_message_delete(message=message, user_name=user_name, interaction=interaction)
    return landed
