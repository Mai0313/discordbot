"""Shared send/edit helpers for economy interaction responses."""

from typing import Any, Protocol, cast

from nextcord import File, Embed, Message, Interaction
from nextcord.ui import View
from nextcord.ext import commands

from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import schedule_public_message_delete


class _MessageOwningView(Protocol):
    """The subset of a loan-decision view needed to record its sent message."""

    message: Message | None


async def send_expiring_followup(
    interaction: Interaction[commands.Bot],
    embed: Embed,
    view: View | None = None,
    file: File | None = None,
) -> None:
    """Sends a game-related economy embed and schedules its cleanup."""
    extra_files = [file] if file is not None else None
    kwargs: dict[str, Any] = {
        "embed": embed,
        "wait": True,
        **embed_spacer_payload(
            embeds=[embed], is_edit=False, target=interaction, extra_files=extra_files
        ),
    }
    if view is not None:
        kwargs["view"] = view
    message = await interaction.followup.send(**kwargs)
    user_name = interaction.user.name if interaction.user is not None else None
    schedule_public_message_delete(message=message, user_name=user_name)


async def send_loan_request_followup(
    interaction: Interaction[commands.Bot], embed: Embed, view: View
) -> None:
    """Sends a loan request message that owns its cleanup after a terminal state."""
    message = await interaction.followup.send(
        embed=embed,
        view=view,
        wait=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )
    cast("_MessageOwningView", view).message = message


async def send_private_followup(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Sends a personal economy embed visible only to the caller."""
    await interaction.followup.send(
        embed=embed,
        ephemeral=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )


async def send_ephemeral_response(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Sends an ephemeral economy embed as the initial interaction response."""
    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )


async def edit_response_embed(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Edits the interaction's public message embed and clears its controls."""
    await interaction.response.edit_message(
        embed=embed,
        view=None,
        **embed_spacer_payload(embeds=[embed], is_edit=True, target=interaction),
    )
