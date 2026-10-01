"""Shared send/edit helpers for interaction responses.

Each embed helper pairs one response shape (public followup, public followup after a private
defer, loan-request followup, private followup, ephemeral response, edit) with the
`embed_spacer_payload` call that keeps embed widths aligned; the plain-text ephemeral notice
needs none. Only the public followups schedule their own deletion up front; the loan-request one
hands that to the view, which schedules it at a terminal state.
"""

from typing import Protocol, cast

import logfire
from nextcord import File, Embed, Message, Interaction, HTTPException
from nextcord.ui import View
from nextcord.ext import commands

from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import track_public_message, schedule_public_message_delete


class _MessageOwningView(Protocol):
    """The subset of a loan-decision view needed to record its sent message."""

    message: Message | None


async def send_expiring_followup(
    interaction: Interaction[commands.Bot], embed: Embed, file: File | None = None
) -> None:
    """Sends a public embed as an interaction followup and schedules its deletion."""
    extra_files = [file] if file is not None else None
    spacer = embed_spacer_payload(
        embeds=[embed], is_edit=False, target=interaction, extra_files=extra_files
    )
    message = await interaction.followup.send(embed=embed, wait=True, **spacer)
    user_name = interaction.user.name if interaction.user is not None else None
    schedule_public_message_delete(message=message, user_name=user_name)


async def send_expiring_followup_after_private_defer(
    interaction: Interaction[commands.Bot], embed: Embed
) -> None:
    """Sends a public expiring embed for a slash command that deferred ephemerally.

    The first followup after that defer fills its placeholder and keeps the defer's flag, so the
    caller gets the embed there first and that copy is withdrawn once the public one is up.
    """
    await send_private_followup(interaction=interaction, embed=embed)
    await send_expiring_followup(interaction=interaction, embed=embed)
    try:
        await interaction.delete_original_message()
    except HTTPException:
        # The public embed is already up; a failure here only leaves the caller a second copy.
        logfire.warn(
            "Could not withdraw the private copy of a public followup",
            interaction_id=interaction.id,
            guild_id=interaction.guild_id,
            _exc_info=True,
        )


async def send_loan_request_followup(
    interaction: Interaction[commands.Bot], embed: Embed, view: View
) -> None:
    """Sends a loan request message that owns its cleanup after a terminal state.

    The message is also recorded at once, since its view dies with the process: a restart's
    sweep is then the only thing left that can take it down.
    """
    message = await interaction.followup.send(
        embed=embed,
        view=view,
        wait=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )
    cast("_MessageOwningView", view).message = message
    user_name = interaction.user.name if interaction.user is not None else None
    await track_public_message(message=message, user_name=user_name)


async def send_private_followup(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Sends a personal embed visible only to the caller."""
    await interaction.followup.send(
        embed=embed,
        ephemeral=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )


async def send_ephemeral_response(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Sends an ephemeral embed as the initial interaction response."""
    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )


async def edit_response_embed(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Edits the interaction's public message embed and clears its controls.

    Edits the original rather than responding with one, because every caller has already
    deferred: the write that decides this embed can outlive the three-second response window,
    so the acknowledgement cannot wait for it.
    """
    await interaction.edit_original_message(
        embed=embed,
        view=None,
        **embed_spacer_payload(embeds=[embed], is_edit=True, target=interaction),
    )


async def send_ephemeral_notice(
    interaction: Interaction[commands.Bot], content: str, log_message: str
) -> None:
    """Sends an ephemeral interaction notice with response/followup fallback."""
    try:
        if interaction.response.is_done():
            await interaction.followup.send(content=content, ephemeral=True)
            return
        await interaction.response.send_message(content=content, ephemeral=True)
    # Broad on purpose: the notice is advisory, and every way Discord can refuse it (an expired
    # token, an already-answered response, a transient HTTP error) must leave the caller's own
    # flow running, whether that is an interaction check or a button callback.
    except Exception as exc:
        logfire.warn(
            log_message,
            user_id=interaction.user.id if interaction.user is not None else None,
            channel_id=interaction.channel_id,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
