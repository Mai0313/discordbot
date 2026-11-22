"""Shared send/edit helpers for interaction responses.

Each embed helper pairs one response shape (public followup, public followup after a private
defer, tracked-view followup, private followup, ephemeral response, edit) with the
`embed_spacer_payload` call that keeps embed widths aligned; the plain-text ephemeral notice
needs none. Only the public followups schedule their own deletion up front; the tracked-view
one hands that to the view, which schedules it at a terminal state.
"""

import logfire
from nextcord import File, Embed, Interaction, HTTPException
from nextcord.ext import commands

from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import (
    PressBoundView,
    track_public_message,
    schedule_public_message_delete,
)


async def send_expiring_followup(
    interaction: Interaction[commands.Bot], embed: Embed, file: File | None = None
) -> None:
    """Sends a public embed as an interaction followup and schedules its deletion.

    Args:
        interaction (Interaction[commands.Bot]): Interaction to send the followup on.
        embed (Embed): Public embed to send.
        file (File | None): Optional attachment sent alongside `embed`.
    """
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

    Args:
        interaction (Interaction[commands.Bot]): Interaction that deferred ephemerally.
        embed (Embed): Public embed to send.
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


async def send_tracked_view_followup(
    interaction: Interaction[commands.Bot],
    embed: Embed,
    view: PressBoundView,
    user_name: str | None = None,
) -> None:
    """Sends a public view followup, binds `view.message`, and tracks it for restart cleanup.

    `view.message` is bound before the tracking write so a timeout or press racing the DB write
    already sees the message. Cleanup after a terminal state stays with `view`.

    Args:
        interaction (Interaction[commands.Bot]): Interaction to send the followup on.
        embed (Embed): Public embed attached to the view.
        view (PressBoundView): View whose `message` attribute receives the sent message.
        user_name (str | None): Account name recorded for cleanup logs; defaults to
            `interaction.user.name` when omitted.
    """
    message = await interaction.followup.send(
        embed=embed,
        view=view,
        wait=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )
    view.message = message
    resolved_user = (
        user_name
        if user_name is not None
        else (interaction.user.name if interaction.user is not None else None)
    )
    await track_public_message(message=message, user_name=resolved_user)


async def send_private_followup(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Sends a personal embed visible only to the caller.

    Args:
        interaction (Interaction[commands.Bot]): Interaction to send the followup on.
        embed (Embed): Ephemeral embed to send.
    """
    await interaction.followup.send(
        embed=embed,
        ephemeral=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )


async def send_ephemeral_response(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Sends an ephemeral embed as the initial interaction response.

    Args:
        interaction (Interaction[commands.Bot]): Interaction to respond to.
        embed (Embed): Ephemeral embed to send.
    """
    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )


async def edit_response_embed(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Edits the interaction's public message embed and clears its controls.

    Edits the original rather than responding with one: callers have already deferred before the
    write that decides this embed.

    Args:
        interaction (Interaction[commands.Bot]): Deferred interaction whose original message is
            edited.
        embed (Embed): Updated public embed.
    """
    await interaction.edit_original_message(
        embed=embed,
        view=None,
        **embed_spacer_payload(embeds=[embed], is_edit=True, target=interaction),
    )


async def send_ephemeral_notice(
    interaction: Interaction[commands.Bot], content: str, log_message: str
) -> None:
    """Sends an ephemeral interaction notice with response/followup fallback.

    Args:
        interaction (Interaction[commands.Bot]): Interaction to notify.
        content (str): Notice text shown to the caller.
        log_message (str): Warning log message if Discord refuses the notice.
    """
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
