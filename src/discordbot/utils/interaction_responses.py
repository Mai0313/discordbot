"""Shared interaction send/edit helpers for the economy and games surfaces.

Five thin wrappers over `interaction.followup` / `interaction.response`, each one a
(visibility, lifetime) pair a cog picks from instead of re-deriving:

- `send_expiring_followup` sends a public followup and schedules its deletion.
- `send_loan_request_followup` sends a public followup whose view owns the cleanup instead.
- `send_private_followup` answers only the caller, after the command deferred.
- `send_ephemeral_response` answers only the caller as the initial, un-deferred response.
- `edit_response_embed` rewrites the message a component was clicked on and drops its controls.

What all five promise is the two things easiest to forget at a call site. Every embed goes out
through `embed_spacer_payload`, so embeds render at an aligned width and an edit retains the
spacer already on the message rather than re-uploading it (Discord error 400009); and a public
message either enters `message_cleanup` here or is handed to a view that will schedule it, so
nothing public is left behind. Nothing else is promised: these do not defer, do not build
embeds, take no `ephemeral` argument (each one's visibility is fixed, so the call site picks it
by picking a helper), and swallow no Discord error, so a failed send still raises into the
command's own error handling.

They sit in `utils/` because `cogs/economy/cog.py`, `cogs/economy/views.py` and
`cogs/games/cog.py` all need the same behavior and a cog may not import a peer cog.
"""

from typing import Protocol, cast

from nextcord import File, Embed, Message, Interaction
from nextcord.ui import View
from nextcord.ext import commands

from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import schedule_public_message_delete


class _MessageOwningView(Protocol):
    """The one attribute `send_loan_request_followup` writes back onto a loan-decision view.

    nextcord's `View` declares no `message`, so casting to this is what lets the assignment
    type-check without `utils/` importing the economy views it is actually handed.
    """

    message: Message | None


async def send_expiring_followup(
    interaction: Interaction[commands.Bot],
    embed: Embed,
    view: View | None = None,
    file: File | None = None,
) -> None:
    """Sends a public followup embed and schedules its deletion after the shared TTL.

    `wait=True` buys a static type rather than a runtime one: `Webhook.send` forces `wait` True
    on an interaction followup anyway, since that webhook's type is `application`, but only the
    `Literal[True]` overload is declared to return a `WebhookMessage` instead of None, and a
    `Message` is what `schedule_public_message_delete` takes. `file` rides inside the spacer
    payload rather than being passed on its own, so it counts against Discord's per-message file
    cap alongside the spacer and both leave in one `files` argument. `view` is omitted entirely
    when None instead of being forwarded: nextcord's `Webhook.send` tests it against MISSING
    rather than None and then calls `view.is_finished()` on it.

    Args:
        interaction (Interaction[commands.Bot]): The already-deferred interaction to answer.
        embed (Embed): The embed to send.
        view (View | None): Controls to attach, or None for a plain embed.
        file (File | None): One extra attachment, such as a rendered leaderboard board.
    """
    extra_files = [file] if file is not None else None
    spacer = embed_spacer_payload(
        embeds=[embed], is_edit=False, target=interaction, extra_files=extra_files
    )
    if view is not None:
        message = await interaction.followup.send(embed=embed, view=view, wait=True, **spacer)
    else:
        message = await interaction.followup.send(embed=embed, wait=True, **spacer)
    user_name = interaction.user.name if interaction.user is not None else None
    schedule_public_message_delete(message=message, user_name=user_name)


async def send_loan_request_followup(
    interaction: Interaction[commands.Bot], embed: Embed, view: View
) -> None:
    """Sends a public loan request and records the sent message on its own view.

    Deliberately schedules no cleanup: a loan request outlives the interaction and is tidied up
    once the view reaches a terminal state, including from `on_timeout`, where there is no
    interaction left to reach the message through. That is the whole reason the message is
    written back onto the view.

    Args:
        interaction (Interaction[commands.Bot]): The already-deferred interaction to answer.
        embed (Embed): The loan request embed.
        view (View): The loan-decision view that will own the message and its cleanup.
    """
    message = await interaction.followup.send(
        embed=embed,
        view=view,
        wait=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )
    cast("_MessageOwningView", view).message = message


async def send_private_followup(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Sends an ephemeral followup embed visible only to the caller.

    Never enters the public cleanup scheduler: nobody else can see it, so there is nothing in
    the channel to tidy up. Use this after the command deferred; `send_ephemeral_response` is
    the un-deferred twin.

    Args:
        interaction (Interaction[commands.Bot]): The already-deferred interaction to answer.
        embed (Embed): The embed to send.
    """
    await interaction.followup.send(
        embed=embed,
        ephemeral=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )


async def send_ephemeral_response(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Sends an ephemeral embed as the interaction's initial response.

    For the rejections that answer before anything is deferred or mutated: a malformed amount,
    a permission check on a component click. An interaction that already deferred must take
    `send_private_followup` instead, since the initial response is spent.

    Args:
        interaction (Interaction[commands.Bot]): The not-yet-answered interaction.
        embed (Embed): The embed to send.
    """
    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )


async def edit_response_embed(interaction: Interaction[commands.Bot], embed: Embed) -> None:
    """Rewrites the message a component was clicked on and removes its controls.

    `view=None` is nextcord's spelling for dropping the components, leaving the message as a
    plain record of the outcome. `is_edit=True` keeps the spacer already uploaded to that
    message instead of sending another one, and carries the retained attachment list an edit
    needs to avoid dropping it.

    Args:
        interaction (Interaction[commands.Bot]): The component interaction to answer.
        embed (Embed): The embed replacing the message's current one.
    """
    await interaction.response.edit_message(
        embed=embed,
        view=None,
        **embed_spacer_payload(embeds=[embed], is_edit=True, target=interaction),
    )
