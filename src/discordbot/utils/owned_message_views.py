"""Shared scaffolding for views that own one public Discord message.

`/stock` and `/games fishing` share the same UX: one public message edited in place, operable
only by the user who opened it, deleted after an idle timeout. Three exports carry that shape.
`OwnedPublicView` contributes the owner gate and the timeout deletion, and holds the message the
panel currently lives on so that deletion can find it. `edit_owned_public_message` decides which
Discord call updates that message for the interaction in hand, and recovers when there is no
message to edit or it has been deleted underneath the panel. What the panel shows stays with
each cog: embeds, controls, notice text, and the decision to transition are none of this
module's business, and every transition still builds its own new view.

`send_ephemeral_notice` is the third export and the most widely used one — blackjack, dragon
gate, the games lobby and the `/games blackjack_history` command take it alone, without the base
view, for their own refusal replies.

It sits in `utils/` rather than inside the stock or the fishing cog because a cog may not import
from a peer cog at all, so the second feature to want this shape would otherwise have copied it.
"""

from io import BytesIO
from typing import cast
from collections.abc import Callable

import logfire
from nextcord import File, Embed, Message, NotFound, Attachment, Interaction
from nextcord.ui import View
from nextcord.ext import commands

from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import (
    track_public_message,
    delete_public_message,
    forget_public_message,
)


async def send_ephemeral_notice(
    interaction: Interaction[commands.Bot], content: str, log_message: str
) -> None:
    """Sends an ephemeral notice, as a followup when the interaction is already answered.

    A notice is never the deliverable — it explains a refusal that has already happened — so a
    delivery failure is logged and swallowed rather than raised into the component callback that
    was rejecting the press.

    Args:
        interaction (Interaction[commands.Bot]): The interaction to answer.
        content (str): The notice text shown to the pressing user only.
        log_message (str): What to log when the notice cannot be delivered.
    """
    try:
        if interaction.response.is_done():
            await interaction.followup.send(content=content, ephemeral=True)
            return
        await interaction.response.send_message(content=content, ephemeral=True)
    # Stays broad: an expired interaction, a lost gateway and a permission change all end the
    # same way here, and none of them may cost the caller its own path.
    except Exception:
        logfire.warn(log_message, _exc_info=True)


class OwnedPublicView(View):
    """Base view for panels that own one public Discord message.

    Subclasses add their own controls; this base adds the owner gate, the idle-timeout deletion,
    and `message`, which is the only handle that deletion has on the panel. nextcord does not
    populate it, so a view that never ran `bind_message` times out doing nothing; the owner gate
    is unaffected, since it weighs the presser against `owner_id` alone.
    """

    def __init__(
        self,
        owner_id: int,
        timeout_seconds: float,
        owner_mismatch_notice: str,
        delete_on_timeout: bool = True,
    ) -> None:
        """Initializes the owned-panel controls with an idle timeout.

        The timeout is nextcord's idle timer, restarted by every interaction it dispatches to
        this view — including one `interaction_check` is about to reject, since the refresh runs
        first — so a stranger pressing a control keeps the owner's panel alive a while longer.

        Args:
            owner_id (int): Discord id of the user allowed to operate this panel.
            timeout_seconds (float): Idle seconds before `on_timeout` runs.
            owner_mismatch_notice (str): Ephemeral text shown to anyone else who presses.
            delete_on_timeout (bool): Whether going idle deletes the bound message; False leaves
                it on screen with its controls dead.
        """
        super().__init__(timeout=timeout_seconds)
        self.owner_id = owner_id
        self.owner_mismatch_notice = owner_mismatch_notice
        self.delete_on_timeout = delete_on_timeout
        self.message: Message | None = None

    def bind_message(self, message: Message | None) -> None:
        """Records the message this view should update or delete.

        Every panel transition replaces the view, so the replacement has to be re-bound before
        it can time out usefully. `edit_owned_public_message` does that for each transition; the
        command that first posts the panel does it at its own call site.

        Args:
            message (Message | None): The message the panel now lives on, or None when it is not
                known yet.
        """
        self.message = message

    async def interaction_check(self, interaction: Interaction[commands.Bot]) -> bool:
        """Allows only the user who opened this panel to operate it.

        A rejection answers the presser ephemerally, so a shared channel never sees one user's
        refusal, and the panel itself is left untouched.

        Args:
            interaction (Interaction[commands.Bot]): The interaction nextcord is dispatching.

        Returns:
            True when the presser opened this panel.

        Raises:
            RuntimeError: The interaction carries no Discord user to check against.
        """
        if interaction.user is None:
            raise RuntimeError("Interaction is missing Discord user identity")
        if self.owner_id == interaction.user.id:
            return True
        await send_ephemeral_notice(
            interaction=interaction,
            content=self.owner_mismatch_notice,
            log_message="Failed to send owner mismatch notice",
        )
        return False

    async def on_timeout(self) -> None:
        """Deletes the panel's message once it has gone idle, unless the panel opted out.

        Goes through `delete_public_message` rather than `Message.delete` so the persisted
        cleanup row goes with it and a later restart sweep does not chase a message that is
        already gone. Two views return quietly instead: an unbound one has nothing to delete, and
        one built with `delete_on_timeout=False` is meant to stay on screen.
        """
        if self.message is None or not self.delete_on_timeout:
            return
        await delete_public_message(message=self.message)


def _fresh_file_factory(file: File | None) -> Callable[[], File] | None:
    """Reads an upload once and returns a factory minting an identical `File` per request.

    nextcord documents a `File` as single-use: the request that sends it reads its buffer to the
    end. `edit_owned_public_message` may issue two requests for one call, so neither of them can
    be handed the caller's object as given. It is left rewound to where it was found instead.

    Args:
        file (File | None): The caller's upload, or None when this edit carries no attachment.

    Returns:
        A factory building a `File` with the same bytes, filename and description, or None when
        there was no file.
    """
    if file is None:
        return None
    file.reset()
    payload = file.fp.read()
    file.reset()
    filename = file.filename
    description = file.description

    def build_file() -> File:
        return File(fp=BytesIO(payload), filename=filename, description=description)

    return build_file


def _fresh_extra_files(file_factory: Callable[[], File] | None) -> list[File] | None:
    """Builds the `extra_files` list for exactly one Discord request.

    Called once per request rather than once per call, so the edit and the followup fallback
    never share a `File`.

    Args:
        file_factory (Callable[[], File] | None): The factory from `_fresh_file_factory`, or
            None when this edit carries no attachment.

    Returns:
        A one-item list holding a new `File`, or None when there is no attachment.
    """
    if file_factory is None:
        return None
    return [file_factory()]


async def edit_owned_public_message(
    interaction: Interaction[commands.Bot],
    embed: Embed,
    view: OwnedPublicView | None,
    file: File | None = None,
    message: Message | None = None,
) -> None:
    """Repaints the panel's message, falling back to a public followup when it is gone.

    Three paths in precedence order. An interaction that has not been answered yet is a button
    or select press, and `response.edit_message` acknowledges and repaints in one request. An
    already-answered one (a deferred press, a modal submit) edits the message object directly.
    With no target message at all, or when that edit 404s because the message has been deleted
    underneath the panel, the result is re-sent as a public followup rather than lost; only the
    404 has a stale cleanup row to drop on the way. That followup is the one path that creates a
    message nothing else knows about, so it is tracked for cleanup here.

    The view is re-bound as the target settles, so the idle timeout deletes the message the
    panel ended up on rather than the one it was aimed at. `response.edit_message` answers with
    None when it cannot resolve its own message, which is why only a real `Message` rebinds.
    Passing `view=None` removes the controls, which the edit paths honor and the followup
    expresses by omitting the argument — nextcord reads None there as a view and calls
    `is_finished()` on it.

    Every request goes through `embed_spacer_payload` so panel embeds keep one rendered width.
    The followup builds its own: a retained-attachment list is an edit-only parameter, and an
    edit that did run has already read the files of the first payload.

    Args:
        interaction (Interaction[commands.Bot]): The interaction being answered.
        embed (Embed): The panel's new embed, mutated in place by the spacer helper.
        view (OwnedPublicView | None): The controls to show, or None to remove them.
        file (File | None): One attachment to ride along, such as a board or chart PNG. Copied
            per request rather than sent as given.
        message (Message | None): The panel message to edit. Required for a modal submit, whose
            interaction is not attached to the message the modal was opened from; defaults to
            `interaction.message`.
    """
    target_message = message or interaction.message
    if view is not None:
        view.bind_message(message=target_message)
    file_factory = _fresh_file_factory(file=file)
    edit_spacer = embed_spacer_payload(
        embeds=[embed],
        is_edit=True,
        target=target_message or interaction,
        extra_files=_fresh_extra_files(file_factory=file_factory),
    )
    edit_files = cast("list[File]", edit_spacer.get("files", []))
    edit_attachments = cast("list[Attachment]", edit_spacer.get("attachments", []))
    if not interaction.response.is_done():
        edited = await interaction.response.edit_message(
            embed=embed, view=view, files=edit_files, attachments=edit_attachments
        )
        if isinstance(edited, Message) and view is not None:
            view.bind_message(message=edited)
        return
    if target_message is not None:
        try:
            await target_message.edit(
                embed=embed, view=view, files=edit_files, attachments=edit_attachments
            )
            return
        except NotFound:
            message_id = getattr(target_message, "id", None)
            if isinstance(message_id, int):
                await forget_public_message(message_id=message_id)
    followup_spacer = embed_spacer_payload(
        embeds=[embed],
        is_edit=False,
        target=interaction,
        extra_files=_fresh_extra_files(file_factory=file_factory),
    )
    followup_files = cast("list[File]", followup_spacer.get("files", []))
    if view is not None:
        sent_message = await interaction.followup.send(
            embed=embed, view=view, wait=True, files=followup_files
        )
    else:
        sent_message = await interaction.followup.send(
            embed=embed, wait=True, files=followup_files
        )
    if view is not None:
        view.bind_message(message=sent_message)
    user_name = getattr(interaction.user, "name", None)
    await track_public_message(message=sent_message, user_name=user_name)
