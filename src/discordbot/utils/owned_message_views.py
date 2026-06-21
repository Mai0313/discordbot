"""Shared scaffolding for views that own one public Discord message.

Stock and fishing panels share the same UX: one public message edited in place,
operable only by the user who opened it, and deleted after an idle timeout. The
base view and the central edit helper live here so each cog only supplies its
own embeds, controls, and notice text.
"""

from io import BytesIO
from collections.abc import Callable

import logfire
from nextcord import File, Embed, Message, NotFound, Interaction
from nextcord.ui import View

from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import (
    track_public_message,
    delete_public_message,
    forget_public_message,
)


async def send_ephemeral_notice(interaction: Interaction, content: str, log_message: str) -> None:
    """Sends an ephemeral interaction notice with response/followup fallback."""
    try:
        if interaction.response.is_done():
            await interaction.followup.send(content=content, ephemeral=True)
            return
        await interaction.response.send_message(content=content, ephemeral=True)
    except Exception:
        logfire.warn(log_message, _exc_info=True)


class OwnedPublicView(View):
    """Base view for panels that own one public Discord message."""

    def __init__(
        self,
        owner_id: int,
        timeout_seconds: float,
        owner_mismatch_notice: str,
        delete_on_timeout: bool = True,
    ) -> None:
        """Initializes the owned-panel controls with an idle timeout."""
        super().__init__(timeout=timeout_seconds)
        self.owner_id = owner_id
        self.owner_mismatch_notice = owner_mismatch_notice
        self.delete_on_timeout = delete_on_timeout
        self.message: Message | None = None

    def bind_message(self, message: Message | None) -> None:
        """Records the message this view should update or delete."""
        self.message = message

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Allows only the user who opened this panel to operate it."""
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
        """Deletes the tracked public message after the idle timeout."""
        if self.message is None or not self.delete_on_timeout:
            return
        await delete_public_message(message=self.message)


def _fresh_file_factory(file: File | None) -> Callable[[], File] | None:
    """Returns a factory that creates fresh uploads for retry or fallback paths."""
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
    """Builds a fresh extra file list for one Discord request."""
    if file_factory is None:
        return None
    return [file_factory()]


async def edit_owned_public_message(
    interaction: Interaction,
    embed: Embed,
    view: OwnedPublicView | None,
    file: File | None = None,
    message: Message | None = None,
) -> None:
    """Edits the panel's current message for a component or modal interaction."""
    target_message = message or interaction.message
    if view is not None:
        view.bind_message(message=target_message)
    file_factory = _fresh_file_factory(file=file)
    kwargs: dict[str, object] = {
        "embed": embed,
        "view": view,
        **embed_spacer_payload(
            embeds=[embed],
            is_edit=True,
            target=target_message or interaction,
            extra_files=_fresh_extra_files(file_factory=file_factory),
        ),
    }
    if not interaction.response.is_done():
        edited = await interaction.response.edit_message(**kwargs)
        if isinstance(edited, Message) and view is not None:
            view.bind_message(message=edited)
        return
    if target_message is not None:
        try:
            await target_message.edit(**kwargs)
            return
        except NotFound:
            message_id = getattr(target_message, "id", None)
            if isinstance(message_id, int):
                await forget_public_message(message_id=message_id)
    followup_kwargs: dict[str, object] = {
        "embed": embed,
        "view": view,
        "wait": True,
        **embed_spacer_payload(
            embeds=[embed],
            is_edit=False,
            target=interaction,
            extra_files=_fresh_extra_files(file_factory=file_factory),
        ),
    }
    sent_message = await interaction.followup.send(**followup_kwargs)
    if view is not None:
        view.bind_message(message=sent_message)
    user_name = getattr(interaction.user, "name", None)
    await track_public_message(message=sent_message, user_name=user_name)
