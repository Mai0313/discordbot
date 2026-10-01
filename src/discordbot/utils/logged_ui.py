"""A view and a modal whose failing callbacks land in `./data/logs`.

nextcord hands an exception raised by a control's callback, a view's `interaction_check` or a
modal's submit to that view's or modal's own `on_error`, never to `DiscordBot.on_error`, and the
default only prints it to `sys.stderr`, which `./data/logs` does not tee.
"""

from typing import Self, ClassVar

import logfire
from nextcord import Interaction
from nextcord.ui import Item, View, Modal
from nextcord.ext import commands


class LoggedView(View):
    """A view whose failing control is logged; a subclass may name its own line."""

    interaction_failure_log: ClassVar[str] = "View control failed"

    async def on_error(
        self, error: Exception, item: Item[Self], interaction: Interaction[commands.Bot]
    ) -> None:
        """Logs a control's failure instead of letting nextcord only print it to stderr."""
        logfire.error(
            self.interaction_failure_log,
            view=type(self).__name__,
            custom_id=getattr(item, "custom_id", None),
            item_label=getattr(item, "label", None),
            user_id=getattr(interaction.user, "id", None),
            message_id=getattr(interaction.message, "id", None),
            error_type=type(error).__name__,
            _exc_info=error,
        )


class LoggedModal(Modal):
    """A modal whose failing submit is logged."""

    async def on_error(self, error: Exception, interaction: Interaction[commands.Bot]) -> None:
        """Logs a submit's failure instead of letting nextcord only print it to stderr."""
        logfire.error(
            "Modal submit failed",
            modal=type(self).__name__,
            user_id=getattr(interaction.user, "id", None),
            message_id=getattr(interaction.message, "id", None),
            error_type=type(error).__name__,
            _exc_info=error,
        )
