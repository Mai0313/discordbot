"""`nextcord.ui.View` and `Modal` subclasses that log callback, submit, and timeout failures.

`nextcord` routes view/modal errors to `on_error` (defaulting to `sys.stderr`) and runs
`on_timeout` as an unawaited task; both bypass `DiscordBot.on_error` and `./data/logs`.
"""

from typing import Self, ClassVar
from functools import wraps
from collections.abc import Callable, Coroutine

import logfire
from nextcord import Interaction
from nextcord.ui import Item, View, Modal
from nextcord.ext import commands


class LoggedView(View):
    """A view whose failing control or timeout is logged; a subclass may name its control line."""

    interaction_failure_log: ClassVar[str] = "View control failed"

    def __init_subclass__(cls) -> None:
        """Wraps the subclass's own `on_timeout` so a failure in it is logged."""
        super().__init_subclass__()
        if "on_timeout" in vars(cls):
            cls.on_timeout = _logging_timeout(on_timeout=cls.on_timeout)

    async def on_error(
        self, error: Exception, item: Item[Self], interaction: Interaction[commands.Bot]
    ) -> None:
        """Logs a control's failure instead of letting nextcord only print it to stderr.

        Args:
            error (Exception): Exception raised by the control or `interaction_check`.
            item (Item[Self]): Control that raised `error`.
            interaction (Interaction[commands.Bot]): Interaction that triggered the control.
        """
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


def _logging_timeout(
    on_timeout: Callable[[LoggedView], Coroutine[None, None, None]],
) -> Callable[[LoggedView], Coroutine[None, None, None]]:
    """Returns `on_timeout` logging whatever escapes it instead of raising it."""

    @wraps(wrapped=on_timeout)
    async def logged(view: LoggedView) -> None:
        # Broad on purpose: nothing awaits the task nextcord runs this in, so whatever escapes
        # here would reach only asyncio's stderr print.
        try:
            await on_timeout(view)
        except Exception as error:
            logfire.error(
                "View timeout failed",
                view=type(view).__name__,
                message_id=getattr(getattr(view, "message", None), "id", None),
                error_type=type(error).__name__,
                _exc_info=error,
            )

    return logged


class LoggedModal(Modal):
    """A modal whose failing submit is logged."""

    async def on_error(self, error: Exception, interaction: Interaction[commands.Bot]) -> None:
        """Logs a submit's failure instead of letting nextcord only print it to stderr.

        Args:
            error (Exception): Exception raised by `callback`.
            interaction (Interaction[commands.Bot]): Interaction that submitted the modal.
        """
        logfire.error(
            "Modal submit failed",
            modal=type(self).__name__,
            user_id=getattr(interaction.user, "id", None),
            message_id=getattr(interaction.message, "id", None),
            error_type=type(error).__name__,
            _exc_info=error,
        )
