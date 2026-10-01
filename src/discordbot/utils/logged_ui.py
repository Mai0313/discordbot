"""A view and a modal whose failures land in `./data/logs`.

nextcord hands an exception raised by a control's callback, a view's `interaction_check` or a
modal's submit to that view's or modal's own `on_error`, never to `DiscordBot.on_error`, and the
default only prints it to `sys.stderr`, which `./data/logs` does not tee. A view's `on_timeout`
reaches neither: nextcord runs it as a task nothing awaits, so asyncio prints its failure there.
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
        """Logs a submit's failure instead of letting nextcord only print it to stderr."""
        logfire.error(
            "Modal submit failed",
            modal=type(self).__name__,
            user_id=getattr(interaction.user, "id", None),
            message_id=getattr(interaction.message, "id", None),
            error_type=type(error).__name__,
            _exc_info=error,
        )
