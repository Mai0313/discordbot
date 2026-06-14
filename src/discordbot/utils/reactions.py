"""Shared helpers for status reactions on a Discord message."""

import asyncio
import contextlib

from nextcord import Message, ClientUser
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation


async def update_reaction(
    message: Message, bot_user: ClientUser | None, emoji: str, previous: str | None = None
) -> str:
    """Adds a status reaction to a message, replacing the bot's previous one.

    Both add and remove are best-effort; transient API failures are suppressed
    so reaction bookkeeping never breaks the surrounding flow.

    Args:
        message: The message to react on.
        bot_user: The bot's own user, used to scope the removal of `previous`.
        emoji: The reaction to add.
        previous: The bot's prior reaction to remove first, if any.

    Returns:
        The emoji that was added, so callers can track the current reaction.
    """
    if previous and bot_user:
        with contextlib.suppress(Exception):
            await message.remove_reaction(emoji=previous, member=bot_user)
    with contextlib.suppress(Exception):
        await message.add_reaction(emoji=emoji)
    return emoji


class ReactionStatusChain(BaseModel):
    """Schedules ordered, best-effort status reactions without blocking the caller.

    Each `advance` chains a background task that waits for the previous update
    before swapping in the new emoji, so the visible order matches the schedule
    order while the reply pipeline never waits on Discord reaction REST calls.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message] = Field(
        ..., description="The user message receiving status reactions."
    )
    bot_user: SkipValidation[ClientUser | None] = Field(
        ..., description="The bot user that scopes removal of the previous reaction."
    )
    current_emoji: str | None = Field(
        default=None, description="The most recently scheduled status emoji."
    )
    _tail: asyncio.Task[str] | None = PrivateAttr(default=None)

    def advance(self, emoji: str) -> None:
        """Schedules `emoji` to replace the previously scheduled status reaction."""
        previous_task = self._tail
        previous_emoji = self.current_emoji

        async def _step() -> str:
            if previous_task is not None:
                with contextlib.suppress(Exception):
                    await previous_task
            return await update_reaction(
                message=self.message, bot_user=self.bot_user, emoji=emoji, previous=previous_emoji
            )

        self.current_emoji = emoji
        self._tail = asyncio.create_task(coro=_step())

    async def flush(self) -> None:
        """Waits for the last scheduled reaction update to finish."""
        if self._tail is None:
            return
        with contextlib.suppress(Exception):
            await self._tail
