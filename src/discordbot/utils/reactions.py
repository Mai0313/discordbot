"""Status reactions on a user's message, best-effort, for the cogs that report progress with them.

`update_reaction` is the whole contract: remove the bot's previous one, add the new emoji, and let
neither failure escape. A status reaction is decoration on top of a deliverable, so a reaction the
API refused must never abort the expansion, download or reply it was reporting on.

Two shapes ride that one call. A `previous` plus a `bot_user` swaps one status emoji for the next,
which is the one-at-a-time chain `parse_threads`, `parse_douyin` and `gen_reply` walk. Leaving
`previous` out adds a reaction nothing here will ever take off again, which is how
`gen_reply/streaming.py` hints a dropped voice clip, image, music or video and how `research` marks
a message it took over. That additive shape is part of the contract rather than a degenerate swap:
those hints have to outlive the status chain moving on.

What this deliberately does not do: it never reads the message's reactions back from Discord, and
it never reports whether either call was accepted. The caller owns the current emoji and hands it
back as `previous` on the next call, which is what keeps the helper stateless and why the return
value is the emoji rather than a success flag. Removal is scoped to `bot_user`, so a human who
reacted with the same emoji keeps their reaction.

`ReactionStatusChain` is the fire-and-forget variant `gen_reply` needs: the reply pipeline steps
through several stage emojis while it works, and awaiting a REST round trip per stage would put
Discord's reaction endpoint on the reply's critical path.

This lives in `utils/` because four unrelated cogs report progress the same way (`gen_reply`,
`parse_threads`, `parse_douyin`, `research`) and none of them may import a peer cog to reach it.
"""

import asyncio
import contextlib

from nextcord import Message, ClientUser
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation


async def update_reaction(
    message: Message, bot_user: ClientUser | None, emoji: str, previous: str | None = None
) -> str:
    """Adds a status reaction to a message, replacing the bot's previous one.

    Both calls are best-effort and every `Exception` is suppressed, including a deleted message or
    a missing `previous` reaction, so reaction bookkeeping never breaks the flow it reports on.
    `CancelledError` is a `BaseException` and still propagates out of either await, which is what
    lets a cancelled `ReactionStatusChain` step stop instead of finishing its swap. Removal runs
    first, so a swap shows no status reaction for a moment rather than two.

    Args:
        message (Message): The message to react on.
        bot_user (ClientUser | None): The bot's own user, which scopes the removal of `previous`
            to the bot's own reaction; a None user skips the removal entirely.
        emoji (str): The reaction to add.
        previous (str | None): The bot's prior status reaction to remove first, if any.

    Returns:
        `emoji`, unconditionally rather than on success, for the caller to pass back as
        `previous` on its next call.
    """
    if previous and bot_user:
        with contextlib.suppress(Exception):
            await message.remove_reaction(emoji=previous, member=bot_user)
    with contextlib.suppress(Exception):
        await message.add_reaction(emoji=emoji)
    return emoji


class ReactionStatusChain(BaseModel):
    """Schedules ordered, best-effort status reactions without blocking the caller.

    Each `advance` starts a background task that awaits the previously scheduled one before
    swapping the emoji, so what the user sees follows the schedule order even though the pipeline
    never awaits a reaction REST call. Nothing outside the chain holds those tasks: the newest
    lives in `_tail` and every pending step holds its predecessor, which is what keeps them from
    being collected mid-flight.

    `current_emoji` advances at schedule time and `update_reaction` returns its emoji whether or
    not Discord accepted the add, so the next step's removal may target an emoji Discord never
    showed, which removal already tolerates. A step cannot fail from a reaction failure, since
    `update_reaction` suppresses those internally, so the guard around awaiting the predecessor is
    defensive only: a cancelled predecessor raises `CancelledError`, which it does not catch.
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
        """Schedules `emoji` to replace the previously scheduled status reaction.

        Returns as soon as the task exists, so `current_emoji` tracks the schedule rather than
        what Discord has shown; `flush` is what waits for the swap to land.

        Args:
            emoji (str): The status reaction to show next.
        """
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
        """Waits for the last scheduled reaction update to finish.

        Awaiting the tail is enough, since every step awaits its predecessor. Its failure is
        suppressed so the usual `finally: await flush()` cannot displace the exception already
        unwinding through it.
        """
        if self._tail is None:
            return
        with contextlib.suppress(Exception):
            await self._tail
