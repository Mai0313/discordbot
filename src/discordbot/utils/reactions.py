"""Shared helper for status reactions on a Discord message."""

import contextlib

from nextcord import Message, ClientUser


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
