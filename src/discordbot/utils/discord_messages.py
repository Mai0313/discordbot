"""Small Discord message operation helpers."""

from typing import Protocol
import contextlib

from nextcord import ClientUser, Message


class BotUserProvider(Protocol):
    """Bot-like object that exposes the current Discord client user."""

    user: ClientUser | None


class DiscordMessageOperations:
    """Reusable operations for mutating Discord messages."""

    def __init__(self, bot: BotUserProvider) -> None:
        """Initializes the operation helper with the owning bot."""
        self.bot = bot

    async def set_status_reaction(
        self, message: Message, emoji: str, previous: str | None = None
    ) -> str:
        """Adds the current status reaction and removes the previous one."""
        if previous and self.bot.user:
            with contextlib.suppress(Exception):
                await message.remove_reaction(emoji=previous, member=self.bot.user)
        with contextlib.suppress(Exception):
            await message.add_reaction(emoji=emoji)
        return emoji
