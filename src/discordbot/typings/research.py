"""The channel permissions a deep-research run needs from the bot.

Shared by every entry point that can start a run, which may not import each other; each reads them
off the cached channel for the bot's own member.
"""

from typing import TYPE_CHECKING

from nextcord import Permissions

if TYPE_CHECKING:
    from nextcord import Member, TextChannel


def has_research_permissions(channel: "TextChannel", member: "Member") -> bool:
    """Whether `member`, the bot's own, holds everything a research run needs in `channel`."""
    # See the channel, post the anchor, open a public thread from it, write into that thread, and
    # attach the `research.md` the last report message always carries.
    needed = Permissions(
        view_channel=True,
        send_messages=True,
        create_public_threads=True,
        send_messages_in_threads=True,
        attach_files=True,
    )
    return channel.permissions_for(member) >= needed
