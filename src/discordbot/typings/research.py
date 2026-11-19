"""Channel permission check required before starting a deep-research run."""

from typing import TYPE_CHECKING

from nextcord import Permissions

if TYPE_CHECKING:
    from nextcord import Member, TextChannel


def has_research_permissions(channel: "TextChannel", member: "Member") -> bool:
    """Returns whether `member` holds all channel permissions required for a deep-research run.

    Args:
        channel ("TextChannel"): Guild text channel where the research thread would start.
        member ("Member"): The bot's guild member in `channel`.
    """
    # `attach_files` is required for the `research.md` file attached to the final report.
    needed = Permissions(
        view_channel=True,
        send_messages=True,
        create_public_threads=True,
        send_messages_in_threads=True,
        attach_files=True,
    )
    return channel.permissions_for(member) >= needed
