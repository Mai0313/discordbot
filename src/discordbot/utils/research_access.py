"""Channel eligibility and permission checks required before starting a deep-research run."""

from typing import TYPE_CHECKING, Literal

from nextcord import Permissions, TextChannel

if TYPE_CHECKING:
    from nextcord import Guild, Member

type ResearchAccessDenial = Literal["unsupported", "forbidden"]


def has_research_permissions(channel: TextChannel, member: "Member") -> bool:
    """Returns whether `member` holds all channel permissions required for a deep-research run.

    Args:
        channel (TextChannel): Guild text channel where the research thread would start.
        member (Member): The bot's guild member in `channel`.
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


def research_access_denial(channel: object, guild: "Guild | None") -> ResearchAccessDenial | None:
    """Returns why a deep-research thread cannot start in `channel`, or None when allowed.

    Returns `"unsupported"` outside a guild `TextChannel` (DMs, threads, or unjoined servers)
    and `"forbidden"` when the bot lacks a required channel permission in `channel`.

    Args:
        channel (object): Discord channel where the research thread would start.
        guild (Guild | None): Guild resolved on the triggering message or interaction.
    """
    if guild is None or not isinstance(channel, TextChannel):
        return "unsupported"
    if not has_research_permissions(channel=channel, member=channel.guild.me):
        return "forbidden"
    return None
