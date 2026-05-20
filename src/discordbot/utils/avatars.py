"""Discord avatar selection helpers."""

from typing import Protocol
import contextlib

import nextcord


class AvatarUser(Protocol):
    """Discord user-like object with enough identity for avatar lookup."""

    id: int
    display_avatar: nextcord.Asset


def _member_avatar_url(member: nextcord.Member, fallback_url: str) -> str:
    """Returns the member's guild avatar URL, falling back to a global avatar."""
    if member.guild_avatar is not None:
        return member.guild_avatar.url
    return fallback_url or member.display_avatar.url


async def guild_avatar_url(user: AvatarUser, guild: nextcord.Guild | None = None) -> str:
    """Returns a guild-scoped avatar URL when Discord exposes one.

    Args:
        user: Discord user or member whose avatar should be stored.
        guild: Guild context used to resolve a member when only a global user is available.

    Returns:
        The member's guild avatar URL when available, otherwise the user's global display avatar.
    """
    fallback_url = user.display_avatar.url
    member: nextcord.Member | None = None
    if isinstance(user, nextcord.Member):
        member = user
    elif guild is not None and hasattr(guild, "get_member"):
        member = guild.get_member(user.id)
        if member is None and hasattr(guild, "fetch_member"):
            with contextlib.suppress(nextcord.HTTPException):
                member = await guild.fetch_member(user.id)

    if member is None:
        return fallback_url
    return _member_avatar_url(member=member, fallback_url=fallback_url)
