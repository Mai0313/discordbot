"""Discord avatar selection helpers."""

from typing import Protocol
import contextlib

from nextcord import Asset, Guild, Member, HTTPException


class AvatarUser(Protocol):
    """Discord user-like object with enough identity for avatar lookup."""

    # Read-only properties: nextcord's User/Member expose these as properties,
    # which cannot satisfy a mutable protocol attribute.
    @property
    def id(self) -> int: ...

    @property
    def display_avatar(self) -> Asset: ...


async def guild_avatar_url(user: AvatarUser, guild: Guild | None = None) -> str:
    """Returns a guild-scoped avatar URL when Discord exposes one.

    Args:
        user: Discord user or member whose avatar should be stored.
        guild: Guild context used to resolve a member when only a global user is available.

    Returns:
        The member's guild avatar URL when available, otherwise the user's global display avatar.
    """
    member: Member | None = None
    if isinstance(user, Member):
        member = user
    elif guild is not None:
        member = guild.get_member(user.id)
        if member is None:
            with contextlib.suppress(HTTPException):
                member = await guild.fetch_member(user.id)
    # A Member's display avatar is its guild avatar when it has one.
    return (user if member is None else member).display_avatar.url
