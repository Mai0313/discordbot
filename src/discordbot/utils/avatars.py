"""Picks the avatar URL to store for a Discord identity, server profile before global.

Discord lets a member override their avatar per server. `Member.guild_avatar` is that
override, while `display_avatar` is whatever Discord actually shows: already guild-aware on a
`Member`, global on a plain `User`, which is why the helper still has to resolve a `Member`
when it is handed a `User`. `guild_avatar_url` is the single place that ladder is written
down, so the write paths that cache `UserAccount.avatar_url` (the per-message reward in
`cli.py`, the economy commands and views, the fishing shop and cast, the games participant
identities, the stock panel) store the face a channel actually shows instead of re-deriving
the rule per cog.

The helper never raises on a Discord-side failure: a member who has left, a fetch Discord
refuses, and a guild-shaped object that resolves no members all degrade to the global avatar.
Only `HTTPException` is suppressed, so a connection or timeout failure from the fetch still
propagates. Degrading is worth it because a global avatar costs less than a failed command or
a lost reward, and most callers do nothing with the URL but pass it into the call that stores
it. It promises nothing about persistence or freshness: it caches nothing and writes nothing,
so a stored URL only changes when a caller calls again and stores the result.

It sits in `utils/` rather than beside the ledger it feeds because the cogs that need it may
not import each other and it holds no domain state of its own. `AvatarUser` is a structural
protocol for the same reason: `User`, `Member` and the bot's own `ClientUser` satisfy it
without any caller having to agree on a shared type, while the body still narrows to `Member`
when it can.
"""

from typing import Protocol
import contextlib

from nextcord import Asset, Guild, Member, HTTPException


class AvatarUser(Protocol):
    """Discord user-like object with enough identity for avatar lookup."""

    # Read-only properties: nextcord's User/Member expose these as properties,
    # which cannot satisfy a mutable protocol attribute.
    @property
    def id(self) -> int:
        """The Discord snowflake, which is what a guild member lookup is keyed on."""
        ...

    @property
    def display_avatar(self) -> Asset:
        """The avatar Discord already shows for this user, used as the fallback."""
        ...


async def guild_avatar_url(user: AvatarUser, guild: Guild | None = None) -> str:
    """Returns the avatar URL this user shows in `guild`, falling back to their global one.

    Costs at most one REST call, and usually none: a `Member` argument answers from itself, a
    cached one from the guild's member cache. The bot runs without the `members` intent
    (`cli.py`), so that cache only holds who this process has already seen and the
    `fetch_member` miss path is ordinary rather than exotic. Every `HTTPException` that fetch
    can raise is swallowed (a member who left, a user Discord will not hand over, a transient
    5xx), because a global avatar is cheaper than the command or reward it would otherwise
    cost; a connection or timeout failure is not an `HTTPException` and still propagates.

    The member lookups are attribute-guarded, so a guild-shaped object that resolves no members
    answers with the global avatar instead of raising.

    Args:
        user (AvatarUser): Whose avatar to resolve. One that is already a `Member` carries its
            own server profile, so no lookup runs and `guild` is ignored.
        guild (Guild | None): Guild whose per-server profile to prefer. None resolves the
            global avatar unless `user` is already a `Member`.

    Returns:
        The guild avatar URL when the member has one, otherwise the global display avatar URL.
    """
    fallback_url = user.display_avatar.url
    member: Member | None = None
    if isinstance(user, Member):
        member = user
    elif guild is not None and hasattr(guild, "get_member"):
        member = guild.get_member(user.id)
        if member is None and hasattr(guild, "fetch_member"):
            with contextlib.suppress(HTTPException):
                member = await guild.fetch_member(user.id)

    if member is None:
        return fallback_url
    if member.guild_avatar is not None:
        return member.guild_avatar.url
    return fallback_url or member.display_avatar.url
