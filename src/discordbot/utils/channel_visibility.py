"""Whether a guild channel's content may enter the server-wide memory any member can read."""

from nextcord import Guild


def channel_is_public(guild: Guild | None, channel: object) -> bool:
    """Whether @everyone can view `channel`, so its content is not private.

    `channel` is whatever messageable a message or an interaction carries, so visibility is read
    defensively (mirrors `utils.discord_embeds`): a private thread is never public; a thread
    otherwise inherits its parent channel's `@everyone` visibility; a regular guild channel uses
    its own. No guild, or any channel whose permissions cannot be resolved, counts as non-public
    — so content from channels members cannot see never enters the server-wide memory any member
    can read via `/memory server show`.
    """
    if guild is None:
        return False
    is_private = getattr(channel, "is_private", None)
    if callable(is_private) and is_private():
        return False
    source = getattr(channel, "parent", None) or channel
    permissions_for = getattr(source, "permissions_for", None)
    if not callable(permissions_for):
        return False
    return bool(getattr(permissions_for(guild.default_role), "view_channel", False))
