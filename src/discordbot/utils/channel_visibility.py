"""Whether a guild channel's content may enter the server-wide memory any member can read."""

from nextcord import Guild


def channel_is_public(guild: Guild | None, channel: object) -> bool:
    """Returns whether `@everyone` can view `channel`.

    A private thread is never public; any other thread inherits its parent channel's `@everyone`
    visibility; a regular guild channel uses its own. Missing guild context or unresolvable
    permissions count as non-public.

    Args:
        guild (Guild | None): Guild that owns `channel`, or None outside a guild.
        channel (object): Channel or thread carried by a message or interaction.
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
