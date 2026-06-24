"""Shared helper for Discord's per-message upload-size ceiling."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nextcord import Guild

# Discord lowered the non-Nitro base upload limit to 10 MiB in 2024; a guild-less context
# (DM) has no boost-tier table to consult, so it falls back to this base.
DEFAULT_NON_NITRO_UPLOAD_LIMIT = 10 * 1024 * 1024


def upload_limit_for(guild: "Guild | None") -> int:
    """Returns the destination's real attachment upload ceiling in bytes.

    A boosted guild's 50/100 MiB is honored via nextcord's `filesize_limit` (its boost-tier
    table lookup keyed on `premium_tier`); a DM has no guild to query, so it falls back to
    Discord's non-Nitro base of 10 MiB.

    Args:
        guild: The destination guild, or None for a DM.

    Returns:
        The maximum attachment size in bytes for that destination.
    """
    return guild.filesize_limit if guild is not None else DEFAULT_NON_NITRO_UPLOAD_LIMIT


__all__ = ["DEFAULT_NON_NITRO_UPLOAD_LIMIT", "upload_limit_for"]
