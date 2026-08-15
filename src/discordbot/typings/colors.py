"""Shared Discord embed color constants.

Every cog that renders an embed draws its accents from here: the same neutral
blurple, and Discord's own red/green/yellow status palette. Defining each hex
once keeps the theme consistent; the semantic aliases document what each color
means rather than where it is used.
"""

from typing import Final

# Discord blurple, used wherever a neutral/info accent is wanted.
NEUTRAL_BLUE: Final[int] = 0x5865F2

# Discord's status palette, reused across every embed-rendering cog.
DISCORD_RED: Final[int] = 0xED4245  # error / loss
DISCORD_GREEN: Final[int] = 0x57F287  # success / win / positive balance
DISCORD_YELLOW: Final[int] = 0xFEE75C  # neutral / push / leaderboard

# Discord's muted grey, for a state that is neither good nor bad (a report the
# developer decided not to act on).
NEUTRAL_GREY: Final[int] = 0x99AAB5

TRANSFER_COLOR: Final[int] = NEUTRAL_BLUE
IN_PROGRESS_COLOR: Final[int] = NEUTRAL_BLUE
