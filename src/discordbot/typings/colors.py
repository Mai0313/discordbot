"""Shared Discord embed color constants.

The help, economy transfer, and in-progress game embeds all use the same
neutral blurple, and the economy / stock / games embeds all use Discord's
red/green/yellow status palette. Defining each hex once keeps the theme
consistent; the semantic aliases document where each color is used.
"""

from typing import Final

# Discord blurple, used wherever a neutral/info accent is wanted.
NEUTRAL_BLUE: Final[int] = 0x5865F2

# Discord's status palette, reused across economy / stock / games embeds.
DISCORD_RED: Final[int] = 0xED4245  # error / loss
DISCORD_GREEN: Final[int] = 0x57F287  # success / win / positive balance
DISCORD_YELLOW: Final[int] = 0xFEE75C  # neutral / push / leaderboard

HELP_COLOR: Final[int] = NEUTRAL_BLUE
TRANSFER_COLOR: Final[int] = NEUTRAL_BLUE
IN_PROGRESS_COLOR: Final[int] = NEUTRAL_BLUE
