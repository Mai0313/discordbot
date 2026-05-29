"""Shared Discord embed color constants.

The help, economy transfer, and in-progress game embeds all use the same
neutral blurple. Defining it once keeps the theme consistent; the semantic
aliases document where each color is used.
"""

from typing import Final

# Discord blurple, used wherever a neutral/info accent is wanted.
NEUTRAL_BLUE: Final[int] = 0x5865F2

HELP_COLOR: Final[int] = NEUTRAL_BLUE
TRANSFER_COLOR: Final[int] = NEUTRAL_BLUE
IN_PROGRESS_COLOR: Final[int] = NEUTRAL_BLUE
