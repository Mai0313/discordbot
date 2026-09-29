"""Discord's own semantic embed palette: the neutral blurple and the red/green/yellow status set.

An embed that means one of these things takes its hex from here, so a status reads the same
wherever it appears. A color that means something else, such as a platform's brand or one
feature's own accent, stays beside the embed that wears it.
"""

from typing import Final

# Discord blurple, used wherever a neutral/info accent is wanted.
NEUTRAL_BLUE: Final[int] = 0x5865F2

# Discord's status palette.
DISCORD_RED: Final[int] = 0xED4245  # error / loss
DISCORD_GREEN: Final[int] = 0x57F287  # success / win / positive balance
DISCORD_YELLOW: Final[int] = 0xFEE75C  # neutral / push / leaderboard

IN_PROGRESS_COLOR: Final[int] = NEUTRAL_BLUE
