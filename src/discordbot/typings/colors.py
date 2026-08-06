"""Discord's own brand palette as embed colors, plus the aliases naming where the blurple goes.

This is the bot's shared theme vocabulary, not a catalogue of every hex in the tree. Only the
four colors the Discord client itself paints with live here, so an embed built from one reads
as part of the client rather than as one cog's taste. A per-cog accent stays a module constant
inside that cog (`economy/embeds.py`'s borrow / check-in / VIP hexes, `stock/presentation.py`'s
market / detail / news hexes) even where two cogs happen to have landed on the same value.

Consumers usually re-alias these under their own domain words, so each surface's meaning can be
re-tuned on its own: `games/presentation.py` as `WIN_COLOR` / `LOSE_COLOR` / `PUSH_COLOR`,
`economy/embeds.py` as `BALANCE_COLOR` / `LEADERBOARD_COLOR` / `ERROR_COLOR`, and so on. That is
a habit rather than a rule: `research/cog.py`, `gen_reply/cog.py` and `memory/views.py` each
pass a constant straight into `Embed(color=...)` where an alias would buy nothing. Neither list
is exhaustive, so grep the constant for the callers of the day.

`TRANSFER_COLOR` and `IN_PROGRESS_COLOR` are that same aliasing kept here rather than in the
cog. An economy transfer receipt and the Blackjack seat currently acting (plus every seat while
the insurance decision is open, dealer included; any other unsettled seat is yellow) share the
blurple today without sharing a meaning, so each is named to keep a retheme of one from moving
the other.
"""

from typing import Final

# Discord blurple, the neutral/info accent.
NEUTRAL_BLUE: Final[int] = 0x5865F2

# Discord's status palette; the trailing note is roughly what consumers use each one for.
DISCORD_RED: Final[int] = 0xED4245  # error / loss
DISCORD_GREEN: Final[int] = 0x57F287  # success / win / positive balance
DISCORD_YELLOW: Final[int] = 0xFEE75C  # warning / neutral / push / leaderboard

TRANSFER_COLOR: Final[int] = NEUTRAL_BLUE
IN_PROGRESS_COLOR: Final[int] = NEUTRAL_BLUE
