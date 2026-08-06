"""The casino tables' shared display vocabulary: the palette, the emoji, and the small line shapes.

Blackjack and 射龍門 are two separate Discord surfaces (`blackjack_views.py`,
`dragon_gate_views.py`) that a player meets in the same channel minutes apart, so what they have
in common is not the rules but the way an amount, a roster row or a settlement footnote is
written. That half lives here rather than in either view module: nothing forbids one table
importing from the other inside a cog, but the constant would then be owned by a surface that has
no say in how the other one looks, and the two drift the moment one is re-tuned. `cog.py` and
`history_text.py` read from here too, for the error accent, the dealer's display name and the
win / lose / push colors.

What lives here:

- The palette. `WIN_COLOR` / `LOSE_COLOR` / `PUSH_COLOR` / `ERROR_COLOR` are aliases over
  `typings/colors.py`'s Discord palette, following that module's convention, so the casino's
  sense of a win can be re-tuned without moving every other cog's greens.
- `SYSTEM_NARRATOR_NAME`, the dealer seat's display name. The casino is a label rather than a
  Discord identity and it has no narrator any more; the name is all that is left of it, and
  `cog.py`'s `_system_identity` stamps it onto every table.
- The emoji constants, split by role: a `*_FIELD_EMOJI` prefixes an embed field name or a
  heading, a `*_RESULT_EMOJI` prefixes an outcome label.
- The line builders. Each returns plain Markdown text and never an `Embed` — this module imports
  no nextcord at all — so a test can assert on the exact string a seat will show without building
  a Discord object. Numbers are formatted by `services/economy/presentation.py`'s `amount_code`,
  which is what makes a figure at a casino table read like every other balance in the bot.

Three exports are leftovers rather than the established spelling: `allin_note` and
`PlayerStatusKind` have no caller in the tree, and `blackjack_outcome_presentation` is only
re-exported through `blackjack_views.__all__`, never called.
"""

from typing import Final, Literal

from discordbot.typings.games import SettleOutcome
from discordbot.typings.colors import DISCORD_RED, DISCORD_GREEN, DISCORD_YELLOW
from discordbot.services.economy.presentation import amount_code

WIN_COLOR = DISCORD_GREEN
LOSE_COLOR = DISCORD_RED
PUSH_COLOR = DISCORD_YELLOW
ERROR_COLOR = DISCORD_RED

SYSTEM_NARRATOR_NAME: Final[str] = "賭場系統"

LOBBY_PLAYERS_FIELD_EMOJI = "👥"
POT_FIELD_EMOJI = "💰"
TURN_FIELD_EMOJI = "🎯"
LAST_HAND_FIELD_EMOJI = "⏮️"
FINISH_REASON_FIELD_EMOJI = "🏁"

WIN_RESULT_EMOJI = "🎉"
LOSE_RESULT_EMOJI = "😢"
BUST_RESULT_EMOJI = "💥"
DEALER_BUST_RESULT_EMOJI = "🎊"
NATURAL_RESULT_EMOJI = "✨"

PlayerStatusKind = Literal["blackjack", "bust", "active", "stand", "waiting"]


def blackjack_outcome_presentation(outcome: SettleOutcome) -> tuple[str, int]:
    """Returns the label and embed color for one settled Blackjack outcome.

    The table covers every `SettleOutcome` member, so a member added there without a row here
    raises `KeyError` at render time instead of quietly showing a blank result. Two pairings are
    deliberate: both 過五關 outcomes share one label, since what separates them is the bonus and
    not the win, and `surrender` takes the loss color even though only half the wager is gone.

    Args:
        outcome (SettleOutcome): Player-facing Blackjack outcome.

    Returns:
        A `(label, color)` tuple for the final embed.
    """
    blackjack_result = {
        "win": ("你贏了", WIN_COLOR),
        "lose": ("你輸了", LOSE_COLOR),
        "push": ("平手", PUSH_COLOR),
        "blackjack": ("Blackjack!", WIN_COLOR),
        "five_card_win": ("過五關", WIN_COLOR),
        "five_card_twenty_one": ("過五關", WIN_COLOR),
        "player_bust": ("你爆牌了", LOSE_COLOR),
        "dealer_bust": ("莊家爆牌, 你贏了", WIN_COLOR),
        "surrender": ("投降 · 退一半", LOSE_COLOR),
    }
    return blackjack_result[outcome]


def allin_note(is_allin: bool) -> str:
    """Returns the pipe-separated suffix marking a round that consumed the whole balance.

    Nothing calls this today, and the two lines that do mark an all-in round
    (`lobby_participant_line`, `settlement_metadata`) spell their own ` · all-in` instead, so this
    is not the wording currently on screen.

    Args:
        is_allin (bool): Whether the requested bet was clamped to the full balance.

    Returns:
        The suffix text, or an empty string for non all-in rounds.
    """
    return " | all-in" if is_allin else ""


def card_line(cards_text: str) -> str:
    """Renders a pre-rendered hand string as an H1 line with doubled inter-card spacing.

    Single-space `A♠ K♥` becomes `# A♠  K♥` so each card breathes a bit more inside the heading.
    An empty string is passed straight back, so a caller with nothing dealt yet short-circuits
    without producing a stray `#`.

    This carries no hand total on purpose: its one caller is the face-down dealer block, and
    printing the total there would give away the card the peek animation exists to reveal. The
    totalled twin for a face-up hand is `blackjack_views.py`'s own `_hand_summary_line`.

    Args:
        cards_text (str): Pre-rendered hand string (e.g. `"A♠ K♥"` or `"🂠 K♥"`).

    Returns:
        Markdown-ready H1 line for an embed description.
    """
    if not cards_text:
        return ""
    spaced = cards_text.replace(" ", "  ")
    return f"# {spaced}"


def metadata_line(text: str) -> str:
    """Wraps one already-formatted line in Discord's `-#` small-text markup.

    Args:
        text (str): The line body.

    Returns:
        The subtext line.
    """
    return f"-# {text}"


def lobby_participant_line(
    index: int, display_name: str, bet: int | None = None, is_allin: bool = False
) -> str:
    """Renders one lobby roster row, carrying the seat's stake only when there is one.

    A Blackjack seat commits its bet on joining and passes it; a 射龍門 seat bets per hand, so it
    passes None and the row is just the name. `is_allin` is therefore only ever visible alongside
    a `bet`.

    Args:
        index (int): 1-based position in the join order.
        display_name (str): Player display name.
        bet (int | None): Stake to append as inline code, or None to omit the segment entirely.
        is_allin (bool): Whether to mark the row with an `all-in` suffix.

    Returns:
        A single Markdown line for the lobby roster.
    """
    bet_suffix = ""
    if bet is not None:
        allin_suffix = " · all-in" if is_allin else ""
        bet_suffix = f" · 下注 {amount_code(amount=bet, compact=True)}{allin_suffix}"
    return f"**{index}. {display_name}**{bet_suffix}"


def settlement_metadata(  # noqa: PLR0913 -- final result metadata has several optional bonus facets
    delta: int,
    new_balance: int,
    is_allin: bool,
    base_delta: int | None = None,
    vip_bonus: int = 0,
    five_card_bonus: int = 0,
) -> str:
    """Renders the small-text settlement footnote under a settled seat.

    `delta` is the whole net change and already contains both bonuses, so the two bonus segments
    break out part of the figure ahead of them rather than adding to it. `base_delta` is read only
    as a presence check — a settlement carrying no bonus detail passes None and the VIP segment is
    dropped — and the pre-bonus figure itself is never shown. The segment order is fixed (本局,
    VIP加成, 過五關 bonus, all-in, 餘額) so every seat's footnote reads the same way.

    Args:
        delta (int): Player net point change for the round, both bonuses included.
        new_balance (int): Player balance after settlement.
        is_allin (bool): Whether the wager consumed the full balance.
        base_delta (int | None): Net change before the bonuses; None marks a settlement with no
            bonus detail and suppresses the VIP segment.
        vip_bonus (int): Part of `delta` contributed by the VIP payout bonus.
        five_card_bonus (int): Part of `delta` contributed by the system-funded five-card 21
            bonus.

    Returns:
        `-# 本局 +X · 餘額 Y` style metadata, with an `· all-in` suffix when the round was all-in.
    """
    segments = [f"本局 {amount_code(amount=delta, signed=True, compact=True)}"]
    if vip_bonus > 0 and base_delta is not None:
        segments.append(f"VIP加成 {amount_code(amount=vip_bonus, signed=True, compact=True)}")
    if five_card_bonus > 0:
        segments.append(
            f"過五關 bonus {amount_code(amount=five_card_bonus, signed=True, compact=True)}"
        )
    if is_allin:
        segments.append("all-in")
    segments.append(f"餘額 {amount_code(amount=new_balance, compact=True)}")
    return "-# " + " · ".join(segments)


def player_result_title(outcome: SettleOutcome, player_total: int, dealer_total: int) -> str:
    """Formats the H2 result line for one settled hand inside a player's seat embed.

    The H2 prefix is the whole difference from `player_result_inline`, whose only caller this is.

    Args:
        outcome (SettleOutcome): Player-facing Blackjack outcome label.
        player_total (int): Final player hand total.
        dealer_total (int): Final dealer hand total.

    Returns:
        Markdown H2 line such as `## 🎉 你贏了 · 20 > 19`.
    """
    return f"## {player_result_inline(outcome=outcome, player_total=player_total, dealer_total=dealer_total)}"


def player_result_inline(outcome: SettleOutcome, player_total: int, dealer_total: int) -> str:  # noqa: PLR0911 -- one branch per SettleOutcome label keeps the mapping obvious
    """Formats one settled hand's result as a single line, with no heading prefix.

    Which total the line shows is per outcome rather than uniform: `dealer_bust` names the
    dealer's total because that is the number that decided the hand, `surrender` names neither
    since the hand never played out, and only `win` / `lose` / the push tail show both. `push` is
    that unmatched tail rather than a branch of its own, so an outcome added to `SettleOutcome`
    without a branch here silently renders as a push.

    Args:
        outcome (SettleOutcome): Player-facing Blackjack outcome label.
        player_total (int): Final player hand total.
        dealer_total (int): Final dealer hand total.

    Returns:
        The label line, such as `🎉 你贏了 · 20 > 19`.
    """
    if outcome == "blackjack":
        return f"{NATURAL_RESULT_EMOJI} Blackjack · {player_total}"
    if outcome == "five_card_twenty_one":
        return f"{NATURAL_RESULT_EMOJI} 過五關 · {player_total}"
    if outcome == "five_card_win":
        return f"{WIN_RESULT_EMOJI} 過五關 · {player_total}"
    if outcome == "dealer_bust":
        return f"{DEALER_BUST_RESULT_EMOJI} 莊家爆牌, 你贏了 · {dealer_total}"
    if outcome == "player_bust":
        return f"{BUST_RESULT_EMOJI} 你爆牌了 · {player_total}"
    if outcome == "win":
        return f"{WIN_RESULT_EMOJI} 你贏了 · {player_total} > {dealer_total}"
    if outcome == "lose":
        return f"{LOSE_RESULT_EMOJI} 你輸了 · {player_total} < {dealer_total}"
    if outcome == "surrender":
        return "🏳️ 投降 · 退一半"
    return f"平手 · {player_total} = {dealer_total}"
