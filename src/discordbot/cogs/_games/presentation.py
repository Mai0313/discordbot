"""Shared presentation helpers for casino game embeds."""

from typing import Final, Literal

from discordbot.typings.games import SettleOutcome
from discordbot.typings.colors import DISCORD_RED, DISCORD_GREEN, DISCORD_YELLOW
from discordbot.cogs._economy.presentation import amount_code

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
    """Returns presentation values for a Blackjack outcome.

    Args:
        outcome: Player-facing Blackjack outcome.

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
    """Returns the shared suffix for auto all-in rounds.

    Args:
        is_allin: Whether the requested bet was clamped to the full balance.

    Returns:
        The suffix text, or an empty string for non all-in rounds.
    """
    return " | all-in" if is_allin else ""


def card_line(cards_text: str) -> str:
    """Renders a hand string as an H1 line with doubled inter-card spacing.

    Single-space `A♠ K♥` becomes `# A♠  K♥` so each card breathes a bit
    more inside the heading. Empty strings are left alone so callers can
    short-circuit without producing a stray `#`.

    Args:
        cards_text: Pre-rendered hand string (e.g. `"A♠ K♥"` or
            `"🂠 K♥"`).

    Returns:
        Markdown-ready H1 line for placement in an embed field value.
    """
    if not cards_text:
        return ""
    spaced = cards_text.replace(" ", "  ")
    return f"# {spaced}"


def metadata_line(text: str) -> str:
    """Formats a `-#` small text metadata line."""
    return f"-# {text}"


def lobby_participant_line(
    index: int, display_name: str, bet: int | None = None, is_allin: bool = False
) -> str:
    """Renders one lobby participant row with optional bet metadata.

    Args:
        index: 1-based position in the join order.
        display_name: Player display name.
        bet: Optional bet amount to append as inline code.
        is_allin: Whether to mark the row with an `all-in` suffix.

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
    """Renders the small-text settlement metadata line.

    Args:
        delta: Player net point change for the round.
        new_balance: Player balance after settlement.
        is_allin: Whether the wager consumed the full balance.
        base_delta: Player net point change before player-facing bonuses.
        vip_bonus: Extra points added by the VIP payout bonus.
        five_card_bonus: System-funded bonus from five-card 21.

    Returns:
        `-# 本局 +X · 餘額 Y` style metadata, with an `· all-in` suffix
        when the round was all-in.
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
    """Formats the H2 result line for one player at Blackjack settlement.

    Args:
        outcome: Player-facing Blackjack outcome label.
        player_total: Final player hand total.
        dealer_total: Final dealer hand total.

    Returns:
        Markdown H2 line such as `## 🎉 你贏了 · 20 > 19`.
    """
    return f"## {player_result_inline(outcome=outcome, player_total=player_total, dealer_total=dealer_total)}"


def player_result_inline(outcome: SettleOutcome, player_total: int, dealer_total: int) -> str:  # noqa: PLR0911 -- one branch per SettleOutcome label keeps the mapping obvious
    """Single-line result label without heading prefix, for embed titles."""
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
