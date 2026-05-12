"""Shared presentation helpers for casino game embeds."""

from discordbot.typings.games import DiceOutcome, SettleOutcome, DragonGateOutcome
from discordbot.cogs._economy.presentation import currency_text

IN_PROGRESS_COLOR = 0x5865F2
WIN_COLOR = 0x57F287
LOSE_COLOR = 0xED4245
PUSH_COLOR = 0xFEE75C
ERROR_COLOR = 0xED4245


def dice_outcome_presentation(outcome: DiceOutcome) -> tuple[str, int]:
    """Returns presentation values for a dice outcome.

    Args:
        outcome: Player-facing dice outcome.

    Returns:
        A `(label, color)` tuple for the final embed.
    """
    dice_result = {
        "win": ("你贏了", WIN_COLOR),
        "lose": ("你輸了", LOSE_COLOR),
        "push": ("平手", PUSH_COLOR),
    }
    return dice_result[outcome]


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
        "player_bust": ("你爆牌了", LOSE_COLOR),
        "dealer_bust": ("莊家爆牌, 你贏了", WIN_COLOR),
    }
    return blackjack_result[outcome]


def dragon_gate_outcome_presentation(outcome: DragonGateOutcome) -> tuple[str, int]:
    """Returns presentation values for a Dragon Gate outcome.

    Args:
        outcome: Player-facing Dragon Gate outcome.

    Returns:
        A `(label, color)` tuple for the final embed.
    """
    dragon_gate_result = {
        "win": ("射進龍門, 你贏了", WIN_COLOR),
        "lose": ("射偏了, 你輸了", LOSE_COLOR),
        "push": ("沒有有效龍門, 退回下注", PUSH_COLOR),
    }
    return dragon_gate_result[outcome]


def allin_note(*, is_allin: bool) -> str:
    """Returns the shared suffix for auto all-in rounds.

    Args:
        is_allin: Whether the requested bet was clamped to the full balance.

    Returns:
        The suffix text, or an empty string for non all-in rounds.
    """
    return " · 已自動 all-in" if is_allin else ""


def bet_field_value(*, bet: int, is_allin: bool) -> str:
    """Formats a bet amount for in-progress embeds.

    Args:
        bet: Effective bet amount in points.
        is_allin: Whether the requested bet was clamped to the full balance.

    Returns:
        A display string for the wager field.
    """
    suffix = " (已自動 all-in)" if is_allin else ""
    return f"{currency_text(amount=bet)}{suffix}"


def settlement_footer(
    *, bet: int, delta: int, new_balance: int, house_balance: int, is_allin: bool
) -> str:
    """Formats the shared final-round settlement footer.

    Args:
        bet: Effective bet amount in points.
        delta: Player net point change for the round.
        new_balance: Player balance after settlement.
        house_balance: Dealer ledger balance after settlement.
        is_allin: Whether the requested bet was clamped to the full balance.

    Returns:
        Footer text for a final game embed.
    """
    return (
        f"下注 {currency_text(amount=bet)} · "
        f"本局淨變動 {currency_text(amount=delta, signed=True)} · "
        f"餘額 {currency_text(amount=new_balance)} · "
        f"莊家餘額 {currency_text(amount=house_balance)}{allin_note(is_allin=is_allin)}"
    )
