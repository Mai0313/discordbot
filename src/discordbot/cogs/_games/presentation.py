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


def settlement_footer(*, delta: int, new_balance: int, is_allin: bool) -> str:
    """Formats the shared final-round settlement footer.

    Keeps only the two numbers the player needs to see at a glance — the
    round delta and the post-settlement balance — and lets ``/house`` carry
    the dealer ledger when the player explicitly asks for it.

    Args:
        delta: Player net point change for the round.
        new_balance: Player balance after settlement.
        is_allin: Whether the requested bet was clamped to the full balance.

    Returns:
        Footer text for a final game embed.
    """
    return (
        f"本局 {currency_text(amount=delta, signed=True)} · "
        f"餘額 {currency_text(amount=new_balance)}{allin_note(is_allin=is_allin)}"
    )
