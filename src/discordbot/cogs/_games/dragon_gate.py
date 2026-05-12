"""Pure helpers for the /dragon_gate command."""

from random import Random

from discordbot.typings.games import Card, DragonGateResult, DragonGateOutcome
from discordbot.cogs._games.blackjack import draw_card


def card_value(*, card: Card) -> int:
    """Returns the Dragon Gate value for a card, with Ace high."""
    values = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }
    return values[card.rank]


def _ordered_gate(*, first_gate: Card, second_gate: Card) -> tuple[Card, Card]:
    if card_value(card=first_gate) <= card_value(card=second_gate):
        return first_gate, second_gate
    return second_gate, first_gate


def settle_dragon_gate(*, first_gate: Card, second_gate: Card, shot: Card) -> DragonGateResult:
    """Settles fixed Dragon Gate cards into a result.

    Args:
        first_gate: First gate card.
        second_gate: Second gate card.
        shot: Card shot between the gates.

    Returns:
        A complete result with sorted gates and the player-facing outcome.
    """
    lower_gate, upper_gate = _ordered_gate(first_gate=first_gate, second_gate=second_gate)
    lower_value = card_value(card=lower_gate)
    upper_value = card_value(card=upper_gate)
    shot_value = card_value(card=shot)

    if upper_value - lower_value <= 1:
        outcome: DragonGateOutcome = "push"
    elif lower_value < shot_value < upper_value:
        outcome = "win"
    else:
        outcome = "lose"

    return DragonGateResult(
        first_gate=first_gate,
        second_gate=second_gate,
        lower_gate=lower_gate,
        upper_gate=upper_gate,
        shot=shot,
        outcome=outcome,
    )


def play_dragon_gate(*, rng: Random) -> DragonGateResult:
    """Deals one Dragon Gate round from a notional infinite shoe."""
    return settle_dragon_gate(
        first_gate=draw_card(rng=rng), second_gate=draw_card(rng=rng), shot=draw_card(rng=rng)
    )


def render_card_value(*, card: Card) -> str:
    """Formats one card with its Dragon Gate rank value."""
    return f"{card} (= {card_value(card=card)})"


def dragon_gate_detail(*, result: DragonGateResult) -> str:
    """Formats a concise round detail for dealer prompts."""
    lower_value = card_value(card=result.lower_gate)
    upper_value = card_value(card=result.upper_gate)
    shot_value = card_value(card=result.shot)
    if result.outcome == "push":
        reason = "兩柱沒有有效空間, 本局 push"
    elif result.outcome == "win":
        reason = "射門牌落在兩柱中間"
    else:
        reason = "射門牌撞柱或落在門外"
    return (
        f"龍門 {result.lower_gate} ({lower_value}) 到 {result.upper_gate} ({upper_value}), "
        f"射門牌 {result.shot} ({shot_value}); {reason}"
    )
