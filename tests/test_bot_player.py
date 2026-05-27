"""Deterministic bot-player Blackjack fallback tests."""

from discordbot.typings.games import Card
from discordbot.cogs._games.bot_player import fallback_action


def _card(rank: str) -> Card:
    """Builds a card with an arbitrary suit for strategy tests."""
    return Card(rank=rank, suit="♠")


def test_fallback_action_stands_on_ten_value_pair() -> None:
    """10-value pairs should not be split by the fallback table."""
    action = fallback_action(
        hand_cards=[_card(rank="10"), _card(rank="K")],
        hand_total=20,
        dealer_up=_card(rank="6"),
        is_pair_hand=True,
        allowed_actions=("hit", "stand", "split"),
    )

    assert action == "stand"


def test_fallback_action_doubles_pair_fives_as_hard_ten() -> None:
    """5/5 is played as hard 10 instead of a split pair."""
    action = fallback_action(
        hand_cards=[_card(rank="5"), _card(rank="5")],
        hand_total=10,
        dealer_up=_card(rank="6"),
        is_pair_hand=True,
        allowed_actions=("hit", "stand", "double", "split"),
    )

    assert action == "double"


def test_fallback_action_surrenders_hard_sixteen_against_ten() -> None:
    """Late surrender takes precedence for hard 16 against dealer 10."""
    action = fallback_action(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        hand_total=16,
        dealer_up=_card(rank="J"),
        is_pair_hand=False,
        allowed_actions=("hit", "stand", "surrender"),
    )

    assert action == "surrender"


def test_fallback_action_splits_eights_against_ten() -> None:
    """8/8 remains a split even against a dealer 10."""
    action = fallback_action(
        hand_cards=[_card(rank="8"), _card(rank="8")],
        hand_total=16,
        dealer_up=_card(rank="10"),
        is_pair_hand=True,
        allowed_actions=("hit", "stand", "surrender", "split"),
    )

    assert action == "split"
