"""Tests for the /blackjack pure-rules module."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random

import pytest

from discordbot.typings.games import GameParticipant
from discordbot.cogs._games.blackjack import (
    Card,
    BlackjackHand,
    BlackjackRound,
    settle,
    is_bust,
    hand_value,
    render_hand,
    is_blackjack,
    dealer_visible_value,
)
from discordbot.cogs._games.settlement import blackjack_early_finish_note


def test_hand_value_no_aces() -> None:
    """Plain numeric cards just sum their face values."""
    assert hand_value(cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")]) == 19


def test_hand_value_face_cards_count_as_ten() -> None:
    """Each of J/Q/K is worth 10 points."""
    assert hand_value(cards=[Card(rank="K", suit="♠"), Card(rank="Q", suit="♥")]) == 20


def test_hand_value_ace_high_when_safe() -> None:
    """An ace counts as 11 when it doesn't push the hand over 21."""
    assert hand_value(cards=[Card(rank="A", suit="♠"), Card(rank="10", suit="♥")]) == 21


def test_hand_value_ace_demoted_when_needed() -> None:
    """Aces drop to 1 to avoid a bust."""
    cards = [Card(rank="A", suit="♠"), Card(rank="10", suit="♥"), Card(rank="3", suit="♣")]
    assert hand_value(cards=cards) == 14


def test_hand_value_double_ace_demotes_one() -> None:
    """A+A starts at 22 and demotes one ace to 1, giving 12."""
    assert hand_value(cards=[Card(rank="A", suit="♠"), Card(rank="A", suit="♥")]) == 12


def test_is_blackjack_only_for_two_card_21() -> None:
    """Three sevens is 21 but not a natural Blackjack."""
    assert is_blackjack(cards=[Card(rank="A", suit="♠"), Card(rank="K", suit="♥")]) is True
    triple_seven = [Card(rank="7", suit="♠"), Card(rank="7", suit="♥"), Card(rank="7", suit="♣")]
    assert is_blackjack(cards=triple_seven) is False


def test_is_bust_above_21() -> None:
    """is_bust returns True only when the value exceeds 21."""
    bust = [Card(rank="K", suit="♠"), Card(rank="Q", suit="♥"), Card(rank="2", suit="♣")]
    safe = [Card(rank="K", suit="♠"), Card(rank="Q", suit="♥")]
    assert is_bust(cards=bust) is True
    assert is_bust(cards=safe) is False


def _hand_with(player: list[Card], dealer: list[Card], bet: int = 100) -> BlackjackHand:
    hand = BlackjackHand(rng=Random(x=0), bet=bet)
    hand.player = player
    hand.dealer = dealer
    hand.finished = True
    return hand


def _participant(user_id: int, display_name: str, bet: int = 100) -> GameParticipant:
    return GameParticipant(
        user_id=user_id,
        account_name=display_name.lower(),
        display_name=display_name,
        bet=bet,
        balance_at_start=1_000,
        is_allin=False,
    )


def test_settle_player_blackjack_pays_three_to_two() -> None:
    """A natural Blackjack pays 1.5x the bet (rounded down)."""
    hand = _hand_with(
        player=[Card(rank="A", suit="♠"), Card(rank="K", suit="♥")],
        dealer=[Card(rank="9", suit="♠"), Card(rank="7", suit="♥")],
    )
    outcome, delta = settle(hand=hand)
    assert outcome == "blackjack"
    assert delta == 150


def test_settle_double_blackjack_is_push() -> None:
    """Two Blackjacks at the table cancel out."""
    hand = _hand_with(
        player=[Card(rank="A", suit="♠"), Card(rank="K", suit="♥")],
        dealer=[Card(rank="A", suit="♣"), Card(rank="Q", suit="♦")],
    )
    outcome, delta = settle(hand=hand)
    assert outcome == "push"
    assert delta == 0


def test_blackjack_early_finish_note_explains_dealer_natural() -> None:
    """A dealer natural Blackjack can end the round before the player acts."""
    hand = _hand_with(
        player=[Card(rank="9", suit="♠"), Card(rank="7", suit="♥")],
        dealer=[Card(rank="A", suit="♣"), Card(rank="Q", suit="♦")],
    )
    assert blackjack_early_finish_note(hand=hand) == "莊家起手 Blackjack, 依規則本局直接結算"


def test_blackjack_early_finish_note_ignores_regular_twenty_one() -> None:
    """A non-natural 21 should not be described as an early Blackjack finish."""
    hand = _hand_with(
        player=[Card(rank="9", suit="♠"), Card(rank="7", suit="♥")],
        dealer=[Card(rank="7", suit="♣"), Card(rank="7", suit="♦"), Card(rank="7", suit="♠")],
    )
    assert blackjack_early_finish_note(hand=hand) is None


def test_settle_player_bust_loses_bet() -> None:
    """When the player busts the dealer wins regardless of dealer total."""
    hand = _hand_with(
        player=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥"), Card(rank="5", suit="♣")],
        dealer=[Card(rank="10", suit="♣"), Card(rank="6", suit="♦")],
        bet=50,
    )
    outcome, delta = settle(hand=hand)
    assert outcome == "player_bust"
    assert delta == -50


def test_settle_dealer_bust_pays_even_money() -> None:
    """When the dealer busts the player wins the bet."""
    hand = _hand_with(
        player=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer=[Card(rank="10", suit="♣"), Card(rank="6", suit="♦"), Card(rank="K", suit="♠")],
        bet=50,
    )
    outcome, delta = settle(hand=hand)
    assert outcome == "dealer_bust"
    assert delta == 50


def test_settle_higher_total_wins() -> None:
    """The higher (non-bust) total wins one bet at even money."""
    hand = _hand_with(
        player=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer=[Card(rank="10", suit="♣"), Card(rank="8", suit="♦")],
        bet=50,
    )
    outcome, delta = settle(hand=hand)
    assert outcome == "win"
    assert delta == 50


def test_settle_lower_total_loses() -> None:
    """A lower (non-bust) total loses the bet."""
    hand = _hand_with(
        player=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer=[Card(rank="10", suit="♣"), Card(rank="8", suit="♦")],
        bet=50,
    )
    outcome, delta = settle(hand=hand)
    assert outcome == "lose"
    assert delta == -50


def test_settle_equal_total_is_push() -> None:
    """Equal totals push regardless of card composition."""
    hand = _hand_with(
        player=[Card(rank="10", suit="♠"), Card(rank="8", suit="♥")],
        dealer=[Card(rank="9", suit="♣"), Card(rank="9", suit="♦")],
        bet=50,
    )
    outcome, delta = settle(hand=hand)
    assert outcome == "push"
    assert delta == 0


def test_settle_unfinished_hand_raises() -> None:
    """Trying to settle a still-live hand is a programmer error."""
    hand = BlackjackHand(rng=Random(x=0), bet=50)
    hand.player = [Card(rank="10", suit="♠")]
    with pytest.raises(expected_exception=ValueError, match="unfinished"):
        settle(hand=hand)


def test_dealer_keeps_drawing_below_17() -> None:
    """Dealer must hit until the hand value is ≥ 17 (or it busts)."""
    hand = BlackjackHand(rng=Random(x=12345), bet=100)
    hand.player = [Card(rank="10", suit="♠"), Card(rank="9", suit="♥")]
    hand.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    hand.stand()
    final = hand.dealer_total()
    assert final >= 17 or is_bust(cards=hand.dealer)


def test_blackjack_round_advances_players_and_dealer_after_all_stand() -> None:
    """The multiplayer round advances in join order and resolves dealer play once."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=12345),
        participants=[
            _participant(user_id=1, display_name="Alice"),
            _participant(user_id=2, display_name="Bob"),
        ],
    )
    round_state.players[0].cards = [Card(rank="10", suit="♠"), Card(rank="8", suit="♥")]
    round_state.players[1].cards = [Card(rank="9", suit="♣"), Card(rank="8", suit="♦")]
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]

    assert round_state.active_player() == round_state.players[0]
    round_state.stand(user_id=1)
    assert round_state.active_player() == round_state.players[1]
    round_state.stand(user_id=2)

    assert round_state.finished is True
    assert round_state.dealer_played is True
    assert round_state.dealer_total() >= 17 or is_bust(cards=round_state.dealer)


def test_blackjack_round_rejects_action_from_non_active_player() -> None:
    """Only the current player can mutate the shared round."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[
            _participant(user_id=1, display_name="Alice"),
            _participant(user_id=2, display_name="Bob"),
        ],
    )
    round_state.players[0].cards = [Card(rank="10", suit="♠"), Card(rank="8", suit="♥")]
    round_state.players[1].cards = [Card(rank="9", suit="♣"), Card(rank="8", suit="♦")]
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]

    with pytest.raises(expected_exception=ValueError, match="turn"):
        round_state.hit(user_id=2)

    assert len(round_state.players[0].cards) == 2


def test_render_hand_hides_first_card() -> None:
    """When the hole card is hidden, only the up-card and a back glyph appear."""
    cards = [Card(rank="A", suit="♠"), Card(rank="K", suit="♥")]
    rendered = render_hand(cards=cards, hide_first=True)
    assert "🂠" in rendered
    assert "A" not in rendered
    assert "K" in rendered


def test_dealer_visible_value_uses_up_card() -> None:
    """The visible value matches the card not hidden from the player."""
    hand = BlackjackHand(rng=Random(x=0), bet=10)
    hand.dealer = [Card(rank="A", suit="♠"), Card(rank="K", suit="♥")]
    assert dealer_visible_value(hand=hand) == 10
    hand.dealer = [Card(rank="K", suit="♠"), Card(rank="A", suit="♥")]
    assert dealer_visible_value(hand=hand) == 11
    hand.dealer = [Card(rank="7", suit="♠")]
    assert dealer_visible_value(hand=hand) == 7
