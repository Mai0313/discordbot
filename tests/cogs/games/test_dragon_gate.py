"""Tests for the 射龍門 rules."""

from __future__ import annotations

import pytest

from discordbot.cogs.games.dragon_gate import (
    DragonGateRound,
    DragonGateDirection,
    DragonGateParticipantUnknownError,
    DragonGatePairChoiceUnavailableError,
    card_value,
    has_open_gate,
)

from tests.helpers.games import card
from tests.helpers.dragon_gate import RiggedRandom, dragon_gate_seat


def test_card_value_uses_ace_low_and_faces_above_ten() -> None:
    """射龍門 compares A as 1 and J/Q/K as 11/12/13."""
    assert card_value(card=card(rank="A")) == 1
    assert card_value(card=card(rank="J")) == 11
    assert card_value(card=card(rank="Q")) == 12
    assert card_value(card=card(rank="K")) == 13


def test_adjacent_non_pair_pillars_are_redealt_without_counting_turn() -> None:
    """Adjacent non-pair pillars have no gate and are skipped before betting."""
    assert has_open_gate(pillars=[card(rank="4"), card(rank="3", suit="♥")]) is False
    assert has_open_gate(pillars=[card(rank="7"), card(rank="7", suit="♥")]) is True

    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("4", "♠", "3", "♥", "5", "♣", "9", "♦", "7", "♠")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )

    assert round_state.turn_number == 1
    assert round_state.active_turn is not None
    assert [card.rank for card in round_state.active_turn.pillars] == ["5", "9"]

    result = round_state.resolve_bet(user_id=1, amount=10_000, jackpot=100_000)
    assert result.outcome == "gate_win"
    assert result.delta == 10_000


def test_gate_win_returns_positive_delta() -> None:
    """A third card between the pillars wins one bet from the pot."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "7", "♣")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )
    result = round_state.resolve_bet(user_id=1, amount=10_000, jackpot=100_000)
    round_state.record_result(result=result)

    assert result.outcome == "gate_win"
    assert result.delta == 10_000
    assert round_state.player_delta(user_id=1) == 10_000


def test_outside_card_returns_negative_one_bet() -> None:
    """A third card outside the gate loses one bet."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "K", "♣", "A", "♦", "K", "♥")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )
    result = round_state.resolve_bet(user_id=1, amount=10_000, jackpot=100_000)
    round_state.record_result(result=result)

    assert result.outcome == "outside_lose"
    assert result.delta == -10_000
    assert round_state.player_delta(user_id=1) == -10_000


def test_pillar_hit_returns_negative_double_bet() -> None:
    """A third card equal to either pillar loses two bets."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "9", "♣", "A", "♦", "K", "♥")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )
    result = round_state.resolve_bet(user_id=1, amount=10_000, jackpot=100_000)
    round_state.record_result(result=result)

    assert result.outcome == "pillar_hit"
    assert result.delta == -20_000
    assert round_state.player_delta(user_id=1) == -20_000


def test_pair_gate_requires_high_or_low_choice() -> None:
    """Same-point pillars require a higher/lower choice before betting."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", "8", "♣")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )

    with pytest.raises(expected_exception=ValueError, match="direction"):
        round_state.resolve_bet(user_id=1, amount=10_000, jackpot=100_000)

    round_state.choose_pair_direction(user_id=1, direction="higher")
    result = round_state.resolve_bet(user_id=1, amount=10_000, jackpot=100_000)
    assert result.outcome == "pair_win"
    assert result.delta == 10_000


def test_pair_pillar_hit_returns_triple_loss() -> None:
    """A same-point third card on a same-point gate loses three bets."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", "7", "♣", "A", "♦", "K", "♥")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )
    round_state.choose_pair_direction(user_id=1, direction="lower")
    result = round_state.resolve_bet(user_id=1, amount=10_000, jackpot=100_000)
    round_state.record_result(result=result)

    assert result.outcome == "pair_pillar_hit"
    assert result.delta == -30_000
    assert round_state.player_delta(user_id=1) == -30_000


@pytest.mark.parametrize(
    argnames=("direction", "third"), argvalues=[("higher", "3"), ("lower", "9")]
)
def test_a_pair_card_on_the_side_not_called_loses_one_bet(
    direction: DragonGateDirection, third: str
) -> None:
    """A third card off the pillar pays only on the side the player called."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("7", "♠", "7", "♥", third, "♣")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )
    round_state.choose_pair_direction(user_id=1, direction=direction)
    result = round_state.resolve_bet(user_id=1, amount=10_000, jackpot=100_000)
    round_state.record_result(result=result)

    assert (result.outcome, result.delta) == ("pair_lose", -10_000)
    assert round_state.player_delta(user_id=1) == -10_000


@pytest.mark.parametrize(
    argnames=("rank", "expected"), argvalues=[("A", "higher"), ("K", "lower")], ids=["ace", "king"]
)
def test_an_ace_or_king_pair_is_dealt_with_the_only_guess_that_can_win(
    rank: str, expected: DragonGateDirection
) -> None:
    """Nothing ranks below an ace or above a king, so the one winnable guess comes preset."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=(rank, "♠", rank, "♥", "7", "♣")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )

    assert round_state.needs_pair_choice() is False
    assert round_state.active_turn is not None
    assert round_state.active_turn.direction == expected

    result = round_state.resolve_bet(user_id=1, amount=10_000, jackpot=100_000)
    assert (result.outcome, result.delta, result.direction) == ("pair_win", 10_000, expected)


@pytest.mark.parametrize(
    argnames=("rank", "losing", "expected"),
    argvalues=[("A", "lower", "higher"), ("K", "higher", "lower")],
    ids=["ace", "king"],
)
def test_an_ace_or_king_pair_refuses_the_guess_that_cannot_win(
    rank: str, losing: DragonGateDirection, expected: DragonGateDirection
) -> None:
    """The guess that can only lose is never accepted on an ace or king pair."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=(rank, "♠", rank, "♥")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )

    with pytest.raises(expected_exception=DragonGatePairChoiceUnavailableError):
        round_state.choose_pair_direction(user_id=1, direction=losing)
    assert round_state.active_turn is not None
    assert round_state.active_turn.direction == expected


def test_a_gate_that_is_not_a_pair_offers_no_guess() -> None:
    """Only a pair takes a high/low guess, even when a pillar is an ace."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("A", "♠", "5", "♥")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )

    assert round_state.active_turn is not None
    assert round_state.active_turn.direction is None
    with pytest.raises(expected_exception=DragonGatePairChoiceUnavailableError):
        round_state.choose_pair_direction(user_id=1, direction="higher")


def test_turns_rotate_through_active_seats() -> None:
    """The next active player is dealt a fresh gate after a bet resolves."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "K", "♣", "4", "♦", "Q", "♣")),
        participants=[
            dragon_gate_seat(user_id=1, display_name="Alice"),
            dragon_gate_seat(user_id=2, display_name="Bob"),
        ],
    )

    round_state.record_result(
        result=round_state.resolve_bet(user_id=1, amount=10_000, jackpot=100_000)
    )

    assert round_state.finished is False
    assert round_state.active_turn is not None
    assert round_state.active_turn.participant.user_id == 2
    assert [card.rank for card in round_state.active_turn.pillars] == ["4", "Q"]


def test_withdraw_advances_to_next_player_and_records_delta() -> None:
    """Withdrawing the active player skips to the next non-withdrawn seat."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥", "4", "♦", "Q", "♣")),
        participants=[
            dragon_gate_seat(user_id=1, display_name="Alice"),
            dragon_gate_seat(user_id=2, display_name="Bob"),
        ],
    )

    leftover = round_state.withdraw(user_id=1)

    assert leftover == 0
    assert round_state.finished is False
    assert round_state.active_turn is not None
    assert round_state.active_turn.participant.user_id == 2


def test_withdraw_finishes_round_when_last_player_leaves() -> None:
    """The round flips finished after the final active player leaves."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )

    round_state.withdraw(user_id=1)

    assert round_state.finished is True
    assert round_state.active_turn is None


def test_withdraw_rejects_non_participant() -> None:
    """Withdrawing someone not at the table is a programmer error."""
    round_state = DragonGateRound.from_participants(
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        participants=[dragon_gate_seat(user_id=1, display_name="Alice")],
    )

    with pytest.raises(expected_exception=DragonGateParticipantUnknownError):
        round_state.withdraw(user_id=999)
