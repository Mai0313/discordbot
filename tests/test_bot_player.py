"""Deterministic bot-player Blackjack decision tests."""

from typing import TYPE_CHECKING

import pytest

from discordbot.cogs.games.bot_player import (
    BOT_TABLE_EDGE,
    kelly_bet,
    fallback_action,
    choose_bot_action,
    bot_takes_insurance,
    count_adjusted_edge,
)

from tests.helpers.games import card

if TYPE_CHECKING:
    from discordbot.typings.games import BotAction


def test_fallback_action_stands_on_ten_value_pair() -> None:
    """10-value pairs should not be split by the fallback table."""
    action = fallback_action(
        hand_cards=[card(rank="10"), card(rank="K")],
        hand_total=20,
        dealer_up=card(rank="6"),
        is_pair_hand=True,
        allowed_actions=("hit", "stand", "split"),
    )

    assert action == "stand"


def test_fallback_action_doubles_pair_fives_as_hard_ten() -> None:
    """5/5 is played as hard 10 instead of a split pair."""
    action = fallback_action(
        hand_cards=[card(rank="5"), card(rank="5")],
        hand_total=10,
        dealer_up=card(rank="6"),
        is_pair_hand=True,
        allowed_actions=("hit", "stand", "double", "split"),
    )

    assert action == "double"


def test_fallback_action_surrenders_hard_sixteen_against_ten() -> None:
    """Late surrender takes precedence for hard 16 against dealer 10."""
    action = fallback_action(
        hand_cards=[card(rank="10"), card(rank="6")],
        hand_total=16,
        dealer_up=card(rank="J"),
        is_pair_hand=False,
        allowed_actions=("hit", "stand", "surrender"),
    )

    assert action == "surrender"


def test_fallback_action_splits_eights_against_ten() -> None:
    """8/8 remains a split even against a dealer 10."""
    action = fallback_action(
        hand_cards=[card(rank="8"), card(rank="8")],
        hand_total=16,
        dealer_up=card(rank="10"),
        is_pair_hand=True,
        allowed_actions=("hit", "stand", "surrender", "split"),
    )

    assert action == "split"


def test_insurance_is_taken_when_the_unseen_shoe_is_ten_rich() -> None:
    """A shoe more than a third ten-value makes insurance +EV, whatever the hole is."""
    assert bot_takes_insurance(shoe=[card(rank="10"), card(rank="J"), card(rank="Q")]) is True


def test_insurance_is_declined_unless_the_ten_density_clears_one_third() -> None:
    """At or under one third ten-value, or with nothing left to count, the bot declines.

    The hole card is never an input, so the bot cannot win insurance on a real dealer
    Blackjack it could not have counted its way to.
    """
    low_shoe = [card(rank="2"), card(rank="3"), card(rank="4"), card(rank="5"), card(rank="6")]

    assert bot_takes_insurance(shoe=low_shoe) is False
    assert bot_takes_insurance(shoe=[card(rank="10"), card(rank="2"), card(rank="3")]) is False
    assert bot_takes_insurance(shoe=[]) is False


def test_action_uses_ev_recommendation() -> None:
    """The played action is the EV engine's hole-aware recommendation, not the table's.

    Hard 16 against a 10 is a hit by the up-card-only table, but the hole is a 6 and the shoe
    holds only tens: the dealer's 16 must draw one and bust, so standing wins, and a hit would
    have busted the bot instead.
    """
    hand_cards = [card(rank="10"), card(rank="6")]
    dealer_up = card(rank="10")
    allowed_actions: tuple[BotAction, ...] = ("hit", "stand")
    action = choose_bot_action(
        hand_cards=hand_cards,
        dealer_cards=[card(rank="6"), dealer_up],
        shoe=[card(rank="10")] * 20,
        allowed_actions=allowed_actions,
        is_pair_hand=False,
        bet=100,
    )

    assert action == "stand"
    table_action = fallback_action(
        hand_cards=hand_cards,
        hand_total=16,
        dealer_up=dealer_up,
        is_pair_hand=False,
        allowed_actions=allowed_actions,
    )
    assert table_action == "hit", "the table must disagree, or this cannot tell the two apart"


def test_action_falls_back_to_the_table_when_the_engine_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing EV engine hands the turn to the up-card table instead of crashing it.

    Hard 16 against an up-card 10 is a hit by the table; reading the 6 in the hole as the
    up-card would make it a stand.
    """

    def engine_down(**_kwargs: object) -> None:
        raise RuntimeError("engine down")

    monkeypatch.setattr("discordbot.cogs.games.bot_player.compute_action_evs", engine_down)

    action = choose_bot_action(
        hand_cards=[card(rank="10"), card(rank="6")],
        dealer_cards=[card(rank="6"), card(rank="10")],
        shoe=[card(rank="10")] * 20,
        allowed_actions=("hit", "stand"),
        is_pair_hand=False,
        bet=100,
    )

    assert action == "hit"


def test_kelly_bet_wagers_half_kelly_fraction_within_bounds() -> None:
    """A positive edge wagers the clamped half-Kelly fraction, floored at the table minimum."""
    bet = kelly_bet(
        balance=100_000, table_minimum=100, edge=0.163, variance=1.334, kelly_fraction=0.5
    )

    assert bet == round(0.5 * 0.163 / 1.334 * 100_000)
    assert 100 <= bet <= 100_000


def test_kelly_bet_floors_at_table_minimum_on_non_positive_edge() -> None:
    """A non-positive edge falls back to the table minimum instead of refusing to play."""
    assert kelly_bet(balance=100_000, table_minimum=500, edge=0.0) == 500
    assert kelly_bet(balance=100_000, table_minimum=500, edge=-0.2) == 500


def test_kelly_bet_caps_fraction_and_clamps_to_balance() -> None:
    """The hard fraction cap bounds the wager even when the edge is extreme."""
    assert kelly_bet(
        balance=1_000, table_minimum=1, edge=10.0, variance=1.0, max_fraction=0.10
    ) == (100)
    assert kelly_bet(balance=0, table_minimum=100) == 1
    # A short stack stays inside the 10% ceiling instead of going all-in to match.
    assert kelly_bet(balance=50, table_minimum=100, edge=0.0) == 5


def test_kelly_bet_caps_a_large_table_stake_at_the_bankroll_fraction() -> None:
    """A table stake larger than the bankroll ceiling no longer drags the bot above it."""
    # The owner opens a 1,000,000 table; the bot has 1,000,000 but stays within its
    # 10% Kelly ceiling instead of matching the whole stake.
    assert kelly_bet(balance=1_000_000, table_minimum=1_000_000, edge=0.13) == 100_000
    # The ceiling also bounds the non-positive-edge floor path.
    assert kelly_bet(balance=1_000_000, table_minimum=1_000_000, edge=0.0) == 100_000


def test_count_adjusted_edge_rises_with_true_count() -> None:
    """The edge equals the base at a neutral count and increases with the true count."""
    assert count_adjusted_edge(true_count=0.0) == BOT_TABLE_EDGE
    assert count_adjusted_edge(true_count=6.0) > count_adjusted_edge(true_count=0.0)
    assert count_adjusted_edge(true_count=-6.0) < count_adjusted_edge(true_count=0.0)


def test_kelly_bet_spreads_higher_on_a_favorable_count() -> None:
    """A favorable true count raises the count-adjusted Kelly wager (bet spread)."""
    neutral = kelly_bet(
        balance=1_000_000, table_minimum=100, edge=count_adjusted_edge(true_count=0.0)
    )
    favorable = kelly_bet(
        balance=1_000_000, table_minimum=100, edge=count_adjusted_edge(true_count=8.0)
    )

    assert favorable > neutral
