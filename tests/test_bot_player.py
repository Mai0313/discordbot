"""Deterministic bot-player Blackjack decision tests."""

from discordbot.typings.games import Card
from discordbot.cogs._games.bot_player import (
    BOT_TABLE_EDGE,
    kelly_bet,
    fallback_action,
    choose_bot_action,
    fallback_insurance,
    count_adjusted_edge,
    build_bot_action_context,
    build_bot_insurance_context,
)


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


def _full_shoe() -> list[Card]:
    """Builds a fresh four-deck shoe (208 cards) as a flat card list for EV-engine tests."""
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    # Four decks of four suits each: every rank appears 16 times.
    return [_card(rank=rank) for rank in ranks] * 16


def test_fallback_action_uses_hole_card_when_dealer_cards_provided() -> None:
    """With dealer cards and shoe, the fallback exploits the known hole card."""
    shoe = _full_shoe()
    weak = fallback_action(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        hand_total=16,
        dealer_up=_card(rank="10"),
        is_pair_hand=False,
        allowed_actions=("hit", "stand", "surrender"),
        dealer_cards=[_card(rank="5"), _card(rank="10")],
        shoe=shoe,
    )
    strong = fallback_action(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        hand_total=16,
        dealer_up=_card(rank="10"),
        is_pair_hand=False,
        allowed_actions=("hit", "stand", "surrender"),
        dealer_cards=[_card(rank="10"), _card(rank="10")],
        shoe=shoe,
    )

    assert weak == "stand"
    assert strong == "surrender"
    assert weak != strong


def test_fallback_action_without_shoe_uses_plain_strategy() -> None:
    """Omitting dealer cards and shoe reproduces the classic basic-strategy table."""
    action = fallback_action(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        hand_total=16,
        dealer_up=_card(rank="J"),
        is_pair_hand=False,
        allowed_actions=("hit", "stand", "surrender"),
    )

    assert action == "surrender"


def test_fallback_insurance_is_count_based() -> None:
    """Insurance fallback takes only when the remaining-shoe ten density makes it +EV."""
    take_context = build_bot_insurance_context(
        dealer_up=_card(rank="A"),
        shoe=[_card(rank="10"), _card(rank="J"), _card(rank="Q")],
        insurance_cost=50,
    )
    decline_context = build_bot_insurance_context(
        dealer_up=_card(rank="A"),
        shoe=[_card(rank="2"), _card(rank="3"), _card(rank="4"), _card(rank="5"), _card(rank="6")],
        insurance_cost=50,
    )

    assert fallback_insurance(insurance_context=take_context) is True
    assert fallback_insurance(insurance_context=decline_context) is False
    assert fallback_insurance() is False


def test_action_context_exposes_up_card_only_without_hole() -> None:
    """Action context exposes rank counts and the dealer up-card, never the hole."""
    context = build_bot_action_context(
        hand_cards=[_card(rank="2"), _card(rank="3"), _card(rank="4"), _card(rank="5")],
        dealer_cards=[_card(rank="K"), _card(rank="A")],
        dealer_up=_card(rank="A"),
        shoe=[_card(rank="7"), _card(rank="2")],
        allowed_actions=("hit", "stand"),
        is_pair_hand=False,
        bet=100,
        balance_remaining=900,
    )

    assert context.dealer.up_card == "A♠"
    assert context.dealer.up_value == 11
    assert context.shoe_summary.total_cards == 2
    assert context.action_analysis.hit_odds is not None
    assert context.action_analysis.hit_odds.five_card_non_bust_probability > 0

    ev_analysis = context.action_analysis.ev_analysis
    assert ev_analysis is not None
    outcome = ev_analysis.dealer_outcome
    distribution_total = (
        outcome.bust_probability
        + outcome.total_17_probability
        + outcome.total_18_probability
        + outcome.total_19_probability
        + outcome.total_20_probability
        + outcome.total_21_probability
    )
    assert abs(distribution_total - 1.0) < 1e-9


def test_insurance_context_uses_remaining_shoe_count_not_hole() -> None:
    """A ten-rich remaining shoe makes insurance +EV without revealing the hole."""
    context = build_bot_insurance_context(
        dealer_up=_card(rank="A"),
        shoe=[_card(rank="10"), _card(rank="J"), _card(rank="Q")],
        insurance_cost=50,
    )

    assert context.ten_value_probability > 1 / 3
    assert context.insurance_recommendation == "take"
    assert context.insurance_expected_value > 0
    # The shown probability matches the shoe-only counts exactly, so it cannot be
    # cross-solved for the hole, and no Blackjack verdict is exposed.
    assert context.ten_value_probability == context.shoe_summary.ten_value_count / (
        context.shoe_summary.total_cards
    )


def test_insurance_declines_in_a_non_ten_rich_shoe() -> None:
    """A non-ten-rich shoe declines insurance regardless of the dealer's hole.

    This is the anti-cheat guarantee: `build_bot_insurance_context` is never even
    given the hole card, so it cannot win insurance on a real dealer Blackjack.
    """
    context = build_bot_insurance_context(
        dealer_up=_card(rank="A"),
        shoe=[_card(rank="2"), _card(rank="3"), _card(rank="4"), _card(rank="5"), _card(rank="6")],
        insurance_cost=50,
    )

    assert context.ten_value_probability < 1 / 3
    assert context.insurance_recommendation == "decline"
    assert context.insurance_expected_value < 0


def test_action_uses_ev_recommendation() -> None:
    """The played action is the EV engine's hole-aware recommendation."""
    action_context = build_bot_action_context(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        dealer_cards=[_card(rank="5"), _card(rank="10")],
        dealer_up=_card(rank="10"),
        shoe=[_card(rank="2"), _card(rank="3"), _card(rank="4")],
        allowed_actions=("hit", "stand"),
        is_pair_hand=False,
        bet=100,
        balance_remaining=900,
    )
    assert action_context.action_analysis.ev_analysis is not None
    assert action_context.action_analysis.basic_strategy_action == "hit"

    chosen = choose_bot_action(
        action_context=action_context,
        hand_cards=[_card(rank="10"), _card(rank="6")],
        hand_total=16,
        dealer_up=_card(rank="10"),
        is_pair_hand=False,
        allowed_actions=("hit", "stand"),
    )

    assert chosen == "hit"


def test_choose_bot_action_without_context_uses_basic_strategy() -> None:
    """With no EV context, the deterministic action falls back to the basic-strategy table."""
    chosen = choose_bot_action(
        action_context=None,
        hand_cards=[_card(rank="10"), _card(rank="6")],
        hand_total=16,
        dealer_up=_card(rank="J"),
        is_pair_hand=False,
        allowed_actions=("hit", "stand", "surrender"),
    )

    assert chosen == "surrender"


def test_insurance_decision_is_count_based() -> None:
    """The insurance decision is the count recommendation."""
    take_context = build_bot_insurance_context(
        dealer_up=_card(rank="A"),
        shoe=[_card(rank="10"), _card(rank="J"), _card(rank="Q")],
        insurance_cost=50,
    )
    decline_context = build_bot_insurance_context(
        dealer_up=_card(rank="A"),
        shoe=[_card(rank="2"), _card(rank="3"), _card(rank="4"), _card(rank="5"), _card(rank="6")],
        insurance_cost=50,
    )

    assert fallback_insurance(insurance_context=take_context) is True
    assert fallback_insurance(insurance_context=decline_context) is False


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
