"""Deterministic tests for the hole-card-aware Blackjack EV engine."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random

from discordbot.typings.games import Card, BotAction
from discordbot.cogs.games.blackjack import build_shoe
from discordbot.cogs.games.blackjack_ev import (
    _add_value,
    _make_context,
    recommend_action,
    _evaluate_actions,
    compute_true_count,
    _dealer_distribution,
    build_shoe_value_counts,
)

from tests.helpers.games import card

# Positions in the dealer distribution `_dealer_distribution` returns.
_DEALER_17 = 0
_DEALER_21 = 4
_DEALER_BUST = 5


def _dealer(*, total: int, soft: bool, shoe: tuple[int, ...]) -> tuple[float, ...]:
    """Returns the dealer's H17 final-total distribution from a known dealer hand."""
    return _dealer_distribution(total=total, soft=soft, shoe=shoe, memo={})


def _evs(
    *,
    hand: list[Card],
    dealer: list[Card],
    allowed: tuple[BotAction, ...],
    shoe: list[Card] | None = None,
    bet: int | None = None,
) -> dict[BotAction, float]:
    """Returns the EV per legal action, the numbers `recommend_action` picks its action from."""
    evs = _evaluate_actions(
        ctx=_make_context(dealer_cards=dealer),
        deck=build_shoe_value_counts(shoe=build_shoe(rng=Random(x=0)) if shoe is None else shoe),
        hand_cards=hand,
        allowed_actions=allowed,
        doubled=False,
        bet=bet,
    )
    return {item.action: item.expected_value for item in evs}


def test_dealer_hard_17_always_stands() -> None:
    """A hard 17 dealer stands with certainty regardless of the shoe."""
    outcome = _dealer(
        total=17, soft=False, shoe=build_shoe_value_counts(shoe=build_shoe(rng=Random(x=0)))
    )

    assert outcome[_DEALER_17] == 1.0
    assert outcome[_DEALER_BUST] == 0.0


def test_dealer_bust_probability_is_exact_on_a_tiny_shoe() -> None:
    """Dealer 16 over a shoe of one ten and one five busts or makes 21 with equal odds."""
    shoe = build_shoe_value_counts(shoe=[card(rank="10"), card(rank="5")])
    outcome = _dealer(total=16, soft=False, shoe=shoe)

    assert abs(outcome[_DEALER_BUST] - 0.5) < 1e-9
    assert abs(outcome[_DEALER_21] - 0.5) < 1e-9


def test_dealer_hits_soft_17_under_h17() -> None:
    """Soft 17 keeps drawing under H17, unlike a hard 17 that stands."""
    shoe = build_shoe_value_counts(shoe=build_shoe(rng=Random(x=0)))
    soft = _dealer(total=17, soft=True, shoe=shoe)
    hard = _dealer(total=17, soft=False, shoe=shoe)

    assert hard[_DEALER_17] == 1.0
    assert soft[_DEALER_17] < 1.0
    assert soft[_DEALER_BUST] > 0.0


def test_dealer_distribution_sums_to_one() -> None:
    """The dealer outcome distribution is a proper probability distribution."""
    shoe = build_shoe_value_counts(shoe=build_shoe(rng=Random(x=0)))
    for dealer_total, dealer_soft in ((12, False), (15, False), (16, False), (13, True)):
        outcome = _dealer(total=dealer_total, soft=dealer_soft, shoe=shoe)
        assert abs(sum(outcome) - 1.0) < 1e-9


def test_standing_beats_hitting_on_hard_twenty() -> None:
    """A hard 20 should stand, never hit, against a weak dealer."""
    hand = [card(rank="10"), card(rank="10")]
    dealer = [card(rank="9"), card(rank="6")]
    action = recommend_action(
        hand_cards=hand,
        dealer_cards=dealer,
        shoe=build_shoe(rng=Random(x=0)),
        allowed_actions=("hit", "stand"),
        doubled=False,
    )
    evs = _evs(hand=hand, dealer=dealer, allowed=("hit", "stand"))

    assert action == "stand"
    assert evs["stand"] > evs["hit"]


def test_recommendation_reads_the_hole_card() -> None:
    """The same hand against the same up-card plays differently once the hole is known.

    Both dealers show a 10, but one holds a weak 5 and the other a strong 10: that hidden
    difference is the bot's private edge.
    """
    shoe = build_shoe(rng=Random(x=0))
    weak = recommend_action(
        hand_cards=[card(rank="10"), card(rank="6")],
        dealer_cards=[card(rank="5"), card(rank="10")],
        shoe=shoe,
        allowed_actions=("hit", "stand", "surrender"),
        doubled=False,
    )
    strong = recommend_action(
        hand_cards=[card(rank="10"), card(rank="6")],
        dealer_cards=[card(rank="10"), card(rank="10")],
        shoe=shoe,
        allowed_actions=("hit", "stand", "surrender"),
        doubled=False,
    )

    assert weak == "stand"
    assert strong == "surrender"


def test_five_card_non_bust_stand_pays_one_unit() -> None:
    """A five-card non-bust hand wins one unit immediately, independent of the dealer."""
    evs = _evs(
        hand=[card(rank="2"), card(rank="3"), card(rank="4"), card(rank="4"), card(rank="5")],
        dealer=[card(rank="10"), card(rank="10")],
        allowed=("stand",),
    )

    assert abs(evs["stand"] - 1.0) < 1e-9


def test_five_card_twenty_one_earns_the_bonus() -> None:
    """A five-card 21 is worth more than a normal win because of the system bonus."""
    evs = _evs(
        hand=[card(rank="10"), card(rank="5"), card(rank="2"), card(rank="3"), card(rank="A")],
        dealer=[card(rank="10"), card(rank="9")],
        allowed=("stand",),
    )

    assert evs["stand"] > 1.0


def test_five_card_chase_beats_standing_into_a_sure_loss() -> None:
    """Hitting a four-card stiff toward a five-card win beats standing against a made dealer."""
    hand = [card(rank="2"), card(rank="3"), card(rank="5"), card(rank="6")]
    dealer = [card(rank="10"), card(rank="10")]
    action = recommend_action(
        hand_cards=hand,
        dealer_cards=dealer,
        shoe=build_shoe(rng=Random(x=0)),
        allowed_actions=("hit", "stand"),
        doubled=False,
    )
    evs = _evs(hand=hand, dealer=dealer, allowed=("hit", "stand"))

    assert action == "hit"
    assert evs["hit"] > evs["stand"]


def test_surrender_ev_is_minus_half_and_only_when_allowed() -> None:
    """Surrender is always exactly -0.5 and absent when not legal."""
    hand = [card(rank="10"), card(rank="6")]
    dealer = [card(rank="10"), card(rank="10")]
    with_surrender = _evs(hand=hand, dealer=dealer, allowed=("hit", "stand", "surrender"))
    without_surrender = _evs(hand=hand, dealer=dealer, allowed=("hit", "stand"))

    assert abs(with_surrender["surrender"] - (-0.5)) < 1e-9
    assert "surrender" not in without_surrender


def test_surrender_ev_uses_rounded_loss_for_odd_bets() -> None:
    """Surrender EV matches settle_hand's rounded half-bet loss for odd and tiny bets."""
    hand = [card(rank="10"), card(rank="6")]
    dealer = [card(rank="10"), card(rank="10")]
    one_point = _evs(hand=hand, dealer=dealer, allowed=("hit", "stand", "surrender"), bet=1)
    three_point = _evs(hand=hand, dealer=dealer, allowed=("hit", "stand", "surrender"), bet=3)

    assert abs(one_point["surrender"] - (-1.0)) < 1e-9
    assert abs(three_point["surrender"] - (-2 / 3)) < 1e-9


def test_split_can_be_recommended() -> None:
    """Splitting eights wins out against a weak dealer, past the split safety margin."""
    action = recommend_action(
        hand_cards=[card(rank="8"), card(rank="8")],
        dealer_cards=[card(rank="10"), card(rank="6")],
        shoe=build_shoe(rng=Random(x=0)),
        allowed_actions=("hit", "stand", "double", "split"),
        doubled=False,
    )

    assert action == "split"


def test_action_evs_only_cover_legal_actions() -> None:
    """The engine never prices, or recommends, an action outside allowed_actions."""
    hand = [card(rank="10"), card(rank="6")]
    dealer = [card(rank="9"), card(rank="7")]
    action = recommend_action(
        hand_cards=hand,
        dealer_cards=dealer,
        shoe=build_shoe(rng=Random(x=0)),
        allowed_actions=("hit", "stand"),
        doubled=False,
    )

    assert set(_evs(hand=hand, dealer=dealer, allowed=("hit", "stand"))) == {"hit", "stand"}
    assert action in {"hit", "stand"}


def test_empty_shoe_does_not_crash() -> None:
    """The engine degrades gracefully when the shoe is empty."""
    action = recommend_action(
        hand_cards=[card(rank="10"), card(rank="6")],
        dealer_cards=[card(rank="9"), card(rank="7")],
        shoe=[],
        allowed_actions=("hit", "stand"),
        doubled=False,
    )

    assert action in {"hit", "stand"}


def test_add_value_demotes_existing_ace_when_drawing_another_ace() -> None:
    """Soft 21 drawing an ace becomes hard 12, mirroring hand_value, not a 22 bust."""
    total, soft = _add_value(total=21, soft=True, bucket=9)

    assert (total, soft) == (12, False)


def test_hitting_soft_twenty_one_never_busts_into_a_five_card_win() -> None:
    """A four-card soft 21 always reaches a non-bust five-card hand, so hit EV is at least +1."""
    evs = _evs(
        hand=[card(rank="A"), card(rank="2"), card(rank="3"), card(rank="5")],
        dealer=[card(rank="10"), card(rank="9")],
        allowed=("hit", "stand"),
    )

    assert evs["hit"] >= 1.0


def test_shoe_value_counts_collapse_ten_values() -> None:
    """Ten, jack, queen, and king collapse into a single ten-value bucket."""
    counts = build_shoe_value_counts(
        shoe=[card(rank="10"), card(rank="J"), card(rank="Q"), card(rank="K"), card(rank="A")]
    )

    assert counts[8] == 4
    assert counts[9] == 1
    assert sum(counts) == 5


def test_compute_true_count_neutral_for_full_and_empty_shoe() -> None:
    """A balanced full shoe and an empty shoe both read as a neutral count of zero."""
    assert compute_true_count(shoe=build_shoe(rng=Random(x=0))) == 0.0
    assert compute_true_count(shoe=[]) == 0.0


def test_compute_true_count_positive_when_low_cards_are_gone() -> None:
    """A shoe drained of its low cards is ten-rich, which is a positive true count."""
    shoe = [
        card for card in build_shoe(rng=Random(x=0)) if card.rank not in ("2", "3", "4", "5", "6")
    ]
    true_count = compute_true_count(shoe=shoe)

    assert true_count > 0


def test_compute_true_count_negative_when_high_cards_are_gone() -> None:
    """A shoe drained of its ten-value cards and aces is low-rich, a negative count."""
    shoe = [
        card for card in build_shoe(rng=Random(x=0)) if card.rank not in ("10", "J", "Q", "K", "A")
    ]
    true_count = compute_true_count(shoe=shoe)

    assert true_count < 0
