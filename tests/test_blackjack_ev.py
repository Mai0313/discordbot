"""Deterministic tests for the hole-card-aware Blackjack EV engine."""

from discordbot.typings.games import Card
from discordbot.cogs._games.blackjack_ev import (
    compute_action_evs,
    build_shoe_value_counts,
    dealer_outcome_distribution,
)


def _card(rank: str) -> Card:
    """Builds a card with an arbitrary suit for EV tests."""
    return Card(rank=rank, suit="♠")


def _full_shoe() -> list[Card]:
    """Builds a fresh four-deck shoe as a flat card list."""
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    return [_card(rank=rank) for rank in ranks] * 4


def _distribution_total(outcome: object) -> float:
    """Sums the six dealer-outcome probabilities."""
    return (
        outcome.bust_probability
        + outcome.total_17_probability
        + outcome.total_18_probability
        + outcome.total_19_probability
        + outcome.total_20_probability
        + outcome.total_21_probability
    )


def _ev_for(analysis: object, action: str) -> float:
    """Returns the computed EV for a single action."""
    return next(item.expected_value for item in analysis.action_evs if item.action == action)


def test_dealer_hard_17_always_stands() -> None:
    """A hard 17 dealer stands with certainty regardless of the shoe."""
    outcome = dealer_outcome_distribution(
        dealer_total=17, dealer_soft=False, shoe=build_shoe_value_counts(shoe=_full_shoe())
    )

    assert outcome.total_17_probability == 1.0
    assert outcome.bust_probability == 0.0


def test_dealer_bust_probability_is_exact_on_a_tiny_shoe() -> None:
    """Dealer 16 over a shoe of one ten and one five busts or makes 21 with equal odds."""
    shoe = build_shoe_value_counts(shoe=[_card(rank="10"), _card(rank="5")])
    outcome = dealer_outcome_distribution(dealer_total=16, dealer_soft=False, shoe=shoe)

    assert abs(outcome.bust_probability - 0.5) < 1e-9
    assert abs(outcome.total_21_probability - 0.5) < 1e-9


def test_dealer_hits_soft_17_under_h17() -> None:
    """Soft 17 keeps drawing under H17, unlike a hard 17 that stands."""
    shoe = build_shoe_value_counts(shoe=_full_shoe())
    soft = dealer_outcome_distribution(dealer_total=17, dealer_soft=True, shoe=shoe)
    hard = dealer_outcome_distribution(dealer_total=17, dealer_soft=False, shoe=shoe)

    assert hard.total_17_probability == 1.0
    assert soft.total_17_probability < 1.0
    assert soft.bust_probability > 0.0


def test_dealer_distribution_sums_to_one() -> None:
    """The dealer outcome distribution is a proper probability distribution."""
    shoe = build_shoe_value_counts(shoe=_full_shoe())
    for dealer_total, dealer_soft in ((12, False), (15, False), (16, False), (13, True)):
        outcome = dealer_outcome_distribution(
            dealer_total=dealer_total, dealer_soft=dealer_soft, shoe=shoe
        )
        assert abs(_distribution_total(outcome=outcome) - 1.0) < 1e-9


def test_standing_beats_hitting_on_hard_twenty() -> None:
    """A hard 20 should stand, never hit, against a weak dealer."""
    analysis = compute_action_evs(
        hand_cards=[_card(rank="10"), _card(rank="10")],
        dealer_cards=[_card(rank="9"), _card(rank="6")],
        shoe=_full_shoe(),
        allowed_actions=("hit", "stand"),
        doubled=False,
    )

    assert analysis.recommended_action == "stand"
    assert _ev_for(analysis=analysis, action="stand") > _ev_for(analysis=analysis, action="hit")


def test_recommendation_depends_on_dealer_hole_card() -> None:
    """Hard 16 stands against a known weak dealer but surrenders against a known strong one."""
    shoe = _full_shoe()
    weak = compute_action_evs(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        dealer_cards=[_card(rank="5"), _card(rank="10")],
        shoe=shoe,
        allowed_actions=("hit", "stand", "surrender"),
        doubled=False,
    )
    strong = compute_action_evs(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        dealer_cards=[_card(rank="10"), _card(rank="10")],
        shoe=shoe,
        allowed_actions=("hit", "stand", "surrender"),
        doubled=False,
    )

    assert weak.recommended_action == "stand"
    assert strong.recommended_action == "surrender"
    assert weak.dealer_outcome.bust_probability > strong.dealer_outcome.bust_probability


def test_five_card_non_bust_stand_pays_one_unit() -> None:
    """A five-card non-bust hand wins one unit immediately, independent of the dealer."""
    analysis = compute_action_evs(
        hand_cards=[
            _card(rank="2"),
            _card(rank="3"),
            _card(rank="4"),
            _card(rank="4"),
            _card(rank="5"),
        ],
        dealer_cards=[_card(rank="10"), _card(rank="10")],
        shoe=_full_shoe(),
        allowed_actions=("stand",),
        doubled=False,
    )

    assert abs(_ev_for(analysis=analysis, action="stand") - 1.0) < 1e-9


def test_five_card_twenty_one_earns_the_bonus() -> None:
    """A five-card 21 is worth more than a normal win because of the system bonus."""
    analysis = compute_action_evs(
        hand_cards=[
            _card(rank="10"),
            _card(rank="5"),
            _card(rank="2"),
            _card(rank="3"),
            _card(rank="A"),
        ],
        dealer_cards=[_card(rank="10"), _card(rank="9")],
        shoe=_full_shoe(),
        allowed_actions=("stand",),
        doubled=False,
    )

    assert _ev_for(analysis=analysis, action="stand") > 1.0


def test_five_card_chase_beats_standing_into_a_sure_loss() -> None:
    """Hitting a four-card stiff toward a five-card win beats standing against a made dealer."""
    analysis = compute_action_evs(
        hand_cards=[_card(rank="2"), _card(rank="3"), _card(rank="5"), _card(rank="6")],
        dealer_cards=[_card(rank="10"), _card(rank="10")],
        shoe=_full_shoe(),
        allowed_actions=("hit", "stand"),
        doubled=False,
    )

    assert analysis.recommended_action == "hit"
    assert _ev_for(analysis=analysis, action="hit") > _ev_for(analysis=analysis, action="stand")


def test_surrender_ev_is_minus_half_and_only_when_allowed() -> None:
    """Surrender is always exactly -0.5 and absent when not legal."""
    with_surrender = compute_action_evs(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        dealer_cards=[_card(rank="10"), _card(rank="10")],
        shoe=_full_shoe(),
        allowed_actions=("hit", "stand", "surrender"),
        doubled=False,
    )
    without_surrender = compute_action_evs(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        dealer_cards=[_card(rank="10"), _card(rank="10")],
        shoe=_full_shoe(),
        allowed_actions=("hit", "stand"),
        doubled=False,
    )

    assert abs(_ev_for(analysis=with_surrender, action="surrender") - (-0.5)) < 1e-9
    assert all(item.action != "surrender" for item in without_surrender.action_evs)


def test_split_is_flagged_as_estimate_and_can_be_recommended() -> None:
    """Splitting eights is flagged as an estimate and wins out against a weak dealer."""
    analysis = compute_action_evs(
        hand_cards=[_card(rank="8"), _card(rank="8")],
        dealer_cards=[_card(rank="10"), _card(rank="6")],
        shoe=_full_shoe(),
        allowed_actions=("hit", "stand", "double", "split"),
        doubled=False,
    )
    split_ev = next(item for item in analysis.action_evs if item.action == "split")

    assert split_ev.is_estimate is True
    assert split_ev.note is not None
    assert analysis.recommended_action == "split"


def test_action_evs_only_cover_legal_actions() -> None:
    """The analysis never reports EV for an action outside allowed_actions."""
    analysis = compute_action_evs(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        dealer_cards=[_card(rank="9"), _card(rank="7")],
        shoe=_full_shoe(),
        allowed_actions=("hit", "stand"),
        doubled=False,
    )

    actions = {item.action for item in analysis.action_evs}
    assert actions == {"hit", "stand"}
    assert analysis.recommended_action in actions


def test_empty_shoe_does_not_crash() -> None:
    """The engine degrades gracefully when the shoe is empty."""
    analysis = compute_action_evs(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        dealer_cards=[_card(rank="9"), _card(rank="7")],
        shoe=[],
        allowed_actions=("hit", "stand"),
        doubled=False,
    )

    assert analysis.recommended_action in {"hit", "stand"}


def test_shoe_value_counts_collapse_ten_values() -> None:
    """Ten, jack, queen, and king collapse into a single ten-value bucket."""
    counts = build_shoe_value_counts(
        shoe=[_card(rank="10"), _card(rank="J"), _card(rank="Q"), _card(rank="K"), _card(rank="A")]
    )

    assert counts[8] == 4
    assert counts[9] == 1
    assert sum(counts) == 5
