"""Deterministic tests for the hole-card-aware Blackjack EV engine.

Pins `cogs/games/blackjack_ev.py`, the pure probability half of the bot player: the H17 dealer
final-total distribution, the per-action EVs in base-bet units, and the recommendation drawn from
them. Every case is exact arithmetic over a fixed shoe rather than a simulation, so an assertion
is a closed-form equality or a strict inequality between two priced actions instead of a sampling
tolerance, and a regression surfaces as a wrong number rather than as flakiness.

Three groups of behavior are worth pinning. The dealer model: H17 keeps drawing on soft 17 while
a hard 17 stands, and the six outcome probabilities always form a proper distribution. This
table's non-standard payouts: a five-card non-bust wins one unit outright, a five-card 21 adds the
system-funded bonus on top, and surrender is charged the rounded half-bet `settle_hand` actually
takes rather than a flat -0.5. And the information boundary that keeps the bot fair to play
against: the recommendation may use the dealer's hole card, but `dealer_outcome` and every exposed
action EV are marginalized over the remaining shoe alone, so two decisions sharing an up-card and
a shoe must expose exactly equal numbers however different their holes are.

The remainder covers what `bot_player.py` and `shoe.py` read out of the engine (the ten-value
bucket collapse, the sign of the Hi-Lo true count) plus the degraded inputs it must survive
without raising, since `bot_player.py` only falls back to basic strategy on an exception: an empty
shoe, and a soft hand drawing a second ace.
"""

from discordbot.typings.games import Card, DealerOutcome, ActionEvAnalysis
from discordbot.cogs.games.blackjack_ev import (
    _add_value,
    compute_action_evs,
    compute_true_count,
    build_shoe_value_counts,
    dealer_outcome_distribution,
)


def _card(rank: str) -> Card:
    """Builds a card of the given rank, with an arbitrary suit the engine never reads.

    Returns:
        A card the engine collapses onto its rank's value bucket.
    """
    return Card(rank=rank, suit="♠")


def _full_shoe() -> list[Card]:
    """Builds a fresh four-deck shoe as a flat card list.

    Returns:
        208 cards: four decks of four suits each, so every rank appears 16 times and the Hi-Lo
        count of the whole shoe is balanced at zero.
    """
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    return [_card(rank=rank) for rank in ranks] * 16


def _distribution_total(outcome: DealerOutcome) -> float:
    """Sums the six dealer-outcome probabilities.

    Returns:
        The total probability mass, which a well-formed distribution puts at 1.0.
    """
    return (
        outcome.bust_probability
        + outcome.total_17_probability
        + outcome.total_18_probability
        + outcome.total_19_probability
        + outcome.total_20_probability
        + outcome.total_21_probability
    )


def _ev_for(analysis: ActionEvAnalysis, action: str) -> float:
    """Reads one action's EV out of the analysis.

    Always the exposed marginal value, never the hole-aware one the recommendation came from,
    since `action_evs` only ever carries the marginal pass.

    Returns:
        The expected value in base-bet units.
    """
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
    """The six dealer outcome probabilities sum to one from every stiff and soft starting total."""
    shoe = build_shoe_value_counts(shoe=_full_shoe())
    for dealer_total, dealer_soft in ((12, False), (15, False), (16, False), (13, True)):
        outcome = dealer_outcome_distribution(
            dealer_total=dealer_total, dealer_soft=dealer_soft, shoe=shoe
        )
        assert abs(_distribution_total(outcome=outcome) - 1.0) < 1e-9


def test_standing_beats_hitting_on_hard_twenty() -> None:
    """A hard 20 against a weak dealer stands, and standing outranks hitting on EV too."""
    analysis = compute_action_evs(
        hand_cards=[_card(rank="10"), _card(rank="10")],
        dealer_cards=[_card(rank="9"), _card(rank="6")],
        shoe=_full_shoe(),
        allowed_actions=("hit", "stand"),
        doubled=False,
    )

    assert analysis.recommended_action == "stand"
    assert _ev_for(analysis=analysis, action="stand") > _ev_for(analysis=analysis, action="hit")


def test_recommendation_uses_hole_but_shown_distribution_hides_it() -> None:
    """A different hole moves the recommendation but leaves every exposed number untouched."""
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
    # Same up-card and shoe, so the exposed distribution must be exactly equal.
    assert weak.dealer_outcome == strong.dealer_outcome
    assert _ev_for(analysis=weak, action="stand") == _ev_for(analysis=strong, action="stand")


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
    """Surrender prices at exactly -0.5 with no bet given, and is absent once it is not legal."""
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


def test_surrender_ev_uses_rounded_loss_for_odd_bets() -> None:
    """Surrender EV matches `settle_hand`'s rounded half-bet loss for odd and tiny bets."""
    one_point = compute_action_evs(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        dealer_cards=[_card(rank="10"), _card(rank="10")],
        shoe=_full_shoe(),
        allowed_actions=("hit", "stand", "surrender"),
        doubled=False,
        bet=1,
    )
    three_point = compute_action_evs(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        dealer_cards=[_card(rank="10"), _card(rank="10")],
        shoe=_full_shoe(),
        allowed_actions=("hit", "stand", "surrender"),
        doubled=False,
        bet=3,
    )

    assert abs(_ev_for(analysis=one_point, action="surrender") - (-1.0)) < 1e-9
    assert abs(_ev_for(analysis=three_point, action="surrender") - (-2 / 3)) < 1e-9


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
    """The analysis never reports an EV for an action outside `allowed_actions`."""
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
    """An exhausted shoe still yields a legal recommendation instead of raising."""
    analysis = compute_action_evs(
        hand_cards=[_card(rank="10"), _card(rank="6")],
        dealer_cards=[_card(rank="9"), _card(rank="7")],
        shoe=[],
        allowed_actions=("hit", "stand"),
        doubled=False,
    )

    assert analysis.recommended_action in {"hit", "stand"}


def test_add_value_demotes_existing_ace_when_drawing_another_ace() -> None:
    """Soft 21 drawing an ace becomes hard 12, mirroring `hand_value`, not a 22 bust."""
    total, soft = _add_value(total=21, soft=True, bucket=9)

    assert (total, soft) == (12, False)


def test_hitting_soft_twenty_one_never_busts_into_a_five_card_win() -> None:
    """A four-card soft 21 always reaches a non-bust five-card hand, so hit EV is at least +1."""
    analysis = compute_action_evs(
        hand_cards=[_card(rank="A"), _card(rank="2"), _card(rank="3"), _card(rank="5")],
        dealer_cards=[_card(rank="10"), _card(rank="9")],
        shoe=_full_shoe(),
        allowed_actions=("hit", "stand"),
        doubled=False,
    )

    assert _ev_for(analysis=analysis, action="hit") >= 1.0


def test_shoe_value_counts_collapse_ten_values() -> None:
    """Ten, jack, queen, and king collapse into a single ten-value bucket."""
    counts = build_shoe_value_counts(
        shoe=[_card(rank="10"), _card(rank="J"), _card(rank="Q"), _card(rank="K"), _card(rank="A")]
    )

    assert counts[8] == 4
    assert counts[9] == 1
    assert sum(counts) == 5


def test_compute_true_count_neutral_for_full_and_empty_shoe() -> None:
    """A balanced full shoe and an empty shoe both read as a neutral count of zero."""
    assert compute_true_count(shoe=_full_shoe()) == 0.0
    assert compute_true_count(shoe=[]) == 0.0


def test_compute_true_count_positive_when_low_cards_are_gone() -> None:
    """A shoe drained of its low cards is ten-rich, which is a positive true count."""
    shoe = [card for card in _full_shoe() if card.rank not in ("2", "3", "4", "5", "6")]
    true_count = compute_true_count(shoe=shoe)

    assert true_count > 0


def test_compute_true_count_negative_when_high_cards_are_gone() -> None:
    """A shoe drained of its ten-value cards and aces is low-rich, a negative count."""
    shoe = [card for card in _full_shoe() if card.rank not in ("10", "J", "Q", "K", "A")]
    true_count = compute_true_count(shoe=shoe)

    assert true_count < 0
