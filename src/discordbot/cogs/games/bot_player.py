"""Deterministic decision logic for the Blackjack bot player.

The bot is a regular Blackjack player; the casino system is the dealer. Its
bet sizing, action, and insurance choices are computed without any LLM:
fractional-Kelly betting off the channel shoe's Hi-Lo true count, the hole-aware
EV engine for the action, and a count-based +EV rule for insurance.
"""

from typing import Final

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.games import Card, BotAction, ActionEvAnalysis
from discordbot.cogs.games.blackjack import is_soft_total, card_blackjack_value
from discordbot.cogs.games.blackjack_ev import compute_action_evs

# Per-round edge (at a neutral count) and variance of the bot's hole-aware optimal
# play, measured by offline simulation (neutral-count edge ~ +0.13, sigma^2 ~ 1.34).
# The edge is large because the EV engine plays the dealer hole card and this
# table's five-card rules are player-favorable; re-measure if those rules change.
BOT_TABLE_EDGE: Final[float] = 0.13
BOT_TABLE_VARIANCE: Final[float] = 1.34
# Half-Kelly keeps drawdown variance down; the hard fraction cap protects the
# bankroll even if the measured edge drifts.
BOT_KELLY_FRACTION: Final[float] = 0.5
BOT_MAX_BET_FRACTION: Final[float] = 0.10
# Edge added per +1 Hi-Lo true count when the shoe persists across rounds, used for
# count-based bet spreading. Measured at ~+0.0175 per true count by offline
# simulation with a persistent shoe, well above the standard Hi-Lo ~0.005 because
# this table's five-card rules amplify a ten-rich shoe. Re-measure if the rules
# change.
BOT_EDGE_PER_TRUE_COUNT: Final[float] = 0.0175
_RANK_ORDER: Final[tuple[str, ...]] = (
    "A",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "J",
    "Q",
    "K",
)
_TEN_VALUE_RANKS: Final[frozenset[str]] = frozenset({"10", "J", "Q", "K"})
_LOW_RANKS: Final[frozenset[str]] = frozenset({"2", "3", "4", "5", "6"})
_NEUTRAL_RANKS: Final[frozenset[str]] = frozenset({"7", "8", "9"})
_PAIR_SPLIT_DEALERS: Final[dict[int, frozenset[int]]] = {
    11: frozenset(range(2, 12)),
    8: frozenset(range(2, 12)),
    9: frozenset({2, 3, 4, 5, 6, 8, 9}),
    7: frozenset(range(2, 8)),
    6: frozenset(range(2, 7)),
    4: frozenset({5, 6}),
    3: frozenset(range(2, 8)),
    2: frozenset(range(2, 8)),
}


class ShoeSummary(BaseModel):
    """Rank-level summary of the true remaining Blackjack shoe."""

    model_config = ConfigDict(frozen=True)

    total_cards: int = Field(..., description="Total cards left in the true remaining shoe.")
    rank_counts: dict[str, int] = Field(
        ..., description="Remaining count per card rank in stable Blackjack rank order."
    )
    ace_count: int = Field(..., description="Number of aces left in the remaining shoe.")
    ten_value_count: int = Field(
        ..., description="Number of ten-value cards (10/J/Q/K) left in the remaining shoe."
    )
    low_card_count: int = Field(
        ..., description="Number of low cards (2-6) left in the remaining shoe."
    )
    neutral_card_count: int = Field(
        ..., description="Number of neutral cards (7-9) left in the remaining shoe."
    )
    high_card_count: int = Field(
        ..., description="Number of high cards (aces plus ten-value) left in the remaining shoe."
    )


class DealerKnowledge(BaseModel):
    """Dealer state exposed to the bot player.

    Only the up-card is shared, exactly what any seated player sees. The hole
    card, the combined two-card total, and the natural-Blackjack flag are
    deliberately withheld so nothing carried here can reveal the hole.
    """

    model_config = ConfigDict(frozen=True)

    up_card: str = Field(
        ..., description="The dealer's face-up card, exactly what a seated player sees."
    )
    up_value: int = Field(
        ..., description="Blackjack value of the dealer up-card (ace counts as 11)."
    )


class ActionAnalysis(BaseModel):
    """Computed reference data for the bot player's action decision."""

    model_config = ConfigDict(frozen=True)

    allowed_actions: tuple[BotAction, ...] = Field(
        ..., description="Actions the bot is legally allowed to take this turn."
    )
    basic_strategy_action: BotAction = Field(
        ..., description="The deterministic hint action the bot would play this turn."
    )
    ev_analysis: ActionEvAnalysis | None = Field(
        default=None,
        description="Per-action EV analysis from the EV engine, or None when unavailable.",
    )


class BotPlayerActionContext(BaseModel):
    """Complete computed context for one bot-player action decision."""

    model_config = ConfigDict(frozen=True)

    shoe_summary: ShoeSummary = Field(
        ..., description="Rank-level summary of the true remaining shoe."
    )
    dealer: DealerKnowledge = Field(
        ..., description="Up-card-only dealer state visible to the bot."
    )
    action_analysis: ActionAnalysis = Field(
        ..., description="Computed reference data for the bot's action decision."
    )


class BotPlayerInsuranceContext(BaseModel):
    """Computed context for one bot-player insurance decision.

    Insurance is priced from the remaining-shoe ten-value density (card
    counting), not the hole card, so the recommendation never reveals whether
    the dealer actually has Blackjack.
    """

    model_config = ConfigDict(frozen=True)

    shoe_summary: ShoeSummary = Field(
        ..., description="Rank-level summary of the true remaining shoe."
    )
    dealer: DealerKnowledge = Field(
        ..., description="Up-card-only dealer state visible to the bot."
    )
    insurance_cost: int = Field(
        ..., description="Cost in currency to take the insurance side bet."
    )
    ten_value_probability: float = Field(
        ..., description="Ten-value card fraction of the remaining shoe used to price insurance."
    )
    insurance_expected_value: float = Field(
        ...,
        description="Expected value in currency of taking insurance at the current shoe density.",
    )
    insurance_recommendation: str = Field(
        ...,
        description="Deterministic recommendation, 'take' or 'decline', from the shoe density.",
    )


_HARD_DOUBLE_DEALERS: Final[dict[int, frozenset[int]]] = {
    9: frozenset({3, 4, 5, 6}),
    10: frozenset(range(2, 10)),
    11: frozenset(range(2, 12)),
}
_SOFT_DOUBLE_DEALERS: Final[dict[int, frozenset[int]]] = {
    13: frozenset({5, 6}),
    14: frozenset({5, 6}),
    15: frozenset({4, 5, 6}),
    16: frozenset({4, 5, 6}),
    17: frozenset({3, 4, 5, 6}),
    18: frozenset(range(2, 7)),
}


def _dealer_up_value(*, up_card: Card | None) -> int:
    """Returns the Blackjack value of the dealer's up-card (A counts as 11)."""
    if up_card is None:
        return 0
    return card_blackjack_value(card=up_card)


def _pair_value(*, cards: list[Card]) -> int | None:
    """Returns the pair value for same-value two-card hands."""
    if len(cards) != 2:
        return None
    first = card_blackjack_value(card=cards[0])
    second = card_blackjack_value(card=cards[1])
    return first if first == second else None


def _should_surrender(*, hand_total: int, dealer_value: int) -> bool:
    """Returns whether late surrender is the fallback table choice."""
    return (hand_total == 16 and dealer_value in {9, 10, 11}) or (
        hand_total == 15 and dealer_value == 10
    )


def _should_double(*, cards: list[Card], hand_total: int, dealer_value: int) -> bool:
    """Returns whether double down is the fallback table choice."""
    is_soft, _total = is_soft_total(cards=cards)
    double_dealers = (
        _SOFT_DOUBLE_DEALERS.get(hand_total, frozenset())
        if is_soft
        else _HARD_DOUBLE_DEALERS.get(hand_total, frozenset())
    )
    return dealer_value in double_dealers


def _should_stand(*, cards: list[Card], hand_total: int, dealer_value: int) -> bool:
    """Returns whether stand is the fallback table choice."""
    is_soft, _total = is_soft_total(cards=cards)
    if is_soft:
        return hand_total >= 19 or (hand_total == 18 and 2 <= dealer_value <= 8)
    return (
        hand_total >= 17
        or (13 <= hand_total <= 16 and dealer_value <= 6)
        or (hand_total == 12 and 4 <= dealer_value <= 6)
    )


def _rank_counts(*, cards: list[Card]) -> dict[str, int]:
    """Counts card ranks in stable Blackjack rank order."""
    counts = dict.fromkeys(_RANK_ORDER, 0)
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return counts


def build_shoe_summary(*, shoe: list[Card]) -> ShoeSummary:
    """Builds rank-count context from the true remaining shoe."""
    counts = _rank_counts(cards=shoe)
    ace_count = counts["A"]
    ten_value_count = sum(counts[rank] for rank in _TEN_VALUE_RANKS)
    low_card_count = sum(counts[rank] for rank in _LOW_RANKS)
    neutral_card_count = sum(counts[rank] for rank in _NEUTRAL_RANKS)
    return ShoeSummary(
        total_cards=len(shoe),
        rank_counts=counts,
        ace_count=ace_count,
        ten_value_count=ten_value_count,
        low_card_count=low_card_count,
        neutral_card_count=neutral_card_count,
        high_card_count=ace_count + ten_value_count,
    )


def build_dealer_knowledge(*, dealer_up: Card | None) -> DealerKnowledge:
    """Builds the up-card-only dealer state exposed to the bot player."""
    return DealerKnowledge(
        up_card=str(dealer_up) if dealer_up is not None else "unknown",
        up_value=_dealer_up_value(up_card=dealer_up),
    )


def kelly_bet(  # noqa: PLR0913 -- exposes the Kelly tuning knobs (fraction, cap) as overridable args.
    *,
    balance: int,
    table_minimum: int,
    edge: float = BOT_TABLE_EDGE,
    variance: float = BOT_TABLE_VARIANCE,
    kelly_fraction: float = BOT_KELLY_FRACTION,
    max_fraction: float = BOT_MAX_BET_FRACTION,
) -> int:
    """Returns the fractional-Kelly wager from the per-round edge.

    The growth-optimal stake is a fraction of the bankroll set by the edge. With
    a fresh shoe the edge is the constant `BOT_TABLE_EDGE`; with a persistent shoe
    it is `count_adjusted_edge(...)` so the bot spreads its bet by true count.

    `max_fraction` of the bankroll is a hard ceiling, not merely a cap on the Kelly
    fraction: the owner-chosen table stake floors the bet only up to that ceiling,
    so a large table stake can no longer drag the bot past its risk limit. The bot
    still sits at any table, but it never wagers more than `max_fraction` of its
    balance in one round. A non-positive edge falls back to that capped table floor
    instead of refusing to play.

    Args:
        balance: The bot's spendable balance.
        table_minimum: The table stake the bot matches, up to the bankroll ceiling.
        edge: Per-round expected value in base-bet units.
        variance: Per-round variance in base-bet units.
        kelly_fraction: Fraction of full Kelly to apply (0.5 is half-Kelly).
        max_fraction: Hard ceiling on the bankroll fraction wagered in one round.

    Returns:
        A positive integer wager within `[1, max_fraction * balance]`, never above
        `balance`. A non-positive balance returns 1, the one case above `balance`.
    """
    if balance <= 0:
        return 1
    # The bankroll fraction is a hard ceiling: the bot never risks more than
    # `max_fraction` of its balance in one round, even when the owner-chosen table
    # stake is larger. The stake only floors the bet up to this ceiling so the bot
    # still sits; it can no longer be dragged past its Kelly risk limit.
    ceiling = max(1, min(round(max_fraction * balance), balance))
    floor = max(1, min(table_minimum, ceiling))
    if edge <= 0 or variance <= 0:
        return floor
    fraction = min(max(kelly_fraction * edge / variance, 0.0), max_fraction)
    wager = round(fraction * balance)
    return max(floor, min(wager, ceiling))


def count_adjusted_edge(*, true_count: float) -> float:
    """Returns the per-round edge adjusted for the Hi-Lo true count.

    A persistent shoe lets the bot read a true count before betting; a positive
    count means the remaining shoe is rich in ten-value cards and aces, which lifts
    the edge. The slope is measured against this table's five-card rules, so the
    bot meaningfully spreads its wager toward favorable counts.
    """
    return BOT_TABLE_EDGE + BOT_EDGE_PER_TRUE_COUNT * true_count


def _safe_compute_action_evs(  # noqa: PLR0913 -- thin EV-engine wrapper mirroring its signature.
    *,
    hand_cards: list[Card],
    dealer_cards: list[Card],
    shoe: list[Card],
    allowed_actions: tuple[BotAction, ...],
    doubled: bool,
    bet: int | None = None,
) -> ActionEvAnalysis | None:
    """Runs the EV engine, returning None on any failure so a bot turn never crashes."""
    try:
        return compute_action_evs(
            hand_cards=hand_cards,
            dealer_cards=dealer_cards,
            shoe=shoe,
            allowed_actions=allowed_actions,
            doubled=doubled,
            bet=bet,
        )
    except Exception as exc:
        logfire.warn(
            "Bot EV engine failed; falling back to basic strategy",
            allowed_actions=allowed_actions,
            hand_ranks=[card.rank for card in hand_cards],
            dealer_ranks=[card.rank for card in dealer_cards],
            shoe_size=len(shoe),
            doubled=doubled,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
        return None


def fallback_action(
    *,
    hand_cards: list[Card],
    hand_total: int,
    dealer_up: Card | None,
    is_pair_hand: bool,
    allowed_actions: tuple[BotAction, ...],
) -> BotAction:
    """Classic up-card-only basic-strategy table, used when the EV engine is unavailable.

    Only emits actions listed in `allowed_actions`.
    """
    dealer_value = _dealer_up_value(up_card=dealer_up)
    pair_value = _pair_value(cards=hand_cards) if is_pair_hand else None
    if (
        pair_value is not None
        and "split" in allowed_actions
        and dealer_value in _PAIR_SPLIT_DEALERS.get(pair_value, frozenset())
    ):
        return "split"
    if "surrender" in allowed_actions and _should_surrender(
        hand_total=hand_total, dealer_value=dealer_value
    ):
        return "surrender"
    if "double" in allowed_actions and _should_double(
        cards=hand_cards, hand_total=hand_total, dealer_value=dealer_value
    ):
        return "double"
    if "stand" in allowed_actions and _should_stand(
        cards=hand_cards, hand_total=hand_total, dealer_value=dealer_value
    ):
        return "stand"
    if "hit" in allowed_actions:
        return "hit"
    return allowed_actions[0]


def build_bot_action_context(  # noqa: PLR0913 -- context builder mirrors the full decision surface.
    *,
    hand_cards: list[Card],
    dealer_cards: list[Card],
    dealer_up: Card | None,
    shoe: list[Card],
    allowed_actions: tuple[BotAction, ...],
    is_pair_hand: bool,
    bet: int,
    doubled: bool = False,
) -> BotPlayerActionContext:
    """Builds the bot's computed decision context without exposing the future shoe order.

    `action_analysis.basic_strategy_action` is the action the bot plays: the EV
    engine's hole-aware recommendation, or the basic-strategy table when the
    engine is unavailable.
    """
    ev_analysis = _safe_compute_action_evs(
        hand_cards=hand_cards,
        dealer_cards=dealer_cards,
        shoe=shoe,
        allowed_actions=allowed_actions,
        doubled=doubled,
        bet=bet,
    )
    if ev_analysis is not None:
        basic_strategy_action = ev_analysis.recommended_action
    else:
        basic_strategy_action = fallback_action(
            hand_cards=hand_cards,
            hand_total=is_soft_total(cards=hand_cards)[1],
            dealer_up=dealer_up,
            is_pair_hand=is_pair_hand,
            allowed_actions=allowed_actions,
        )
    return BotPlayerActionContext(
        shoe_summary=build_shoe_summary(shoe=shoe),
        dealer=build_dealer_knowledge(dealer_up=dealer_up),
        action_analysis=ActionAnalysis(
            allowed_actions=allowed_actions,
            basic_strategy_action=basic_strategy_action,
            ev_analysis=ev_analysis,
        ),
    )


def build_bot_insurance_context(
    *, dealer_up: Card | None, shoe: list[Card], insurance_cost: int
) -> BotPlayerInsuranceContext:
    """Builds insurance context from the remaining-shoe ten density only.

    The dealer hole card is never passed in, so it cannot reach the decision.
    Insurance pays only on a ten-value hole, so the remaining shoe's
    ten-value fraction is the fair probability a counter would use. Insurance is
    +EV only when that fraction clears 1/3; the probability matches the exposed
    shoe counts exactly, leaving nothing to cross-solve.
    """
    ten_count = sum(1 for card in shoe if card.rank in _TEN_VALUE_RANKS)
    total = len(shoe)
    ten_probability = ten_count / total if total > 0 else 0.0
    # Take pays +2x cost on a ten hole, loses cost otherwise: EV = cost*(3p - 1),
    # so it only turns positive once ten-value density clears one third.
    break_even = 1.0 / 3.0
    expected_value = insurance_cost * (3.0 * ten_probability - 1.0)
    recommendation = "take" if ten_probability > break_even else "decline"
    return BotPlayerInsuranceContext(
        shoe_summary=build_shoe_summary(shoe=shoe),
        dealer=build_dealer_knowledge(dealer_up=dealer_up),
        insurance_cost=insurance_cost,
        ten_value_probability=ten_probability,
        insurance_expected_value=expected_value,
        insurance_recommendation=recommendation,
    )


def fallback_insurance(*, insurance_context: BotPlayerInsuranceContext | None = None) -> bool:
    """Count-based insurance decision: take only when the unseen deck makes it +EV.

    This is the bot's authoritative insurance choice, not just a failure
    fallback: insurance is +EV only when the remaining-shoe ten density clears
    one third, so the deterministic count drives the decision.
    """
    if insurance_context is None:
        return False
    return insurance_context.insurance_recommendation == "take"
