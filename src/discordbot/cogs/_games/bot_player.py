"""AI that decides the bot player's bet, action, and insurance choices.

The bot is a regular Blackjack player; the casino system is the dealer. This
module mirrors `SystemNarrator` shape (BaseModel + AsyncOpenAI client +
ModelSettings) and provides deterministic fallbacks so a slow or failing LLM
never blocks the bot's turn at the table.

Decision-time context is verbose by design: the LLM sees its lifetime balance,
today's casino loss/win/net, this round's bet and remaining wallet, every
other player's hands plus bets, and its own other split hands. The goal is to
let the model reason from the actual table state rather than fall back on a
prescriptive script.
"""

from typing import Final

from openai import AsyncOpenAI
import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.utils.llm import parse_responses_or_none
from discordbot.typings.games import (
    Card,
    BotAction,
    OtherPlayerView,
    ActionEvAnalysis,
    BotFinancialContext,
    BotPlayerActionDecision,
    BotPlayerInsuranceDecision,
)
from discordbot.typings.models import ModelSettings
from discordbot.cogs._games.prompts import BOT_PLAYER_ACTION_PROMPT, BOT_PLAYER_INSURANCE_PROMPT
from discordbot.cogs._games.blackjack import is_soft_total, _card_blackjack_value
from discordbot.cogs._games.blackjack_ev import compute_action_evs
from discordbot.cogs._economy.presentation import CURRENCY_NAME

BOT_ACTION_AI_TIMEOUT_SECONDS = 30.0
BOT_INSURANCE_AI_TIMEOUT_SECONDS = 30.0
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
# Bot decisions are system-side LLM calls. ASCII labels per method let LiteLLM
# telemetry split bet / action / insurance traffic, mirroring the
# `auto_unmute.py` / `_stock/news.py` / `prompt_dev.py` pattern.
_ACTION_END_USER_ID: Final[str] = "bot_player_action"
_INSURANCE_END_USER_ID: Final[str] = "bot_player_insurance"
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
_INFO_BOUNDARY: Final[str] = (
    "server_true_remaining_shoe_counts_and_dealer_up_card; no hole card, "
    "no next-card field, and no ordered future shoe"
)
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
    """Dealer state exposed to the bot player AI.

    Only the up-card is shared, exactly what any seated player sees. The hole
    card, the combined two-card total, and the natural-Blackjack flag are
    deliberately withheld so nothing the model receives can reveal the hole.
    """

    model_config = ConfigDict(frozen=True)

    up_card: str = Field(
        ..., description="The dealer's face-up card, exactly what a seated player sees."
    )
    up_value: int = Field(
        ..., description="Blackjack value of the dealer up-card (ace counts as 11)."
    )


class DrawOdds(BaseModel):
    """One-card draw probabilities derived from current hand plus rank counts."""

    model_config = ConfigDict(frozen=True)

    total_draws: int = Field(
        ..., description="Number of possible next cards (size of the remaining shoe)."
    )
    bust_probability: float = Field(
        ..., description="Probability the next single card busts the hand."
    )
    twenty_one_probability: float = Field(
        ..., description="Probability the next single card makes the hand total 21."
    )
    seventeen_to_twenty_one_probability: float = Field(
        ..., description="Probability the next single card leaves a total between 17 and 21."
    )
    five_card_non_bust_probability: float = Field(
        ..., description="Probability the next card reaches a five-plus-card hand without busting."
    )
    five_card_twenty_one_probability: float = Field(
        ..., description="Probability the next card reaches a five-plus-card hand totaling 21."
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
    basic_strategy_reason: str = Field(
        ..., description="Short English explanation of the basic-strategy hint action."
    )
    ev_analysis: ActionEvAnalysis | None = Field(
        default=None,
        description="Per-action EV analysis from the EV engine, or None when unavailable.",
    )
    hit_odds: DrawOdds | None = Field(
        default=None,
        description="One-card draw odds for hitting, or None when hit is not allowed.",
    )
    double_odds: DrawOdds | None = Field(
        default=None,
        description="One-card draw odds for doubling, or None when double is not allowed.",
    )
    stand_summary: str | None = Field(
        default=None,
        description="Human-readable summary of standing, or None when not applicable.",
    )
    split_summary: str | None = Field(
        default=None, description="Human-readable summary of splitting, or None when not allowed."
    )
    surrender_summary: str | None = Field(
        default=None,
        description="Human-readable summary of surrendering, or None when not allowed.",
    )


class BotPlayerActionContext(BaseModel):
    """Complete computed context for one bot-player action decision."""

    model_config = ConfigDict(frozen=True)

    information_boundary: str = Field(
        ...,
        description="Text describing exactly which table information the bot is allowed to see.",
    )
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

    information_boundary: str = Field(
        ...,
        description="Text describing exactly which table information the bot is allowed to see.",
    )
    shoe_summary: ShoeSummary = Field(
        ..., description="Rank-level summary of the true remaining shoe."
    )
    dealer: DealerKnowledge = Field(
        ..., description="Up-card-only dealer state visible to the bot."
    )
    insurance_cost: int = Field(
        ..., description="Cost in currency to take the insurance side bet."
    )
    insurance_payout: int = Field(..., description="Payout in currency if the insurance bet wins.")
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
    summary: str = Field(
        ..., description="Human-readable summary of the insurance pricing analysis."
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
    if up_card.rank == "A":
        return 11
    if up_card.rank in ("J", "Q", "K"):
        return 10
    return int(up_card.rank)


def _hand_total_and_soft(*, cards: list[Card]) -> tuple[int, bool]:
    """Returns the best total and whether at least one Ace remains high."""
    is_soft, total = is_soft_total(cards=cards)
    return total, is_soft


def _pair_value(*, cards: list[Card]) -> int | None:
    """Returns the pair value for same-value two-card hands."""
    if len(cards) != 2:
        return None
    first = _card_blackjack_value(card=cards[0])
    second = _card_blackjack_value(card=cards[1])
    return first if first == second else None


def _should_surrender(*, hand_total: int, dealer_value: int) -> bool:
    """Returns whether late surrender is the fallback table choice."""
    return (hand_total == 16 and dealer_value in {9, 10, 11}) or (
        hand_total == 15 and dealer_value == 10
    )


def _should_double(*, cards: list[Card], hand_total: int, dealer_value: int) -> bool:
    """Returns whether double down is the fallback table choice."""
    _, is_soft = _hand_total_and_soft(cards=cards)
    double_dealers = (
        _SOFT_DOUBLE_DEALERS.get(hand_total, frozenset())
        if is_soft
        else _HARD_DOUBLE_DEALERS.get(hand_total, frozenset())
    )
    return dealer_value in double_dealers


def _should_stand(*, cards: list[Card], hand_total: int, dealer_value: int) -> bool:
    """Returns whether stand is the fallback table choice."""
    _, is_soft = _hand_total_and_soft(cards=cards)
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
    """Builds the up-card-only dealer state exposed to the bot player AI."""
    return DealerKnowledge(
        up_card=str(dealer_up) if dealer_up is not None else "unknown",
        up_value=_dealer_up_value(up_card=dealer_up),
    )


def _probability(*, count: int, total: int) -> float:
    """Returns a zero-safe probability in the range [0.0, 1.0]."""
    if total <= 0:
        return 0.0
    return count / total


def _draw_odds(*, hand_cards: list[Card], shoe: list[Card], doubled: bool) -> DrawOdds:
    """Computes one-card draw odds from rank counts, not shoe order."""
    counts = _rank_counts(cards=shoe)
    total_draws = len(shoe)
    busts = 0
    twenty_ones = 0
    strong_totals = 0
    five_card_non_busts = 0
    five_card_twenty_ones = 0
    for rank, count in counts.items():
        if count <= 0:
            continue
        drawn = Card(rank=rank, suit="♠")
        next_cards = [*hand_cards, drawn]
        next_total = _hand_total_and_soft(cards=next_cards)[0]
        if next_total > 21:
            busts += count
        if next_total == 21:
            twenty_ones += count
        if 17 <= next_total <= 21:
            strong_totals += count
        if not doubled and len(next_cards) >= 5 and next_total <= 21:
            five_card_non_busts += count
        if not doubled and len(next_cards) >= 5 and next_total == 21:
            five_card_twenty_ones += count
    return DrawOdds(
        total_draws=total_draws,
        bust_probability=_probability(count=busts, total=total_draws),
        twenty_one_probability=_probability(count=twenty_ones, total=total_draws),
        seventeen_to_twenty_one_probability=_probability(count=strong_totals, total=total_draws),
        five_card_non_bust_probability=_probability(count=five_card_non_busts, total=total_draws),
        five_card_twenty_one_probability=_probability(
            count=five_card_twenty_ones, total=total_draws
        ),
    )


def _basic_strategy_reason(*, action: BotAction) -> str:
    """Returns a compact English reason for the fallback action hint."""
    return f"Deterministic fallback table would choose {action}; use as a hint, not a hard rule."


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
        `balance`.
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
    except Exception:
        logfire.warn("Bot EV engine failed; falling back to basic strategy", _exc_info=True)
        return None


def _basic_strategy_table_action(
    *,
    hand_cards: list[Card],
    hand_total: int,
    dealer_up: Card | None,
    is_pair_hand: bool,
    allowed_actions: tuple[BotAction, ...],
) -> BotAction:
    """Classic up-card-only basic-strategy table, used when the EV engine is unavailable."""
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


def fallback_action(  # noqa: PLR0913 -- hole-card-aware fallback also accepts dealer cards and shoe.
    *,
    hand_cards: list[Card],
    hand_total: int,
    dealer_up: Card | None,
    is_pair_hand: bool,
    allowed_actions: tuple[BotAction, ...],
    dealer_cards: list[Card] | None = None,
    shoe: list[Card] | None = None,
) -> BotAction:
    """Deterministic fallback that only emits allowed actions.

    When the full dealer cards and remaining shoe are supplied, the exact EV
    engine drives the choice (hole-card-aware). Otherwise it degrades to the
    classic up-card-only basic-strategy table.
    """
    if dealer_cards is not None and shoe is not None:
        analysis = _safe_compute_action_evs(
            hand_cards=hand_cards,
            dealer_cards=dealer_cards,
            shoe=shoe,
            allowed_actions=allowed_actions,
            doubled=False,
        )
        if analysis is not None and analysis.recommended_action in allowed_actions:
            return analysis.recommended_action
    return _basic_strategy_table_action(
        hand_cards=hand_cards,
        hand_total=hand_total,
        dealer_up=dealer_up,
        is_pair_hand=is_pair_hand,
        allowed_actions=allowed_actions,
    )


def build_bot_action_context(  # noqa: PLR0913 -- context builder mirrors the full decision surface.
    *,
    hand_cards: list[Card],
    dealer_cards: list[Card],
    dealer_up: Card | None,
    shoe: list[Card],
    allowed_actions: tuple[BotAction, ...],
    is_pair_hand: bool,
    bet: int,
    balance_remaining: int,
    doubled: bool = False,
) -> BotPlayerActionContext:
    """Builds computed AI context without exposing the future shoe order."""
    hand_total, _is_soft = _hand_total_and_soft(cards=hand_cards)
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
        basic_strategy_reason = (
            "EV-max legal action given the dealer up-card and remaining shoe; "
            f"expected_value={ev_analysis.recommended_expected_value:+.2f} base bets."
        )
    else:
        basic_strategy_action = fallback_action(
            hand_cards=hand_cards,
            hand_total=hand_total,
            dealer_up=dealer_up,
            is_pair_hand=is_pair_hand,
            allowed_actions=allowed_actions,
        )
        basic_strategy_reason = _basic_strategy_reason(action=basic_strategy_action)
    dealer = build_dealer_knowledge(dealer_up=dealer_up)
    hit_odds = (
        _draw_odds(hand_cards=hand_cards, shoe=shoe, doubled=doubled)
        if "hit" in allowed_actions
        else None
    )
    double_odds = (
        _draw_odds(hand_cards=hand_cards, shoe=shoe, doubled=True)
        if "double" in allowed_actions
        else None
    )
    pair_value = _pair_value(cards=hand_cards)
    split_summary = None
    if "split" in allowed_actions:
        split_summary = (
            f"Pair value {pair_value}; split costs an extra {bet} {CURRENCY_NAME}; "
            "double after split is not allowed; split Aces receive one card and stand."
        )
    surrender_summary = None
    if "surrender" in allowed_actions:
        surrender_summary = (
            f"Surrender locks in a half-bet loss of {(bet + 1) // 2} {CURRENCY_NAME}."
        )
    return BotPlayerActionContext(
        information_boundary=_INFO_BOUNDARY,
        shoe_summary=build_shoe_summary(shoe=shoe),
        dealer=dealer,
        action_analysis=ActionAnalysis(
            allowed_actions=allowed_actions,
            basic_strategy_action=basic_strategy_action,
            basic_strategy_reason=basic_strategy_reason,
            ev_analysis=ev_analysis,
            hit_odds=hit_odds,
            double_odds=double_odds,
            stand_summary=(
                f"Standing leaves active hand total {hand_total} against dealer up-card "
                f"{dealer.up_card} (value {dealer.up_value}); uncommitted balance after current "
                f"wagers: {balance_remaining} {CURRENCY_NAME}."
            ),
            split_summary=split_summary,
            surrender_summary=surrender_summary,
        ),
    )


def choose_bot_action(  # noqa: PLR0913 -- deterministic action picker mirrors the full decision surface.
    *,
    action_context: BotPlayerActionContext | None,
    hand_cards: list[Card],
    hand_total: int,
    dealer_up: Card | None,
    is_pair_hand: bool,
    allowed_actions: tuple[BotAction, ...],
) -> BotAction:
    """Returns the deterministic action the bot plays this turn.

    The action is the EV engine's hole-aware recommendation carried on the
    action context (always one of `allowed_actions`); it degrades to the
    up-card-only basic-strategy fallback only when the context is missing. The
    LLM never chooses the action; it only narrates the reason afterwards.
    """
    if action_context is not None:
        return action_context.action_analysis.basic_strategy_action
    return fallback_action(
        hand_cards=hand_cards,
        hand_total=hand_total,
        dealer_up=dealer_up,
        is_pair_hand=is_pair_hand,
        allowed_actions=allowed_actions,
    )


def action_decision_reason(*, action_context: BotPlayerActionContext | None) -> str:
    """Returns the instant Traditional Chinese reason shown before LLM narration."""
    if action_context is not None and action_context.action_analysis.ev_analysis is not None:
        return (
            f"EV {action_context.action_analysis.ev_analysis.recommended_expected_value:+.2f} 最佳"
        )
    return "基本策略最佳"


def build_bot_insurance_context(
    *, dealer_up: Card | None, shoe: list[Card], insurance_cost: int
) -> BotPlayerInsuranceContext:
    """Builds insurance context from the remaining-shoe ten density only.

    The dealer hole card is never passed in, so it cannot reach the decision or
    the prompt. Insurance pays only on a ten-value hole, so the remaining shoe's
    ten-value fraction is the fair probability a counter would use. Insurance is
    +EV only when that fraction clears 1/3; the probability matches the exposed
    shoe counts exactly, leaving nothing to cross-solve.
    """
    ten_count = sum(1 for card in shoe if card.rank in _TEN_VALUE_RANKS)
    total = len(shoe)
    ten_probability = ten_count / total if total > 0 else 0.0
    insurance_payout = insurance_cost * 2
    # Take pays +2x cost on a ten hole, loses cost otherwise: EV = cost*(3p - 1),
    # so it only turns positive once ten-value density clears one third.
    break_even = 1.0 / 3.0
    expected_value = insurance_cost * (3.0 * ten_probability - 1.0)
    recommendation = "take" if ten_probability > break_even else "decline"
    return BotPlayerInsuranceContext(
        information_boundary=_INFO_BOUNDARY,
        shoe_summary=build_shoe_summary(shoe=shoe),
        dealer=build_dealer_knowledge(dealer_up=dealer_up),
        insurance_cost=insurance_cost,
        insurance_payout=insurance_payout,
        ten_value_probability=ten_probability,
        insurance_expected_value=expected_value,
        insurance_recommendation=recommendation,
        summary=(
            "Insurance pays only on a ten-value hole; estimated ten-value probability "
            f"{ten_probability * 100:.1f}% from the remaining shoe; +EV only above 33.3%."
        ),
    )


def fallback_insurance(*, insurance_context: BotPlayerInsuranceContext | None = None) -> bool:
    """Count-based insurance decision: take only when the unseen deck makes it +EV.

    This is the bot's authoritative insurance choice, not just a failure
    fallback: insurance is +EV only when the remaining-shoe ten density clears
    one third, so the deterministic count drives the decision and the LLM only
    narrates it.
    """
    if insurance_context is None:
        return False
    return insurance_context.insurance_recommendation == "take"


def insurance_decision_reason(
    *, take_insurance: bool, insurance_context: BotPlayerInsuranceContext | None
) -> str:
    """Returns the instant Traditional Chinese reason shown before LLM narration."""
    if insurance_context is None:
        return "無牌堆資料, 不買保險"
    ten_percent = insurance_context.ten_value_probability * 100
    if take_insurance:
        return f"牌堆十點 {ten_percent:.0f}%, 保險划算"
    return f"牌堆十點僅 {ten_percent:.0f}%, 不買"


def _format_percent(value: float) -> str:
    """Formats a probability as one decimal-place percentage."""
    return f"{value * 100:.1f}%"


def _format_rank_counts(rank_counts: dict[str, int]) -> str:
    """Formats stable rank counts for prompt context."""
    return ", ".join(f"{rank}:{rank_counts.get(rank, 0)}" for rank in _RANK_ORDER)


def _format_draw_odds(*, label: str, odds: DrawOdds | None) -> list[str]:
    """Formats one-card draw odds as prompt lines."""
    if odds is None:
        return [f"- {label}: not_allowed"]
    return [
        f"- {label}.total_draws: {odds.total_draws}",
        f"- {label}.bust_probability: {_format_percent(odds.bust_probability)}",
        f"- {label}.twenty_one_probability: {_format_percent(odds.twenty_one_probability)}",
        (
            f"- {label}.seventeen_to_twenty_one_probability: "
            f"{_format_percent(odds.seventeen_to_twenty_one_probability)}"
        ),
        (
            f"- {label}.five_card_non_bust_probability: "
            f"{_format_percent(odds.five_card_non_bust_probability)}"
        ),
        (
            f"- {label}.five_card_twenty_one_probability: "
            f"{_format_percent(odds.five_card_twenty_one_probability)}"
        ),
    ]


def _format_ev_block(*, ev_analysis: ActionEvAnalysis | None) -> list[str]:
    """Renders the dealer outcome distribution and per-action EV as prompt lines."""
    if ev_analysis is None:
        return ["- ev_analysis: unavailable"]
    outcome = ev_analysis.dealer_outcome
    lines = [
        f"- dealer_outcome.bust_probability: {_format_percent(outcome.bust_probability)}",
        f"- dealer_outcome.total_17_probability: {_format_percent(outcome.total_17_probability)}",
        f"- dealer_outcome.total_18_probability: {_format_percent(outcome.total_18_probability)}",
        f"- dealer_outcome.total_19_probability: {_format_percent(outcome.total_19_probability)}",
        f"- dealer_outcome.total_20_probability: {_format_percent(outcome.total_20_probability)}",
        f"- dealer_outcome.total_21_probability: {_format_percent(outcome.total_21_probability)}",
    ]
    for action_ev in ev_analysis.action_evs:
        suffix = " (estimate)" if action_ev.is_estimate else ""
        lines.append(
            f"- expected_value.{action_ev.action}: {action_ev.expected_value:+.2f}{suffix}"
        )
    lines.append(f"- recommended_action.action: {ev_analysis.recommended_action}")
    lines.append(
        f"- recommended_action.expected_value: {ev_analysis.recommended_expected_value:+.2f}"
    )
    lines.append(
        "- ev_units_note: EV is in multiples of the base hand bet; higher is better; the dealer "
        "outcome and EVs are estimated from the dealer up-card and the remaining shoe with the "
        "hole card unknown, and include the five-card-21 bonus and this table's payouts."
    )
    return lines


def format_action_context(*, context: BotPlayerActionContext | None) -> str:
    """Renders computed action context for the LLM prompt."""
    if context is None:
        return "server_computed_context: unavailable"
    shoe = context.shoe_summary
    dealer = context.dealer
    analysis = context.action_analysis
    lines = [
        "server_computed_context:",
        f"- information_boundary: {context.information_boundary}",
        f"- remaining_shoe.total_cards: {shoe.total_cards}",
        f"- remaining_shoe.rank_counts: {_format_rank_counts(shoe.rank_counts)}",
        f"- remaining_shoe.ace_count: {shoe.ace_count}",
        f"- remaining_shoe.ten_value_count: {shoe.ten_value_count}",
        f"- remaining_shoe.low_card_count_2_to_6: {shoe.low_card_count}",
        f"- remaining_shoe.neutral_card_count_7_to_9: {shoe.neutral_card_count}",
        f"- remaining_shoe.high_card_count_A_or_10_value: {shoe.high_card_count}",
        f"- dealer.up_card: {dealer.up_card}",
        f"- dealer.up_value: {dealer.up_value}",
        *_format_ev_block(ev_analysis=analysis.ev_analysis),
        f"- allowed_actions: {', '.join(analysis.allowed_actions)}",
        f"- basic_strategy_hint.action: {analysis.basic_strategy_action}",
        f"- basic_strategy_hint.reason: {analysis.basic_strategy_reason}",
        f"- stand_summary: {analysis.stand_summary or 'not_applicable'}",
        f"- split_summary: {analysis.split_summary or 'not_allowed'}",
        f"- surrender_summary: {analysis.surrender_summary or 'not_allowed'}",
    ]
    lines.extend(_format_draw_odds(label="hit_odds", odds=analysis.hit_odds))
    lines.extend(_format_draw_odds(label="double_odds", odds=analysis.double_odds))
    return "\n".join(lines)


def format_insurance_context(*, context: BotPlayerInsuranceContext | None) -> str:
    """Renders computed insurance context for the LLM prompt."""
    if context is None:
        return "server_computed_context: unavailable"
    shoe = context.shoe_summary
    dealer = context.dealer
    return "\n".join([
        "server_computed_context:",
        f"- information_boundary: {context.information_boundary}",
        f"- remaining_shoe.total_cards: {shoe.total_cards}",
        f"- remaining_shoe.rank_counts: {_format_rank_counts(shoe.rank_counts)}",
        f"- remaining_shoe.ace_count: {shoe.ace_count}",
        f"- remaining_shoe.ten_value_count: {shoe.ten_value_count}",
        f"- dealer.up_card: {dealer.up_card}",
        f"- dealer.up_value: {dealer.up_value}",
        f"- ten_value_probability: {_format_percent(context.ten_value_probability)}",
        f"- insurance_cost: {context.insurance_cost}",
        f"- insurance_payout: {context.insurance_payout}",
        f"- insurance_expected_value: {context.insurance_expected_value:+.0f} {CURRENCY_NAME}",
        f"- insurance_recommendation: {context.insurance_recommendation}",
        f"- insurance_analysis: {context.summary}",
    ])


def _format_finance_block(finance: BotFinancialContext) -> str:
    """Renders the bot's lifetime + daily financial state as a prompt block."""
    return (
        f"bankroll_context:\n"
        f"- current_balance_{CURRENCY_NAME}: {finance.balance}\n"
        f"- lifetime_earned_{CURRENCY_NAME}: {finance.total_earned}\n"
        f"- lifetime_spent_{CURRENCY_NAME}: {finance.total_spent}\n"
        f"- today_win_{CURRENCY_NAME}: {finance.daily_win}\n"
        f"- today_loss_{CURRENCY_NAME}: {finance.daily_loss}\n"
        f"- today_net_{CURRENCY_NAME}: {finance.daily_net:+d}"
    )


def _format_other_players_block(other_players: list[OtherPlayerView]) -> str:
    """Renders other players' visible table state, or a placeholder when empty."""
    if not other_players:
        return "other_players: none (only bot player and casino)"
    lines: list[str] = ["other_players:"]
    for index, other in enumerate(other_players, start=1):
        status = "finished" if other.is_finished else "active"
        hands_repr = " | ".join(other.hands) if other.hands else "not_dealt"
        lines.append(f"- Player{index} (bet {other.bet} {CURRENCY_NAME}, {status}): {hands_repr}")
    return "\n".join(lines)


def _format_other_player_bets_block(other_player_bets: list[tuple[str, int]]) -> str:
    """Renders the per-player bet list visible during the bet phase."""
    if not other_player_bets:
        return "other_player_bets: none"
    lines: list[str] = ["other_player_bets:"]
    for index, (_display_name, bet) in enumerate(other_player_bets, start=1):
        lines.append(f"- Player{index}: {bet} {CURRENCY_NAME}")
    return "\n".join(lines)


class BotActionReasonRequest(BaseModel):
    """Computed inputs for the background LLM narration of a chosen bot action.

    The action is already decided deterministically by the EV engine; these
    fields only let the model write a faithful Traditional Chinese reason for
    that fixed action.
    """

    model_config = ConfigDict(frozen=True)

    action: BotAction = Field(
        ..., description="The action already chosen by the EV engine to narrate."
    )
    hand_repr: str = Field(..., description="Text representation of the bot's active hand.")
    hand_total: int = Field(..., description="Best total of the bot's active hand.")
    dealer_up: Card | None = Field(
        ..., description="The dealer's face-up card, or None when not yet dealt."
    )
    allowed_actions: tuple[BotAction, ...] = Field(
        ..., description="Actions the bot was legally allowed to take this turn."
    )
    bet: int = Field(..., description="Current wager on the bot's active hand.")
    balance_remaining: int = Field(
        ..., description="Bot balance still uncommitted after current wagers."
    )
    finance: BotFinancialContext = Field(
        ..., description="The bot's lifetime and daily financial state."
    )
    other_players: list[OtherPlayerView] = Field(
        ..., description="Visible table state of the other seated players."
    )
    own_other_hands: list[str] = Field(
        ..., description="Text representations of the bot's other split hands, if any."
    )
    action_context: BotPlayerActionContext | None = Field(
        ..., description="Full computed action context, or None when unavailable."
    )


class BotInsuranceReasonRequest(BaseModel):
    """Computed inputs for the background LLM narration of an insurance choice."""

    model_config = ConfigDict(frozen=True)

    take_insurance: bool = Field(
        ..., description="The insurance decision already made, True to take it, to narrate."
    )
    dealer_up: Card | None = Field(
        ..., description="The dealer's face-up card, or None when not yet dealt."
    )
    hand_repr: str = Field(..., description="Text representation of the bot's opening hand.")
    bet: int = Field(..., description="The bot's main wager this round.")
    finance: BotFinancialContext = Field(
        ..., description="The bot's lifetime and daily financial state."
    )
    other_players: list[OtherPlayerView] = Field(
        ..., description="Visible table state of the other seated players."
    )
    insurance_context: BotPlayerInsuranceContext | None = Field(
        ..., description="Full computed insurance context, or None when unavailable."
    )


class BotPlayerAI(BaseModel):
    """Wraps slow-model calls for the bot's player-side decisions.

    Attributes:
        client: The shared AsyncOpenAI client.
        model: Slow-model settings for strategic Blackjack reasoning.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: AsyncOpenAI = Field(..., description="The shared AsyncOpenAI client.")
    model: ModelSettings = Field(
        ..., description="Slow-model settings for strategic Blackjack reasoning."
    )

    async def narrate_bot_action_reason(self, *, request: BotActionReasonRequest) -> str:
        """Returns a Traditional Chinese reason for the already-chosen action.

        The action is fixed by the EV engine; this call only produces flavor
        text and runs off the table's critical path, so any timeout or failure
        falls back to the deterministic template reason.
        """
        template = action_decision_reason(action_context=request.action_context)
        dealer_label = str(request.dealer_up) if request.dealer_up else "unknown"
        own_other = (
            "own_other_split_hands: " + " | ".join(request.own_other_hands)
            if request.own_other_hands
            else "own_other_split_hands: none"
        )
        user_text = (
            f"chosen_action: {request.action}\n\n"
            f"{_format_finance_block(finance=request.finance)}\n\n"
            f"active_hand_bet_{CURRENCY_NAME}: {request.bet}\n"
            f"uncommitted_balance_{CURRENCY_NAME}: {request.balance_remaining}\n\n"
            f"active_hand: {request.hand_repr}\n"
            f"active_hand_total: {request.hand_total}\n"
            f"{own_other}\n\n"
            f"dealer_up_card: {dealer_label}\n"
            f"{_format_other_players_block(other_players=request.other_players)}\n\n"
            f"allowed_actions: [{', '.join(request.allowed_actions)}]\n\n"
            f"{format_action_context(context=request.action_context)}"
        )
        decision = await parse_responses_or_none(
            client=self.client,
            model=self.model,
            instructions=BOT_PLAYER_ACTION_PROMPT,
            user_text=user_text,
            end_user_id=_ACTION_END_USER_ID,
            text_format=BotPlayerActionDecision,
            timeout_seconds=BOT_ACTION_AI_TIMEOUT_SECONDS,
        )
        return decision.reason if decision is not None else template

    async def narrate_bot_insurance_reason(self, *, request: BotInsuranceReasonRequest) -> str:
        """Returns a Traditional Chinese reason for the already-made insurance choice.

        The take/decline decision is fixed by the remaining-shoe ten density;
        this call only narrates it off the critical path and degrades to the
        deterministic template reason on timeout or failure.
        """
        template = insurance_decision_reason(
            take_insurance=request.take_insurance, insurance_context=request.insurance_context
        )
        dealer_label = str(request.dealer_up) if request.dealer_up else "unknown"
        insurance_cost = request.bet // 2
        user_text = (
            f"chosen_decision: {'take' if request.take_insurance else 'decline'}\n\n"
            f"{_format_finance_block(finance=request.finance)}\n\n"
            f"main_bet_{CURRENCY_NAME}: {request.bet}\n"
            f"insurance_cost_{CURRENCY_NAME}: {insurance_cost}\n"
            f"insurance_payout_if_won_{CURRENCY_NAME}: {insurance_cost * 2}\n\n"
            f"opening_hand: {request.hand_repr}\n"
            f"dealer_up_card: {dealer_label}\n"
            f"{_format_other_players_block(other_players=request.other_players)}\n\n"
            f"{format_insurance_context(context=request.insurance_context)}"
        )
        decision = await parse_responses_or_none(
            client=self.client,
            model=self.model,
            instructions=BOT_PLAYER_INSURANCE_PROMPT,
            user_text=user_text,
            end_user_id=_INSURANCE_END_USER_ID,
            text_format=BotPlayerInsuranceDecision,
            timeout_seconds=BOT_INSURANCE_AI_TIMEOUT_SECONDS,
        )
        return decision.reason if decision is not None else template
