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

from typing import Final, cast
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import BaseModel, ConfigDict
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.games import (
    Card,
    BotAction,
    OtherPlayerView,
    BotFinancialContext,
    BotPlayerBetDecision,
    BotPlayerActionDecision,
    BotPlayerInsuranceDecision,
)
from discordbot.typings.models import ModelSettings
from discordbot.cogs._games.prompts import (
    BOT_PLAYER_BET_PROMPT,
    BOT_PLAYER_ACTION_PROMPT,
    BOT_PLAYER_INSURANCE_PROMPT,
)
from discordbot.cogs._economy.presentation import CURRENCY_NAME

BOT_BET_AI_TIMEOUT_SECONDS = 30.0
BOT_ACTION_AI_TIMEOUT_SECONDS = 30.0
BOT_INSURANCE_AI_TIMEOUT_SECONDS = 30.0
# Bot decisions are system-side LLM calls. ASCII labels per method let LiteLLM
# telemetry split bet / action / insurance traffic, mirroring the
# `auto_unmute.py` / `_stock/news.py` / `prompt_dev.py` pattern.
_BET_END_USER_ID: Final[str] = "bot_player_bet"
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
    "server_true_counts_plus_dealer_hole; no next-card field and no ordered future shoe"
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

    total_cards: int
    rank_counts: dict[str, int]
    ace_count: int
    ten_value_count: int
    low_card_count: int
    neutral_card_count: int
    high_card_count: int


class DealerKnowledge(BaseModel):
    """Server-visible dealer state exposed to the bot player AI."""

    model_config = ConfigDict(frozen=True)

    up_card: str
    hole_card: str
    cards: str
    total: int
    natural_blackjack: bool
    h17_status: str


class DrawOdds(BaseModel):
    """One-card draw probabilities derived from current hand plus rank counts."""

    model_config = ConfigDict(frozen=True)

    total_draws: int
    bust_probability: float
    twenty_one_probability: float
    seventeen_to_twenty_one_probability: float
    five_card_non_bust_probability: float
    five_card_twenty_one_probability: float


class ActionAnalysis(BaseModel):
    """Computed reference data for the bot player's action decision."""

    model_config = ConfigDict(frozen=True)

    allowed_actions: tuple[BotAction, ...]
    basic_strategy_action: BotAction
    basic_strategy_reason: str
    hit_odds: DrawOdds | None = None
    double_odds: DrawOdds | None = None
    stand_summary: str | None = None
    split_summary: str | None = None
    surrender_summary: str | None = None


class BotPlayerActionContext(BaseModel):
    """Complete computed context for one bot-player action decision."""

    model_config = ConfigDict(frozen=True)

    information_boundary: str
    shoe_summary: ShoeSummary
    dealer: DealerKnowledge
    action_analysis: ActionAnalysis


class BotPlayerInsuranceContext(BaseModel):
    """Complete computed context for one bot-player insurance decision."""

    model_config = ConfigDict(frozen=True)

    information_boundary: str
    shoe_summary: ShoeSummary
    dealer: DealerKnowledge
    insurance_cost: int
    insurance_payout: int
    dealer_blackjack: bool
    side_bet_delta_if_taken: int
    summary: str


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


def _card_blackjack_value(*, card: Card) -> int:
    """Returns the Blackjack value used by fallback strategy tables."""
    if card.rank == "A":
        return 11
    if card.rank in ("J", "Q", "K"):
        return 10
    return int(card.rank)


def _hand_total_and_soft(*, cards: list[Card]) -> tuple[int, bool]:
    """Returns the best total and whether at least one Ace remains high."""
    total = 0
    aces = 0
    for card in cards:
        if card.rank == "A":
            aces += 1
        total += _card_blackjack_value(card=card)
    aces_high = aces
    while total > 21 and aces_high > 0:
        total -= 10
        aces_high -= 1
    return total, aces_high > 0


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


def _is_blackjack_cards(*, cards: list[Card]) -> bool:
    """Returns whether cards form a natural Blackjack."""
    return len(cards) == 2 and _hand_total_and_soft(cards=cards)[0] == 21


def _card_list_text(*, cards: list[Card]) -> str:
    """Returns a compact card list for prompt context."""
    return " ".join(str(card) for card in cards) if cards else "none"


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


def _dealer_h17_status(*, dealer_cards: list[Card]) -> str:
    """Returns the dealer's forced H17 status from server-visible cards."""
    if not dealer_cards:
        return "unknown"
    total, is_soft = _hand_total_and_soft(cards=dealer_cards)
    if _is_blackjack_cards(cards=dealer_cards):
        return "natural_blackjack"
    if total > 21:
        return "bust"
    if total < 17:
        return "must_hit_below_17"
    if total == 17 and is_soft:
        return "must_hit_soft_17"
    return "stand_hard_17_or_more"


def build_dealer_knowledge(*, dealer_cards: list[Card], dealer_up: Card | None) -> DealerKnowledge:
    """Builds the dealer state intentionally exposed to the bot player AI."""
    hole_card = dealer_cards[0] if dealer_cards else None
    total = _hand_total_and_soft(cards=dealer_cards)[0] if dealer_cards else 0
    return DealerKnowledge(
        up_card=str(dealer_up) if dealer_up is not None else "unknown",
        hole_card=str(hole_card) if hole_card is not None else "unknown",
        cards=_card_list_text(cards=dealer_cards),
        total=total,
        natural_blackjack=_is_blackjack_cards(cards=dealer_cards),
        h17_status=_dealer_h17_status(dealer_cards=dealer_cards),
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


def fallback_bet(*, balance: int, table_bet: int) -> int:
    """Deterministic bet fallback when the LLM is slow or fails.

    Matches the table bet, clamped into [1, balance]. Returns 1 when balance
    is non-positive; callers guard auto-join against that already.
    """
    if balance <= 0:
        return 1
    return max(1, min(balance, table_bet))


def fallback_action(
    *,
    hand_cards: list[Card],
    hand_total: int,
    dealer_up: Card | None,
    is_pair_hand: bool,
    allowed_actions: tuple[BotAction, ...],
) -> BotAction:
    """Deterministic basic-strategy fallback that only emits allowed actions."""
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
    balance_remaining: int,
    doubled: bool = False,
) -> BotPlayerActionContext:
    """Builds computed AI context without exposing the future shoe order."""
    hand_total, _is_soft = _hand_total_and_soft(cards=hand_cards)
    basic_strategy_action = fallback_action(
        hand_cards=hand_cards,
        hand_total=hand_total,
        dealer_up=dealer_up,
        is_pair_hand=is_pair_hand,
        allowed_actions=allowed_actions,
    )
    dealer = build_dealer_knowledge(dealer_cards=dealer_cards, dealer_up=dealer_up)
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
            basic_strategy_reason=_basic_strategy_reason(action=basic_strategy_action),
            hit_odds=hit_odds,
            double_odds=double_odds,
            stand_summary=(
                f"Standing leaves active hand total {hand_total} against known dealer total "
                f"{dealer.total}; dealer status: {dealer.h17_status}; "
                f"uncommitted balance after current wagers: {balance_remaining} {CURRENCY_NAME}."
            ),
            split_summary=split_summary,
            surrender_summary=surrender_summary,
        ),
    )


def build_bot_insurance_context(
    *, dealer_cards: list[Card], dealer_up: Card | None, shoe: list[Card], insurance_cost: int
) -> BotPlayerInsuranceContext:
    """Builds computed AI context for the insurance side bet."""
    dealer = build_dealer_knowledge(dealer_cards=dealer_cards, dealer_up=dealer_up)
    dealer_blackjack = dealer.natural_blackjack
    insurance_payout = insurance_cost * 2
    delta_if_taken = insurance_payout if dealer_blackjack else -insurance_cost
    return BotPlayerInsuranceContext(
        information_boundary=_INFO_BOUNDARY,
        shoe_summary=build_shoe_summary(shoe=shoe),
        dealer=dealer,
        insurance_cost=insurance_cost,
        insurance_payout=insurance_payout,
        dealer_blackjack=dealer_blackjack,
        side_bet_delta_if_taken=delta_if_taken,
        summary=(
            "Dealer has natural Blackjack; insurance side bet wins."
            if dealer_blackjack
            else "Dealer does not have natural Blackjack; insurance side bet loses."
        ),
    )


def fallback_insurance() -> bool:
    """Deterministic insurance fallback: never take (negative EV)."""
    return False


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
        f"- dealer.hole_card: {dealer.hole_card}",
        f"- dealer.cards: {dealer.cards}",
        f"- dealer.known_total: {dealer.total}",
        f"- dealer.natural_blackjack: {dealer.natural_blackjack}",
        f"- dealer.h17_status: {dealer.h17_status}",
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
        f"- dealer.hole_card: {dealer.hole_card}",
        f"- dealer.cards: {dealer.cards}",
        f"- dealer.known_total: {dealer.total}",
        f"- dealer.natural_blackjack: {dealer.natural_blackjack}",
        f"- insurance_cost: {context.insurance_cost}",
        f"- insurance_payout: {context.insurance_payout}",
        f"- dealer_blackjack: {context.dealer_blackjack}",
        f"- side_bet_delta_if_taken: {context.side_bet_delta_if_taken}",
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


class BotPlayerAI(BaseModel):
    """Wraps slow-model calls for the bot's player-side decisions.

    Attributes:
        client: The shared AsyncOpenAI client.
        model: Slow-model settings for strategic Blackjack reasoning.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: AsyncOpenAI
    model: ModelSettings

    async def decide_bot_bet(
        self,
        *,
        finance: BotFinancialContext,
        table_bet: int,
        other_player_bets: list[tuple[str, int]],
    ) -> int:
        """Returns the bot's bet for the upcoming round, falling back on error."""
        fallback = fallback_bet(balance=finance.balance, table_bet=table_bet)
        user_text = (
            f"{_format_finance_block(finance=finance)}\n\n"
            f"table_bet_{CURRENCY_NAME}: {table_bet}\n"
            f"{_format_other_player_bets_block(other_player_bets=other_player_bets)}"
        )
        try:
            async with asyncio.timeout(delay=BOT_BET_AI_TIMEOUT_SECONDS):
                responses = await self.client.responses.parse(
                    model=self.model.name,
                    instructions=BOT_PLAYER_BET_PROMPT,
                    input=cast(
                        "ResponseInputParam",
                        [EasyInputMessageParam(role="user", content=user_text)],
                    ),
                    text_format=BotPlayerBetDecision,
                    reasoning=self.model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": _BET_END_USER_ID},
                    extra_body={"mock_testing_fallbacks": False},
                )
        except TimeoutError:
            logfire.warn(
                "Bot bet decision timed out; using deterministic fallback",
                timeout_seconds=BOT_BET_AI_TIMEOUT_SECONDS,
            )
            return fallback
        except Exception:
            logfire.warn("Bot bet decision failed; using deterministic fallback", _exc_info=True)
            return fallback
        if responses.output_parsed is None:
            return fallback
        candidate = responses.output_parsed.bet_amount
        return max(1, min(finance.balance, candidate)) if finance.balance > 0 else fallback

    async def decide_bot_action(  # noqa: PLR0913 -- bot action decision needs full table context
        self,
        *,
        hand_cards: list[Card],
        hand_total: int,
        hand_repr: str,
        dealer_up: Card | None,
        is_pair_hand: bool,
        allowed_actions: tuple[BotAction, ...],
        bet: int,
        balance_remaining: int,
        finance: BotFinancialContext,
        other_players: list[OtherPlayerView],
        own_other_hands: list[str],
        action_context: BotPlayerActionContext | None = None,
    ) -> BotPlayerActionDecision:
        """Returns the bot's next action with reasoning, falling back to basic strategy."""
        fallback_decision = BotPlayerActionDecision(
            action=fallback_action(
                hand_cards=hand_cards,
                hand_total=hand_total,
                dealer_up=dealer_up,
                is_pair_hand=is_pair_hand,
                allowed_actions=allowed_actions,
            ),
            reason="基本策略 fallback",
        )
        dealer_label = str(dealer_up) if dealer_up else "unknown"
        allowed_text = ", ".join(allowed_actions)
        own_other = (
            "own_other_split_hands: " + " | ".join(own_other_hands)
            if own_other_hands
            else "own_other_split_hands: none"
        )
        user_text = (
            f"{_format_finance_block(finance=finance)}\n\n"
            f"active_hand_bet_{CURRENCY_NAME}: {bet}\n"
            f"uncommitted_balance_{CURRENCY_NAME}: {balance_remaining}\n\n"
            f"active_hand: {hand_repr}\n"
            f"active_hand_total: {hand_total}\n"
            f"is_pair_hand_for_split: {is_pair_hand}\n"
            f"{own_other}\n\n"
            f"dealer_up_card: {dealer_label}\n"
            f"{_format_other_players_block(other_players=other_players)}\n\n"
            f"allowed_actions: [{allowed_text}]\n\n"
            f"{format_action_context(context=action_context)}"
        )
        try:
            async with asyncio.timeout(delay=BOT_ACTION_AI_TIMEOUT_SECONDS):
                responses = await self.client.responses.parse(
                    model=self.model.name,
                    instructions=BOT_PLAYER_ACTION_PROMPT,
                    input=cast(
                        "ResponseInputParam",
                        [EasyInputMessageParam(role="user", content=user_text)],
                    ),
                    text_format=BotPlayerActionDecision,
                    reasoning=self.model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": _ACTION_END_USER_ID},
                    extra_body={"mock_testing_fallbacks": False},
                )
        except TimeoutError:
            logfire.warn(
                "Bot action decision timed out; using basic-strategy fallback",
                timeout_seconds=BOT_ACTION_AI_TIMEOUT_SECONDS,
            )
            return fallback_decision
        except Exception:
            logfire.warn(
                "Bot action decision failed; using basic-strategy fallback", _exc_info=True
            )
            return fallback_decision
        if responses.output_parsed is None:
            return fallback_decision
        candidate = responses.output_parsed
        if candidate.action in allowed_actions:
            return candidate
        return fallback_decision

    async def decide_bot_insurance(  # noqa: PLR0913 -- insurance prompt needs full table context.
        self,
        *,
        dealer_up: Card | None,
        hand_repr: str,
        bet: int,
        finance: BotFinancialContext,
        other_players: list[OtherPlayerView],
        insurance_context: BotPlayerInsuranceContext | None = None,
    ) -> BotPlayerInsuranceDecision:
        """Returns whether the bot takes insurance with reasoning, falling back to False."""
        fallback_decision = BotPlayerInsuranceDecision(
            take_insurance=fallback_insurance(), reason="保險長期 EV 為負, 直接拒絕"
        )
        dealer_label = str(dealer_up) if dealer_up else "未知"
        insurance_cost = bet // 2
        user_text = (
            f"{_format_finance_block(finance=finance)}\n\n"
            f"main_bet_{CURRENCY_NAME}: {bet}\n"
            f"insurance_cost_{CURRENCY_NAME}: {insurance_cost}\n"
            f"insurance_payout_if_won_{CURRENCY_NAME}: {insurance_cost * 2}\n\n"
            f"opening_hand: {hand_repr}\n"
            f"dealer_up_card: {dealer_label}\n"
            f"{_format_other_players_block(other_players=other_players)}\n\n"
            f"{format_insurance_context(context=insurance_context)}"
        )
        try:
            async with asyncio.timeout(delay=BOT_INSURANCE_AI_TIMEOUT_SECONDS):
                responses = await self.client.responses.parse(
                    model=self.model.name,
                    instructions=BOT_PLAYER_INSURANCE_PROMPT,
                    input=cast(
                        "ResponseInputParam",
                        [EasyInputMessageParam(role="user", content=user_text)],
                    ),
                    text_format=BotPlayerInsuranceDecision,
                    reasoning=self.model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": _INSURANCE_END_USER_ID},
                    extra_body={"mock_testing_fallbacks": False},
                )
        except TimeoutError:
            logfire.warn(
                "Bot insurance decision timed out; declining insurance",
                timeout_seconds=BOT_INSURANCE_AI_TIMEOUT_SECONDS,
            )
            return fallback_decision
        except Exception:
            logfire.warn("Bot insurance decision failed; declining insurance", _exc_info=True)
            return fallback_decision
        if responses.output_parsed is None:
            return fallback_decision
        return responses.output_parsed
