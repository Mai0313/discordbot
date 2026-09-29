"""Deterministic decision logic for the Blackjack bot player.

The bot is a regular Blackjack player; the casino system is the dealer. Its
bet sizing, action, and insurance choices are computed without any LLM:
fractional-Kelly betting off the channel shoe's Hi-Lo true count, the hole-aware
EV engine for the action, and a count-based +EV rule for insurance.
"""

from typing import Final

import logfire

from discordbot.typings.games import Card, BotAction
from discordbot.cogs.games.blackjack import (
    TEN_VALUE_RANKS,
    is_soft_total,
    dealer_up_card,
    card_blackjack_value,
)
from discordbot.cogs.games.blackjack_ev import recommend_action

# Per-round edge (at a neutral count) and variance of the bot's hole-aware optimal
# play, both measured by offline simulation. The edge is large because the EV engine
# plays the dealer hole card and this table's five-card rules are player-favorable;
# re-measure offline if those rules change.
BOT_TABLE_EDGE: Final[float] = 0.13
BOT_TABLE_VARIANCE: Final[float] = 1.34
# Half-Kelly keeps drawdown variance down; the hard fraction cap protects the
# bankroll even if the measured edge drifts.
BOT_KELLY_FRACTION: Final[float] = 0.5
BOT_MAX_BET_FRACTION: Final[float] = 0.10
# Edge added per +1 Hi-Lo true count when the shoe persists across rounds, used for
# count-based bet spreading. Measured by offline simulation against a persistent shoe,
# and well above the standard Hi-Lo ~0.005 because this table's five-card rules amplify
# a ten-rich shoe; re-measure offline if those rules change.
BOT_EDGE_PER_TRUE_COUNT: Final[float] = 0.0175
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


def kelly_bet(*, balance: int, table_minimum: int, edge: float = BOT_TABLE_EDGE) -> int:
    """Returns the fractional-Kelly wager from the per-round edge.

    The growth-optimal stake is a fraction of the bankroll set by the edge. With
    a fresh shoe the edge is the constant `BOT_TABLE_EDGE`; with a persistent shoe
    it is `count_adjusted_edge(...)` so the bot spreads its bet by true count.

    `BOT_MAX_BET_FRACTION` of the bankroll is a hard ceiling, not merely a cap on the
    Kelly fraction: the owner-chosen table stake floors the bet only up to that ceiling,
    so a large table stake cannot drag the bot past its risk limit. The bot still sits
    at any table, but it never wagers more than that fraction of its balance in one
    round. A non-positive edge falls back to that capped table floor instead of
    refusing to play.

    Args:
        balance: The bot's spendable balance.
        table_minimum: The table stake the bot matches, up to the bankroll ceiling.
        edge: Per-round expected value in base-bet units.

    Returns:
        A positive integer wager within `[1, BOT_MAX_BET_FRACTION * balance]`, never
        above `balance`. A non-positive balance returns 1, the one case above `balance`.
    """
    if balance <= 0:
        return 1
    ceiling = max(1, min(round(BOT_MAX_BET_FRACTION * balance), balance))
    floor = max(1, min(table_minimum, ceiling))
    if edge <= 0:
        return floor
    fraction = min(max(BOT_KELLY_FRACTION * edge / BOT_TABLE_VARIANCE, 0.0), BOT_MAX_BET_FRACTION)
    wager = round(fraction * balance)
    return max(floor, min(wager, ceiling))


def count_adjusted_edge(*, true_count: float) -> float:
    """Returns the per-round edge adjusted for the Hi-Lo true count.

    A persistent shoe lets the bot read a true count before betting; a positive
    count means the remaining shoe is rich in ten-value cards and aces, which lifts
    the edge.
    """
    return BOT_TABLE_EDGE + BOT_EDGE_PER_TRUE_COUNT * true_count


def _safe_recommend_action(  # noqa: PLR0913 -- thin EV-engine wrapper mirroring its signature.
    *,
    hand_cards: list[Card],
    dealer_cards: list[Card],
    shoe: list[Card],
    allowed_actions: tuple[BotAction, ...],
    doubled: bool,
    bet: int,
) -> BotAction | None:
    """Runs the EV engine, returning None on any failure so a bot turn never crashes."""
    try:
        return recommend_action(
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


def choose_bot_action(  # noqa: PLR0913 -- the decision reads the hand, the dealer, and the shoe.
    *,
    hand_cards: list[Card],
    dealer_cards: list[Card],
    shoe: list[Card],
    allowed_actions: tuple[BotAction, ...],
    is_pair_hand: bool,
    bet: int,
    doubled: bool = False,
) -> BotAction:
    """Returns the action the bot plays on its active hand.

    The EV engine's hole-aware recommendation, or the up-card-only basic-strategy table
    only when the engine is unavailable.
    """
    ev_action = _safe_recommend_action(
        hand_cards=hand_cards,
        dealer_cards=dealer_cards,
        shoe=shoe,
        allowed_actions=allowed_actions,
        doubled=doubled,
        bet=bet,
    )
    if ev_action is not None:
        return ev_action
    return fallback_action(
        hand_cards=hand_cards,
        hand_total=is_soft_total(cards=hand_cards)[1],
        dealer_up=dealer_up_card(dealer=dealer_cards),
        is_pair_hand=is_pair_hand,
        allowed_actions=allowed_actions,
    )


def bot_takes_insurance(*, shoe: list[Card]) -> bool:
    """Returns whether the bot buys insurance: only when the unseen shoe makes it +EV.

    Priced from the remaining shoe's ten-value density alone; the dealer's hole card is never
    an input, so it cannot reach the decision. Insurance pays +2x its cost on a ten-value hole
    and loses the cost otherwise, EV = cost * (3p - 1), so it only turns positive once that
    density clears one third.
    """
    ten_count = sum(1 for card in shoe if card.rank in TEN_VALUE_RANKS)
    return bool(shoe) and ten_count / len(shoe) > 1.0 / 3.0
