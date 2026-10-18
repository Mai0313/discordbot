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
# play; re-measure offline if the dealer or five-card rules change.
BOT_TABLE_EDGE: Final[float] = 0.13
BOT_TABLE_VARIANCE: Final[float] = 1.34
# Half-Kelly keeps drawdown variance down; the hard fraction cap bounds risk per round.
BOT_KELLY_FRACTION: Final[float] = 0.5
BOT_MAX_BET_FRACTION: Final[float] = 0.10
# Edge added per +1 Hi-Lo true count on a persistent shoe; re-measure offline if rules change.
BOT_EDGE_PER_TRUE_COUNT: Final[float] = 0.0175


def _dealer_up_value(up_card: Card | None) -> int:
    """Returns the Blackjack value of the dealer's up-card (A counts as 11)."""
    if up_card is None:
        return 0
    return card_blackjack_value(card=up_card)


def _should_surrender(hand_total: int, dealer_value: int) -> bool:
    """Returns whether late surrender is the fallback table choice."""
    return (hand_total == 16 and dealer_value in {9, 10, 11}) or (
        hand_total == 15 and dealer_value == 10
    )


def _should_double(cards: list[Card], hand_total: int, dealer_value: int) -> bool:
    """Returns whether double down is the fallback table choice."""
    hard_double_dealers: dict[int, frozenset[int]] = {
        9: frozenset({3, 4, 5, 6}),
        10: frozenset(range(2, 10)),
        11: frozenset(range(2, 12)),
    }
    soft_double_dealers: dict[int, frozenset[int]] = {
        13: frozenset({5, 6}),
        14: frozenset({5, 6}),
        15: frozenset({4, 5, 6}),
        16: frozenset({4, 5, 6}),
        17: frozenset({3, 4, 5, 6}),
        18: frozenset(range(2, 7)),
    }
    is_soft, _total = is_soft_total(cards=cards)
    double_dealers = (
        soft_double_dealers.get(hand_total, frozenset())
        if is_soft
        else hard_double_dealers.get(hand_total, frozenset())
    )
    return dealer_value in double_dealers


def _should_stand(cards: list[Card], hand_total: int, dealer_value: int) -> bool:
    """Returns whether stand is the fallback table choice."""
    is_soft, _total = is_soft_total(cards=cards)
    if is_soft:
        return hand_total >= 19 or (hand_total == 18 and 2 <= dealer_value <= 8)
    return (
        hand_total >= 17
        or (13 <= hand_total <= 16 and dealer_value <= 6)
        or (hand_total == 12 and 4 <= dealer_value <= 6)
    )


def kelly_bet(balance: int, table_minimum: int, edge: float = BOT_TABLE_EDGE) -> int:
    """Returns the fractional-Kelly wager from the per-round edge.

    `BOT_MAX_BET_FRACTION` of `balance` is a hard ceiling, and `table_minimum`
    floors the wager only up to that ceiling; a non-positive edge falls back to
    that capped table floor, and a non-positive `balance` returns 1.

    Args:
        balance (int): The bot's spendable balance.
        table_minimum (int): The table stake the bot matches, up to the bankroll ceiling.
        edge (float): Per-round expected value in base-bet units.
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


def count_adjusted_edge(true_count: float) -> float:
    """Returns the per-round edge adjusted for the Hi-Lo `true_count`.

    Args:
        true_count (float): Hi-Lo true count of the remaining shoe.
    """
    return BOT_TABLE_EDGE + BOT_EDGE_PER_TRUE_COUNT * true_count


def fallback_action(
    hand_cards: list[Card], dealer_up: Card | None, allowed_actions: tuple[BotAction, ...]
) -> BotAction:
    """Returns the up-card-only basic-strategy action from `allowed_actions`.

    Args:
        hand_cards (list[Card]): The bot's active sub-hand cards.
        dealer_up (Card | None): The dealer's visible up-card, if dealt.
        allowed_actions (tuple[BotAction, ...]): Legal actions for the active hand;
            `"split"` is included only when `hand_cards` is a pair.
    """
    pair_split_dealers: dict[int, frozenset[int]] = {
        11: frozenset(range(2, 12)),
        8: frozenset(range(2, 12)),
        9: frozenset({2, 3, 4, 5, 6, 8, 9}),
        7: frozenset(range(2, 8)),
        6: frozenset(range(2, 7)),
        4: frozenset({5, 6}),
        3: frozenset(range(2, 8)),
        2: frozenset(range(2, 8)),
    }
    dealer_value = _dealer_up_value(up_card=dealer_up)
    hand_total = is_soft_total(cards=hand_cards)[1]
    if "split" in allowed_actions and dealer_value in pair_split_dealers.get(
        card_blackjack_value(card=hand_cards[0]), frozenset()
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


def choose_bot_action(
    hand_cards: list[Card],
    dealer_cards: list[Card],
    shoe: list[Card],
    allowed_actions: tuple[BotAction, ...],
    bet: int,
) -> BotAction:
    """Returns the action the bot plays on its active hand.

    Uses the hole-aware EV engine, falling back to `fallback_action` if the
    engine raises.

    Args:
        hand_cards (list[Card]): The bot's active sub-hand cards.
        dealer_cards (list[Card]): The dealer's cards (hole card first, then up-card).
        shoe (list[Card]): Remaining undealt cards in the shoe.
        allowed_actions (tuple[BotAction, ...]): Legal actions for the active hand.
        bet (int): Active wager on the hand.
    """
    try:
        return recommend_action(
            hand_cards=hand_cards,
            dealer_cards=dealer_cards,
            shoe=shoe,
            allowed_actions=allowed_actions,
            bet=bet,
        )
    # Broad on purpose: whatever breaks the engine, the table still plays the turn.
    except Exception as exc:
        logfire.warn(
            "Bot EV engine failed; falling back to basic strategy",
            allowed_actions=allowed_actions,
            hand_ranks=[card.rank for card in hand_cards],
            dealer_ranks=[card.rank for card in dealer_cards],
            shoe_size=len(shoe),
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
    return fallback_action(
        hand_cards=hand_cards,
        dealer_up=dealer_up_card(dealer=dealer_cards),
        allowed_actions=allowed_actions,
    )


def bot_takes_insurance(shoe: list[Card], dealer_cards: list[Card]) -> bool:
    """Returns whether the bot buys insurance: only when the face-down cards make it +EV.

    Priced from the ten-value density of every card still face down at the table: the remaining
    shoe plus the dealer's hole card (`dealer_cards[0]`, hole first as dealt). Insurance pays 2:1,
    so it turns +EV once that density exceeds one third; the hole card is pooled with `shoe` so
    its identity never leaks into the insurance decision.

    Args:
        shoe (list[Card]): Remaining undealt cards in the shoe.
        dealer_cards (list[Card]): Dealer cards in draw order (`dealer_cards[0]` is the hole card).
    """
    face_down = [*shoe, dealer_cards[0]]
    ten_count = sum(1 for card in face_down if card.rank in TEN_VALUE_RANKS)
    return bool(shoe) and ten_count / len(face_down) > 1.0 / 3.0
