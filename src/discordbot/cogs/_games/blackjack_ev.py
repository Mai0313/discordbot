"""Exact, hole-card-aware Blackjack expected-value engine.

The bot player already sees the dealer's hole card, but an LLM reasons poorly
about multi-step no-replacement probability. This pure module turns that known
state into decision-grade numbers: the dealer's exact final-total distribution
under H17 and the expected value of every legal action, measured in multiples
of the base hand bet.

Everything here is deterministic and order-independent: the shoe is collapsed
to a 10-bucket value-count multiset (`2..9`, ten-value, ace), so results depend
only on which cards remain, not their order. The recursions terminate naturally
(dealer stands at hard 17+, a player hand auto-finishes at five non-bust cards),
and per-call memoization keeps a single decision well under a millisecond.

This table's non-standard payouts are modeled directly: a five-card non-bust
wins immediately, and a five-card 21 also earns a system-funded bonus. The
VIP payout bonus is intentionally excluded; it is a settlement-layer perk, not
a per-hand strategic lever.
"""

from typing import Final

from pydantic import BaseModel, ConfigDict

from discordbot.typings.games import Card, ActionEv, BotAction, DealerOutcome, ActionEvAnalysis
from discordbot.cogs._games.blackjack import hand_value, is_soft_total

# Bucket index -> Blackjack draw value. Index 8 is any ten-value card, index 9
# is an ace counted high (11). Indices 0..7 map ranks 2..9 directly.
_BUCKET_VALUES: Final[tuple[int, ...]] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
_TEN_BUCKET: Final[int] = 8
_ACE_BUCKET: Final[int] = 9
# Split EV is an independence approximation that is slightly optimistic, so the
# recommendation only prefers a split when it clears the best exact action by
# this margin (in base-bet units).
SPLIT_EV_MARGIN: Final[float] = 0.02

# (p17, p18, p19, p20, p21, p_bust); indices 0..4 are dealer totals 17..21.
_DealerDist = tuple[float, ...]
_BUST_INDEX: Final[int] = 5
_DealerMemo = dict[tuple[int, bool, tuple[int, ...]], _DealerDist]
_PlayerMemo = dict[tuple[int, bool, int, tuple[int, ...]], float]


class _EvContext(BaseModel):
    """Fixed per-decision state threaded through the EV recursions."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dealer_total: int
    dealer_soft: bool
    dealer_memo: _DealerMemo
    player_memo: _PlayerMemo


def _bucket_for_rank(*, rank: str) -> int:
    """Maps a card rank to its value bucket index."""
    if rank == "A":
        return _ACE_BUCKET
    if rank in ("10", "J", "Q", "K"):
        return _TEN_BUCKET
    return int(rank) - 2


def build_shoe_value_counts(*, shoe: list[Card]) -> tuple[int, ...]:
    """Collapses a card shoe into a 10-bucket value-count vector (2..9, ten, ace)."""
    counts = [0] * 10
    for card in shoe:
        counts[_bucket_for_rank(rank=card.rank)] += 1
    return tuple(counts)


def _decrement(*, shoe: tuple[int, ...], bucket: int) -> tuple[int, ...]:
    """Returns a copy of the shoe vector with one card removed from a bucket."""
    mutable = list(shoe)
    mutable[bucket] -= 1
    return tuple(mutable)


def _add_value(*, total: int, soft: bool, bucket: int) -> tuple[int, bool]:
    """Adds one drawn card to a running `(total, soft)` Blackjack hand state.

    Mirrors `hand_value`/`is_soft_total`: at most one ace is ever counted high,
    and a non-ace draw that busts a soft hand demotes that ace to 1.

    Args:
        total: Best current total with any high ace already counted as 11.
        soft: Whether an ace is currently counted as 11.
        bucket: Value bucket index of the drawn card.

    Returns:
        The updated `(total, soft)` state.
    """
    if bucket == _ACE_BUCKET:
        if total + 11 <= 21:
            return total + 11, True
        return total + 1, soft
    new_total = total + _BUCKET_VALUES[bucket]
    if new_total > 21 and soft:
        return new_total - 10, False
    return new_total, soft


def _dealer_distribution(
    *, total: int, soft: bool, shoe: tuple[int, ...], memo: _DealerMemo
) -> _DealerDist:
    """Computes the exact dealer final-total distribution under H17.

    The dealer hits while below 17 and on soft 17, and stands on hard 17+.
    Probabilities are over `{17, 18, 19, 20, 21, bust}`.
    """
    if total > 21:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    if total > 17 or (total == 17 and not soft):
        index = total - 17
        return tuple(1.0 if position == index else 0.0 for position in range(6))
    key = (total, soft, shoe)
    cached = memo.get(key)
    if cached is not None:
        return cached
    shoe_total = sum(shoe)
    if shoe_total == 0:
        # Unreachable with a real 208-card shoe; never divide by zero.
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    accumulator = [0.0] * 6
    for bucket, count in enumerate(shoe):
        if count <= 0:
            continue
        probability = count / shoe_total
        next_total, next_soft = _add_value(total=total, soft=soft, bucket=bucket)
        child = _dealer_distribution(
            total=next_total, soft=next_soft, shoe=_decrement(shoe=shoe, bucket=bucket), memo=memo
        )
        for position in range(6):
            accumulator[position] += probability * child[position]
    result = tuple(accumulator)
    memo[key] = result
    return result


def _dealer_dist_for(*, ctx: _EvContext, shoe: tuple[int, ...]) -> _DealerDist:
    """Resolves the dealer distribution from the fixed dealer hand and a shoe."""
    return _dealer_distribution(
        total=ctx.dealer_total, soft=ctx.dealer_soft, shoe=shoe, memo=ctx.dealer_memo
    )


def _stand_ev_unit(
    *, player_total: int, five_card_eligible: bool, shoe: tuple[int, ...], ctx: _EvContext
) -> float:
    """Returns the per-unit EV of standing on a non-bust hand against the dealer.

    A five-card non-bust wins immediately (+1); a five-card 21 also earns the
    system bonus, so its EV is `2 - P(dealer 21)`.
    """
    if five_card_eligible:
        if player_total == 21:
            return 2.0 - _dealer_dist_for(ctx=ctx, shoe=shoe)[4]
        return 1.0
    distribution = _dealer_dist_for(ctx=ctx, shoe=shoe)
    win = distribution[_BUST_INDEX]
    lose = 0.0
    for index in range(5):
        dealer_total = 17 + index
        if player_total > dealer_total:
            win += distribution[index]
        elif player_total < dealer_total:
            lose += distribution[index]
    return win - lose


def _player_optimal_ev(
    *, total: int, soft: bool, num_cards: int, shoe: tuple[int, ...], ctx: _EvContext
) -> float:
    """Returns the best EV reachable from a player state via optimal hit/stand."""
    if total > 21:
        return -1.0
    stand_ev = _stand_ev_unit(
        player_total=total, five_card_eligible=num_cards >= 5, shoe=shoe, ctx=ctx
    )
    if num_cards >= 5:
        # A five-card non-bust hand auto-finishes; hitting is impossible.
        return stand_ev
    key = (total, soft, num_cards, shoe)
    cached = ctx.player_memo.get(key)
    if cached is not None:
        return cached
    shoe_total = sum(shoe)
    if shoe_total == 0:
        ctx.player_memo[key] = stand_ev
        return stand_ev
    hit_ev = 0.0
    for bucket, count in enumerate(shoe):
        if count <= 0:
            continue
        next_total, next_soft = _add_value(total=total, soft=soft, bucket=bucket)
        hit_ev += (count / shoe_total) * _player_optimal_ev(
            total=next_total,
            soft=next_soft,
            num_cards=num_cards + 1,
            shoe=_decrement(shoe=shoe, bucket=bucket),
            ctx=ctx,
        )
    best = max(stand_ev, hit_ev)
    ctx.player_memo[key] = best
    return best


def _hit_action_ev(
    *, total: int, soft: bool, num_cards: int, shoe: tuple[int, ...], ctx: _EvContext
) -> float:
    """Returns the EV of hitting now and then playing optimally."""
    shoe_total = sum(shoe)
    if shoe_total == 0:
        return -1.0
    expected = 0.0
    for bucket, count in enumerate(shoe):
        if count <= 0:
            continue
        next_total, next_soft = _add_value(total=total, soft=soft, bucket=bucket)
        expected += (count / shoe_total) * _player_optimal_ev(
            total=next_total,
            soft=next_soft,
            num_cards=num_cards + 1,
            shoe=_decrement(shoe=shoe, bucket=bucket),
            ctx=ctx,
        )
    return expected


def _double_ev(*, total: int, soft: bool, shoe: tuple[int, ...], ctx: _EvContext) -> float:
    """Returns the EV of doubling: one card at double stake, then stand."""
    shoe_total = sum(shoe)
    if shoe_total == 0:
        return 2.0 * _stand_ev_unit(
            player_total=total, five_card_eligible=False, shoe=shoe, ctx=ctx
        )
    expected = 0.0
    for bucket, count in enumerate(shoe):
        if count <= 0:
            continue
        next_total, _next_soft = _add_value(total=total, soft=soft, bucket=bucket)
        probability = count / shoe_total
        if next_total > 21:
            expected += probability * -2.0
        else:
            expected += (
                probability
                * 2.0
                * _stand_ev_unit(
                    player_total=next_total,
                    five_card_eligible=False,
                    shoe=_decrement(shoe=shoe, bucket=bucket),
                    ctx=ctx,
                )
            )
    return expected


def _single_split_hand_ev(
    *, pair_bucket: int, is_ace_pair: bool, shoe: tuple[int, ...], ctx: _EvContext
) -> float:
    """Returns the optimal EV of one post-split hand under split constraints."""
    shoe_total = sum(shoe)
    if shoe_total == 0:
        return 0.0
    base_total, base_soft = _add_value(total=0, soft=False, bucket=pair_bucket)
    expected = 0.0
    for bucket, count in enumerate(shoe):
        if count <= 0:
            continue
        next_total, next_soft = _add_value(total=base_total, soft=base_soft, bucket=bucket)
        next_shoe = _decrement(shoe=shoe, bucket=bucket)
        if is_ace_pair:
            value = _stand_ev_unit(
                player_total=next_total, five_card_eligible=False, shoe=next_shoe, ctx=ctx
            )
        else:
            value = _player_optimal_ev(
                total=next_total, soft=next_soft, num_cards=2, shoe=next_shoe, ctx=ctx
            )
        expected += (count / shoe_total) * value
    return expected


def _split_estimate(*, hand_cards: list[Card], shoe: tuple[int, ...], ctx: _EvContext) -> float:
    """Estimates split EV as twice one independent split hand (shared-shoe approximation)."""
    pair_bucket = _bucket_for_rank(rank=hand_cards[0].rank)
    single = _single_split_hand_ev(
        pair_bucket=pair_bucket, is_ace_pair=pair_bucket == _ACE_BUCKET, shoe=shoe, ctx=ctx
    )
    return 2.0 * single


def _dist_to_outcome(*, dist: _DealerDist) -> DealerOutcome:
    """Converts the internal dealer distribution tuple into the public model."""
    return DealerOutcome(
        total_17_probability=dist[0],
        total_18_probability=dist[1],
        total_19_probability=dist[2],
        total_20_probability=dist[3],
        total_21_probability=dist[4],
        bust_probability=dist[_BUST_INDEX],
    )


def dealer_outcome_distribution(
    *, dealer_total: int, dealer_soft: bool, shoe: tuple[int, ...], memo: _DealerMemo | None = None
) -> DealerOutcome:
    """Public entry: exact dealer final-total distribution from a known dealer hand."""
    distribution = _dealer_distribution(
        total=dealer_total, soft=dealer_soft, shoe=shoe, memo={} if memo is None else memo
    )
    return _dist_to_outcome(dist=distribution)


def _select_recommended(*, ordered: tuple[ActionEv, ...]) -> ActionEv:
    """Picks the EV-max action, only preferring split past the safety margin."""
    best = ordered[0]
    if best.action != "split":
        return best
    non_split = [candidate for candidate in ordered if candidate.action != "split"]
    if not non_split:
        return best
    top_non_split = non_split[0]
    if best.expected_value > top_non_split.expected_value + SPLIT_EV_MARGIN:
        return best
    return top_non_split


def compute_action_evs(
    *,
    hand_cards: list[Card],
    dealer_cards: list[Card],
    shoe: list[Card],
    allowed_actions: tuple[BotAction, ...],
    doubled: bool,
) -> ActionEvAnalysis:
    """Computes the exact per-action EV analysis for one bot-player decision.

    EV is expressed in multiples of the base hand bet. The dealer outcome and
    every action value already account for the known dealer hole card, the true
    remaining shoe, H17 rules, and this table's five-card payouts.

    Args:
        hand_cards: The bot's active sub-hand cards.
        dealer_cards: The dealer's known cards (hole card first, then up-card).
        shoe: The true remaining undealt shoe.
        allowed_actions: Legal actions for the active hand.
        doubled: Whether the active hand has already doubled.

    Returns:
        The dealer distribution, per-allowed-action EVs, and the EV-max action.
    """
    shoe_counts = build_shoe_value_counts(shoe=shoe)
    ctx = _EvContext(
        dealer_total=hand_value(cards=dealer_cards),
        dealer_soft=is_soft_total(cards=dealer_cards)[0],
        dealer_memo={},
        player_memo={},
    )
    player_total = hand_value(cards=hand_cards)
    player_soft = is_soft_total(cards=hand_cards)[0]
    num_cards = len(hand_cards)
    dealer_dist = _dealer_dist_for(ctx=ctx, shoe=shoe_counts)
    evs: list[ActionEv] = []
    if "stand" in allowed_actions:
        stand_ev = _stand_ev_unit(
            player_total=player_total,
            five_card_eligible=num_cards >= 5 and not doubled,
            shoe=shoe_counts,
            ctx=ctx,
        )
        evs.append(
            ActionEv(action="stand", expected_value=2.0 * stand_ev if doubled else stand_ev)
        )
    if "hit" in allowed_actions:
        evs.append(
            ActionEv(
                action="hit",
                expected_value=_hit_action_ev(
                    total=player_total,
                    soft=player_soft,
                    num_cards=num_cards,
                    shoe=shoe_counts,
                    ctx=ctx,
                ),
            )
        )
    if "double" in allowed_actions:
        evs.append(
            ActionEv(
                action="double",
                expected_value=_double_ev(
                    total=player_total, soft=player_soft, shoe=shoe_counts, ctx=ctx
                ),
            )
        )
    if "surrender" in allowed_actions:
        evs.append(ActionEv(action="surrender", expected_value=-0.5))
    if "split" in allowed_actions:
        evs.append(
            ActionEv(
                action="split",
                expected_value=_split_estimate(hand_cards=hand_cards, shoe=shoe_counts, ctx=ctx),
                is_estimate=True,
                note="估計值: 兩手共用牌堆的獨立性近似",
            )
        )
    ordered = tuple(sorted(evs, key=lambda candidate: candidate.expected_value, reverse=True))
    recommended = _select_recommended(ordered=ordered)
    return ActionEvAnalysis(
        dealer_outcome=_dist_to_outcome(dist=dealer_dist),
        action_evs=ordered,
        recommended_action=recommended.action,
        recommended_expected_value=recommended.expected_value,
    )
