"""Blackjack expected-value engine for the bot player.

An LLM reasons poorly about multi-step no-replacement probability, so this pure
module turns the table state into decision-grade numbers: the dealer's H17
final-total distribution and the expected value of every legal action, measured
in multiples of the base hand bet.

The engine runs two passes. The exact pass knows the dealer hole card and drives
the recommended action only; it is the bot's private informational edge and is
never surfaced verbatim. The marginal pass integrates a hypothetical hole out
over the remaining shoe (the real hole is never added back, so every exposed
number depends only on the up-card and the shoe) and, when the dealer has
already peeked under an Ace/ten up-card, conditions on "no dealer Blackjack".
Only the marginal dealer distribution and marginal per-action EVs are exposed,
so nothing handed to the model can reveal or reconstruct the actual hole card.

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

from pydantic import Field, BaseModel, ConfigDict

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
# Marginal dealer distribution keyed by the unseen deck at the node; the
# up-card and peek flag are fixed per context, so the deck alone is the key.
_MarginalMemo = dict[tuple[int, ...], _DealerDist]


class _EvContext(BaseModel):
    """Fixed per-decision state threaded through the EV recursions.

    `marginalize` selects the dealer model. When False (exact pass) the known
    two-card dealer total drives the H17 distribution. When True (marginal pass)
    only the up-card is known and the hole is integrated out over the unseen
    deck, conditioning on no dealer Blackjack whenever the dealer peeked.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    marginalize: bool = Field(
        ..., description="Selects the marginal (hole integrated out) vs exact dealer model."
    )
    dealer_total: int = Field(..., description="Exact-pass dealer two-card total.")
    dealer_soft: bool = Field(
        ..., description="Whether the exact-pass dealer total counts an ace as 11."
    )
    up_total: int = Field(..., description="Dealer up-card total.")
    up_soft: bool = Field(..., description="Whether the up-card total counts an ace as 11.")
    up_bucket: int = Field(..., description="Value bucket index of the dealer up-card.")
    peek_no_blackjack: bool = Field(
        ..., description="Whether to condition on no dealer Blackjack after an Ace/ten peek."
    )
    dealer_memo: _DealerMemo = Field(..., description="Memo cache for exact dealer distributions.")
    player_memo: _PlayerMemo = Field(..., description="Memo cache for optimal player-hand EVs.")
    marginal_memo: _MarginalMemo = Field(
        ..., description="Memo cache for marginal dealer distributions by deck."
    )


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


_CARDS_PER_DECK: Final[int] = 52


def compute_true_count(*, shoe: list[Card]) -> float:
    """Returns the Hi-Lo true count of the cards already dealt out of a shoe.

    Hi-Lo assigns +1 to 2-6, 0 to 7-9, and -1 to ten-value cards and aces. A
    full balanced shoe sums to zero, so the running count of the dealt cards is
    the negative of the Hi-Lo sum still in `shoe`. The true count divides that
    running count by the decks remaining; a positive true count means the
    remaining shoe is rich in ten-value cards and aces, which favors the player.
    Returns 0.0 for an empty shoe (a neutral, just-shuffled count).
    """
    if not shoe:
        return 0.0
    counts = build_shoe_value_counts(shoe=shoe)
    low_remaining = counts[0] + counts[1] + counts[2] + counts[3] + counts[4]
    high_remaining = counts[_TEN_BUCKET] + counts[_ACE_BUCKET]
    running_count = high_remaining - low_remaining
    decks_remaining = len(shoe) / _CARDS_PER_DECK
    return running_count / decks_remaining if decks_remaining > 0 else 0.0


def _decrement(*, shoe: tuple[int, ...], bucket: int) -> tuple[int, ...]:
    """Returns a copy of the shoe vector with one card removed from a bucket."""
    mutable = list(shoe)
    mutable[bucket] -= 1
    return tuple(mutable)


def _hole_completes_blackjack(*, up_bucket: int, hole_bucket: int) -> bool:
    """Returns whether an up-card plus this hole would be a natural Blackjack."""
    return (up_bucket == _ACE_BUCKET and hole_bucket == _TEN_BUCKET) or (
        up_bucket == _TEN_BUCKET and hole_bucket == _ACE_BUCKET
    )


def _add_value(*, total: int, soft: bool, bucket: int) -> tuple[int, bool]:
    """Adds one drawn card to a running `(total, soft)` Blackjack hand state.

    Mirrors `hand_value`/`is_soft_total`: at most one ace is ever counted high,
    and any draw that would bust a soft hand demotes that high ace to 1. This
    includes drawing a second ace into a soft hand (e.g. soft 21 + ace becomes
    hard 12, not a bust).

    Args:
        total: Best current total with any high ace already counted as 11.
        soft: Whether an ace is currently counted as 11.
        bucket: Value bucket index of the drawn card.

    Returns:
        The updated `(total, soft)` state.
    """
    if bucket == _ACE_BUCKET and total + 11 <= 21:
        return total + 11, True
    increment = 1 if bucket == _ACE_BUCKET else _BUCKET_VALUES[bucket]
    new_total = total + increment
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
        # Defensive guard for a degraded/empty shoe (callers may pass one): the
        # dealer cannot draw, so treat the sub-17 total as terminal and never
        # divide by zero.
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


def _marginalize_hole(*, ctx: _EvContext, deck: tuple[int, ...], apply_peek: bool) -> _DealerDist:
    """Integrates the unknown hole out of the dealer distribution over a deck.

    The dealer's first (hole) card is drawn from `deck` and the dealer then
    plays out H17 from the remaining cards. When `apply_peek` is set, holes that
    would complete a natural Blackjack are excluded and the rest renormalized,
    matching the information a player has once an Ace/ten up-card round survives
    the dealer's peek.
    """
    accumulator = [0.0] * 6
    weight = 0.0
    for hole_bucket, count in enumerate(deck):
        if count <= 0:
            continue
        if apply_peek and _hole_completes_blackjack(
            up_bucket=ctx.up_bucket, hole_bucket=hole_bucket
        ):
            continue
        start_total, start_soft = _add_value(
            total=ctx.up_total, soft=ctx.up_soft, bucket=hole_bucket
        )
        child = _dealer_distribution(
            total=start_total,
            soft=start_soft,
            shoe=_decrement(shoe=deck, bucket=hole_bucket),
            memo=ctx.dealer_memo,
        )
        for position in range(6):
            accumulator[position] += count * child[position]
        weight += count
    if weight == 0.0:
        # Either an empty deck or a deck holding only the Blackjack-completing
        # rank under peek conditioning (a contradictory state). Fall back to the
        # up-card alone as a terminal hand so we never divide by zero.
        if apply_peek:
            return _marginalize_hole(ctx=ctx, deck=deck, apply_peek=False)
        return _dealer_distribution(
            total=ctx.up_total, soft=ctx.up_soft, shoe=deck, memo=ctx.dealer_memo
        )
    return tuple(value / weight for value in accumulator)


def _dealer_marginal_distribution(*, ctx: _EvContext, deck: tuple[int, ...]) -> _DealerDist:
    """Memoized marginal dealer distribution from the up-card and unseen deck."""
    cached = ctx.marginal_memo.get(deck)
    if cached is not None:
        return cached
    result = _marginalize_hole(ctx=ctx, deck=deck, apply_peek=ctx.peek_no_blackjack)
    ctx.marginal_memo[deck] = result
    return result


def _dealer_dist_for(*, ctx: _EvContext, shoe: tuple[int, ...]) -> _DealerDist:
    """Resolves the dealer distribution for the current pass and node deck.

    Exact pass: the known two-card dealer total plays out over `shoe`. Marginal
    pass: `shoe` is the unseen deck and the hole is integrated out of it.
    """
    if ctx.marginalize:
        return _dealer_marginal_distribution(ctx=ctx, deck=shoe)
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


def _evaluate_actions(  # noqa: PLR0913 -- mirrors the full per-action decision surface.
    *,
    ctx: _EvContext,
    deck: tuple[int, ...],
    hand_cards: list[Card],
    allowed_actions: tuple[BotAction, ...],
    doubled: bool,
    bet: int | None,
) -> list[ActionEv]:
    """Computes each legal action's EV for one pass over a deck."""
    player_total = hand_value(cards=hand_cards)
    player_soft = is_soft_total(cards=hand_cards)[0]
    num_cards = len(hand_cards)
    evs: list[ActionEv] = []
    if "stand" in allowed_actions:
        stand_ev = _stand_ev_unit(
            player_total=player_total,
            five_card_eligible=num_cards >= 5 and not doubled,
            shoe=deck,
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
                    total=player_total, soft=player_soft, num_cards=num_cards, shoe=deck, ctx=ctx
                ),
            )
        )
    if "double" in allowed_actions:
        evs.append(
            ActionEv(
                action="double",
                expected_value=_double_ev(
                    total=player_total, soft=player_soft, shoe=deck, ctx=ctx
                ),
            )
        )
    if "surrender" in allowed_actions:
        surrender_ev = -0.5 if bet is None else -((bet + 1) // 2) / bet
        evs.append(ActionEv(action="surrender", expected_value=surrender_ev))
    if "split" in allowed_actions:
        evs.append(
            ActionEv(
                action="split",
                expected_value=_split_estimate(hand_cards=hand_cards, shoe=deck, ctx=ctx),
                is_estimate=True,
                note="估計值: 兩手共用牌堆的獨立性近似",
            )
        )
    return evs


def _make_context(*, marginalize: bool, dealer_cards: list[Card], up_card: Card) -> _EvContext:
    """Builds a fixed per-pass EV context from the dealer's cards and up-card."""
    return _EvContext(
        marginalize=marginalize,
        dealer_total=hand_value(cards=dealer_cards),
        dealer_soft=is_soft_total(cards=dealer_cards)[0],
        up_total=hand_value(cards=[up_card]),
        up_soft=is_soft_total(cards=[up_card])[0],
        up_bucket=_bucket_for_rank(rank=up_card.rank),
        peek_no_blackjack=_bucket_for_rank(rank=up_card.rank) in (_ACE_BUCKET, _TEN_BUCKET),
        dealer_memo={},
        player_memo={},
        marginal_memo={},
    )


def compute_action_evs(  # noqa: PLR0913 -- one EV-engine entry point mirroring the full decision surface.
    *,
    hand_cards: list[Card],
    dealer_cards: list[Card],
    shoe: list[Card],
    allowed_actions: tuple[BotAction, ...],
    doubled: bool,
    bet: int | None = None,
) -> ActionEvAnalysis:
    """Computes the per-action EV analysis for one bot-player decision.

    EV is expressed in multiples of the base hand bet. Two passes run over H17
    rules and this table's five-card payouts:

    - The exact pass knows the hole card and selects `recommended_action`; this
      is the bot's private edge and is never surfaced directly.
    - The marginal pass integrates the hole out over the unseen deck (remaining
      shoe plus the hole as one anonymous unknown) and supplies the
      `dealer_outcome` distribution and every `action_evs` value. These hole-free
      numbers are all that the model ever sees, so they cannot reveal the hole.

    `recommended_expected_value` is reported as the recommended action's marginal
    EV to stay consistent with the exposed numbers.

    Args:
        hand_cards: The bot's active sub-hand cards.
        dealer_cards: The dealer's cards (hole card first, then up-card).
        shoe: The true remaining undealt shoe.
        allowed_actions: Legal actions for the active hand.
        doubled: Whether the active hand has already doubled.
        bet: The base hand bet, used to price surrender from its actual rounded
            half-bet loss (`settle_hand` charges `-((bet + 1) // 2)`). When None
            the theoretical -0.5 is used.

    Returns:
        The marginal dealer distribution and per-action EVs, plus the EV-max
        action selected from the exact (hole-aware) pass.
    """
    shoe_counts = build_shoe_value_counts(shoe=shoe)
    up_card = dealer_cards[1] if len(dealer_cards) >= 2 else dealer_cards[0]

    # Exact pass: the known two-card dealer total drives the recommendation only.
    exact_ctx = _make_context(marginalize=False, dealer_cards=dealer_cards, up_card=up_card)
    exact_evs = _evaluate_actions(
        ctx=exact_ctx,
        deck=shoe_counts,
        hand_cards=hand_cards,
        allowed_actions=allowed_actions,
        doubled=doubled,
        bet=bet,
    )
    exact_ordered = tuple(
        sorted(exact_evs, key=lambda candidate: candidate.expected_value, reverse=True)
    )
    recommended = _select_recommended(ordered=exact_ordered)

    # Marginal pass: a hypothetical hole is integrated out over the remaining
    # shoe. The real hole is never added back, so every exposed number depends
    # only on the up-card and the shoe and cannot reveal the actual hole.
    marginal_ctx = _make_context(marginalize=True, dealer_cards=dealer_cards, up_card=up_card)
    marginal_evs = _evaluate_actions(
        ctx=marginal_ctx,
        deck=shoe_counts,
        hand_cards=hand_cards,
        allowed_actions=allowed_actions,
        doubled=doubled,
        bet=bet,
    )
    marginal_ordered = tuple(
        sorted(marginal_evs, key=lambda candidate: candidate.expected_value, reverse=True)
    )
    dealer_dist = _dealer_dist_for(ctx=marginal_ctx, shoe=shoe_counts)
    # Surface the recommended action's marginal EV. The default stays a marginal
    # value so an exact (hole-aware) EV can never reach the exposed field.
    marginal_by_action = {item.action: item.expected_value for item in marginal_ordered}
    shown_recommended_ev = marginal_by_action.get(
        recommended.action, marginal_ordered[0].expected_value if marginal_ordered else 0.0
    )
    return ActionEvAnalysis(
        dealer_outcome=_dist_to_outcome(dist=dealer_dist),
        action_evs=marginal_ordered,
        recommended_action=recommended.action,
        recommended_expected_value=shown_recommended_ev,
    )
