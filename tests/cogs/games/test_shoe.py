"""Tests for the in-memory per-channel persistent Blackjack shoe store."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random
from itertools import count

from discordbot.cogs.games.shoe import RESHUFFLE_THRESHOLD_CARDS, BlackjackShoeStore
from discordbot.typings.economy import MAX_SINGLE_BET
from discordbot.cogs.games.blackjack import (
    CARD_RANKS,
    SHOE_DECK_COUNT,
    BlackjackRound,
    can_split,
    dealer_must_hit,
    is_five_card_win,
)
from discordbot.cogs.games.blackjack_views import MAX_BLACKJACK_PLAYERS

from tests.helpers.games import card, seat


def test_first_take_builds_a_fresh_shoe() -> None:
    """A channel with no stored shoe gets a full fresh shoe."""
    store = BlackjackShoeStore()
    shoe, _generation = store.take_shoe(channel_id=1, rng=Random(x=0))

    assert len(shoe) == 208


def test_take_returns_the_stored_shoe_down_to_the_threshold() -> None:
    """A stored shoe holding exactly the threshold is handed back unchanged and removed."""
    store = BlackjackShoeStore()
    stored = [card(rank="10")] * RESHUFFLE_THRESHOLD_CARDS
    store.save_shoe(channel_id=7, cards=stored)

    shoe, _generation = store.take_shoe(channel_id=7, rng=Random(x=0))

    assert shoe == stored
    # Taking removes it so a concurrent game cannot share the same list.
    assert 7 not in store.shoes


def test_take_reshuffles_below_the_threshold() -> None:
    """A worn-down shoe is replaced by a fresh build."""
    store = BlackjackShoeStore()
    store.save_shoe(channel_id=3, cards=[card(rank="5")] * (RESHUFFLE_THRESHOLD_CARDS - 1))

    shoe, _generation = store.take_shoe(channel_id=3, rng=Random(x=0))

    assert len(shoe) == 208


def test_true_count_is_neutral_without_a_countable_shoe() -> None:
    """A missing or about-to-reshuffle shoe reads as a neutral count for bet sizing."""
    store = BlackjackShoeStore()

    assert store.true_count(channel_id=1) == 0.0

    store.save_shoe(channel_id=1, cards=[card(rank="10")] * (RESHUFFLE_THRESHOLD_CARDS - 1))
    assert store.true_count(channel_id=1) == 0.0


def test_true_count_reads_a_countable_stored_shoe() -> None:
    """A ten-rich stored shoe holding exactly the threshold yields a positive true count."""
    store = BlackjackShoeStore()
    store.save_shoe(channel_id=1, cards=[card(rank="10")] * RESHUFFLE_THRESHOLD_CARDS)

    assert store.true_count(channel_id=1) > 0


def test_older_round_does_not_clobber_a_newer_shoe() -> None:
    """An earlier-started overlapping round cannot overwrite a newer table's saved shoe."""
    store = BlackjackShoeStore()
    # Two tables open in the same channel: the first take pops the (empty) channel, the
    # second take starts from a fresh shoe; both carry their own generation token.
    _first_shoe, first_generation = store.take_shoe(channel_id=5, rng=Random(x=0))
    _second_shoe, second_generation = store.take_shoe(channel_id=5, rng=Random(x=1))
    assert second_generation > first_generation

    newer = [card(rank="K")] * (RESHUFFLE_THRESHOLD_CARDS + 2)
    older = [card(rank="2")] * (RESHUFFLE_THRESHOLD_CARDS + 2)

    # The newer table settles first and persists its shoe.
    store.save_shoe(channel_id=5, cards=newer, generation=second_generation)
    # The older table settles later and must not clobber the newer shoe.
    store.save_shoe(channel_id=5, cards=older, generation=first_generation)

    assert store.shoes[5] == newer


def test_a_put_back_shoe_never_displaces_another_rounds_shoe() -> None:
    """A shoe returned from a start that never showed a card yields to every round that settles.

    The table already in play when the start failed holds the channel's real shoe, while the
    failed start had to take a fresh one.
    """
    store = BlackjackShoeStore()
    _in_play, in_play_generation = store.take_shoe(channel_id=5, rng=Random(x=0))
    unshown, _generation = store.take_shoe(channel_id=5, rng=Random(x=1))
    store.put_back_shoe(channel_id=5, cards=unshown)

    settled = [card(rank="2")] * (RESHUFFLE_THRESHOLD_CARDS + 2)
    store.save_shoe(channel_id=5, cards=settled, generation=in_play_generation)
    assert store.shoes[5] == settled

    store.put_back_shoe(channel_id=5, cards=unshown)
    assert store.shoes[5] == settled


def _longest_hand_the_dealer_must_draw_on() -> int:
    """Returns the most cards an H17 dealer can hold while the rules still make it draw.

    Searched rather than reasoned out, because it is not the number anyone reaches by hand:
    eleven aces and a five is hard 16 and twelve cards, where a hand of small cards runs out
    sooner and a hand of aces stands early on the soft total. A rules change re-derives it.
    """
    per_shoe = 4 * SHOE_DECK_COUNT

    def must_draw(counts: dict[str, int]) -> bool:
        return dealer_must_hit(
            cards=[card(rank=rank) for rank, held in counts.items() for _ in range(held)]
        )

    longest = 0
    seen: set[tuple[tuple[str, int], ...]] = set()
    # A hand is its multiset: `must_draw` cannot read an order, so one ordering settles them all.
    stack: list[dict[str, int]] = [{}]
    while stack:
        counts = stack.pop()
        key = tuple(sorted(counts.items()))
        if key in seen:
            continue
        seen.add(key)
        held = sum(counts.values())
        # The dealer's own first two cards are dealt, not drawn, so they are unconstrained.
        if held >= 2 and not must_draw(counts=counts):
            continue
        longest = max(longest, held)
        for rank in CARD_RANKS:
            if counts.get(rank, 0) < per_shoe:
                stack.append({**counts, rank: counts.get(rank, 0) + 1})
    return longest


def test_the_reshuffle_threshold_outlasts_the_longest_round_a_full_table_can_deal() -> None:
    """A round that starts at the threshold never draws past the end of its shoe.

    Past the end `draw_card` deals from a notional infinite deck, which corrupts the count the
    bot bets on. The longest round is a full table where every seat splits and takes both
    hands to the card count at which the rules stand them, while the dealer is forced to the
    longest hand it must still draw on and then draws the card that hand is owed. Each factor
    is read off the rules rather than restated, so raising the seat cap re-checks this.
    """
    cards_per_hand = next(
        held for held in count(start=1) if is_five_card_win(cards=[card(rank="2")] * held)
    )
    pair_round = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat()], shoe=[card(rank="8")] * 2
    )
    pair_round.players[0].hands[0].cards = [card(rank="8"), card(rank="8")]
    pair_round.split(user_id=1)
    split_hands = pair_round.players[0].hands
    assert not any(
        can_split(hand=hand, balance_remaining=MAX_SINGLE_BET) for hand in split_hands
    ), "a split hand could split again, so a seat can hold more than two hands"
    dealer_cards = _longest_hand_the_dealer_must_draw_on() + 1

    longest_round = MAX_BLACKJACK_PLAYERS * len(split_hands) * cards_per_hand + dealer_cards

    assert longest_round <= RESHUFFLE_THRESHOLD_CARDS, (
        f"a full table can deal {longest_round} cards in one round, past the "
        f"{RESHUFFLE_THRESHOLD_CARDS} a round may start with; raise RESHUFFLE_THRESHOLD_CARDS"
    )
