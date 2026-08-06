"""Deterministic tests for the in-memory per-channel persistent Blackjack shoe store.

Pins `cogs/games/shoe.py`, the one piece of Blackjack state that outlives a round: the shoe a
channel carries between tables so the bot's Hi-Lo counting and Kelly bet sizing have real
depletion to read. Every case builds its own shoe as a run of identical ranks at an exact offset
from `RESHUFFLE_THRESHOLD_CARDS`, so an assertion is a boundary condition or a card count rather
than a sample; the seeded `Random` only feeds `build_shoe`'s shuffle, which nothing here reads.

Four behaviors are worth pinning. The reshuffle policy: a fresh shoe is the full 208 cards of the
4-deck build, a stored shoe under the threshold is cut before the round starts (the cut is what
keeps a shoe from emptying mid-round into `draw_card`'s infinite fallback and corrupting the
count), and a channel's first shoe is deliberately not flagged as a reshuffle, or every new
channel would open by announcing one. Taking removes the shoe from the store, so two tables open
in the same channel deal from separate lists instead of interleaving draws on a shared one, and
saving it back is what carries depletion into the next round. `true_count` reads the stored shoe
without taking it, since bet sizing consults the count before a table is dealt, and reports a
neutral 0.0 whenever the next round would deal from a fresh shoe anyway. And the generation
token, the ordering guard: two overlapping tables settle in whatever order their rounds finish,
so a save carrying an older token is dropped rather than allowed to clobber the newer shoe.
"""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random

from discordbot.typings.games import Card
from discordbot.cogs.games.shoe import RESHUFFLE_THRESHOLD_CARDS, BlackjackShoeStore


def _card(rank: str) -> Card:
    """Builds a card of the given rank; the suit is arbitrary since nothing here counts suits.

    Returns:
        A `Card` of that rank in spades.
    """
    return Card(rank=rank, suit="♠")


def test_first_take_builds_a_fresh_shoe_without_announcing_a_reshuffle() -> None:
    """A channel with no stored shoe gets a full fresh shoe and no reshuffle flag."""
    store = BlackjackShoeStore()
    shoe, reshuffled, _generation = store.take_shoe(channel_id=1, rng=Random(0))

    assert len(shoe) == 208
    assert reshuffled is False


def test_take_returns_the_stored_shoe_above_the_threshold() -> None:
    """A healthy stored shoe is handed back unchanged and removed from the store."""
    store = BlackjackShoeStore()
    stored = [_card(rank="10")] * (RESHUFFLE_THRESHOLD_CARDS + 5)
    store.save_shoe(channel_id=7, cards=stored)

    shoe, reshuffled, _generation = store.take_shoe(channel_id=7, rng=Random(0))

    assert shoe == stored
    assert reshuffled is False
    # Taking removes it so a concurrent game cannot share the same list.
    assert 7 not in store.shoes


def test_take_reshuffles_and_announces_below_the_threshold() -> None:
    """A worn-down shoe triggers a fresh build flagged as a reshuffle."""
    store = BlackjackShoeStore()
    store.save_shoe(channel_id=3, cards=[_card(rank="5")] * (RESHUFFLE_THRESHOLD_CARDS - 1))

    shoe, reshuffled, _generation = store.take_shoe(channel_id=3, rng=Random(0))

    assert len(shoe) == 208
    assert reshuffled is True


def test_save_then_take_round_trips_card_depletion() -> None:
    """Saving a depleted shoe lets the next round continue from the same cards."""
    store = BlackjackShoeStore()
    remaining = [_card(rank="A")] * (RESHUFFLE_THRESHOLD_CARDS + 1)
    store.save_shoe(channel_id=9, cards=remaining)

    shoe, reshuffled, _generation = store.take_shoe(channel_id=9, rng=Random(0))

    assert shoe == remaining
    assert reshuffled is False


def test_true_count_is_neutral_without_a_countable_shoe() -> None:
    """A missing or about-to-reshuffle shoe reads as a neutral count for bet sizing."""
    store = BlackjackShoeStore()

    assert store.true_count(channel_id=1) == 0.0

    store.save_shoe(channel_id=1, cards=[_card(rank="10")] * (RESHUFFLE_THRESHOLD_CARDS - 1))
    assert store.true_count(channel_id=1) == 0.0


def test_true_count_reads_a_countable_stored_shoe() -> None:
    """A ten-rich stored shoe above the threshold yields a positive true count."""
    store = BlackjackShoeStore()
    store.save_shoe(channel_id=1, cards=[_card(rank="10")] * (RESHUFFLE_THRESHOLD_CARDS + 4))

    assert store.true_count(channel_id=1) > 0


def test_older_round_does_not_clobber_a_newer_shoe() -> None:
    """An earlier-started overlapping round cannot overwrite a newer table's saved shoe."""
    store = BlackjackShoeStore()
    # Two tables open in the same channel: the first take pops the (empty) channel, the
    # second take starts from a fresh shoe; both carry their own generation token.
    _first_shoe, _first_reshuffled, first_generation = store.take_shoe(channel_id=5, rng=Random(0))
    _second_shoe, _second_reshuffled, second_generation = store.take_shoe(
        channel_id=5, rng=Random(1)
    )
    assert second_generation > first_generation

    newer = [_card(rank="K")] * (RESHUFFLE_THRESHOLD_CARDS + 2)
    older = [_card(rank="2")] * (RESHUFFLE_THRESHOLD_CARDS + 2)

    # The newer table settles first and persists its shoe.
    store.save_shoe(channel_id=5, cards=newer, generation=second_generation)
    # The older table settles later and must not clobber the newer shoe.
    store.save_shoe(channel_id=5, cards=older, generation=first_generation)

    assert store.shoes[5] == newer
