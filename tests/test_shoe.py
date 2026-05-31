"""Tests for the in-memory per-channel persistent Blackjack shoe store."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random

from discordbot.typings.games import Card
from discordbot.cogs._games.shoe import RESHUFFLE_THRESHOLD_CARDS, BlackjackShoeStore


def _card(rank: str) -> Card:
    """Builds a card with an arbitrary suit for shoe tests."""
    return Card(rank=rank, suit="♠")


def test_first_take_builds_a_fresh_shoe_without_announcing_a_reshuffle() -> None:
    """A channel with no stored shoe gets a full fresh shoe and no reshuffle flag."""
    store = BlackjackShoeStore()
    shoe, reshuffled = store.take_shoe(channel_id=1, rng=Random(0))

    assert len(shoe) == 208
    assert reshuffled is False


def test_take_returns_the_stored_shoe_above_the_threshold() -> None:
    """A healthy stored shoe is handed back unchanged and removed from the store."""
    store = BlackjackShoeStore()
    stored = [_card(rank="10")] * (RESHUFFLE_THRESHOLD_CARDS + 5)
    store.save_shoe(channel_id=7, cards=stored)

    shoe, reshuffled = store.take_shoe(channel_id=7, rng=Random(0))

    assert shoe is stored
    assert reshuffled is False
    # Taking removes it so a concurrent game cannot share the same list.
    assert 7 not in store.shoes


def test_take_reshuffles_and_announces_below_the_threshold() -> None:
    """A worn-down shoe triggers a fresh build flagged as a reshuffle."""
    store = BlackjackShoeStore()
    store.save_shoe(channel_id=3, cards=[_card(rank="5")] * (RESHUFFLE_THRESHOLD_CARDS - 1))

    shoe, reshuffled = store.take_shoe(channel_id=3, rng=Random(0))

    assert len(shoe) == 208
    assert reshuffled is True


def test_save_then_take_round_trips_card_depletion() -> None:
    """Saving a depleted shoe lets the next round continue from the same cards."""
    store = BlackjackShoeStore()
    remaining = [_card(rank="A")] * (RESHUFFLE_THRESHOLD_CARDS + 1)
    store.save_shoe(channel_id=9, cards=remaining)

    shoe, reshuffled = store.take_shoe(channel_id=9, rng=Random(0))

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
