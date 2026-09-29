"""Tests for the in-memory per-channel persistent Blackjack shoe store."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random

from discordbot.cogs.games.shoe import RESHUFFLE_THRESHOLD_CARDS, BlackjackShoeStore

from tests.helpers.games import card


def test_first_take_builds_a_fresh_shoe_without_announcing_a_reshuffle() -> None:
    """A channel with no stored shoe gets a full fresh shoe and no reshuffle flag."""
    store = BlackjackShoeStore()
    shoe, reshuffled, _generation = store.take_shoe(channel_id=1, rng=Random(0))

    assert len(shoe) == 208
    assert reshuffled is False


def test_take_returns_the_stored_shoe_down_to_the_threshold() -> None:
    """A stored shoe holding exactly the threshold is handed back unchanged and removed."""
    store = BlackjackShoeStore()
    stored = [card(rank="10")] * RESHUFFLE_THRESHOLD_CARDS
    store.save_shoe(channel_id=7, cards=stored)

    shoe, reshuffled, _generation = store.take_shoe(channel_id=7, rng=Random(0))

    assert shoe == stored
    assert reshuffled is False
    # Taking removes it so a concurrent game cannot share the same list.
    assert 7 not in store.shoes


def test_take_reshuffles_and_announces_below_the_threshold() -> None:
    """A worn-down shoe triggers a fresh build flagged as a reshuffle."""
    store = BlackjackShoeStore()
    store.save_shoe(channel_id=3, cards=[card(rank="5")] * (RESHUFFLE_THRESHOLD_CARDS - 1))

    shoe, reshuffled, _generation = store.take_shoe(channel_id=3, rng=Random(0))

    assert len(shoe) == 208
    assert reshuffled is True


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
    _first_shoe, _first_reshuffled, first_generation = store.take_shoe(channel_id=5, rng=Random(0))
    _second_shoe, _second_reshuffled, second_generation = store.take_shoe(
        channel_id=5, rng=Random(1)
    )
    assert second_generation > first_generation

    newer = [card(rank="K")] * (RESHUFFLE_THRESHOLD_CARDS + 2)
    older = [card(rank="2")] * (RESHUFFLE_THRESHOLD_CARDS + 2)

    # The newer table settles first and persists its shoe.
    store.save_shoe(channel_id=5, cards=newer, generation=second_generation)
    # The older table settles later and must not clobber the newer shoe.
    store.save_shoe(channel_id=5, cards=older, generation=first_generation)

    assert store.shoes[5] == newer
