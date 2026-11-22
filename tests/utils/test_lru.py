"""Tests for the bounded LRU cache."""

import pytest
from pydantic import ValidationError

from discordbot.utils.lru import BoundedLRU


def test_evicts_least_recently_used_and_refreshes_on_get_and_overwrite() -> None:
    """`get` and overwriting `set` both move an entry ahead of older keys before eviction."""
    cache: BoundedLRU[str, int] = BoundedLRU(max_entries=3)
    cache.set(key="a", value=1)
    cache.set(key="b", value=2)
    cache.set(key="c", value=3)

    assert cache.get(key="a") == 1
    cache.set(key="b", value=20)
    cache.set(key="d", value=4)

    assert len(cache) == 3
    assert "c" not in cache
    assert cache.peek(key="a") == 1
    assert cache.peek(key="b") == 20
    assert cache.peek(key="d") == 4


def test_peek_and_contains_do_not_refresh_recency() -> None:
    """`peek` and `in` inspect entries without altering eviction order, unlike `touch`."""
    cache: BoundedLRU[str, str] = BoundedLRU(max_entries=2)
    cache.set(key="a", value="first")
    cache.set(key="b", value="second")

    assert "a" in cache
    assert cache.peek(key="a") == "first"
    cache.set(key="c", value="third")

    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache

    cache.touch(key="b")
    cache.touch(key="missing")
    cache.set(key="d", value="fourth")

    assert "b" in cache
    assert "c" not in cache
    assert "d" in cache


def test_get_handles_none_values_and_pop_and_clear_remove_entries() -> None:
    """A stored `None` value stays present and refreshes recency; `pop` and `clear` drop items."""
    cache: BoundedLRU[str, None] = BoundedLRU(max_entries=2)
    cache.set(key="a", value=None)
    cache.set(key="b", value=None)

    assert cache.get(key="a") is None
    cache.set(key="c", value=None)
    assert "a" in cache
    assert "b" not in cache

    assert cache.pop(key="a") is None
    assert "a" not in cache
    assert cache.pop(key="missing") is None

    cache.clear()
    assert len(cache) == 0


def test_max_entries_must_be_positive() -> None:
    """A non-positive capacity is rejected at construction."""
    with pytest.raises(ValidationError):
        BoundedLRU(max_entries=0)
