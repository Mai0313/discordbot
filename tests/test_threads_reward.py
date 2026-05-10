"""Tests for the parse_threads reward cooldown logic."""

from datetime import UTC, datetime, timedelta

import pytest

from discordbot.cogs.parse_threads import ThreadsCogs


@pytest.fixture
def cog() -> ThreadsCogs:
    """Builds a bare ThreadsCogs without invoking commands.Cog setup machinery."""
    instance = ThreadsCogs.__new__(ThreadsCogs)
    instance._recent_awards = {}
    return instance


def test_first_claim_succeeds(cog: ThreadsCogs) -> None:
    """A brand-new (user, url) pair is always allowed."""
    assert cog._claim_award_slot(user_id=1, url="https://threads.com/@x/post/abc") is True


def test_second_claim_within_cooldown_fails(cog: ThreadsCogs) -> None:
    """The same (user, url) pair is rejected during the cooldown window."""
    url = "https://threads.com/@x/post/abc"
    assert cog._claim_award_slot(user_id=1, url=url) is True
    assert cog._claim_award_slot(user_id=1, url=url) is False


def test_different_url_same_user_is_independent(cog: ThreadsCogs) -> None:
    """Cooldown is per-(user, url), not per-user."""
    assert cog._claim_award_slot(user_id=1, url="https://threads.com/@x/post/a") is True
    assert cog._claim_award_slot(user_id=1, url="https://threads.com/@x/post/b") is True


def test_different_user_same_url_is_independent(cog: ThreadsCogs) -> None:
    """Two different users sharing the same link both get a reward."""
    url = "https://threads.com/@x/post/abc"
    assert cog._claim_award_slot(user_id=1, url=url) is True
    assert cog._claim_award_slot(user_id=2, url=url) is True


def test_expired_entry_is_evicted_and_re_claimable(cog: ThreadsCogs) -> None:
    """Past-cooldown entries get cleaned up and the slot becomes claimable again."""
    url = "https://threads.com/@x/post/abc"
    # Pre-seed with a stale timestamp two hours ago.
    cog._recent_awards[(1, url)] = datetime.now(tz=UTC) - timedelta(hours=2)
    assert cog._claim_award_slot(user_id=1, url=url) is True


def test_stale_entries_are_cleaned_during_unrelated_writes(cog: ThreadsCogs) -> None:
    """Writing for one key sweeps stale entries even for other keys."""
    stale_url = "https://threads.com/@x/post/old"
    fresh_url = "https://threads.com/@x/post/new"
    cog._recent_awards[(1, stale_url)] = datetime.now(tz=UTC) - timedelta(hours=2)
    cog._claim_award_slot(user_id=1, url=fresh_url)
    assert (1, stale_url) not in cog._recent_awards
