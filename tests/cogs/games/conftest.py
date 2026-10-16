"""Fixtures the games tests share."""

import pytest

from tests.helpers.message_cleanup import ScheduledDeletes, record_scheduled_deletes


@pytest.fixture
def scheduled_cleanups(monkeypatch: pytest.MonkeyPatch) -> ScheduledDeletes:
    """Records each table the view hands to the public-message cleanup instead of scheduling it."""
    return record_scheduled_deletes(monkeypatch=monkeypatch)
