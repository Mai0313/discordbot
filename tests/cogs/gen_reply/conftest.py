"""Fixtures shared by every test in this directory."""

from pathlib import Path

import pytest

from tests.helpers.reply_fakes import FakeMessage


@pytest.fixture(autouse=True)
def memory_isolated(memory_isolated_dir: Path) -> Path:
    """Keeps every test here off the live memory store.

    A reply always reads memory, with no caller-side switch to turn it off.
    """
    return memory_isolated_dir


@pytest.fixture(autouse=True)
def fake_messages_count_as_replies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lets a `FakeMessage` pass the replied-to lookup's `isinstance(resolved, Message)` check."""
    monkeypatch.setattr("discordbot.cogs.gen_reply.references.Message", FakeMessage)


@pytest.fixture
def no_memory_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps the answer from scheduling its fire-and-forget memory review."""
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.answer.schedule_memory_update", lambda **_: None
    )
