"""Tests for casino game response cleanup helpers."""

import asyncio

import pytest
import nextcord

from discordbot.cogs._games import cleanup


class _ResponseStub:
    """Minimal HTTP response shape used by nextcord exceptions."""

    status = 404
    reason = "Not Found"


class _DeletableMessageStub:
    """Minimal message stub that records deletion."""

    def __init__(self) -> None:
        self.delete_calls = 0

    async def delete(self) -> None:
        """Records a Discord message deletion."""
        self.delete_calls += 1


class _AlreadyDeletedMessageStub:
    """Message stub that behaves like Discord already removed it."""

    async def delete(self) -> None:
        """Raises the same exception nextcord raises for missing messages."""
        raise nextcord.NotFound(response=_ResponseStub(), message="missing")


async def test_delete_game_message_after_waits_then_deletes() -> None:
    """Game response cleanup deletes the message after the configured delay."""
    message = _DeletableMessageStub()

    await cleanup.delete_game_message_after(message=message, delay=0)

    assert message.delete_calls == 1


async def test_delete_game_message_after_ignores_already_deleted_message() -> None:
    """Manual deletion before cleanup should not surface as a task failure."""
    await cleanup.delete_game_message_after(message=_AlreadyDeletedMessageStub(), delay=0)


async def test_schedule_game_message_delete_uses_default_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduling uses the shared three-minute TTL by default."""
    scheduled_delay: float | None = None

    async def fake_delete_game_message_after(*, message: object, delay: float) -> None:
        nonlocal scheduled_delay
        scheduled_delay = delay

    monkeypatch.setattr(
        target=cleanup, name="delete_game_message_after", value=fake_delete_game_message_after
    )

    cleanup.schedule_game_message_delete(message=_DeletableMessageStub())
    await asyncio.sleep(delay=0)

    assert scheduled_delay == cleanup.GAME_RESPONSE_TTL_SECONDS
