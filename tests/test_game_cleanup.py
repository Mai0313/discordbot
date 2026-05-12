"""Tests for casino game response cleanup helpers."""

from typing import cast
import asyncio

import pytest
import nextcord
from nextcord import Interaction

from discordbot.cogs import economy
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


class _FollowupStub:
    """Minimal followup stub that records send arguments."""

    def __init__(self) -> None:
        self.message = object()
        self.sent_wait: bool | None = None
        self.sent_embed: nextcord.Embed | None = None

    async def send(self, *, embed: nextcord.Embed, wait: bool) -> object:
        """Records the embed send and returns the message object."""
        self.sent_embed = embed
        self.sent_wait = wait
        return self.message


class _InteractionStub:
    """Minimal interaction shape for expiring economy followups."""

    def __init__(self) -> None:
        self.followup = _FollowupStub()


async def test_delete_game_message_after_waits_then_deletes() -> None:
    """Game response cleanup deletes the message after the configured delay."""
    message = _DeletableMessageStub()

    await cleanup.delete_game_message_after(message=message, delay=0)

    assert message.delete_calls == 1


async def test_send_expiring_followup_waits_for_message_and_schedules_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Game-related economy embeds must retrieve their message before cleanup."""
    scheduled_messages: list[object] = []

    def fake_schedule_game_message_delete(*, message: object, delay: float = 180) -> None:
        scheduled_messages.append(message)

    monkeypatch.setattr(
        target=economy,
        name="schedule_game_message_delete",
        value=fake_schedule_game_message_delete,
    )
    interaction = _InteractionStub()
    embed = nextcord.Embed(title="balance")

    await economy._send_expiring_followup(
        interaction=cast("Interaction", interaction), embed=embed
    )

    assert interaction.followup.sent_wait is True
    assert interaction.followup.sent_embed is embed
    assert scheduled_messages == [interaction.followup.message]


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
