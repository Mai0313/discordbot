"""Tests for casino game response cleanup helpers."""

from typing import TYPE_CHECKING, cast
import asyncio
from pathlib import Path

import pytest
import nextcord
from nextcord import Message, Interaction

from discordbot.cogs import economy
from discordbot.cogs._games import cleanup

if TYPE_CHECKING:
    from nextcord.ext import commands


class _ResponseStub:
    """Minimal HTTP response shape used by nextcord exceptions."""

    status = 404
    reason = "Not Found"


class _DeletableMessageStub:
    """Minimal message stub that records deletion."""

    def __init__(self, message_id: int = 123, channel_id: int = 456) -> None:
        """Initializes message and channel IDs for cleanup tracking."""
        self.id = message_id
        self.channel = _ChannelStub(channel_id=channel_id)
        self.delete_calls = 0

    async def delete(self) -> None:
        """Records a Discord message deletion."""
        self.delete_calls += 1


class _AlreadyDeletedMessageStub:
    """Message stub that behaves like Discord already removed it."""

    def __init__(self) -> None:
        """Initializes a message identity that will fail deletion."""
        self.id = 789
        self.channel = _ChannelStub(channel_id=456)

    async def delete(self) -> None:
        """Raises the same exception nextcord raises for missing messages."""
        raise nextcord.NotFound(response=_ResponseStub(), message="missing")


class _ChannelStub:
    """Minimal channel shape for persistent cleanup identity."""

    def __init__(self, channel_id: int) -> None:
        """Stores the Discord channel ID."""
        self.id = channel_id


class _FetchedMessageStub:
    """Fetched message returned by a fake channel for startup cleanup."""

    def __init__(self, channel_id: int, message_id: int, deleted: list[tuple[int, int]]) -> None:
        """Initializes a fetched message with shared deletion recording."""
        self.channel_id = channel_id
        self.message_id = message_id
        self.deleted = deleted

    async def delete(self) -> None:
        """Records deletion by channel/message pair."""
        self.deleted.append((self.channel_id, self.message_id))


class _FetchMessageChannelStub:
    """Minimal message channel returned by the fake bot."""

    def __init__(self, channel_id: int, deleted: list[tuple[int, int]]) -> None:
        """Stores channel identity and the shared deletion recorder."""
        self.channel_id = channel_id
        self.deleted = deleted

    async def fetch_message(self, message_id: int, /) -> _FetchedMessageStub:
        """Returns a fetched message stub."""
        return _FetchedMessageStub(
            channel_id=self.channel_id, message_id=message_id, deleted=self.deleted
        )


class _BotStub:
    """Minimal bot shape for startup cleanup."""

    def __init__(self, cached_channel: object | None = None) -> None:
        """Initializes cached-channel behavior and cleanup call records."""
        self.deleted: list[tuple[int, int]] = []
        self.cached_channel = cached_channel
        self.fetch_calls: list[int] = []

    def get_channel(self, channel_id: int, /) -> object | None:
        """Returns the configured cached channel."""
        return self.cached_channel

    async def fetch_channel(self, channel_id: int, /) -> _FetchMessageChannelStub:
        """Returns a concrete message channel stub."""
        self.fetch_calls.append(channel_id)
        return _FetchMessageChannelStub(channel_id=channel_id, deleted=self.deleted)


class _UnfetchableBotStub:
    """Bot stub that resolves a channel object that cannot fetch messages."""

    def get_channel(self, channel_id: int, /) -> None:
        """Returns no cached channel."""
        return

    async def fetch_channel(self, channel_id: int, /) -> object:
        """Returns a non-messageable channel shape."""
        return object()


class _FollowupStub:
    """Minimal followup stub that records send arguments."""

    def __init__(self) -> None:
        """Initializes followup send records."""
        self.message = object()
        self.sent_wait: bool | None = None
        self.sent_ephemeral: bool | None = None
        self.sent_embed: nextcord.Embed | None = None

    async def send(
        self, embed: nextcord.Embed, wait: bool = False, ephemeral: bool = False
    ) -> object:
        """Records the embed send and returns the message object."""
        self.sent_embed = embed
        self.sent_wait = wait
        self.sent_ephemeral = ephemeral
        return self.message


class _InteractionStub:
    """Minimal interaction shape for expiring economy followups."""

    def __init__(self) -> None:
        """Initializes the followup stub used by the helper under test."""
        self.followup = _FollowupStub()


@pytest.fixture(autouse=True)
def isolated_cleanup_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps cleanup DB writes out of the real data directory."""
    monkeypatch.setattr(
        target=cleanup, name="_PENDING_GAME_MESSAGE_DB_PATH", value=tmp_path / "game_cleanup.db"
    )


async def test_delete_game_message_after_waits_then_deletes() -> None:
    """Game response cleanup deletes the message after the configured delay."""
    message = _DeletableMessageStub()

    await cleanup.delete_game_message_after(message=cast("Message", message), delay=0)

    assert message.delete_calls == 1


async def test_track_game_message_persists_message_identity() -> None:
    """Game response tracking stores the channel/message pair needed for restart cleanup."""
    message = _DeletableMessageStub(message_id=10, channel_id=20)

    record = await cleanup.track_game_message(message=cast("Message", message))

    assert record == cleanup.PendingGameMessage(channel_id=20, message_id=10)
    assert await cleanup.list_pending_game_messages() == [
        cleanup.PendingGameMessage(channel_id=20, message_id=10)
    ]


async def test_delete_game_message_after_forgets_successful_cleanup() -> None:
    """Successful TTL cleanup removes the persisted restart record."""
    message = _DeletableMessageStub(message_id=10, channel_id=20)
    await cleanup.track_game_message(message=cast("Message", message))

    await cleanup.delete_game_message_after(message=cast("Message", message), delay=0)

    assert message.delete_calls == 1
    assert await cleanup.list_pending_game_messages() == []


async def test_delete_tracked_game_messages_deletes_stale_restart_records() -> None:
    """Startup cleanup deletes persisted Discord messages and clears the records."""
    message = _DeletableMessageStub(message_id=10, channel_id=20)
    await cleanup.track_game_message(message=cast("Message", message))
    bot = _BotStub()

    await cleanup.delete_tracked_game_messages(bot=cast("commands.Bot", bot))

    assert bot.deleted == [(20, 10)]
    assert bot.fetch_calls == [20]
    assert await cleanup.list_pending_game_messages() == []


async def test_delete_tracked_game_messages_skips_non_messageable_cached_channel() -> None:
    """Cached PartialMessageable-like channels should be resolved via fetch_channel first."""
    message = _DeletableMessageStub(message_id=10, channel_id=20)
    await cleanup.track_game_message(message=cast("Message", message))
    bot = _BotStub(cached_channel=object())

    await cleanup.delete_tracked_game_messages(bot=cast("commands.Bot", bot))

    assert bot.deleted == [(20, 10)]
    assert bot.fetch_calls == [20]
    assert await cleanup.list_pending_game_messages() == []


async def test_delete_tracked_game_messages_keeps_unresolved_channel_records() -> None:
    """Startup cleanup keeps records when it cannot resolve a message-fetchable channel."""
    message = _DeletableMessageStub(message_id=10, channel_id=20)
    await cleanup.track_game_message(message=cast("Message", message))

    await cleanup.delete_tracked_game_messages(bot=cast("commands.Bot", _UnfetchableBotStub()))

    assert await cleanup.list_pending_game_messages() == [
        cleanup.PendingGameMessage(channel_id=20, message_id=10)
    ]


async def test_send_expiring_followup_waits_for_message_and_schedules_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Game-related economy embeds must retrieve their message before cleanup."""
    scheduled_messages: list[object] = []

    def fake_schedule_game_message_delete(message: object, delay: float = 180) -> None:
        """Records the message scheduled for later deletion."""
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


async def test_send_private_followup_is_ephemeral_and_not_scheduled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Personal economy embeds should not enter the game cleanup scheduler."""
    scheduled_messages: list[object] = []

    def fake_schedule_game_message_delete(message: object, delay: float = 180) -> None:
        """Records any unexpected scheduler calls."""
        scheduled_messages.append(message)

    monkeypatch.setattr(
        target=economy,
        name="schedule_game_message_delete",
        value=fake_schedule_game_message_delete,
    )
    interaction = _InteractionStub()
    embed = nextcord.Embed(title="balance")

    await economy._send_private_followup(interaction=cast("Interaction", interaction), embed=embed)

    assert interaction.followup.sent_ephemeral is True
    assert interaction.followup.sent_embed is embed
    assert scheduled_messages == []


async def test_delete_game_message_after_ignores_already_deleted_message() -> None:
    """Manual deletion before cleanup should not surface as a task failure."""
    await cleanup.delete_game_message_after(
        message=cast("Message", _AlreadyDeletedMessageStub()), delay=0
    )


async def test_schedule_game_message_delete_uses_default_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduling uses the shared three-minute TTL by default."""
    scheduled_delay: float | None = None

    async def fake_delete_game_message_after(message: object, delay: float) -> None:
        """Records the delay requested by the scheduler."""
        nonlocal scheduled_delay
        scheduled_delay = delay

    monkeypatch.setattr(
        target=cleanup, name="delete_game_message_after", value=fake_delete_game_message_after
    )

    cleanup.schedule_game_message_delete(message=cast("Message", _DeletableMessageStub()))
    await asyncio.sleep(delay=0)

    assert scheduled_delay == cleanup.GAME_RESPONSE_TTL_SECONDS
