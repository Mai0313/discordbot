"""Tests for public response cleanup helpers."""

import asyncio

import pytest
import nextcord
from nextcord import Message
from nextcord.abc import Messageable

from discordbot.utils import message_cleanup as cleanup_module
from discordbot.utils import interaction_responses as interactions
from discordbot.utils.message_cleanup import (
    PUBLIC_MESSAGE_TTL_SECONDS,
    PendingPublicMessage,
    track_public_message,
    delete_public_message_after,
    list_pending_public_messages,
    delete_tracked_public_messages,
    schedule_public_message_delete,
)

from tests.helpers.casting import (
    as_bot,
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
)


class _DeletableMessageStub:
    """Minimal message stub that records deletion."""

    def __init__(
        self,
        message_id: int = 123,
        channel_id: int = 456,
        guild_name: str | None = None,
        channel_name: str | None = None,
    ) -> None:
        """Initializes message and channel IDs for cleanup tracking."""
        self.id = message_id
        guild = _GuildStub(name=guild_name) if guild_name is not None else None
        self.guild = guild
        self.channel = _ChannelStub(channel_id=channel_id, name=channel_name, guild=guild)
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
        raise make_not_found()


class _ChannelStub:
    """Minimal channel shape for persistent cleanup identity."""

    def __init__(
        self, channel_id: int, name: str | None = None, guild: "_GuildStub | None" = None
    ) -> None:
        """Stores the Discord channel identity."""
        self.id = channel_id
        self.name = name
        self.guild = guild


class _GuildStub:
    """Minimal guild shape for persistent cleanup readability."""

    def __init__(self, name: str) -> None:
        """Stores the Discord guild name."""
        self.name = name


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


class _FetchMessageChannelStub(Messageable):
    """Minimal message channel returned by the fake bot.

    Subclasses ``Messageable`` because production code narrows channels with
    ``isinstance(channel, Messageable)`` before fetching.
    """

    def __init__(self, channel_id: int, deleted: list[tuple[int, int]]) -> None:
        """Stores channel identity and the shared deletion recorder."""
        self.channel_id = channel_id
        self.deleted = deleted

    async def fetch_message(self, message_id: int, /) -> Message:
        """Returns a fetched message stub typed as the Message the base declares."""
        return as_message(
            fake=_FetchedMessageStub(
                channel_id=self.channel_id, message_id=message_id, deleted=self.deleted
            )
        )


class _NonMessageableChannelStub:
    """Channel shape without `fetch_message` used to exercise fallback paths."""

    pass


class _BotStub:
    """Minimal bot shape for startup cleanup."""

    def __init__(
        self, cached_channel: _FetchMessageChannelStub | _NonMessageableChannelStub | None = None
    ) -> None:
        """Initializes cached-channel behavior and cleanup call records."""
        self.deleted: list[tuple[int, int]] = []
        self.cached_channel = cached_channel
        self.fetch_calls: list[int] = []

    def get_channel(
        self, channel_id: int, /
    ) -> _FetchMessageChannelStub | _NonMessageableChannelStub | None:
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

    async def fetch_channel(self, channel_id: int, /) -> _NonMessageableChannelStub:
        """Returns a non-messageable channel shape."""
        return _NonMessageableChannelStub()


class _SentFollowupMessageStub:
    """Message shape returned by fake followup sends."""

    pass


class _FollowupStub:
    """Minimal followup stub that records send arguments."""

    def __init__(self) -> None:
        """Initializes followup send records."""
        self.message = _SentFollowupMessageStub()
        self.sent_wait: bool | None = None
        self.sent_ephemeral: bool | None = None
        self.sent_embed: nextcord.Embed | None = None
        self.sent_view: nextcord.ui.View | None = None
        self.sent_files: list[nextcord.File] | None = None

    async def send(
        self,
        embed: nextcord.Embed,
        wait: bool = False,
        ephemeral: bool = False,
        view: nextcord.ui.View | None = None,
        files: list[nextcord.File] | None = None,
    ) -> _SentFollowupMessageStub:
        """Records the embed send and returns the message object."""
        self.sent_embed = embed
        self.sent_wait = wait
        self.sent_ephemeral = ephemeral
        self.sent_view = view
        self.sent_files = files
        return self.message


class _InteractionStub:
    """Minimal interaction shape for expiring economy followups."""

    def __init__(self) -> None:
        """Initializes the followup stub used by the helper under test."""
        self.user = _UserStub(name="alice")
        self.followup = _FollowupStub()


class _UserStub:
    """Minimal interaction user shape for cleanup readability."""

    def __init__(self, name: str) -> None:
        """Stores the Discord account name."""
        self.name = name


async def test_delete_public_message_after_waits_then_deletes() -> None:
    """Public response cleanup deletes the message after the configured delay."""
    message = _DeletableMessageStub()

    await delete_public_message_after(message=as_message(fake=message), delay=0)

    assert message.delete_calls == 1


async def test_track_public_message_persists_message_identity() -> None:
    """Public response tracking stores IDs plus readable guild/channel names."""
    message = _DeletableMessageStub(
        message_id=10, channel_id=20, guild_name="Mai Server", channel_name="casino"
    )
    expected = PendingPublicMessage(
        channel_id=20,
        message_id=10,
        guild_name="Mai Server",
        channel_name="casino",
        user_name="alice",
    )

    record = await track_public_message(message=as_message(fake=message), user_name="alice")

    assert record == expected
    assert await list_pending_public_messages() == [expected]
    await track_public_message(message=as_message(fake=message))
    assert await list_pending_public_messages() == [expected]


async def test_delete_public_message_after_forgets_successful_cleanup() -> None:
    """Successful TTL cleanup removes the persisted restart record."""
    message = _DeletableMessageStub(message_id=10, channel_id=20)
    await track_public_message(message=as_message(fake=message))

    await delete_public_message_after(message=as_message(fake=message), delay=0)

    assert message.delete_calls == 1
    assert await list_pending_public_messages() == []


async def test_delete_tracked_public_messages_deletes_stale_restart_records() -> None:
    """Startup cleanup deletes persisted Discord messages and clears the records."""
    message = _DeletableMessageStub(message_id=10, channel_id=20)
    await track_public_message(message=as_message(fake=message))
    bot = _BotStub()

    await delete_tracked_public_messages(bot=as_bot(fake=bot))

    assert bot.deleted == [(20, 10)]
    assert bot.fetch_calls == [20]
    assert await list_pending_public_messages() == []


async def test_delete_tracked_public_messages_skips_non_messageable_cached_channel() -> None:
    """A cached channel that is not Messageable is re-resolved through fetch_channel."""
    message = _DeletableMessageStub(message_id=10, channel_id=20)
    await track_public_message(message=as_message(fake=message))
    bot = _BotStub(cached_channel=_NonMessageableChannelStub())

    await delete_tracked_public_messages(bot=as_bot(fake=bot))

    assert bot.deleted == [(20, 10)]
    assert bot.fetch_calls == [20]
    assert await list_pending_public_messages() == []


async def test_delete_tracked_public_messages_keeps_unresolved_channel_records() -> None:
    """Startup cleanup keeps records when it cannot resolve a message-fetchable channel."""
    message = _DeletableMessageStub(message_id=10, channel_id=20)
    await track_public_message(message=as_message(fake=message))

    await delete_tracked_public_messages(bot=as_bot(fake=_UnfetchableBotStub()))

    assert await list_pending_public_messages() == [
        PendingPublicMessage(channel_id=20, message_id=10)
    ]


class _ForbiddenChannelBotStub:
    """A bot whose identity is shut out of the channel holding the record."""

    def __init__(self) -> None:
        """Initializes the cleanup call record."""
        self.fetch_calls: list[int] = []

    def get_channel(self, channel_id: int, /) -> None:
        """Nothing cached, so the sweep reaches for `fetch_channel`."""
        return

    async def fetch_channel(self, channel_id: int, /) -> object:
        """Refuses the way Discord refuses a channel the bot may not view."""
        self.fetch_calls.append(channel_id)
        raise make_forbidden(message="Missing Access")


def _recorded_warns(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, object]]]:
    """Captures what the sweep reports, the way the rest of the suite reads logfire."""
    warns: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        target=cleanup_module.logfire,
        name="warn",
        value=lambda message, **fields: warns.append((message, fields)),
    )
    return warns


async def test_a_channel_the_bot_cannot_read_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permission the bot never had is expected, so it says so and attaches nothing.

    The channel's overwrites belong to whoever administers that server, and an interaction
    reaches a channel the bot's own identity cannot, so this is a standing condition rather
    than something to diagnose. Sixteen lines of identical stack per record per boot is what
    the carve-out in `.github/CONTRIBUTING.md#logging` exists to stop.
    """
    message = _DeletableMessageStub(message_id=10, channel_id=20)
    await track_public_message(message=as_message(fake=message))
    bot = _ForbiddenChannelBotStub()
    warns = _recorded_warns(monkeypatch=monkeypatch)

    await delete_tracked_public_messages(bot=as_bot(fake=bot))

    assert bot.fetch_calls == [20]
    assert len(warns) == 1, f"expected one report, got {warns}"
    text, fields = warns[0]
    assert "cannot read" in text
    assert fields == {"channel_id": 20, "message_id": 10}, (
        "a permission the bot cannot earn needs the ids and no traceback"
    )


async def test_a_transient_http_failure_keeps_its_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The carve-out is for `Forbidden` alone; a 5xx is still something to look at.

    Both used to share one `except HTTPException`, so dropping the traceback there would have
    taken it from every transport failure too.
    """

    class _ServerErrorBotStub(_ForbiddenChannelBotStub):
        async def fetch_channel(self, channel_id: int, /) -> object:
            """Fails the way Discord fails when it is Discord that is broken."""
            self.fetch_calls.append(channel_id)
            raise make_server_error()

    message = _DeletableMessageStub(message_id=11, channel_id=21)
    await track_public_message(message=as_message(fake=message))
    warns = _recorded_warns(monkeypatch=monkeypatch)

    await delete_tracked_public_messages(bot=as_bot(fake=_ServerErrorBotStub()))

    assert len(warns) == 1
    text, fields = warns[0]
    assert "Failed to fetch" in text
    assert fields.get("_exc_info") is True, "a transport failure still needs its traceback"


async def test_send_expiring_followup_waits_for_message_and_schedules_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public economy embeds must retrieve their message before cleanup."""
    scheduled_messages: list[_SentFollowupMessageStub] = []
    scheduled_user_names: list[str | None] = []

    def fake_schedule_public_message_delete(
        message: _SentFollowupMessageStub, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records the message scheduled for later deletion."""
        scheduled_messages.append(message)
        scheduled_user_names.append(user_name)

    monkeypatch.setattr(
        target=interactions,
        name="schedule_public_message_delete",
        value=fake_schedule_public_message_delete,
    )
    interaction = _InteractionStub()
    embed = nextcord.Embed(title="balance")

    await interactions.send_expiring_followup(
        interaction=as_interaction(fake=interaction), embed=embed
    )

    assert interaction.followup.sent_wait is True
    assert interaction.followup.sent_embed is embed
    assert interaction.followup.sent_view is None
    assert scheduled_messages == [interaction.followup.message]
    assert scheduled_user_names == ["alice"]


async def test_send_private_followup_is_ephemeral_and_not_scheduled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Personal economy embeds should not enter the public cleanup scheduler."""
    scheduled_messages: list[_SentFollowupMessageStub] = []

    def fake_schedule_public_message_delete(
        message: _SentFollowupMessageStub, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records any unexpected scheduler calls."""
        scheduled_messages.append(message)

    monkeypatch.setattr(
        target=interactions,
        name="schedule_public_message_delete",
        value=fake_schedule_public_message_delete,
    )
    interaction = _InteractionStub()
    embed = nextcord.Embed(title="balance")

    await interactions.send_private_followup(
        interaction=as_interaction(fake=interaction), embed=embed
    )

    assert interaction.followup.sent_ephemeral is True
    assert interaction.followup.sent_embed is embed
    assert scheduled_messages == []


async def test_delete_public_message_after_ignores_already_deleted_message() -> None:
    """Manual deletion before cleanup should not surface as a task failure."""
    await delete_public_message_after(
        message=as_message(fake=_AlreadyDeletedMessageStub()), delay=0
    )


async def test_schedule_public_message_delete_uses_default_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduling uses the shared three-minute TTL by default."""
    scheduled_delay: float | None = None

    async def fake_delete_public_message_after(
        message: Message, delay: float, user_name: str | None = None
    ) -> None:
        """Records the delay requested by the scheduler."""
        nonlocal scheduled_delay
        scheduled_delay = delay

    monkeypatch.setattr(
        "discordbot.utils.message_cleanup.delete_public_message_after",
        fake_delete_public_message_after,
    )

    schedule_public_message_delete(message=as_message(fake=_DeletableMessageStub()))
    await asyncio.sleep(delay=0)

    assert scheduled_delay == PUBLIC_MESSAGE_TTL_SECONDS
