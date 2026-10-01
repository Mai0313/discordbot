"""Tests for public response cleanup helpers."""

import asyncio
from collections import Counter

import pytest
from nextcord import Message, Interaction
from nextcord.abc import Messageable
from nextcord.ext import commands

from discordbot.utils import message_cleanup as cleanup_module
from discordbot.utils.message_cleanup import (
    PUBLIC_MESSAGE_TTL_SECONDS,
    PendingPublicMessage,
    track_public_message,
    delete_public_message,
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
    make_invalid_webhook_token,
)
from tests.helpers.discord_mocks import FakeGuild, FakeInteraction, FakeDiscordMessage


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


async def test_delete_public_message_after_waits_then_deletes() -> None:
    """Public response cleanup deletes the message after the configured delay."""
    message = FakeDiscordMessage()

    await delete_public_message_after(message=as_message(fake=message), delay=0)

    assert message.deleted is True


async def test_track_public_message_persists_message_identity() -> None:
    """Public response tracking stores IDs plus readable guild/channel names."""
    message = FakeDiscordMessage(guild=FakeGuild(guild_name="Mai Server"), channel_name="casino")
    expected = PendingPublicMessage(
        channel_id=message.channel.id,
        message_id=message.id,
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
    message = FakeDiscordMessage()
    await track_public_message(message=as_message(fake=message))

    await delete_public_message_after(message=as_message(fake=message), delay=0)

    assert message.deleted is True
    assert await list_pending_public_messages() == []


async def test_delete_tracked_public_messages_deletes_stale_restart_records() -> None:
    """Startup cleanup deletes persisted Discord messages and clears the records."""
    message = FakeDiscordMessage()
    await track_public_message(message=as_message(fake=message))
    bot = _BotStub()

    await delete_tracked_public_messages(bot=as_bot(fake=bot))

    assert bot.deleted == [(message.channel.id, message.id)]
    assert bot.fetch_calls == [message.channel.id]
    assert await list_pending_public_messages() == []


async def test_delete_tracked_public_messages_skips_non_messageable_cached_channel() -> None:
    """A cached channel that is not Messageable is re-resolved through fetch_channel."""
    message = FakeDiscordMessage()
    await track_public_message(message=as_message(fake=message))
    bot = _BotStub(cached_channel=_NonMessageableChannelStub())

    await delete_tracked_public_messages(bot=as_bot(fake=bot))

    assert bot.deleted == [(message.channel.id, message.id)]
    assert bot.fetch_calls == [message.channel.id]
    assert await list_pending_public_messages() == []


async def test_delete_tracked_public_messages_keeps_unresolved_channel_records() -> None:
    """Startup cleanup keeps records when it cannot resolve a message-fetchable channel."""
    message = FakeDiscordMessage()
    await track_public_message(message=as_message(fake=message))

    await delete_tracked_public_messages(bot=as_bot(fake=_UnfetchableBotStub()))

    assert await list_pending_public_messages() == [
        PendingPublicMessage(channel_id=message.channel.id, message_id=message.id)
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
    message = FakeDiscordMessage()
    await track_public_message(message=as_message(fake=message))
    bot = _ForbiddenChannelBotStub()
    warns = _recorded_warns(monkeypatch=monkeypatch)

    await delete_tracked_public_messages(bot=as_bot(fake=bot))

    assert bot.fetch_calls == [message.channel.id]
    assert len(warns) == 1, f"expected one report, got {warns}"
    text, fields = warns[0]
    assert "cannot read" in text
    assert fields == {"channel_id": message.channel.id, "message_id": message.id}, (
        "a permission the bot cannot earn needs the ids and no traceback"
    )


async def test_a_transient_http_failure_keeps_its_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The carve-out is for `Forbidden` alone; a 5xx is still something to look at.

    One `except HTTPException` for both would drop the traceback from every transport failure.
    """

    class _ServerErrorBotStub(_ForbiddenChannelBotStub):
        async def fetch_channel(self, channel_id: int, /) -> object:
            """Fails the way Discord fails when it is Discord that is broken."""
            self.fetch_calls.append(channel_id)
            raise make_server_error()

    await track_public_message(message=as_message(fake=FakeDiscordMessage()))
    warns = _recorded_warns(monkeypatch=monkeypatch)

    await delete_tracked_public_messages(bot=as_bot(fake=_ServerErrorBotStub()))

    assert len(warns) == 1
    text, fields = warns[0]
    assert "Failed to fetch" in text
    assert fields.get("_exc_info") is True, "a transport failure still needs its traceback"


async def test_delete_public_message_after_ignores_already_deleted_message() -> None:
    """A message someone deleted first is cleaned up all the same, restart record included."""
    message = FakeDiscordMessage()
    message.delete_failure = make_not_found()

    await delete_public_message_after(message=as_message(fake=message), delay=0)

    assert await list_pending_public_messages() == []


async def test_a_refused_delete_keeps_the_record_for_the_next_sweep() -> None:
    """A delete Discord refuses reports failure and leaves the record for the restart sweep."""
    message = FakeDiscordMessage()
    message.delete_failure = make_forbidden(message="Missing Permissions")
    await track_public_message(message=as_message(fake=message))

    assert await delete_public_message(message=as_message(fake=message)) is False
    assert await list_pending_public_messages() == [
        PendingPublicMessage(channel_id=message.channel.id, message_id=message.id)
    ]


@pytest.mark.parametrize(
    ("failure", "text", "traceback"),
    [
        (make_forbidden(message="Missing Access"), "refused", False),
        (make_server_error(), "Failed to delete", True),
    ],
    ids=["refused", "broke"],
)
async def test_a_refused_delete_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, text: str, traceback: bool
) -> None:
    """The delete gets the same carve-out as the sweep's fetch; a 5xx keeps its traceback."""
    message = FakeDiscordMessage()
    message.delete_failure = failure
    warns = _recorded_warns(monkeypatch=monkeypatch)

    assert await delete_public_message(message=as_message(fake=message)) is False
    assert [
        (text in logged, fields["message_id"], fields["channel_id"], "_exc_info" in fields)
        for logged, fields in warns
    ] == [(True, message.id, message.channel.id, traceback)]


@pytest.mark.parametrize(
    argnames=("expired", "failure", "through_token", "report"),
    argvalues=[
        (False, None, True, []),
        (True, None, False, []),
        (False, make_not_found(), False, [("info", False)]),
        (False, make_invalid_webhook_token(), False, [("info", False)]),
        (False, make_forbidden(message="Missing Access"), False, [("warn", False)]),
        (False, make_server_error(), False, [("warn", True)]),
    ],
    ids=["live", "expired", "token_404", "token_401", "token_refused", "token_broke"],
)
async def test_a_press_on_the_message_deletes_it_while_its_token_lives(
    monkeypatch: pytest.MonkeyPatch,
    expired: bool,
    failure: Exception | None,
    through_token: bool,
    report: list[tuple[str, bool]],
) -> None:
    """A press's token reaches a message the channel no longer lets the bot touch.

    A token that cannot do it hands the delete to the channel and leaves a record: a 404 is
    routine, since the channel tells a message already gone apart, and only a failure that is
    neither keeps its traceback.
    """
    reports: list[tuple[str, dict[str, object]]] = []
    for level in ("info", "warn"):
        monkeypatch.setattr(
            target=cleanup_module.logfire,
            name=level,
            value=lambda message, level=level, **fields: reports.append((level, fields)),
        )
    message = FakeDiscordMessage()
    press = FakeInteraction()
    press.expired = expired
    press.delete_failure = failure

    await delete_public_message_after(
        message=as_message(fake=message), delay=0, interaction=as_interaction(fake=press)
    )

    assert (press.original_deleted, message.deleted) == (through_token, not through_token)
    assert Counter((level, "_exc_info" in fields) for level, fields in reports) == Counter(report)
    assert await list_pending_public_messages() == []


async def test_schedule_public_message_delete_uses_default_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduling uses the shared three-minute TTL by default."""
    scheduled_delay: float | None = None

    async def fake_delete_public_message_after(
        message: Message,
        delay: float,
        user_name: str | None = None,
        interaction: Interaction[commands.Bot] | None = None,
    ) -> None:
        """Records the delay requested by the scheduler."""
        nonlocal scheduled_delay
        scheduled_delay = delay

    monkeypatch.setattr(
        "discordbot.utils.message_cleanup.delete_public_message_after",
        fake_delete_public_message_after,
    )

    schedule_public_message_delete(message=as_message(fake=FakeDiscordMessage()))
    await asyncio.sleep(delay=0)

    assert scheduled_delay == PUBLIC_MESSAGE_TTL_SECONDS
