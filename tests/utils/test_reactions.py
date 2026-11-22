"""Tests for the shared status-reaction helper."""

from types import SimpleNamespace
import asyncio

import pytest
from nextcord import HTTPException

from discordbot.utils.reactions import ReactionStatusChain, update_reaction

from tests.helpers.casting import as_message, make_forbidden, make_not_found, make_server_error
from tests.helpers.reply_fakes import FakeMessage
from tests.helpers.discord_mocks import FakeDiscordMessage
from tests.helpers.logfire_capture import capture_logs


async def test_a_refused_reaction_is_a_warn_with_its_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """A channel that refuses the reaction is logged with the ids, never swallowed silently."""
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    message = FakeDiscordMessage()

    async def refuse(emoji: str) -> None:
        """Refuses the reaction the way a channel overwrite does."""
        del emoji
        raise make_forbidden(message="Missing Permissions")

    monkeypatch.setattr(target=message, name="add_reaction", value=refuse)

    added = await update_reaction(message=as_message(fake=message), bot_user=None, emoji="✅")

    assert added == "✅"
    assert [fields for _message, fields in warns] == [
        {
            "message_id": message.id,
            "channel_id": message.channel.id,
            "emoji": "✅",
            "code": 0,
            "error_type": "Forbidden",
        }
    ]


async def test_update_reaction_suppresses_not_found_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding a reaction to a deleted message is inert cleanup and produces no warning."""
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    message = FakeDiscordMessage()

    async def gone(emoji: str) -> None:
        del emoji
        raise make_not_found(message="Unknown Message")

    monkeypatch.setattr(target=message, name="add_reaction", value=gone)

    added = await update_reaction(message=as_message(fake=message), bot_user=None, emoji="✅")

    assert added == "✅"
    assert warns == []


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [(make_server_error(), "HTTPException"), (RuntimeError("network reset"), "RuntimeError")],
    ids=["http_500", "unexpected_error"],
)
async def test_update_reaction_logs_unexpected_errors_at_warn(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, error_type: str
) -> None:
    """Unexpected reaction failures log at `warn` with `error_type` and `_exc_info` without raising."""
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    message = FakeDiscordMessage()

    async def broken(emoji: str) -> None:
        del emoji
        raise failure

    monkeypatch.setattr(target=message, name="add_reaction", value=broken)

    added = await update_reaction(message=as_message(fake=message), bot_user=None, emoji="✅")

    assert added == "✅"
    expected: dict[str, object] = {
        "message_id": message.id,
        "channel_id": message.channel.id,
        "emoji": "✅",
        "error_type": error_type,
        "_exc_info": failure,
    }
    if isinstance(failure, HTTPException):
        expected["code"] = failure.code
    assert [fields for _message, fields in warns] == [expected]


async def test_reaction_status_chain_orders_and_replaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advance schedules ordered swaps without blocking; flush waits for the tail."""
    events: list[tuple[str, str | None]] = []

    async def fake_reaction(
        message: FakeMessage, bot_user: object, emoji: str, previous: str | None = None
    ) -> str:
        """Records each scheduled reaction swap, yielding on the first so ordering is tested."""
        del message, bot_user
        if emoji == "🔀":
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        events.append((emoji, previous))
        return emoji

    monkeypatch.setattr("discordbot.utils.reactions.update_reaction", fake_reaction)
    chain = ReactionStatusChain(
        message=FakeMessage(content="hi"), bot_user=SimpleNamespace(id=999)
    )
    chain.advance(emoji="🔀")
    chain.advance(emoji="❓")
    chain.advance(emoji="🆗")
    assert events == []  # nothing awaited yet: scheduling never blocks the caller
    await chain.flush()
    # order-contract: ReactionStatusChain promises FIFO reaction swaps.
    assert events == [("🔀", None), ("❓", "🔀"), ("🆗", "❓")]
