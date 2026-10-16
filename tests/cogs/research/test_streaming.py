"""Tests for the research progress streamer: the reasoning preview and its editor."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn, cast
import asyncio

import pytest

from discordbot.cogs.research import streaming as research_streaming
from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT
from discordbot.cogs.research.streaming import ResearchProgressStreamer

from tests.helpers.casting import (
    make_forbidden,
    make_not_found,
    make_server_error,
    as_interaction_event_stream,
)
from tests.helpers.research import (
    _as_event,
    _RunStatus,
    _RunThread,
    _FakeStream,
    _created_event,
    _thought_event,
)
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from nextcord import AllowedMentions


def test_streamer_feed_accumulates_only_thought_summaries() -> None:
    streamer = ResearchProgressStreamer(status=None, label="Antigravity")
    streamer._feed(event=_as_event(_thought_event("searching...")))
    text_delta = SimpleNamespace(
        event_type="step.delta", event_id="x", delta=SimpleNamespace(type="text", text="body")
    )
    streamer._feed(
        event=_as_event(text_delta)
    )  # report text is delivered separately, not reasoning
    streamer._feed(event=_as_event(_created_event()))  # non-delta events are ignored
    assert streamer.reasoning == "searching..."


def test_streamer_render_preview_windows_and_escapes_mentions() -> None:
    streamer = ResearchProgressStreamer(status=None, label="Antigravity")
    streamer.reasoning = "first line\n@everyone please\nlast line"
    preview = streamer._render_preview()
    assert preview.startswith("-# Researching... (Antigravity,")
    assert "@everyone" not in preview  # agent text is escaped so the thinking can never ping
    assert "last line" in preview

    # Two windows narrow a long think and each drops what the other would have kept: the
    # renderer sees only the last 1500 characters, then keeps only the newest of THOSE lines
    # that fit one Discord message. The lines are short so the second window bites too.
    lines = [f"t{index:03d}" for index in range(500)]
    streamer.reasoning = "\n".join(["oldest thought", *lines])
    preview = streamer._render_preview()
    assert "oldest thought" not in preview  # outside the character tail
    assert lines[-1] in preview  # the newest thought always survives
    assert len(preview) <= DISCORD_MESSAGE_LIMIT
    # Fewer lines than the character tail holds: the per-line budget dropped the rest.
    assert 0 < preview.count("\n-# ") < len(streamer.reasoning[-1500:].splitlines())


async def test_streamer_write_snapshot_edits_and_skips_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The header's elapsed timer is frozen, so a second render of the same reasoning is the
    # same snapshot however long the first write took.
    monkeypatch.setattr(
        target=research_streaming, name="time", value=SimpleNamespace(monotonic=lambda: 100.0)
    )
    thread = _RunThread()
    streamer = ResearchProgressStreamer(
        status=_RunStatus(thread=thread),
        label="Antigravity",
        reasoning="thinking",
        started_at=100.0,
    )
    await streamer._write_preview_snapshot()
    assert len(thread.writes) == 1
    assert cast("AllowedMentions", thread.writes[0]["allowed_mentions"]).everyone is False
    # A second write of the same rendered snapshot is a no-op, so the editor never spams edits.
    await streamer._write_preview_snapshot()
    assert len(thread.writes) == 1
    streamer.reasoning += " more"
    await streamer._write_preview_snapshot()
    assert len(thread.writes) == 2


async def test_streamer_stream_accumulates_and_stops_editor_cleanly() -> None:
    streamer = ResearchProgressStreamer(
        status=_RunStatus(thread=_RunThread()), label="Antigravity", preview_interval_seconds=0.01
    )
    await streamer.stream(
        events=as_interaction_event_stream(
            fake=_FakeStream([_thought_event("aaa"), _thought_event("bbb")])
        )
    )
    assert streamer.reasoning == "aaabbb"
    assert streamer._editor_task is None  # the cadence editor is always stopped in finally


class _FailingStatusMessage:
    """A status message every edit of which fails with `error`, counting the attempts."""

    id = 7

    def __init__(self, error: Exception, attempts_seen: int = 1) -> None:
        """Initializes the failure and the attempt count that sets `seen`."""
        self.error = error
        self.attempts = 0
        self.attempts_seen = attempts_seen
        self.seen = asyncio.Event()

    async def edit(self, **_kwargs: object) -> NoReturn:
        """Fails the edit the way Discord does."""
        self.attempts += 1
        if self.attempts >= self.attempts_seen:
            self.seen.set()
        raise self.error


@pytest.mark.parametrize(
    ("failure", "level"),
    [(make_not_found(message="Unknown Message"), "info"), (make_forbidden(), "warn")],
    ids=["status_deleted", "shut_out"],
)
async def test_a_status_message_that_can_no_longer_be_edited_stops_the_preview(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, level: str
) -> None:
    """Neither failure clears on a retry, so the editor stops at once and says why, untraced."""
    status = _FailingStatusMessage(error=failure)
    streamer = ResearchProgressStreamer(
        status=status, label="Antigravity", reasoning="thinking", preview_interval_seconds=0.01
    )
    records = {
        name: capture_logs(monkeypatch=monkeypatch, level=name) for name in ("info", "warn")
    }

    await asyncio.wait_for(streamer._preview_editor(), timeout=5)

    assert status.attempts == 1
    assert [(name, fields) for name, found in records.items() for _, fields in found] == [
        (level, {"message_id": 7})
    ]


async def test_a_failing_preview_edit_is_logged_once_and_the_editor_keeps_going(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient failure may clear, so the editor retries, but the log names it only once."""
    status = _FailingStatusMessage(error=make_server_error(), attempts_seen=3)
    streamer = ResearchProgressStreamer(
        status=status, label="Antigravity", reasoning="thinking", preview_interval_seconds=0.01
    )
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")

    streamer._ensure_editor_started()
    await asyncio.wait_for(status.seen.wait(), timeout=5)
    await streamer._stop_editor()

    assert [(message, fields["error_type"]) for message, fields in warns] == [
        ("research preview edit failed; continuing the run", "HTTPException")
    ]
    assert warns[0][1]["_exc_info"] is not None
