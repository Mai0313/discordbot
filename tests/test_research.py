"""Tests for the deep-research feature: marker extraction, delivery, agent helpers, and store."""

from types import SimpleNamespace
import base64
from typing import TYPE_CHECKING, NoReturn, cast
import asyncio
from pathlib import Path
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest
from nextcord import File, Embed, Thread, Permissions, TextChannel, AllowedMentions
from sqlalchemy.exc import OperationalError

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.research import cog as research_cog
from discordbot.cogs.research import agent
from discordbot.cogs.research import database as rdb
from discordbot.cogs.research import streaming as research_streaming
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.asyncio_locks import KeyedLockManager
from discordbot.utils.model_pricing import ModelPriceEntry
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT
from discordbot.utils.media_delivery import MediaHostingService, MediaDeliveryPlanner
from discordbot.cogs.gen_reply.markers import extract_inline_markers, scrub_markers_for_preview
from discordbot.cogs.research.delivery import (
    split_report,
    deliver_report,
    owner_allowed_mentions,
    split_report_by_sections,
)
from discordbot.cogs.research.streaming import ResearchProgressStreamer

from tests.helpers.casting import (
    as_bot,
    as_client,
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
    make_invalid_form_body,
    make_media_hosting_config,
    as_interaction_event_stream,
)
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from google.genai.interactions import InteractionSSEEvent

    from discordbot.cogs.research.database import ResearchPhase
    from discordbot.cogs.gen_reply.attachment.base import AttachmentRenderer


def _disabled_delivery() -> MediaDeliveryPlanner:
    """A planner whose host is off, so report files attach natively."""
    return MediaDeliveryPlanner(
        media_hosting=MediaHostingService(config=make_media_hosting_config(enabled=False))
    )


# ----- marker extraction --------------------------------------------------------------------


def test_deep_research_block_is_pulled_and_brief_captured() -> None:
    markers = extract_inline_markers(
        text="好喔幫你查 <deep-research>研究 TPU 的競爭格局</deep-research> 等等貼到 thread"
    )
    assert markers.research_brief == "研究 TPU 的競爭格局"
    assert "TPU" not in markers.cleaned_text
    assert "thread" in markers.cleaned_text


def test_unclosed_trailing_deep_research_is_still_pulled() -> None:
    markers = extract_inline_markers(text="開查囉 <deep-research>研究量子計算最新進展")
    assert markers.research_brief == "研究量子計算最新進展"
    assert "量子" not in markers.cleaned_text


def test_deep_research_coexists_with_voice() -> None:
    markers = extract_inline_markers(
        text="<generate-voice>馬上幫你查</generate-voice> <deep-research>研究 X</deep-research>"
    )
    assert markers.voice_requested
    assert "馬上幫你查" in markers.cleaned_text
    assert markers.research_brief == "研究 X"
    assert "X" not in markers.cleaned_text


def test_scrub_hides_deep_research_mid_stream() -> None:
    assert "TPU" not in scrub_markers_for_preview(text="好喔 <deep-research>研究 TPU")


def test_no_marker_leaves_text_and_brief_untouched() -> None:
    markers = extract_inline_markers(text="這只是一般回覆,沒有任何 marker")
    assert markers.research_brief is None
    assert markers.cleaned_text == "這只是一般回覆,沒有任何 marker"


# ----- delivery splitting -------------------------------------------------------------------


def test_split_report_keeps_short_text_as_one_chunk() -> None:
    assert split_report(text="short report") == ["short report"]


def test_split_report_prefers_paragraph_boundaries() -> None:
    para_a = "A" * 1200
    para_b = "B" * 1200
    chunks = split_report(text=f"{para_a}\n\n{para_b}")
    assert len(chunks) == 2
    assert chunks[0] == para_a
    assert chunks[1] == para_b


def test_split_report_hard_cuts_an_oversized_line() -> None:
    chunks = split_report(text="C" * 5000, limit=2000)
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert "".join(chunks) == "C" * 5000


def test_split_report_by_sections_splits_on_thematic_breaks() -> None:
    chunks = split_report_by_sections(text="## A\n\nAlpha body\n\n---\n\n## B\n\nBeta body")
    assert chunks == ["## A\n\nAlpha body", "## B\n\nBeta body"]


def test_split_report_by_sections_subsplits_oversized_section() -> None:
    chunks = split_report_by_sections(text="intro\n\n---\n\n" + "X" * 2500, limit=2000)
    assert chunks[0] == "intro"
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert "".join(chunks[1:]) == "X" * 2500
    assert len(chunks) == 3


def test_split_report_by_sections_falls_back_to_paragraph_packing() -> None:
    text = f"{'A' * 1200}\n\n{'B' * 1200}"
    assert split_report_by_sections(text=text) == split_report(text=text)


def test_split_report_by_sections_ignores_break_inside_code_fence() -> None:
    chunks = split_report_by_sections(text="before\n\n```\n---\n```\n\nafter")
    assert len(chunks) == 1
    assert "---" in chunks[0]


def test_split_report_by_sections_keeps_table_delimiter_row() -> None:
    chunks = split_report_by_sections(text="| Col | Val |\n| --- | --- |\n| a | 1 |")
    assert len(chunks) == 1


def test_split_report_by_sections_keeps_setext_heading() -> None:
    chunks = split_report_by_sections(text="Heading\n---\n\nbody")
    assert len(chunks) == 1


def test_split_report_by_sections_drops_empty_sections() -> None:
    chunks = split_report_by_sections(text="---\n\nonly body\n\n---")
    assert chunks == ["only body"]


# ----- agent helpers ------------------------------------------------------------------------


class _FakeStream:
    """Async iterator over scripted SSE events; can raise after a prefix to simulate a drop."""

    def __init__(self, events: list[object], *, raise_after: int | None = None) -> None:
        self._events = list(events)
        self._raise_after = raise_after
        self._yielded = 0

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> object:
        if self._raise_after is not None and self._yielded >= self._raise_after:
            raise RuntimeError("stream dropped")
        if not self._events:
            raise StopAsyncIteration
        self._yielded += 1
        return self._events.pop(0)


class _FakeInteractions:
    """Fakes `client.aio.interactions`: `create`/`get(stream=True)` yield scripted streams.

    A non-stream `get(id=...)` returns the terminal interaction (the authoritative final read).
    """

    def __init__(self, *, streams: list[_FakeStream], terminal: object) -> None:
        self._streams = list(streams)
        self._terminal = terminal
        self.create_kwargs: dict[str, object] = {}
        self.stream_get_calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> _FakeStream:
        self.create_kwargs = kwargs
        return self._streams.pop(0)

    async def get(self, **kwargs: object) -> object:
        if kwargs.get("stream"):
            self.stream_get_calls.append(kwargs)
            return self._streams.pop(0)
        return self._terminal


def _fake_client(*, streams: list[_FakeStream], terminal: object) -> SimpleNamespace:
    return SimpleNamespace(
        aio=SimpleNamespace(interactions=_FakeInteractions(streams=streams, terminal=terminal))
    )


def _as_event(fake: object) -> "InteractionSSEEvent":
    """Views a fabricated SSE event double as the real SDK union a production signature expects.

    Production discriminates on `.event_type`, not isinstance, so a SimpleNamespace event is safe.
    """
    return cast("InteractionSSEEvent", fake)


def _created_event(*, interaction_id: str = "int_9", event_id: str = "e1") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="interaction.created",
        event_id=event_id,
        interaction=SimpleNamespace(id=interaction_id, model="m"),
    )


def _thought_event(text: str, *, event_id: str = "e2") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="step.delta",
        event_id=event_id,
        delta=SimpleNamespace(type="thought_summary", content=SimpleNamespace(text=text)),
    )


def _completed_event(*, event_id: str = "e9") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="interaction.completed", event_id=event_id, interaction=SimpleNamespace()
    )


def _terminal_interaction(
    *, status: str = "completed", input_tokens: int = 10, output_tokens: int = 5
) -> SimpleNamespace:
    return SimpleNamespace(
        id="int_9",
        status=status,
        output_text="# Report\nbody",
        usage=SimpleNamespace(total_input_tokens=input_tokens, total_output_tokens=output_tokens),
        steps=[],
    )


async def test_stream_antigravity_persists_id_streams_and_returns_terminal_result() -> None:
    client = _fake_client(
        streams=[_FakeStream([_created_event(), _thought_event("searching"), _completed_event()])],
        terminal=_terminal_interaction(),
    )
    streamer = ResearchProgressStreamer(
        status=None, label="Antigravity", preview_interval_seconds=0.01
    )
    persisted: list[str] = []

    async def _persist(interaction_id: str) -> None:
        persisted.append(interaction_id)

    result = await agent.stream_antigravity(
        client=as_client(fake=client),
        agent="antigravity-preview-09-2026",
        brief="b",
        system_instruction="sys",
        streamer=streamer,
        on_created=_persist,
    )
    kwargs = client.aio.interactions.create_kwargs
    # The id is persisted on the first event (before the long wait) and the built-in grounding
    # tool set rides every streaming create; the final result comes from the terminal get.
    assert persisted == ["int_9"]
    assert kwargs["stream"] is True
    assert kwargs["background"] is True
    assert kwargs["tools"] is agent.RESEARCH_TOOLS
    # The config block names the agent's own family and carries no knob of its own; a `deep-research`
    # one here would be the removed tiers' `collaborative_planning` coming back, and nothing upstream
    # rejects a mismatched family, so the discriminator is pinned here instead.
    assert kwargs["agent_config"] is agent.RESEARCH_AGENT_CONFIG
    assert kwargs["agent_config"]["type"] == "antigravity"
    assert streamer.reasoning == "searching"
    assert result.ok is True
    assert result.report_text.startswith("# Report")
    assert result.input_tokens == 10
    assert result.output_tokens == 5


async def test_stream_reconnects_when_stream_ends_without_terminal(monkeypatch) -> None:  # noqa: ANN001 -- pytest monkeypatch fixture
    # The SDK can close a bounded request mid-run; ending WITHOUT a terminal event must re-attach
    # (from the last event id), not be mistaken for completion.
    monkeypatch.setattr(agent, "RESEARCH_POLL_INTERVAL_SECONDS", 0.0)
    client = _fake_client(
        streams=[
            _FakeStream([_created_event(event_id="e1"), _thought_event("part1", event_id="e2")]),
            _FakeStream([_completed_event(event_id="e3")]),
        ],
        terminal=_terminal_interaction(),
    )
    streamer = ResearchProgressStreamer(status=None, label="Antigravity")

    async def _persist(_interaction_id: str) -> None:
        return None

    result = await agent.stream_antigravity(
        client=as_client(fake=client),
        agent="a",
        brief="b",
        system_instruction="s",
        streamer=streamer,
        on_created=_persist,
    )
    stream_gets = client.aio.interactions.stream_get_calls
    assert stream_gets
    assert stream_gets[0]["last_event_id"] == "e2"
    assert result.ok is True


async def test_stream_reconnects_after_a_mid_stream_drop(monkeypatch) -> None:  # noqa: ANN001 -- pytest monkeypatch fixture
    monkeypatch.setattr(agent, "RESEARCH_POLL_INTERVAL_SECONDS", 0.0)
    client = _fake_client(
        streams=[
            _FakeStream(
                [_created_event(event_id="e1"), _thought_event("x", event_id="e2")], raise_after=2
            ),
            _FakeStream([_completed_event(event_id="e3")]),
        ],
        terminal=_terminal_interaction(),
    )
    streamer = ResearchProgressStreamer(status=None, label="Antigravity")

    async def _persist(_interaction_id: str) -> None:
        return None

    result = await agent.stream_antigravity(
        client=as_client(fake=client),
        agent="a",
        brief="b",
        system_instruction="s",
        streamer=streamer,
        on_created=_persist,
    )
    assert client.aio.interactions.stream_get_calls[0]["last_event_id"] == "e2"
    assert result.ok is True


async def test_stream_falls_back_to_poll_when_streaming_gives_up(monkeypatch) -> None:  # noqa: ANN001 -- pytest monkeypatch fixture
    monkeypatch.setattr(agent, "RESEARCH_POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(agent, "MAX_STREAM_RECONNECTS", 2)
    # More dead streams than the bound allows, so running out of them cannot pass for giving up.
    client = _fake_client(
        streams=[
            _FakeStream([_created_event(event_id="e1")], raise_after=1),
            *(_FakeStream([], raise_after=0) for _ in range(10)),
        ],
        terminal=_terminal_interaction(),
    )
    streamer = ResearchProgressStreamer(status=None, label="Antigravity")
    warns = _recorded(monkeypatch=monkeypatch, level="warn")

    async def _persist(_interaction_id: str) -> None:
        return None

    # Streaming exhausts its reconnects, so the driver degrades to the poll and still returns the
    # authoritative terminal result.
    result = await agent.stream_antigravity(
        client=as_client(fake=client),
        agent="a",
        brief="b",
        system_instruction="s",
        streamer=streamer,
        on_created=_persist,
    )
    # The bound's worth of re-attaches that made no progress, plus the one that gave up.
    assert len(client.aio.interactions.stream_get_calls) == agent.MAX_STREAM_RECONNECTS + 1
    assert result.ok is True
    assert [
        (fields["interaction_id"], fields["error_type"])
        for message, fields in warns
        if message == "research stream failed; polling for the terminal result"
    ] == [("int_9", "RuntimeError")]


async def test_stream_antigravity_reraises_when_create_never_yields_an_id() -> None:
    client = _fake_client(
        streams=[_FakeStream([], raise_after=0)], terminal=_terminal_interaction()
    )
    streamer = ResearchProgressStreamer(status=None, label="Antigravity")

    async def _persist(_interaction_id: str) -> None:
        return None

    # No interaction.created ever arrived, so there is no id to resume: the error propagates to the
    # cog's failure path instead of being swallowed into a poll.
    raised = False
    try:
        await agent.stream_antigravity(
            client=as_client(fake=client),
            agent="a",
            brief="b",
            system_instruction="s",
            streamer=streamer,
            on_created=_persist,
        )
    except RuntimeError:
        raised = True
    assert raised is True


async def test_resume_research_stream_drives_from_get_stream() -> None:
    client = _fake_client(
        streams=[_FakeStream([_completed_event(event_id="e1")])], terminal=_terminal_interaction()
    )
    streamer = ResearchProgressStreamer(status=None, label="Antigravity")
    result = await agent.resume_research_stream(
        client=as_client(fake=client), interaction_id="int_9", streamer=streamer
    )
    # Resume re-attaches via get(stream=True) and never calls create.
    assert client.aio.interactions.create_kwargs == {}
    assert result.ok is True


def test_is_terminal_event_classifies_statuses() -> None:
    assert agent._is_terminal_event(event=_as_event(_completed_event())) is True
    assert (
        agent._is_terminal_event(
            event=_as_event(
                SimpleNamespace(event_type="error", error=SimpleNamespace(message="boom"))
            )
        )
        is True
    )
    running = SimpleNamespace(event_type="interaction.status_update", status="in_progress")
    assert agent._is_terminal_event(event=_as_event(running)) is False
    # `requires_action` stays non-terminal: it is a generic Interactions status, not a leftover of
    # the removed plan-approval flow, and calling it terminal would end a live stream early.
    waiting = SimpleNamespace(event_type="interaction.status_update", status="requires_action")
    assert agent._is_terminal_event(event=_as_event(waiting)) is False
    failed = SimpleNamespace(event_type="interaction.status_update", status="budget_exceeded")
    assert agent._is_terminal_event(event=_as_event(failed)) is True
    assert agent._is_terminal_event(event=_as_event(_thought_event("x"))) is False


def test_to_result_extracts_text_image_and_usage() -> None:
    image_b64 = base64.b64encode(b"PNGBYTES").decode()
    interaction = SimpleNamespace(
        id="int_123",
        status="completed",
        output_text="# Report\nbody",
        usage=SimpleNamespace(total_input_tokens=250000, total_output_tokens=60000),
        steps=[
            SimpleNamespace(
                type="model_output", content=[SimpleNamespace(type="image", data=image_b64)]
            )
        ],
    )
    result = agent._to_result(interaction=interaction)
    assert result.ok is True
    assert result.report_text.startswith("# Report")
    assert result.image_bytes == b"PNGBYTES"
    assert result.input_tokens == 250000
    assert result.output_tokens == 60000


def test_to_result_handles_failure_and_missing_fields() -> None:
    # Every field the SDK leaves unset on a failed run comes back None, not absent.
    interaction = SimpleNamespace(
        id="int_x", status="failed", output_text=None, usage=None, steps=None
    )
    result = agent._to_result(interaction=interaction)
    assert result.ok is False
    assert result.report_text == ""
    assert result.image_bytes is None
    assert result.input_tokens == 0


# ----- progress streamer --------------------------------------------------------------------


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
    status = _FakeStatusMessage()
    streamer = ResearchProgressStreamer(
        status=status, label="Antigravity", reasoning="thinking", started_at=100.0
    )
    await streamer._write_preview_snapshot()
    assert len(status.edits) == 1
    assert cast("AllowedMentions", status.edits[0]["allowed_mentions"]).everyone is False
    # A second write of the same rendered snapshot is a no-op, so the editor never spams edits.
    await streamer._write_preview_snapshot()
    assert len(status.edits) == 1
    streamer.reasoning += " more"
    await streamer._write_preview_snapshot()
    assert len(status.edits) == 2


async def test_streamer_stream_accumulates_and_stops_editor_cleanly() -> None:
    status = _FakeStatusMessage()
    streamer = ResearchProgressStreamer(
        status=status, label="Antigravity", preview_interval_seconds=0.01
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

    def __init__(self, *, error: Exception, attempts_seen: int = 1) -> None:
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
    records = {name: _recorded(monkeypatch=monkeypatch, level=name) for name in ("info", "warn")}

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
    warns = _recorded(monkeypatch=monkeypatch, level="warn")

    streamer._ensure_editor_started()
    await asyncio.wait_for(status.seen.wait(), timeout=5)
    await streamer._stop_editor()

    assert [(message, fields["error_type"]) for message, fields in warns] == [
        ("research preview edit failed; continuing the run", "HTTPException")
    ]
    assert warns[0][1]["_exc_info"] is not None


# ----- research module helpers --------------------------------------------------------------


def test_fallback_thread_name_uses_first_line() -> None:
    name = research_cog._fallback_thread_name(brief="研究 TPU 的歷史與競爭格局\n更多細節")
    assert name.startswith("研究 TPU")
    assert "\n" not in name
    assert research_cog._fallback_thread_name(brief="   ") == "深度研究"


def test_terminal_phase_mapping() -> None:
    assert research_cog._terminal_phase(status="completed") == "done"
    assert research_cog._terminal_phase(status="cancelled") == "cancelled"
    assert research_cog._terminal_phase(status="budget_exceeded") == "failed"


def test_deep_research_available_requires_enabled_and_key() -> None:
    config = LLMConfig()
    config.deep_research_enabled = True
    config.gemini_api_key = "AIza-key"
    assert config.deep_research_available is True
    config.gemini_api_key = "   "
    assert config.deep_research_available is False
    config.gemini_api_key = "AIza-key"
    config.deep_research_enabled = False
    assert config.deep_research_available is False


def test_owner_allowed_mentions_blocks_everyone_and_roles() -> None:
    mentions = owner_allowed_mentions(owner_id=42)
    assert mentions.everyone is False
    assert mentions.roles is False
    users = mentions.users
    assert isinstance(users, list)
    assert [obj.id for obj in users] == [42]


def test_failure_text_distinguishes_budget() -> None:
    assert "成本上限" in research_cog._failure_text(status="budget_exceeded")
    assert "取消" in research_cog._failure_text(status="cancelled")
    assert research_cog._failure_text(status="failed")


# ----- persistence (reply.db) ---------------------------------------------------------------


async def _only_resumable(*, thread_id: int) -> rdb.PersistentResearchSession | None:
    """The one resumable row for a thread; the store has no single-row reader left to use."""
    return next((row for row in await rdb.list_resumable() if row.thread_id == thread_id), None)


async def test_session_round_trip(research_isolated_db: None) -> None:
    await rdb.insert_session(
        thread_id=1,
        owner_id=99,
        channel_id=7,
        guild_id=5,
        source_message_id=3,
        agent="antigravity-preview-09-2026",
        brief="研究 X",
    )
    session = await _only_resumable(thread_id=1)
    assert session is not None
    assert session.owner_id == 99
    assert session.interaction_id is None
    assert await _only_resumable(thread_id=999) is None


async def test_set_interaction_and_phase(research_isolated_db: None) -> None:
    await rdb.insert_session(
        thread_id=2,
        owner_id=1,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="antigravity-preview-09-2026",
        brief="b",
    )
    await rdb.set_interaction(thread_id=2, interaction_id="int_abc")
    session = await _only_resumable(thread_id=2)
    assert session is not None
    assert session.interaction_id == "int_abc"
    assert session.agent == "antigravity-preview-09-2026"
    await rdb.set_phase(thread_id=2, phase="done")
    assert await _only_resumable(thread_id=2) is None
    assert await rdb.active_thread_for_owner(owner_id=1) is None


async def test_active_thread_for_owner_excludes_terminal(research_isolated_db: None) -> None:
    await rdb.insert_session(
        thread_id=10,
        owner_id=500,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="antigravity-preview-09-2026",
        brief="b",
    )
    assert await rdb.active_thread_for_owner(owner_id=500) == 10
    await rdb.set_phase(thread_id=10, phase="done")
    assert await rdb.active_thread_for_owner(owner_id=500) is None
    assert await rdb.active_thread_for_owner(owner_id=12345) is None


async def test_list_resumable_only_returns_researching(research_isolated_db: None) -> None:
    # A researching session beside two terminal ones: only the first may come back resumable.
    seeded: tuple[tuple[int, ResearchPhase], ...] = (
        (20, "researching"),
        (21, "cancelled"),
        (22, "done"),
    )
    for thread_id, phase in seeded:
        await rdb.insert_session(
            thread_id=thread_id,
            owner_id=thread_id,
            channel_id=1,
            guild_id=1,
            source_message_id=1,
            agent="antigravity-preview-09-2026",
            brief="b",
        )
        await rdb.set_interaction(thread_id=thread_id, interaction_id="int_x")
        await rdb.set_phase(thread_id=thread_id, phase=phase)
    resumable = await rdb.list_resumable()
    assert {session.thread_id for session in resumable} == {20}


async def test_a_legacy_planning_row_no_longer_blocks_its_owner(
    research_isolated_db: None,
) -> None:
    # Written the way the removed escalation wrote it: the phase literal is gone from the model, so
    # seed it through the ORM. A stuck row must not hold the one-per-owner slot forever.
    async with rdb.open_session() as session:
        session.add(
            rdb.ResearchSessionRow(
                thread_id=60,
                owner_id=61,
                channel_id=1,
                guild_id=1,
                source_message_id=1,
                agent="deep-research-preview-04-2026",
                interaction_id="plan_1",
                brief="b",
                phase="planning",
            )
        )
        await session.commit()
    assert await rdb.active_thread_for_owner(owner_id=61) is None
    assert await rdb.list_resumable() == []


# ----- delivery completion footer -----------------------------------------------------------


class _FakeStatusMessage:
    """Records `edit` calls on the opening status message."""

    id = 2

    def __init__(self) -> None:
        self.edits: list[dict[str, object]] = []

    async def edit(self, **kwargs: object) -> None:
        self.edits.append(kwargs)


class _FakeThread:
    """Records `send` calls and exposes a guild upload limit, like a real Thread."""

    id = 1

    def __init__(self) -> None:
        self.sends: list[dict[str, object]] = []
        self.guild = SimpleNamespace(filesize_limit=10 * 1024 * 1024)

    async def send(self, **kwargs: object) -> None:
        self.sends.append(kwargs)


def _completed_result(
    *, report_text: str, image_bytes: bytes | None = None
) -> agent.ResearchResult:
    return agent.ResearchResult(
        status="completed", report_text=report_text, image_bytes=image_bytes
    )


async def test_delivery_keeps_footer_message_under_the_limit() -> None:
    status = _FakeStatusMessage()
    thread = _FakeThread()
    footer = "-# antigravity-preview-09-2026 · ⬆ 0 ⬇ 0 · $0.00000000"
    # A report chunk that sits just under the 2000-char message cap; appending the footer inline
    # would overflow, so it must ride its own trailing message.
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text="X" * 1990),
        footer=footer,
        media_delivery=_disabled_delivery(),
    )
    contents = [str(edit["content"]) for edit in status.edits]
    contents += [str(send["content"]) for send in thread.sends]
    assert all(len(content) <= 2000 for content in contents)
    # The footer + owner ping + research.md ride the trailing send, not the near-limit chunk.
    footer_send = thread.sends[-1]
    assert "<@1>" in str(footer_send["content"])
    assert footer in str(footer_send["content"])
    assert footer_send["files"]
    # Every report message carries the owner-only mention policy so agent text can't mass-ping.
    mentions = cast("AllowedMentions", footer_send["allowed_mentions"])
    assert mentions.everyone is False
    assert mentions.roles is False
    assert isinstance(mentions.users, list)
    assert [user.id for user in mentions.users] == [1]
    assert status.edits[0]["allowed_mentions"] is mentions


async def test_delivery_inlines_footer_for_short_reports() -> None:
    status = _FakeStatusMessage()
    thread = _FakeThread()
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text="# Report\nbody"),
        footer="-# footer",
        media_delivery=_disabled_delivery(),
    )
    # One message: the opening status edited into report + footer + the research.md attachment.
    assert not thread.sends
    assert len(status.edits) == 1
    assert "<@1>" in str(status.edits[0]["content"])
    assert status.edits[0]["files"]


async def test_delivery_hosts_oversized_report_file(tmp_path: Path) -> None:
    """A report file too big to attach is hosted and its URL linked instead of silently dropped."""
    status = _FakeStatusMessage()
    thread = _FakeThread()
    thread.guild = SimpleNamespace(filesize_limit=4)  # tiny ceiling so research.md is oversize
    planner = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(
            config=make_media_hosting_config(
                enabled=True, base_url="https://media.test", serve_dir=str(tmp_path)
            )
        )
    )
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text="# Report\nbody"),
        footer="-# footer",
        media_delivery=planner,
    )
    # The report .md was hosted (no native attachment); its URL rides the message content.
    edit = status.edits[0]
    assert not edit.get("files")
    content = str(edit["content"])
    assert any(line.startswith("https://media.test/") for line in content.splitlines())


async def test_delivery_attaches_both_files_when_each_fits_but_combined_over() -> None:
    """Host-off contract: md + png that each fit but jointly exceed the limit BOTH attach natively.

    Routing both through one `plan()` call would fire the planner's combined-peel and drop the
    larger (the report), so delivery decides each attachment on its own.
    """
    status = _FakeStatusMessage()
    thread = _FakeThread()
    thread.guild = SimpleNamespace(filesize_limit=100)  # each file fits, md + png together do not
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text="R" * 60, image_bytes=b"x" * 60),
        footer="-# footer",
        media_delivery=_disabled_delivery(),
    )
    edit = status.edits[0]
    files = edit["files"]
    assert isinstance(files, list)
    assert len(files) == 2  # research.md AND research.png both attached, neither dropped
    assert "https://" not in str(edit["content"])  # nothing was hosted


async def test_delivery_names_a_report_file_it_leaves_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """With hosting off, a file too big to attach is left out, and the log says which one."""
    warns = _recorded(monkeypatch=monkeypatch, level="warn")
    status = _FakeStatusMessage()
    thread = _FakeThread()
    thread.guild = SimpleNamespace(filesize_limit=4)  # tiny ceiling so research.md is oversize
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text="# Report\nbody"),
        footer="-# footer",
        media_delivery=_disabled_delivery(),
    )
    assert not status.edits[0].get("files")
    assert warns == [
        (
            "research report file too big to attach and not hosted; left out",
            {"thread_id": 1, "filename": "research.md"},
        )
    ]


@pytest.mark.parametrize(
    ("report_text", "hosted"),
    [("# Report\nbody", False), ("X" * 1990, False), ("# Report\nbody", True)],
    ids=["inline", "own_message", "hosted_file"],
)
async def test_a_delivered_reports_usage_footer_never_reaches_the_bots_history(
    tmp_path: Path, report_text: str, hosted: bool
) -> None:
    """Every delivered shape renders back as the bot's own history with the footer gone."""
    status = _FakeStatusMessage()
    thread = _FakeThread()
    planner = _disabled_delivery()
    if hosted:
        thread.guild = SimpleNamespace(filesize_limit=4)  # tiny ceiling so research.md is oversize
        planner = MediaDeliveryPlanner(
            media_hosting=MediaHostingService(
                config=make_media_hosting_config(
                    enabled=True, base_url="https://media.test", serve_dir=str(tmp_path)
                )
            )
        )
    footer = "-# antigravity-preview-09-2026 · ⬆ 1,234 ⬇ 567 · $0.00236800"
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_id=1,
        result=_completed_result(report_text=report_text),
        footer=footer,
        media_delivery=planner,
    )
    posted = [str(write["content"]) for write in [*status.edits, *thread.sends]]
    assert footer in posted[-1]
    builder = MessageInputBuilder(
        bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))),
        runtime_models=RuntimeModelCatalog(),
        # Rendering a message's text never reaches its attachments.
        attachment_handler=cast("AttachmentRenderer", SimpleNamespace()),
    )

    history = [
        await builder.get_cleaned_content(
            message=as_message(
                fake=SimpleNamespace(
                    content=content,
                    author=SimpleNamespace(id=999),
                    embeds=[],
                    snapshots=[],
                    is_system=lambda: False,
                )
            )
        )
        for content in posted
    ]

    assert not any("⬆" in text for text in history)
    # Only the footer goes: the report, the owner ping and any hosted link all stay.
    assert history == [content.replace(f"\n\n{footer}", "").strip() for content in posted]


# ----- restart resume sweep -----------------------------------------------------------------


def _research_cog(*, enabled: bool) -> research_cog.ResearchCogs:
    """A cog carrying only what the resume sweep touches: no bot, no client, no gateway.

    The key is always present so the switch alone decides `deep_research_available`, and neither
    field is left to a deployment's `.env`.
    """
    cog = research_cog.ResearchCogs.__new__(research_cog.ResearchCogs)
    config = LLMConfig()
    config.deep_research_enabled = enabled
    config.gemini_api_key = "AIza-key"
    cog.config = config
    cog._active_threads = set()
    cog._tasks = set()
    return cog


async def _seed_researching(
    *,
    thread_id: int,
    owner_id: int,
    stored_id: bool = True,
    agent: str = "antigravity-preview-09-2026",
) -> None:
    """Seeds one in-flight row, as a launch that never reached a terminal phase left it.

    `stored_id=False` is a launch that restarted before its interaction id was persisted.
    """
    await rdb.insert_session(
        thread_id=thread_id,
        owner_id=owner_id,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent=agent,
        brief="b",
    )
    if stored_id:
        await rdb.set_interaction(thread_id=thread_id, interaction_id=f"int_{thread_id}")


async def test_resume_sweep_reattaches_to_nothing_while_the_switch_is_off(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(**_kwargs: object) -> None:
        raise AssertionError("the resume must not reach the provider while the switch is off")

    monkeypatch.setattr(research_cog, "resume_research_stream", _boom)
    cog = _research_cog(enabled=False)
    await _seed_researching(thread_id=30, owner_id=300)

    await cog._resume_all()

    # Nothing is attached and nothing is delivered, and the row stays `researching` because that
    # is what it is: the interaction runs server-side and a later start with the switch on may
    # still deliver it.
    assert not cog._tasks
    assert cog._active_threads == set()
    assert [session.thread_id for session in await rdb.list_resumable()] == [30]


async def test_resume_sweep_stays_off_without_a_gemini_key(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(**_kwargs: object) -> None:
        raise AssertionError("a keyless deployment must not reach the provider either")

    monkeypatch.setattr(research_cog, "resume_research_stream", _boom)
    # The gate is `deep_research_available`, so a switched-on deployment with no key is refused
    # here rather than at `genai.Client` inside the resume's own try.
    cog = _research_cog(enabled=True)
    cog.config.gemini_api_key = "   "
    await _seed_researching(thread_id=35, owner_id=350)

    await cog._resume_all()

    assert not cog._tasks
    assert [session.thread_id for session in await rdb.list_resumable()] == [35]


async def test_resume_sweep_still_resumes_when_the_switch_is_on(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cog = _research_cog(enabled=True)
    resumed: list[int] = []

    async def _fake_resume_one(*, session: rdb.PersistentResearchSession) -> None:
        resumed.append(session.thread_id)

    monkeypatch.setattr(cog, "_resume_one", _fake_resume_one)
    await _seed_researching(thread_id=40, owner_id=400)

    await cog._resume_all()
    await asyncio.gather(*cog._tasks)

    assert resumed == [40]
    assert cog._active_threads == {40}
    # The sweep leaves the phase alone: the resumed run decides its own terminal phase.
    assert [session.thread_id for session in await rdb.list_resumable()] == [40]


# ----- permission refusals ------------------------------------------------------------------


class _ResearchInteraction(FakeInteraction):
    """A `/deep_research` invocation from a guild text channel whose posts the test decides."""

    def __init__(self, *, channel: MagicMock) -> None:
        """Initializes the shared fake plus the channel the command was run in."""
        super().__init__()
        self.channel = channel


class _Anchor(FakeDiscordMessage):
    """A message the research thread hangs off, refusing it with `error` or opening `thread`."""

    def __init__(
        self, *, channel: MagicMock, error: Exception | None = None, thread: object = None
    ) -> None:
        """Initializes the identity `_start_for` reads on top of the shared message fake."""
        super().__init__()
        self.id = 10
        self.guild = SimpleNamespace(id=1)
        self.channel = channel
        self.author = FakeUser(user_id=300)
        self.error = error
        self.thread = thread

    async def create_thread(self, **_kwargs: object) -> object:
        """Opens `thread`, or fails the way Discord fails the thread."""
        if self.error is not None:
            raise self.error
        return self.thread


def _text_channel(*, permissions: Permissions | None = None) -> MagicMock:
    """A guild text channel resolving `permissions` (default: all) for the bot's own member."""
    channel = MagicMock(spec=TextChannel)
    channel.id = 20
    channel.guild = SimpleNamespace(me=object())
    channel.permissions_for.return_value = permissions or Permissions.all()
    return channel


def _launching_cog(*, monkeypatch: pytest.MonkeyPatch) -> research_cog.ResearchCogs:
    """A cog that gets as far as `create_thread` without a title model behind it."""
    cog = _research_cog(enabled=True)
    cog._owner_locks = KeyedLockManager()

    async def _title(*, brief: str) -> str:
        del brief
        return "research"

    monkeypatch.setattr(target=cog, name="_generate_thread_name", value=_title)
    return cog


def _recorded(
    *, monkeypatch: pytest.MonkeyPatch, level: str
) -> list[tuple[str, dict[str, object]]]:
    """Captures what the cog reports at one level, the way the rest of the suite reads logfire."""
    records: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        target=research_cog.logfire,
        name=level,
        value=lambda message, **fields: records.append((message, fields)),
    )
    return records


async def test_deep_research_answers_when_the_bot_cannot_post_in_the_channel(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slash command reaches a channel the bot may not post in, so the answer rides the token."""
    channel = _text_channel()
    channel.send = AsyncMock(side_effect=make_forbidden(message="Missing Access"))
    interaction = _ResearchInteraction(channel=channel)
    warns = _recorded(monkeypatch=monkeypatch, level="warn")

    await _launching_cog(monkeypatch=monkeypatch).deep_research(
        as_interaction(fake=interaction), topic="topic"
    )

    assert interaction.response.deferred is True
    assert [edit.get("content") for edit in interaction.edits] == [
        "我在這個頻道的權限不夠,開不了研究串"
    ]
    assert warns == [
        ("deep research cannot post its anchor in this channel", {"channel_id": 20, "owner_id": 1})
    ], "a permission the bot cannot earn needs the ids and no traceback"


async def test_deep_research_withdraws_its_anchor_when_the_thread_is_refused(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anchor posted but the thread did not, so the requester is told why, not to retry."""
    channel = _text_channel()
    anchor = _Anchor(error=make_forbidden(message="Missing Permissions"), channel=channel)
    channel.send = AsyncMock(return_value=anchor)
    interaction = _ResearchInteraction(channel=channel)
    warns = _recorded(monkeypatch=monkeypatch, level="warn")

    await _launching_cog(monkeypatch=monkeypatch).deep_research(
        as_interaction(fake=interaction), topic="topic"
    )

    assert anchor.deleted is True
    assert [edit.get("content") for edit in interaction.edits] == [
        "我在這個頻道的權限不夠,開不了研究串"
    ]
    assert [fields for _, fields in warns] == [{"message_id": 10, "owner_id": 1, "channel_id": 20}]


async def test_a_marker_launch_says_so_when_the_thread_is_refused(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `<deep-research>` entry shares `_start_for`, so it answers the refusal the same way."""
    anchor = _Anchor(error=make_forbidden(message="Missing Permissions"), channel=_text_channel())
    warns = _recorded(monkeypatch=monkeypatch, level="warn")

    await _launching_cog(monkeypatch=monkeypatch).launch(
        message=as_message(fake=anchor), brief="b"
    )

    assert [reply.get("content") for reply in anchor.replies] == [
        "我在這個頻道的權限不夠,開不了研究串"
    ]
    assert warns == [
        (
            "deep research cannot open a thread in this channel",
            {"message_id": 10, "owner_id": 300, "channel_id": 20},
        )
    ]


async def test_both_entry_points_name_the_owners_running_research_alike(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second launch points at the running thread in one wording, however it was asked for."""
    await _seed_researching(thread_id=_THREAD_ID, owner_id=_OWNER_ID)
    channel = _text_channel()
    request = _Anchor(channel=channel)
    slash_anchor = _Anchor(channel=channel)
    channel.send = AsyncMock(return_value=slash_anchor)
    interaction = _ResearchInteraction(channel=channel)
    interaction.user = FakeUser(user_id=_OWNER_ID)
    cog = _launching_cog(monkeypatch=monkeypatch)

    await cog.launch(message=as_message(fake=request), brief="b")
    await cog.deep_research(as_interaction(fake=interaction), topic="topic")

    running = f"你已經有一個深度研究在進行了:<#{_THREAD_ID}>"
    assert [reply.get("content") for reply in request.replies] == [running]
    assert [edit.get("content") for edit in interaction.edits] == [running]
    assert slash_anchor.deleted is True


async def test_a_thread_failure_that_is_not_a_refusal_keeps_its_traceback(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The carve-out is for `Forbidden` alone; a 5xx is still something to look at."""
    anchor = _Anchor(error=make_server_error(), channel=_text_channel())
    errors = _recorded(monkeypatch=monkeypatch, level="error")

    await _launching_cog(monkeypatch=monkeypatch).launch(
        message=as_message(fake=anchor), brief="b"
    )

    assert [reply.get("content") for reply in anchor.replies] == ["開研究串失敗了,等等再試一次"]
    assert len(errors) == 1
    assert errors[0][1].get("_exc_info") is not None, (
        "a transport failure still needs its traceback"
    )


@pytest.mark.parametrize(
    ("failure", "level", "traceback"),
    [
        (make_forbidden(message="Missing Permissions"), "warn", False),
        (make_not_found(message="Unknown Message"), "info", False),
        (make_invalid_form_body(), "info", False),
        (make_server_error(), "warn", True),
    ],
    ids=["refused", "message_gone", "reply_target_gone", "broke"],
)
async def test_a_launch_that_cannot_say_why_it_stopped_logs_it(
    research_isolated_db: None,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    level: str,
    traceback: bool,
) -> None:
    """The launch has already ended, so its notice failing is logged here and raises nothing."""
    anchor = _Anchor(error=make_server_error(), channel=_text_channel())

    async def refuse(**kwargs: object) -> NoReturn:
        """Fails the reply the way Discord does."""
        del kwargs
        raise failure

    anchor.reply = refuse  # ty: ignore[invalid-assignment]
    records = {name: _recorded(monkeypatch=monkeypatch, level=name) for name in ("info", "warn")}

    await _launching_cog(monkeypatch=monkeypatch).launch(
        message=as_message(fake=anchor), brief="b"
    )

    assert [
        (name, fields["message_id"], fields["channel_id"], "_exc_info" in fields)
        for name, found in records.items()
        for _, fields in found
    ] == [(level, 10, 20, traceback)]


@pytest.mark.parametrize(
    ("failure", "level"),
    [(make_not_found(message="Unknown Channel"), "info"), (make_forbidden(), "warn")],
    ids=["deleted", "shut_out"],
)
async def test_a_thread_the_bot_lost_access_to_is_not_reported_as_deleted(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, level: str
) -> None:
    """Both end the resume the same way, but only a deletion is the owner's doing."""
    cog = _research_cog(enabled=True)
    cog.bot = as_bot(
        fake=SimpleNamespace(
            get_channel=lambda channel_id: None, fetch_channel=AsyncMock(side_effect=failure)
        )
    )
    records = {name: _recorded(monkeypatch=monkeypatch, level=name) for name in ("info", "warn")}

    assert await cog._fetch_thread(thread_id=5) is None
    assert [(name, fields) for name, found in records.items() for _, fields in found] == [
        (level, {"thread_id": 5})
    ]


async def test_deep_research_refuses_up_front_where_it_cannot_open_a_thread(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The channel already says so, so nothing is posted, pinged or titled before the refusal."""
    channel = _text_channel(
        permissions=Permissions(
            view_channel=True, send_messages=True, send_messages_in_threads=True, attach_files=True
        )
    )
    channel.send = AsyncMock()
    interaction = _ResearchInteraction(channel=channel)

    await _launching_cog(monkeypatch=monkeypatch).deep_research(
        as_interaction(fake=interaction), topic="topic"
    )

    assert interaction.response.deferred is False
    assert interaction.response.sent == [
        {"content": "我在這個頻道的權限不夠,開不了研究串", "ephemeral": True}
    ]
    channel.send.assert_not_called()
    channel.permissions_for.assert_called_once_with(channel.guild.me)


class _RefusingThread(_FakeThread):
    """A research thread whose overwrites changed under the run, so every send is refused."""

    async def send(self, **kwargs: object) -> None:
        """Refuses the way Discord refuses a thread the bot may no longer write in."""
        del kwargs
        raise make_forbidden(message="Missing Permissions")


async def test_a_refused_thread_write_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permission that changed mid-run is expected, so it is a warn carrying only the id."""
    warns = _recorded(monkeypatch=monkeypatch, level="warn")

    sent = await _research_cog(enabled=True)._safe_send(
        thread=cast("Thread", _RefusingThread()), content="-# Researching..."
    )

    assert sent is None
    assert warns == [("research thread refused a message", {"thread_id": 1})]


async def test_a_refused_report_is_a_warn_even_on_its_last_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last chunk logs `error` for a real failure, but a refusal is the server's setting."""
    warns = _recorded(monkeypatch=monkeypatch, level="warn")
    errors = _recorded(monkeypatch=monkeypatch, level="error")

    await deliver_report(
        thread=cast("Thread", _RefusingThread()),
        status=None,
        owner_id=1,
        result=_completed_result(report_text="report"),
        footer="-# footer",
        media_delivery=_disabled_delivery(),
    )

    assert errors == []
    assert warns == [
        (
            "research thread refused a report message",
            {"thread_id": 1, "chunk_index": 0, "is_last": True},
        )
    ]


class _RefusingStatus:
    """An opening status message whose edit Discord refuses after reading the upload body."""

    async def edit(self, *, files: list[File] | None = None, **kwargs: object) -> None:
        """Consumes every file the way a real multipart edit does, then refuses."""
        del kwargs
        for file in files or []:
            file.fp.read()
        raise make_forbidden(message="Missing Permissions")


class _ReadingThread:
    """A thread that records what each attached file actually carried when it was sent."""

    id = 1

    def __init__(self) -> None:
        """Initializes the guild upload limit and the bodies read off each send's files."""
        self.guild = SimpleNamespace(filesize_limit=10 * 1024 * 1024)
        self.bodies: list[bytes] = []

    async def send(self, *, files: list[File] | None = None, **kwargs: object) -> None:
        """Reads every attached file, as the upload would."""
        del kwargs
        self.bodies.extend(file.fp.read() for file in files or [])


async def test_a_refused_status_edit_still_hands_the_fallback_a_full_report_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refused edit already read the file, so the fallback send must get it rewound."""
    warns = _recorded(monkeypatch=monkeypatch, level="warn")
    thread = _ReadingThread()

    await deliver_report(
        thread=cast("Thread", thread),
        status=as_message(fake=_RefusingStatus()),
        owner_id=1,
        result=_completed_result(report_text="the whole report"),
        footer="-# footer",
        media_delivery=_disabled_delivery(),
    )

    assert thread.bodies == [b"the whole report"]
    assert warns == [
        ("research thread refused the report edit", {"thread_id": 1, "chunk_index": 0})
    ]


# ----- run exits ----------------------------------------------------------------------------

# The `_Anchor` author owns every run below, and each run lands in this thread.
_OWNER_ID = 300
_THREAD_ID = 50


class _RunStatus:
    """A run's opening status message, whose edits land in its thread's write log."""

    id = 2

    def __init__(self, *, thread: "_RunThread") -> None:
        """Binds the status to the thread whose write log and failure it shares."""
        self.thread = thread

    async def edit(self, **kwargs: object) -> None:
        """Records the edit, or fails it the way its thread fails every later write."""
        if self.thread.error is not None:
            raise self.thread.error
        self.thread.writes.append(kwargs)


class _RunThread:
    """A research thread logging every write of a run in order, status edits included.

    With `error` set, every write fails with it except the opening status post while
    `status_posts` holds.
    """

    id = _THREAD_ID

    def __init__(
        self,
        *,
        error: Exception | None = None,
        status_posts: bool = True,
        earlier: list[FakeDiscordMessage] | None = None,
    ) -> None:
        """Initializes the write log, the guild upload limit, how writes fail, and the history."""
        self.guild = SimpleNamespace(filesize_limit=10 * 1024 * 1024)
        self.writes: list[dict[str, object]] = []
        self.error = error
        self.status_posts = status_posts
        self.sends = 0
        self.deleted = False
        self.earlier = earlier or []

    async def history(self, **kwargs: object) -> "AsyncIterator[FakeDiscordMessage]":
        """Hands back what the thread held before the restart, oldest first."""
        del kwargs
        for message in self.earlier:
            yield message

    async def send(self, **kwargs: object) -> _RunStatus:
        """Records a post and answers with the message it created, as Discord does."""
        self.sends += 1
        if self.error is not None and not (self.sends == 1 and self.status_posts):
            raise self.error
        self.writes.append(kwargs)
        return _RunStatus(thread=self)

    async def delete(self) -> None:
        """Records the thread's deletion, or fails it the way every later write fails."""
        if self.error is not None:
            raise self.error
        self.deleted = True


class _ThreadBot:
    """The bot surface a run reads: its own user, and the thread lookup a resume starts from."""

    user = FakeUser(user_id=900, name="bot", bot=True)

    def __init__(self, *, thread: _RunThread | None) -> None:
        """Initializes the one thread the cache holds; None is a thread deleted meanwhile."""
        self.thread = thread

    def get_channel(self, channel_id: int) -> MagicMock | None:
        """Answers the cache with the thread dressed as the nextcord `Thread` a resume wants."""
        if self.thread is None or channel_id != self.thread.id:
            return None
        channel = MagicMock(spec=Thread)
        channel.id = self.thread.id
        channel.guild = self.thread.guild
        channel.send = self.thread.send
        channel.history = self.thread.history
        return channel

    async def fetch_channel(self, channel_id: int) -> None:
        """Answers the REST lookup the way Discord does for a deleted thread."""
        del channel_id
        raise make_not_found(message="Unknown Channel")


def _settling_client(*, status: str) -> SimpleNamespace:
    """A Gemini client whose research streams to its end and settles with `status`."""
    return _fake_client(
        streams=[_FakeStream([_created_event(), _completed_event()])],
        terminal=_terminal_interaction(status=status, input_tokens=1234, output_tokens=567),
    )


def _failing_client(*, error: Exception) -> SimpleNamespace:
    """A Gemini client whose research create fails outright."""

    async def create(**_kwargs: object) -> None:
        raise error

    return SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=create)))


def _running_cog(
    *, monkeypatch: pytest.MonkeyPatch, client: object, thread: _RunThread | None = None
) -> research_cog.ResearchCogs:
    """A cog that runs research on `client` end to end and delivers with hosting off.

    `thread` is what a resume finds by id. The agent is priced, so the usage footer's cost is
    a known number rather than whatever price table the worker holds.
    """
    cog = _launching_cog(monkeypatch=monkeypatch)
    cog.bot = as_bot(fake=_ThreadBot(thread=thread))
    cog.runtime_models = RuntimeModelCatalog()
    cog.media_delivery = _disabled_delivery()
    cog.interactions_client = as_client(fake=client)
    rates = {
        cog.runtime_models.antigravity_model.name: ModelPriceEntry(
            input_cost_per_token=1e-6, output_cost_per_token=2e-6
        )
    }
    monkeypatch.setattr("discordbot.utils.model_pricing.load_model_info", lambda: rates)
    return cog


async def _launch_run(*, cog: research_cog.ResearchCogs, thread: _RunThread) -> None:
    """Launches a research from a marker and waits out the run it spawned."""
    anchor = _Anchor(channel=_text_channel(), thread=thread)
    await cog.launch(message=as_message(fake=anchor), brief="b")
    await asyncio.gather(*cog._tasks)


async def _resume_run(
    *,
    cog: research_cog.ResearchCogs,
    stored_id: bool = True,
    agent: str = "antigravity-preview-09-2026",
) -> None:
    """Resumes the owner's in-flight row after a restart and waits out the run it spawned."""
    await _seed_researching(
        thread_id=_THREAD_ID, owner_id=_OWNER_ID, stored_id=stored_id, agent=agent
    )
    await cog._resume_all()
    await asyncio.gather(*cog._tasks)


async def _assert_owner_released(*, cog: research_cog.ResearchCogs, phase: str) -> None:
    """The run's row ended in `phase` and nothing holds its owner's one slot any more."""
    async with rdb.open_session() as session:
        row = await session.get(entity=rdb.ResearchSessionRow, ident=_THREAD_ID)
    assert row is not None
    assert row.phase == phase
    assert await rdb.active_thread_for_owner(owner_id=_OWNER_ID) is None
    assert cog._active_threads == set()


def _assert_pings_only_the_owner(*, write: dict[str, object]) -> None:
    """The write mentions its owner, and its mention policy lets nobody else be pinged."""
    assert f"<@{_OWNER_ID}>" in str(write["content"])
    mentions = cast("AllowedMentions", write["allowed_mentions"])
    assert mentions.everyone is False
    assert mentions.roles is False
    assert isinstance(mentions.users, list)
    assert [user.id for user in mentions.users] == [_OWNER_ID]


@pytest.mark.parametrize(
    ("settles", "phase"),
    [
        ("completed", "done"),
        ("cancelled", "cancelled"),
        ("budget_exceeded", "failed"),
        (RuntimeError("quota"), "failed"),
    ],
    ids=["completed", "cancelled", "budget_exceeded", "create_fails"],
)
async def test_every_exit_of_a_launched_run_records_its_phase_and_frees_the_owner(
    research_isolated_db: None,
    monkeypatch: pytest.MonkeyPatch,
    settles: str | Exception,
    phase: str,
) -> None:
    client = (
        _failing_client(error=settles)
        if isinstance(settles, Exception)
        else _settling_client(status=settles)
    )
    cog = _running_cog(monkeypatch=monkeypatch, client=client)

    await _launch_run(cog=cog, thread=_RunThread())

    await _assert_owner_released(cog=cog, phase=phase)


async def test_a_run_whose_delivery_raises_still_ends_failed_and_frees_the_owner(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cog = _running_cog(monkeypatch=monkeypatch, client=_settling_client(status="completed"))
    cog.media_delivery = cast(
        "MediaDeliveryPlanner", SimpleNamespace(plan=AsyncMock(side_effect=OSError("host down")))
    )
    thread = _RunThread()

    await _launch_run(cog=cog, thread=thread)

    await _assert_owner_released(cog=cog, phase="failed")
    assert thread.writes[-1]["content"] == "-# Research failed (Antigravity)"


def _lock_reply_db(
    *, monkeypatch: pytest.MonkeyPatch, call: str
) -> list[tuple[str, dict[str, object]]]:
    """Makes one research store call fail the way a locked `reply.db` does; returns the errors."""

    async def _locked(**_kwargs: object) -> None:
        raise OperationalError("research", None, sqlite3.OperationalError("database is locked"))

    monkeypatch.setattr(target=rdb, name=call, value=_locked)
    return _recorded(monkeypatch=monkeypatch, level="error")


@pytest.mark.parametrize("call", ["active_thread_for_owner", "insert_session"])
async def test_deep_research_answers_and_withdraws_its_posts_when_reply_db_fails(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch, call: str
) -> None:
    errors = _lock_reply_db(monkeypatch=monkeypatch, call=call)
    channel = _text_channel()
    thread = _RunThread()
    anchor = _Anchor(channel=channel, thread=thread)
    channel.send = AsyncMock(return_value=anchor)
    interaction = _ResearchInteraction(channel=channel)
    cog = _running_cog(monkeypatch=monkeypatch, client=SimpleNamespace())

    await cog.deep_research(as_interaction(fake=interaction), topic="topic")

    assert [edit.get("content") for edit in interaction.edits] == ["開研究串失敗了,等等再試一次"]
    assert anchor.deleted is True
    # Only the insert comes after the thread is opened, and a thread with no row is not a run.
    assert thread.deleted is (call == "insert_session")
    assert thread.writes == []
    assert cog._active_threads == set()
    assert not cog._tasks
    assert len(errors) == 1
    assert errors[0][1].get("_exc_info") is not None


@pytest.mark.parametrize("refused", [False, True], ids=["deleted", "delete_refused"])
async def test_a_marker_launch_says_so_and_withdraws_its_thread_when_reply_db_fails(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch, refused: bool
) -> None:
    _lock_reply_db(monkeypatch=monkeypatch, call="insert_session")
    warns = _recorded(monkeypatch=monkeypatch, level="warn")
    thread = _RunThread(error=make_forbidden(message="Missing Permissions") if refused else None)
    anchor = _Anchor(channel=_text_channel(), thread=thread)
    cog = _running_cog(monkeypatch=monkeypatch, client=SimpleNamespace())

    await cog.launch(message=as_message(fake=anchor), brief="b")

    assert [reply.get("content") for reply in anchor.replies] == ["開研究串失敗了,等等再試一次"]
    assert thread.deleted is not refused
    # The marker's anchor is the reply that promised the run, so it stays.
    assert anchor.deleted is False
    assert cog._active_threads == set()
    # A launch never asks for the Manage Threads a delete takes, so a refusal logs the ids alone.
    assert warns == (
        [
            (
                "research thread of a failed launch could not be deleted",
                {"thread_id": _THREAD_ID, "owner_id": _OWNER_ID},
            )
        ]
        if refused
        else []
    )


@pytest.mark.parametrize("stored_id", [True, False], ids=["resume_fails", "no_stored_id"])
async def test_a_resume_that_cannot_reattach_frees_the_owner_and_tells_only_them(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch, stored_id: bool
) -> None:
    async def _expired(**_kwargs: object) -> None:
        raise RuntimeError("interaction expired")

    monkeypatch.setattr(target=research_cog, name="resume_research_stream", value=_expired)
    thread = _RunThread()
    cog = _running_cog(monkeypatch=monkeypatch, client=SimpleNamespace(), thread=thread)

    await _resume_run(cog=cog, stored_id=stored_id)

    await _assert_owner_released(cog=cog, phase="failed")
    notice = thread.writes[-1]
    assert notice["content"] == "<@300> 重啟後沒辦法接回剛剛的研究,麻煩重新發起一次"
    _assert_pings_only_the_owner(write=notice)


async def test_a_resume_that_cannot_reattach_ends_its_own_status_as_failed(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _expired(**_kwargs: object) -> None:
        raise RuntimeError("interaction expired")

    monkeypatch.setattr(target=research_cog, name="resume_research_stream", value=_expired)
    thread = _RunThread()
    cog = _running_cog(monkeypatch=monkeypatch, client=SimpleNamespace(), thread=thread)
    errors = _recorded(monkeypatch=monkeypatch, level="error")

    await _resume_run(cog=cog)

    # The same failure a fresh run reports at `error`: the report it was waiting for is lost.
    assert [(message, fields["error_type"]) for message, fields in errors] == [
        ("research resume failed", "RuntimeError")
    ]
    assert [write["content"] for write in thread.writes] == [
        "-# Researching... (Antigravity)",
        "-# Research failed (Antigravity)",
        "<@300> 重啟後沒辦法接回剛剛的研究,麻煩重新發起一次",
    ]
    assert thread.sends == 2, "the failed line is the resume's own status edited, not a new post"


@pytest.mark.parametrize(
    ("settles", "stored_id", "ends_as"),
    [
        ("completed", True, "# Report\nbody"),
        (RuntimeError("interaction expired"), True, "-# Research failed (Antigravity)"),
        ("completed", False, "-# Research failed (Antigravity)"),
    ],
    ids=["delivers", "resume_fails", "no_stored_id"],
)
async def test_a_resume_ends_the_status_line_posted_before_the_restart(
    research_isolated_db: None,
    monkeypatch: pytest.MonkeyPatch,
    settles: str | Exception,
    stored_id: bool,
    ends_as: str,
) -> None:
    """The resume takes the pre-restart line over, so nothing is left claiming to research."""
    if isinstance(settles, Exception):

        async def _expired(**_kwargs: object) -> None:
            raise settles

        monkeypatch.setattr(target=research_cog, name="resume_research_stream", value=_expired)
    before_restart = FakeDiscordMessage(
        author=_ThreadBot.user, content="-# Researching... (Antigravity, 12m30s)\n-# Weighing"
    )
    # Neither a member's copy of the line nor a bot line of another kind is the status.
    thread = _RunThread(
        earlier=[
            FakeDiscordMessage(author=FakeUser(user_id=_OWNER_ID), content="-# Researching... ?"),
            FakeDiscordMessage(author=_ThreadBot.user, content="-# Research failed (Antigravity)"),
            before_restart,
        ]
    )
    cog = _running_cog(
        monkeypatch=monkeypatch, client=_settling_client(status="completed"), thread=thread
    )

    await _resume_run(cog=cog, stored_id=stored_id)

    assert str(before_restart.edits[-1]["content"]).startswith(ends_as)
    assert not any(str(write["content"]).startswith("-# Researching") for write in thread.writes)


@pytest.mark.parametrize("refused", [True, False], ids=["missing_access", "failing"])
async def test_a_resume_whose_history_read_fails_runs_on_a_status_line_of_its_own(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch, refused: bool
) -> None:
    """A 403 is the server's setting and logs the id alone; any other failure keeps its trace."""
    error = make_forbidden(message="Missing Access") if refused else make_server_error()
    warns = _recorded(monkeypatch=monkeypatch, level="warn")
    thread = _RunThread()

    def _failed(**_kwargs: object) -> NoReturn:
        raise error

    monkeypatch.setattr(target=thread, name="history", value=_failed)
    cog = _running_cog(
        monkeypatch=monkeypatch, client=_settling_client(status="completed"), thread=thread
    )

    await _resume_run(cog=cog)

    await _assert_owner_released(cog=cog, phase="done")
    assert thread.writes[0]["content"] == "-# Researching... (Antigravity)"
    assert warns == [
        ("research thread refused the history read", {"thread_id": _THREAD_ID})
        if refused
        else (
            "failed to read research thread history",
            {"thread_id": _THREAD_ID, "error_type": "HTTPException", "_exc_info": error},
        )
    ]


async def test_a_resume_whose_thread_is_gone_still_records_how_the_run_settled(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cog = _running_cog(monkeypatch=monkeypatch, client=_settling_client(status="cancelled"))

    await _resume_run(cog=cog)

    await _assert_owner_released(cog=cog, phase="cancelled")


@pytest.mark.parametrize(
    ("settles", "reason", "footer"),
    [(RuntimeError("quota"), "quota", "RuntimeError"), ("cancelled", "研究被取消了", None)],
    ids=["create_fails", "cancelled"],
)
async def test_a_failed_run_tells_only_its_owner_why(
    research_isolated_db: None,
    monkeypatch: pytest.MonkeyPatch,
    settles: str | Exception,
    reason: str,
    footer: str | None,
) -> None:
    client = (
        _failing_client(error=settles)
        if isinstance(settles, Exception)
        else _settling_client(status=settles)
    )
    thread = _RunThread()

    await _launch_run(cog=_running_cog(monkeypatch=monkeypatch, client=client), thread=thread)

    assert [write["content"] for write in thread.writes] == [
        "-# Researching... (Antigravity)",
        "<@300> ⚠️",
        "-# Research failed (Antigravity)",
    ]
    notice = thread.writes[1]
    _assert_pings_only_the_owner(write=notice)
    embed = cast("Embed", notice["embed"])
    assert embed.description == f"```\n{reason}\n```"
    assert embed.footer.text == footer


async def test_a_delivered_report_pings_only_its_owner_over_the_runs_own_usage(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cog = _running_cog(monkeypatch=monkeypatch, client=_settling_client(status="completed"))
    thread = _RunThread()

    await _launch_run(cog=cog, thread=thread)

    # 1,234 in at $1e-6 plus 567 out at $2e-6: the counts the interaction reported, priced.
    report = thread.writes[-1]
    agent_name = cog.runtime_models.antigravity_model.name
    assert report["content"] == (
        f"# Report\nbody\n\n<@300>\n\n-# {agent_name} · ⬆ 1,234 ⬇ 567 · $0.00236800"
    )
    _assert_pings_only_the_owner(write=report)


async def test_a_resumed_report_pings_only_its_owner_over_the_runs_own_usage(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    thread = _RunThread()
    cog = _running_cog(
        monkeypatch=monkeypatch, client=_settling_client(status="completed"), thread=thread
    )

    # The row's own agent, not the catalog's: a restart after a repoint still names and prices
    # the agent the run was launched on, which the patched table leaves unpriced.
    await _resume_run(cog=cog, agent="antigravity-launched-agent")

    await _assert_owner_released(cog=cog, phase="done")
    report = thread.writes[-1]
    assert report["content"] == (
        "# Report\nbody\n\n<@300>\n\n-# antigravity-launched-agent · ⬆ 1,234 ⬇ 567 · $0.00000000"
    )
    _assert_pings_only_the_owner(write=report)


# What each write of a failing run logs when Discord refuses it, and when it fails otherwise.
_FAILED_RUN_WRITE_LOGS = {
    "status": ("research thread refused a message", "failed to send research thread message"),
    "notice": (
        "research thread refused the failure notice",
        "failed to post research failure notice",
    ),
    "status edit": (
        "research thread refused the status edit",
        "failed to edit research status message",
    ),
    "terminal status": (
        "research thread refused the terminal status",
        "failed to post terminal research status",
    ),
}


@pytest.mark.parametrize("refused", [True, False], ids=["refused", "failing"])
@pytest.mark.parametrize(
    ("status_posts", "writes"),
    [
        (True, ("notice", "status edit", "terminal status")),
        (False, ("status", "notice", "terminal status")),
    ],
    ids=["status_posted", "status_lost"],
)
async def test_a_failed_run_on_a_broken_thread_logs_each_write_and_still_ends(
    research_isolated_db: None,
    monkeypatch: pytest.MonkeyPatch,
    refused: bool,
    status_posts: bool,
    writes: tuple[str, ...],
) -> None:
    """A refusal is the server's setting and logs the id alone; any other failure keeps its trace."""
    error = make_forbidden(message="Missing Access") if refused else make_server_error()
    warns = _recorded(monkeypatch=monkeypatch, level="warn")
    errors = _recorded(monkeypatch=monkeypatch, level="error")
    cog = _running_cog(
        monkeypatch=monkeypatch, client=_failing_client(error=RuntimeError("quota"))
    )

    await _launch_run(cog=cog, thread=_RunThread(error=error, status_posts=status_posts))

    trace = {} if refused else {"error_type": "HTTPException", "_exc_info": error}
    assert warns == [
        (_FAILED_RUN_WRITE_LOGS[write][0 if refused else 1], {"thread_id": _THREAD_ID, **trace})
        for write in writes
    ]
    assert [message for message, _ in errors] == ["research failed"]
    await _assert_owner_released(cog=cog, phase="failed")


async def test_a_refused_report_logs_each_write_without_a_traceback_and_still_ends(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    warns = _recorded(monkeypatch=monkeypatch, level="warn")
    errors = _recorded(monkeypatch=monkeypatch, level="error")
    cog = _running_cog(monkeypatch=monkeypatch, client=_settling_client(status="completed"))

    await _launch_run(cog=cog, thread=_RunThread(error=make_forbidden(message="Missing Access")))

    where = {"thread_id": _THREAD_ID, "chunk_index": 0}
    assert warns == [
        ("research thread refused the report edit", where),
        ("research thread refused a report message", {**where, "is_last": True}),
    ]
    assert errors == []
    await _assert_owner_released(cog=cog, phase="done")


async def test_a_failing_report_logs_each_write_with_its_traceback_and_still_ends(
    research_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = make_server_error()
    warns = _recorded(monkeypatch=monkeypatch, level="warn")
    errors = _recorded(monkeypatch=monkeypatch, level="error")
    cog = _running_cog(monkeypatch=monkeypatch, client=_settling_client(status="completed"))

    await _launch_run(cog=cog, thread=_RunThread(error=error))

    where = {"thread_id": _THREAD_ID, "chunk_index": 0}
    trace = {"error_type": "HTTPException", "_exc_info": error}
    assert warns == [("failed to edit research status into report", {**where, **trace})]
    # The last message carries the file, ping and footer, so losing it is an error.
    assert errors == [
        (
            "failed to post research report message",
            {**where, "is_last": True, "has_files": True, **trace},
        )
    ]
    await _assert_owner_released(cog=cog, phase="done")
