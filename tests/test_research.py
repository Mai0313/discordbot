"""Tests for the deep-research feature: marker extraction, delivery, agent helpers, and store."""

from types import SimpleNamespace
import base64
from typing import TYPE_CHECKING, cast
import asyncio
from pathlib import Path

import pytest
from nextcord import AllowedMentions

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.research import cog as research_cog
from discordbot.cogs.research import agent
from discordbot.cogs.research import database as rdb
from discordbot.utils.media_delivery import MediaHostingService, MediaDeliveryPlanner
from discordbot.cogs.gen_reply.markers import extract_inline_markers, scrub_markers_for_preview
from discordbot.cogs.research.delivery import (
    split_report,
    deliver_report,
    split_report_by_sections,
)
from discordbot.cogs.research.streaming import DISCORD_MESSAGE_LIMIT, ResearchProgressStreamer

from tests.helpers.casting import (
    as_client,
    as_message,
    make_media_hosting_config,
    as_interaction_event_stream,
)

if TYPE_CHECKING:
    from nextcord import Thread
    from google.genai.interactions import InteractionSSEEvent

    from discordbot.cogs.research.database import ResearchPhase


def _disabled_delivery() -> MediaDeliveryPlanner:
    """A planner whose host is off, so report files attach natively exactly as before hosting."""
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


def _terminal_interaction() -> SimpleNamespace:
    return SimpleNamespace(
        id="int_9",
        status="completed",
        output_text="# Report\nbody",
        usage=SimpleNamespace(total_input_tokens=10, total_output_tokens=5),
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
        agent="antigravity-preview-05-2026",
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
    # The only agent runs with no `agent_config`: that plumbing carried the removed Deep Research
    # tiers' `collaborative_planning`, and nothing may put it back on the one surviving call.
    assert "agent_config" not in kwargs
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
    monkeypatch.setattr(agent, "MAX_STREAM_RECONNECTS", 0)
    client = _fake_client(
        streams=[
            _FakeStream([_created_event(event_id="e1")], raise_after=1),
            _FakeStream([], raise_after=0),
        ],
        terminal=_terminal_interaction(),
    )
    streamer = ResearchProgressStreamer(status=None, label="Antigravity")

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
    assert result.ok is True


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
    assert result.interaction_id == "int_123"
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


async def test_streamer_write_snapshot_edits_and_skips_unchanged() -> None:
    status = _FakeStatusMessage()
    streamer = ResearchProgressStreamer(status=status, label="Antigravity", reasoning="thinking")
    await streamer._write_preview_snapshot()
    assert len(status.edits) == 1
    assert cast("AllowedMentions", status.edits[0]["allowed_mentions"]).everyone is False
    # A second write of the same rendered snapshot is a no-op, so the editor never spams edits.
    streamer._displayed = streamer._render_preview()
    await streamer._write_preview_snapshot()
    assert len(status.edits) == 1


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
    mentions = research_cog._owner_allowed_mentions(owner_id=42)
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
    await rdb.upsert_session(
        thread_id=1,
        owner_id=99,
        channel_id=7,
        guild_id=5,
        source_message_id=3,
        agent="antigravity-preview-05-2026",
        interaction_id=None,
        brief="研究 X",
        phase="researching",
    )
    session = await _only_resumable(thread_id=1)
    assert session is not None
    assert session.owner_id == 99
    assert session.brief == "研究 X"
    assert session.phase == "researching"
    assert session.interaction_id is None
    assert await _only_resumable(thread_id=999) is None


async def test_set_interaction_and_phase(research_isolated_db: None) -> None:
    await rdb.upsert_session(
        thread_id=2,
        owner_id=1,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="antigravity-preview-05-2026",
        interaction_id=None,
        brief="b",
        phase="researching",
    )
    await rdb.set_interaction(
        thread_id=2,
        interaction_id="int_abc",
        agent="antigravity-preview-05-2026",
        phase="researching",
    )
    session = await _only_resumable(thread_id=2)
    assert session is not None
    assert session.interaction_id == "int_abc"
    assert session.agent == "antigravity-preview-05-2026"
    assert session.phase == "researching"
    await rdb.set_phase(thread_id=2, phase="done")
    assert await _only_resumable(thread_id=2) is None
    assert await rdb.active_thread_for_owner(owner_id=1) is None


async def test_active_thread_for_owner_excludes_terminal(research_isolated_db: None) -> None:
    await rdb.upsert_session(
        thread_id=10,
        owner_id=500,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="antigravity-preview-05-2026",
        interaction_id=None,
        brief="b",
        phase="researching",
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
        await rdb.upsert_session(
            thread_id=thread_id,
            owner_id=thread_id,
            channel_id=1,
            guild_id=1,
            source_message_id=1,
            agent="antigravity-preview-05-2026",
            interaction_id="int_x",
            brief="b",
            phase=phase,
        )
    resumable = await rdb.list_resumable()
    assert {session.thread_id for session in resumable} == {20}


def test_cast_phase_defaults_unknown_to_failed() -> None:
    assert rdb.cast_phase(value="researching") == "researching"
    assert rdb.cast_phase(value="bogus") == "failed"
    # A row the removed escalation tiers left in `planning` is no longer a phase this store knows.
    assert rdb.cast_phase(value="planning") == "failed"


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
        interaction_id="int_1",
        status="completed",
        report_text=report_text,
        image_bytes=image_bytes,
    )


async def test_delivery_keeps_footer_message_under_the_limit() -> None:
    status = _FakeStatusMessage()
    thread = _FakeThread()
    footer = "-# antigravity-preview-05-2026 · ⬆ 0 ⬇ 0 · $0.00000000"
    mentions = AllowedMentions(everyone=False, roles=False, users=[])
    # A report chunk that sits just under the 2000-char message cap; appending the footer inline
    # would overflow, so it must ride its own trailing message.
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_mention="<@1>",
        result=_completed_result(report_text="X" * 1990),
        footer=footer,
        allowed_mentions=mentions,
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
    assert footer_send["allowed_mentions"] is mentions
    assert status.edits[0]["allowed_mentions"] is mentions


async def test_delivery_inlines_footer_for_short_reports() -> None:
    status = _FakeStatusMessage()
    thread = _FakeThread()
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_mention="<@1>",
        result=_completed_result(report_text="# Report\nbody"),
        footer="-# footer",
        allowed_mentions=AllowedMentions(everyone=False, roles=False, users=[]),
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
        owner_mention="<@1>",
        result=_completed_result(report_text="# Report\nbody"),
        footer="-# footer",
        allowed_mentions=AllowedMentions(everyone=False, roles=False, users=[]),
        media_delivery=planner,
    )
    # The report .md was hosted (no native attachment); its URL rides the message content.
    edit = status.edits[0]
    assert not edit.get("files")
    content = str(edit["content"])
    assert any(line.startswith("https://media.test/") for line in content.splitlines())


async def test_delivery_attaches_both_files_when_each_fits_but_combined_over() -> None:
    """Host-off contract: md + png that each fit but jointly exceed the limit BOTH attach natively.

    Pre-fold-in `_final_files` attached each file independently with no combined-body check. Routing
    both through one `plan()` call would have fired the planner's combined-peel and dropped the
    larger (the report), so delivery decides each attachment on its own to keep host-off parity.
    """
    status = _FakeStatusMessage()
    thread = _FakeThread()
    thread.guild = SimpleNamespace(filesize_limit=100)  # each file fits, md + png together do not
    await deliver_report(
        thread=cast("Thread", thread),  # minimal Thread double for the delivery path
        status=as_message(fake=status),  # minimal status-message double
        owner_mention="<@1>",
        result=_completed_result(report_text="R" * 60, image_bytes=b"x" * 60),
        footer="-# footer",
        allowed_mentions=AllowedMentions(everyone=False, roles=False, users=[]),
        media_delivery=_disabled_delivery(),
    )
    edit = status.edits[0]
    files = edit["files"]
    assert isinstance(files, list)
    assert len(files) == 2  # research.md AND research.png both attached, neither dropped
    assert "https://" not in str(edit["content"])  # nothing was hosted


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


async def _seed_researching(*, thread_id: int, owner_id: int) -> None:
    """Seeds one in-flight row, as a launch that never reached a terminal phase left it."""
    await rdb.upsert_session(
        thread_id=thread_id,
        owner_id=owner_id,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="antigravity-preview-05-2026",
        interaction_id=f"int_{thread_id}",
        brief="b",
        phase="researching",
    )


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
