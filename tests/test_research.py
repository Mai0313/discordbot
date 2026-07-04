"""Tests for the deep-research feature: marker extraction, delivery, agent helpers, and store."""

from types import SimpleNamespace
import base64
from pathlib import Path

from nextcord import AllowedMentions

from discordbot.cogs import research as research_cog
from discordbot.typings.llm import LLMConfig
from discordbot.cogs._research import agent
from discordbot.cogs._research import database as rdb
from discordbot.utils.media_delivery import (
    MediaHostingConfig,
    MediaHostingService,
    MediaDeliveryPlanner,
)
from discordbot.cogs._gen_reply.markers import extract_inline_markers, scrub_markers_for_preview
from discordbot.cogs._research.delivery import split_report, deliver_report
from discordbot.cogs._research.streaming import ResearchProgressStreamer


def _disabled_delivery() -> MediaDeliveryPlanner:
    """A planner whose host is off, so report files attach natively exactly as before hosting."""
    return MediaDeliveryPlanner(
        media_hosting=MediaHostingService(config=MediaHostingConfig(MEDIA_HOSTING_ENABLED=False))
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


# ----- agent helpers ------------------------------------------------------------------------


def test_deep_research_agent_config_shape() -> None:
    plan_config = agent._deep_research_agent_config(collaborative_planning=True)
    assert plan_config["type"] == "deep-research"
    assert plan_config["collaborative_planning"] is True
    run_config = agent._deep_research_agent_config(collaborative_planning=False)
    assert run_config["collaborative_planning"] is False


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
        client=client,
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
        client=client,
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
        client=client,
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
        client=client,
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
            client=client,
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
    streamer = ResearchProgressStreamer(status=None, label="Deep Research")
    result = await agent.resume_research_stream(
        client=client, interaction_id="int_9", streamer=streamer
    )
    # Resume re-attaches via get(stream=True) and never calls create.
    assert client.aio.interactions.create_kwargs == {}
    assert result.ok is True


async def test_stream_plan_passes_research_tools_and_returns_plan() -> None:
    client = _fake_client(
        streams=[
            _FakeStream([
                _created_event(interaction_id="plan_x", event_id="e1"),
                _thought_event("outlining", event_id="e2"),
                _completed_event(event_id="e3"),
            ])
        ],
        terminal=SimpleNamespace(id="plan_x", status="completed", output_text="a plan", steps=[]),
    )
    streamer = ResearchProgressStreamer(status=None, label="Deep Research", action="Planning")
    plan = await agent.stream_plan(
        client=client,
        agent="deep-research-preview-04-2026",
        brief="b",
        system_instruction="sys",
        streamer=streamer,
    )
    # Planning must pass the restricted search/url tool set so the agent default (which can include
    # code execution) cannot leak raw tool-call text into the plan; it now streams reasoning too.
    assert client.aio.interactions.create_kwargs["tools"] is agent.RESEARCH_TOOLS
    assert client.aio.interactions.create_kwargs["stream"] is True
    assert streamer.reasoning == "outlining"
    assert plan.status == "completed"
    assert plan.plan_text == "a plan"
    assert plan.interaction_id == "plan_x"


async def test_stream_refine_passes_research_tools_and_returns_plan() -> None:
    client = _fake_client(
        streams=[
            _FakeStream([
                _created_event(interaction_id="plan_x", event_id="e1"),
                _completed_event(),
            ])
        ],
        terminal=SimpleNamespace(id="plan_x", status="completed", output_text="a plan", steps=[]),
    )
    streamer = ResearchProgressStreamer(status=None, label="Deep Research", action="Re-planning")
    plan = await agent.stream_refine(
        client=client,
        agent="deep-research-preview-04-2026",
        previous_interaction_id="plan_v1",
        feedback="tighten the scope",
        system_instruction="sys",
        streamer=streamer,
    )
    assert client.aio.interactions.create_kwargs["tools"] is agent.RESEARCH_TOOLS
    assert client.aio.interactions.create_kwargs["previous_interaction_id"] == "plan_v1"
    assert plan.status == "completed"


def test_is_terminal_event_classifies_statuses() -> None:
    assert agent._is_terminal_event(event=_completed_event()) is True
    assert (
        agent._is_terminal_event(
            event=SimpleNamespace(event_type="error", error=SimpleNamespace(message="boom"))
        )
        is True
    )
    running = SimpleNamespace(event_type="interaction.status_update", status="in_progress")
    assert agent._is_terminal_event(event=running) is False
    failed = SimpleNamespace(event_type="interaction.status_update", status="budget_exceeded")
    assert agent._is_terminal_event(event=failed) is True
    assert agent._is_terminal_event(event=_thought_event("x")) is False


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
    interaction = SimpleNamespace(id="int_x", status="failed")
    result = agent._to_result(interaction=interaction)
    assert result.ok is False
    assert result.report_text == ""
    assert result.image_bytes is None
    assert result.input_tokens == 0


def test_latest_thought_returns_last_summary() -> None:
    interaction = SimpleNamespace(
        steps=[
            SimpleNamespace(content=[SimpleNamespace(type="thought_summary", text="first")]),
            SimpleNamespace(content=[SimpleNamespace(type="thought_summary", text="second")]),
        ]
    )
    assert agent._latest_thought(interaction=interaction) == "second"
    assert agent._latest_thought(interaction=SimpleNamespace()) is None


def test_latest_thought_reads_thought_step_summary() -> None:
    interaction = SimpleNamespace(
        steps=[
            SimpleNamespace(type="thought", summary=[SimpleNamespace(text="planning the search")]),
            SimpleNamespace(
                type="model_output", content=[SimpleNamespace(type="text", text="report")]
            ),
        ]
    )
    assert agent._latest_thought(interaction=interaction) == "planning the search"


# ----- progress streamer --------------------------------------------------------------------


def test_streamer_feed_accumulates_only_thought_summaries() -> None:
    streamer = ResearchProgressStreamer(status=None, label="Antigravity")
    streamer._feed(event=_thought_event("searching..."))
    text_delta = SimpleNamespace(
        event_type="step.delta", event_id="x", delta=SimpleNamespace(type="text", text="body")
    )
    streamer._feed(event=text_delta)  # report text is delivered separately, not shown as reasoning
    streamer._feed(event=_created_event())  # non-delta events are ignored
    assert streamer.reasoning == "searching..."


def test_streamer_render_preview_windows_and_escapes_mentions() -> None:
    streamer = ResearchProgressStreamer(status=None, label="Deep Research", action="Planning")
    streamer.reasoning = "first line\n@everyone please\nlast line"
    preview = streamer._render_preview()
    assert preview.startswith("-# Planning... (Deep Research,")
    assert "@everyone" not in preview  # agent text is escaped so the thinking can never ping
    assert "last line" in preview


async def test_streamer_write_snapshot_edits_and_skips_unchanged() -> None:
    status = _FakeStatusMessage()
    streamer = ResearchProgressStreamer(status=status, label="Antigravity", reasoning="thinking")
    await streamer._write_preview_snapshot()
    assert len(status.edits) == 1
    assert status.edits[0]["allowed_mentions"].everyone is False
    # A second write of the same rendered snapshot is a no-op, so the editor never spams edits.
    streamer._displayed = streamer._render_preview()
    await streamer._write_preview_snapshot()
    assert len(status.edits) == 1


async def test_streamer_stream_accumulates_and_stops_editor_cleanly() -> None:
    status = _FakeStatusMessage()
    streamer = ResearchProgressStreamer(
        status=status, label="Antigravity", preview_interval_seconds=0.01
    )
    await streamer.stream(events=_FakeStream([_thought_event("aaa"), _thought_event("bbb")]))
    assert streamer.reasoning == "aaabbb"
    assert streamer._editor_task is None  # the cadence editor is always stopped in finally


# ----- research module helpers --------------------------------------------------------------


def test_fallback_thread_name_uses_first_line() -> None:
    name = research_cog._fallback_thread_name(brief="研究 TPU 的歷史與競爭格局\n更多細節")
    assert name.startswith("研究 TPU")
    assert "\n" not in name
    assert research_cog._fallback_thread_name(brief="   ") == "深度研究"


def test_tier_label_maps_agent_strings() -> None:
    assert research_cog._tier_label(agent="antigravity-preview-05-2026") == "Antigravity"
    assert research_cog._tier_label(agent="deep-research-preview-04-2026") == "Deep Research"
    assert (
        research_cog._tier_label(agent="deep-research-max-preview-04-2026") == "Deep Research Max"
    )


def test_terminal_phase_mapping() -> None:
    assert research_cog._terminal_phase(status="completed") == "done"
    assert research_cog._terminal_phase(status="cancelled") == "cancelled"
    assert research_cog._terminal_phase(status="budget_exceeded") == "failed"


def test_owner_id_from_mention_parses_digits() -> None:
    assert research_cog._owner_id_from_mention(mention="<@123456789>") == 123456789
    assert research_cog._owner_id_from_mention(mention="<@!42>") == 42
    assert research_cog._owner_id_from_mention(mention="nobody") == 0


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
    assert [obj.id for obj in mentions.users] == [42]


def test_failure_text_distinguishes_budget() -> None:
    assert "成本上限" in research_cog._failure_text(status="budget_exceeded")
    assert "取消" in research_cog._failure_text(status="cancelled")
    assert research_cog._failure_text(status="failed")


# ----- persistence (reply.db) ---------------------------------------------------------------


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
    session = await rdb.get_session(thread_id=1)
    assert session is not None
    assert session.owner_id == 99
    assert session.brief == "研究 X"
    assert session.phase == "researching"
    assert session.interaction_id is None
    assert await rdb.get_session(thread_id=999) is None


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
        agent="deep-research-preview-04-2026",
        phase="planning",
    )
    session = await rdb.get_session(thread_id=2)
    assert session is not None
    assert session.interaction_id == "int_abc"
    assert session.agent == "deep-research-preview-04-2026"
    assert session.phase == "planning"
    await rdb.set_phase(thread_id=2, phase="done")
    refreshed = await rdb.get_session(thread_id=2)
    assert refreshed is not None
    assert refreshed.phase == "done"


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
    for thread_id, phase in ((20, "researching"), (21, "planning"), (22, "done")):
        await rdb.upsert_session(
            thread_id=thread_id,
            owner_id=thread_id,
            channel_id=1,
            guild_id=1,
            source_message_id=1,
            agent="deep-research-preview-04-2026",
            interaction_id="int_x",
            brief="b",
            phase=phase,  # type: ignore[arg-type]  # the test deliberately exercises each phase
        )
    resumable = await rdb.list_resumable()
    assert {session.thread_id for session in resumable} == {20}


def test_cast_phase_defaults_unknown_to_failed() -> None:
    assert rdb.cast_phase(value="researching") == "researching"
    assert rdb.cast_phase(value="bogus") == "failed"


async def test_claim_research_is_idempotent_and_clears_stale_id(
    research_isolated_db: None,
) -> None:
    await rdb.upsert_session(
        thread_id=30,
        owner_id=1,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="deep-research-preview-04-2026",
        interaction_id="plan_1",
        brief="b",
        phase="planning",
    )
    assert await rdb.claim_research(thread_id=30, plan_interaction_id="plan_1") is True
    session = await rdb.get_session(thread_id=30)
    assert session is not None
    assert session.phase == "researching"
    assert session.interaction_id is None
    # A second (double-click) claim loses: the row is no longer in `planning`.
    assert await rdb.claim_research(thread_id=30, plan_interaction_id="plan_1") is False


async def test_claim_research_rejects_stale_plan_interaction(research_isolated_db: None) -> None:
    await rdb.upsert_session(
        thread_id=31,
        owner_id=1,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="deep-research-preview-04-2026",
        interaction_id="plan_v2",
        brief="b",
        phase="planning",
    )
    # A stale approval view (its plan interaction was superseded by a refine) loses the claim, so
    # research never launches from the old plan; the row stays `planning` for the fresh view.
    assert await rdb.claim_research(thread_id=31, plan_interaction_id="plan_v1") is False
    session = await rdb.get_session(thread_id=31)
    assert session is not None
    assert session.phase == "planning"
    assert session.interaction_id == "plan_v2"
    # The current plan's view wins the claim.
    assert await rdb.claim_research(thread_id=31, plan_interaction_id="plan_v2") is True


async def test_claim_planning_is_idempotent(research_isolated_db: None) -> None:
    await rdb.upsert_session(
        thread_id=50,
        owner_id=1,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="antigravity-preview-05-2026",
        interaction_id="int_done",
        brief="b",
        phase="done",
    )
    assert await rdb.claim_planning(thread_id=50) is True
    claimed = await rdb.get_session(thread_id=50)
    assert claimed is not None
    assert claimed.phase == "planning"
    # A second escalation click finds the row already planning and loses.
    assert await rdb.claim_planning(thread_id=50) is False


async def test_cancel_stale_plan_matches_current_interaction(research_isolated_db: None) -> None:
    await rdb.upsert_session(
        thread_id=51,
        owner_id=1,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="deep-research-preview-04-2026",
        interaction_id="plan_v2",
        brief="b",
        phase="planning",
    )
    # A superseded view (an older plan interaction id) is a no-op, leaving the fresh plan intact.
    assert await rdb.cancel_stale_plan(thread_id=51, plan_interaction_id="plan_v1") is False
    still_planning = await rdb.get_session(thread_id=51)
    assert still_planning is not None
    assert still_planning.phase == "planning"
    # The current plan's expired view cancels it and frees the owner's slot.
    assert await rdb.cancel_stale_plan(thread_id=51, plan_interaction_id="plan_v2") is True
    cancelled = await rdb.get_session(thread_id=51)
    assert cancelled is not None
    assert cancelled.phase == "cancelled"


async def test_clear_stale_planning_cancels_only_planning(research_isolated_db: None) -> None:
    for thread_id, phase in ((40, "planning"), (41, "researching"), (42, "planning")):
        await rdb.upsert_session(
            thread_id=thread_id,
            owner_id=thread_id,
            channel_id=1,
            guild_id=1,
            source_message_id=1,
            agent="deep-research-preview-04-2026",
            interaction_id="int_x",
            brief="b",
            phase=phase,  # type: ignore[arg-type]  # the test deliberately exercises each phase
        )
    cleared = await rdb.clear_stale_planning()
    assert {session.thread_id for session in cleared} == {40, 42}
    cancelled = await rdb.get_session(thread_id=40)
    assert cancelled is not None
    assert cancelled.phase == "cancelled"
    researching = await rdb.get_session(thread_id=41)
    assert researching is not None
    assert researching.phase == "researching"
    # The owner whose plan was cleared is no longer blocked from launching new research.
    assert await rdb.active_thread_for_owner(owner_id=40) is None


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
        thread=thread,  # type: ignore[arg-type]  # minimal Thread double for the delivery path
        status=status,  # type: ignore[arg-type]  # minimal status-message double
        owner_mention="<@1>",
        result=_completed_result(report_text="X" * 1990),
        footer=footer,
        view=None,
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
        thread=thread,  # type: ignore[arg-type]  # minimal Thread double for the delivery path
        status=status,  # type: ignore[arg-type]  # minimal status-message double
        owner_mention="<@1>",
        result=_completed_result(report_text="# Report\nbody"),
        footer="-# footer",
        view=None,
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
            config=MediaHostingConfig(
                MEDIA_HOSTING_ENABLED=True,
                MEDIA_HOSTING_BASE_URL="https://media.test",
                MEDIA_HOSTING_SERVE_DIR=str(tmp_path),
            )
        )
    )
    await deliver_report(
        thread=thread,  # type: ignore[arg-type]  # minimal Thread double for the delivery path
        status=status,  # type: ignore[arg-type]  # minimal status-message double
        owner_mention="<@1>",
        result=_completed_result(report_text="# Report\nbody"),
        footer="-# footer",
        view=None,
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
        thread=thread,  # type: ignore[arg-type]  # minimal Thread double for the delivery path
        status=status,  # type: ignore[arg-type]  # minimal status-message double
        owner_mention="<@1>",
        result=_completed_result(report_text="R" * 60, image_bytes=b"x" * 60),
        footer="-# footer",
        view=None,
        allowed_mentions=AllowedMentions(everyone=False, roles=False, users=[]),
        media_delivery=_disabled_delivery(),
    )
    edit = status.edits[0]
    files = edit["files"]
    assert isinstance(files, list)
    assert len(files) == 2  # research.md AND research.png both attached, neither dropped
    assert "https://" not in str(edit["content"])  # nothing was hosted
