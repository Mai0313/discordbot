"""Pins the deep-research feature end to end: marker, chunking, streaming agent, store, delivery.

A research run costs minutes of wall clock and real money on a managed Gemini agent, so almost
none of this is reachable from a live call. Every guard here drives the production code over
fabricated SSE events and a fake `client.aio.interactions`, which is the only way to exercise the
paths that appear solely when something goes wrong.

The streaming driver carries most of the file. `agent.py` treats a stream that ends or drops
without a terminal event as a re-attach (`get(stream=True, last_event_id=...)`) rather than as
completion, degrades to the poll once the re-attaches stop making progress, and re-raises only
when the create itself died before `interaction.created` left an id to resume from. Those are
three different outcomes behind one identical-looking symptom, so each has its own test. Beside
them sit the id being persisted on the first event rather than after the long wait,
`RESEARCH_TOOLS` riding every create, and no `agent_config` going out at all, since it carried
the removed escalation tiers' `collaborative_planning` that #447 dropped. The result is always
read back from the terminal non-stream `get(id)`, because `interaction.completed` carries an
empty payload on purpose.

Delivery is the other half where a bug is expensive. The report is agent-written text of
arbitrary length that may quote arbitrary mentions, so the chunker has to cut on the report's own
`---` / paragraph / line structure without losing a character, the owner-only mention policy has
to reach every message, and a near-limit final chunk must not swallow the footer, the owner ping
or the `research.md` attachment. One test pins that a file past the guild's upload ceiling becomes
a hosted URL instead of disappearing, and its host-off twin pins that the two attachments are
still decided independently, so a merely combined overflow never peels the report away.

The rest is smaller but load-bearing. The `<deep-research>` marker is the QA route's only inline
trigger, so a brief leaking into visible text would post the prompt instead of running it. The
reply.db row is both the restart-resume set and the one-per-owner concurrency slot, so a row
parked in a phase the store no longer knows (`planning`, left behind by those same removed tiers)
must hold neither. The store tests take `research_isolated_db`; nothing here needs credentials.
"""

from types import SimpleNamespace
import base64
from typing import TYPE_CHECKING, cast
from pathlib import Path

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
from discordbot.cogs.research.streaming import ResearchProgressStreamer

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
    """Builds a delivery planner whose media host is switched off.

    Returns:
        A planner that never hosts, so a report file either attaches natively or is dropped,
        which is byte-for-byte the pre-hosting delivery path.
    """
    return MediaDeliveryPlanner(
        media_hosting=MediaHostingService(config=make_media_hosting_config(enabled=False))
    )


# ----- marker extraction --------------------------------------------------------------------


def test_deep_research_block_is_pulled_and_brief_captured() -> None:
    """A complete `<deep-research>` block yields the brief and vanishes from the visible text."""
    markers = extract_inline_markers(
        text="好喔幫你查 <deep-research>研究 TPU 的競爭格局</deep-research> 等等貼到 thread"
    )
    assert markers.research_brief == "研究 TPU 的競爭格局"
    assert "TPU" not in markers.cleaned_text
    assert "thread" in markers.cleaned_text


def test_unclosed_trailing_deep_research_is_still_pulled() -> None:
    """An unclosed trailing marker still yields its brief instead of leaking it into chat."""
    markers = extract_inline_markers(text="開查囉 <deep-research>研究量子計算最新進展")
    assert markers.research_brief == "研究量子計算最新進展"
    assert "量子" not in markers.cleaned_text


def test_deep_research_coexists_with_voice() -> None:
    """One reply carrying both markers keeps the spoken segment visible and pulls the brief."""
    markers = extract_inline_markers(
        text="<generate-voice>馬上幫你查</generate-voice> <deep-research>研究 X</deep-research>"
    )
    assert markers.voice_requested
    assert "馬上幫你查" in markers.cleaned_text
    assert markers.research_brief == "研究 X"
    assert "X" not in markers.cleaned_text


def test_scrub_hides_deep_research_mid_stream() -> None:
    """The live preview hides a half-streamed brief, so it never flickers into the message."""
    assert "TPU" not in scrub_markers_for_preview(text="好喔 <deep-research>研究 TPU")


def test_no_marker_leaves_text_and_brief_untouched() -> None:
    """A marker-free reply comes back byte-for-byte and asks for no research."""
    markers = extract_inline_markers(text="這只是一般回覆,沒有任何 marker")
    assert markers.research_brief is None
    assert markers.cleaned_text == "這只是一般回覆,沒有任何 marker"


# ----- delivery splitting -------------------------------------------------------------------


def test_split_report_keeps_short_text_as_one_chunk() -> None:
    """Text already under the message cap is one chunk, unsplit."""
    assert split_report(text="short report") == ["short report"]


def test_split_report_prefers_paragraph_boundaries() -> None:
    """An over-limit report cuts on its blank-line paragraph boundary, not mid-paragraph."""
    para_a = "A" * 1200
    para_b = "B" * 1200
    chunks = split_report(text=f"{para_a}\n\n{para_b}")
    assert len(chunks) == 2
    assert chunks[0] == para_a
    assert chunks[1] == para_b


def test_split_report_hard_cuts_an_oversized_line() -> None:
    """A single line longer than the limit is hard-cut at the limit with nothing dropped."""
    chunks = split_report(text="C" * 5000, limit=2000)
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert "".join(chunks) == "C" * 5000


def test_split_report_by_sections_splits_on_thematic_breaks() -> None:
    """A `---` thematic break starts its own message, and the separator line itself is dropped."""
    chunks = split_report_by_sections(text="## A\n\nAlpha body\n\n---\n\n## B\n\nBeta body")
    assert chunks == ["## A\n\nAlpha body", "## B\n\nBeta body"]


def test_split_report_by_sections_subsplits_oversized_section() -> None:
    """A section still over the limit is sub-packed, leaving the sections beside it whole."""
    chunks = split_report_by_sections(text="intro\n\n---\n\n" + "X" * 2500, limit=2000)
    assert chunks[0] == "intro"
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert "".join(chunks[1:]) == "X" * 2500
    assert len(chunks) == 3


def test_split_report_by_sections_falls_back_to_paragraph_packing() -> None:
    """A report with no thematic break packs byte-for-byte like plain `split_report`."""
    text = f"{'A' * 1200}\n\n{'B' * 1200}"
    assert split_report_by_sections(text=text) == split_report(text=text)


def test_split_report_by_sections_ignores_break_inside_code_fence() -> None:
    """A `---` line inside a fenced code block is content, never a section break."""
    chunks = split_report_by_sections(text="before\n\n```\n---\n```\n\nafter")
    assert len(chunks) == 1
    assert "---" in chunks[0]


def test_split_report_by_sections_keeps_table_delimiter_row() -> None:
    """A markdown table's `| --- |` delimiter row never cuts the table in half."""
    chunks = split_report_by_sections(text="| Col | Val |\n| --- | --- |\n| a | 1 |")
    assert len(chunks) == 1


def test_split_report_by_sections_keeps_setext_heading() -> None:
    """A setext heading underline, having no blank line above it, is not a section break."""
    chunks = split_report_by_sections(text="Heading\n---\n\nbody")
    assert len(chunks) == 1


def test_split_report_by_sections_drops_empty_sections() -> None:
    """A leading or trailing break yields no empty message."""
    chunks = split_report_by_sections(text="---\n\nonly body\n\n---")
    assert chunks == ["only body"]


# ----- agent helpers ------------------------------------------------------------------------


class _FakeStream:
    """Async iterator over scripted SSE events; can raise after a prefix to simulate a drop."""

    def __init__(self, events: list[object], *, raise_after: int | None = None) -> None:
        """Scripts the events to yield, and how many of them may pass before it raises."""
        self._events = list(events)
        self._raise_after = raise_after
        self._yielded = 0

    def __aiter__(self) -> "_FakeStream":
        """Iterates itself, like the SDK's own stream object.

        Returns:
            This same stream.
        """
        return self

    async def __anext__(self) -> object:
        """Hands back the next scripted event, or drops the stream once `raise_after` is reached.

        Returns:
            The next scripted event.

        Raises:
            RuntimeError: `raise_after` events have been yielded, standing in for a dropped stream.
            StopAsyncIteration: The script ran out, standing in for a bounded request closing.
        """
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
        """Queues the streams each open hands out, and starts the call recorders."""
        self._streams = list(streams)
        self._terminal = terminal
        self.create_kwargs: dict[str, object] = {}
        self.stream_get_calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> _FakeStream:
        """Records the create kwargs and hands out the next queued stream.

        Returns:
            The next scripted stream.
        """
        self.create_kwargs = kwargs
        return self._streams.pop(0)

    async def get(self, **kwargs: object) -> object:
        """Hands out the next queued stream for a streaming get, else the terminal interaction.

        Returns:
            The next scripted stream when `stream=True`, else the terminal interaction.
        """
        if kwargs.get("stream"):
            self.stream_get_calls.append(kwargs)
            return self._streams.pop(0)
        return self._terminal


def _fake_client(*, streams: list[_FakeStream], terminal: object) -> SimpleNamespace:
    """Builds the `client.aio.interactions` chain `agent.py` reaches through.

    Returns:
        The client double; its `.aio.interactions` is the `_FakeInteractions` a test reads the
        recorded create kwargs and streaming-get calls back off.
    """
    return SimpleNamespace(
        aio=SimpleNamespace(interactions=_FakeInteractions(streams=streams, terminal=terminal))
    )


def _as_event(fake: object) -> "InteractionSSEEvent":
    """Views a fabricated SSE event double as the real SDK union a production signature expects.

    Production discriminates on `.event_type`, not isinstance, so a SimpleNamespace event is safe.

    Returns:
        `fake` unchanged, typed as `InteractionSSEEvent`.
    """
    return cast("InteractionSSEEvent", fake)


def _created_event(*, interaction_id: str = "int_9", event_id: str = "e1") -> SimpleNamespace:
    """Builds the `interaction.created` event the driver captures the interaction id off.

    Returns:
        The event double, carrying the id under `.interaction.id`.
    """
    return SimpleNamespace(
        event_type="interaction.created",
        event_id=event_id,
        interaction=SimpleNamespace(id=interaction_id, model="m"),
    )


def _thought_event(text: str, *, event_id: str = "e2") -> SimpleNamespace:
    """Builds the one delta shape the progress streamer accumulates: a thought-summary delta.

    Returns:
        The event double, shaped so `_feed`'s `event_type` then `delta.type` narrowing reaches it.
    """
    return SimpleNamespace(
        event_type="step.delta",
        event_id=event_id,
        delta=SimpleNamespace(type="thought_summary", content=SimpleNamespace(text=text)),
    )


def _completed_event(*, event_id: str = "e9") -> SimpleNamespace:
    """Builds the terminal `interaction.completed` event that stops the driver re-attaching.

    Returns:
        The event double, its `.interaction` deliberately empty: the real one carries no report
        body either, which is why the result is read from the terminal `get(id)` instead.
    """
    return SimpleNamespace(
        event_type="interaction.completed", event_id=event_id, interaction=SimpleNamespace()
    )


def _terminal_interaction() -> SimpleNamespace:
    """Builds the settled interaction the non-stream `get(id)` answers with.

    Returns:
        A completed interaction carrying the report text and usage `_to_result` maps.
    """
    return SimpleNamespace(
        id="int_9",
        status="completed",
        output_text="# Report\nbody",
        usage=SimpleNamespace(total_input_tokens=10, total_output_tokens=5),
        steps=[],
    )


async def test_stream_antigravity_persists_id_streams_and_returns_terminal_result() -> None:
    """The happy path: id persisted on the first event, reasoning painted, result from the get."""
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
    """A stream that simply ends with no terminal event re-attaches from the last event id."""
    # The SDK closes a bounded request mid-run, so a clean end is not evidence the run finished.
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
    """A stream that raises mid-run re-attaches from the last event id rather than failing."""
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
    """Once the re-attaches give up, the run is settled by the poll instead of being lost."""
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
    """A stream that dies before `interaction.created` raises: there is no id to poll or resume."""
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
    """A restart resume re-attaches through `get(stream=True)` and never calls `create`."""
    client = _fake_client(
        streams=[_FakeStream([_completed_event(event_id="e1")])], terminal=_terminal_interaction()
    )
    streamer = ResearchProgressStreamer(status=None, label="Antigravity")
    result = await agent.resume_research_stream(
        client=as_client(fake=client), interaction_id="int_9", streamer=streamer
    )
    assert client.aio.interactions.create_kwargs == {}
    assert result.ok is True


def test_is_terminal_event_classifies_statuses() -> None:
    """Which SSE events settle a run and stop the driver re-attaching."""
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
    """A completed interaction maps to its report text, decoded chart bytes and token counts."""
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
    """An interaction carrying only id and status degrades to defaults instead of raising."""
    interaction = SimpleNamespace(id="int_x", status="failed")
    result = agent._to_result(interaction=interaction)
    assert result.ok is False
    assert result.report_text == ""
    assert result.image_bytes is None
    assert result.input_tokens == 0


def test_latest_thought_returns_last_summary() -> None:
    """The content-shaped steps yield the newest summary; a step-less interaction yields None."""
    interaction = SimpleNamespace(
        steps=[
            SimpleNamespace(content=[SimpleNamespace(type="thought_summary", text="first")]),
            SimpleNamespace(content=[SimpleNamespace(type="thought_summary", text="second")]),
        ]
    )
    assert agent._latest_thought(interaction=interaction) == "second"
    assert agent._latest_thought(interaction=SimpleNamespace()) is None


def test_latest_thought_reads_thought_step_summary() -> None:
    """A materialized `thought` step keeps its text in `summary[].text`, not in `content`."""
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
    """Only a thought-summary delta reaches the live view, not report text or lifecycle events."""
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
    """The preview keeps the newest lines under the timed header, and escapes agent mentions."""
    streamer = ResearchProgressStreamer(status=None, label="Antigravity")
    streamer.reasoning = "first line\n@everyone please\nlast line"
    preview = streamer._render_preview()
    assert preview.startswith("-# Researching... (Antigravity,")
    assert "@everyone" not in preview  # agent text is escaped so the thinking can never ping
    assert "last line" in preview


async def test_streamer_write_snapshot_edits_and_skips_unchanged() -> None:
    """A snapshot write carries a no-ping policy, and an unchanged snapshot writes nothing."""
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
    """Streaming concatenates the thought deltas and leaves no cadence editor task behind."""
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
    """With no LLM title, the thread is named after the brief's first line, or `深度研究`."""
    name = research_cog._fallback_thread_name(brief="研究 TPU 的歷史與競爭格局\n更多細節")
    assert name.startswith("研究 TPU")
    assert "\n" not in name
    assert research_cog._fallback_thread_name(brief="   ") == "深度研究"


def test_terminal_phase_mapping() -> None:
    """A terminal status maps onto a stored phase; all but completed and cancelled read failed."""
    assert research_cog._terminal_phase(status="completed") == "done"
    assert research_cog._terminal_phase(status="cancelled") == "cancelled"
    assert research_cog._terminal_phase(status="budget_exceeded") == "failed"


def test_owner_id_from_mention_parses_digits() -> None:
    """Both `<@id>` spellings parse back to the owner id, and a non-mention yields 0."""
    assert research_cog._owner_id_from_mention(mention="<@123456789>") == 123456789
    assert research_cog._owner_id_from_mention(mention="<@!42>") == 42
    assert research_cog._owner_id_from_mention(mention="nobody") == 0


def test_deep_research_available_requires_enabled_and_key() -> None:
    """Deep research needs the kill-switch on AND a non-blank key: it runs direct to Google."""
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
    """The report's mention policy resolves the owner alone, never `@everyone` or a role."""
    mentions = research_cog._owner_allowed_mentions(owner_id=42)
    assert mentions.everyone is False
    assert mentions.roles is False
    users = mentions.users
    assert isinstance(users, list)
    assert [obj.id for obj in users] == [42]


def test_failure_text_distinguishes_budget() -> None:
    """A cost stop and a cancellation each say so; every other failure gets the generic line."""
    assert "成本上限" in research_cog._failure_text(status="budget_exceeded")
    assert "取消" in research_cog._failure_text(status="cancelled")
    assert research_cog._failure_text(status="failed")


# ----- persistence (reply.db) ---------------------------------------------------------------


async def _only_resumable(*, thread_id: int) -> rdb.PersistentResearchSession | None:
    """Picks one thread out of the resumable set; the store has no single-row reader left to use.

    Returns:
        That thread's session while it is still `researching`, or None once it is terminal or
        was never written.
    """
    return next((row for row in await rdb.list_resumable() if row.thread_id == thread_id), None)


async def test_session_round_trip(research_isolated_db: None) -> None:
    """A launched session reads back every column the resume needs; an unknown thread has none."""
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
    """The id lands on the row, and leaving `researching` drops it from resume and owner slot."""
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
    """The one-per-owner slot is held only while researching, and only against that owner."""
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
    """The restart sweep picks up in-flight sessions alone, never a terminal one."""
    # One session per phase, so only the researching one may come back as resumable.
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
    """A stored phase this store no longer knows narrows to `failed`: nothing is in flight."""
    assert rdb.cast_phase(value="researching") == "researching"
    assert rdb.cast_phase(value="bogus") == "failed"
    # A row the removed escalation tiers left in `planning` is no longer a phase this store knows.
    assert rdb.cast_phase(value="planning") == "failed"


async def test_a_legacy_planning_row_no_longer_blocks_its_owner(
    research_isolated_db: None,
) -> None:
    """A row stuck in the removed tiers' `planning` holds no owner slot and is never resumed."""
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
        """Starts the edit recorder."""
        self.edits: list[dict[str, object]] = []

    async def edit(self, **kwargs: object) -> None:
        """Records one edit payload whole, so a test can read its content, files and mentions."""
        self.edits.append(kwargs)


class _FakeThread:
    """Records `send` calls and exposes a guild upload limit, like a real Thread."""

    id = 1

    def __init__(self) -> None:
        """Starts the send recorder and a guild reporting a plain 10MB upload ceiling.

        `_upload_limit` reads `guild.filesize_limit`, so a test that wants an oversize attachment
        overwrites `guild` with a tiny one rather than growing the file.
        """
        self.sends: list[dict[str, object]] = []
        self.guild = SimpleNamespace(filesize_limit=10 * 1024 * 1024)

    async def send(self, **kwargs: object) -> None:
        """Records one send payload whole, so a test can read its content, files and mentions."""
        self.sends.append(kwargs)


def _completed_result(
    *, report_text: str, image_bytes: bytes | None = None
) -> agent.ResearchResult:
    """Builds a settled research result the delivery path treats as a finished run.

    Returns:
        A `completed` result carrying that report markdown and, when given, a chart image.
    """
    return agent.ResearchResult(
        interaction_id="int_1",
        status="completed",
        report_text=report_text,
        image_bytes=image_bytes,
    )


async def test_delivery_keeps_footer_message_under_the_limit() -> None:
    """A near-limit last chunk pushes footer, ping and `research.md` onto a trailing message."""
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
    """A short report is one edit of the opening status message, footer, ping and file included."""
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
