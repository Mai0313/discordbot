"""Tests for the deep-research agent driver: streaming, reconnects, polling, result mapping."""

from types import SimpleNamespace
import base64
from typing import TYPE_CHECKING

import httpx
from google import genai
import pytest
from google.genai import types

from discordbot.cogs.research import agent
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs.research.streaming import ResearchProgressStreamer

from tests.helpers.casting import as_client
from tests.helpers.research import (
    _as_event,
    _FakeStream,
    _fake_client,
    _created_event,
    _thought_event,
    _completed_event,
    _terminal_interaction,
)
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from collections.abc import Callable


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
    # The config block names the agent's own family and the catalog's reasoning model; a
    # `deep-research` one here would be the removed tiers' `collaborative_planning` coming back, and
    # nothing upstream rejects a mismatched family, so the discriminator is pinned here instead.
    assert kwargs["agent_config"] is agent.RESEARCH_AGENT_CONFIG
    assert kwargs["agent_config"]["type"] == "antigravity"
    assert (
        kwargs["agent_config"]["model"] == RuntimeModelCatalog().antigravity_reasoning_model.name
    )
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

    result = await agent.stream_antigravity(
        client=as_client(fake=client),
        agent="a",
        brief="b",
        system_instruction="s",
        streamer=streamer,
        on_created=agent._noop_created,
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

    result = await agent.stream_antigravity(
        client=as_client(fake=client),
        agent="a",
        brief="b",
        system_instruction="s",
        streamer=streamer,
        on_created=agent._noop_created,
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
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")

    # Streaming exhausts its reconnects, so the driver degrades to the poll and still returns the
    # authoritative terminal result.
    result = await agent.stream_antigravity(
        client=as_client(fake=client),
        agent="a",
        brief="b",
        system_instruction="s",
        streamer=streamer,
        on_created=agent._noop_created,
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
            on_created=agent._noop_created,
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


_POLLED_TERMINAL = {
    "id": "int_9",
    "status": "completed",
    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "# Report"}]}],
}


class _PolledInteraction:
    """A real Gemini client whose first `failures` poll requests each meet `failure`.

    Every later request answers the completed interaction. The SDK's own re-sends count as
    requests, so `polls` is what actually went over the wire.
    """

    def __init__(
        self, failure: "Callable[[httpx.Request], httpx.Response]", failures: int
    ) -> None:
        self.failure = failure
        self.failures = failures
        self.polls = 0
        self.client = genai.Client(
            api_key="test",
            http_options=types.HttpOptions(
                httpx_async_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(handler=self._respond)
                ),
                # Keeps the SDK's own re-sends but makes them sleepless.
                retry_options=types.HttpRetryOptions(initial_delay=0),
            ),
        )

    def _respond(self, request: httpx.Request) -> httpx.Response:
        self.polls += 1
        if self.polls <= self.failures:
            return self.failure(request)
        return httpx.Response(status_code=200, json=_POLLED_TERMINAL)


def _poll_answered(status: int) -> "Callable[[httpx.Request], httpx.Response]":
    """A poll the API answers with `status`."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status,
            json={"error": {"code": status, "message": f"provider answered {status}"}},
        )

    return respond


def _poll_connection_refused(request: httpx.Request) -> httpx.Response:
    """A poll that fails before any response arrives."""
    raise httpx.ConnectError("connection refused", request=request)


@pytest.mark.parametrize("status", [400, 403, 404])
async def test_a_poll_the_api_refuses_ends_the_run_at_once(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setattr(agent, "RESEARCH_POLL_INTERVAL_SECONDS", 0.0)
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    polled = _PolledInteraction(failure=_poll_answered(status=status), failures=100)

    # Asking again cannot change a refusal, so the first one ends the poll.
    with pytest.raises(Exception, match=f"provider answered {status}"):
        await agent._poll_until_terminal(client=polled.client, interaction_id="int_9")
    assert polled.polls == 1
    assert warns == []


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(_poll_answered(status=503), id="http-503"),
        pytest.param(_poll_answered(status=429), id="http-429"),
        pytest.param(_poll_connection_refused, id="connection-refused"),
    ],
)
async def test_a_transient_poll_error_is_still_retried(
    monkeypatch: pytest.MonkeyPatch, failure: "Callable[[httpx.Request], httpx.Response]"
) -> None:
    monkeypatch.setattr(agent, "RESEARCH_POLL_INTERVAL_SECONDS", 0.0)
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    # More failures than one request's own SDK re-sends absorb, so only the poll's retry gets past.
    polled = _PolledInteraction(failure=failure, failures=10)

    interaction = await agent._poll_until_terminal(client=polled.client, interaction_id="int_9")

    assert interaction.status == "completed"
    assert any(message == "research poll error; retrying" for message, _ in warns)


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
