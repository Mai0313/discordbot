"""Direct Gemini Interactions call layer for the deep-research cog.

All three agents (`antigravity-preview-05-2026` default, `deep-research-preview-04-2026`
and `deep-research-max-preview-04-2026` escalation) run through one injected
`genai.Client` that talks DIRECT to Google (`gemini_api_key`, no proxy). The proxy is
bypassed on purpose: it drops `agent_config` in its interactions->responses transform, so
`collaborative_planning` only works direct (verified 2026-06-20). Every call uses
`background=True` + `store=True` + `stream=True`, so the agent's reasoning streams live to
the thread (`_StreamDriver` + `ResearchProgressStreamer`) while it works.

Call shapes share `_StreamDriver` / `_drive` (SSE consume + reconnect + terminal extract):
- `stream_antigravity`: streams the default one-shot agent in a remote sandbox environment.
- `stream_plan` / `stream_refine`: Deep Research collaborative planning (returns a plan, not a report).
- `stream_deep_research`: approves the planned interaction and streams the full research.
- `resume_research_stream`: re-attaches a live stream to an already-running interaction (restart resume).

Robustness: the SDK can close a long-lived streaming request mid-run while the agent keeps
working server-side, so `_StreamDriver` re-attaches via `interactions.get(stream=True,
last_event_id=...)`. The final result is ALWAYS read through `_poll_until_terminal` (a terminal
non-stream `interactions.get(id)` with retry-on-error): the streamed deltas are the live view
only, `interaction.completed` carries no report body on purpose, and the poll both settles a run
whose stream died and waits out any brief `in_progress` visibility lag, then `_to_result` maps it.
These functions can raise (network errors, `TimeoutError`); the cog maps failure to a friendly message.
"""

import time
import base64
from typing import TYPE_CHECKING, cast
import asyncio

from google import genai
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from google.genai._interactions.types import EnvironmentParam, DeepResearchAgentConfigParam
from google.genai._interactions.types.tool_param import URLContext, GoogleSearch
from google.genai._interactions.types.environment_param import (
    NetworkAllowlist,
    NetworkAllowlistAllowlist,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable, AsyncIterator

    from google.genai._interactions.types import InteractionSSEEvent

    from discordbot.cogs._research.streaming import ResearchProgressStreamer

# Built-in tools enabled for every research run: web grounding + URL reading. Passed explicitly
# (not left to the agent default) so search/url grounding is guaranteed. Code execution is left
# OFF on purpose: the agent's bash/python tool calls were leaking into `output_text` as raw
# `call:default_api:bash{command:...}` text and corrupting the report.
RESEARCH_TOOLS = [
    URLContext(type="url_context"),
    GoogleSearch(type="google_search", search_types=["web_search"]),
]

# The poll-fallback interval + the re-attach backoff; research is minutes-long so coarse is plenty.
RESEARCH_POLL_INTERVAL_SECONDS = 15.0
# A transient get() error mid-research (e.g. a server 504 gateway timeout) is retried, not fatal;
# only this many CONSECUTIVE failures give up. There is no wall-clock timeout: the Gemini SDK
# bounds each request and the agent settles server-side (Deep Research caps at 60 min).
MAX_CONSECUTIVE_POLL_ERRORS = 30

# Reports progress to the thread: (latest thought summary or None, elapsed seconds).
type ProgressCallback = Callable[[str | None, float], Awaitable[None]]


class ResearchPlan(BaseModel):
    """A proposed research plan returned by Deep Research collaborative planning."""

    interaction_id: str = Field(
        ..., description="The plan interaction's id, used to refine or approve it."
    )
    plan_text: str = Field(
        ..., description="The proposed research plan text, in the user's language."
    )
    status: str = Field(..., description="Terminal interaction status the plan was read from.")


class ResearchResult(BaseModel):
    """The terminal outcome of a research run."""

    interaction_id: str = Field(..., description="The research interaction's id.")
    status: str = Field(
        ..., description="Terminal interaction status (completed / failed / cancelled / ...)."
    )
    report_text: str = Field(
        default="", description="The final report markdown; empty on a non-completed status."
    )
    image_bytes: bytes | None = Field(
        default=None, description="First generated chart/visualization image, if any."
    )
    input_tokens: int = Field(default=0, description="Reported input tokens for the interaction.")
    output_tokens: int = Field(
        default=0, description="Reported output tokens for the interaction."
    )

    @property
    def ok(self) -> bool:
        """Whether the research finished cleanly."""
        return self.status == "completed"


def _deep_research_agent_config(*, collaborative_planning: bool) -> DeepResearchAgentConfigParam:
    """Builds the Deep Research `agent_config` for a plan or a full run."""
    return DeepResearchAgentConfigParam(
        type="deep-research",
        thinking_summaries="auto",
        visualization="auto",
        collaborative_planning=collaborative_planning,
    )


def _latest_thought(*, interaction: object) -> str | None:
    """Returns the most recent thought-summary text from an interaction's steps, if any.

    A materialized `thought` step carries its text in `step.summary[].text` (verified by spike
    dump), not in `step.content`; the older content-based shape is kept as a fallback.
    """
    latest: str | None = None
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) == "thought":
            for item in getattr(step, "summary", None) or []:
                text = getattr(item, "text", None)
                if text:
                    latest = text
        for item in getattr(step, "content", None) or []:
            if getattr(item, "type", None) in ("thought_summary", "thought"):
                text = getattr(item, "text", None)
                if text:
                    latest = text
    return latest


def _extract_image(*, interaction: object) -> bytes | None:
    """Returns the first generated image (decoded) from an interaction's model_output steps."""
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for item in getattr(step, "content", None) or []:
            if getattr(item, "type", None) == "image" and getattr(item, "data", None):
                try:
                    return base64.b64decode(item.data)
                except Exception:
                    return None
    return None


def _extract_usage(*, interaction: object) -> tuple[int, int]:
    """Returns `(input_tokens, output_tokens)` from an interaction, defaulting to zero."""
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return 0, 0
    inp = getattr(usage, "total_input_tokens", None) or getattr(usage, "input_tokens", None) or 0
    out = getattr(usage, "total_output_tokens", None) or getattr(usage, "output_tokens", None) or 0
    return int(inp), int(out)


def _to_result(*, interaction: object) -> ResearchResult:
    """Maps a terminal interaction to a `ResearchResult`."""
    input_tokens, output_tokens = _extract_usage(interaction=interaction)
    return ResearchResult(
        interaction_id=str(getattr(interaction, "id", "")),
        status=str(getattr(interaction, "status", "failed")),
        report_text=(getattr(interaction, "output_text", "") or ""),
        image_bytes=_extract_image(interaction=interaction),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def _poll_until_terminal(
    *,
    client: genai.Client,
    interaction_id: str,
    on_progress: "ProgressCallback | None",
    poll_interval_seconds: float,
) -> object:
    """Polls `interactions.get` until the status leaves `in_progress`.

    No wall-clock timeout (the SDK bounds each request; the agent settles server-side). A
    transient get() error mid-research is retried so one 504 does not kill a long run; it gives
    up only after `MAX_CONSECUTIVE_POLL_ERRORS` consecutive failures (re-raising the last error).
    """
    started = time.monotonic()
    consecutive_errors = 0
    while True:
        try:
            interaction = await client.aio.interactions.get(id=interaction_id)
        except Exception:
            consecutive_errors += 1
            logfire.warn(
                "research poll error; retrying",
                interaction_id=interaction_id,
                consecutive_errors=consecutive_errors,
            )
            if consecutive_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                raise
            await asyncio.sleep(poll_interval_seconds)
            continue
        consecutive_errors = 0
        if getattr(interaction, "status", None) != "in_progress":
            return interaction
        if on_progress is not None:
            await on_progress(_latest_thought(interaction=interaction), time.monotonic() - started)
        await asyncio.sleep(poll_interval_seconds)


# The SDK can close a create/get(stream=True) SSE request mid-run (each request is bounded) while
# the agent keeps working server-side; `_StreamDriver` re-attaches via get(stream=True). This caps
# CONSECUTIVE re-attaches that make no progress so a truly dead stream gives up (mirrors
# MAX_CONSECUTIVE_POLL_ERRORS), while a healthy long run that just needs periodic re-attach never trips.
MAX_STREAM_RECONNECTS = 20

# Called with the interaction id the moment `interaction.created` arrives (the stream's first event),
# so the cog persists the id BEFORE the minutes-long run, exactly as the old create-then-store split did.
type CreatedCallback = Callable[[str], Awaitable[None]]


async def _noop_created(_interaction_id: str) -> None:
    """A `CreatedCallback` that persists nothing (resume/plan do not re-store the id)."""
    return


def _is_terminal_event(*, event: "InteractionSSEEvent") -> bool:
    """Whether an SSE event marks the interaction as settled (so the driver stops re-attaching).

    `interaction.completed` and `error` are terminal; a `status_update` is terminal once it leaves
    the two non-final states. Any other status (`failed` / `cancelled` / `budget_exceeded` /
    `incomplete`) is a real terminal outcome the terminal `get(id)` then maps to a friendly result.
    """
    if event.event_type in ("interaction.completed", "error"):
        return True
    if event.event_type == "interaction.status_update":
        return event.status not in ("in_progress", "requires_action")
    return False


class _StreamDriver(BaseModel):
    """Drives one research interaction's SSE stream with reconnect + id capture.

    Yields every event to the `ResearchProgressStreamer` (for the live view) while capturing the
    interaction id on `interaction.created` (persisted via `on_created` before the long wait) and
    the resume token on every event. When the SDK closes the long-lived request without a terminal
    event, it re-attaches via `get(stream=True, last_event_id=...)` so the run continues seamlessly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[genai.Client] = Field(
        ..., description="Direct-to-Google Gemini client the stream is opened on."
    )
    interaction_id: str = Field(
        default="", description="Captured on interaction.created; empty until the first event."
    )
    last_event_id: str | None = Field(
        default=None, description="Resume token of the last received event for a re-attach."
    )

    async def _reopen(self) -> "AsyncIterator[InteractionSSEEvent]":
        """Re-attaches a live stream to the running interaction from the last resume token."""
        responses = await self.client.aio.interactions.get(
            id=self.interaction_id, stream=True, last_event_id=self.last_event_id
        )
        return cast("AsyncIterator[InteractionSSEEvent]", responses)

    async def events(
        self,
        *,
        open_initial: "Callable[[], Awaitable[AsyncIterator[InteractionSSEEvent]]]",
        on_created: "CreatedCallback",
    ) -> "AsyncIterator[InteractionSSEEvent]":
        """Yields events across reconnects until the interaction reaches a terminal event."""
        stream = await open_initial()
        empty_reconnects = 0
        while True:
            terminal = False
            progressed = False
            try:
                async for event in stream:
                    progressed = True
                    if event.event_id:
                        self.last_event_id = event.event_id
                    if event.event_type == "interaction.created" and not self.interaction_id:
                        self.interaction_id = str(event.interaction.id or "")
                        await on_created(self.interaction_id)
                    if _is_terminal_event(event=event):
                        terminal = True
                    yield event
            except Exception:
                logfire.warn(
                    "research stream dropped; will reconnect",
                    interaction_id=self.interaction_id,
                    _exc_info=True,
                )
            if terminal:
                return
            if not self.interaction_id:
                # The first stream ended before `interaction.created`: the create itself failed, so
                # there is no id to re-attach to. Surface it to the caller's fallback / cog failure.
                raise RuntimeError("research stream ended before interaction.created")
            empty_reconnects = 0 if progressed else empty_reconnects + 1
            if empty_reconnects > MAX_STREAM_RECONNECTS:
                raise RuntimeError("research stream reconnect gave up")
            await asyncio.sleep(RESEARCH_POLL_INTERVAL_SECONDS)
            stream = await self._reopen()


async def _drive(
    *,
    client: genai.Client,
    driver: _StreamDriver,
    streamer: "ResearchProgressStreamer",
    open_initial: "Callable[[], Awaitable[AsyncIterator[InteractionSSEEvent]]]",
    on_created: "CreatedCallback",
) -> object:
    """Runs the streamer over the driver's events, then returns the authoritative terminal interaction.

    The streamed deltas are the live view only; the result is ALWAYS read through
    `_poll_until_terminal` (a terminal non-stream `get(id)`) because `interaction.completed` carries
    an empty payload on purpose, so the existing `_to_result` extraction is reused unchanged. Routing
    the terminal read through the poll (not a single `get`) gives it the poll's retry-on-error and
    waits out any brief `in_progress` visibility lag, so a completed run is never misread as failed;
    it also transparently finishes a run whose stream died mid-way (the interaction lives server-side
    via `store=True`). A streaming failure BEFORE any id (the create itself failed) re-raises so the
    cog hits its normal failure path; once an id exists, streaming errors are swallowed and the poll
    settles the run.
    """
    try:
        await streamer.stream(
            events=driver.events(open_initial=open_initial, on_created=on_created)
        )
    except Exception:
        if not driver.interaction_id:
            raise
        logfire.warn(
            "research stream failed; polling for the terminal result",
            interaction_id=driver.interaction_id,
            _exc_info=True,
        )
    return await _poll_until_terminal(
        client=client,
        interaction_id=driver.interaction_id,
        on_progress=None,
        poll_interval_seconds=RESEARCH_POLL_INTERVAL_SECONDS,
    )


def _plan_from_interaction(*, interaction: object, fallback_id: str) -> ResearchPlan:
    """Maps a terminal planning interaction to a `ResearchPlan` (id falls back to the captured one)."""
    return ResearchPlan(
        interaction_id=str(getattr(interaction, "id", "") or fallback_id),
        plan_text=(getattr(interaction, "output_text", "") or ""),
        status=str(getattr(interaction, "status", "failed")),
    )


async def stream_antigravity(  # noqa: PLR0913 -- the streaming create inputs plus the streamer + id callback
    *,
    client: genai.Client,
    agent: str,
    brief: str,
    system_instruction: str,
    streamer: "ResearchProgressStreamer",
    on_created: "CreatedCallback",
) -> ResearchResult:
    """Streams the default Antigravity research (reasoning live); returns the terminal result."""
    environment = EnvironmentParam(
        type="remote", network=NetworkAllowlist(allowlist=[NetworkAllowlistAllowlist(domain="*")])
    )
    driver = _StreamDriver(client=client)

    async def _open() -> "AsyncIterator[InteractionSSEEvent]":
        responses = await client.aio.interactions.create(
            agent=agent,
            input=brief,
            system_instruction=system_instruction,
            environment=environment,
            tools=RESEARCH_TOOLS,
            background=True,
            store=True,
            stream=True,
        )
        return cast("AsyncIterator[InteractionSSEEvent]", responses)

    logfire.info("research antigravity streaming", agent=agent)
    interaction = await _drive(
        client=client, driver=driver, streamer=streamer, open_initial=_open, on_created=on_created
    )
    return _to_result(interaction=interaction)


async def stream_plan(
    *,
    client: genai.Client,
    agent: str,
    brief: str,
    system_instruction: str,
    streamer: "ResearchProgressStreamer",
) -> ResearchPlan:
    """Streams a Deep Research collaborative-planning turn (reasoning live); returns the plan."""
    driver = _StreamDriver(client=client)

    async def _open() -> "AsyncIterator[InteractionSSEEvent]":
        responses = await client.aio.interactions.create(
            agent=agent,
            input=brief,
            system_instruction=system_instruction,
            agent_config=_deep_research_agent_config(collaborative_planning=True),
            tools=RESEARCH_TOOLS,
            background=True,
            store=True,
            stream=True,
        )
        return cast("AsyncIterator[InteractionSSEEvent]", responses)

    interaction = await _drive(
        client=client,
        driver=driver,
        streamer=streamer,
        open_initial=_open,
        on_created=_noop_created,
    )
    return _plan_from_interaction(interaction=interaction, fallback_id=driver.interaction_id)


async def stream_refine(  # noqa: PLR0913 -- the streaming refine inputs plus the streamer
    *,
    client: genai.Client,
    agent: str,
    previous_interaction_id: str,
    feedback: str,
    system_instruction: str,
    streamer: "ResearchProgressStreamer",
) -> ResearchPlan:
    """Streams a plan refinement with the owner's feedback (reasoning live); returns the plan."""
    driver = _StreamDriver(client=client)

    async def _open() -> "AsyncIterator[InteractionSSEEvent]":
        responses = await client.aio.interactions.create(
            agent=agent,
            input=feedback,
            previous_interaction_id=previous_interaction_id,
            system_instruction=system_instruction,
            agent_config=_deep_research_agent_config(collaborative_planning=True),
            tools=RESEARCH_TOOLS,
            background=True,
            store=True,
            stream=True,
        )
        return cast("AsyncIterator[InteractionSSEEvent]", responses)

    interaction = await _drive(
        client=client,
        driver=driver,
        streamer=streamer,
        open_initial=_open,
        on_created=_noop_created,
    )
    return _plan_from_interaction(interaction=interaction, fallback_id=driver.interaction_id)


async def stream_deep_research(  # noqa: PLR0913 -- the streaming create inputs plus the streamer + id callback
    *,
    client: genai.Client,
    agent: str,
    previous_interaction_id: str,
    system_instruction: str,
    streamer: "ResearchProgressStreamer",
    on_created: "CreatedCallback",
) -> ResearchResult:
    """Approves a planned interaction and streams the full Deep Research run; returns the result."""
    driver = _StreamDriver(client=client)

    async def _open() -> "AsyncIterator[InteractionSSEEvent]":
        responses = await client.aio.interactions.create(
            agent=agent,
            input="The plan looks good, please proceed with the research.",
            previous_interaction_id=previous_interaction_id,
            system_instruction=system_instruction,
            agent_config=_deep_research_agent_config(collaborative_planning=False),
            tools=RESEARCH_TOOLS,
            background=True,
            store=True,
            stream=True,
        )
        return cast("AsyncIterator[InteractionSSEEvent]", responses)

    logfire.info("research deep-research streaming", agent=agent)
    interaction = await _drive(
        client=client, driver=driver, streamer=streamer, open_initial=_open, on_created=on_created
    )
    return _to_result(interaction=interaction)


async def resume_research_stream(
    *, client: genai.Client, interaction_id: str, streamer: "ResearchProgressStreamer"
) -> ResearchResult:
    """Re-attaches a live stream to an already-running research (restart resume); returns the result."""
    driver = _StreamDriver(client=client, interaction_id=interaction_id)

    async def _open() -> "AsyncIterator[InteractionSSEEvent]":
        responses = await client.aio.interactions.get(id=interaction_id, stream=True)
        return cast("AsyncIterator[InteractionSSEEvent]", responses)

    interaction = await _drive(
        client=client,
        driver=driver,
        streamer=streamer,
        open_initial=_open,
        on_created=_noop_created,
    )
    return _to_result(interaction=interaction)
