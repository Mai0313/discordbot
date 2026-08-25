"""Direct Gemini Interactions call layer for the deep-research cog.

The one research agent (`antigravity-preview-05-2026`) runs through an injected `genai.Client`
that talks DIRECT to Google (`gemini_api_key`, no proxy): a managed agent rides the native
Interactions API, which this project always calls direct rather than through the LiteLLM proxy's
interactions transform. The create uses `background=True` + `store=True` + `stream=True`, so the
agent's reasoning streams live to the thread (`_StreamDriver` + `ResearchProgressStreamer`) while
it works.

Both call shapes share `_StreamDriver` / `_drive` (SSE consume + reconnect + terminal extract):
- `stream_antigravity`: streams the one-shot agent in a remote sandbox environment.
- `resume_research_stream`: re-attaches a live stream to an already-running interaction (restart resume).

Robustness: the SDK can close a long-lived streaming request mid-run while the agent keeps
working server-side, so `_StreamDriver` re-attaches via `interactions.get(stream=True,
last_event_id=...)`. The final result is ALWAYS read through `_poll_until_terminal` (a terminal
non-stream `interactions.get(id)` with retry-on-error): the streamed deltas are the live view
only, `interaction.completed` carries no report body on purpose, and the poll both settles a run
whose stream died and waits out any brief `in_progress` visibility lag, then `_to_result` maps it.
There is no wall-clock timeout anywhere here, so what escapes is either the SDK error from a create
or a poll that never recovered, or a `RuntimeError` when the stream ended before any interaction id
existed; the cog maps both to a friendly message.
"""

import base64
from typing import TYPE_CHECKING, Protocol, cast
import asyncio

from google import genai
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from google.genai.interactions import (
    URLContext,
    GoogleSearch,
    AllowlistParam,
    EnvironmentParam,
    AllowlistEntryParam,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable, AsyncIterator

    from google.genai.interactions import Step, InteractionSSEEvent

    from discordbot.cogs.research.streaming import ResearchProgressStreamer

# Built-in tools enabled for every research run: web grounding + URL reading. Passed explicitly
# (not left to the agent default) so search/url grounding is guaranteed. Code execution is left
# OFF on purpose: the agent's bash/python tool calls were leaking into `output_text` as raw
# `call:default_api:bash{command:...}` text and corrupting the report.
RESEARCH_TOOLS = [
    URLContext(type="url_context"),
    GoogleSearch(search_types=["web_search"], type="google_search"),
]

# The poll-fallback interval + the re-attach backoff; research is minutes-long so coarse is plenty.
RESEARCH_POLL_INTERVAL_SECONDS = 15.0
# A transient get() error mid-research (e.g. a server 504 gateway timeout) is retried, not fatal;
# only this many CONSECUTIVE failures give up. There is no wall-clock timeout: the Gemini SDK
# bounds each request and the agent settles server-side on its own budget.
MAX_CONSECUTIVE_POLL_ERRORS = 30


class _InteractionUsage(Protocol):
    """Structural view of the token counts a terminal interaction reports.

    Both are Optional on the SDK model and absent on an interaction that never billed, so a
    None is read as zero rather than as a missing field.
    """

    @property
    def total_input_tokens(self) -> int | None: ...

    @property
    def total_output_tokens(self) -> int | None: ...


class _ResearchInteraction(Protocol):
    """Structural view of a terminal research interaction, as this module reads it.

    `interactions.get` returns `Interaction | AsyncStream[...]`, and the stream cannot be excluded
    with `isinstance(x, AsyncIterator)` because genai's `AsyncStream` is only structurally an
    `AsyncIterator` and so stays in the union; naming the response class instead is its own trap
    (`google.genai.interactions` star-imports `Interaction` from both the request-union alias and
    the response module, so which one wins rests on import order). The attributes actually read
    are declared here and cast to, exactly as `gen_reply/generation.py::_InteractionResult` does.

    `steps` stays the SDK's open `Step` union: its members carry genuinely different payloads, so
    `_extract_image` still probes each one rather than reading a shape this could declare.
    """

    @property
    def id(self) -> str | None: ...

    @property
    def status(self) -> str: ...

    @property
    def output_text(self) -> str | None: ...

    @property
    def usage(self) -> _InteractionUsage | None: ...

    @property
    def steps(self) -> "list[Step] | None": ...


class _TokenUsage(BaseModel):
    """The token counts read off one terminal interaction."""

    input_tokens: int = Field(default=0, description="Reported input tokens for the interaction.")
    output_tokens: int = Field(
        default=0, description="Reported output tokens for the interaction."
    )


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


def _extract_image(*, interaction: _ResearchInteraction) -> bytes | None:
    """Returns the first generated image (decoded) from an interaction's model_output steps.

    A step that fails to decode is reported rather than swallowed: it is the one path that loses
    a generated chart from an otherwise complete report, and the report itself carries no trace
    of the missing figure.
    """
    for step in interaction.steps or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for item in getattr(step, "content", None) or []:
            if getattr(item, "type", None) == "image" and getattr(item, "data", None):
                try:
                    return base64.b64decode(item.data)
                except Exception as exc:
                    # Broad: a corrupt payload only costs the chart, so the report is still
                    # delivered; without this the drop leaves nothing at all in the logs.
                    logfire.warn(
                        "research chart image could not be decoded; delivering without it",
                        interaction_id=interaction.id,
                        error_type=type(exc).__name__,
                        _exc_info=exc,
                    )
                    return None
    return None


def _extract_usage(*, interaction: _ResearchInteraction) -> _TokenUsage:
    """Returns the interaction's token counts, defaulting to zero."""
    usage = interaction.usage
    if usage is None:
        return _TokenUsage()
    return _TokenUsage(
        input_tokens=int(usage.total_input_tokens or 0),
        output_tokens=int(usage.total_output_tokens or 0),
    )


def _to_result(*, interaction: _ResearchInteraction) -> ResearchResult:
    """Maps a terminal interaction to a `ResearchResult`."""
    usage = _extract_usage(interaction=interaction)
    return ResearchResult(
        interaction_id=str(interaction.id or ""),
        status=str(interaction.status),
        report_text=(interaction.output_text or ""),
        image_bytes=_extract_image(interaction=interaction),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


async def _poll_until_terminal(
    *, client: genai.Client, interaction_id: str, poll_interval_seconds: float
) -> _ResearchInteraction:
    """Polls `interactions.get` until the status leaves `in_progress`.

    No wall-clock timeout (the SDK bounds each request; the agent settles server-side). A
    transient get() error mid-research is retried so one 504 does not kill a long run; it gives
    up only after `MAX_CONSECUTIVE_POLL_ERRORS` consecutive failures (re-raising the last error).
    """
    consecutive_errors = 0
    while True:
        try:
            interaction = cast(
                "_ResearchInteraction", await client.aio.interactions.get(id=interaction_id)
            )
        except Exception as exc:
            consecutive_errors += 1
            logfire.warn(
                "research poll error; retrying",
                interaction_id=interaction_id,
                consecutive_errors=consecutive_errors,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            if consecutive_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                raise
            await asyncio.sleep(poll_interval_seconds)
            continue
        consecutive_errors = 0
        if interaction.status != "in_progress":
            return interaction
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
    """A `CreatedCallback` that persists nothing (a resume does not re-store the id)."""
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

    async def _persist_created(
        self, *, interaction_id: str, on_created: "CreatedCallback"
    ) -> None:
        """Records the captured interaction id and hands it to the caller's persist callback."""
        self.interaction_id = interaction_id
        try:
            await on_created(interaction_id)
        except Exception as exc:
            # Broad: the callback is a caller-supplied DB write. Losing it does not stop this run,
            # but a restart cannot resume an unpersisted id, and raising here would kill the live
            # view for the rest of the run.
            logfire.error(
                "failed to persist research interaction id",
                interaction_id=interaction_id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

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
                    event_id = getattr(event, "event_id", None)
                    if event_id:
                        self.last_event_id = event_id
                    if event.event_type == "interaction.created" and not self.interaction_id:
                        await self._persist_created(
                            interaction_id=str(event.interaction.id or ""), on_created=on_created
                        )
                    if _is_terminal_event(event=event):
                        terminal = True
                    yield event
            except Exception as exc:
                # Broad: the SDK surfaces transport/SSE failures as arbitrary exception types; the
                # reconnect below is the handling, so every type is tolerated here.
                logfire.warn(
                    "research stream dropped; will reconnect",
                    interaction_id=self.interaction_id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
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
) -> _ResearchInteraction:
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
        poll_interval_seconds=RESEARCH_POLL_INTERVAL_SECONDS,
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
    """Streams the Antigravity research (reasoning live); returns the terminal result."""
    environment = EnvironmentParam(
        type="remote", network=AllowlistParam(allowlist=[AllowlistEntryParam(domain="*")])
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
