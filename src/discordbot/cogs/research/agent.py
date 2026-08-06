"""Direct Gemini Interactions call layer for the deep-research cog.

The one research agent (`antigravity-preview-05-2026`) runs through an injected `genai.Client`
that talks DIRECT to Google (`gemini_api_key`, no proxy): a managed agent rides the native
Interactions API, which this project always calls direct rather than through the LiteLLM proxy's
interactions transform. Every call uses `background=True` + `store=True` + `stream=True`, so the
agent's reasoning streams live to the thread (`_StreamDriver` + `ResearchProgressStreamer`) while
it works.

Nothing here touches Discord: this half owns the SDK call, the reconnect loop and the terminal
read, and hands the cog one `ResearchResult`; the thread, the status message and the report
delivery are `cog.py` / `streaming.py` / `delivery.py`.

Both call shapes share `_StreamDriver` / `_drive` (SSE consume + reconnect + terminal extract):
- `stream_antigravity`: streams the one-shot agent in a remote sandbox environment.
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
from google.genai.interactions import (
    URLContext,
    GoogleSearch,
    AllowlistParam,
    EnvironmentParam,
    AllowlistEntryParam,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable, AsyncIterator

    from google.genai.interactions import InteractionSSEEvent

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

# Reports progress to the thread: (latest thought summary or None, elapsed seconds).
type ProgressCallback = Callable[[str | None, float], Awaitable[None]]


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
        """Whether the research finished cleanly.

        Returns:
            True when the terminal status is `completed`, otherwise False.
        """
        return self.status == "completed"


def _latest_thought(*, interaction: object) -> str | None:
    """Returns the most recent thought-summary text from an interaction's steps, if any.

    A materialized `thought` step carries its text in `step.summary[].text` (verified by spike
    dump), not in `step.content`; the older content-based shape is kept as a fallback. Every
    field is read with `getattr`, so an interaction shaped differently yields None rather than
    raising into the poll loop. Only `_poll_until_terminal`'s progress branch calls this, and its
    one caller passes no callback, so it is unreached at runtime today.

    Args:
        interaction (object): A fetched interaction; both step shapes are tolerated.

    Returns:
        The last summary seen while walking the steps in order, or None when there is none.
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
    """Returns the first generated image (decoded) from an interaction's model_output steps.

    Args:
        interaction (object): The terminal interaction to scan.

    Returns:
        The decoded image bytes, or None when no model_output step carried a decodable image.
    """
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for item in getattr(step, "content", None) or []:
            if getattr(item, "type", None) == "image" and getattr(item, "data", None):
                try:
                    return base64.b64decode(item.data)
                except Exception:
                    # Broad: a payload that will not decode is simply "no chart"; the report is
                    # the deliverable and a malformed image must not fail the run.
                    return None
    return None


def _extract_usage(*, interaction: object) -> tuple[int, int]:
    """Returns `(input_tokens, output_tokens)` from an interaction, defaulting to zero.

    Either the `total_input_tokens` / `total_output_tokens` or the bare `input_tokens` /
    `output_tokens` spelling is accepted, so the footer still prices a run whichever one the
    usage block carries.

    Args:
        interaction (object): The terminal interaction to read usage off.

    Returns:
        `(input_tokens, output_tokens)`, both zero when the interaction reports no usage.
    """
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return 0, 0
    inp = getattr(usage, "total_input_tokens", None) or getattr(usage, "input_tokens", None) or 0
    out = getattr(usage, "total_output_tokens", None) or getattr(usage, "output_tokens", None) or 0
    return int(inp), int(out)


def _to_result(*, interaction: object) -> ResearchResult:
    """Maps a terminal interaction to a `ResearchResult`.

    Every field is read with `getattr`, so a payload missing one degrades to that field's default
    (an absent status reads as `failed`) instead of raising into the cog's failure path.

    Args:
        interaction (object): The terminal interaction `_poll_until_terminal` returned.

    Returns:
        The mapped result; `ok` is False for any status but `completed`.
    """
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
    The error counter resets on any successful poll, so the cap is on a dead endpoint rather than
    on the run's total bad luck. `_drive` is the only caller and passes `on_progress=None` (the
    SSE deltas are the live view), so the progress branch is unreached at runtime today.

    Args:
        client (genai.Client): Direct-to-Google client the interaction lives on.
        interaction_id (str): Id of the interaction to poll.
        on_progress (ProgressCallback | None): Called after each non-terminal poll with the latest
            thought summary and the seconds since polling started; None reports nothing.
        poll_interval_seconds (float): Sleep between polls, and between retries after an error.

    Returns:
        The first interaction whose status is not `in_progress`.
    """
    started = time.monotonic()
    consecutive_errors = 0
    while True:
        try:
            interaction = await client.aio.interactions.get(id=interaction_id)
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
    """A `CreatedCallback` that persists nothing (a resume does not re-store the id).

    Args:
        _interaction_id (str): The captured id, deliberately ignored.
    """
    return


def _is_terminal_event(*, event: "InteractionSSEEvent") -> bool:
    """Whether an SSE event marks the interaction as settled (so the driver stops re-attaching).

    `interaction.completed` and `error` are terminal; a `status_update` is terminal once it leaves
    the two non-final states. Any other status (`failed` / `cancelled` / `budget_exceeded` /
    `incomplete`) is a real terminal outcome the terminal `get(id)` then maps to a friendly result.
    `requires_action` stays non-terminal: it is a generic Interactions status, so treating it as
    settled would end a live stream early.

    Args:
        event (InteractionSSEEvent): One event off the interaction's SSE stream.

    Returns:
        True once the run has settled, otherwise False.
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
        default="", description="Captured on interaction.created, or seeded up front by a resume."
    )
    last_event_id: str | None = Field(
        default=None, description="Resume token of the last received event for a re-attach."
    )

    async def _reopen(self) -> "AsyncIterator[InteractionSSEEvent]":
        """Re-attaches a live stream to the running interaction from the last resume token.

        Returns:
            The re-opened SSE stream, resuming after `last_event_id` rather than replaying.
        """
        responses = await self.client.aio.interactions.get(
            id=self.interaction_id, stream=True, last_event_id=self.last_event_id
        )
        return cast("AsyncIterator[InteractionSSEEvent]", responses)

    async def _persist_created(
        self, *, interaction_id: str, on_created: "CreatedCallback"
    ) -> None:
        """Records the captured interaction id and hands it to the caller's persist callback.

        The callback failing is logged and swallowed, so the live view survives; the cost is that
        a restart cannot resume that run.

        Args:
            interaction_id (str): Id read off the `interaction.created` event.
            on_created (CreatedCallback): Caller-supplied persist hook, run once per stream.
        """
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
        """Yields events across reconnects until the interaction reaches a terminal event.

        Captures the interaction id off the first `interaction.created` and persists it before the
        minutes-long wait, and tracks `last_event_id` so a re-attach resumes instead of replaying.
        A stream that ends or drops without a terminal event is re-opened; only a reconnect that
        yields nothing counts against `MAX_STREAM_RECONNECTS`, so a healthy long run that merely
        needs periodic re-attaching never trips it.

        Args:
            open_initial (Callable[[], Awaitable[AsyncIterator[InteractionSSEEvent]]]): Opens the
                first stream (a streaming `create` for a new run, a streaming `get` for a resume).
            on_created (CreatedCallback): Persist hook for the captured id.

        Yields:
            Every event off the stream in arrival order, the terminal one included.

        Raises:
            RuntimeError: The first stream ended before `interaction.created`, so there is no id
                to re-attach to, or the re-attaches gave up without progress.
        """  # noqa: DOC201 -- the quoted AsyncIterator annotation hides the generator from ruff
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

    Args:
        client (genai.Client): Direct-to-Google client for the terminal read.
        driver (_StreamDriver): Driver whose events feed the streamer and whose id the poll uses.
        streamer (ResearchProgressStreamer): Live reasoning view on the thread's status message.
        open_initial (Callable[[], Awaitable[AsyncIterator[InteractionSSEEvent]]]): Opens the
            first stream (a streaming `create` for a new run, a streaming `get` for a resume).
        on_created (CreatedCallback): Persist hook for the captured interaction id.

    Returns:
        The authoritative terminal interaction, for `_to_result` to map.
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


async def stream_antigravity(  # noqa: PLR0913 -- the streaming create inputs plus the streamer + id callback
    *,
    client: genai.Client,
    agent: str,
    brief: str,
    system_instruction: str,
    streamer: "ResearchProgressStreamer",
    on_created: "CreatedCallback",
) -> ResearchResult:
    """Streams the Antigravity research (reasoning live); returns the terminal result.

    The create rides `background=True` + `store=True`, so the run keeps going server-side and
    survives both a dropped stream and a bot restart. The remote sandbox opens with an
    allow-everything network policy because fetching the open web is the whole job, and
    `RESEARCH_TOOLS` is passed explicitly rather than left to the agent default. No `agent_config`
    is sent: it carried the removed tiers' `collaborative_planning` and nothing may put it back.

    Args:
        client (genai.Client): Direct-to-Google Gemini client (no proxy).
        agent (str): Managed-agent name; the cog passes `antigravity_model.name`.
        brief (str): The research request, sent as the interaction input.
        system_instruction (str): The agent system instruction the cog builds per run.
        streamer (ResearchProgressStreamer): Live reasoning view on the thread's status message.
        on_created (CreatedCallback): Persists the interaction id the moment it is captured, so a
            restart can resume this run.

    Returns:
        The terminal result; `ok` is False when the run did not complete.
    """
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
    """Re-attaches a live stream to an already-running research (restart resume); returns the result.

    Never calls `create`: `store=True` kept the interaction alive server-side, so the driver starts
    with the id already known and `on_created` is a no-op. A run that settled while the bot was
    down still resolves, because the terminal read is a plain `get(id)`.

    Args:
        client (genai.Client): Direct-to-Google Gemini client (no proxy).
        interaction_id (str): Id persisted before the restart.
        streamer (ResearchProgressStreamer): Live reasoning view on the thread's status message.

    Returns:
        The terminal result; `ok` is False when the run did not complete.
    """
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
