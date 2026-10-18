"""Direct Gemini Interactions call layer for the deep-research cog.

Runs `RuntimeModelCatalog.antigravity_model` through an injected `genai.Client` directly against
Google (`gemini_api_key`, no proxy) with `background=True`, `store=True`, and `stream=True`.

Both `stream_antigravity` (fresh run in a remote sandbox) and `resume_research_stream` (restart
re-attachment) share `_StreamDriver` and `_drive`. When the SDK closes a long-lived SSE request
mid-run, `_StreamDriver` re-attaches via `interactions.get(stream=True, last_event_id=...)`.
The final result is always read through `_poll_until_terminal` (`interactions.get(id)` with
transient-error retries) rather than from the stream, since `interaction.completed` carries no
report body and the poll also settles runs whose stream died or lags in `in_progress`. Neither
streaming nor polling imposes a wall-clock timeout.
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
    AntigravityAgentConfigParam,
)

from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.llm_errors import is_retryable_llm_error

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable, AsyncIterator

    from google.genai.interactions import Step, InteractionSSEEvent

    from discordbot.cogs.research.streaming import ResearchProgressStreamer

# Built-in tools enabled for every research run: web grounding + URL reading. Passed explicitly
# (not left to the agent default) so search/url grounding is guaranteed. Code execution is left
# off on purpose: the agent's bash/python tool calls leak into `output_text` as raw
# `call:default_api:bash{command:...}` text and corrupt the report.
RESEARCH_TOOLS = [
    URLContext(type="url_context"),
    GoogleSearch(search_types=["web_search"], type="google_search"),
]

# The agent family's own config block: `model` picks the reasoning model, while
# `max_total_tokens` stays at its default. Uses the `...Param` TypedDict that
# `interactions.create` declares. Do not read `agent_config` back off the interaction to verify
# it took: `interactions.get` can report `agent_config=None` even after a 200 create.
RESEARCH_AGENT_CONFIG = AntigravityAgentConfigParam(
    type="antigravity", model=RuntimeModelCatalog().antigravity_reasoning_model.name
)

# The poll-fallback interval + the re-attach backoff; research is minutes-long so coarse is plenty.
RESEARCH_POLL_INTERVAL_SECONDS = 15.0


class _InteractionUsage(Protocol):
    """Structural view of the token counts a terminal interaction reports.

    Both fields are optional on the SDK model and absent when an interaction never billed.
    """

    @property
    def total_input_tokens(self) -> int | None: ...

    @property
    def total_output_tokens(self) -> int | None: ...


class _ResearchInteraction(Protocol):
    """Structural view of a terminal research interaction, as this module reads it.

    `interactions.get` returns `Interaction | AsyncStream[...]`, so the attributes actually read
    are declared here and cast to; `gen_reply/generation.py::_InteractionResult` has why neither
    an `isinstance` check nor the response class narrows it.

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


class ResearchResult(BaseModel):
    """The terminal outcome of a research run."""

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
        """Returns whether the research finished with status `"completed"`."""
        return self.status == "completed"


def _extract_image(interaction: _ResearchInteraction) -> bytes | None:
    """Returns the first decoded image from `interaction`'s `model_output` steps, if any."""
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


def _to_result(interaction: _ResearchInteraction) -> ResearchResult:
    """Maps a terminal interaction to a `ResearchResult`, reading absent token counts as zero."""
    return ResearchResult(
        status=str(interaction.status),
        report_text=(interaction.output_text or ""),
        image_bytes=_extract_image(interaction=interaction),
        input_tokens=(
            int(interaction.usage.total_input_tokens or 0) if interaction.usage is not None else 0
        ),
        output_tokens=(
            int(interaction.usage.total_output_tokens or 0) if interaction.usage is not None else 0
        ),
    )


async def _poll_until_terminal(client: genai.Client, interaction_id: str) -> _ResearchInteraction:
    """Polls `interactions.get` without a wall-clock timeout until status leaves `in_progress`.

    Retries transient errors up to `max_consecutive_errors` consecutive times and re-raises
    non-retryable errors immediately.
    """
    max_consecutive_errors = 30
    consecutive_errors = 0
    while True:
        try:
            interaction = cast(
                "_ResearchInteraction", await client.aio.interactions.get(id=interaction_id)
            )
        except Exception as exc:
            if not is_retryable_llm_error(exc=exc):
                raise
            consecutive_errors += 1
            logfire.warn(
                "research poll error; retrying",
                interaction_id=interaction_id,
                consecutive_errors=consecutive_errors,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            if consecutive_errors >= max_consecutive_errors:
                raise
            await asyncio.sleep(RESEARCH_POLL_INTERVAL_SECONDS)
            continue
        consecutive_errors = 0
        if interaction.status != "in_progress":
            return interaction
        await asyncio.sleep(RESEARCH_POLL_INTERVAL_SECONDS)


# The SDK can close a create/get(stream=True) SSE request mid-run (each request is bounded) while
# the agent keeps working server-side; `_StreamDriver` re-attaches via get(stream=True). This caps
# consecutive re-attaches that make no progress so a dead stream gives up, while a healthy long
# run that needs periodic re-attach never trips.
MAX_STREAM_RECONNECTS = 20

# Called with the interaction id the moment `interaction.created` arrives (the stream's first event),
# so the cog persists the id before the minutes-long run.
type CreatedCallback = Callable[[str], Awaitable[None]]


async def _noop_created(_interaction_id: str) -> None:
    """A `CreatedCallback` that persists nothing (a resume does not re-store the id)."""
    return


def _is_terminal_event(event: "InteractionSSEEvent") -> bool:
    """Returns whether `event` marks the interaction as settled."""
    if event.event_type in ("interaction.completed", "error"):
        return True
    if event.event_type == "interaction.status_update":
        return event.status not in ("in_progress", "requires_action")
    return False


class _StreamDriver(BaseModel):
    """Drives one research interaction's SSE stream with reconnect and id capture.

    Captures the interaction id on `interaction.created` via `on_created` and the resume token on
    every event, re-attaching via `get(stream=True, last_event_id=...)` when the SDK closes the
    request before a terminal event.
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

    async def _persist_created(self, interaction_id: str, on_created: "CreatedCallback") -> None:
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
    driver: _StreamDriver,
    streamer: "ResearchProgressStreamer",
    open_initial: "Callable[[], Awaitable[AsyncIterator[InteractionSSEEvent]]]",
    on_created: "CreatedCallback",
) -> _ResearchInteraction:
    """Streams events through `streamer`, then polls for the authoritative terminal interaction.

    Re-raises stream errors that happen before `interaction.created` assigns an id; once an id
    exists, stream errors are logged and `_poll_until_terminal` settles the run.
    """
    try:
        await streamer.stream(
            events=driver.events(open_initial=open_initial, on_created=on_created)
        )
    except Exception as exc:
        if not driver.interaction_id:
            raise
        logfire.warn(
            "research stream failed; polling for the terminal result",
            interaction_id=driver.interaction_id,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
    return await _poll_until_terminal(client=driver.client, interaction_id=driver.interaction_id)


async def stream_antigravity(  # noqa: PLR0913 -- the streaming create inputs plus the streamer + id callback
    client: genai.Client,
    agent: str,
    brief: str,
    system_instruction: str,
    streamer: "ResearchProgressStreamer",
    on_created: "CreatedCallback",
) -> ResearchResult:
    """Returns the terminal `ResearchResult` after streaming an Antigravity research run.

    Args:
        client (genai.Client): Direct-to-Google Gemini client.
        agent (str): Managed-agent identifier to invoke.
        brief (str): User research brief passed as interaction input.
        system_instruction (str): System instruction for the research agent.
        streamer (ResearchProgressStreamer): Live progress streamer for reasoning deltas.
        on_created (CreatedCallback): Callback invoked with the interaction ID on creation.
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
            agent_config=RESEARCH_AGENT_CONFIG,
            tools=RESEARCH_TOOLS,
            background=True,
            store=True,
            stream=True,
        )
        return cast("AsyncIterator[InteractionSSEEvent]", responses)

    logfire.info("research antigravity streaming", agent=agent)
    interaction = await _drive(
        driver=driver, streamer=streamer, open_initial=_open, on_created=on_created
    )
    return _to_result(interaction=interaction)


async def resume_research_stream(
    client: genai.Client, interaction_id: str, streamer: "ResearchProgressStreamer"
) -> ResearchResult:
    """Returns the terminal `ResearchResult` after re-attaching to a running interaction.

    Args:
        client (genai.Client): Direct-to-Google Gemini client.
        interaction_id (str): ID of the running interaction to resume.
        streamer (ResearchProgressStreamer): Live progress streamer for reasoning deltas.
    """
    driver = _StreamDriver(client=client, interaction_id=interaction_id)

    async def _open() -> "AsyncIterator[InteractionSSEEvent]":
        responses = await client.aio.interactions.get(id=interaction_id, stream=True)
        return cast("AsyncIterator[InteractionSSEEvent]", responses)

    interaction = await _drive(
        driver=driver, streamer=streamer, open_initial=_open, on_created=_noop_created
    )
    return _to_result(interaction=interaction)
