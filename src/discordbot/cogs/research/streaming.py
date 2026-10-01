"""Live reasoning preview for a running deep-research interaction.

Paints the agent's thought-summary text onto the research thread's opening status
message while the run is in flight, mirroring the QA reply streamer's cadence-editor
UX (`gen_reply/streaming.py`): a snapshot editor task edits the message on a fixed
interval with a windowed tail of `-#` reasoning lines under a `Researching...` header,
so the user watches the agent think.

Purpose-built and self-contained on purpose: it never touches the SDK, the interaction
id, the reconnect loop, or the final result (those live in `agent.py::_StreamDriver`);
it only consumes the already-opened Interactions SSE stream and renders reasoning. The
report itself is delivered separately by `delivery.py` after the run settles, so this
only ever shows thinking. It deliberately does NOT reuse `ResponseStreamer` (tightly
coupled to the QA reply) nor extract a shared base from it.
"""

import time
from typing import TYPE_CHECKING
import asyncio
from collections.abc import AsyncIterator

import logfire
from nextcord import Message, NotFound, Forbidden, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.utils import escape_mentions

from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT

if TYPE_CHECKING:
    from google.genai.interactions import InteractionSSEEvent

# Every status line of a running research opens with this, which is how a resume finds the one
# posted before a restart.
RESEARCHING_PREFIX = "-# Researching..."


class ResearchProgressStreamer(BaseModel):
    """Renders a running research interaction's reasoning onto its status message.

    The driver in `agent.py` opens the SSE stream (and handles the id, reconnects, and
    the terminal result); this consumes that stream via `stream(events=...)`, accumulates
    the thought-summary deltas, and a cadence editor task edits the status message with a
    windowed `-#` preview. Everything else in the stream (report text, images, tool steps)
    is ignored here because the finished report is delivered by `delivery.py`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: SkipValidation[Message | None] = Field(
        ..., description="Opening status message edited in place; None disables the live view."
    )
    label: str = Field(
        ..., description="Agent label shown in the header; the cog passes its `RESEARCH_LABEL`."
    )
    reasoning: str = Field(
        default="", description="Accumulated thought-summary text; only a windowed tail is shown."
    )
    preview_interval_seconds: float = Field(
        default=3.0,
        description=(
            "Cadence of the editor's Discord edits; research runs for minutes, so a coarse "
            "interval keeps the single message well under Discord's edit rate limit."
        ),
    )
    started_at: float = Field(
        default_factory=time.monotonic,
        description="Monotonic start time, for the elapsed timer in the header.",
    )
    _editor_task: asyncio.Task[None] | None = PrivateAttr(default=None)
    _editor_stop: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _displayed: str = PrivateAttr(default="")
    _preview_error_logged: bool = PrivateAttr(default=False)

    def _feed(self, *, event: "InteractionSSEEvent") -> None:
        """Accumulates one event's thought-summary text; ignores every other event/delta.

        Branch on `event.event_type` (then `delta.type`) directly so the discriminated unions
        narrow for the checker, exactly like `adapt_interactions_stream`. Only a `step.delta` carrying a
        `thought_summary` contributes to the live view; unknown deltas/events are skipped per the
        API's forward-compatibility guidance.
        """
        if event.event_type != "step.delta":
            return
        delta = event.delta
        if delta.type != "thought_summary":
            return
        text = getattr(delta.content, "text", "") if delta.content is not None else ""
        if text:
            self.reasoning += text
            self._ensure_editor_started()

    def _render_preview(self) -> str:
        """Builds the header plus the newest `-#` reasoning lines that fit one Discord message.

        The header carries an elapsed timer so the message reads as alive even before the first
        thought. Reasoning is agent text that may quote a mention, so it is `escape_mentions`-wrapped
        (a second guard on top of the edit's `AllowedMentions.none()`): the thinking must never ping.
        """
        elapsed = int(time.monotonic() - self.started_at)
        mins, secs = divmod(elapsed, 60)
        header = f"{RESEARCHING_PREFIX} ({self.label}, {mins}m{secs:02d}s)"
        if not self.reasoning:
            return header
        tail = escape_mentions(self.reasoning[-1500:])
        lines = [f"-# {line}" for line in tail.splitlines() if line.strip()]
        budget = DISCORD_MESSAGE_LIMIT - len(header)
        kept: list[str] = []
        for line in reversed(lines):
            if budget - (len(line) + 1) < 0:
                break
            kept.append(line)
            budget -= len(line) + 1
        kept.reverse()
        return "\n".join([header, *kept])

    async def _write_preview_snapshot(self) -> None:
        """Writes the latest preview to the status message, skipping unchanged snapshots."""
        if self.status is None:
            return
        preview = self._render_preview()
        if preview == self._displayed:
            return
        await self.status.edit(content=preview, allowed_mentions=AllowedMentions.none())
        self._displayed = preview

    async def _preview_editor(self) -> None:
        """Edits the status message with the latest snapshot on a fixed cadence until stopped.

        Stops via the event rather than task cancellation so an in-flight Discord write always
        lands before `deliver_report` reuses the same status message (a cancel could orphan it).
        """
        message_id = self.status.id if self.status is not None else None
        while True:
            try:
                await asyncio.wait_for(
                    self._editor_stop.wait(), timeout=self.preview_interval_seconds
                )
            except TimeoutError:
                try:
                    await self._write_preview_snapshot()
                except NotFound:
                    logfire.info(
                        "research status message deleted; stopping preview edits",
                        message_id=message_id,
                    )
                    return
                except Forbidden:
                    # The channel's permissions changed under the run; the id is the whole finding.
                    logfire.warn(
                        "research thread refused the preview edit; stopping preview edits",
                        message_id=message_id,
                    )
                    return
                except Exception as exc:
                    # Broad: the preview is best-effort and must never end the run. A failed write
                    # leaves the snapshot unadvanced, so the same error repeats every tick and only
                    # the first is logged.
                    if not self._preview_error_logged:
                        self._preview_error_logged = True
                        logfire.warn(
                            "research preview edit failed; continuing the run",
                            message_id=message_id,
                            error_type=type(exc).__name__,
                            _exc_info=exc,
                        )
            else:
                return

    def _ensure_editor_started(self) -> None:
        """Starts the cadence editor task once, only when there is a status message to edit."""
        if self.status is None:
            return
        if self._editor_task is None:
            self._editor_task = asyncio.create_task(coro=self._preview_editor())

    async def _stop_editor(self) -> None:
        """Signals the editor to stop and waits out any in-flight Discord write."""
        if self._editor_task is None:
            return
        self._editor_stop.set()
        try:
            await self._editor_task
        except Exception as exc:
            # Broad: the editor already catches its own Discord write failures, so anything here is
            # unexpected; swallowing keeps the preview best-effort. Catching Exception rather than
            # BaseException keeps cancellation propagating, as the stop-by-event contract needs.
            logfire.warn(
                "research preview editor ended with an error",
                message_id=self.status.id if self.status is not None else None,
                label=self.label,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        self._editor_task = None

    async def stream(self, *, events: AsyncIterator["InteractionSSEEvent"]) -> None:
        """Paints reasoning onto the status message until the event stream ends.

        Starts the editor up front so the elapsed timer ticks from t=0 even before the first
        thought, then feeds every event; the editor is always stopped in `finally` so any
        in-flight edit lands before the caller delivers the report on the same message.
        """
        self._ensure_editor_started()
        try:
            async for event in events:
                self._feed(event=event)
        finally:
            await self._stop_editor()
        logfire.debug("research progress stream ended", reasoning_chars=len(self.reasoning))
