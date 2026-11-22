"""Live reasoning preview for a running deep-research interaction.

Edits the research thread's opening status message on a fixed interval with a windowed tail of
`-#` thought-summary lines under a `Researching...` header. Consumes an already-opened
Interactions SSE stream without managing the SDK client, interaction ID, reconnect loop, or
final report delivery (`agent.py` and `delivery.py`). Does not reuse `ResponseStreamer`.
"""

import time
from typing import TYPE_CHECKING
from collections.abc import AsyncIterator

import logfire
from nextcord import Message, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.utils import escape_mentions

from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT
from discordbot.utils.preview_editor import CadencePreviewEditor

if TYPE_CHECKING:
    from google.genai.interactions import InteractionSSEEvent

# Prefix matched by restart resume to locate the pre-restart status message.
RESEARCHING_PREFIX = "-# Researching..."


class ResearchProgressStreamer(BaseModel):
    """Renders a running research interaction's reasoning onto its status message.

    Accumulates `thought_summary` deltas from `stream(events=...)` while a background cadence
    task edits `status` with a windowed `-#` preview, ignoring report text, images, and tool
    steps.
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
    _editor: CadencePreviewEditor | None = PrivateAttr(default=None)
    _displayed: str = PrivateAttr(default="")

    def _feed(self, event: "InteractionSSEEvent") -> None:
        """Appends `thought_summary` delta text from `event` and starts the editor if needed."""
        if event.event_type != "step.delta":
            return
        delta = event.delta
        if delta.type != "thought_summary":
            return
        text = getattr(delta.content, "text", "") if delta.content is not None else ""
        if text:
            self.reasoning += text
            editor = self._preview_editor()
            if editor is not None:
                editor.start()

    def _render_preview(self) -> str:
        """Returns the elapsed-time header and the newest `-#` reasoning lines that fit one message."""
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

    def _preview_editor(self) -> CadencePreviewEditor | None:
        """Returns the cadence editor for `status`, or None when live preview is disabled."""
        if self.status is None:
            return None
        if self._editor is None:
            self._editor = CadencePreviewEditor(
                write_snapshot=self._write_preview_snapshot,
                interval_seconds=self.preview_interval_seconds,
                message_id=self.status.id,
                deleted_log="research status message deleted; stopping preview edits",
                forbidden_log="research thread refused the preview edit; stopping preview edits",
                error_log="research preview edit failed; continuing the run",
                crash_log="research preview editor ended with an error",
            )
        return self._editor

    async def stream(self, events: AsyncIterator["InteractionSSEEvent"]) -> None:
        """Paints reasoning onto `status` until `events` ends, stopping the editor in `finally`.

        Args:
            events (AsyncIterator[InteractionSSEEvent]): Stream of interaction SSE events.
        """
        editor = self._preview_editor()
        if editor is not None:
            editor.start()
        try:
            async for event in events:
                self._feed(event=event)
        finally:
            if self._editor is not None:
                await self._editor.stop()
        logfire.debug("research progress stream ended", reasoning_chars=len(self.reasoning))
