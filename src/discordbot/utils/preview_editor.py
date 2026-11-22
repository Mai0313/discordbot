"""Fixed-cadence background editor for live Discord preview messages."""

import asyncio
import contextlib
from collections.abc import Callable, Awaitable

import logfire
from nextcord import NotFound, Forbidden
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr


class CadencePreviewEditor(BaseModel):
    """Drives a Discord preview snapshot writer on a fixed interval until stopped.

    Stopped via an event rather than task cancellation so an in-flight Discord write finishes
    before the caller overwrites or reuses the message.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    write_snapshot: Callable[[], Awaitable[None]] = Field(
        ..., description="Async callback that renders and writes the latest preview snapshot."
    )
    interval_seconds: float = Field(
        ..., description="Delay between successive snapshot writes while running."
    )
    message_id: int | None = Field(
        default=None, description="Discord message id attached to preview lifecycle logs."
    )
    channel_id: int | None = Field(
        default=None,
        description="Optional Discord channel id attached to `Forbidden` logs when provided.",
    )
    deleted_log: str = Field(
        default="Preview message deleted; stopping preview edits",
        description="Info log emitted when Discord answers `NotFound`.",
    )
    forbidden_log: str = Field(
        default="Channel refused a preview write; stopping preview edits",
        description="Warn log emitted when Discord answers `Forbidden`.",
    )
    error_log: str = Field(
        default="Preview snapshot edit failed; continuing",
        description="Warn log emitted once on the first unexpected snapshot error.",
    )
    crash_log: str = Field(
        default="Preview editor task ended with an error",
        description="Warn log emitted if the background task exits with an uncaught exception.",
    )
    _task: asyncio.Task[None] | None = PrivateAttr(default=None)
    _stop_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _error_logged: bool = PrivateAttr(default=False)

    @property
    def is_running(self) -> bool:
        """Whether the background cadence task is currently active."""
        return self._task is not None and not self._task.done()

    async def run(self) -> None:
        """Writes snapshots on `interval_seconds` cadence until stopped or a terminal Discord error."""
        while not self._stop_event.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
            if self._stop_event.is_set():
                return
            if not await self._tick():
                return

    async def _tick(self) -> bool:
        """Runs one preview snapshot write, returning False on terminal Discord errors."""
        try:
            await self.write_snapshot()
        except NotFound:
            logfire.info(self.deleted_log, message_id=self.message_id)
            return False
        except Forbidden:
            if self.channel_id is not None:
                logfire.warn(
                    self.forbidden_log, channel_id=self.channel_id, message_id=self.message_id
                )
            else:
                logfire.warn(self.forbidden_log, message_id=self.message_id)
            return False
        except Exception as exc:
            # Broad: live preview is best-effort; log once so repeated ticks do not spam warnings.
            if not self._error_logged:
                self._error_logged = True
                logfire.warn(
                    self.error_log,
                    message_id=self.message_id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
        return True

    def start(self) -> None:
        """Starts the background cadence task if it is not already running."""
        if self._task is None:
            self._task = asyncio.create_task(coro=self.run())

    def reset(self) -> None:
        """Clears the stop signal so `start` can launch a fresh cycle (e.g. after a retry)."""
        self._stop_event.clear()

    async def stop(self) -> None:
        """Signals the cadence loop to stop and waits out any in-flight snapshot write."""
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await self._task
        except Exception as exc:
            # Broad: preview failure must not sink the deliverable; cancellation still propagates.
            logfire.warn(
                self.crash_log,
                message_id=self.message_id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        finally:
            self._task = None
