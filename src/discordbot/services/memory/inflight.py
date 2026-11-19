"""In-flight scheduling, replay merging, concurrency cap, and `reply.db` staging for memory.

Holds overlapping updates per scope and replays them sequentially per conversation source
(`subject`), merging memory notes from superseded turns within the same source unless a clear
intervened. Also owns the process-wide `memory_semaphore` and detached `reply.db` write queue.
"""

import time
from typing import Any
import asyncio
from collections.abc import Callable, Awaitable, Coroutine

import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation

from discordbot.typings.memory import MemoryWriteSummary

# Imported as a module so a test that swaps one database call on it reaches the calls here.
from discordbot.services.memory import database as memory_db
from discordbot.utils.asyncio_locks import (
    KeyedLockManager,
    LoopLocalRegistry,
    LoopLocalSemaphore,
    spawn_tracked,
)
from discordbot.services.memory.store import flavor_of, cleared_since
from discordbot.services.memory.writer import (
    NoteRound,
    MemoryWriterAI,
    parse_turn_payload,
    render_turn_payload,
)
from discordbot.typings.context_budgets import MEMORY_MERGED_NOTES_MAX
from discordbot.services.memory.constants import MEMORY_GLOBAL_CONCURRENCY

# In-memory callback notifying the cog of recorded writes; omitted for restart-resumed turns.
type MemoryWriteReport = Callable[[MemoryWriteSummary], Awaitable[None]]


class MemoryTurn(BaseModel):
    """One turn's memory review request, whether it runs now or is held for replay."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scope: str = Field(..., description="The memory scope this turn writes into.")
    subject: str = Field(..., description="The phase-1 directive naming the memory target.")
    transcript: str = Field(
        ...,
        description=(
            "The turn payload captured for the turn: the rendered transcript (reply folded "
            "in) plus its memory notes (`render_turn_payload`), so a replay needs no re-render."
        ),
    )
    writer: SkipValidation[MemoryWriterAI] = Field(
        ..., description="The memory writing service to run the update with."
    )
    identity: str = Field(
        ...,
        description=(
            "Single-line target identity `parse_identity` splits into the `owner_id` / "
            "`owner_name` stamped on every fact this scope writes."
        ),
    )
    token: int = Field(
        ...,
        description=(
            "Process-local logical version token persisted with the turn's DB row, reused on "
            "replay so the terminal write guards on the same id."
        ),
    )
    captured_at: float = Field(
        default_factory=time.monotonic,
        description=(
            "`time.monotonic()` when the turn was captured, so a clear that lands "
            "before it runs can abort it via `cleared_since`. Re-stamped by "
            "`enqueue_memory_update`, which is the moment that actually counts."
        ),
    )
    report: SkipValidation[MemoryWriteReport | None] = Field(
        default=None, description="Callback reporting what the turn recorded."
    )
    retry: bool = Field(
        default=False,
        description=(
            "Whether this turn retries a review that already failed: its forgets are "
            "already filed, and failing again closes the row instead of parking it."
        ),
    )


# Injected turn body from pipeline.py (Coroutine required by asyncio.create_task).
type TurnRunner = Callable[[MemoryTurn], Coroutine[Any, Any, None]]

# Per-scope in-flight task and held turns keyed by `subject` (which carries the source line).
# Keying by `subject` prevents merging turns from different guilds under one source stamp.
# Loop-local so an asyncio.Task never outlives its event loop.
_inflight_tasks: LoopLocalRegistry[str, asyncio.Task[None]] = LoopLocalRegistry()
_pending_updates: LoopLocalRegistry[str, dict[str, MemoryTurn]] = LoopLocalRegistry()

# Strong references to detached DB tasks until completion.
_db_tasks: set[asyncio.Task[None]] = set()

# Short per-scope lock serializing reply.db staging against clear tombstones (distinct from
# scope_lock).
staging_locks = KeyedLockManager[str]()

# Process-wide semaphore capping background memory updates; rebuilt per loop.
_memory_semaphore_holder = LoopLocalSemaphore(capacity_provider=lambda: MEMORY_GLOBAL_CONCURRENCY)


def memory_semaphore() -> asyncio.Semaphore:
    """Returns the process-wide semaphore, rebuilt when the event loop changes."""
    return _memory_semaphore_holder.get()


async def safe_db_write(coro: Awaitable[None], scope: str) -> None:
    """Awaits a best-effort `reply.db` write for `scope`, swallowing any failure.

    Persistence is an augmentation layer: a `reply.db` failure must never break the
    in-memory fire-and-forget memory pipeline.

    Args:
        coro (Awaitable[None]): Database write coroutine to execute.
        scope (str): Memory scope key for logging.
    """
    try:
        await coro
    except Exception as exc:
        logfire.warn(
            "memory_job persistence write failed",
            scope=scope,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )


def _spawn_db(coro: Awaitable[None], scope: str) -> None:
    """Spawns a tracked best-effort coroutine whose failures log as `memory_job` write errors."""
    spawn_tracked(
        coro=safe_db_write(coro=coro, scope=scope), tasks=_db_tasks, name="memory-db-write"
    )


async def stage_turn(  # noqa: PLR0913 -- one row's columns plus the turn's capture time
    scope: str, subject: str, transcript: str, identity: str, token: int, captured_at: float
) -> None:
    """Stages one turn only in the memory lifetime that captured it.

    Serializes staging against clear tombstones under `staging_locks`; returns without writing if
    the scope was cleared after `captured_at`.

    Args:
        scope (str): Memory scope key.
        subject (str): Phase-1 directive naming the review target.
        transcript (str): Rendered turn payload (transcript plus memory notes).
        identity (str): Single-line owner identity stamped onto written facts.
        token (int): Logical version token for ordering.
        captured_at (float): `time.monotonic()` timestamp when the turn was captured.
    """
    async with staging_locks.hold(key=scope):
        if cleared_since(scope=scope, started_at=captured_at):
            return
        await memory_db.upsert_pending(
            scope=scope,
            flavor=flavor_of(scope=scope),
            subject=subject,
            transcript=transcript,
            identity=identity,
            token=token,
        )


def enqueue_memory_update(turn: MemoryTurn, run: TurnRunner) -> None:
    """Schedules (or defers) one rendered-transcript update, backed by a `reply.db` row.

    Args:
        turn (MemoryTurn): Captured turn to review.
        run (TurnRunner): Turn body, carried through to the replay by the done-callback.
    """
    # Stamp before scheduling so a clear landing before the task starts aborts it.
    turn = turn.model_copy(update={"captured_at": time.monotonic()})
    running = _inflight_tasks.get(key=turn.scope)
    if running is not None and not running.done():
        by_subject = _pending_updates.setdefault(key=turn.scope, default={})
        superseded = by_subject.get(turn.subject)
        if superseded is not None and cleared_since(
            scope=turn.scope, started_at=superseded.captured_at
        ):
            # Superseded turn predates a clear; drop it unmerged and settle its reply.
            _release_pending_report(pending=superseded)
        elif superseded is not None:
            turn = turn.model_copy(
                update={
                    "transcript": _merged_payload(
                        newer=turn.transcript, older=superseded.transcript
                    ),
                    "report": _merged_report(
                        newer=turn.report, older=superseded.report, scope=turn.scope
                    ),
                    # Merged payload starts with the older round, whose forgets are not filed yet.
                    "retry": False,
                }
            )
        by_subject[turn.subject] = turn
        # Persist deferred turn for restart recovery; its newer token wins over the running turn.
        _spawn_db(
            coro=stage_turn(
                scope=turn.scope,
                subject=turn.subject,
                transcript=turn.transcript,
                identity=turn.identity,
                token=turn.token,
                captured_at=turn.captured_at,
            ),
            scope=turn.scope,
        )
        return
    task = asyncio.create_task(run(turn))
    _inflight_tasks.set(key=turn.scope, value=task)
    task.add_done_callback(
        lambda finished: _finish_memory_update(scope=turn.scope, task=finished, run=run)
    )


def drop_pending_updates(scope: str) -> None:
    """Drops every held turn for `scope` and tells each one's reply nothing was recorded.

    Args:
        scope (str): Memory scope key being cleared.
    """
    for dropped in (_pending_updates.pop(key=scope) or {}).values():
        _release_pending_report(pending=dropped)


async def report_writes(
    report: MemoryWriteReport, summary: MemoryWriteSummary, scope: str
) -> None:
    """Invokes `report` with `summary`, swallowing any Discord failure.

    Reports empty summaries too, settling any reply showing `正在整理記憶⋯`.

    Args:
        report (MemoryWriteReport): Callback reporting what the turn recorded.
        summary (MemoryWriteSummary): Recorded memory write summary.
        scope (str): Memory scope key for logging.
    """
    try:
        await report(summary)
    except Exception as exc:
        # Broad on purpose: the callback reaches Discord, and this runs in a background task
        # whose failure would otherwise be logged as the memory update having failed.
        logfire.warn(
            "Reporting a memory write back to the reply failed",
            scope=scope,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )


def _merged_payload(newer: str, older: str) -> str:
    """Merges `older`'s note rounds ahead of `newer`'s while keeping `newer`'s transcript."""
    transcript, rounds = parse_turn_payload(payload=newer)
    _, older_rounds = parse_turn_payload(payload=older)
    return render_turn_payload(
        transcript=transcript, rounds=_capped(rounds=(*older_rounds, *rounds))
    )


def _merged_report(
    newer: MemoryWriteReport | None, older: MemoryWriteReport | None, scope: str
) -> MemoryWriteReport | None:
    """Returns a callback that reports the merged outcome to both replies independently."""
    if newer is None or older is None:
        return newer or older

    async def both(summary: MemoryWriteSummary) -> None:
        """Reports one merged outcome to every reply whose notes went into it."""
        await report_writes(report=older, summary=summary, scope=scope)
        await report_writes(report=newer, summary=summary, scope=scope)

    return both


def _capped(rounds: tuple[NoteRound, ...]) -> tuple[NoteRound, ...]:
    """Deduplicates rounds (newest kept), caps at `MEMORY_MERGED_NOTES_MAX`, and folds them."""
    seen: tuple[set[str], set[str]] = (set(), set())
    newest_first: list[NoteRound] = []
    for round_ in reversed(rounds):
        kept: list[tuple[str, ...]] = []
        for notes, taken in zip(round_, seen, strict=True):
            picked: list[str] = []
            for note in reversed(notes):
                if note in taken or len(taken) >= MEMORY_MERGED_NOTES_MAX:
                    continue
                taken.add(note)
                picked.append(note)
            kept.append(tuple(reversed(picked)))
        newest_first.append((kept[0], kept[1]))
    merged: list[NoteRound] = []
    for remember, forget in reversed(newest_first):
        if merged and not forget:
            merged[-1] = ((*merged[-1][0], *remember), merged[-1][1])
        elif remember or forget:
            merged.append((remember, forget))
    return tuple(merged)


def _release_pending_report(pending: MemoryTurn) -> None:
    """Detaches an empty write report for a dropped pending turn so its reply settles."""
    if pending.report is None:
        return
    _spawn_db(
        coro=report_writes(
            report=pending.report, summary=MemoryWriteSummary(), scope=pending.scope
        ),
        scope=pending.scope,
    )


def _finish_memory_update(scope: str, task: asyncio.Task[None], run: TurnRunner) -> None:
    """Clears the in-flight slot, logs failures, and replays a pending update."""
    if _inflight_tasks.get(key=scope) is task:
        _inflight_tasks.pop(key=scope)
    if task.cancelled():
        # Cancelled tasks raise CancelledError from result(); skip replay on shutdown.
        return
    try:
        task.result()
    except Exception as exc:
        # Broad on purpose: `result()` re-raises whatever the whole pipeline raised
        # (LLM, network, file IO), and this runs on the event loop as a done-callback,
        # so anything escaping here is dropped by asyncio instead of handled.
        logfire.warn(
            "Background memory update failed",
            scope=scope,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
    by_subject = _pending_updates.get(key=scope)
    if not by_subject:
        _pending_updates.pop(key=scope)
        return
    # Replay one pending source at a time (FIFO); skip cleared turns so later ones do not stall.
    while by_subject:
        pending = by_subject.pop(next(iter(by_subject)))
        if not by_subject:
            _pending_updates.pop(key=scope)
        if cleared_since(scope=scope, started_at=pending.captured_at):
            # Best-effort DB cleanup for store-level clears; the tombstone owns the guarantee.
            _spawn_db(coro=memory_db.mark_done(scope=scope, token=pending.token), scope=scope)
            _release_pending_report(pending=pending)
            continue
        enqueue_memory_update(turn=pending, run=run)
        return
