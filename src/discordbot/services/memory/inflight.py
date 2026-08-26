"""One memory review job per scope, and the reply.db bookkeeping around it.

While a scope's update runs, later turns for that scope are held rather than dropped and
replayed one at a time afterwards. Within ONE conversation source only the newest is kept,
its history window already covering the earlier ones, and its memory notes merged in so
nothing a marker wrote is lost.

The turn's own body is not here: `enqueue_memory_update` takes it as `run`, and the replay
carries it on through the done-callback. That is what keeps this module below `pipeline.py`
rather than beside it — the queue owns when a turn runs and what happens to the ones it
displaces, and nothing about what a turn does.

The detached reply.db writes live here too, because the queue and the clear are the only
two things that touch them and they have to agree on one staging lock.
"""

import time
from typing import Any
import asyncio
from collections.abc import Callable, Awaitable, Coroutine

import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation

from discordbot.typings.memory import MemoryWriteSummary
from discordbot.services.memory import database as memory_db
from discordbot.utils.asyncio_locks import KeyedLockManager, LoopLocalRegistry
from discordbot.services.memory.store import flavor_of, cleared_since
from discordbot.services.memory.writer import (
    MemoryWriterAI,
    parse_turn_payload,
    render_turn_payload,
)
from discordbot.typings.context_budgets import MEMORY_MERGED_NOTES_MAX

# What a caller is handed once a turn's memory writes land. `services/` never composes what
# a user reads, so this reports the shape and lets the cog word it. In-memory only: a resumed
# turn runs after a restart, long past the reply it belonged to, and has no report to make.
type MemoryWriteReport = Callable[[MemoryWriteSummary], Awaitable[None]]


class MemoryTurn(BaseModel):
    """One turn's memory review request, whether it runs now or is held for replay.

    Attributes:
        scope: The memory scope this turn writes into.
        subject: The phase-1 directive naming the memory target.
        transcript: The rendered phase-1 input captured for the turn
            (already folds in the reply), so a replay needs no re-render.
        writer: The memory writing service to run the update with.
        identity: Single-line target identity `parse_identity` splits into the
            `owner_id` / `owner_name` stamped on every fact this scope writes.
        captured_at: `time.monotonic()` when the turn was captured, so a clear
            that lands before it runs can abort it via `cleared_since`.
        token: Process-local logical token persisted with the turn's DB
            row, reused on replay so the terminal write guards on the same id.
        report: Callback reporting what the turn recorded.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scope: str = Field(..., description="The memory scope this turn writes into.")
    subject: str = Field(..., description="The phase-1 directive naming the memory target.")
    transcript: str = Field(..., description="The rendered phase-1 input captured for the turn.")
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
        ..., description="Logical version token reused on replay for the DB row guard."
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


# The turn body `enqueue_memory_update` schedules. Injected rather than imported: it lives in
# `pipeline.py`, one layer up, and importing it here would close the loop. A coroutine
# function rather than any awaitable, because `asyncio.create_task` accepts nothing looser.
type TurnRunner = Callable[[MemoryTurn], Coroutine[Any, Any, None]]

# Process-level per-scope in-flight de-dupe; while one update runs, the skipped turns are
# held and replayed afterwards.
#
# The second key is the subject, which carries the source line, and it is a correctness
# boundary rather than bookkeeping: the scope is guild-independent, so a user active in two
# guilds at once would otherwise have one conversation's notes replayed under the other's
# source stamp, filing a `source_only` observation in a compartment the speaker never spoke
# in. Sources are replayed one after another, each keeping its own subject.
#
# Loop-local, like `regeneration`'s task registry: an `asyncio.Task` belongs to the loop that
# created it, so an entry surviving a loop change parks every later turn for that scope
# behind a task nothing on this loop can ever see finish.
_inflight_tasks: LoopLocalRegistry[str, asyncio.Task[None]] = LoopLocalRegistry()
_pending_updates: LoopLocalRegistry[str, dict[str, MemoryTurn]] = LoopLocalRegistry()

# Detached best-effort reply.db writes (the deferred-turn persist), held so the
# event loop keeps a strong reference until they finish; reset by the test fixture.
_db_tasks: set[asyncio.Task[None]] = set()

# A clear and the short reply.db staging transaction must not pass each other:
# otherwise an INSERT can commit after the clear's tombstone write. This is separate
# from the minutes-long file-write lock and is held only around reply.db staging
# or the clear's tombstone plus synchronous file removal.
staging_locks = KeyedLockManager[str]()


async def safe_db_write(coro: Awaitable[None]) -> None:
    """Awaits a best-effort reply.db write, swallowing any failure.

    Persistence is an augmentation layer (the opposite of `cogs/research/cog.py`,
    which lets DB errors raise into its run loop): a reply.db failure must never
    break the in-memory fire-and-forget memory pipeline.
    """
    try:
        await coro
    except Exception:
        logfire.warn("memory_job persistence write failed", _exc_info=True)


def _spawn_db(coro: Awaitable[None]) -> None:
    """Runs a detached best-effort DB write, tracked so it is not GC'd mid-flight."""
    task = asyncio.ensure_future(safe_db_write(coro=coro))
    _db_tasks.add(task)
    task.add_done_callback(_db_tasks.discard)


async def stage_turn(  # noqa: PLR0913 -- one row's columns plus the turn's capture time
    *, scope: str, subject: str, transcript: str, identity: str, token: int, captured_at: float
) -> None:
    """Stages one turn only in the memory lifetime that captured it.

    The short per-scope lock serializes staging with the clear's tombstone write.
    A row committed before a clear is scrubbed by that newer logical token. A row
    waiting while a clear is in progress sees its closing stamp before it can
    write, so it cannot leave a during-clear transcript for restart to resume.
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
    """Schedules (or defers) one rendered-transcript update, backed by a reply.db row.

    Args:
        turn: The captured turn to review.
        run: The turn body, carried through to the replay by the done-callback.
    """
    # Stamped here, not inside the worker: a clear landing between this call and
    # the task actually starting must still abort the turn, and a worker that
    # timed itself would read the clear as older than its own work and write on.
    turn = turn.model_copy(update={"captured_at": time.monotonic()})
    running = _inflight_tasks.get(key=turn.scope)
    if running is not None and not running.done():
        by_subject = _pending_updates.setdefault(key=turn.scope, default={})
        superseded = by_subject.get(turn.subject)
        if superseded is not None:
            turn = turn.model_copy(
                update={
                    "transcript": _merged_payload(
                        newer=turn.transcript, older=superseded.transcript
                    ),
                    "report": _merged_report(newer=turn.report, older=superseded.report),
                }
            )
        by_subject[turn.subject] = turn
        # Persist the deferred turn so a redeploy before it runs still resumes it.
        # Safe from a same-token race: this turn's worker only starts after the
        # in-flight one ends, long after this detached write lands, and it carries
        # a newer token than the running turn so newest-wins keeps it.
        _spawn_db(
            coro=stage_turn(
                scope=turn.scope,
                subject=turn.subject,
                transcript=turn.transcript,
                identity=turn.identity,
                token=turn.token,
                captured_at=turn.captured_at,
            )
        )
        return
    task = asyncio.create_task(run(turn))
    _inflight_tasks.set(key=turn.scope, value=task)
    task.add_done_callback(
        lambda finished: _finish_memory_update(scope=turn.scope, task=finished, run=run)
    )


def drop_pending_updates(scope: str) -> None:
    """Drops every held turn for a scope and tells each one's reply nothing was recorded.

    The clear's own step: dropping the retained transcripts now rather than waiting for the
    in-flight task to finish and discard them means `_finish_memory_update` then finds no
    pending turn and replays nothing.
    """
    for dropped in (_pending_updates.pop(key=scope) or {}).values():
        _release_pending_report(pending=dropped)


async def report_writes(report: MemoryWriteReport, summary: MemoryWriteSummary) -> None:
    """Hands the caller what this turn recorded, never letting the report cost the write.

    The write is already durable by the time this runs, so a Discord edit that fails, or a
    reply that has since been deleted, must not surface as a memory failure.

    An empty summary is reported like any other. It is the answer to a real question — the
    turn is over and kept nothing — and the caller is holding a reply that says the work is
    still going, so silence here reads as the work never having finished.
    """
    try:
        await report(summary)
    except Exception:
        # Broad on purpose: the callback reaches Discord, and this runs in a background task
        # whose failure would otherwise be logged as the memory update having failed.
        logfire.warn("Reporting a memory write back to the reply failed", _exc_info=True)


def _merged_payload(newer: str, older: str) -> str:
    """Carries a superseded turn's memory notes into the turn replacing it.

    Only the newest skipped turn is replayed, and for a TRANSCRIPT that is right: its history
    window already contains the earlier skipped turns, which is what makes one replay enough. A
    marker note is not in that window. It exists only in the reply that emitted it, so letting
    the newer payload simply overwrite the older one would drop it with nothing in the logs to
    say a note had ever been written.

    Merging is only ever within one conversation source; `_pending_updates` is keyed on the
    subject for that reason, and this function never sees two sources.
    """
    transcript, remember, forget = parse_turn_payload(payload=newer)
    _, older_remember, older_forget = parse_turn_payload(payload=older)
    return render_turn_payload(
        transcript=transcript,
        remember=_deduped(notes=(*older_remember, *remember)),
        forget=_deduped(notes=(*older_forget, *forget)),
    )


def _merged_report(
    newer: MemoryWriteReport | None, older: MemoryWriteReport | None
) -> MemoryWriteReport | None:
    """Answers both replies when one turn's notes are merged into another's.

    The counterpart to `_merged_payload`, and needed for the same reason: the superseded turn's
    notes ride on into the merged payload, so its reply is still waiting to be told what became
    of them. Overwriting the report the way the transcript is overwritten would leave that reply
    saying `正在整理記憶⋯` for good.

    Both are handed the same summary, which is the only honest one available: the two turns'
    notes were reviewed as one payload and there is nothing left that says which note came from
    which reply.

    Each is answered independently. A merge exists because the older reply is still standing
    there promising work, and that reply is also the likelier of the two to have been deleted
    under it, so a callback that raises must not take the other one's report with it.
    """
    if newer is None or older is None:
        return newer or older

    async def both(summary: MemoryWriteSummary) -> None:
        """Reports one merged outcome to every reply whose notes went into it."""
        await report_writes(report=older, summary=summary)
        await report_writes(report=newer, summary=summary)

    return both


def _deduped(notes: tuple[str, ...]) -> tuple[str, ...]:
    """Drops exact repeats, keeps writing order, and caps what a merge can accumulate.

    The per-reply cap the markers enforce does not survive a merge: each skipped turn stacks
    onto the pending payload, so without a bound here a long stretch of skipped turns grows
    the review request and the raw entry together. The OLDEST are dropped, since the newest
    notes are the ones the user is still in the middle of.
    """
    unique = tuple(dict.fromkeys(notes))
    return unique[-MEMORY_MERGED_NOTES_MAX:]


def _release_pending_report(pending: MemoryTurn) -> None:
    """Tells a dropped turn's reply that nothing was recorded, detached.

    A deferred turn is dropped rather than replayed when the scope was cleared under it, and
    its reply is still showing `正在整理記憶⋯`. Detached because both callers are synchronous —
    a done-callback and the clear orchestration — while the report reaches Discord.
    """
    if pending.report is None:
        return
    _spawn_db(coro=report_writes(report=pending.report, summary=MemoryWriteSummary()))


def _finish_memory_update(scope: str, task: asyncio.Task[None], run: TurnRunner) -> None:
    """Clears the in-flight slot, logs failures, and replays a pending update."""
    if _inflight_tasks.get(key=scope) is task:
        _inflight_tasks.pop(key=scope)
    if task.cancelled():
        # Cancelled (e.g. bot shutdown): reading result() would raise
        # CancelledError (a BaseException on 3.11+) out of this callback, and a
        # pre-shutdown turn is not worth replaying.
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
    # Oldest source first (dicts keep insertion order), and only one: each replay ends in
    # this same callback, which picks up the next one.
    pending = by_subject.pop(next(iter(by_subject)))
    if not by_subject:
        _pending_updates.pop(key=scope)
    if cleared_since(scope=scope, started_at=pending.captured_at):
        # The durable clear tombstone owns the privacy guarantee. This remains a
        # best-effort cleanup for store-level clears that only stamped the process.
        _spawn_db(coro=memory_db.mark_done(scope=scope, token=pending.token))
        _release_pending_report(pending=pending)
        return
    enqueue_memory_update(turn=pending, run=run)
