"""Background orchestration for the two-phase memory pipeline.

The pipeline is keyed by an opaque scope (see ``store``), so the same
orchestration drives both per-user and per-server (bot self) memory. The
flavor-specific bits are injected: ``subject`` names the memory target and
``writer`` carries the flavor's prompts.
"""

import time
from typing import Literal
import asyncio
from datetime import UTC, datetime
from collections.abc import Callable, Awaitable

import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.memory import MemoryOwner, MemoryWriteSummary
from discordbot.services.memory import database as memory_db
from discordbot.typings.timeouts import MEMORY_CONSOLIDATE_TIMEOUT_SECONDS
from discordbot.utils.asyncio_locks import KeyedLockManager, LoopLocalRegistry, LoopLocalSemaphore
from discordbot.services.memory.facts import MemoryFlavor, parse_identity, sections_for_flavor
from discordbot.services.memory.store import (
    DM_COMPARTMENT,
    GLOBAL_COMPARTMENT,
    clear_raw,
    read_tone,
    clear_tone,
    read_facts,
    scope_lock,
    write_tone,
    mark_cleared,
    append_detail,
    cleared_since,
    raw_file_bytes,
    scope_owner_id,
    append_raw_entry,
    read_detail_tail,
    read_raw_entries,
    count_raw_entries,
    detail_file_bytes,
    list_compartments,
    prune_compartment,
    delete_memory_files,
    read_memory_document,
)
from discordbot.services.memory.deltas import (
    DeltaOutcome,
    today_utc,
    apply_deltas,
    sweep_stale_facts,
    partition_raw_entries,
    render_existing_facts,
    tone_evidence_from_raw,
    partition_forget_requests,
)
from discordbot.services.memory.writer import (
    MemoryWriterAI,
    MemoryObservation,
    ConsolidatedMemory,
    ConsolidationRequest,
    parse_turn_payload,
    render_turn_payload,
    parse_subject_source,
    render_forget_requests,
    transcript_from_messages,
    render_memory_observations,
    filter_duplicate_observations,
)
from discordbot.typings.context_budgets import (
    MEMORY_MERGED_NOTES_MAX,
    MEMORY_INJECTION_MAX_CHARS,
    MEMORY_INJECTION_WARN_CHARS,
    MEMORY_DETAIL_CONTEXT_MAX_CHARS,
)
from discordbot.services.memory.constants import (
    COMPACTION_TRIGGER_CHARS,
    MEMORY_GLOBAL_CONCURRENCY,
    RAW_CONSOLIDATION_MAX_BYTES,
    RAW_CONSOLIDATION_THRESHOLD,
    MEMORY_REGENERATION_COOLDOWN_SECONDS,
    MEMORY_CONSOLIDATION_COOLDOWN_SECONDS,
)
from discordbot.services.memory.git_history import memory_git

# What a caller is handed once a turn's memory writes land. `services/` never composes what
# a user reads, so this reports the shape and lets the cog word it. In-memory only: a resumed
# turn runs after a restart, long past the reply it belonged to, and has no report to make.
type MemoryWriteReport = Callable[[MemoryWriteSummary], Awaitable[None]]

# The ways a from-scratch rebuild can end, carried on `RegenerationReport.result`.
_RegenerationResult = Literal["regenerated", "no_evidence", "failed", "cooldown"]


class RegenerationReport(BaseModel):
    """What one from-scratch rebuild did, for a caller with no logfire to read.

    Attributes:
        result: How the rebuild ended.
        unreadable_removed: Fact files it destroyed that no reader could parse.
    """

    model_config = ConfigDict(frozen=True)

    result: _RegenerationResult = Field(..., description="How the rebuild ended.")
    unreadable_removed: int = Field(
        default=0, description="Fact files removed that no reader could parse."
    )


class _PendingMemoryUpdate(BaseModel):
    """The newest skipped update request, replayed once the in-flight task ends.

    Attributes:
        subject: The phase-1 extraction directive naming the memory target.
        transcript: The rendered phase-1 input captured for the skipped turn
            (already folds in the reply), so the replay needs no re-render.
        writer: The memory writing service to run the replayed update with.
        identity: Single-line target identity `parse_identity` splits into the
            `owner_id` / `owner_name` stamped on every fact this scope writes.
        captured_at: `time.monotonic()` when the turn was captured, so a clear
            that lands before the replay can abort it via `cleared_since`.
        token: Process-local logical token persisted with the deferred turn's DB
            row, reused on replay so the terminal write guards on the same id.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subject: str = Field(
        ..., description="The phase-1 extraction directive naming the memory target."
    )
    transcript: str = Field(
        ..., description="The rendered phase-1 input captured for the skipped turn."
    )
    writer: SkipValidation[MemoryWriterAI] = Field(
        ..., description="The memory writing service to run the replayed update with."
    )
    identity: str = Field(
        ...,
        description=(
            "Single-line target identity `parse_identity` splits into the `owner_id` / "
            "`owner_name` stamped on every fact this scope writes."
        ),
    )
    captured_at: float = Field(
        ...,
        description=(
            "`time.monotonic()` when the turn was captured, so a clear that lands "
            "before the replay can abort it via `cleared_since`."
        ),
    )
    token: int = Field(
        ..., description="Logical version token reused on replay for the DB row guard."
    )
    report: SkipValidation[MemoryWriteReport | None] = Field(
        default=None, description="Callback reporting what the replayed turn recorded."
    )


# Process-level per-scope in-flight de-dupe; while one update runs, the skipped turns are
# held and replayed afterwards. Within ONE conversation source only the newest is kept, its
# history window already covering the earlier ones, and its memory notes merged in so nothing
# a marker wrote is lost.
#
# The second key is the subject, which carries the source line, and it is a correctness
# boundary rather than bookkeeping: the scope is guild-independent, so a user active in two
# guilds at once would otherwise have one conversation's notes replayed under the other's
# source stamp, filing a `source_only` observation in a compartment the speaker never spoke
# in. Sources are replayed one after another, each keeping its own subject.
_inflight_tasks: dict[str, asyncio.Task[None]] = {}
_pending_updates: dict[str, dict[str, _PendingMemoryUpdate]] = {}
_inflight_loop: asyncio.AbstractEventLoop | None = None

# Per-scope consolidation attempt times for the cooldown; monotonic, so it does
# not need a loop-change reset. Tests clear it through the conftest fixture.
_last_consolidation: dict[str, float] = {}

# The exact header a tone note must lead with; the tier is injected on every reply,
# so anything else is a rewrite that did not land and must not be written.
_TONE_HEADER = "## 語氣偏好"

# Consecutive refused consolidation batches per (scope, compartment). A refusal keeps the
# raw batch for retry, which is right for a transient LLM failure and wrong for a
# mass-delete the model reproduces verbatim: that retries every cooldown forever while the
# scope's memory silently stops updating. Past the threshold the log escalates from warn to
# error, since CONTRIBUTING puts "an unexpected failure that lost a deliverable" at error.
_consecutive_rejections: dict[tuple[str, str], int] = {}
_MAX_QUIET_REJECTIONS = 3

# Per-scope regeneration attempt times, separate from the consolidation cooldown
# so a manual `/memory regenerate` never starves the automatic background
# consolidation or vice versa. Recorded at attempt time so failures cool down too.
_last_regeneration: dict[str, float] = {}

# Per-scope in-flight regeneration tasks so a manual rebuild runs in the
# background without blocking the command, and a second request while one is
# still running cannot double-schedule the rebuild. Kept separate
# from `_inflight_tasks` because regeneration is a distinct, user-triggered job.
_regeneration_tasks: LoopLocalRegistry[str, asyncio.Task[RegenerationReport]] = LoopLocalRegistry()

# Process-wide semaphore capping concurrent background memory updates so a busy server
# cannot fan out unbounded LLM work; shared across flavors and rebuilt per loop. The cap
# is read at build time so a test that lowers MEMORY_GLOBAL_CONCURRENCY first still applies.
_memory_semaphore_holder = LoopLocalSemaphore(capacity_provider=lambda: MEMORY_GLOBAL_CONCURRENCY)


def _memory_semaphore() -> asyncio.Semaphore:
    """Returns the process-wide semaphore, rebuilt when the event loop changes."""
    return _memory_semaphore_holder.get()


# Detached best-effort reply.db writes (the deferred-turn persist), held so the
# event loop keeps a strong reference until they finish; reset by the test fixture.
_db_tasks: set[asyncio.Task[None]] = set()

# A clear and the short reply.db staging transaction must not pass each other:
# otherwise an INSERT can commit after the clear's tombstone write. This is separate
# from the minutes-long file-write lock and is held only around reply.db staging
# or the clear's tombstone plus synchronous file removal.
_staging_locks = KeyedLockManager[str]()


def flavor_of(scope: str) -> memory_db.MemoryJobFlavor:
    """Maps a scope to its persisted memory flavor (`server_scope` carries a '/')."""
    return "server" if "/" in scope else "user"


async def _safe(coro: Awaitable[None]) -> None:
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
    task = asyncio.ensure_future(_safe(coro=coro))
    _db_tasks.add(task)
    task.add_done_callback(_db_tasks.discard)


async def _stage_turn(  # noqa: PLR0913 -- one row's columns plus the turn's capture time
    *, scope: str, subject: str, transcript: str, identity: str, token: int, captured_at: float
) -> None:
    """Stages one turn only in the memory lifetime that captured it.

    The short per-scope lock serializes staging with the clear's tombstone write.
    A row committed before a clear is scrubbed by that newer logical token. A row
    waiting while a clear is in progress sees its closing stamp before it can
    write, so it cannot leave a during-clear transcript for restart to resume.
    """
    async with _staging_locks.hold(key=scope):
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


async def _clear_scope_critical(scope: str) -> tuple[bool, bool]:
    """Completes the non-interruptible durable portion of one memory clear."""
    async with _staging_locks.hold(key=scope):
        removed_job = await memory_db.clear_job(
            scope=scope, flavor=flavor_of(scope=scope), token=memory_db.new_token()
        )
        try:
            removed_files = delete_memory_files(scope=scope)
        finally:
            # A caller may cancel while this task is running, but the next memory
            # lifetime still begins only after the tombstone and file pass finish.
            mark_cleared(scope=scope)
    return removed_files, removed_job


async def _await_clear_critical(
    task: asyncio.Task[tuple[bool, bool]],
) -> tuple[tuple[bool, bool], bool]:
    """Drains the clear task and reports whether its caller requested cancellation."""
    caller_cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            caller_cancelled = True
    return task.result(), caller_cancelled


async def clear_scope_memory(scope: str) -> bool:
    """Erases everything the pipeline holds for a scope, on the user's request.

    Four tiers go at once, because leaving any one of them behind rebuilds the
    memory the user just asked to remove: the on-disk files, the deferred replay
    still holding a pre-clear transcript in this process, the persisted phase-1
    row (whose transcript is that same conversation, and which the restart sweep
    would otherwise resume), and any in-flight update, which aborts itself once
    `mark_cleared` has stamped the scope. The DB tier remains as a scrubbed
    `cleared` tombstone so a stale staging commit cannot recreate the transcript.

    Completion is the deliberate user-visible boundary. The opening stamp
    protects the erase; a closing stamp after file removal also rejects turns
    captured while the reply.db deletion was suspended. Only turns captured
    after this coroutine returns belong to the new memory lifetime.

    The scope lock is deliberately NOT taken, since waiting for it would park a
    user-facing command behind a minutes-long consolidation. Every FILE write
    sits immediately after a `cleared_since` guard with no `await` in between,
    so an in-flight task cannot interleave one past the stamp. The much shorter
    per-scope staging lock serializes reply.db staging with the tombstone write:
    an earlier INSERT is scrubbed by the newer tombstone, and a turn captured
    during the clear waits for the closing stamp then writes nothing.

    Raises:
        Exception: From the `memory_job` tombstone write, the one memory DB call that is
            not best-effort: swallowing it would leave a resumable row that
            resurrects the memory on the next restart. It runs before the file
            deletion, so that failure alone leaves every tier in place — including
            the case where a row newer than this clear makes the tombstone
            unwritable, which `clear_job` refuses rather than erasing behind it.
        OSError: From the file deletion, which walks the tiers one at a time and
            can therefore stop part way. A clear is idempotent, so the caller
            recovers by retrying rather than by claiming either outcome.
        asyncio.CancelledError: Re-raised for a caller that cancelled, after the
            durable work has drained AND after the erase has been recorded. What
            was erased does not shrink because the caller went away, so neither
            does the record of it.

    Note that neither failure rolls back the stamp or the dropped replay: a
    failed clear still aborts the turns that were in flight for this scope. That
    is deliberate — the alternative is letting a turn the user tried to erase
    survive because the erase failed — but it is why the caller must not report a
    failure as "nothing happened".

    Returns:
        True when anything was actually removed.
    """
    # Stamped before anything else so a row write already in flight sees the
    # clear on its own re-check. Ordering the stamp first is what makes the
    # tombstone below safe without draining or taking the minutes-long scope lock.
    mark_cleared(scope=scope)
    # Drops the retained transcript now rather than waiting for the in-flight
    # task to finish and discard it; `_finish_memory_update` then finds no
    # pending turn and replays nothing. Each dropped turn's reply is told, or it
    # would keep saying it was still working on memory this clear just erased.
    for dropped in _pending_updates.pop(scope, {}).values():
        _release_pending_report(pending=dropped)
    critical_task = asyncio.create_task(_clear_scope_critical(scope=scope))
    (removed_files, removed_job), caller_cancelled = await _await_clear_critical(
        task=critical_task
    )
    # Both traces are taken before cancellation propagates. The shielded task has already
    # finished by now, so a cancelled clear erased exactly as much as an uncancelled one;
    # re-raising first left an irreversible user-driven erase with nothing on record that it
    # happened. Neither call awaits, so nothing can interleave before the re-raise below.
    #
    # Commits the deletion so the working tree stops carrying it, which is all this can
    # do: the commits before it still hold the content, and no reachable-object pruning
    # changes that. Local history outliving a clear is a recorded decision on #408.
    memory_git.enqueue(scope=scope, reason="clear")
    # A user-driven, irreversible erase of their own data, and the only trace of it outside
    # reply.db's own `cleared` row, since nothing about it is visible in the files afterwards.
    # `caller_cancelled` earns its place because that path is silent at the Discord end: the
    # button deferred before the clear began and nextcord's dispatcher catches only `Exception`,
    # so a cancelled clear leaves the confirmation prompt simply never edited — no error, no
    # confirmation, and nothing anywhere but this line saying the wipe happened.
    logfire.info(
        "Cleared personal memory on request",
        scope=scope,
        removed_files=removed_files,
        removed_job=removed_job,
        caller_cancelled=caller_cancelled,
    )
    if caller_cancelled:
        raise asyncio.CancelledError
    return removed_files or removed_job


def schedule_memory_update(  # noqa: PLR0913 -- flavor (scope/subject/identity) plus the turn payload
    scope: str,
    subject: str,
    message_list: list[EasyInputMessageParam],
    full_reply: str,
    writer: MemoryWriterAI,
    identity: str,
    remember_notes: tuple[str, ...],
    forget_notes: tuple[str, ...] = (),
    report: MemoryWriteReport | None = None,
) -> None:
    """Starts a background memory update without delaying the reply path.

    `remember_notes` and `forget_notes` are the inline memory markers the answer model wrote in
    the reply it just gave. A turn that carried none does nothing at all: no row, no model call,
    no background task. That is the normal case now, and it is the whole saving over the
    extraction pass this replaced, which ran on every single reply to find out whether there was
    anything to find.

    The transcript is rendered eagerly here (pure, sub-ms, already past the reply)
    so the persisted job and the in-memory replay both carry a plain string and
    `_run_memory_update` re-renders nothing.

    `report`, when given, is awaited once the turn's writes land, so the caller can tell the
    user what was recorded. It is deliberately in-memory only and is NOT persisted with the
    row: a resumed turn runs after a restart, long after the reply it belonged to, and
    reporting onto it then would edit a message the conversation has moved past.
    """
    if not remember_notes and not forget_notes:
        return
    transcript = transcript_from_messages(message_list=message_list, full_reply=full_reply)
    _enqueue_memory_update(
        scope=scope,
        subject=subject,
        transcript=render_turn_payload(
            transcript=transcript, remember=remember_notes, forget=forget_notes
        ),
        writer=writer,
        identity=identity,
        token=memory_db.new_token(),
        report=report,
    )


def resume_memory_update(  # noqa: PLR0913 -- mirrors a persisted row's columns
    *, scope: str, subject: str, transcript: str, writer: MemoryWriterAI, identity: str, token: int
) -> None:
    """Re-enqueues a persisted phase-1 turn on restart, reusing its stored token."""
    _enqueue_memory_update(
        scope=scope,
        subject=subject,
        transcript=transcript,
        writer=writer,
        identity=identity,
        token=token,
    )


def _enqueue_memory_update(  # noqa: PLR0913 -- flavor (scope/subject/identity) plus the rendered turn
    scope: str,
    subject: str,
    transcript: str,
    writer: MemoryWriterAI,
    identity: str,
    token: int,
    report: MemoryWriteReport | None = None,
) -> None:
    """Schedules (or defers) one rendered-transcript update, backed by a reply.db row."""
    global _inflight_loop  # noqa: PLW0603 -- process task de-dupe
    loop = asyncio.get_running_loop()
    if _inflight_loop is not loop:
        _inflight_tasks.clear()
        _pending_updates.clear()
        _inflight_loop = loop
    # Stamped here, not inside the worker: a clear landing between this call and
    # the task actually starting must still abort the turn, and a worker that
    # timed itself would read the clear as older than its own work and write on.
    captured_at = time.monotonic()
    running = _inflight_tasks.get(scope)
    if running is not None and not running.done():
        by_subject = _pending_updates.setdefault(scope, {})
        superseded = by_subject.get(subject)
        if superseded is not None:
            transcript = _merged_payload(newer=transcript, older=superseded.transcript)
            report = _merged_report(newer=report, older=superseded.report)
        by_subject[subject] = _PendingMemoryUpdate(
            subject=subject,
            transcript=transcript,
            report=report,
            writer=writer,
            identity=identity,
            captured_at=captured_at,
            token=token,
        )
        # Persist the deferred turn so a redeploy before it runs still resumes it.
        # Safe from a same-token race: this turn's worker only starts after the
        # in-flight one ends, long after this detached write lands, and it carries
        # a newer token than the running turn so newest-wins keeps it.
        _spawn_db(
            coro=_stage_turn(
                scope=scope,
                subject=subject,
                transcript=transcript,
                identity=identity,
                token=token,
                captured_at=captured_at,
            )
        )
        return
    task = asyncio.create_task(
        _run_memory_update(
            scope=scope,
            subject=subject,
            transcript=transcript,
            writer=writer,
            identity=identity,
            token=token,
            captured_at=captured_at,
            report=report,
        )
    )
    _inflight_tasks[scope] = task
    task.add_done_callback(lambda finished: _finish_memory_update(scope=scope, task=finished))


def _write_summary(
    observations: tuple[MemoryObservation, ...], forgotten: tuple[str, ...]
) -> MemoryWriteSummary:
    """Builds what the user is told about this turn's memory work.

    Reported at raw-append time rather than after consolidation: that is the point the turn's
    work is durable, and consolidation can be minutes away. So the wording the caller builds
    from this has to be "taken down", never "stored as a fact" -- the merge that turns these
    into facts can still fold or drop any of them.
    """
    return MemoryWriteSummary(
        remembered=tuple(
            observation.summary_zh
            for observation in observations
            if observation.sharing != "source_only"
        ),
        private=sum(1 for observation in observations if observation.sharing == "source_only"),
        forgotten=forgotten,
    )


async def _report_writes(report: MemoryWriteReport, summary: MemoryWriteSummary) -> None:
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
    """
    if newer is None or older is None:
        return newer or older

    async def both(summary: MemoryWriteSummary) -> None:
        """Reports one merged outcome to every reply whose notes went into it."""
        await older(summary)
        await newer(summary)

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


def _release_pending_report(pending: _PendingMemoryUpdate) -> None:
    """Tells a dropped turn's reply that nothing was recorded, detached.

    A deferred turn is dropped rather than replayed when the scope was cleared under it, and
    its reply is still showing `正在整理記憶⋯`. Detached because both callers are synchronous —
    a done-callback and the clear orchestration — while the report reaches Discord.
    """
    if pending.report is None:
        return
    _spawn_db(coro=_report_writes(report=pending.report, summary=MemoryWriteSummary()))


def _finish_memory_update(scope: str, task: asyncio.Task[None]) -> None:
    """Clears the in-flight slot, logs failures, and replays a pending update."""
    if _inflight_tasks.get(scope) is task:
        _inflight_tasks.pop(scope, None)
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
    by_subject = _pending_updates.get(scope)
    if not by_subject:
        _pending_updates.pop(scope, None)
        return
    # Oldest source first (dicts keep insertion order), and only one: each replay ends in
    # this same callback, which picks up the next one.
    pending = by_subject.pop(next(iter(by_subject)))
    if not by_subject:
        _pending_updates.pop(scope, None)
    if cleared_since(scope=scope, started_at=pending.captured_at):
        # The durable clear tombstone owns the privacy guarantee. This remains a
        # best-effort cleanup for store-level clears that only stamped the process.
        _spawn_db(coro=memory_db.mark_done(scope=scope, token=pending.token))
        _release_pending_report(pending=pending)
        return
    _enqueue_memory_update(
        scope=scope,
        subject=pending.subject,
        transcript=pending.transcript,
        writer=pending.writer,
        identity=pending.identity,
        token=pending.token,
        report=pending.report,
    )


async def _run_memory_update(  # noqa: PLR0913 -- schedule_memory_update's flavor + payload, and one early exit per way a turn can end
    scope: str,
    subject: str,
    transcript: str,
    writer: MemoryWriterAI,
    identity: str,
    token: int,
    captured_at: float,
    report: MemoryWriteReport | None = None,
) -> None:
    """Reviews the turn's memory notes and, past the raw threshold, consolidates.

    The reply.db row is written `pending` at the top (awaited, before the lock) so
    a redeploy mid-extraction resumes this turn; it is marked `done` once phase-1
    is terminal (extracted, no signal, all dupes, or cleared) and `failed` only
    when the LLM call itself fails, so the restart sweep retries just that case.
    Consolidation needs no DB row: `raw.md` is its durable, re-entrant queue.

    `captured_at` is when the turn was scheduled, and every clear check runs
    against it rather than a worker-local clock, so a clear that lands while this
    turn is still queued aborts it too.

    The caller's `report` is answered exactly once, and the `finally` is what guarantees it:
    the reply already carries `正在整理記憶⋯`, and this function has more ways to end than
    ways to record something — two clear checks before the lock, an unread exception out of
    the review, a shutdown cancellation. Making each of those remember to report is how one
    of them ends up not doing it and leaves a reply promising work that finished long ago.
    """
    settled = False

    async def settle(summary: MemoryWriteSummary) -> None:
        """Answers the caller's report once, whichever way this turn ends."""
        nonlocal settled
        if settled or report is None:
            return
        settled = True
        await _report_writes(report=report, summary=summary)

    try:
        if cleared_since(scope=scope, started_at=captured_at):
            # Cleared between capture and start: staging the row would hand the
            # restart sweep the very conversation the clear erased, so drop the turn
            # before it writes anything at all.
            return
        await _safe(
            coro=_stage_turn(
                scope=scope,
                subject=subject,
                transcript=transcript,
                identity=identity,
                token=token,
                captured_at=captured_at,
            )
        )
        if cleared_since(scope=scope, started_at=captured_at):
            # The clear landed while the row was being written. Its durable tombstone
            # owns the DB ordering, so drop the in-memory turn before extraction.
            return
        async with scope_lock(scope=scope), _memory_semaphore():
            forced = await _review_and_stage(
                scope=scope,
                subject=subject,
                payload=transcript,
                writer=writer,
                token=token,
                captured_at=captured_at,
                report=settle if report is not None else None,
            )
            if forced is None or not _should_consolidate(scope=scope, forced=forced):
                return
            await _consolidate_now(
                scope=scope, started_at=captured_at, writer=writer, identity=identity
            )
    finally:
        # Nothing recorded is the honest answer for every path that reaches here without having
        # reported: a cleared scope wrote nothing, and neither did a turn that raised. The one
        # case where it is a guess is a review whose LLM call failed, which the restart sweep
        # will retry -- but a resumed turn carries no report (in-memory only, by design), so the
        # choice is between saying "kept nothing" now and leaving the reply promising work
        # forever. It says "kept nothing". The forget half of such a turn is reported for real
        # above, being durable before the evaluator ever runs.
        await settle(MemoryWriteSummary())


async def _review_and_stage(  # noqa: PLR0913 -- the turn's identity and payload, and one early exit per way a review can end
    scope: str,
    subject: str,
    payload: str,
    writer: MemoryWriterAI,
    token: int,
    captured_at: float,
    report: MemoryWriteReport | None,
) -> bool | None:
    """Reviews one turn's notes and stages what survives, under the caller's scope lock.

    Returns whether consolidation should be FORCED (a forget is waiting), or None when the
    turn is finished and must not consolidate at all. Split out of `_run_memory_update` so the
    lock-holding half reads as one thing; the caller owns only the consolidation decision.
    """
    transcript, remember_notes, forget_notes = parse_turn_payload(payload=payload)
    # The subject's source line survives the memory_job round-trip, so a resumed
    # turn stamps the same source; a pre-source row (or the server flavor) parses
    # to None and renders without the source/sharing fields.
    source = parse_subject_source(subject=subject)
    # Written before the evaluator runs, and deliberately not undone by its failure: a
    # forget needs no model, and making it wait behind one would let a failed call keep
    # the bot repeating what it was just asked to drop. A retried row therefore writes it
    # twice, which costs nothing: consolidation deletes the fact the first time and finds
    # nothing to delete the second.
    forget_text = render_forget_requests(notes=forget_notes, source=source)
    if forget_text and not cleared_since(scope=scope, started_at=captured_at):
        append_raw_entry(scope=scope, entry_text=forget_text)
    draft = await writer.evaluate(subject=subject, transcript=transcript, notes=remember_notes)
    if cleared_since(scope=scope, started_at=captured_at):
        # Cleared while this update was in flight; dropping the result beats
        # resurrecting deleted memory. The tombstone already owns the durable
        # ordering; this terminal write is only best-effort cleanup for a
        # process-local store clear.
        await _safe(coro=memory_db.mark_done(scope=scope, token=token))
        return None
    if draft is None:
        # The LLM path itself failed: keep the row (payload intact) so the
        # restart sweep retries it, no extra timeout needed. The cause detail is
        # already logged upstream; this line adds the scope attribution.
        logfire.warn(
            "Memory note review returned no draft; job parked for restart retry",
            scope=scope,
            flavor=flavor_of(scope=scope),
        )
        await _safe(coro=memory_db.mark_failed(scope=scope, token=token, error="evaluate failed"))
        if report is not None and forget_text:
            # The forget is durable regardless of the review, so the reply may say so. Its
            # remembered half is left empty rather than guessed at: the notes that would have
            # filled it are exactly what the failed call was reviewing.
            await _report_writes(report=report, summary=MemoryWriteSummary(forgotten=forget_notes))
        # The forget above is already durable and has nothing to do with the review that
        # failed, so it still gets the immediate pass it was written for. Without this it
        # would wait for an unrelated turn to push the backlog over threshold, and the bot
        # would go on repeating what it was asked to drop -- exactly what writing it first
        # was meant to prevent.
        return True if forget_text else None
    recent_detail = read_detail_tail(scope=scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS)
    deduped_observations = filter_duplicate_observations(
        observations=draft.observations,
        existing_text="\n\n".join((read_raw_entries(scope=scope), recent_detail)),
        source=source,
    )
    if deduped_observations:
        append_raw_entry(
            scope=scope,
            entry_text=render_memory_observations(
                observations=deduped_observations, source=source
            ),
        )
    elif not forget_text:
        logfire.debug(
            "Memory notes survived nothing",
            scope=scope,
            notes=len(remember_notes),
            candidates=len(draft.observations),
        )
        await _safe(coro=memory_db.mark_done(scope=scope, token=token))
        return None
    # The turn is durable in raw.md now; record success before the (best-effort,
    # self-healing) consolidation so a consolidation crash never re-runs the review.
    await _safe(coro=memory_db.mark_done(scope=scope, token=token))
    if report is not None:
        await _report_writes(
            report=report,
            summary=_write_summary(observations=deduped_observations, forgotten=forget_notes),
        )
    return bool(forget_text)


async def _consolidate_now(
    scope: str, started_at: float, writer: MemoryWriterAI, identity: str
) -> None:
    """Runs the fan-out immediately, stamping the cooldown first.

    Recorded at attempt time, not success time, so repeated LLM failures are rate-limited by
    the same cooldown instead of retrying every turn. Assumes the caller holds the scope lock
    and a semaphore permit, like `_consolidate_locked` itself.
    """
    _last_consolidation[scope] = time.monotonic()
    await _consolidate_locked(scope=scope, started_at=started_at, writer=writer, identity=identity)


async def safe_list_resumable() -> list[memory_db.MemoryJob]:
    """Returns the persisted `pending` and `failed` jobs for the restart sweep, best-effort.

    Wrapped so a reply.db read failure degrades to "nothing to resume" instead of
    breaking `on_ready`; the in-memory pipeline keeps working regardless.
    """
    try:
        return await memory_db.list_resumable()
    except Exception:
        logfire.warn("memory_job resume read failed", _exc_info=True)
        return []


def needs_consolidation(scope: str) -> bool:
    """Public sync pre-check for the boot sweep so it only spawns over-threshold scopes.

    A cheap file read (no lock), used to avoid queuing a per-scope task on the
    global semaphore just to discover it is under threshold; `consolidate_if_needed`
    re-checks under the lock, which stays the authority.
    """
    return _should_consolidate(scope=scope)


async def consolidate_if_needed(scope: str, writer: MemoryWriterAI, identity: str) -> None:
    """Consolidates a scope whose raw backlog is over threshold; best-effort, self-logging.

    The boot-sweep entry point: `_consolidate_locked` is private and assumes the
    scope lock and the semaphore permit are held, so this wrapper takes both and
    re-checks the threshold under the lock. It swallows its own errors (a background
    digest must never surface), so the caller just spawns it.
    """
    try:
        async with scope_lock(scope=scope), _memory_semaphore():
            if not _should_consolidate(scope=scope):
                return
            _last_consolidation[scope] = time.monotonic()
            await _consolidate_locked(
                scope=scope, started_at=time.monotonic(), writer=writer, identity=identity
            )
    except Exception:
        logfire.warn("Background memory consolidation sweep failed", scope=scope, _exc_info=True)


def _should_consolidate(scope: str, forced: bool = False) -> bool:
    """Whether the raw backlog warrants a consolidation right now.

    `forced` is a batch carrying a forget request. It skips BOTH gates below rather than just
    the cooldown: the entry count is checked first and a lone forget is one entry against a
    threshold of two, so bypassing the cooldown alone would still leave the user waiting for
    their next message before the bot stopped repeating what they asked it to drop.
    """
    if forced:
        return True
    if raw_file_bytes(scope=scope) >= RAW_CONSOLIDATION_MAX_BYTES:
        # A verbose burst consolidates regardless of the cooldown so the raw
        # file cannot sit large until the timer expires.
        return True
    if count_raw_entries(scope=scope) < RAW_CONSOLIDATION_THRESHOLD:
        return False
    last_attempt = _last_consolidation.get(scope)
    if last_attempt is None or cleared_since(scope=scope, started_at=last_attempt):
        # No prior attempt, or the memory was cleared since it: the fresh
        # post-clear state deserves a prompt first consolidation instead of
        # waiting out a cooldown that belonged to the wiped memory.
        return True
    return time.monotonic() - last_attempt >= MEMORY_CONSOLIDATION_COOLDOWN_SECONDS


async def _consolidate_locked(
    scope: str, started_at: float, writer: MemoryWriterAI, identity: str
) -> None:
    """Fans one raw batch out over the scope's compartments, applying each one's deltas.

    Ordering is load-bearing. `global` runs first, so every later compartment can be
    handed its facts as read-only reference and neither restates them nor silently
    contradicts them — the one thing the old whole-file rewrite got for free by seeing
    everything at once. Each call sees only the evidence routed to the compartment it
    writes, which is what makes the boundary structural; the tone note, the one tier that
    is genuinely cross-compartment, is written afterwards by its own call.

    The whole fan-out sits inside the caller's scope lock and semaphore permit and is
    bounded by one timeout, so the worst-case lock hold stays what it is today instead
    of multiplying by the number of compartments. Compartments run sequentially for the
    same reason: proxy load per scope stays exactly as it was.

    The raw batch is retired only when every compartment applied. A retry re-runs the
    ones that already landed, which is safe because a delta is an upsert keyed by an id
    the model echoes back and, failing that, by the evidence keys the fact carries.
    """
    flavor = flavor_of(scope=scope)
    owner = parse_identity(identity=identity, fallback_owner_id=scope_owner_id(scope=scope))
    raw_entries = read_raw_entries(scope=scope)
    buckets = partition_raw_entries(raw_text=raw_entries, flavor=flavor)
    # Forget requests are a separate pass over the same batch: they are copied into every
    # compartment their speaker could read from, and each of those calls may only delete.
    forget_buckets = partition_forget_requests(
        raw_text=raw_entries, compartments=tuple(list_compartments(scope=scope))
    )
    detail_tail = read_detail_tail(scope=scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS)
    # The detail window is up to MEMORY_DETAIL_CONTEXT_MAX_CHARS and used to be sliced
    # rather than parsed; splitting it into observation blocks is a real stall on a
    # heavy scope, and this runs on the same loop as the reply path. Pure function, no
    # shared state, so a thread costs nothing — and the await is safe here because every
    # write still sits immediately after its own `cleared_since` guard downstream.
    detail_buckets = await asyncio.to_thread(
        partition_raw_entries, raw_text=detail_tail, flavor=flavor
    )
    today = datetime.now(UTC).date().isoformat()
    compartments = _compartments_to_run(scope=scope, buckets=buckets)
    global_reference = ""
    try:
        async with asyncio.timeout(MEMORY_CONSOLIDATE_TIMEOUT_SECONDS):
            # Forgets first, so a fact this batch also re-confirms is not deleted right after
            # being written, and so the observation pass sees the tree the deletions left.
            if not await _apply_forget_buckets(
                scope=scope,
                flavor=flavor,
                owner=owner,
                started_at=started_at,
                writer=writer,
                buckets=forget_buckets,
                today=today,
            ):
                return
            for compartment in compartments:
                if compartment != GLOBAL_COMPARTMENT and not global_reference:
                    # Read from disk rather than from this run: when the batch carried no
                    # cross-server evidence there was no global call to take it from, and
                    # a guild compartment still must not restate what is already shared.
                    global_reference = render_existing_facts(
                        facts=read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)
                    )
                if cleared_since(scope=scope, started_at=started_at):
                    return
                outcome = await _consolidate_compartment(
                    scope=scope,
                    compartment=compartment,
                    flavor=flavor,
                    owner=owner,
                    started_at=started_at,
                    writer=writer,
                    request_parts=_CompartmentInput(
                        raw_entries=buckets.get(compartment, ""),
                        recent_detail=detail_buckets.get(compartment, ""),
                        global_reference=global_reference,
                        today=today,
                    ),
                )
                if outcome is None or not outcome.applied:
                    # Keep the whole batch so the next run retries it; a partially
                    # applied fan-out is fine to replay, an unread bucket is not.
                    return
                if compartment == GLOBAL_COMPARTMENT:
                    global_reference = render_existing_facts(
                        facts=read_facts(scope=scope, compartment=compartment)
                    )
    except TimeoutError:
        logfire.warn(
            "Memory consolidation fan-out timed out; keeping raw batch",
            scope=scope,
            compartments=len(compartments),
        )
        return
    if cleared_since(scope=scope, started_at=started_at):
        return
    await _update_tone_note(
        scope=scope,
        flavor=flavor,
        started_at=started_at,
        writer=writer,
        raw_entries=raw_entries,
        today=today,
    )
    # Every compartment ran, so age the ones this batch did not touch too: a guild the
    # user has stopped visiting otherwise keeps its `recent` facts forever and hands them
    # back on their next visit, which is the aging the whole-file rewrite used to do for
    # free. Synchronous and after the clear guard, like every other write here.
    for compartment in list_compartments(scope=scope):
        sweep_stale_facts(scope=scope, compartment=compartment, today=today_utc())
    _report_injection_size(scope=scope, flavor=flavor)
    # The consumed batch's content is preserved in the cold-tier detail file; every
    # failure path above returns before this, so it can never retire an unread bucket.
    append_detail(scope=scope, text=raw_entries)
    clear_raw(scope=scope)
    # Best-effort and deliberately fire-and-forget: the worker takes this same scope
    # lock, so it commits once the caller releases it and never sees a half-written batch.
    memory_git.enqueue(scope=scope, reason="update")


async def _apply_forget_buckets(  # noqa: PLR0913 -- the scope's identity plus the buckets, their stamp and the LLM handle
    scope: str,
    flavor: MemoryFlavor,
    owner: MemoryOwner,
    started_at: float,
    writer: MemoryWriterAI,
    buckets: dict[str, str],
    today: str,
) -> bool:
    """Runs one deletion-only pass per compartment a forget was copied into.

    Returns False when the caller must keep the raw batch for a retry, on the same terms as
    the observation fan-out: an unread bucket is not safe to retire.

    Every call here is `deletes_only`. That is the whole reason forgets are partitioned
    separately rather than folded into the observation buckets: the flag is per call, so a
    turn that both remembered and forgot something would otherwise hand a possibly-private
    sentence to a call that is allowed to write.
    """
    for compartment, forget_text in sorted(buckets.items()):
        if cleared_since(scope=scope, started_at=started_at):
            return False
        outcome = await _consolidate_compartment(
            scope=scope,
            compartment=compartment,
            flavor=flavor,
            owner=owner,
            started_at=started_at,
            writer=writer,
            deletes_only=True,
            request_parts=_CompartmentInput(
                raw_entries=forget_text, recent_detail="", global_reference="", today=today
            ),
        )
        if outcome is None or not outcome.applied:
            return False
    return True


class _CompartmentInput(BaseModel):
    """The per-compartment half of a consolidation request, before the store is read.

    Split out so `_consolidate_locked` can build the parts that differ per compartment
    without `_consolidate_compartment` growing a dozen positional arguments.
    """

    model_config = ConfigDict(frozen=True)

    raw_entries: str = Field(..., description="This compartment's share of the raw batch.")
    recent_detail: str = Field(..., description="Cold evidence filtered to this compartment.")
    global_reference: str = Field(..., description="Global facts already stored, or empty.")
    today: str = Field(..., description="ISO date for dating and aging.")


async def _consolidate_compartment(  # noqa: PLR0913 -- one compartment's identity plus the shared stamp, the LLM handle and the write gate
    scope: str,
    compartment: str,
    flavor: MemoryFlavor,
    owner: MemoryOwner,
    started_at: float,
    writer: MemoryWriterAI,
    request_parts: _CompartmentInput,
    deletes_only: bool = False,
) -> DeltaOutcome | None:
    """Runs and applies one compartment's consolidation; None means the LLM path failed.

    This call only ever sees the evidence routed to the compartment it is writing, which
    is what makes "a guild-locked observation cannot reach `global/`" structural rather
    than a rule the prompt asks the model to follow. The tone note, which is genuinely
    cross-compartment, is therefore NOT written here — see `_update_tone_note`.
    """
    existing = read_facts(scope=scope, compartment=compartment)
    rendered = render_existing_facts(facts=existing)
    is_global = compartment == GLOBAL_COMPARTMENT
    result = await writer.consolidate(
        request=ConsolidationRequest(
            compartment_note=_compartment_note(compartment=compartment, flavor=flavor),
            allowed_sections=tuple(sorted(sections_for_flavor(flavor=flavor))),
            existing_facts=rendered,
            existing_tone="",
            raw_entries=request_parts.raw_entries,
            recent_detail=request_parts.recent_detail,
            tone_evidence="",
            global_reference="" if is_global else request_parts.global_reference,
            today=request_parts.today,
            compact=len(rendered) > COMPACTION_TRIGGER_CHARS,
            emit_tone=False,
        )
    )
    if result is None:
        logfire.warn(
            "Memory consolidation LLM call failed; keeping raw batch",
            scope=scope,
            compartment=compartment,
            raw_entries=count_raw_entries(scope=scope),
        )
        return None
    if cleared_since(scope=scope, started_at=started_at):
        # Checked immediately before the first write, with no await in between, so an
        # in-flight clear can never be overtaken by this batch.
        return None
    outcome = apply_deltas(
        scope=scope,
        compartment=compartment,
        flavor=flavor,
        deltas=result.deltas,
        owner=owner,
        allow_mass_delete=False,
        deletes_only=deletes_only,
    )
    if not outcome.applied:
        _record_rejection(
            scope=scope, compartment=compartment, outcome=outcome, stored=len(existing)
        )
        return outcome
    _consecutive_rejections.pop((scope, compartment), None)
    swept = sweep_stale_facts(scope=scope, compartment=compartment, today=today_utc())
    logfire.debug(
        "Memory compartment consolidated",
        scope=scope,
        compartment=compartment,
        created=outcome.created,
        updated=outcome.updated,
        deleted=outcome.deleted,
        dropped=outcome.dropped,
        swept=swept,
    )
    return outcome


def _record_rejection(scope: str, compartment: str, outcome: DeltaOutcome, stored: int) -> None:
    """Logs a refused batch, escalating once refusals stop looking transient.

    A refusal is only a retry if the next run would decide differently. An LLM failure
    would; a mass deletion the model re-derives from the same unchanged inputs would not,
    and that scope then burns a consolidation call every cooldown while its memory quietly
    stops moving. The count is what tells an operator which of the two they are looking at.
    """
    # Keyed on the compartment as well: a scope with several compartments would
    # otherwise have one compartment's success reset another's stuck counter, and the
    # escalation this exists for would never fire.
    count = _consecutive_rejections.get((scope, compartment), 0) + 1
    _consecutive_rejections[(scope, compartment)] = count
    log = logfire.error if count >= _MAX_QUIET_REJECTIONS else logfire.warn
    log(
        "Memory consolidation batch refused; keeping raw batch",
        scope=scope,
        compartment=compartment,
        reason=outcome.rejected,
        existing_facts=stored,
        consecutive=count,
    )


async def _update_tone_note(  # noqa: PLR0913 -- the scope's identity plus the batch, its stamp, and the LLM handle
    scope: str,
    flavor: MemoryFlavor,
    started_at: float,
    writer: MemoryWriterAI,
    raw_entries: str,
    today: str,
) -> None:
    """Rewrites the per-user tone note from the WHOLE batch, in its own call.

    Tone is the one tier that is cross-server safe by construction, so it is the one
    thing that must not be partitioned: nearly half of all observations are
    `source_only`, and a tone note fed only the `global` bucket would simply stop
    updating for those conversations.

    That is exactly why it gets its own call rather than riding on the `global`
    compartment's. A compartment call sees only the evidence routed to the compartment
    it writes, which is what makes "a guild-locked observation cannot reach `global/`"
    structural; handing that same call the unpartitioned tone evidence would have
    demoted the boundary back to a rule the prompt asks the model to follow. Here the
    deltas are discarded by CODE — this call cannot write a fact anywhere, whatever it
    returns — so the unpartitioned input is safe by the same structural argument.

    Best-effort throughout: the note is a small always-read tier and the next
    consolidation repairs a bad write, so a failure never touches the raw batch.
    """
    if flavor != "user":
        return
    tone_evidence = tone_evidence_from_raw(raw_text=raw_entries)
    if not tone_evidence:
        # No tone signal in this batch is the normal case, and an empty output must
        # never delete the note; only the evidence-complete rebuild may do that.
        return
    result = await writer.consolidate(
        request=ConsolidationRequest(
            compartment_note="the user's persona-independent tone note, read in every conversation",
            allowed_sections=(),
            existing_facts="",
            existing_tone=read_tone(scope=scope),
            raw_entries="",
            recent_detail="",
            tone_evidence=tone_evidence,
            global_reference="",
            today=today,
            compact=False,
            emit_tone=True,
        )
    )
    if result is None or cleared_since(scope=scope, started_at=started_at):
        return
    _write_tone_result(scope=scope, tone_markdown=result.tone_markdown)


def _compartments_to_run(scope: str, buckets: dict[str, str]) -> list[str]:
    """Returns the compartments this run touches, `global` first.

    Only compartments the batch actually routed evidence to: a call with an empty bucket
    has nothing to consolidate. `global` leads because every later compartment is handed
    its facts as read-only reference, so it must be up to date before they run.
    """
    ordered = [GLOBAL_COMPARTMENT] if buckets.get(GLOBAL_COMPARTMENT) else []
    ordered.extend(
        compartment for compartment in sorted(buckets) if compartment != GLOBAL_COMPARTMENT
    )
    return ordered


def _compartment_note(compartment: str, flavor: MemoryFlavor) -> str:
    """Describes, in plain English, who may read the compartment being written.

    Handed to the consolidation prompt so the model's own sense of what belongs here
    matches the directory it is writing into. It is guidance, not enforcement: the
    partition above already decided what evidence this call can see.
    """
    if flavor == "server":
        return "this server's own community memory; every member of this server can read it"
    if compartment == GLOBAL_COMPARTMENT:
        return "cross-server safe memory; readable in every server and DM this user takes part in"
    if compartment == DM_COMPARTMENT:
        return "private memory; readable only in this user's own direct messages with the bot"
    return f"memory readable only inside Discord server {compartment.removeprefix('g/')}"


def _report_injection_size(scope: str, flavor: MemoryFlavor) -> None:
    """Logs when a scope's injectable document approaches or passes the hard cap.

    A post-write backstop, not a budget: the read path already stops rendering at the
    cap, so this exists to tell the operator that the prompt-side sizing stopped
    working. Nothing is deleted here, which is what keeps the cap from fighting the
    next consolidation over facts it would immediately write back.
    """
    compartments = list_compartments(scope=scope)
    if not compartments:
        return
    # The owner's own DM reads every compartment at once, so it is the only combination
    # that can overflow while each individual reading context stays inside the cap.
    widest = len(
        read_memory_document(
            scope=scope,
            compartments=compartments,
            flavor=flavor,
            max_chars=MEMORY_INJECTION_MAX_CHARS * 4,
        )
    )
    if widest > MEMORY_INJECTION_MAX_CHARS:
        logfire.error(
            "Memory exceeds the injectable size cap; older facts are being dropped on read",
            scope=scope,
            chars=widest,
            cap=MEMORY_INJECTION_MAX_CHARS,
        )
    elif widest > MEMORY_INJECTION_WARN_CHARS:
        logfire.warn(
            "Memory is approaching the injectable size cap",
            scope=scope,
            chars=widest,
            cap=MEMORY_INJECTION_MAX_CHARS,
        )


def _write_tone_result(scope: str, tone_markdown: str) -> None:
    """Persists a tone-note call's output when it is acceptable for this scope.

    User scopes only, and only a note starting with the exact `## 語氣偏好` header;
    an empty or malformed output never deletes the existing note — the tier is
    best-effort and the next consolidation repairs it. Only `_rebuild_tone_note`,
    which saw the whole evidence corpus, may clear it.
    """
    if flavor_of(scope=scope) != "user":
        return
    if not _tone_is_well_formed(tone_markdown=tone_markdown):
        return
    write_tone(scope=scope, content=tone_markdown)


def _tone_is_well_formed(tone_markdown: str) -> bool:
    """Whether a tone note carries the exact header the injected tier is contracted to."""
    return tone_markdown.startswith(_TONE_HEADER)


def regeneration_has_evidence(scope: str) -> bool:
    """Whether any cold-tier evidence exists for a from-scratch rebuild.

    Mirrors the evidence guard inside `regenerate_main_memory` cheaply (no full
    window read), so the command can surface "no observations yet" up front
    instead of scheduling a background rebuild that would silently do nothing.
    """
    return bool(read_raw_entries(scope=scope)) or detail_file_bytes(scope=scope) > 0


def regeneration_on_cooldown(scope: str) -> bool:
    """Whether a recent regeneration attempt blocks another one right now."""
    last_attempt = _last_regeneration.get(scope)
    if last_attempt is None or cleared_since(scope=scope, started_at=last_attempt):
        # A clear since the last attempt wiped the memory that cooldown
        # belonged to; the fresh post-clear state deserves a prompt rebuild.
        return False
    return time.monotonic() - last_attempt < MEMORY_REGENERATION_COOLDOWN_SECONDS


def schedule_memory_regeneration(scope: str, writer: MemoryWriterAI, identity: str) -> bool:
    """Starts a background main-memory rebuild without blocking the command.

    Returns False when a rebuild is already in flight for this scope (so the
    caller can report "still rebuilding" instead of double-scheduling the
    rebuild); True when a fresh background task was started.
    """
    running = _regeneration_tasks.get(key=scope)
    if running is not None and not running.done():
        return False
    task = asyncio.create_task(
        regenerate_main_memory(scope=scope, writer=writer, identity=identity)
    )
    _regeneration_tasks.set(key=scope, value=task)
    task.add_done_callback(
        lambda finished: _finish_memory_regeneration(scope=scope, task=finished)
    )
    return True


def _finish_memory_regeneration(scope: str, task: asyncio.Task[RegenerationReport]) -> None:
    """Clears the in-flight slot and logs failures of a background rebuild."""
    if _regeneration_tasks.get(key=scope) is task:
        _regeneration_tasks.pop(key=scope)
    if task.cancelled():
        # Cancelled (e.g. bot shutdown): reading result() would raise
        # CancelledError out of this callback, and an aborted rebuild leaves the
        # existing memory untouched, so there is nothing to recover.
        return
    try:
        task.result()
    except Exception as exc:
        # Broad on purpose: this is a done-callback boundary, so anything the
        # rebuild raised must be swallowed here or asyncio drops it silently.
        logfire.error(
            "Background memory regeneration crashed",
            scope=scope,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )


async def regenerate_main_memory(
    scope: str, writer: MemoryWriterAI, identity: str
) -> RegenerationReport:
    """Rebuilds every compartment from cold-tier evidence alone.

    The existing facts are deliberately NOT fed to the model: the rebuild distills the
    detail tail window plus any unconsumed raw entries from scratch, e.g. to redo an
    unsatisfying consolidation with another model. Facts the rebuild did not re-emit are
    then deleted, which is the one path allowed to lose most of a compartment at once:
    replacing the whole set is what it is for, and that exemption is also what let the
    #408 compartment migration run through here instead of needing a writer of its own.
    `scripts/regen_memories.py` drives the same path offline.

    On an LLM failure the compartment is left exactly as it was, and the raw batch is
    retired only when every compartment rebuilt. The report carries what the run removed
    unread whichever way it ended, so a rebuild that gave up on its third compartment
    still accounts for what the first two destroyed.
    """
    started_at = time.monotonic()
    unreadable_removed = 0
    async with scope_lock(scope=scope), _memory_semaphore():
        if regeneration_on_cooldown(scope=scope):
            # Invocations queued behind a held lock all pass the command-level
            # cooldown check before the first one stamps the attempt; the
            # re-check under the lock keeps the per-scope limit on the rewrite.
            return RegenerationReport(result="cooldown")
        flavor = flavor_of(scope=scope)
        owner = parse_identity(identity=identity, fallback_owner_id=scope_owner_id(scope=scope))
        raw_entries = read_raw_entries(scope=scope)
        recent_detail = read_detail_tail(scope=scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS)
        # Detail entries are retired raw entries verbatim with the same
        # `## <ISO timestamp>` headers, so the combined corpus (oldest first)
        # slots into the raw-entries consolidation input unchanged.
        evidence = "\n\n".join(part for part in (recent_detail, raw_entries) if part)
        if not evidence:
            return RegenerationReport(result="no_evidence")
        # Recorded at attempt time, not success time, so repeated LLM failures
        # are rate-limited by the same cooldown.
        _last_regeneration[scope] = time.monotonic()
        buckets = partition_raw_entries(raw_text=evidence, flavor=flavor)
        today = datetime.now(UTC).date().isoformat()
        # Every compartment that has evidence, plus every one that still has files, so a
        # compartment whose evidence is gone is emptied rather than left stale.
        compartments = _compartments_to_rebuild(scope=scope, buckets=buckets)
        try:
            # Bounded as a whole like the incremental fan-out, and for the same reason: the
            # individual calls carry no deadline of their own (`constants.py` has why), so
            # this is the only thing standing between a stuck rebuild and a scope lock held
            # for as long as the client will keep one compartment's request alive.
            async with asyncio.timeout(MEMORY_CONSOLIDATE_TIMEOUT_SECONDS):
                for compartment in compartments:
                    raw_bucket = buckets.get(compartment, "")
                    if not raw_bucket and not read_facts(scope=scope, compartment=compartment):
                        # A leftover directory with nothing to distil and nothing to keep:
                        # the model would be handed an empty corpus and could only answer
                        # with an empty batch, so the prune alone reaches the same state.
                        # It also removes the emptied directory, which is what stops the
                        # leftover costing another call — and another way to fail the
                        # compartments that do have something — on every later rebuild.
                        unreadable_removed += _prune_rebuilt_compartment(
                            scope=scope, compartment=compartment, keep=set()
                        )
                        continue
                    result = await writer.consolidate(
                        request=ConsolidationRequest(
                            compartment_note=_compartment_note(
                                compartment=compartment, flavor=flavor
                            ),
                            allowed_sections=tuple(sorted(sections_for_flavor(flavor=flavor))),
                            existing_facts="",
                            existing_tone="",
                            raw_entries=raw_bucket,
                            recent_detail="",
                            tone_evidence="",
                            global_reference="",
                            today=today,
                            compact=True,
                            emit_tone=False,
                        )
                    )
                    if result is None:
                        # The LLM path logs the cause but not the scope, and the command
                        # already told the user a rebuild was scheduled, so this is its
                        # only attribution.
                        logfire.warn(
                            "Memory regeneration LLM call failed; memory left untouched",
                            scope=scope,
                            compartment=compartment,
                        )
                        return RegenerationReport(
                            result="failed", unreadable_removed=unreadable_removed
                        )
                    if cleared_since(scope=scope, started_at=started_at):
                        return RegenerationReport(
                            result="failed", unreadable_removed=unreadable_removed
                        )
                    unreadable_removed += _replace_compartment(
                        scope=scope,
                        compartment=compartment,
                        flavor=flavor,
                        owner=owner,
                        result=result,
                    )
                await _reapply_forgets(
                    scope=scope,
                    flavor=flavor,
                    owner=owner,
                    started_at=started_at,
                    writer=writer,
                    evidence=evidence,
                    today=today,
                )
                await _rebuild_tone_note(
                    scope=scope,
                    flavor=flavor,
                    started_at=started_at,
                    writer=writer,
                    evidence=evidence,
                    today=today,
                )
        except TimeoutError:
            logfire.warn(
                "Memory regeneration timed out", scope=scope, compartments=len(compartments)
            )
            return RegenerationReport(result="failed", unreadable_removed=unreadable_removed)
        _report_injection_size(scope=scope, flavor=flavor)
        if raw_entries:
            # The rebuild consumed the raw batch; retire it to the cold tier
            # exactly like a consolidation so it cannot be re-ingested.
            append_detail(scope=scope, text=raw_entries)
            clear_raw(scope=scope)
        memory_git.enqueue(scope=scope, reason="rebuild")
        return RegenerationReport(result="regenerated", unreadable_removed=unreadable_removed)


async def _reapply_forgets(  # noqa: PLR0913 -- the scope's identity plus the corpus, its stamp, and the LLM handle
    scope: str,
    flavor: MemoryFlavor,
    owner: MemoryOwner,
    started_at: float,
    writer: MemoryWriterAI,
    evidence: str,
    today: str,
) -> None:
    """Re-runs every forget request in the corpus against the freshly rebuilt facts.

    A rebuild derives facts from evidence rather than from the current facts, and the
    observation a forget was aimed at is still sitting in `detail.md` verbatim: consolidation
    retires the raw batch there and never prunes it. So the rebuild re-creates exactly what
    the user asked to have removed, and `/memory regenerate` quietly undoes every forget they
    ever asked for.

    Replaying the requests afterwards fixes that without weakening anything: each runs as its
    own `deletes_only` call, the same shape the incremental path uses, so the forget's own
    sentence still cannot be written anywhere. Feeding the requests INTO the rebuild instead
    would have put a possibly-private sentence in front of a call whose whole job is creating
    facts, which is the one thing `deletes_only` exists to prevent.

    Best-effort: the rebuild has already landed by this point, and a failure here leaves a
    resurrected fact rather than a broken store. The next forget removes it again.
    """
    await _apply_forget_buckets(
        scope=scope,
        flavor=flavor,
        owner=owner,
        started_at=started_at,
        writer=writer,
        buckets=partition_forget_requests(
            raw_text=evidence, compartments=tuple(list_compartments(scope=scope))
        ),
        today=today,
    )


async def _rebuild_tone_note(  # noqa: PLR0913 -- the scope's identity plus the corpus, its stamp, and the LLM handle
    scope: str,
    flavor: MemoryFlavor,
    started_at: float,
    writer: MemoryWriterAI,
    evidence: str,
    today: str,
) -> None:
    """Rebuilds the tone note from the whole evidence corpus, in its own call.

    Unlike an incremental consolidation — whose empty tone output only means "no tone
    signal in this batch" — this pass saw everything, so no signal anywhere means a
    surviving note is stale and would keep injecting a preference the evidence no longer
    supports. This is the only path allowed to delete the note.
    """
    if flavor != "user":
        return
    tone_evidence = tone_evidence_from_raw(raw_text=evidence)
    result = (
        None
        if not tone_evidence
        else await writer.consolidate(
            request=ConsolidationRequest(
                compartment_note=(
                    "the user's persona-independent tone note, read in every conversation"
                ),
                allowed_sections=(),
                existing_facts="",
                existing_tone="",
                raw_entries="",
                recent_detail="",
                tone_evidence=tone_evidence,
                global_reference="",
                today=today,
                compact=False,
                emit_tone=True,
            )
        )
    )
    if cleared_since(scope=scope, started_at=started_at):
        return
    if result is None or not result.tone_markdown:
        clear_tone(scope=scope)
        return
    _write_tone_result(scope=scope, tone_markdown=result.tone_markdown)


def _compartments_to_rebuild(scope: str, buckets: dict[str, str]) -> list[str]:
    """Returns every compartment a rebuild touches, `global` first.

    Compartments that still hold files but have no surviving evidence are included so
    the rebuild empties them; leaving them alone would keep pre-rebuild facts visible
    alongside the new ones with no evidence behind them. Touching one does not always
    mean consolidating it: an entry that turns out to hold neither evidence nor a
    readable fact is pruned without a model call (`regenerate_main_memory` has the why).
    """
    ordered = [GLOBAL_COMPARTMENT]
    ordered.extend(
        compartment
        for compartment in sorted({*buckets, *list_compartments(scope=scope)})
        if compartment != GLOBAL_COMPARTMENT
    )
    return ordered


def _replace_compartment(
    scope: str,
    compartment: str,
    flavor: MemoryFlavor,
    owner: MemoryOwner,
    result: ConsolidatedMemory,
) -> int:
    """Replaces a compartment's contents with a from-scratch rebuild's facts.

    Applies the batch first and prunes afterwards, so a fact the rebuild kept is never
    momentarily absent. What survives is decided by the ids the batch actually WROTE, not
    by what is left on disk: a rebuild says "this fact is gone" by simply not mentioning
    it, so comparing against the post-apply state would only re-delete what the batch
    already deleted and leave every stale fact standing.

    The mass-delete guard is off here for the reason it was skipped by the old whole-file
    rebuild: replacing the entire set is what this path is for.
    """
    outcome = apply_deltas(
        scope=scope,
        compartment=compartment,
        flavor=flavor,
        deltas=result.deltas,
        owner=owner,
        allow_mass_delete=True,
    )
    return _prune_rebuilt_compartment(
        scope=scope, compartment=compartment, keep=set(outcome.written)
    )


def _prune_rebuilt_compartment(scope: str, compartment: str, keep: set[str]) -> int:
    """Reduces a rebuilt compartment to `keep`, reporting what it left and what it took.

    The prune reads the directory rather than the facts read back from it, so a file no
    reader can parse cannot outlive a rebuild that reports the compartment replaced
    (`prune_compartment` carries the why). What it could not account for is reported
    here instead, since the store never removes a file it did not write.

    What it DID remove unread is reported here too, and returned for the offline
    rebuild's own report. Renaming a section or a durability value makes every fact
    carrying the old one unparsable, so the next rebuild of a scope drops all of them in
    one pass; a run that says only what it spared reads as one that destroyed nothing.

    Shared with the skip path, which prunes a compartment it never handed to the model,
    so a file the store never wrote is named there on the same terms — and so is the
    unreadable one, which is the ONLY thing that path ever removes: a compartment reaches
    it precisely when nothing in it could be read.
    """
    pruned = prune_compartment(scope=scope, compartment=compartment, keep=keep)
    if pruned.unaccounted:
        logfire.warn(
            "Memory rebuild left files it cannot account for",
            scope=scope,
            compartment=compartment,
            files=pruned.unaccounted,
        )
    if pruned.unreadable:
        logfire.warn(
            "Memory rebuild removed fact files it could not read",
            scope=scope,
            compartment=compartment,
            files=pruned.unreadable,
        )
    return len(pruned.unreadable)
