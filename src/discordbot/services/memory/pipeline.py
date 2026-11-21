"""Per-turn memory review pipeline and full-scope clear protocol.

Drives both user and server scopes through the same orchestration: `subject` names the review
target and `flavor_of(scope)` selects the writer's prompts.

Every disk write sits immediately after a `cleared_since` check with no `await` in between so
`clear_scope_memory` can abort in-flight updates without waiting on `scope_lock`.
"""

import asyncio
from datetime import datetime

import logfire
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.memory import MemoryWriteSummary

# Imported as a module so a test that swaps one database call on it reaches the calls here.
from discordbot.services.memory import database as memory_db
from discordbot.services.memory.store import (
    flavor_of,
    scope_lock,
    trim_detail,
    mark_cleared,
    cleared_since,
    append_raw_entry,
    delete_memory_files,
)
from discordbot.services.memory.writer import (
    MemoryWriterAI,
    MemoryObservation,
    parse_turn_payload,
    render_turn_payload,
    parse_subject_source,
    transcript_from_messages,
)
from discordbot.services.memory.inflight import (
    MemoryTurn,
    MemoryWriteReport,
    stage_turn,
    report_writes,
    safe_db_write,
    staging_locks,
    memory_semaphore,
    drop_pending_updates,
    enqueue_memory_update,
)
from discordbot.services.memory.git_history import memory_git
from discordbot.services.memory.raw_entries import (
    render_forget_requests,
    render_memory_observations,
)
from discordbot.services.memory.consolidation import consolidate_after_turn


async def _clear_scope_critical(scope: str) -> tuple[bool, bool]:
    """Completes the non-interruptible durable portion of one memory clear."""
    async with staging_locks.hold(key=scope):
        removed_job = await memory_db.clear_job(
            scope=scope, flavor=flavor_of(scope=scope), token=memory_db.new_token()
        )
        try:
            removed_files = delete_memory_files(scope=scope)
        finally:
            # Stamp even on cancellation so the next lifetime starts after tombstone and erasure.
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
    """Erases all pipeline state for `scope` across all four tiers.

    Neutralizes:
    - On-disk files under the scope directory.
    - Deferred in-process replay turns held in `inflight`.
    - The persisted `reply.db` row, replaced with a transcript-free `cleared` tombstone.
    - In-flight updates, aborted via opening and closing `mark_cleared` stamps.

    Does not acquire `scope_lock`; serializes only against `staging_locks`. Only turns captured
    after this coroutine returns belong to the new memory lifetime. On failure, the clear stamp
    and dropped replay are not rolled back.

    Args:
        scope (str): Memory scope key to erase.

    Returns:
        True when either files or a non-cleared DB row were removed.

    Raises:
        Exception: From the `memory_job` tombstone write (runs before file deletion, leaving
            files intact on failure, including when a newer DB token blocks the tombstone).
        OSError: From partial file deletion; retry is idempotent.
        asyncio.CancelledError: Re-raised after the shielded durable clear finishes and logs.
    """
    # Stamp first so in-flight updates abort before their next write without taking scope_lock.
    mark_cleared(scope=scope)
    # Drop deferred turns immediately and settle their replies rather than waiting for replay.
    drop_pending_updates(scope=scope)
    critical_task = asyncio.create_task(_clear_scope_critical(scope=scope))
    (removed_files, removed_job), caller_cancelled = await _await_clear_critical(
        task=critical_task
    )
    # Record both git commit and audit log before re-raising cancellation; neither awaits.
    memory_git.enqueue(scope=scope, reason="clear")
    # Log caller_cancelled too: nextcord ignores CancelledError so cancelled clears stay silent.
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

    No-ops when both `remember_notes` and `forget_notes` are empty; renders the transcript
    eagerly into a plain string.

    Args:
        scope (str): Memory scope key.
        subject (str): Phase-1 directive naming the review target.
        message_list (list[EasyInputMessageParam]): Input messages for the turn.
        full_reply (str): Assistant reply text for the turn.
        writer (MemoryWriterAI): Memory writing service running the review and consolidation.
        identity (str): Single-line owner identity stamped onto written facts.
        remember_notes (tuple[str, ...]): Inline `<write-memory>` or `<write-server-memory>`
            notes from the reply.
        forget_notes (tuple[str, ...]): Inline `<forget-memory>` notes from the reply.
        report (MemoryWriteReport | None): In-memory callback awaited once the turn's writes
            land; not persisted across restarts.
    """
    if not remember_notes and not forget_notes:
        return
    transcript = transcript_from_messages(message_list=message_list, full_reply=full_reply)
    enqueue_memory_update(
        turn=MemoryTurn(
            scope=scope,
            subject=subject,
            transcript=render_turn_payload(
                transcript=transcript, rounds=((remember_notes, forget_notes),)
            ),
            writer=writer,
            identity=identity,
            token=memory_db.new_token(),
            report=report,
        ),
        run=_run_memory_update,
    )


def resume_memory_update(  # noqa: PLR0913 -- mirrors a persisted row's columns
    scope: str,
    subject: str,
    transcript: str,
    writer: MemoryWriterAI,
    identity: str,
    token: int,
    status: memory_db.MemoryJobStatus,
) -> None:
    """Re-enqueues a persisted phase-1 turn on restart, reusing its stored token.

    Args:
        scope (str): Memory scope key.
        subject (str): Phase-1 directive naming the review target.
        transcript (str): Stored turn payload (transcript plus memory notes).
        writer (MemoryWriterAI): Memory writing service running the review and consolidation.
        identity (str): Single-line owner identity stamped onto written facts.
        token (int): Stored logical version token.
        status (memory_db.MemoryJobStatus): Stored row status (`'pending'` or `'failed'`, where
            `'failed'` marks the turn's single retry).
    """
    enqueue_memory_update(
        turn=MemoryTurn(
            scope=scope,
            subject=subject,
            transcript=transcript,
            writer=writer,
            identity=identity,
            token=token,
            retry=status == "failed",
        ),
        run=_run_memory_update,
    )


def _write_summary(
    observations: tuple[MemoryObservation, ...], forgotten: tuple[str, ...]
) -> MemoryWriteSummary:
    """Builds the raw-append `MemoryWriteSummary` from staged observations and filed forgets."""
    return MemoryWriteSummary(
        remembered=tuple(
            observation.summary_zh
            for observation in observations
            if observation.sharing != "source_only"
        ),
        private=sum(1 for observation in observations if observation.sharing == "source_only"),
        forgotten=forgotten,
    )


async def _run_memory_update(turn: MemoryTurn) -> None:
    """Stages `turn`, reviews its notes, consolidates if needed, and settles its report once."""
    settled = False

    async def settle(summary: MemoryWriteSummary) -> None:
        """Answers the caller's report once, whichever way this turn ends."""
        nonlocal settled
        if settled or turn.report is None:
            return
        settled = True
        await report_writes(report=turn.report, summary=summary, scope=turn.scope)

    try:
        if cleared_since(scope=turn.scope, started_at=turn.captured_at):
            # Drop pre-clear turns before staging so restart recovery never resurrects erased data.
            return
        await safe_db_write(
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
        if cleared_since(scope=turn.scope, started_at=turn.captured_at):
            # Clear landed during staging; drop the in-memory turn (tombstone owns DB ordering).
            return
        async with scope_lock(scope=turn.scope), memory_semaphore():
            forced = await _review_and_stage(
                turn=turn, report=settle if turn.report is not None else None
            )
            if forced is None:
                return
            await consolidate_after_turn(
                scope=turn.scope,
                forced=forced,
                started_at=turn.captured_at,
                writer=turn.writer,
                identity=turn.identity,
            )
    finally:
        # Settle any exit path that did not report (clear, error, or failed review) so replies
        # do not stay stuck on `正在整理記憶⋯`.
        await settle(MemoryWriteSummary())


async def _review_and_stage(  # noqa: C901 -- one review per round, and every write after an await needs its own clear check
    turn: MemoryTurn, report: MemoryWriteReport | None
) -> bool | None:
    """Reviews `turn`'s notes and stages survivors, returning forced consolidation or None.

    Caller must hold `scope_lock(turn.scope)` and a semaphore permit; `report` must swallow its own
    errors.
    """
    scope = turn.scope
    transcript, rounds = parse_turn_payload(payload=turn.transcript)
    # Server flavor or legacy subjects without a source line parse to None.
    source = parse_subject_source(subject=turn.subject)
    forget_notes: tuple[str, ...] = ()
    remember_notes: tuple[str, ...] = ()
    kept: list[MemoryObservation] = []
    reviewed = True
    # Process oldest round first so older remembers land in raw.md ahead of newer forgets.
    for remember, forget in rounds:
        # File forgets before the LLM review so a failed review never delays deletion.
        # On a retry, skip already-filed forgets until this attempt re-stages new observations.
        filing = () if turn.retry and not kept else forget
        forget_text = render_forget_requests(notes=filing, source=source)
        if forget_text and not cleared_since(scope=scope, started_at=turn.captured_at):
            append_raw_entry(scope=scope, entry_text=forget_text)
            await trim_detail(scope=scope, started_at=turn.captured_at)
        forget_notes += filing
        remember_notes += remember
        if not remember or not reviewed:
            # Skip remaining reviews after a failure, but keep filing forgets.
            continue
        draft = await turn.writer.evaluate(
            flavor=flavor_of(scope=scope),
            subject=turn.subject,
            transcript=transcript,
            notes=remember,
        )
        if cleared_since(scope=scope, started_at=turn.captured_at):
            # Cleared mid-review; drop results and best-effort mark_done (tombstone owns ordering).
            await safe_db_write(
                coro=memory_db.mark_done(scope=scope, token=turn.token), scope=scope
            )
            return None
        if draft is None:
            reviewed = False
            continue
        if draft.observations:
            # Stage even duplicate keys so restatements refresh last_confirmed and avoid aging out.
            append_raw_entry(
                scope=scope,
                entry_text=render_memory_observations(
                    observations=draft.observations, source=source
                ),
            )
            await trim_detail(scope=scope, started_at=turn.captured_at)
            kept.extend(draft.observations)
    if not reviewed and turn.retry:
        # Close the row on second failure so restarts do not loop on a permanent refusal.
        logfire.warn(
            "Memory note review failed again on its restart retry; notes dropped",
            scope=scope,
            flavor=flavor_of(scope=scope),
        )
        await safe_db_write(coro=memory_db.mark_done(scope=scope, token=turn.token), scope=scope)
        return True if forget_notes else None
    if not reviewed:
        # Park payload intact so the next restart retries once.
        logfire.warn(
            "Memory note review returned no draft; job parked for restart retry",
            scope=scope,
            flavor=flavor_of(scope=scope),
        )
        await safe_db_write(
            coro=memory_db.mark_failed(scope=scope, token=turn.token, error="evaluate failed"),
            scope=scope,
        )
        if report is not None and (kept or forget_notes):
            # Report already-staged rounds and forgets now; resumed retries carry no report.
            await report(_write_summary(observations=tuple(kept), forgotten=forget_notes))
        # Still force consolidation for filed forgets even though the review failed.
        return True if forget_notes else None
    if not kept and not forget_notes:
        logfire.debug("Memory notes survived nothing", scope=scope, notes=len(remember_notes))
        await safe_db_write(coro=memory_db.mark_done(scope=scope, token=turn.token), scope=scope)
        return None
    # Mark done before consolidation so a consolidation crash never re-runs phase-1 review.
    await safe_db_write(coro=memory_db.mark_done(scope=scope, token=turn.token), scope=scope)
    if report is not None:
        await report(_write_summary(observations=tuple(kept), forgotten=forget_notes))
    return bool(forget_notes)


async def safe_list_resumable(updated_before: datetime) -> list[memory_db.MemoryJob]:
    """Returns persisted `pending` and `failed` jobs for the restart sweep, or `[]` on DB error.

    Args:
        updated_before (datetime): Process start timestamp; see `database.py::list_resumable`.
    """
    try:
        return await memory_db.list_resumable(updated_before=updated_before)
    except Exception as exc:
        logfire.warn("memory_job resume read failed", error_type=type(exc).__name__, _exc_info=exc)
        return []
