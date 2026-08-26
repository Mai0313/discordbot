"""One reply turn's memory lifetime, from the notes it wrote to the clear that ends it.

The pipeline is keyed by an opaque scope (see ``store``), so the same orchestration drives
both per-user and per-server (bot self) memory. The flavor-specific bits are injected:
``subject`` names the memory target and ``writer`` carries the flavor's prompts.

Two things live here, and they are the two ends of one protocol rather than two subsystems
sharing a file. The turn reviews the reply's memory notes, stages what survives, and checks
``cleared_since`` before every write it makes; ``clear_scope_memory`` is what stamps that
flag, and its docstring is the only thing that says why each of those checks has to sit
immediately before its write with no ``await`` in between. Splitting them would file the
guard away from what it guards against.

Everything below is a module of its own: ``inflight`` holds the one-job-per-scope queue, the
process-wide semaphore and the reply.db bookkeeping, ``consolidation`` the compartment
fan-out, ``regeneration`` the from-scratch rebuild, ``tone`` the unpartitioned tone tier.
None of them imports this module back. The turn body ``inflight`` runs reaches it as an
argument rather than an import, and a staged turn meets the fan-out at exactly one call
(``consolidate_after_turn``), which is what let that cluster leave in #613.
"""

import asyncio

import logfire
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.memory import MemoryWriteSummary
from discordbot.services.memory import database as memory_db
from discordbot.services.memory.store import (
    flavor_of,
    scope_lock,
    mark_cleared,
    cleared_since,
    append_raw_entry,
    read_detail_tail,
    read_raw_entries,
    delete_memory_files,
)
from discordbot.services.memory.writer import (
    MemoryWriterAI,
    MemoryObservation,
    parse_turn_payload,
    render_turn_payload,
    parse_subject_source,
    render_forget_requests,
    transcript_from_messages,
    render_memory_observations,
    filter_duplicate_observations,
)
from discordbot.typings.context_budgets import MEMORY_DETAIL_CONTEXT_MAX_CHARS
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
    # Drops the retained transcripts now rather than waiting for the in-flight
    # task to finish and discard them; `inflight` then finds no pending turn and
    # replays nothing. Each dropped turn's reply is told, or it would keep saying
    # it was still working on memory this clear just erased.
    drop_pending_updates(scope=scope)
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
    enqueue_memory_update(
        turn=MemoryTurn(
            scope=scope,
            subject=subject,
            transcript=render_turn_payload(
                transcript=transcript, remember=remember_notes, forget=forget_notes
            ),
            writer=writer,
            identity=identity,
            token=memory_db.new_token(),
            report=report,
        ),
        run=_run_memory_update,
    )


def resume_memory_update(  # noqa: PLR0913 -- mirrors a persisted row's columns
    *, scope: str, subject: str, transcript: str, writer: MemoryWriterAI, identity: str, token: int
) -> None:
    """Re-enqueues a persisted phase-1 turn on restart, reusing its stored token."""
    enqueue_memory_update(
        turn=MemoryTurn(
            scope=scope,
            subject=subject,
            transcript=transcript,
            writer=writer,
            identity=identity,
            token=token,
        ),
        run=_run_memory_update,
    )


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


async def _run_memory_update(turn: MemoryTurn) -> None:
    """Reviews the turn's memory notes and, past the raw threshold, consolidates.

    The reply.db row is written `pending` at the top (awaited, before the lock) so
    a redeploy mid-review resumes this turn; it is marked `done` once phase-1
    is terminal (staged, no signal, all dupes, or cleared) and `failed` only
    when the LLM call itself fails, so the restart sweep retries just that case.
    Consolidation needs no DB row: `raw.md` is its durable, re-entrant queue.

    `turn.captured_at` is when the turn was scheduled, and every clear check runs
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
        if settled or turn.report is None:
            return
        settled = True
        await report_writes(report=turn.report, summary=summary)

    try:
        if cleared_since(scope=turn.scope, started_at=turn.captured_at):
            # Cleared between capture and start: staging the row would hand the
            # restart sweep the very conversation the clear erased, so drop the turn
            # before it writes anything at all.
            return
        await safe_db_write(
            coro=stage_turn(
                scope=turn.scope,
                subject=turn.subject,
                transcript=turn.transcript,
                identity=turn.identity,
                token=turn.token,
                captured_at=turn.captured_at,
            )
        )
        if cleared_since(scope=turn.scope, started_at=turn.captured_at):
            # The clear landed while the row was being written. Its durable tombstone
            # owns the DB ordering, so drop the in-memory turn before the review.
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
        # Nothing recorded is the honest answer for every path that reaches here without having
        # reported: a cleared scope wrote nothing, and neither did a turn that raised. The one
        # case where it is a guess is a review whose LLM call failed, which the restart sweep
        # will retry -- but a resumed turn carries no report (in-memory only, by design), so the
        # choice is between saying "kept nothing" now and leaving the reply promising work
        # forever. It says "kept nothing". The forget half of such a turn is reported for real
        # above, being durable before the evaluator ever runs.
        await settle(MemoryWriteSummary())


async def _review_and_stage(turn: MemoryTurn, report: MemoryWriteReport | None) -> bool | None:
    """Reviews one turn's notes and stages what survives, under the caller's scope lock.

    Returns whether consolidation should be FORCED (a forget is waiting), or None when the
    turn is finished and must not consolidate at all. Split out of `_run_memory_update` so the
    lock-holding half reads as one thing; the caller owns only the consolidation decision.
    """
    scope = turn.scope
    transcript, remember_notes, forget_notes = parse_turn_payload(payload=turn.transcript)
    # The subject's source line survives the memory_job round-trip, so a resumed
    # turn stamps the same source; a pre-source row (or the server flavor) parses
    # to None and renders without the source/sharing fields.
    source = parse_subject_source(subject=turn.subject)
    # Written before the evaluator runs, and deliberately not undone by its failure: a
    # forget needs no model, and making it wait behind one would let a failed call keep
    # the bot repeating what it was just asked to drop. A retried row therefore writes it
    # twice, which costs nothing: consolidation deletes the fact the first time and finds
    # nothing to delete the second.
    forget_text = render_forget_requests(notes=forget_notes, source=source)
    if forget_text and not cleared_since(scope=scope, started_at=turn.captured_at):
        append_raw_entry(scope=scope, entry_text=forget_text)
    draft = await turn.writer.evaluate(
        subject=turn.subject, transcript=transcript, notes=remember_notes
    )
    if cleared_since(scope=scope, started_at=turn.captured_at):
        # Cleared while this update was in flight; dropping the result beats
        # resurrecting deleted memory. The tombstone already owns the durable
        # ordering; this terminal write is only best-effort cleanup for a
        # process-local store clear.
        await safe_db_write(coro=memory_db.mark_done(scope=scope, token=turn.token))
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
        await safe_db_write(
            coro=memory_db.mark_failed(scope=scope, token=turn.token, error="evaluate failed")
        )
        if report is not None and forget_text:
            # The forget is durable regardless of the review, so the reply may say so. Its
            # remembered half is left empty rather than guessed at: the notes that would have
            # filled it are exactly what the failed call was reviewing.
            await report_writes(report=report, summary=MemoryWriteSummary(forgotten=forget_notes))
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
        await safe_db_write(coro=memory_db.mark_done(scope=scope, token=turn.token))
        return None
    # The turn is durable in raw.md now; record success before the (best-effort,
    # self-healing) consolidation so a consolidation crash never re-runs the review.
    await safe_db_write(coro=memory_db.mark_done(scope=scope, token=turn.token))
    if report is not None:
        await report_writes(
            report=report,
            summary=_write_summary(observations=deduped_observations, forgotten=forget_notes),
        )
    return bool(forget_text)


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
