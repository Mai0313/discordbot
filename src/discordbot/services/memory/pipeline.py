"""Per-scope orchestration for the memory pipeline.

The pipeline is keyed by an opaque scope (see ``store``), so the same
orchestration drives both per-user and per-server (bot self) memory. The
flavor-specific bits are injected: ``subject`` names the memory target and
``writer`` carries the flavor's prompts.

What lives here is the part that binds the rest together: one turn's review, the
consolidation fan-out that turn can trigger, and the clear that has to neutralise every
tier at once. The three subsystems that touch almost nothing else moved out beside it in
#607 — ``inflight`` holds the one-job-per-scope queue and the reply.db bookkeeping,
``regeneration`` the from-scratch rebuild, ``tone`` the unpartitioned tone tier.

They do not all sit on the same side of this module. ``inflight`` and ``tone`` are below it
and import nothing from here — the turn body ``inflight`` runs reaches it as an argument
rather than an import, which is what keeps that edge one-way. ``regeneration`` is ABOVE it
and takes the compartment machinery (``CompartmentInput``, ``compartment_request``,
``apply_forget_buckets``, ``global_first``, ``memory_semaphore``, ``report_injection_size``)
from here, so nothing in this module may import it back.
"""

import time
import asyncio
from datetime import UTC, datetime

import logfire
from pydantic import Field, BaseModel, ConfigDict
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.memory import MemoryOwner, MemoryWriteSummary
from discordbot.services.memory import database as memory_db
from discordbot.typings.timeouts import MEMORY_CONSOLIDATE_TIMEOUT_SECONDS
from discordbot.utils.asyncio_locks import LoopLocalSemaphore
from discordbot.services.memory.tone import update_tone_note
from discordbot.services.memory.facts import MemoryFlavor, parse_identity, sections_for_flavor
from discordbot.services.memory.store import (
    DM_COMPARTMENT,
    GLOBAL_COMPARTMENT,
    clear_raw,
    flavor_of,
    read_facts,
    scope_lock,
    mark_cleared,
    append_detail,
    cleared_since,
    raw_file_bytes,
    scope_owner_id,
    append_raw_entry,
    read_detail_tail,
    read_raw_entries,
    count_raw_entries,
    list_compartments,
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
    partition_forget_requests,
)
from discordbot.services.memory.writer import (
    MemoryWriterAI,
    MemoryObservation,
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
    MEMORY_INJECTION_MAX_CHARS,
    MEMORY_INJECTION_WARN_CHARS,
    MEMORY_DETAIL_CONTEXT_MAX_CHARS,
)
from discordbot.services.memory.inflight import (
    MemoryTurn,
    MemoryWriteReport,
    stage_turn,
    report_writes,
    safe_db_write,
    staging_locks,
    drop_pending_updates,
    enqueue_memory_update,
)
from discordbot.services.memory.constants import (
    COMPACTION_TRIGGER_CHARS,
    MEMORY_GLOBAL_CONCURRENCY,
    RAW_CONSOLIDATION_MAX_BYTES,
    RAW_CONSOLIDATION_THRESHOLD,
    MEMORY_CONSOLIDATION_COOLDOWN_SECONDS,
)
from discordbot.services.memory.git_history import memory_git

# Per-scope consolidation attempt times for the cooldown; monotonic, so it does
# not need a loop-change reset. Tests clear it through the conftest fixture.
_last_consolidation: dict[str, float] = {}

# Consecutive refused consolidation batches per (scope, compartment). A refusal keeps the
# raw batch for retry, which is right for a transient LLM failure and wrong for a
# mass-delete the model reproduces verbatim: that retries every cooldown forever while the
# scope's memory silently stops updating. Past the threshold the log escalates from warn to
# error, since CONTRIBUTING puts "an unexpected failure that lost a deliverable" at error.
_consecutive_rejections: dict[tuple[str, str], int] = {}
_MAX_QUIET_REJECTIONS = 3

# Process-wide semaphore capping concurrent background memory updates so a busy server
# cannot fan out unbounded LLM work; shared across flavors and rebuilt per loop. The cap
# is read at build time so a test that lowers MEMORY_GLOBAL_CONCURRENCY first still applies.
_memory_semaphore_holder = LoopLocalSemaphore(capacity_provider=lambda: MEMORY_GLOBAL_CONCURRENCY)


def memory_semaphore() -> asyncio.Semaphore:
    """Returns the process-wide semaphore, rebuilt when the event loop changes."""
    return _memory_semaphore_holder.get()


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
            if forced is None or not _should_consolidate(scope=turn.scope, forced=forced):
                return
            await _consolidate_now(
                scope=turn.scope,
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
        async with scope_lock(scope=scope), memory_semaphore():
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
    compartments = _compartments_to_run(buckets=buckets)
    global_reference = ""
    try:
        async with asyncio.timeout(MEMORY_CONSOLIDATE_TIMEOUT_SECONDS):
            # Forgets first, so a fact this batch also re-confirms is not deleted right after
            # being written, and so the observation pass sees the tree the deletions left.
            if not await apply_forget_buckets(
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
                    request_parts=CompartmentInput(
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
    await update_tone_note(
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
    report_injection_size(scope=scope, flavor=flavor)
    # The consumed batch's content is preserved in the cold-tier detail file; every
    # failure path above returns before this, so it can never retire an unread bucket.
    append_detail(scope=scope, text=raw_entries)
    clear_raw(scope=scope)
    # Best-effort and deliberately fire-and-forget: the worker takes this same scope
    # lock, so it commits once the caller releases it and never sees a half-written batch.
    memory_git.enqueue(scope=scope, reason="update")


async def apply_forget_buckets(  # noqa: PLR0913 -- the scope's identity plus the buckets, their stamp and the LLM handle
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
            request_parts=CompartmentInput(
                raw_entries=forget_text, recent_detail="", global_reference="", today=today
            ),
        )
        if outcome is None or not outcome.applied:
            return False
    return True


class CompartmentInput(BaseModel):
    """The per-compartment half of a consolidation request, before the store is read.

    Split out so a caller can build the parts that differ per compartment without
    `compartment_request` growing a dozen positional arguments. The incremental fan-out
    fills all four; a from-scratch rebuild has only the raw bucket and the date.
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
    request_parts: CompartmentInput,
    deletes_only: bool = False,
) -> DeltaOutcome | None:
    """Runs and applies one compartment's consolidation; None means the LLM path failed.

    This call only ever sees the evidence routed to the compartment it is writing, which
    is what makes "a guild-locked observation cannot reach `global/`" structural rather
    than a rule the prompt asks the model to follow. The tone note, which is genuinely
    cross-compartment, is therefore NOT written here — see `tone.update_tone_note`.
    """
    existing = read_facts(scope=scope, compartment=compartment)
    rendered = render_existing_facts(facts=existing)
    result = await writer.consolidate(
        request=compartment_request(
            compartment=compartment,
            flavor=flavor,
            existing_facts=rendered,
            parts=request_parts,
            # Never on a forget-only call, however large the compartment is: compaction
            # asks the model to merge and condense, and `apply_deltas` then drops every
            # non-delete it produced with a warning apiece. The block would only buy a
            # rewrite nobody can apply.
            compact=not deletes_only and len(rendered) > COMPACTION_TRIGGER_CHARS,
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


def compartment_request(
    compartment: str,
    flavor: MemoryFlavor,
    existing_facts: str,
    parts: CompartmentInput,
    compact: bool,
) -> ConsolidationRequest:
    """Builds one compartment's consolidation request, filling in what never varies.

    The compartment note and the flavor's legal sections are what both callers have to get
    right and neither varies with the batch, so they are derived here once; the incremental
    fan-out and the from-scratch rebuild differ only in which blocks they can offer. Neither
    writes the tone note — `tone` owns that call — so `emit_tone` is false for both.
    """
    return ConsolidationRequest(
        compartment_note=_compartment_note(compartment=compartment, flavor=flavor),
        allowed_sections=tuple(sorted(sections_for_flavor(flavor=flavor))),
        existing_facts=existing_facts,
        raw_entries=parts.raw_entries,
        recent_detail=parts.recent_detail,
        # `global` is never handed its own facts as reference: they are already its
        # `existing_facts`, and offering them twice only invites the model to restate them.
        global_reference="" if compartment == GLOBAL_COMPARTMENT else parts.global_reference,
        today=parts.today,
        compact=compact,
        emit_tone=False,
    )


def global_first(compartments: set[str]) -> list[str]:
    """Orders one run's compartments with `global` leading and the rest by name.

    `global` leads because every later compartment is handed its facts as read-only
    reference, so it must be up to date before they run. Only the ordering is shared: which
    compartments go in is the caller's own question, and the two answer it differently.
    """
    ordered = [GLOBAL_COMPARTMENT] if GLOBAL_COMPARTMENT in compartments else []
    ordered.extend(sorted(compartments - {GLOBAL_COMPARTMENT}))
    return ordered


def _compartments_to_run(buckets: dict[str, str]) -> list[str]:
    """Returns the compartments this run touches, `global` first.

    Only compartments the batch actually routed evidence to: a call with an empty bucket
    has nothing to consolidate.
    """
    return global_first(
        compartments={
            compartment
            for compartment in buckets
            if compartment != GLOBAL_COMPARTMENT or buckets[compartment]
        }
    )


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


def report_injection_size(scope: str, flavor: MemoryFlavor) -> None:
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
