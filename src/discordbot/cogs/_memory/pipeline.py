"""Background orchestration for the two-phase memory pipeline.

The pipeline is keyed by an opaque scope (see ``store``), so the same
orchestration drives both per-user and per-server (bot self) memory. The
flavor-specific bits are injected: ``subject`` names the extraction target and
``extractor`` carries the flavor's prompts.
"""

import time
from typing import Literal
import asyncio
from datetime import UTC, datetime
from collections.abc import Awaitable

import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.cogs._memory import database as memory_db
from discordbot.cogs._memory.store import (
    clear_raw,
    read_tone,
    scope_lock,
    write_tone,
    append_detail,
    cleared_since,
    raw_file_bytes,
    append_raw_entry,
    read_detail_tail,
    read_main_memory,
    read_raw_entries,
    count_raw_entries,
    detail_file_bytes,
    write_main_memory,
)
from discordbot.utils.asyncio_locks import LoopLocalRegistry, LoopLocalSemaphore
from discordbot.cogs._memory.constants import (
    MEMORY_GLOBAL_CONCURRENCY,
    RAW_CONSOLIDATION_MAX_BYTES,
    RAW_CONSOLIDATION_THRESHOLD,
    MAIN_COMPACTION_TARGET_CHARS,
    MAIN_COMPACTION_TRIGGER_CHARS,
    MEMORY_DETAIL_CONTEXT_MAX_CHARS,
    MEMORY_REGENERATION_COOLDOWN_SECONDS,
    MEMORY_CONSOLIDATION_COOLDOWN_SECONDS,
)
from discordbot.cogs._memory.extraction import (
    MemoryExtractorAI,
    parse_subject_source,
    transcript_from_messages,
    render_memory_observations,
    filter_duplicate_observations,
)

# Outcome of a from-scratch main-file rebuild. Aliased so the background
# scheduler's task dict shares the exact type (asyncio.Task is invariant in its
# result type, so a Literal cannot stand in for a bare str).
_RegenerationResult = Literal["regenerated", "no_evidence", "failed", "cooldown"]


class _PendingMemoryUpdate(BaseModel):
    """The newest skipped update request, replayed once the in-flight task ends.

    Attributes:
        subject: The phase-1 extraction directive naming the memory target.
        transcript: The rendered phase-1 input captured for the skipped turn
            (already folds in the reply), so the replay needs no re-render.
        extractor: The extraction service to run the replayed update with.
        identity: Single-line target identity stamped into the main memory
            file as human-inspection metadata.
        captured_at: `time.monotonic()` when the turn was captured, so a clear
            that lands before the replay can abort it via `cleared_since`.
        token: `time.time_ns()` version token persisted with the deferred turn's
            DB row, reused on replay so the terminal write guards on the same id.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subject: str = Field(
        ..., description="The phase-1 extraction directive naming the memory target."
    )
    transcript: str = Field(
        ..., description="The rendered phase-1 input captured for the skipped turn."
    )
    extractor: SkipValidation[MemoryExtractorAI] = Field(
        ..., description="The extraction service to run the replayed update with."
    )
    identity: str = Field(
        ...,
        description=(
            "Single-line target identity stamped into the main memory file as "
            "human-inspection metadata."
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
        ..., description="time.time_ns() version token reused on replay for the DB row guard."
    )


# Process-level per-scope in-flight de-dupe; while one extraction runs, only the
# NEWEST skipped turn is kept and replayed afterwards. Its history window
# already contains the earlier skipped turns, so one replay recovers the
# dropped signal without a real queue.
_inflight_tasks: dict[str, asyncio.Task[None]] = {}
_pending_updates: dict[str, _PendingMemoryUpdate] = {}
_inflight_loop: asyncio.AbstractEventLoop | None = None

# Per-scope consolidation attempt times for the cooldown; monotonic, so it does
# not need a loop-change reset. Tests clear it through the conftest fixture.
_last_consolidation: dict[str, float] = {}

# Per-scope regeneration attempt times, separate from the consolidation cooldown
# so a manual `/memory regenerate` never starves the automatic background
# consolidation or vice versa. Recorded at attempt time so failures cool down too.
_last_regeneration: dict[str, float] = {}

# Per-scope in-flight regeneration tasks so a manual rebuild runs in the
# background without blocking the command, and a second request while one is
# still running cannot double-schedule the whole-file rewrite. Kept separate
# from `_inflight_tasks` because regeneration is a distinct, user-triggered job.
_regeneration_tasks: LoopLocalRegistry[str, asyncio.Task[_RegenerationResult]] = (
    LoopLocalRegistry()
)

# Process-wide semaphore capping concurrent background memory updates so a busy server
# cannot fan out unbounded LLM work; shared across flavors and rebuilt per loop. The cap
# is read at build time so a test that lowers MEMORY_GLOBAL_CONCURRENCY first still applies.
_memory_semaphore_holder = LoopLocalSemaphore(capacity_provider=lambda: MEMORY_GLOBAL_CONCURRENCY)


def _memory_semaphore() -> asyncio.Semaphore:
    """Returns the process-wide semaphore, rebuilt when the event loop changes."""
    return _memory_semaphore_holder.get()


# Detached best-effort reply.db writes (the deferred-turn persist), held so the
# event loop keeps a strong reference until they finish; rebuilt per loop.
_db_tasks: set[asyncio.Task[None]] = set()


def flavor_of(scope: str) -> memory_db.MemoryJobFlavor:
    """Maps a scope to its persisted memory flavor (`server_scope` carries a '/')."""
    return "server" if "/" in scope else "user"


async def _safe(coro: Awaitable[None]) -> None:
    """Awaits a best-effort reply.db write, swallowing any failure.

    Persistence is an augmentation layer (the opposite of `research.py`, which
    lets DB errors raise into its run loop): a reply.db failure must never break
    the in-memory fire-and-forget memory pipeline.
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


def schedule_memory_update(  # noqa: PLR0913 -- flavor (scope/subject/identity) plus the turn payload
    scope: str,
    subject: str,
    message_list: list[EasyInputMessageParam],
    full_reply: str,
    extractor: MemoryExtractorAI,
    identity: str,
) -> None:
    """Starts a background memory update without delaying the reply path.

    The transcript is rendered eagerly here (pure, sub-ms, already past the reply)
    so the persisted job and the in-memory replay both carry a plain string and
    `_run_memory_update` re-renders nothing.
    """
    transcript = transcript_from_messages(message_list=message_list, full_reply=full_reply)
    _enqueue_memory_update(
        scope=scope,
        subject=subject,
        transcript=transcript,
        extractor=extractor,
        identity=identity,
        token=time.time_ns(),
    )


def resume_memory_update(  # noqa: PLR0913 -- mirrors a persisted row's columns
    *,
    scope: str,
    subject: str,
    transcript: str,
    extractor: MemoryExtractorAI,
    identity: str,
    token: int,
) -> None:
    """Re-enqueues a persisted phase-1 turn on restart, reusing its stored token."""
    _enqueue_memory_update(
        scope=scope,
        subject=subject,
        transcript=transcript,
        extractor=extractor,
        identity=identity,
        token=token,
    )


def _enqueue_memory_update(  # noqa: PLR0913 -- flavor (scope/subject/identity) plus the rendered turn
    scope: str,
    subject: str,
    transcript: str,
    extractor: MemoryExtractorAI,
    identity: str,
    token: int,
) -> None:
    """Schedules (or defers) one rendered-transcript update, backed by a reply.db row."""
    global _inflight_loop  # noqa: PLW0603 -- process task de-dupe
    loop = asyncio.get_running_loop()
    if _inflight_loop is not loop:
        _inflight_tasks.clear()
        _pending_updates.clear()
        _inflight_loop = loop
    running = _inflight_tasks.get(scope)
    if running is not None and not running.done():
        _pending_updates[scope] = _PendingMemoryUpdate(
            subject=subject,
            transcript=transcript,
            extractor=extractor,
            identity=identity,
            captured_at=time.monotonic(),
            token=token,
        )
        # Persist the deferred turn so a redeploy before it runs still resumes it.
        # Safe from a same-token race: this turn's worker only starts after the
        # in-flight one ends, long after this detached write lands, and it carries
        # a newer token than the running turn so newest-wins keeps it.
        _spawn_db(
            coro=memory_db.upsert_pending(
                scope=scope,
                flavor=flavor_of(scope=scope),
                subject=subject,
                transcript=transcript,
                identity=identity,
                token=token,
            )
        )
        return
    task = asyncio.create_task(
        _run_memory_update(
            scope=scope,
            subject=subject,
            transcript=transcript,
            extractor=extractor,
            identity=identity,
            token=token,
        )
    )
    _inflight_tasks[scope] = task
    task.add_done_callback(lambda finished: _finish_memory_update(scope=scope, task=finished))


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
    except Exception:
        logfire.warn("Background memory update failed", scope=scope, _exc_info=True)
    pending = _pending_updates.pop(scope, None)
    if pending is None:
        return
    if cleared_since(scope=scope, started_at=pending.captured_at):
        # The memory was cleared after this turn was captured; replaying it
        # would write the pre-clear conversation back into storage. Mark the
        # persisted deferred row done too (mirrors the in-flight clear branch in
        # `_run_memory_update`) so a restart does not resume the cleared turn.
        _spawn_db(coro=memory_db.mark_done(scope=scope, token=pending.token))
        return
    _enqueue_memory_update(
        scope=scope,
        subject=pending.subject,
        transcript=pending.transcript,
        extractor=pending.extractor,
        identity=pending.identity,
        token=pending.token,
    )


async def _run_memory_update(  # noqa: PLR0913 -- mirrors schedule_memory_update's flavor + payload
    scope: str,
    subject: str,
    transcript: str,
    extractor: MemoryExtractorAI,
    identity: str,
    token: int,
) -> None:
    """Runs phase-1 extraction and, past the raw threshold, phase-2 consolidation.

    The reply.db row is written `pending` at the top (awaited, before the lock) so
    a redeploy mid-extraction resumes this turn; it is marked `done` once phase-1
    is terminal (extracted, no signal, all dupes, or cleared) and `failed` only
    when the LLM call itself fails, so the restart sweep retries just that case.
    Consolidation needs no DB row: `raw.md` is its durable, re-entrant queue.
    """
    started_at = time.monotonic()
    await _safe(
        coro=memory_db.upsert_pending(
            scope=scope,
            flavor=flavor_of(scope=scope),
            subject=subject,
            transcript=transcript,
            identity=identity,
            token=token,
        )
    )
    async with scope_lock(scope=scope), _memory_semaphore():
        draft = await extractor.extract(subject=subject, transcript=transcript)
        if draft is None:
            # The LLM path itself failed: keep the row (transcript intact) so the
            # restart sweep retries it, no extra timeout needed.
            await _safe(
                coro=memory_db.mark_failed(scope=scope, token=token, error="extract failed")
            )
            return
        if not draft.has_signal or not draft.observations:
            await _safe(coro=memory_db.mark_done(scope=scope, token=token))
            return
        if cleared_since(scope=scope, started_at=started_at):
            # The memory was cleared while this update was in flight; dropping
            # the write beats resurrecting deleted memory.
            await _safe(coro=memory_db.mark_done(scope=scope, token=token))
            return
        # The subject's source line survives the memory_job round-trip, so a resumed
        # turn stamps the same source; a pre-source row (or the server flavor) parses
        # to None and renders without the source/sharing fields.
        source = parse_subject_source(subject=subject)
        recent_detail = read_detail_tail(scope=scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS)
        deduped_observations = filter_duplicate_observations(
            observations=draft.observations,
            existing_text="\n\n".join((read_raw_entries(scope=scope), recent_detail)),
            source=source,
        )
        if not deduped_observations:
            await _safe(coro=memory_db.mark_done(scope=scope, token=token))
            return
        append_raw_entry(
            scope=scope,
            entry_text=render_memory_observations(
                observations=deduped_observations, source=source
            ),
        )
        # Phase-1 is durable in raw.md now; record success before the (best-effort,
        # self-healing) consolidation so a consolidation crash never re-runs extraction.
        await _safe(coro=memory_db.mark_done(scope=scope, token=token))
        if not _should_consolidate(scope=scope):
            return
        # Recorded at attempt time, not success time, so repeated LLM failures
        # are rate-limited by the same cooldown instead of retrying every turn.
        _last_consolidation[scope] = time.monotonic()
        await _consolidate_locked(
            scope=scope, started_at=started_at, extractor=extractor, identity=identity
        )


async def safe_list_resumable() -> list[memory_db.MemoryJob]:
    """Returns the persisted non-`done` jobs for the restart sweep, best-effort.

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


async def consolidate_if_needed(scope: str, extractor: MemoryExtractorAI, identity: str) -> None:
    """Consolidates a scope whose raw backlog is over threshold; best-effort, self-logging.

    The boot-sweep entry point: `_consolidate_locked` / `_should_consolidate` are
    private and assume the scope lock + semaphore are held, so this wrapper takes
    both and re-checks the threshold under the lock. It swallows its own errors
    (a background digest must never surface), so the caller just spawns it.
    """
    try:
        async with scope_lock(scope=scope), _memory_semaphore():
            if not _should_consolidate(scope=scope):
                return
            _last_consolidation[scope] = time.monotonic()
            await _consolidate_locked(
                scope=scope, started_at=time.monotonic(), extractor=extractor, identity=identity
            )
    except Exception:
        logfire.warn("Background memory consolidation sweep failed", scope=scope, _exc_info=True)


def _should_consolidate(scope: str) -> bool:
    """Whether the raw backlog warrants a consolidation right now."""
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
    scope: str, started_at: float, extractor: MemoryExtractorAI, identity: str
) -> None:
    """Consolidates accumulated raw entries into the main memory file (and tone note)."""
    existing_main = read_main_memory(scope=scope)
    compact = len(existing_main) > MAIN_COMPACTION_TRIGGER_CHARS
    result = await extractor.consolidate(
        existing_main=existing_main,
        existing_tone=read_tone(scope=scope),
        raw_entries=read_raw_entries(scope=scope),
        recent_detail=read_detail_tail(scope=scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS),
        today=datetime.now(UTC).date().isoformat(),
        compact=compact,
    )
    if result is None:
        # LLM path failed; keep the raw entries so the next update retries.
        return
    if cleared_since(scope=scope, started_at=started_at):
        return
    is_well_formed = result.memory_markdown.startswith("v1\n")
    if result.memory_markdown and not is_well_formed:
        # Any non-empty output that is not a well-formed `v1` rewrite is
        # malformed (missing the exact header line, near-misses like `v10...` /
        # `v1: ...`): keep the raw batch for retry regardless of `changed`,
        # instead of discarding the accumulated signal.
        return
    if result.tone_markdown and not result.tone_markdown.startswith("## 語氣偏好"):
        # A malformed tone note rejects the batch like a malformed main rewrite: the
        # rewritten main may have moved tone bullets out on the promise they land in
        # the note, so consuming the batch here would silently lose the preference
        # from every injected tier. Keeping raw retries the whole consolidation.
        return
    # The first rewrite that brings an untagged (pre-migration) file into the tagged
    # format legitimately sheds a lot of text (tone bullets move to tone.md, private
    # profile prose turns into tagged bullets), so it is judged by the deeper
    # compaction floor; once tags exist the normal halving guard resumes.
    first_tagged_rewrite = "[src:" not in existing_main and "[src:" in result.memory_markdown
    if is_well_formed and _rewrite_shrank_too_much(
        existing_main=existing_main,
        rewritten=result.memory_markdown,
        compact=compact or first_tagged_rewrite,
    ):
        # A drastic surprise shrink is almost always a lossy LLM failure, not
        # a merge; refusing it keeps raw for retry and protects main.bak.md
        # (one generation deep) from being overwritten by the bad rewrite.
        logfire.warn(
            "Memory consolidation shrank too much; keeping previous memory",
            scope=scope,
            existing_chars=len(existing_main),
            rewritten_chars=len(result.memory_markdown),
            compact=compact,
        )
        return
    if is_well_formed:
        # Accept any well-formed `v1` rewrite, even one the model flagged
        # `changed=false`, so a single contradictory boolean cannot silently
        # discard the whole raw batch.
        write_main_memory(scope=scope, content=result.memory_markdown, identity=identity)
    _write_tone_result(scope=scope, tone_markdown=result.tone_markdown)
    # Reached only by a well-formed write or a genuine empty no-op: the batch is
    # consumed either way, since an unchanged verdict on the same raw entries
    # would just re-burn a consolidation call on every following extraction.
    # The consumed batch's content is preserved in the cold-tier detail file;
    # the failure paths above keep raw for retry and therefore must not retire it.
    append_detail(scope=scope, text=read_raw_entries(scope=scope))
    clear_raw(scope=scope)


def _write_tone_result(scope: str, tone_markdown: str) -> None:
    """Persists a consolidation's tone note when it is acceptable for this scope.

    Sits on the tone-consuming paths (accepted rewrite AND genuine no-op, both of
    which retire the raw batch — a main no-op can still carry fresh tone signal
    that would otherwise be consumed without landing). User scopes only, and only
    a note starting with the exact `## 語氣偏好` header; an empty or malformed
    output never deletes the existing note — the tier is best-effort and the next
    consolidation repairs it.
    """
    if flavor_of(scope=scope) != "user":
        return
    if not tone_markdown.startswith("## 語氣偏好"):
        return
    write_tone(scope=scope, content=tone_markdown)


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


def schedule_memory_regeneration(scope: str, extractor: MemoryExtractorAI, identity: str) -> bool:
    """Starts a background main-memory rebuild without blocking the command.

    Returns False when a rebuild is already in flight for this scope (so the
    caller can report "still rebuilding" instead of double-scheduling the
    whole-file rewrite); True when a fresh background task was started.
    """
    running = _regeneration_tasks.get(key=scope)
    if running is not None and not running.done():
        return False
    task = asyncio.create_task(
        regenerate_main_memory(scope=scope, extractor=extractor, identity=identity)
    )
    _regeneration_tasks.set(key=scope, value=task)
    task.add_done_callback(
        lambda finished: _finish_memory_regeneration(scope=scope, task=finished)
    )
    return True


def _finish_memory_regeneration(scope: str, task: asyncio.Task[_RegenerationResult]) -> None:
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
    except Exception:
        logfire.warn("Background memory regeneration failed", scope=scope, _exc_info=True)


async def regenerate_main_memory(
    scope: str, extractor: MemoryExtractorAI, identity: str
) -> _RegenerationResult:
    """Rebuilds the main memory file from cold-tier evidence alone.

    The existing main file is deliberately NOT fed to the model: the rebuild
    distills the detail tail window plus any unconsumed raw entries from
    scratch, e.g. to redo an unsatisfying consolidation with another model.
    The compaction block is always applied so a window-sized evidence corpus
    summarizes into the output-token budget instead of failing `incomplete`.
    On any failure the current main file and the raw batch stay untouched;
    `write_main_memory` keeps the previous generation in `main.bak.md`.
    """
    started_at = time.monotonic()
    async with scope_lock(scope=scope), _memory_semaphore():
        if regeneration_on_cooldown(scope=scope):
            # Invocations queued behind a held lock all pass the command-level
            # cooldown check before the first one stamps the attempt; the
            # re-check under the lock keeps the per-scope limit on the rewrite.
            return "cooldown"
        raw_entries = read_raw_entries(scope=scope)
        recent_detail = read_detail_tail(scope=scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS)
        # Detail entries are retired raw entries verbatim with the same
        # `## <ISO timestamp>` headers, so the combined corpus (oldest first)
        # slots into the raw-entries consolidation input unchanged.
        evidence = "\n\n".join(part for part in (recent_detail, raw_entries) if part)
        if not evidence:
            return "no_evidence"
        # Recorded at attempt time, not success time, so repeated LLM failures
        # are rate-limited by the same cooldown.
        _last_regeneration[scope] = time.monotonic()
        result = await extractor.consolidate(
            existing_main="",
            existing_tone="",
            raw_entries=evidence,
            recent_detail="",
            today=datetime.now(UTC).date().isoformat(),
            compact=True,
        )
        if result is None or not result.memory_markdown.startswith("v1\n"):
            # LLM failure or malformed rewrite; a from-scratch rebuild has no
            # prior size to compare, so the `v1` header check is the guard.
            return "failed"
        if cleared_since(scope=scope, started_at=started_at):
            return "failed"
        write_main_memory(scope=scope, content=result.memory_markdown, identity=identity)
        _write_tone_result(scope=scope, tone_markdown=result.tone_markdown)
        if raw_entries:
            # The rebuild consumed the raw batch; retire it to the cold tier
            # exactly like a consolidation so it cannot be re-ingested.
            append_detail(scope=scope, text=raw_entries)
            clear_raw(scope=scope)
        return "regenerated"


def _rewrite_shrank_too_much(existing_main: str, rewritten: str, compact: bool) -> bool:
    """Whether a well-formed rewrite lost so much text it reads as a lossy failure."""
    if compact:
        # Compaction legitimately shrinks toward the target; collapsing below
        # a tenth of the input reads as dropped content rather than
        # summarization. The target-based floor keeps a main file that grew
        # far past ten times the target compactable instead of stuck retrying.
        floor = min(len(existing_main) // 10, MAIN_COMPACTION_TARGET_CHARS // 3)
        return len(rewritten) < floor
    # Consolidation merges and dedupes, so mild shrinkage is normal; losing
    # over half of a non-trivial file is not. Small files are exempt because
    # legitimate restructuring dominates at that scale.
    return len(existing_main) > 2_000 and len(rewritten) < len(existing_main) // 2
