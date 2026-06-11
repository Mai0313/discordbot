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

import logfire
from pydantic import BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.cogs._memory.store import (
    clear_raw,
    scope_lock,
    append_detail,
    cleared_since,
    raw_file_bytes,
    append_raw_entry,
    read_detail_tail,
    read_main_memory,
    read_raw_entries,
    count_raw_entries,
    write_main_memory,
)
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
    transcript_from_messages,
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
        message_list: Reply-pipeline input messages captured for the skipped turn.
        full_reply: The streamed reply text for the skipped turn.
        extractor: The extraction service to run the replayed update with.
        identity: Single-line target identity stamped into the main memory
            file as human-inspection metadata.
        captured_at: `time.monotonic()` when the turn was captured, so a clear
            that lands before the replay can abort it via `cleared_since`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subject: str
    message_list: SkipValidation[list[EasyInputMessageParam]]
    full_reply: str
    extractor: SkipValidation[MemoryExtractorAI]
    identity: str
    captured_at: float


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
_regeneration_tasks: dict[str, asyncio.Task[_RegenerationResult]] = {}
_regeneration_loop: asyncio.AbstractEventLoop | None = None

# Loop-keyed process-wide semaphore capping concurrent background memory
# updates so a busy server cannot fan out unbounded LLM work. Shared across
# flavors, so per-user and per-server updates draw from the same budget.
_memory_semaphore_obj: asyncio.Semaphore | None = None
_memory_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _memory_semaphore() -> asyncio.Semaphore:
    """Returns the process-wide semaphore, rebuilt when the event loop changes."""
    global _memory_semaphore_obj, _memory_semaphore_loop  # noqa: PLW0603 -- loop-keyed singleton
    loop = asyncio.get_running_loop()
    if _memory_semaphore_loop is not loop or _memory_semaphore_obj is None:
        _memory_semaphore_obj = asyncio.Semaphore(MEMORY_GLOBAL_CONCURRENCY)
        _memory_semaphore_loop = loop
    return _memory_semaphore_obj


def schedule_memory_update(  # noqa: PLR0913 -- flavor (scope/subject/identity) plus the turn payload
    scope: str,
    subject: str,
    message_list: list[EasyInputMessageParam],
    full_reply: str,
    extractor: MemoryExtractorAI,
    identity: str,
) -> None:
    """Starts a background memory update without delaying the reply path."""
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
            message_list=message_list,
            full_reply=full_reply,
            extractor=extractor,
            identity=identity,
            captured_at=time.monotonic(),
        )
        return
    task = asyncio.create_task(
        _run_memory_update(
            scope=scope,
            subject=subject,
            message_list=message_list,
            full_reply=full_reply,
            extractor=extractor,
            identity=identity,
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
        # would write the pre-clear conversation back into storage.
        return
    schedule_memory_update(
        scope=scope,
        subject=pending.subject,
        message_list=pending.message_list,
        full_reply=pending.full_reply,
        extractor=pending.extractor,
        identity=pending.identity,
    )


async def _run_memory_update(  # noqa: PLR0913 -- mirrors schedule_memory_update's flavor + payload
    scope: str,
    subject: str,
    message_list: list[EasyInputMessageParam],
    full_reply: str,
    extractor: MemoryExtractorAI,
    identity: str,
) -> None:
    """Runs phase-1 extraction and, past the raw threshold, phase-2 consolidation."""
    started_at = time.monotonic()
    transcript = transcript_from_messages(message_list=message_list, full_reply=full_reply)
    async with scope_lock(scope=scope), _memory_semaphore():
        draft = await extractor.extract(subject=subject, transcript=transcript)
        if draft is None or not draft.has_signal or not draft.memory_markdown:
            return
        if cleared_since(scope=scope, started_at=started_at):
            # The memory was cleared while this update was in flight; dropping
            # the write beats resurrecting deleted memory.
            return
        recent_detail = read_detail_tail(scope=scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS)
        deduped_observations = filter_duplicate_observations(
            observations=draft.observations,
            existing_text="\n\n".join((read_raw_entries(scope=scope), recent_detail)),
        )
        if not deduped_observations:
            return
        append_raw_entry(
            scope=scope,
            entry_text=draft.model_copy(
                update={"observations": deduped_observations}
            ).memory_markdown,
        )
        if not _should_consolidate(scope=scope):
            return
        # Recorded at attempt time, not success time, so repeated LLM failures
        # are rate-limited by the same cooldown instead of retrying every turn.
        _last_consolidation[scope] = time.monotonic()
        await _consolidate_locked(
            scope=scope, started_at=started_at, extractor=extractor, identity=identity
        )


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
    """Consolidates accumulated raw entries into the main memory file."""
    existing_main = read_main_memory(scope=scope)
    compact = len(existing_main) > MAIN_COMPACTION_TRIGGER_CHARS
    result = await extractor.consolidate(
        existing_main=existing_main,
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
    if is_well_formed and _rewrite_shrank_too_much(
        existing_main=existing_main, rewritten=result.memory_markdown, compact=compact
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
    # Reached only by a well-formed write or a genuine empty no-op: the batch is
    # consumed either way, since an unchanged verdict on the same raw entries
    # would just re-burn a consolidation call on every following extraction.
    # The consumed batch's content is preserved in the cold-tier detail file,
    # minus legacy identity header suffixes; the failure paths above keep raw
    # for retry and therefore must not retire it.
    append_detail(scope=scope, text=read_raw_entries(scope=scope))
    clear_raw(scope=scope)


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
    global _regeneration_loop  # noqa: PLW0603 -- process task de-dupe
    loop = asyncio.get_running_loop()
    if _regeneration_loop is not loop:
        _regeneration_tasks.clear()
        _regeneration_loop = loop
    running = _regeneration_tasks.get(scope)
    if running is not None and not running.done():
        return False
    task = asyncio.create_task(
        regenerate_main_memory(scope=scope, extractor=extractor, identity=identity)
    )
    _regeneration_tasks[scope] = task
    task.add_done_callback(
        lambda finished: _finish_memory_regeneration(scope=scope, task=finished)
    )
    return True


def _finish_memory_regeneration(scope: str, task: asyncio.Task[_RegenerationResult]) -> None:
    """Clears the in-flight slot and logs failures of a background rebuild."""
    if _regeneration_tasks.get(scope) is task:
        _regeneration_tasks.pop(scope, None)
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
