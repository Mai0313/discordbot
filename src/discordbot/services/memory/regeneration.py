"""From-scratch memory rebuilds from cold-tier evidence (`detail.md` plus `raw.md`).

Distills each compartment without showing existing facts, replays recorded forget requests, dates
rebuilt facts from their supporting evidence, and sweeps stale facts.

Runs either in-process under `scope_lock` (`/memory regenerate`) or offline
(`admin/regen_memories.py`). The two are not equivalent: an offline run holds no live process
lock, so its closing `clear_raw` unlinks any entries appended to `raw.md` while it ran.
"""

import time
from typing import Literal
import asyncio
from functools import partial
from collections.abc import Callable

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.memory import MemoryFact
from discordbot.typings.timeouts import MEMORY_CONSOLIDATE_TIMEOUT_SECONDS
from discordbot.services.memory.run import ConsolidationRun, start_run
from discordbot.utils.asyncio_locks import LoopLocalRegistry
from discordbot.services.memory.tone import forget_tone, rebuild_tone_note
from discordbot.services.memory.facts import today_utc
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    clear_raw,
    read_facts,
    scope_lock,
    write_fact,
    trim_detail,
    append_detail,
    cleared_since,
    read_evidence,
    read_raw_entries,
    rewrite_evidence,
    detail_file_bytes,
    list_compartments,
    prune_compartment,
)
from discordbot.services.memory.deltas import (
    apply_deltas,
    forget_segments,
    sweep_stale_facts,
    partition_raw_entries,
    render_existing_facts,
    date_facts_from_evidence,
)
from discordbot.services.memory.writer import MemoryWriterAI, ConsolidatedMemory
from discordbot.services.memory.inflight import memory_semaphore
from discordbot.services.memory.constants import MEMORY_REGENERATION_COOLDOWN_SECONDS
from discordbot.services.memory.git_history import memory_git
from discordbot.services.memory.consolidation import (
    CompartmentInput,
    global_first,
    compartment_request,
    apply_forget_buckets,
    report_injection_size,
)

# Outcome tags carried on `RegenerationReport.result`.
_RegenerationResult = Literal["regenerated", "no_evidence", "failed", "cooldown"]

# Per-scope monotonic attempt times (separate from consolidation cooldown; stamped at attempt).
_last_regeneration: dict[str, float] = {}

# Loop-local in-flight rebuild tasks to prevent overlapping manual rebuilds per scope.
_regeneration_tasks: LoopLocalRegistry[str, asyncio.Task["RegenerationReport"]] = (
    LoopLocalRegistry()
)


class RegenerationReport(BaseModel):
    """What one from-scratch rebuild did, for a caller with no logfire to read."""

    model_config = ConfigDict(frozen=True)

    result: _RegenerationResult = Field(..., description="How the rebuild ended.")
    unreadable_removed: int = Field(
        default=0, description="Fact files removed that no reader could parse."
    )


def regeneration_has_evidence(scope: str) -> bool:
    """Returns whether any cold-tier or raw evidence exists for a from-scratch rebuild.

    Args:
        scope (str): Memory scope key to check.
    """
    return bool(read_raw_entries(scope=scope)) or detail_file_bytes(scope=scope) > 0


def regeneration_on_cooldown(scope: str) -> bool:
    """Returns whether a recent regeneration attempt in `scope` blocks another one right now.

    Args:
        scope (str): Memory scope key to check.
    """
    last_attempt = _last_regeneration.get(scope)
    if last_attempt is None or cleared_since(scope=scope, started_at=last_attempt):
        # Post-clear state ignores the wiped memory's prior cooldown.
        return False
    return time.monotonic() - last_attempt < MEMORY_REGENERATION_COOLDOWN_SECONDS


def schedule_memory_regeneration(scope: str, writer: MemoryWriterAI, identity: str) -> bool:
    """Starts a background rebuild of `scope`'s memory without blocking the command.

    Args:
        scope (str): Memory scope key to rebuild.
        writer (MemoryWriterAI): Memory writing service running the rebuild calls.
        identity (str): Single-line owner identity stamped onto written facts.

    Returns:
        True when a fresh background task was started; False when a rebuild is already in
        flight for `scope`.
    """
    running = _regeneration_tasks.get(key=scope)
    if running is not None and not running.done():
        return False
    task = asyncio.create_task(
        regenerate_scope_memory(scope=scope, writer=writer, identity=identity)
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
        # Cancelled tasks raise CancelledError from result(); skip logging on shutdown.
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


class _RebuildState(BaseModel):
    """Mutable rollback, deferred evidence edits, and accounting across rebuild segments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    replaced: dict[str, list[MemoryFact]] = Field(
        default_factory=dict,
        description="Pre-rebuild facts per compartment for rollback before the commit point.",
    )
    pending_edits: list[Callable[[str], str]] = Field(
        default_factory=list,
        description="Deferred evidence-file edits flushed once all forget segments succeed.",
    )
    unreadable_removed: int = Field(
        default=0, description="Unreadable fact files removed across rebuilt compartments."
    )


async def regenerate_scope_memory(
    scope: str, writer: MemoryWriterAI, identity: str
) -> RegenerationReport:
    """Rebuilds every compartment of `scope` from cold-tier evidence alone.

    Distills `detail.md` plus any unconsumed `raw.md` one forget segment at a time in recorded
    order without showing existing facts. Buffers forget-driven evidence rewrites and restores
    replaced compartments if the run fails or times out before the final forget replay applies;
    retires `raw.md` only after all compartments and the tone note succeed.

    Args:
        scope (str): Memory scope key to rebuild.
        writer (MemoryWriterAI): Memory writing service running the rebuild calls.
        identity (str): Single-line owner identity stamped onto written facts.

    Returns:
        The `RegenerationReport` summarizing how the rebuild ended and how many unreadable
        fact files were removed.
    """
    started_at = time.monotonic()
    state = _RebuildState()
    async with scope_lock(scope=scope), memory_semaphore():
        if regeneration_on_cooldown(scope=scope):
            # Re-check under the lock because concurrent invocations can pass the command check.
            return RegenerationReport(result="cooldown")
        run = start_run(scope=scope, writer=writer, identity=identity, started_at=started_at)
        evidence = read_evidence(scope=scope)
        if not evidence:
            return RegenerationReport(result="no_evidence")
        _last_regeneration[scope] = time.monotonic()
        compartments = _compartments_to_rebuild(
            scope=scope, buckets=partition_raw_entries(raw_text=evidence, flavor=run.flavor)
        )
        succeeded = False
        try:
            # Single deadline across the compartment loop; client calls carry no tighter bound.
            async with asyncio.timeout(MEMORY_CONSOLIDATE_TIMEOUT_SECONDS):
                succeeded = await _replay_rebuild_segments(
                    run=run, evidence=evidence, compartments=compartments, state=state
                ) and await _finalize_rebuild(run=run, state=state)
        except TimeoutError as error:
            logfire.warn(
                "Memory regeneration timed out",
                scope=scope,
                compartments=len(compartments),
                _exc_info=error,
            )
        finally:
            _restore_compartments(run=run, replaced=state.replaced)
        if not succeeded or cleared_since(scope=scope, started_at=started_at):
            return RegenerationReport(result="failed", unreadable_removed=state.unreadable_removed)
        report_injection_size(scope=scope)
        raw_entries = read_raw_entries(scope=scope)
        if raw_entries:
            # Retire consumed raw entries into detail.md so they are not re-ingested.
            append_detail(scope=scope, text=raw_entries)
            clear_raw(scope=scope)
            await trim_detail(scope=scope, started_at=started_at)
        memory_git.enqueue(scope=scope, reason="rebuild")
        return RegenerationReport(
            result="regenerated", unreadable_removed=state.unreadable_removed
        )


async def _replay_rebuild_segments(
    run: ConsolidationRun, evidence: str, compartments: list[str], state: _RebuildState
) -> bool:
    """Replays each forget segment in recorded order so forgets reach earlier facts only."""
    for index, (observations, forgets) in enumerate(forget_segments(raw_text=evidence)):
        segment = partition_raw_entries(raw_text=observations, flavor=run.flavor)
        # First segment visits every compartment to empty obsolete ones; later ones only active buckets.
        targets = compartments if index == 0 else global_first(compartments=set(segment))
        global_reference = ""
        for compartment in targets:
            if compartment != GLOBAL_COMPARTMENT and not global_reference:
                global_reference = render_existing_facts(
                    facts=read_facts(scope=run.scope, compartment=GLOBAL_COMPARTMENT)
                )
            ok = await _rebuild_compartment(
                run=run,
                compartment=compartment,
                raw_bucket=segment.get(compartment, ""),
                state=state,
                global_reference=global_reference,
            )
            if not ok:
                return False
            if compartment == GLOBAL_COMPARTMENT:
                global_reference = render_existing_facts(
                    facts=read_facts(scope=run.scope, compartment=compartment)
                )
        if not await _reapply_forgets(run=run, forgets=forgets, pending_edits=state.pending_edits):
            return False
    return True


async def _finalize_rebuild(run: ConsolidationRun, state: _RebuildState) -> bool:
    """Flushes deferred forget evidence rewrites, commits replaced compartments, dates facts, and rebuilds tone."""
    if state.pending_edits:
        await rewrite_evidence(
            scope=run.scope,
            edit=partial(_apply_evidence_edits, edits=state.pending_edits),
            started_at=run.started_at,
        )
    if cleared_since(scope=run.scope, started_at=run.started_at):
        return False
    # All forget replays and evidence rewrites landed; commit point (skip restore on later exit).
    state.replaced.clear()
    # Re-read evidence after forget replays stripped released entries from both files.
    evidence = read_evidence(scope=run.scope)
    _date_rebuilt_facts(run=run, evidence=evidence)
    return await rebuild_tone_note(run=run, evidence=evidence)


def _apply_evidence_edits(text: str, edits: list[Callable[[str], str]]) -> str:
    """Applies buffered evidence rewrites in recorded segment order."""
    for edit in edits:
        text = edit(text)
    return text


async def _rebuild_compartment(
    run: ConsolidationRun,
    compartment: str,
    raw_bucket: str,
    state: _RebuildState,
    global_reference: str,
) -> bool:
    """Rebuilds, merges, or prunes one segment into `compartment`, returning whether it succeeded."""
    rebuilt = compartment in state.replaced
    if not raw_bucket:
        # No pre-forget evidence: prune directly instead of an empty LLM call.
        if not rebuilt:
            state.replaced[compartment] = read_facts(scope=run.scope, compartment=compartment)
        state.unreadable_removed += _prune_rebuilt_compartment(
            scope=run.scope, compartment=compartment, keep=set()
        )
        return True
    existing = read_facts(scope=run.scope, compartment=compartment) if rebuilt else []
    result = await run.writer.consolidate(
        flavor=run.flavor,
        request=compartment_request(
            run=run,
            compartment=compartment,
            existing_facts=render_existing_facts(facts=existing),
            parts=CompartmentInput(
                raw_entries=raw_bucket, recent_detail="", global_reference=global_reference
            ),
            compact=not existing,
        ),
    )
    if result is None:
        logfire.warn(
            "Memory regeneration LLM call failed; this and later compartments left untouched",
            scope=run.scope,
            compartment=compartment,
        )
        return False
    if cleared_since(scope=run.scope, started_at=run.started_at):
        return False
    if not rebuilt:
        state.replaced[compartment] = read_facts(scope=run.scope, compartment=compartment)
        state.unreadable_removed += _replace_compartment(
            run=run, compartment=compartment, result=result
        )
        return True
    outcome = apply_deltas(
        scope=run.scope,
        compartment=compartment,
        deltas=result.deltas,
        owner=run.owner,
        allow_mass_delete=False,
    )
    if not outcome.applied:
        logfire.warn(
            "Memory regeneration batch refused",
            scope=run.scope,
            compartment=compartment,
            reason=outcome.rejected,
        )
        return False
    return True


def _restore_compartments(run: ConsolidationRun, replaced: dict[str, list[MemoryFact]]) -> None:
    """Restores compartments replaced during an aborted rebuild unless the scope was cleared."""
    if not replaced or cleared_since(scope=run.scope, started_at=run.started_at):
        return
    for compartment, facts in replaced.items():
        for fact in facts:
            write_fact(scope=run.scope, fact=fact)
        prune_compartment(
            scope=run.scope, compartment=compartment, keep={fact.fact_id for fact in facts}
        )
    logfire.info(
        "Memory regeneration stopped before its forget replay completed; "
        "replaced compartments restored",
        scope=run.scope,
        compartments=sorted(replaced),
    )


async def _reapply_forgets(
    run: ConsolidationRun, forgets: str, pending_edits: list[Callable[[str], str]]
) -> bool:
    """Replays one segment's forget requests against rebuilt facts and tone evidence."""
    return await apply_forget_buckets(
        run=run, forgets=forgets, pending_edits=pending_edits
    ) and await forget_tone(
        run=run, forgets=forgets, evidence_only=True, pending_edits=pending_edits
    )


def _date_rebuilt_facts(run: ConsolidationRun, evidence: str) -> None:
    """Re-dates rebuilt facts per compartment from their evidence and sweeps stale facts."""
    buckets = partition_raw_entries(raw_text=evidence, flavor=run.flavor)
    today = today_utc()
    keyed = swept = facts = 0
    for compartment in list_compartments(scope=run.scope):
        facts += len(read_facts(scope=run.scope, compartment=compartment))
        keyed += date_facts_from_evidence(
            scope=run.scope, compartment=compartment, raw_text=buckets.get(compartment, "")
        )
        swept += sweep_stale_facts(scope=run.scope, compartment=compartment, today=today)
    logfire.info(
        "Memory rebuild dated its facts from their evidence",
        scope=run.scope,
        facts=facts,
        keyed=keyed,
        swept=swept,
    )


def _compartments_to_rebuild(scope: str, buckets: dict[str, str]) -> list[str]:
    """Returns all compartments to rebuild (`global` first), including ones without evidence."""
    return global_first(
        compartments={GLOBAL_COMPARTMENT, *buckets, *list_compartments(scope=scope)}
    )


def _replace_compartment(
    run: ConsolidationRun, compartment: str, result: ConsolidatedMemory
) -> int:
    """Applies a rebuild batch with `allow_mass_delete=True` and prunes unemitted facts."""
    outcome = apply_deltas(
        scope=run.scope,
        compartment=compartment,
        deltas=result.deltas,
        owner=run.owner,
        allow_mass_delete=True,
    )
    return _prune_rebuilt_compartment(
        scope=run.scope, compartment=compartment, keep=set(outcome.written)
    )


def _prune_rebuilt_compartment(scope: str, compartment: str, keep: set[str]) -> int:
    """Reduces a rebuilt compartment to `keep`, logging and returning unreadable files removed."""
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
