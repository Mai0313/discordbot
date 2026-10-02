"""Rebuilding a scope's memory from cold-tier evidence alone.

The incremental fan-out in `consolidation.py` merges one raw batch into the facts already
stored. This is the other direction: the existing facts are not shown to the model at all,
every compartment is distilled from the detail tail plus any unconsumed raw entries, and
whatever the rebuild did not re-emit is deleted. Only it and a forget pass may lose most of a
compartment at once, because replacing the whole set is what it is for.

It runs either inside the bot under the scope lock, or offline in a second process — where
it is NOT equivalent, since its closing `clear_raw` unlinks whatever `raw.md` gained while
it worked.
"""

import time
from typing import Literal
import asyncio

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.memory import MemoryFact
from discordbot.typings.timeouts import MEMORY_CONSOLIDATE_TIMEOUT_SECONDS
from discordbot.services.memory.run import ConsolidationRun, start_run
from discordbot.utils.asyncio_locks import LoopLocalRegistry
from discordbot.services.memory.tone import forget_tone, rebuild_tone_note
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    clear_raw,
    read_facts,
    scope_lock,
    write_fact,
    append_detail,
    cleared_since,
    read_evidence,
    read_raw_entries,
    detail_file_bytes,
    list_compartments,
    prune_compartment,
)
from discordbot.services.memory.deltas import (
    apply_deltas,
    forget_segments,
    partition_raw_entries,
    render_existing_facts,
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

# The ways a from-scratch rebuild can end, carried on `RegenerationReport.result`.
_RegenerationResult = Literal["regenerated", "no_evidence", "failed", "cooldown"]

# Per-scope regeneration attempt times, separate from the consolidation cooldown
# so a manual `/memory regenerate` never starves the automatic background
# consolidation or vice versa. Recorded at attempt time so failures cool down too.
_last_regeneration: dict[str, float] = {}

# Per-scope in-flight regeneration tasks so a manual rebuild runs in the
# background without blocking the command, and a second request while one is
# still running cannot double-schedule the rebuild.
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
    """Whether any cold-tier evidence exists for a from-scratch rebuild.

    Mirrors the evidence guard inside `regenerate_scope_memory` cheaply (no full
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
    """Starts a background rebuild of the scope's memory without blocking the command.

    Returns False when a rebuild is already in flight for this scope (so the
    caller can report "still rebuilding" instead of double-scheduling the
    rebuild); True when a fresh background task was started.
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
        # Cancelled (e.g. bot shutdown): reading result() would raise
        # CancelledError out of this callback. A cancel before the last forget replay has
        # applied puts back every compartment the rebuild replaced, and raw.md is not
        # yet retired either way.
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


async def regenerate_scope_memory(  # noqa: C901, PLR0911 -- one early report per way a rebuild stops short, a clear and a failed forget replay included
    scope: str, writer: MemoryWriterAI, identity: str
) -> RegenerationReport:
    """Rebuilds every compartment from cold-tier evidence alone.

    The existing facts are deliberately NOT fed to the model: the rebuild distills the
    detail tail window plus any unconsumed raw entries from scratch, e.g. to redo an
    unsatisfying consolidation with another model, and facts it did not re-emit are then
    deleted. The corpus is taken one forget at a time in recorded order, as consolidation takes
    a batch: what came before a forget is rebuilt, the forget replayed, and what came after
    merged into the facts this run rebuilt, so a restatement survives the forget it followed.

    A run that stops before its last forget replay has applied puts every compartment it replaced
    back to the facts it held, unless the process dies without unwinding: until the replay, a
    replaced compartment can hold a fact a forget had already removed. The raw batch is
    retired only when every compartment and the tone note rebuilt. The report carries what
    the run removed unread whichever way it ended, since putting a compartment back restores
    only what could be read, so a rebuild that gave up on its third compartment still
    accounts for what the first two destroyed.
    """
    started_at = time.monotonic()
    unreadable_removed = 0
    async with scope_lock(scope=scope), memory_semaphore():
        if regeneration_on_cooldown(scope=scope):
            # Invocations queued behind a held lock all pass the command-level
            # cooldown check before the first one stamps the attempt; the
            # re-check under the lock keeps the per-scope limit on the rewrite.
            return RegenerationReport(result="cooldown")
        run = start_run(scope=scope, writer=writer, identity=identity, started_at=started_at)
        raw_entries = read_raw_entries(scope=scope)
        evidence = read_evidence(scope=scope)
        if not evidence:
            return RegenerationReport(result="no_evidence")
        _last_regeneration[scope] = time.monotonic()
        buckets = partition_raw_entries(raw_text=evidence, flavor=run.flavor)
        compartments = _compartments_to_rebuild(scope=scope, buckets=buckets)
        replaced: dict[str, list[MemoryFact]] = {}
        try:
            # The individual calls carry no deadline of their own, so this is the only
            # thing standing between a stuck rebuild and a scope lock held for as long as
            # the client will keep one compartment's request alive.
            async with asyncio.timeout(MEMORY_CONSOLIDATE_TIMEOUT_SECONDS):
                # One forget at a time, in the order they were recorded, as consolidation takes
                # a batch: a forget reaches only the facts made from what came before it, and
                # what was said after it is a restatement it must not reach.
                for index, (observations, forgets) in enumerate(
                    forget_segments(raw_text=evidence)
                ):
                    segment = partition_raw_entries(raw_text=observations, flavor=run.flavor)
                    # The first pass touches every compartment, so one with nothing left to
                    # rebuild from is emptied; a later one only those it has evidence for.
                    for compartment in (
                        compartments if index == 0 else global_first(compartments=set(segment))
                    ):
                        raw_bucket = segment.get(compartment, "")
                        if not raw_bucket:
                            # Nothing to distil before the first forget: the model would be
                            # handed an empty corpus and could only answer with an empty batch,
                            # so the prune alone reaches the same state without a call that
                            # could fail the compartments that do have something. A later
                            # segment merges into the emptied compartment.
                            replaced[compartment] = read_facts(
                                scope=scope, compartment=compartment
                            )
                            unreadable_removed += _prune_rebuilt_compartment(
                                scope=scope, compartment=compartment, keep=set()
                            )
                            continue
                        removed = await _rebuild_compartment(
                            run=run,
                            compartment=compartment,
                            raw_bucket=raw_bucket,
                            replaced=replaced,
                        )
                        if removed is None:
                            return RegenerationReport(
                                result="failed", unreadable_removed=unreadable_removed
                            )
                        unreadable_removed += removed
                    if not await _reapply_forgets(run=run, forgets=forgets):
                        return RegenerationReport(
                            result="failed", unreadable_removed=unreadable_removed
                        )
                # The replay has reached every compartment, so the rebuild stands from here.
                replaced.clear()
                # The replay takes the evidence of what it deleted out of both files, so the
                # tone rebuild and the retirement below must not work from the copies read
                # before it.
                raw_entries = read_raw_entries(scope=scope)
                evidence = read_evidence(scope=scope)
                tone_rebuilt = await rebuild_tone_note(run=run, evidence=evidence)
        except TimeoutError:
            logfire.warn(
                "Memory regeneration timed out", scope=scope, compartments=len(compartments)
            )
            return RegenerationReport(result="failed", unreadable_removed=unreadable_removed)
        finally:
            _restore_compartments(run=run, replaced=replaced)
        if not tone_rebuilt or cleared_since(scope=scope, started_at=started_at):
            return RegenerationReport(result="failed", unreadable_removed=unreadable_removed)
        report_injection_size(scope=scope, flavor=run.flavor)
        if raw_entries:
            # The rebuild consumed the raw batch; retire it to the cold tier so it
            # cannot be re-ingested.
            append_detail(scope=scope, text=raw_entries)
            clear_raw(scope=scope)
        memory_git.enqueue(scope=scope, reason="rebuild")
        return RegenerationReport(result="regenerated", unreadable_removed=unreadable_removed)


async def _rebuild_compartment(
    run: ConsolidationRun, compartment: str, raw_bucket: str, replaced: dict[str, list[MemoryFact]]
) -> int | None:
    """Rebuilds one compartment from one segment's evidence; None means the run stops here.

    The compartment's first pass replaces it from that evidence alone, recording what it held
    for `_restore_compartments`. A later pass merges the segment into what this run rebuilt, as
    consolidation merges a batch: shown no earlier evidence, so a fact-writing call never sees
    what a forget left behind, and never compacting, since it could not check a fact against
    evidence it was not shown. Returns how many unreadable files the replace removed.
    """
    rebuilt = compartment in replaced
    existing = read_facts(scope=run.scope, compartment=compartment) if rebuilt else []
    result = await run.writer.consolidate(
        flavor=run.flavor,
        request=compartment_request(
            run=run,
            compartment=compartment,
            existing_facts=render_existing_facts(facts=existing),
            parts=CompartmentInput(raw_entries=raw_bucket, recent_detail="", global_reference=""),
            compact=not existing,
        ),
    )
    if result is None:
        logfire.warn(
            "Memory regeneration LLM call failed; this and later compartments left untouched",
            scope=run.scope,
            compartment=compartment,
        )
        return None
    if cleared_since(scope=run.scope, started_at=run.started_at):
        return None
    if not rebuilt:
        replaced[compartment] = read_facts(scope=run.scope, compartment=compartment)
        return _replace_compartment(run=run, compartment=compartment, result=result)
    outcome = apply_deltas(
        scope=run.scope,
        compartment=compartment,
        flavor=run.flavor,
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
        return None
    return 0


def _restore_compartments(run: ConsolidationRun, replaced: dict[str, list[MemoryFact]]) -> None:
    """Puts every compartment the run replaced back to the facts it held before the run.

    Only what `read_facts` could read was kept, so a file the replace pass removed unread
    stays gone and stays counted. Skipped after a clear, which already deleted both
    generations: writing the old facts back would undo it.
    """
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


async def _reapply_forgets(run: ConsolidationRun, forgets: str) -> bool:
    """Re-runs one segment's forget requests against the facts rebuilt from what came before.

    A rebuild derives facts from evidence rather than from the current facts, and the
    observation a forget was aimed at can still be sitting in `detail.md` verbatim: the forget
    pass takes out only the evidence of a fact it actually deleted, so a forget that found
    nothing to delete left it there. So the rebuild re-creates exactly what the user asked to
    have removed, and without this `/memory regenerate` would quietly undo those forgets. The
    replay then takes that evidence out as the incremental pass does, so the next rebuild no
    longer sees it.

    The tone evidence gets the same replay, since a tone preference is never a fact and the tone
    rebuild reads whatever evidence is left: a forget whose tone pass failed, its batch kept in
    `raw.md`, or missed a line would otherwise come back from it (#971). Only the evidence, as
    the note is about to be rewritten from it (`forget_tone` has why).

    Replaying the requests afterwards fixes that without weakening anything: each runs as its
    own `deletes_only` call, or for the tone evidence one answering line numbers only, so the
    forget's own sentence still cannot be written anywhere. Feeding a forget INTO the rebuild
    instead would hand a possibly-private sentence to a call whose whole job is creating facts.

    Returns False when a replay call failed. The caller then puts the replaced compartments
    back rather than keep them: a fact the replay did not reach can be one the user asked to
    forget, and nothing but a later rebuild that completes would remove it.
    """
    return await apply_forget_buckets(run=run, forgets=forgets) and await forget_tone(
        run=run, forgets=forgets, evidence_only=True
    )


def _compartments_to_rebuild(scope: str, buckets: dict[str, str]) -> list[str]:
    """Returns every compartment a rebuild touches, `global` first.

    Compartments that still hold files but have no surviving evidence are included so
    the rebuild empties them; leaving them alone would keep pre-rebuild facts visible
    alongside the new ones with no evidence behind them. Touching one does not always
    mean consolidating it: an entry with no evidence before the first forget is pruned
    without a model call (`regenerate_scope_memory` has the why).
    """
    return global_first(
        compartments={GLOBAL_COMPARTMENT, *buckets, *list_compartments(scope=scope)}
    )


def _replace_compartment(
    run: ConsolidationRun, compartment: str, result: ConsolidatedMemory
) -> int:
    """Replaces a compartment's contents with a from-scratch rebuild's facts.

    Applies the batch first and prunes afterwards, so a fact the rebuild kept is never
    momentarily absent. What survives is decided by the ids the batch actually WROTE, not
    by what is left on disk: a rebuild says "this fact is gone" by simply not mentioning
    it, so comparing against the post-apply state would only re-delete what the batch
    already deleted and leave every stale fact standing.

    The mass-delete guard is off here: replacing the entire set is what this path is for.
    """
    outcome = apply_deltas(
        scope=run.scope,
        compartment=compartment,
        flavor=run.flavor,
        deltas=result.deltas,
        owner=run.owner,
        allow_mass_delete=True,
    )
    return _prune_rebuilt_compartment(
        scope=run.scope, compartment=compartment, keep=set(outcome.written)
    )


def _prune_rebuilt_compartment(scope: str, compartment: str, keep: set[str]) -> int:
    """Reduces a rebuilt compartment to `keep`, reporting what it left and what it took.

    The prune reads the directory rather than the facts read back from it, so a file no
    reader can parse cannot outlive a rebuild that reports the compartment replaced
    (`prune_compartment` carries the why). What it could not account for is reported
    here instead, since the store never removes a file it did not write.

    What it DID remove unread is reported and returned. Renaming a section or a durability
    value makes every fact carrying the old one unparsable, so the next rebuild of a scope
    drops all of them in one pass; a run that says only what it spared reads as one that
    destroyed nothing.
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
