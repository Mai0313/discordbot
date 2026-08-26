"""Rebuilding a scope's memory from cold-tier evidence alone.

The incremental fan-out in `pipeline.py` merges one raw batch into the facts already
stored. This is the other direction: the existing facts are not shown to the model at all,
every compartment is distilled from the detail tail plus any unconsumed raw entries, and
whatever the rebuild did not re-emit is deleted. It is the one path allowed to lose most of
a compartment at once, because replacing the whole set is what it is for.

Two entry points, one path: `/memory regenerate` schedules it in the background under the
running bot's scope lock, and `scripts/regen_memories.py` drives the same coroutine offline.
They are NOT equivalent — the script is a second process, and its closing `clear_raw`
unlinks whatever `raw.md` gained while it worked.
"""

import time
from typing import Literal
import asyncio
from datetime import UTC, datetime

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.memory import MemoryOwner
from discordbot.typings.timeouts import MEMORY_CONSOLIDATE_TIMEOUT_SECONDS
from discordbot.utils.asyncio_locks import LoopLocalRegistry
from discordbot.services.memory.tone import rebuild_tone_note
from discordbot.services.memory.facts import MemoryFlavor, parse_identity
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    clear_raw,
    flavor_of,
    read_facts,
    scope_lock,
    append_detail,
    cleared_since,
    scope_owner_id,
    read_detail_tail,
    read_raw_entries,
    detail_file_bytes,
    list_compartments,
    prune_compartment,
)
from discordbot.services.memory.deltas import (
    apply_deltas,
    partition_raw_entries,
    partition_forget_requests,
)
from discordbot.services.memory.writer import MemoryWriterAI, ConsolidatedMemory
from discordbot.typings.context_budgets import MEMORY_DETAIL_CONTEXT_MAX_CHARS
from discordbot.services.memory.inflight import memory_semaphore
from discordbot.services.memory.pipeline import (
    CompartmentInput,
    global_first,
    compartment_request,
    apply_forget_buckets,
    report_injection_size,
)
from discordbot.services.memory.constants import MEMORY_REGENERATION_COOLDOWN_SECONDS
from discordbot.services.memory.git_history import memory_git

# The ways a from-scratch rebuild can end, carried on `RegenerationReport.result`.
_RegenerationResult = Literal["regenerated", "no_evidence", "failed", "cooldown"]

# Per-scope regeneration attempt times, separate from the consolidation cooldown
# so a manual `/memory regenerate` never starves the automatic background
# consolidation or vice versa. Recorded at attempt time so failures cool down too.
_last_regeneration: dict[str, float] = {}

# Per-scope in-flight regeneration tasks so a manual rebuild runs in the
# background without blocking the command, and a second request while one is
# still running cannot double-schedule the rebuild. Kept separate from the reply
# turn queue because regeneration is a distinct, user-triggered job.
_regeneration_tasks: LoopLocalRegistry[str, asyncio.Task["RegenerationReport"]] = (
    LoopLocalRegistry()
)


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


async def regenerate_scope_memory(
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
    async with scope_lock(scope=scope), memory_semaphore():
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
                        request=compartment_request(
                            compartment=compartment,
                            flavor=flavor,
                            existing_facts="",
                            parts=CompartmentInput(
                                raw_entries=raw_bucket,
                                recent_detail="",
                                global_reference="",
                                today=today,
                            ),
                            compact=True,
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
                await rebuild_tone_note(
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
        report_injection_size(scope=scope, flavor=flavor)
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
    await apply_forget_buckets(
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


def _compartments_to_rebuild(scope: str, buckets: dict[str, str]) -> list[str]:
    """Returns every compartment a rebuild touches, `global` first.

    Compartments that still hold files but have no surviving evidence are included so
    the rebuild empties them; leaving them alone would keep pre-rebuild facts visible
    alongside the new ones with no evidence behind them. Touching one does not always
    mean consolidating it: an entry that turns out to hold neither evidence nor a
    readable fact is pruned without a model call (`regenerate_scope_memory` has the why).
    """
    return global_first(
        compartments={GLOBAL_COMPARTMENT, *buckets, *list_compartments(scope=scope)}
    )


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
