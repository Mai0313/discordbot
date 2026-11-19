"""Consolidation fan-out from raw entries into compartmentalized memory facts.

Routes `raw.md` entries across compartments and applies per-compartment model deltas under the
scope lock. `global` always runs before guild and DM compartments so downstream calls receive
current global facts as read-only reference.
"""

import time
import asyncio
from functools import partial

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.memory import MemoryFact, MemoryFlavor
from discordbot.typings.timeouts import MEMORY_CONSOLIDATE_TIMEOUT_SECONDS
from discordbot.services.memory.run import ConsolidationRun, start_run
from discordbot.services.memory.tone import forget_tone, update_tone_note
from discordbot.services.memory.facts import today_utc, is_past_recent_ttl, sections_for_flavor
from discordbot.services.memory.store import (
    DM_COMPARTMENT,
    GLOBAL_COMPARTMENT,
    read_facts,
    scope_lock,
    replace_raw,
    append_detail,
    cleared_since,
    raw_file_bytes,
    read_detail_tail,
    read_raw_entries,
    rewrite_evidence,
    count_raw_entries,
    list_compartments,
    compartment_guild_id,
    read_memory_document,
)
from discordbot.services.memory.deltas import (
    DeltaOutcome,
    apply_deltas,
    forget_segments,
    reconfirm_facts,
    sweep_stale_facts,
    net_deletion_ceiling,
    partition_raw_entries,
    render_existing_facts,
    drop_released_evidence,
    partition_forget_requests,
)
from discordbot.services.memory.writer import MemoryWriterAI, ConsolidationRequest
from discordbot.typings.context_budgets import (
    MEMORY_INJECTION_MAX_CHARS,
    MEMORY_INJECTION_WARN_CHARS,
    MEMORY_DETAIL_RETRY_MAX_CHARS,
    MEMORY_DETAIL_CONTEXT_MAX_CHARS,
)
from discordbot.services.memory.inflight import memory_semaphore
from discordbot.services.memory.constants import (
    COMPACTION_TRIGGER_CHARS,
    RAW_CONSOLIDATION_MAX_BYTES,
    RAW_CONSOLIDATION_THRESHOLD,
    MEMORY_CONSOLIDATION_COOLDOWN_SECONDS,
)
from discordbot.services.memory.git_history import memory_git
from discordbot.services.memory.raw_entries import (
    iter_observations,
    split_raw_entries,
    render_file_entries,
)

# Per-scope monotonic attempt times for the cooldown.
_last_consolidation: dict[str, float] = {}

# Consecutive refused batches per (scope, compartment); drives warning-to-error escalation.
_consecutive_rejections: dict[tuple[str, str], int] = {}

# (scope, compartment) pairs whose last observation call failed; retried with less detail.
_failed_observations: set[tuple[str, str]] = set()

# UTC day of each (scope, compartment)'s last failed or refused compaction.
_failed_compactions: dict[tuple[str, str], str] = {}


def needs_consolidation(scope: str) -> bool:
    """Returns whether `scope`'s raw backlog is over threshold without taking the scope lock.

    `consolidate_if_needed` re-checks under the lock. An unreadable or undecodable `raw.md` logs
    a warning and returns False.

    Args:
        scope (str): Memory scope to check.
    """
    try:
        return _should_consolidate(scope=scope)
    except (OSError, UnicodeDecodeError) as error:
        logfire.warn(
            "Memory raw.md could not be read; scope left out of the consolidation sweep",
            scope=scope,
            _exc_info=error,
        )
        return False


async def consolidate_if_needed(scope: str, writer: MemoryWriterAI, identity: str) -> None:
    """Consolidates a scope whose raw backlog is over threshold; best-effort and self-logging.

    Takes the scope lock and a semaphore permit around `consolidate_after_turn`, re-checking
    the threshold under the lock, and swallows its own errors.

    Args:
        scope (str): Memory scope to consolidate.
        writer (MemoryWriterAI): Memory writing service running the consolidation calls.
        identity (str): Rendered owner identity stamped onto written facts.
    """
    try:
        async with scope_lock(scope=scope), memory_semaphore():
            await consolidate_after_turn(
                scope=scope,
                forced=False,
                started_at=time.monotonic(),
                writer=writer,
                identity=identity,
            )
    except Exception as exc:
        logfire.warn(
            "Background memory consolidation sweep failed",
            scope=scope,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )


async def consolidate_after_turn(
    scope: str, forced: bool, started_at: float, writer: MemoryWriterAI, identity: str
) -> None:
    """Consolidates `scope` when its backlog warrants it, as of `started_at`.

    Caller must hold the scope lock and a semaphore permit. Stamps the cooldown at attempt time,
    regardless of outcome.

    Args:
        scope (str): Memory scope to consolidate.
        forced (bool): Whether the batch carries a forget request and bypasses threshold and
            cooldown checks.
        started_at (float): `time.monotonic()` timestamp for `cleared_since` checks.
        writer (MemoryWriterAI): Memory writing service running the consolidation calls.
        identity (str): Rendered owner identity stamped onto written facts.
    """
    if not _should_consolidate(scope=scope, forced=forced):
        return
    _last_consolidation[scope] = time.monotonic()
    await _consolidate_locked(
        run=start_run(scope=scope, writer=writer, identity=identity, started_at=started_at)
    )


def _should_consolidate(scope: str, forced: bool = False) -> bool:
    """Returns whether `scope`'s raw backlog warrants consolidation right now."""
    if forced:
        return True
    if raw_file_bytes(scope=scope) >= RAW_CONSOLIDATION_MAX_BYTES:
        # Byte cap bypasses the cooldown so a verbose burst cannot leave raw.md oversized.
        return True
    if count_raw_entries(scope=scope) < RAW_CONSOLIDATION_THRESHOLD:
        return False
    last_attempt = _last_consolidation.get(scope)
    if last_attempt is None or cleared_since(scope=scope, started_at=last_attempt):
        # Post-clear state ignores the wiped memory's prior cooldown.
        return True
    return time.monotonic() - last_attempt >= MEMORY_CONSOLIDATION_COOLDOWN_SECONDS


async def _consolidate_locked(run: ConsolidationRun) -> None:
    """Fans one raw batch out over `run.scope`'s compartments and retires completed segments.

    Caller must hold the scope lock and a semaphore permit.
    """
    segments = forget_segments(raw_text=read_raw_entries(scope=run.scope))
    completed = 0
    try:
        async with asyncio.timeout(MEMORY_CONSOLIDATE_TIMEOUT_SECONDS):
            # Process segments in chronological order so each forget reaches prior facts without
            # wiping later restatements.
            observed = True
            for observations, forgets in segments:
                # After a failed observation pass, keep later observations for retry but still run
                # forgets so deleted facts stop injecting immediately.
                if observed:
                    observed = await _consolidate_observations(run=run, raw_text=observations)
                if not await apply_forget_buckets(run=run, forgets=forgets):
                    break
                # Tone preferences live outside facts; pass only this segment's forgets so later
                # restatements survive.
                if not await forget_tone(run=run, forgets=forgets):
                    break
                completed += observed
            else:
                # Compact only after all forgets finish, and even after a failed observation pass
                # to shrink oversized compartments before retry.
                await _compact_oversized(run=run)
    except TimeoutError:
        logfire.warn(
            "Memory consolidation fan-out timed out; keeping the unfinished part of the raw batch",
            scope=run.scope,
            segments=len(segments),
            completed_segments=completed,
        )
    if not completed or cleared_since(scope=run.scope, started_at=run.started_at):
        return
    # Re-read raw.md after forgets strip released evidence; forget entries remain as boundaries.
    boundary = (
        iter_observations(text=segments[completed - 1][1])[-1]
        if completed < len(segments)
        else None
    )
    raw_entries, remaining = _split_after(
        raw_text=read_raw_entries(scope=run.scope), boundary=boundary
    )
    if remaining:
        logfire.info(
            "Memory consolidation retiring part of the raw batch",
            scope=run.scope,
            segments=len(segments),
            completed_segments=completed,
        )
    tone_updated = await update_tone_note(run=run, raw_entries=raw_entries)
    if not tone_updated or cleared_since(scope=run.scope, started_at=run.started_at):
        return
    # Sweep every compartment (not only touched ones) so idle guilds still age out expired facts.
    for compartment in list_compartments(scope=run.scope):
        sweep_stale_facts(scope=run.scope, compartment=compartment, today=today_utc())
    report_injection_size(scope=run.scope, flavor=run.flavor)
    # Every failure path returns earlier, so retiring into detail.md never drops unread entries.
    append_detail(scope=run.scope, text=raw_entries)
    replace_raw(scope=run.scope, text=remaining)
    # Worker acquires the same scope lock, so the commit runs after this call releases it.
    memory_git.enqueue(scope=run.scope, reason="update")


async def _consolidate_observations(run: ConsolidationRun, raw_text: str) -> bool:
    """Runs the observation fan-out over one slice of the raw batch, returning whether it applied."""
    buckets = partition_raw_entries(raw_text=raw_text, flavor=run.flavor)
    if not buckets:
        return True
    detail_tail = read_detail_tail(scope=run.scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS)
    # Offload parsing to avoid blocking the reply loop; writes remain guarded by cleared_since.
    detail_buckets = await asyncio.to_thread(
        partition_raw_entries, raw_text=detail_tail, flavor=run.flavor
    )
    global_reference = ""
    for compartment in global_first(compartments=set(buckets)):
        if compartment != GLOBAL_COMPARTMENT and not global_reference:
            # Load stored global facts when this batch had no global observations so guild passes
            # still avoid restating them.
            global_reference = render_existing_facts(
                facts=_live_facts(scope=run.scope, compartment=GLOBAL_COMPARTMENT)
            )
        if cleared_since(scope=run.scope, started_at=run.started_at):
            return False
        recent_detail = detail_buckets.get(compartment, "")
        if (run.scope, compartment) in _failed_observations:
            # Retry failed compartments with a smaller detail window so oversized evidence does
            # not deadlock them.
            recent_detail = _newest_entries(
                text=recent_detail, max_chars=MEMORY_DETAIL_RETRY_MAX_CHARS
            )
            logfire.info(
                "Memory consolidation retrying a failed compartment with less evidence",
                scope=run.scope,
                compartment=compartment,
                detail_chars=len(recent_detail),
            )
        outcome = await _consolidate_compartment(
            run=run,
            compartment=compartment,
            request_parts=CompartmentInput(
                raw_entries=buckets.get(compartment, ""),
                recent_detail=recent_detail,
                global_reference=global_reference,
            ),
        )
        if outcome is None:
            _failed_observations.add((run.scope, compartment))
            return False
        _failed_observations.discard((run.scope, compartment))
        if not outcome.applied:
            return False
        if compartment == GLOBAL_COMPARTMENT:
            global_reference = render_existing_facts(
                facts=_live_facts(scope=run.scope, compartment=compartment)
            )
    return True


async def apply_forget_buckets(run: ConsolidationRun, forgets: str) -> bool:
    """Runs one deletion-only pass per compartment reachable by `forgets`.

    Removes released evidence keys from `raw.md` and `detail.md` after each applied compartment;
    callers holding earlier file contents must re-read.

    Args:
        run (ConsolidationRun): Active consolidation or rebuild run context.
        forgets (str): Raw forget-request entries for this segment.

    Returns:
        True when every compartment's forget pass applied; False when the caller must keep the
        raw batch for a retry.
    """
    buckets = partition_forget_requests(
        raw_text=forgets, compartments=tuple(list_compartments(scope=run.scope))
    )
    for compartment, forget_text in sorted(buckets.items()):
        if cleared_since(scope=run.scope, started_at=run.started_at):
            return False
        outcome = await _consolidate_compartment(
            run=run,
            compartment=compartment,
            deletes_only=True,
            request_parts=CompartmentInput(
                raw_entries=forget_text, recent_detail="", global_reference=""
            ),
        )
        if outcome is None or not outcome.applied:
            return False
        if outcome.released_keys:
            if cleared_since(scope=run.scope, started_at=run.started_at):
                return False
            rewrite_evidence(
                scope=run.scope,
                edit=partial(
                    drop_released_evidence,
                    released={compartment: outcome.released_keys},
                    forgets=buckets,
                ),
            )
    return True


async def _compact_oversized(run: ConsolidationRun) -> None:
    """Compacts compartments over `COMPACTION_TRIGGER_CHARS`, at most once per failed UTC day."""
    empty = CompartmentInput(raw_entries="", recent_detail="", global_reference="")
    for compartment in list_compartments(scope=run.scope):
        facts = _live_facts(scope=run.scope, compartment=compartment)
        # Measure fact bodies only; rendered headers would let alias rows trip the trigger.
        before = sum(len(fact.text) for fact in facts)
        if before <= COMPACTION_TRIGGER_CHARS:
            continue
        key = (run.scope, compartment)
        if _failed_compactions.get(key) == run.today:
            logfire.debug(
                "Memory compaction skipped; it already failed today",
                scope=run.scope,
                compartment=compartment,
            )
            continue
        if cleared_since(scope=run.scope, started_at=run.started_at):
            return
        outcome = await _consolidate_compartment(
            run=run, compartment=compartment, request_parts=empty, compact=True
        )
        if outcome is None or not outcome.applied:
            _failed_compactions[key] = run.today
            continue
        _failed_compactions.pop(key, None)
        logfire.info(
            "Memory compartment compacted",
            scope=run.scope,
            compartment=compartment,
            facts_before=len(facts),
            chars_before=before,
            chars_after=sum(
                len(fact.text) for fact in _live_facts(scope=run.scope, compartment=compartment)
            ),
        )


class CompartmentInput(BaseModel):
    """Per-compartment evidence and reference inputs for a consolidation request."""

    model_config = ConfigDict(frozen=True)

    raw_entries: str = Field(..., description="This compartment's share of the raw batch.")
    recent_detail: str = Field(..., description="Cold evidence filtered to this compartment.")
    global_reference: str = Field(..., description="Global facts already stored, or empty.")


async def _consolidate_compartment(
    run: ConsolidationRun,
    compartment: str,
    request_parts: CompartmentInput,
    deletes_only: bool = False,
    compact: bool = False,
) -> DeltaOutcome | None:
    """Runs and applies one compartment's consolidation pass, returning None on LLM failure."""
    # Forgets see expired facts too so their delete releases evidence keys; TTL sweeps do not.
    existing = (
        read_facts(scope=run.scope, compartment=compartment)
        if deletes_only
        else _live_facts(scope=run.scope, compartment=compartment)
    )
    # Logged body-only length matching the compaction metric in `_compact_oversized`.
    stored_chars = sum(len(fact.text) for fact in existing)
    request = compartment_request(
        run=run,
        compartment=compartment,
        existing_facts=render_existing_facts(facts=existing),
        parts=request_parts,
        compact=compact,
    )
    if compact:
        # Tell the model up front because apply_deltas rejects the whole pass if exceeded.
        request = request.model_copy(
            update={"net_removal_limit": net_deletion_ceiling(existing=len(existing))}
        )
    result = await run.writer.consolidate(flavor=run.flavor, request=request)
    if result is None:
        logfire.warn(
            "Memory compaction LLM call failed"
            if compact
            else "Memory consolidation LLM call failed; keeping raw batch",
            scope=run.scope,
            compartment=compartment,
            raw_entries=count_raw_entries(scope=run.scope),
            compacted=compact,
        )
        return None
    if cleared_since(scope=run.scope, started_at=run.started_at):
        # Checked right before disk writes with no await in between.
        return None
    outcome = apply_deltas(
        scope=run.scope,
        compartment=compartment,
        flavor=run.flavor,
        deltas=result.deltas,
        owner=run.owner,
        # Forget passes are exempt from the net-deletion cap (see MAX_NET_FACT_DELETIONS_FLOOR).
        allow_mass_delete=deletes_only,
        deletes_only=deletes_only,
    )
    if not outcome.applied:
        if compact:
            # Best-effort and holds nothing back, so exclude it from stuck-batch counts.
            return outcome
        _record_rejection(
            scope=run.scope, compartment=compartment, outcome=outcome, stored=len(existing)
        )
        return outcome
    if not compact:
        _consecutive_rejections.pop((run.scope, compartment), None)
    reconfirmed = 0
    if not deletes_only:
        # Run before sweep_stale_facts so restated facts are re-stamped instead of aged out.
        reconfirmed = reconfirm_facts(
            scope=run.scope,
            compartment=compartment,
            raw_text=request_parts.raw_entries,
            written=outcome.written,
        )
    swept = sweep_stale_facts(scope=run.scope, compartment=compartment, today=today_utc())
    logfire.debug(
        "Memory compartment consolidated",
        scope=run.scope,
        compartment=compartment,
        created=outcome.created,
        updated=outcome.updated,
        deleted=outcome.deleted,
        dropped=outcome.dropped,
        reconfirmed=reconfirmed,
        swept=swept,
        compacted=compact,
        stored_chars=stored_chars,
    )
    return outcome


def _live_facts(scope: str, compartment: str) -> list[MemoryFact]:
    """Returns non-expired facts in `compartment` visible to replies and writing passes."""
    today = today_utc()
    return [
        fact
        for fact in read_facts(scope=scope, compartment=compartment)
        if not is_past_recent_ttl(fact=fact, today=today)
    ]


def _record_rejection(scope: str, compartment: str, outcome: DeltaOutcome, stored: int) -> None:
    """Logs a refused batch, escalating to error after repeated consecutive refusals."""
    # Track per (scope, compartment) so another compartment's success cannot reset a stuck one.
    count = _consecutive_rejections.get((scope, compartment), 0) + 1
    _consecutive_rejections[(scope, compartment)] = count
    max_quiet_rejections = 3
    log = logfire.error if count >= max_quiet_rejections else logfire.warn
    log(
        "Memory consolidation batch refused; keeping raw batch",
        scope=scope,
        compartment=compartment,
        reason=outcome.rejected,
        existing_facts=stored,
        consecutive=count,
    )


def compartment_request(
    run: ConsolidationRun,
    compartment: str,
    existing_facts: str,
    parts: CompartmentInput,
    compact: bool,
) -> ConsolidationRequest:
    """Returns one compartment's consolidation request with flavor sections and compartment note.

    Args:
        run (ConsolidationRun): Active consolidation or rebuild run context.
        compartment (str): Compartment being written.
        existing_facts (str): Rendered existing facts in `compartment`.
        parts (CompartmentInput): Per-compartment raw entries, detail tail, and global reference.
        compact (bool): Whether this pass should compact the compartment.
    """
    return ConsolidationRequest(
        compartment_note=_compartment_note(compartment=compartment, flavor=run.flavor),
        allowed_sections=tuple(sorted(sections_for_flavor(flavor=run.flavor))),
        existing_facts=existing_facts,
        raw_entries=parts.raw_entries,
        recent_detail=parts.recent_detail,
        # Omit global_reference for global itself; its facts are already in existing_facts.
        global_reference="" if compartment == GLOBAL_COMPARTMENT else parts.global_reference,
        today=run.today,
        compact=compact,
        emit_tone=False,
    )


def _newest_entries(text: str, max_chars: int) -> str:
    """Returns the newest whole entries of oldest-first evidence `text` within `max_chars`."""
    kept: list[str] = []
    size = 0
    for entry in reversed(split_raw_entries(text=text)):
        size += len(entry) + 2
        if size > max_chars:
            break
        kept.append(entry)
    return "\n\n".join(reversed(kept))


def _split_after(raw_text: str, boundary: tuple[str, str] | None) -> tuple[str, str]:
    """Returns `raw_text` split right after the `boundary` entry, or whole when it is None."""
    pairs = iter_observations(text=raw_text)
    cut = len(pairs) if boundary is None else pairs.index(boundary) + 1
    return render_file_entries(pairs=pairs[:cut]), render_file_entries(pairs=pairs[cut:])


def global_first(compartments: set[str]) -> list[str]:
    """Returns `compartments` ordered with `global` first and the rest sorted by name.

    Args:
        compartments (set[str]): Compartment names to order.
    """
    ordered = [GLOBAL_COMPARTMENT] if GLOBAL_COMPARTMENT in compartments else []
    ordered.extend(sorted(compartments - {GLOBAL_COMPARTMENT}))
    return ordered


def _compartment_note(compartment: str, flavor: MemoryFlavor) -> str:
    """Returns a plain-English description of who can read `compartment` for prompts."""
    if flavor == "server":
        return "this server's own community memory; every member of this server can read it"
    if compartment == GLOBAL_COMPARTMENT:
        return "cross-server safe memory; readable in every server and DM this user takes part in"
    if compartment == DM_COMPARTMENT:
        return "private memory; readable only in this user's own direct messages with the bot"
    guild_id = compartment_guild_id(compartment=compartment)
    return f"memory readable only inside Discord server {guild_id}"


def report_injection_size(scope: str, flavor: MemoryFlavor) -> None:
    """Logs when a scope's injectable document approaches or exceeds the hard cap.

    Never deletes facts on disk.

    Args:
        scope (str): Memory scope to measure.
        flavor (MemoryFlavor): Scope flavor used to render the document.
    """
    compartments = list_compartments(scope=scope)
    if not compartments:
        return
    # Measure across all compartments (matching the owner's DM view, the widest read context).
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
