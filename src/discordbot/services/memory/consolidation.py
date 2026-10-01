"""How a raw batch becomes facts: the compartment fan-out and the decision to run it.

Consolidation reads the entries `raw.md` has accumulated for one scope, routes them over the
compartments that may hold them, and asks the model for a delta batch per compartment.
`_consolidate_locked` owns the ordering that makes the boundary structural; the decision of
whether to run at all is `_should_consolidate`, with the cooldown and the refusal counters
beside it.

Nothing here imports the layer above it. The fan-out is what its callers are built on rather
than a step in any of them, so each reaches it at one call of its own.
"""

import time
import asyncio
from functools import partial

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.memory import MemoryFlavor
from discordbot.typings.timeouts import MEMORY_CONSOLIDATE_TIMEOUT_SECONDS
from discordbot.services.memory.run import ConsolidationRun, start_run
from discordbot.services.memory.tone import forget_tone, update_tone_note
from discordbot.services.memory.facts import sections_for_flavor
from discordbot.services.memory.store import (
    DM_COMPARTMENT,
    GLOBAL_COMPARTMENT,
    clear_raw,
    read_facts,
    scope_lock,
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
    today_utc,
    apply_deltas,
    forget_segments,
    reconfirm_facts,
    sweep_stale_facts,
    partition_raw_entries,
    render_existing_facts,
    drop_released_evidence,
    partition_forget_requests,
)
from discordbot.services.memory.writer import MemoryWriterAI, ConsolidationRequest
from discordbot.typings.context_budgets import (
    MEMORY_INJECTION_MAX_CHARS,
    MEMORY_INJECTION_WARN_CHARS,
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

# Per-scope consolidation attempt times for the cooldown; monotonic, so it does
# not need a loop-change reset.
_last_consolidation: dict[str, float] = {}

# Consecutive refused consolidation batches per (scope, compartment); `_record_rejection`
# owns what the count is for and when it escalates.
_consecutive_rejections: dict[tuple[str, str], int] = {}


def needs_consolidation(scope: str) -> bool:
    """Public sync pre-check for the boot sweep so it only spawns over-threshold scopes.

    A cheap file read (no lock), used to avoid queuing a per-scope task on the
    global semaphore just to discover it is under threshold; `consolidate_if_needed`
    re-checks under the lock, which stays the authority.
    """
    return _should_consolidate(scope=scope)


async def consolidate_if_needed(scope: str, writer: MemoryWriterAI, identity: str) -> None:
    """Consolidates a scope whose raw backlog is over threshold; best-effort, self-logging.

    The boot-sweep entry point: `consolidate_after_turn` assumes the scope lock and a
    semaphore permit are held, so this takes both around it, and the threshold is re-checked
    under the lock. It swallows its own errors (a background digest must never surface), so
    the caller just spawns it.
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
    """Consolidates the scope when its backlog warrants it, as of `started_at`.

    The caller already holds the scope lock and a semaphore permit, so this takes neither and
    folds the threshold check in under them rather than leaving it to be read on the caller's
    side.

    `forced` says the batch carries a forget request; `_should_consolidate` owns what that
    skips. The cooldown is stamped at attempt time, not success time, so repeated LLM
    failures are rate-limited by the same cooldown instead of retrying on every turn.
    """
    if not _should_consolidate(scope=scope, forced=forced):
        return
    _last_consolidation[scope] = time.monotonic()
    await _consolidate_locked(
        run=start_run(scope=scope, writer=writer, identity=identity, started_at=started_at)
    )


def _should_consolidate(scope: str, forced: bool = False) -> bool:
    """Whether the raw backlog warrants a consolidation right now.

    `forced` is a batch carrying a forget request. It skips BOTH gates below rather than just
    the cooldown: the entry count is checked first and a lone forget is one entry, so bypassing
    the cooldown alone would still leave the user waiting for their next message before the bot
    stopped repeating what they asked it to drop.
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


async def _consolidate_locked(run: ConsolidationRun) -> None:
    """Fans one raw batch out over the scope's compartments, applying each one's deltas.

    Ordering is load-bearing. `global` runs first, so every later compartment can be
    handed its facts as read-only reference and neither restates them nor silently
    contradicts them. Each call sees only the evidence routed to the compartment it writes,
    which is what makes the boundary structural; the tone note, the one tier that is
    genuinely cross-compartment, is written afterwards by its own call.

    The whole fan-out sits inside the caller's scope lock and semaphore permit under ONE
    timeout, and compartments run sequentially, so neither the worst-case lock hold nor the
    per-scope proxy load scales with the compartment count.

    The raw batch is retired only when every compartment applied and the tone call answered.
    A retry re-runs the ones that already landed, which is safe because a delta is an upsert
    keyed by an id the model echoes back and, failing that, by the evidence keys the fact
    carries.
    """
    raw_entries = read_raw_entries(scope=run.scope)
    try:
        async with asyncio.timeout(MEMORY_CONSOLIDATE_TIMEOUT_SECONDS):
            # In the order they were recorded: a forget can reach the facts made from what came
            # before it, and a fact re-confirmed after it is not deleted right after being
            # written. A forget only removes evidence stamped before it, so the segments still
            # ahead are the same text they were when this batch was split.
            observed = True
            for observations, forgets in forget_segments(raw_text=raw_entries):
                # Once a pass has failed the batch is kept for a retry, so the passes after it
                # are skipped; the forgets are not, since a fact an earlier batch stored must not
                # go on being injected until that retry, which a stuck compartment may never let
                # succeed.
                if observed:
                    observed = await _consolidate_observations(
                        run=run,
                        raw_text=observations,
                        # Not ahead of a forget: merging the fact it names into others first
                        # would leave it only the choice of deleting all of them or none.
                        may_compact=not forgets,
                    )
                # Forget requests are a separate pass: they are copied into every compartment
                # their speaker could read from, including one the pass above just made, and
                # each of those calls may only delete.
                if not await apply_forget_buckets(run=run, forgets=forgets):
                    return
                # A tone preference is never stored as a fact, so the pass above cannot reach
                # one; it gets its own, taking only this segment's forgets so a tone
                # restatement made after one of them is not offered to it.
                if not await forget_tone(run=run, forgets=forgets):
                    return
            if not observed:
                return
            # Read again after the forget passes, which may have taken evidence out of the
            # file: the batch retired at the end no longer carries it.
            raw_entries = read_raw_entries(scope=run.scope)
    except TimeoutError:
        logfire.warn(
            "Memory consolidation fan-out timed out; keeping raw batch",
            scope=run.scope,
            compartments=len(partition_raw_entries(raw_text=raw_entries, flavor=run.flavor)),
        )
        return
    if cleared_since(scope=run.scope, started_at=run.started_at):
        return
    tone_updated = await update_tone_note(run=run, raw_entries=raw_entries)
    if not tone_updated or cleared_since(scope=run.scope, started_at=run.started_at):
        return
    # Age every compartment, not just the ones this batch touched: a guild the user has
    # stopped visiting otherwise keeps its `recent` facts forever and hands them back on
    # their next visit. The ones that did consolidate were already swept inside the
    # fan-out, and that is not redundant — a timeout or a clear returns before this line,
    # so the per-compartment sweep is the only aging those runs get.
    for compartment in list_compartments(scope=run.scope):
        sweep_stale_facts(scope=run.scope, compartment=compartment, today=today_utc())
    report_injection_size(scope=run.scope, flavor=run.flavor)
    # The consumed batch's content is preserved in the cold-tier detail file; every
    # failure path above returns before this, so it can never retire an unread bucket.
    append_detail(scope=run.scope, text=raw_entries)
    clear_raw(scope=run.scope)
    # Best-effort and deliberately fire-and-forget: the worker takes this same scope
    # lock, so it commits once the caller releases it and never sees a half-written batch.
    memory_git.enqueue(scope=run.scope, reason="update")


async def _consolidate_observations(
    run: ConsolidationRun, raw_text: str, may_compact: bool
) -> bool:
    """Runs the observation fan-out over one slice of the raw batch.

    Returns False when the caller must keep the raw batch for a retry: a partially applied
    fan-out is fine to replay, an unread bucket is not. A slice with no observation in it
    costs nothing, not even the detail read. `may_compact` False keeps every call from
    compacting, however large its compartment.
    """
    buckets = partition_raw_entries(raw_text=raw_text, flavor=run.flavor)
    if not buckets:
        return True
    detail_tail = read_detail_tail(scope=run.scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS)
    # Splitting that window into observation blocks is a real stall on a heavy scope, and this
    # runs on the same loop as the reply path. Pure function, no shared state, so a thread costs
    # nothing — and the await is safe here because every write still sits immediately after its
    # own `cleared_since` guard downstream.
    detail_buckets = await asyncio.to_thread(
        partition_raw_entries, raw_text=detail_tail, flavor=run.flavor
    )
    global_reference = ""
    # Every key is a compartment this batch routed evidence to: the partition only creates a
    # bucket when something lands in it.
    for compartment in global_first(compartments=set(buckets)):
        if compartment != GLOBAL_COMPARTMENT and not global_reference:
            # Read from disk rather than from this run: when the batch carried no
            # cross-server evidence there was no global call to take it from, and
            # a guild compartment still must not restate what is already shared.
            global_reference = render_existing_facts(
                facts=read_facts(scope=run.scope, compartment=GLOBAL_COMPARTMENT)
            )
        if cleared_since(scope=run.scope, started_at=run.started_at):
            return False
        outcome = await _consolidate_compartment(
            run=run,
            compartment=compartment,
            request_parts=CompartmentInput(
                raw_entries=buckets.get(compartment, ""),
                recent_detail=detail_buckets.get(compartment, ""),
                global_reference=global_reference,
            ),
            may_compact=may_compact,
        )
        if outcome is None or not outcome.applied:
            return False
        if compartment == GLOBAL_COMPARTMENT:
            global_reference = render_existing_facts(
                facts=read_facts(scope=run.scope, compartment=compartment)
            )
    return True


async def apply_forget_buckets(run: ConsolidationRun, forgets: str) -> bool:
    """Runs one deletion-only pass per compartment the forget requests in `forgets` can reach.

    Returns False when the caller must keep the raw batch for a retry, on the same terms as
    the observation fan-out: an unread bucket is not safe to retire.

    Every call here is `deletes_only`; `partition_forget_requests` decides which compartments
    each request is copied into, and owns why a forget is partitioned on its own rather than
    folded into the observation buckets. Each applied call then takes the evidence of the facts
    it deleted out of `raw.md` and `detail.md` at once, not after the whole pass: a retry finds
    those facts already gone and releases nothing, so evidence left behind by a later call
    failing would stay for good. A caller holding either file's text from before this pass must
    read it again.
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


class CompartmentInput(BaseModel):
    """The per-compartment half of a consolidation request, before the store is read.

    Split out so a caller can build what differs per compartment without
    `compartment_request` growing a parameter per block. A field may be empty: not every
    caller has every part to offer.
    """

    model_config = ConfigDict(frozen=True)

    raw_entries: str = Field(..., description="This compartment's share of the raw batch.")
    recent_detail: str = Field(..., description="Cold evidence filtered to this compartment.")
    global_reference: str = Field(..., description="Global facts already stored, or empty.")


async def _consolidate_compartment(
    run: ConsolidationRun,
    compartment: str,
    request_parts: CompartmentInput,
    deletes_only: bool = False,
    may_compact: bool = True,
) -> DeltaOutcome | None:
    """Runs and applies one compartment's consolidation; None means the LLM path failed.

    The request carries only the evidence routed to the compartment being written, which is
    what makes "a guild-locked observation cannot reach `global/`" structural rather than a
    rule the prompt asks the model to follow.
    """
    existing = read_facts(scope=run.scope, compartment=compartment)
    rendered = render_existing_facts(facts=existing)
    result = await run.writer.consolidate(
        flavor=run.flavor,
        request=compartment_request(
            run=run,
            compartment=compartment,
            existing_facts=rendered,
            parts=request_parts,
            # Never on a forget-only call, however large the compartment is: compaction
            # asks the model to merge and condense, and `apply_deltas` then drops every
            # non-delete it produced with a warning apiece. The block would only buy a
            # rewrite nobody can apply.
            compact=may_compact and not deletes_only and len(rendered) > COMPACTION_TRIGGER_CHARS,
        ),
    )
    if result is None:
        logfire.warn(
            "Memory consolidation LLM call failed; keeping raw batch",
            scope=run.scope,
            compartment=compartment,
            raw_entries=count_raw_entries(scope=run.scope),
        )
        return None
    if cleared_since(scope=run.scope, started_at=run.started_at):
        # Checked immediately before the first write, with no await in between, so an
        # in-flight clear can never be overtaken by this batch.
        return None
    outcome = apply_deltas(
        scope=run.scope,
        compartment=compartment,
        flavor=run.flavor,
        deltas=result.deltas,
        owner=run.owner,
        allow_mass_delete=False,
        deletes_only=deletes_only,
    )
    if not outcome.applied:
        _record_rejection(
            scope=run.scope, compartment=compartment, outcome=outcome, stored=len(existing)
        )
        return outcome
    _consecutive_rejections.pop((run.scope, compartment), None)
    reconfirmed = 0
    if not deletes_only:
        # Ahead of the sweep, which would otherwise age out a fact this batch restated.
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
    """Builds one compartment's consolidation request, filling in what never varies.

    The compartment note and the flavor's legal sections never vary with the batch, so they
    are derived here once and a caller supplies only the blocks it has. `emit_tone` is false
    because the tone note is its own call, never a field on a compartment's request.
    """
    return ConsolidationRequest(
        compartment_note=_compartment_note(compartment=compartment, flavor=run.flavor),
        allowed_sections=tuple(sorted(sections_for_flavor(flavor=run.flavor))),
        existing_facts=existing_facts,
        raw_entries=parts.raw_entries,
        recent_detail=parts.recent_detail,
        # `global` is never handed its own facts as reference: they are already its
        # `existing_facts`, and offering them twice only invites the model to restate them.
        global_reference="" if compartment == GLOBAL_COMPARTMENT else parts.global_reference,
        today=run.today,
        compact=compact,
        emit_tone=False,
    )


def global_first(compartments: set[str]) -> list[str]:
    """Orders one run's compartments with `global` leading and the rest by name.

    `global` leads because every later compartment is handed its facts as read-only
    reference, so it must be up to date before they run. Only the ordering is shared; which
    compartments go in is the caller's own question.
    """
    ordered = [GLOBAL_COMPARTMENT] if GLOBAL_COMPARTMENT in compartments else []
    ordered.extend(sorted(compartments - {GLOBAL_COMPARTMENT}))
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
    guild_id = compartment_guild_id(compartment=compartment)
    return f"memory readable only inside Discord server {guild_id}"


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
