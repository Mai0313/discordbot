"""How a raw batch becomes facts: the compartment fan-out and the decision to run it.

Consolidation reads the entries `raw.md` has accumulated for one scope, routes them over the
compartments that may hold them, and asks the model for a delta batch per compartment.
`_consolidate_locked` owns the ordering that makes the boundary structural; the decision of
whether to run at all is `_should_consolidate`, with the cooldown and the refusal counters
beside it.

Three callers, one per way a batch can be raised: a reply turn that just staged evidence
(`consolidate_after_turn`, the only name `pipeline` takes from here), the boot sweep over
every over-threshold scope (`needs_consolidation` then `consolidate_if_needed`), and the
from-scratch rebuild in `regeneration`, which reuses the request vocabulary and the
deletion-only pass but decides its own compartments and never consults the cooldown.

Nothing here imports `pipeline` or `regeneration`. The fan-out is what both of them are
built on rather than a step in either, which is why it left `pipeline.py` in #613 — a turn
meets it at exactly one call, and `regeneration` used to have to import upward to reach it.
"""

import time
import asyncio
from datetime import UTC, datetime

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.memory import MemoryOwner
from discordbot.typings.timeouts import MEMORY_CONSOLIDATE_TIMEOUT_SECONDS
from discordbot.services.memory.tone import update_tone_note
from discordbot.services.memory.facts import MemoryFlavor, parse_identity, sections_for_flavor
from discordbot.services.memory.store import (
    DM_COMPARTMENT,
    GLOBAL_COMPARTMENT,
    clear_raw,
    flavor_of,
    read_facts,
    scope_lock,
    append_detail,
    cleared_since,
    raw_file_bytes,
    scope_owner_id,
    read_detail_tail,
    read_raw_entries,
    count_raw_entries,
    list_compartments,
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
# not need a loop-change reset. Tests clear it through the conftest fixture.
_last_consolidation: dict[str, float] = {}

# Consecutive refused consolidation batches per (scope, compartment). A refusal keeps the
# raw batch for retry, which is right for a transient LLM failure and wrong for a
# mass-delete the model reproduces verbatim: that retries every cooldown forever while the
# scope's memory silently stops updating. Past the threshold the log escalates from warn to
# error, since CONTRIBUTING puts "an unexpected failure that lost a deliverable" at error.
_consecutive_rejections: dict[tuple[str, str], int] = {}
_MAX_QUIET_REJECTIONS = 3


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


async def consolidate_after_turn(
    scope: str, forced: bool, started_at: float, writer: MemoryWriterAI, identity: str
) -> None:
    """Consolidates right after a turn staged evidence, when the backlog warrants it.

    The whole of what a reply turn asks of this module. `pipeline._run_memory_update` already
    holds the scope lock and a semaphore permit, so this is the threshold check and the
    fan-out under them rather than an entry point that takes either — which is also why the
    check is folded in here instead of being read on the caller's side.

    `forced` says the batch carries a forget request; `_should_consolidate` owns what that
    skips. The cooldown is stamped at attempt time, not success time, so repeated LLM
    failures are rate-limited by the same cooldown instead of retrying on every turn.
    """
    if not _should_consolidate(scope=scope, forced=forced):
        return
    _last_consolidation[scope] = time.monotonic()
    await _consolidate_locked(scope=scope, started_at=started_at, writer=writer, identity=identity)


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
