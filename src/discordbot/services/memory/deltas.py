"""Turning one compartment's consolidation deltas into files, and aging what is left.

Everything here is pure-plus-store and synchronous, so the caller can hold it inside the
scope lock without an await in between (the `cleared_since` guard depends on that). The
main jobs:

* **Partitioning.** `raw.md` stays one file per scope — it is staging, never injected,
  and splitting it would multiply job rows and cooldowns by the number of guilds for
  nothing. So the fan-out partitions it here instead, deterministically, off the
  `- sharing:` and `- source:` fields code already stamps onto every observation. The
  model never sees a compartment it is not writing.
* **Applying.** Deltas are validated one at a time and a bad one is dropped rather than
  failing the batch. That asymmetry is deliberate: a whole-batch rejection is only a
  retry if the next run would produce something different, and a deterministic content
  check re-run against the same raw batch and the same existing facts will not, so it
  would freeze the scope's memory permanently while burning a consolidation call every
  cooldown. Only shape failures — the call itself failing, or a mass deletion — reject.
* **Aging.** `last_confirmed` is code-stamped, so the freshness rules are a deterministic
  sweep here rather than something a prompt has to re-apply. Stable facts age by displacement
  against the freshest fact *in the same compartment*, so an active guild cannot evict
  the memory of one the user visits less often.
"""

from datetime import UTC, datetime, timedelta

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.memory import MemoryFact, MemoryOwner, MemoryFlavor, MemorySection
from discordbot.services.memory.facts import (
    FACT_ID_RE,
    utc_now,
    mint_fact_id,
    node_type_for,
    sections_for_flavor,
    render_member_alias_text,
)
from discordbot.services.memory.store import (
    DM_COMPARTMENT,
    GLOBAL_COMPARTMENT,
    read_facts,
    write_fact,
    delete_fact,
    guild_compartment,
)
from discordbot.services.memory.writer import MemoryFactDelta
from discordbot.services.memory.constants import (
    RECENT_CONTEXT_TTL_DAYS,
    MAX_NET_FACT_DELETIONS_FLOOR,
    STABLE_FRESHNESS_WINDOW_DAYS,
)
from discordbot.services.memory.raw_entries import (
    fields_of,
    newest_stamp,
    render_entries,
    source_guild_id,
    is_forget_request,
    iter_observations,
    render_file_entries,
)


class DeltaOutcome(BaseModel):
    """What one compartment's delta batch did."""

    model_config = ConfigDict(frozen=True)

    created: int = Field(default=0, description="Facts written that did not exist before.")
    updated: int = Field(default=0, description="Existing facts rewritten in place.")
    deleted: int = Field(default=0, description="Facts removed.")
    dropped: int = Field(
        default=0,
        description="Deltas refused individually: unknown section, empty body, or bad id.",
    )
    rejected: str = Field(default="", description="Why the batch was refused; empty when applied.")
    written: tuple[str, ...] = Field(
        default=(),
        description="Ids this batch created or updated, so a rebuild can drop the rest.",
    )
    released_keys: tuple[str, ...] = Field(
        default=(),
        description=(
            "Evidence keys the deleted facts carried that no fact left in the compartment "
            "carries, so a forget can take the evidence with the fact."
        ),
    )

    @property
    def applied(self) -> bool:
        """Returns whether the batch landed (a batch that changed nothing still counts)."""
        return not self.rejected


def partition_raw_entries(raw_text: str, flavor: MemoryFlavor) -> dict[str, str]:
    """Returns `raw_text` split into per-compartment observation texts keyed by compartment.

    `sharing: global` routes to `global` and `source_only` routes to the conversation source it
    was learned in; server-flavor observations route to `global`. Forget requests are excluded
    and handled separately by `partition_forget_requests`. Each observation keeps its
    `## <timestamp>` header.

    Args:
        raw_text (str): Raw or detail entries to partition.
        flavor (MemoryFlavor): Scope flavor (`'user'` or `'server'`).
    """
    buckets: dict[str, list[tuple[str, str]]] = {}
    for timestamp, block in iter_observations(text=raw_text):
        if is_forget_request(block=block):
            continue
        compartment = (
            GLOBAL_COMPARTMENT if flavor == "server" else _compartment_for_block(block=block)
        )
        buckets.setdefault(compartment, []).append((timestamp, block))
    return {compartment: render_entries(blocks=blocks) for compartment, blocks in buckets.items()}


def partition_forget_requests(raw_text: str, compartments: tuple[str, ...]) -> dict[str, str]:
    """Returns the forget requests in `raw_text` split into per-compartment texts.

    Kept separate from `partition_raw_entries` so a forget never shares a consolidation call
    with observations: a forget call runs with `deletes_only` per call, preventing a turn that
    both remembered and forgot something from writing the forget's own sentence into another
    compartment. Each request is copied into every compartment its speaker could read from
    (`_forget_targets`); an empty `compartments` yields an empty dict.

    Args:
        raw_text (str): Raw entries containing forget requests.
        compartments (tuple[str, ...]): Existing compartments for the scope.
    """
    buckets: dict[str, list[tuple[str, str]]] = {}
    for timestamp, block in iter_observations(text=raw_text):
        if not is_forget_request(block=block):
            continue
        for compartment in _forget_targets(block=block, compartments=compartments):
            buckets.setdefault(compartment, []).append((timestamp, block))
    return {compartment: render_entries(blocks=blocks) for compartment, blocks in buckets.items()}


def _forget_targets(block: str, compartments: tuple[str, ...]) -> tuple[str, ...]:
    """Which compartments one forget request is copied into.

    A forget can only sensibly name a fact its speaker could see, so the copy follows the
    `source` stamp: under `guild <id>`, the shared compartment plus that guild's own; under
    `dm`, everything, since the owner's own DMs read their whole memory. Copying wider would let
    a forget spoken in one guild reach a fact stored for another, and copying narrower would
    leave the ordinary case, forgetting something the bot just told them, unable to reach a
    fact that happens to live in `global/`.

    The stamp is coarser than the read side: `dm` also stamps an `/ask` turn in a group DM or
    someone else's DM, where the reply read only `global`, and a forget spoken there is still
    copied everywhere.

    A `source` that is neither of those shapes falls open to every compartment rather than to
    none, so a stamping change cannot silently drop a user's forget on the floor. Code writes
    that field and only ever writes those two shapes, so the branch is unreachable today; it is
    the direction to fail in, not a case being handled.
    """
    source = fields_of(block=block).get("source", "")
    if source == "dm" or not source:
        return compartments
    guild_id = source_guild_id(source=source)
    if guild_id is None:
        return compartments
    guild = guild_compartment(guild_id=guild_id)
    return tuple(
        compartment for compartment in compartments if compartment in {GLOBAL_COMPARTMENT, guild}
    )


def drop_released_evidence(
    text: str, released: dict[str, tuple[str, ...]], forgets: dict[str, str]
) -> str:
    """Returns `text` with the evidence behind forget-deleted facts removed, or `text` unchanged.

    Every later consolidation reads the detail tail as `<recent_detail>` with forget requests
    stripped out, so leaving the released observations would hand forgotten evidence back to the
    model. An observation is removed when it routes to a compartment in `released`, carries one
    of that compartment's released `normalized_key`s, and predates the newest forget request
    copied there; observations after the cutoff are restatements and stay, as do forget
    requests themselves for `/memory regenerate` replay.

    Args:
        text (str): Raw or detail text to filter.
        released (dict[str, tuple[str, ...]]): Released `normalized_key`s per compartment.
        forgets (dict[str, str]): Per-compartment forget-request text from the pass.
    """
    cutoffs = {
        compartment: newest_stamp(text=forgets[compartment])
        for compartment, keys in released.items()
        if keys
    }
    pairs = iter_observations(text=text)
    kept = [
        (timestamp, block)
        for timestamp, block in pairs
        if not _is_released(timestamp=timestamp, block=block, released=released, cutoffs=cutoffs)
    ]
    if len(kept) == len(pairs):
        return text
    return render_file_entries(pairs=kept)


def forget_segments(raw_text: str) -> list[tuple[str, str]]:
    """Returns `raw_text` split in file order into `(observations, trailing_forgets)` segments.

    Observations before a forget must be consolidated before that forget's pass runs so the
    pass can delete the resulting facts, while observations after it are restatements it must
    not reach.

    Args:
        raw_text (str): Raw or evidence text to split.
    """
    segments: list[tuple[str, str]] = []
    observations: list[tuple[str, str]] = []
    forgets: list[tuple[str, str]] = []
    for pair in iter_observations(text=raw_text):
        if is_forget_request(block=pair[1]):
            forgets.append(pair)
            continue
        if forgets:
            segments.append((
                render_file_entries(pairs=observations),
                render_file_entries(pairs=forgets),
            ))
            observations, forgets = [], []
        observations.append(pair)
    segments.append((render_file_entries(pairs=observations), render_file_entries(pairs=forgets)))
    return segments


def _is_released(
    timestamp: str, block: str, released: dict[str, tuple[str, ...]], cutoffs: dict[str, str]
) -> bool:
    """Whether one observation is evidence a forget released; see `drop_released_evidence`."""
    if is_forget_request(block=block):
        return False
    compartment = _compartment_for_block(block=block)
    return (
        fields_of(block=block).get("normalized_key") in released.get(compartment, ())
        and timestamp < cutoffs[compartment]
    )


def render_existing_facts(facts: list[MemoryFact]) -> str:
    """Returns a compartment's current facts rendered with their `[<fact_id>]` headers.

    Args:
        facts (list[MemoryFact]): Stored facts in the compartment.
    """
    blocks: list[str] = []
    for fact in sorted(facts, key=lambda item: (item.section, item.fact_id)):
        header = f"[{fact.fact_id}] section={fact.section} durability={fact.durability}"
        keys = ",".join(fact.keys)
        lines = [header, f"summary: {fact.summary}"]
        if keys:
            lines.append(f"from_keys: {keys}")
        if fact.subject_id is not None:
            lines.append(f"subject_id: {fact.subject_id}")
        lines.append(fact.text)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def apply_deltas(  # noqa: PLR0913 -- one compartment's identity (scope/compartment/flavor) plus the batch, its stamp, and the two write exemptions
    scope: str,
    compartment: str,
    flavor: MemoryFlavor,
    deltas: tuple[MemoryFactDelta, ...],
    owner: MemoryOwner,
    allow_mass_delete: bool,
    deletes_only: bool = False,
) -> DeltaOutcome:
    """Validates and applies one compartment's delta batch.

    Deletes run before writes so a fact narrowed from one compartment to another is never
    present in both at once.

    Args:
        scope (str): Memory scope key.
        compartment (str): Compartment being modified.
        flavor (MemoryFlavor): Scope flavor (`'user'` or `'server'`).
        deltas (tuple[MemoryFactDelta, ...]): Candidate deltas emitted by consolidation.
        owner (MemoryOwner): Owner identity stamped onto created and updated facts.
        allow_mass_delete (bool): Whether to bypass the net-deletion ceiling.
        deletes_only (bool): Whether to drop every non-delete delta in the batch.

    Returns:
        Outcome counts, written fact ids, and any evidence keys released by deletions.
    """
    existing = {fact.fact_id: fact for fact in read_facts(scope=scope, compartment=compartment)}
    allowed = sections_for_flavor(flavor=flavor)
    now = utc_now()
    dropped = 0
    to_delete: set[str] = set()
    to_write: list[MemoryFact] = []
    for delta in deltas:
        if deletes_only and delta.action != "delete":
            logfire.warn(
                "Memory delta writes into a forget-only batch; dropping", action=delta.action
            )
            dropped += 1
            continue
        resolved = _resolve_delta(
            delta=delta, compartment=compartment, existing=existing, allowed=allowed
        )
        if resolved is None:
            dropped += 1
            continue
        target_id, is_delete = resolved
        if is_delete:
            to_delete.add(target_id)
            continue
        to_delete.discard(target_id)
        previous = existing.get(target_id)
        to_write.append(
            MemoryFact(
                fact_id=target_id,
                summary=" ".join(delta.summary.split()),
                section=delta.section,
                durability=delta.durability,
                text=_delta_body(delta=delta),
                compartment=compartment,
                owner_id=owner.owner_id,
                owner_name=owner.owner_name,
                subject_id=_subject_id_of(delta=delta),
                node_type=node_type_for(section=delta.section),
                created=previous.created if previous is not None else now,
                last_confirmed=now,
                # Unioned, never replaced: the keys are what lets a retried batch
                # recognise this fact again, so a rewrite that cites fewer of them must
                # not shrink the handle it will be found by next time.
                keys=_merged_keys(delta=delta, previous=previous),
            )
        )
    written_ids = {fact.fact_id for fact in to_write}
    created = len(written_ids - existing.keys())
    net_loss = len(to_delete) - created
    ceiling = max(MAX_NET_FACT_DELETIONS_FLOOR, len(existing) // 2)
    if not allow_mass_delete and net_loss > ceiling:
        # Net rather than raw deletes: merging several near-duplicates into one is
        # consolidation's whole job, so a raw-delete ceiling would refuse the common case.
        return DeltaOutcome(dropped=dropped, rejected="mass deletion")
    for fact_id in sorted(to_delete):
        delete_fact(scope=scope, compartment=compartment, fact_id=fact_id)
    for fact in to_write:
        write_fact(scope=scope, fact=fact)
    kept_keys = {key for fact in to_write for key in fact.keys}
    kept_keys.update(
        key for fact_id, fact in existing.items() if fact_id not in to_delete for key in fact.keys
    )
    return DeltaOutcome(
        created=created,
        updated=len(to_write) - created,
        deleted=len(to_delete),
        dropped=dropped,
        written=tuple(sorted(written_ids)),
        released_keys=tuple(
            sorted({key for fact_id in to_delete for key in existing[fact_id].keys} - kept_keys)
        ),
    )


def _resolve_delta(  # noqa: PLR0911 -- one early return per way a delta can be dropped or re-aimed
    delta: MemoryFactDelta,
    compartment: str,
    existing: dict[str, MemoryFact],
    allowed: frozenset[MemorySection],
) -> tuple[str, bool] | None:
    """Resolves one delta to `(fact_id, is_delete)`, or None when it must be dropped.

    An `update` naming an id that is gone becomes a `create`, and a `create` whose evidence keys
    already back an existing fact becomes an `update` of that fact so retried batches stay
    idempotent. A delete is resolved by its id alone before the section gate, since a delete's
    section is unused and gating on it would drop a `<forget-memory>` deletion when the model
    names a section from the other flavor.
    """
    named_id = delta.fact_id.strip()
    known = named_id if FACT_ID_RE.match(named_id) and named_id in existing else ""
    if delta.action == "delete":
        return (known, True) if known else None
    if delta.section not in allowed:
        logfire.warn("Memory delta names an unknown section; dropping", section=delta.section)
        return None
    if not delta.summary.strip() or not _delta_body(delta=delta):
        logfire.warn("Memory delta carries no content; dropping", action=delta.action)
        return None
    if delta.section == "member_alias" and _subject_id_of(delta=delta) is None:
        logfire.warn("Member-alias delta carries no member id; dropping")
        return None
    if known:
        return known, False
    matched = _fact_sharing_keys(delta=delta, existing=existing)
    if matched is not None:
        return matched, False
    return mint_fact_id(compartment=compartment, summary=delta.summary), False


def _subject_id_of(delta: MemoryFactDelta) -> int | None:
    """Returns the member id this delta names, or None when it names nothing usable.

    Both `isdecimal` and `int()` are checked: `isdigit` accepts superscripts like `"²"` that
    `int()` refuses, and `isdecimal` alone accepts digit strings longer than CPython converts.
    """
    if not delta.subject_id.isdecimal():
        return None
    try:
        return int(delta.subject_id)
    except ValueError:
        # Past the int-conversion limit.
        return None


def _delta_body(delta: MemoryFactDelta) -> str:
    """Returns the body this delta writes, which for an alias row the code renders itself.

    The model's `text` is not read for that section at all: it is asked for the member's
    name and aliases as fields instead, so the row cannot come out as a sentence with a
    personal aside attached to it.
    """
    if delta.section == "member_alias":
        return render_member_alias_text(display_name=delta.display_name, aliases=delta.aliases)
    return delta.text.strip()


def _merged_keys(delta: MemoryFactDelta, previous: MemoryFact | None) -> tuple[str, ...]:
    """Unions a delta's evidence keys with whatever the fact already carried."""
    existing_keys = previous.keys if previous is not None else ()
    return tuple(sorted({*existing_keys, *(key for key in delta.from_keys if key)}))


def _fact_sharing_keys(delta: MemoryFactDelta, existing: dict[str, MemoryFact]) -> str | None:
    """Returns an existing fact id whose evidence keys overlap this delta's."""
    if not delta.from_keys:
        return None
    wanted = set(delta.from_keys)
    for fact_id, fact in existing.items():
        if wanted & set(fact.keys):
            return fact_id
    return None


def reconfirm_facts(scope: str, compartment: str, raw_text: str, written: tuple[str, ...]) -> int:
    """Stamps `last_confirmed` on facts restated in `raw_text` that `written` did not rewrite.

    A restatement that adds nothing emits no delta, so matching `normalized_key` overlap keeps
    repeated facts from aging out.

    Args:
        scope (str): Memory scope key.
        compartment (str): Compartment whose facts are checked.
        raw_text (str): Raw observations from the consolidated batch.
        written (tuple[str, ...]): Fact ids already written by `apply_deltas` in this pass.

    Returns:
        Number of existing facts whose `last_confirmed` timestamp was refreshed.
    """
    observed = {
        fields_of(block=block).get("normalized_key")
        for _, block in iter_observations(text=raw_text)
    }
    now = utc_now()
    reconfirmed = 0
    for fact in read_facts(scope=scope, compartment=compartment):
        if fact.fact_id not in written and observed & set(fact.keys):
            write_fact(scope=scope, fact=fact.model_copy(update={"last_confirmed": now}))
            reconfirmed += 1
    return reconfirmed


def sweep_stale_facts(scope: str, compartment: str, today: datetime) -> int:
    """Deletes facts aged out by the freshness rules, returning how many were removed.

    `stable` facts outside `recent` age out once they fall `STABLE_FRESHNESS_WINDOW_DAYS`
    behind the freshest `stable` fact in the same compartment; other non-permanent facts
    expire `RECENT_CONTEXT_TTL_DAYS` after `last_confirmed`. `permanent` facts, facts in the
    `permanent` section, and `member_alias` rows never age.

    Args:
        scope (str): Memory scope key.
        compartment (str): Compartment to sweep.
        today (datetime): UTC day boundary for TTL comparison.

    Returns:
        Number of expired facts deleted from disk.
    """
    facts = read_facts(scope=scope, compartment=compartment)
    stable = [fact.last_confirmed for fact in facts if fact.durability == "stable"]
    latest_stable = max(stable) if stable else None
    removed = 0
    for fact in facts:
        if (
            fact.durability == "permanent"
            or fact.section == "permanent"
            or fact.node_type == "member_alias"
        ):
            # The section counts as well as the durability: nothing couples the two, and
            # `render_existing_facts` feeds a mismatched pairing back on every later
            # update, so one slip would otherwise age out an enforced standing directive.
            continue
        if fact.durability == "stable" and fact.section != "recent" and latest_stable is not None:
            expired = latest_stable - fact.last_confirmed > timedelta(
                days=STABLE_FRESHNESS_WINDOW_DAYS
            )
        else:
            expired = today - fact.last_confirmed > timedelta(days=RECENT_CONTEXT_TTL_DAYS)
        if expired and delete_fact(scope=scope, compartment=compartment, fact_id=fact.fact_id):
            removed += 1
    return removed


def _compartment_for_block(block: str) -> str:
    """Routes one observation block to its compartment from its stamped fields."""
    fields = fields_of(block=block)
    if fields.get("sharing") != "source_only":
        # Anything not marked `source_only`, a block with no `sharing` field included, is
        # cross-server safe.
        return GLOBAL_COMPARTMENT
    source = fields.get("source", "")
    if source == "dm":
        return DM_COMPARTMENT
    guild_id = source_guild_id(source=source)
    if guild_id is not None:
        return guild_compartment(guild_id=guild_id)
    # `source_only` with no usable source cannot be placed in a guild, and putting it
    # in `global` would publish exactly what the flag asked to confine, so it goes to
    # the owner's own DMs — visible to them alone.
    return DM_COMPARTMENT


def today_utc() -> datetime:
    """Returns the current UTC day boundary used by the freshness sweep."""
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
