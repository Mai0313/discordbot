"""Synchronous partitioning, delta application, and fact aging for one memory compartment.

All operations are synchronous so callers can run them under the scope lock without an `await`
between the `cleared_since` check and disk writes.

* **Partitioning:** splits `raw.md` deterministically by stamped `sharing` and `source` fields.
* **Applying:** drops invalid deltas individually and rejects the batch only on mass deletion.
* **Aging:** displaces `stable` facts relative to the newest `stable` fact in the same
  compartment, and expires `recent`/non-permanent facts by UTC day TTL.
"""

from datetime import datetime, timedelta

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.memory import MemoryFact, MemoryOwner, MemoryFlavor, MemorySection
from discordbot.services.memory.facts import (
    FACT_ID_RE,
    utc_now,
    mint_fact_id,
    node_type_for,
    is_past_recent_ttl,
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
    MAX_NET_FACT_DELETIONS_FLOOR,
    MAX_NET_FACT_DELETIONS_RATIO,
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
        description=(
            "Deltas refused individually: unknown section, empty body, bad id, or one losing to "
            "another write or delete of the same fact."
        ),
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

    Each request is copied into every compartment its speaker could read from (`_forget_targets`).
    Returns an empty dict when `compartments` is empty.

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
    """Returns the compartments a forget request can reach based on its `source` stamp."""
    source = fields_of(block=block).get("source", "")
    # DM and unrecognized sources fail open to every compartment so a stamp change never drops a
    # forget (`dm` also covers group/other DMs because the stamp is coarser than the read side).
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
    """Returns `text` with evidence behind forget-deleted facts removed, or `text` unchanged.

    Removes observations routing to a compartment in `released` whose `normalized_key` is released
    and whose timestamp predates that compartment's newest forget request. Later restatements and
    forget requests themselves are preserved.

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

    Separates observations preceding a forget from later restatements.

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
        header = (
            f"[{fact.fact_id}] section={fact.section} durability={fact.durability} "
            f"last_confirmed={fact.last_confirmed.date().isoformat()}"
        )
        keys = ",".join(fact.keys)
        lines = [header, f"summary: {fact.summary}"]
        if keys:
            lines.append(f"from_keys: {keys}")
        if fact.subject_id is not None:
            lines.append(f"subject_id: {fact.subject_id}")
        lines.append(fact.text)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def net_deletion_ceiling(existing: int) -> int:
    """Returns how many facts one batch may remove net from a compartment holding `existing`.

    Args:
        existing (int): Facts the compartment holds before the batch.
    """
    return max(MAX_NET_FACT_DELETIONS_FLOOR, int(existing * MAX_NET_FACT_DELETIONS_RATIO))


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

    Applies deletes before writes. When two writes target one id, the permanent one wins (else the
    later one); when a fact is both rewritten and deleted, the rewrite wins. Dropped deltas are
    logged and counted.

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
    to_write: dict[str, MemoryFact] = {}
    # Pre-claim every existing id named in this batch so creates never re-aim onto them.
    claimed = {delta.fact_id.strip() for delta in deltas} & existing.keys()
    for delta in deltas:
        if deletes_only and delta.action != "delete":
            logfire.warn(
                "Memory delta writes into a forget-only batch; dropping", action=delta.action
            )
            dropped += 1
            continue
        resolved = _resolve_delta(
            delta=delta,
            compartment=compartment,
            existing=existing,
            allowed=allowed,
            claimed=claimed,
        )
        if resolved is None:
            dropped += 1
            continue
        target_id, is_delete = resolved
        if is_delete:
            to_delete.add(target_id)
            continue
        claimed.add(target_id)
        previous = existing.get(target_id)
        fact = MemoryFact(
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
            # Union keys rather than replacing so retries can still match the fact.
            keys=_merged_keys(delta=delta, previous=previous),
        )
        kept = to_write.get(target_id)
        if kept is not None:
            fact = _settle_collision(scope=scope, earlier=kept, later=fact)
            dropped += 1
        to_write[target_id] = fact
    for fact_id in sorted(to_delete & to_write.keys()):
        logfire.warn(
            "Memory delta deletes a fact the same batch rewrites; keeping the rewrite",
            scope=scope,
            compartment=compartment,
            fact_id=fact_id,
        )
        dropped += 1
    to_delete -= to_write.keys()
    written_ids = set(to_write)
    created = len(written_ids - existing.keys())
    net_loss = len(to_delete) - created
    if not allow_mass_delete and net_loss > net_deletion_ceiling(existing=len(existing)):
        # Net (not raw) deletes so merging duplicates is not rejected as mass deletion.
        return DeltaOutcome(dropped=dropped, rejected="mass deletion")
    for fact_id in sorted(to_delete):
        delete_fact(scope=scope, compartment=compartment, fact_id=fact_id)
    for fact in to_write.values():
        write_fact(scope=scope, fact=fact)
    kept_keys = {key for fact in to_write.values() for key in fact.keys}
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
    claimed: set[str],
) -> tuple[str, bool] | None:
    """Resolves one delta to `(fact_id, is_delete)`, or None when invalid."""
    named_id = delta.fact_id.strip()
    known = named_id if FACT_ID_RE.match(named_id) and named_id in existing else ""
    if delta.action == "delete":
        # Resolve deletes before the section gate; a delete's section is unused.
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
    matched = _fact_sharing_keys(delta=delta, existing=existing, claimed=claimed)
    if matched is not None:
        return matched, False
    return mint_fact_id(compartment=compartment, summary=delta.summary), False


def _subject_id_of(delta: MemoryFactDelta) -> int | None:
    """Returns the integer member id from `delta.subject_id`, or None when invalid."""
    # isdecimal rejects superscripts like "²"; int() also guards CPython's digit limit.
    if not delta.subject_id.isdecimal():
        return None
    try:
        return int(delta.subject_id)
    except ValueError:
        return None


def _delta_body(delta: MemoryFactDelta) -> str:
    """Returns the body text for `delta`, rendering `member_alias` rows from structured fields."""
    if delta.section == "member_alias":
        return render_member_alias_text(display_name=delta.display_name, aliases=delta.aliases)
    return delta.text.strip()


def _merged_keys(delta: MemoryFactDelta, previous: MemoryFact | None) -> tuple[str, ...]:
    """Unions a delta's evidence keys with whatever the fact already carried."""
    existing_keys = previous.keys if previous is not None else ()
    return tuple(sorted({*existing_keys, *(key for key in delta.from_keys if key)}))


def _fact_sharing_keys(
    delta: MemoryFactDelta, existing: dict[str, MemoryFact], claimed: set[str]
) -> str | None:
    """Returns an existing fact id outside `claimed` whose evidence keys overlap this delta's."""
    if not delta.from_keys:
        return None
    wanted = set(delta.from_keys)
    for fact_id, fact in existing.items():
        if fact_id not in claimed and wanted & set(fact.keys):
            return fact_id
    return None


def _settle_collision(scope: str, earlier: MemoryFact, later: MemoryFact) -> MemoryFact:
    """Returns the winning write for a colliding fact id (permanent wins, else later)."""
    kept, lost = later, earlier
    if _is_permanent(fact=earlier) and not _is_permanent(fact=later):
        kept, lost = earlier, later
    logfire.warn(
        "Memory delta lands on a fact the same batch already writes; dropping one",
        scope=scope,
        compartment=kept.compartment,
        fact_id=kept.fact_id,
        kept_durability=kept.durability,
        dropped_durability=lost.durability,
    )
    return kept


def _is_permanent(fact: MemoryFact) -> bool:
    """Whether `fact` is one the sweep never ages, an enforced directive included."""
    return fact.durability == "permanent" or fact.section == "permanent"


def reconfirm_facts(scope: str, compartment: str, raw_text: str, written: tuple[str, ...]) -> int:
    """Refreshes `last_confirmed` on facts restated in `raw_text` that `written` did not rewrite.

    Matches existing facts whose `keys` overlap any `normalized_key` in `raw_text`.

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


def date_facts_from_evidence(scope: str, compartment: str, raw_text: str) -> int:
    """Re-dates facts in `compartment` from observations in `raw_text` sharing their keys.

    Sets `last_confirmed` to the newest matching observation timestamp and `created` to the oldest
    (preserving any earlier `created`). Facts with no matching key take the newest timestamp in
    `raw_text` for both.

    Args:
        scope (str): Memory scope key.
        compartment (str): Compartment whose facts are re-dated.
        raw_text (str): Evidence routed to this compartment.

    Returns:
        Number of facts dated by key overlap rather than the fallback newest timestamp.
    """
    stamps: dict[str, list[datetime]] = {}
    every: list[datetime] = []
    for timestamp, block in iter_observations(text=raw_text):
        stamp = datetime.fromisoformat(timestamp)
        every.append(stamp)
        key = fields_of(block=block).get("normalized_key")
        if key:
            stamps.setdefault(key, []).append(stamp)
    newest = [max(every)] if every else []
    keyed = 0
    for fact in read_facts(scope=scope, compartment=compartment):
        times = [stamp for key in fact.keys for stamp in stamps.get(key, ())]
        keyed += bool(times)
        times = times or newest
        if not times:
            continue
        dates = {"last_confirmed": max(times), "created": min(fact.created, *times)}
        write_fact(scope=scope, fact=fact.model_copy(update=dates))
    return keyed


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
            # Exempt section == "permanent" too so a mismatched standing directive never ages out.
            continue
        if fact.durability == "stable" and fact.section != "recent" and latest_stable is not None:
            expired = latest_stable - fact.last_confirmed > timedelta(
                days=STABLE_FRESHNESS_WINDOW_DAYS
            )
        else:
            # Same date predicate used on read so hidden facts match swept facts.
            expired = is_past_recent_ttl(fact=fact, today=today)
        if expired and delete_fact(scope=scope, compartment=compartment, fact_id=fact.fact_id):
            removed += 1
    return removed


def _compartment_for_block(block: str) -> str:
    """Routes one observation block to its compartment from its stamped fields."""
    fields = fields_of(block=block)
    if fields.get("sharing") != "source_only":
        # Missing or non-source_only sharing defaults to global.
        return GLOBAL_COMPARTMENT
    source = fields.get("source", "")
    if source == "dm":
        return DM_COMPARTMENT
    guild_id = source_guild_id(source=source)
    if guild_id is not None:
        return guild_compartment(guild_id=guild_id)
    # Confine source_only with an unrecognized source to DM rather than leaking into global.
    return DM_COMPARTMENT
