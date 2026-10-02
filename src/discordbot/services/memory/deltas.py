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
        """Whether the batch landed (a batch that changed nothing still counts)."""
        return not self.rejected


def partition_raw_entries(raw_text: str, flavor: MemoryFlavor) -> dict[str, str]:
    """Splits a raw batch into per-compartment texts, keyed by compartment.

    Routing is entirely deterministic: `sharing: global` is cross-server safe and goes
    to `global`, and `source_only` goes to whichever conversation it was learned in.
    Server-flavor observations carry neither field by design (a server memory is one
    guild by construction), so they all land in that scope's single compartment.

    Forget requests are NOT here: they are their own pass, in `partition_forget_requests`,
    for the reason that function gives.

    Each observation keeps the `## <timestamp>` header of the entry it came from, so
    the consolidation prompt still sees dated, oldest-first evidence.
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
    """Splits the forget requests in a raw batch into their own per-compartment texts.

    Deliberately separate from `partition_raw_entries` rather than a bucket alongside the
    observations, because a forget must never share a consolidation call with them. The call
    that carries one is applied with `deletes_only`, and that flag is per CALL: a combined
    bucket on a turn that both remembered and forgot something would leave nothing but a prompt
    line stopping the model writing the forget's own sentence into a compartment it was copied
    into precisely because it could not reach the fact any other way.

    A request is COPIED into every compartment its speaker could read from, since the fact it
    names may be stored in any of them; `_forget_targets` decides which. An empty
    `compartments` yields nothing at all.
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
    """Removes from a raw or detail text the evidence behind the facts a forget deleted.

    Leaving it would hand it straight back: every later consolidation reads the detail tail as
    `<recent_detail>`, and `partition_raw_entries` strips the forget request out of it, so the
    model would see the evidence with nothing saying it had been forgotten.

    `released` maps each compartment to the keys its forget pass freed
    (`DeltaOutcome.released_keys`), and `forgets` is that pass's input. An observation goes when
    it routes to one of those compartments, carries one of its keys as `normalized_key`, and
    predates the newest forget request copied there: something said after every forget is a
    restatement, not what was forgotten. The newest, because one pass cannot tell which of its
    requests deleted which fact, and its requests were made back to back, with nothing said
    between them. Forget requests themselves stay, since `/memory regenerate` replays them.

    Returns `text` itself when nothing matched, so the caller can skip the rewrite.
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
    """Splits a raw batch, in file order, into `(observations, the forget requests after them)`.

    What came before a forget is what it can be about, so it has to be consolidated before that
    forget's pass runs, or the pass finds nothing to delete and the observation right behind it
    stores what the user asked to drop. What came after is a restatement it must not reach, and
    a batch can hold several forgets (a forced run that failed keeps its batch), so each one
    gets its own segment. Only a batch ending in observations has a last segment with no
    forget; a batch without any forget is that one segment.
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
    """Renders a compartment's current facts with their ids, for the model to edit.

    The id leads each entry because it is the only handle an `update` or `delete` delta
    has; everything else is what the model needs to decide whether this batch changes
    the fact at all.
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

    Deletes run before writes so a fact narrowed from one compartment to another can
    only ever be temporarily missing (it re-forms from evidence) instead of temporarily
    present in both — the one ordering that cannot widen a fact's reach.

    `deletes_only` refuses every create and update in the batch, and is set when the bucket
    carried nothing but forget requests; `partition_forget_requests` owns why a forget arrives
    in a call of its own.
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

    An `update` naming an id that is gone becomes a `create` (the fact was aged out or
    the batch is a retry against a changed tree), and a `create` whose evidence keys
    already back an existing fact becomes an `update` of that fact. The second rule is
    what makes a retried batch idempotent: ids are minted from the summary, so a model
    that rewords slightly on the retry would otherwise file a duplicate.

    A delete is resolved by its id alone, BEFORE the section gate: the section a delete
    names is decoration, since the fact being removed already exists and carries its own,
    and a flavor's section vocabulary is not the other's — so gating a delete on it loses
    the deletion outright when the model names a section legal on the other flavor
    (`member_alias` against a user scope). That is the path every `<forget-memory>` runs
    through, where losing the delta means the bot goes on repeating what it was asked to drop.
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

    The field is model-authored free text and only `member_alias` renders it, so on every
    other section a junk id costs the field and nothing else — where raising instead would
    abort the whole fan-out, `apply_deltas` going up past a broad handler that abandons the
    compartments still queued behind it. An alias row that resolves to None is dropped in
    `_resolve_delta` instead, because there the id IS the row.

    The test and the cast live in one place so they cannot disagree, and it takes both
    halves: `isdigit` accepts a "²" that `int()` refuses, and `isdecimal` alone still
    accepts a digit string longer than CPython converts.
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
    """Stamps `last_confirmed` on each fact a batch observed again but did not rewrite.

    Returns how many it stamped.

    A restatement that adds nothing draws no delta, which the prompt prefers, and only a
    written delta stamps the date, so a fact the user keeps repeating would still age out.
    Observed again means one of the fact's keys is the `normalized_key` of an observation in
    `raw_text`, the overlap `_fact_sharing_keys` already reads as the same fact. The `written`
    ids are skipped, since the batch stamped them itself.
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
    """Deletes facts the freshness rules have aged out, returning how many went.

    Two rules, deterministic because `last_confirmed` is code-stamped:

    * a `stable` fact outside the `recent` section is displaced once it falls
      `STABLE_FRESHNESS_WINDOW_DAYS` behind the freshest stable fact in the SAME
      compartment, so a quiet compartment ages nothing and forgets nothing while a busy
      one self-trims;
    * every other fact expires `RECENT_CONTEXT_TTL_DAYS` after it was last confirmed,
      whatever section and durability the model paired it with.

    `permanent` facts, anything filed in the `permanent` section, and member-alias
    rows never age.
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
