"""Turning one compartment's consolidation deltas into files, and aging what is left.

Three jobs live here, all pure-plus-store and synchronous so the caller can hold them
inside the scope lock without an await in between (the `cleared_since` guard depends on
that):

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
* **Aging.** `last_confirmed` is code-stamped, so the freshness rules that used to be
  prose in the consolidation prompt are a sweep here. Stable facts age by displacement
  against the freshest fact *in the same compartment*, so an active guild cannot evict
  the memory of one the user visits less often.
"""

import re
from datetime import UTC, datetime, timedelta

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.memory import MemoryFact, MemoryOwner, MemorySection
from discordbot.services.memory.facts import (
    FACT_ID_RE,
    MemoryFlavor,
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
from discordbot.services.memory.writer import FORGET_REQUEST_CATEGORY, MemoryFactDelta
from discordbot.services.memory.constants import (
    RECENT_CONTEXT_TTL_DAYS,
    MAX_NET_FACT_DELETIONS_FLOOR,
    STABLE_FRESHNESS_WINDOW_DAYS,
)

# One raw entry's `## <ISO timestamp>` header, and one observation block inside it.
_ENTRY_HEADER_RE = re.compile(r"^## (?P<timestamp>\d{4}-\d{2}-\d{2}T\S+)\s*$")
_OBSERVATION_HEADER_RE = re.compile(r"^### (?P<category>\S+)")
# Observation categories that carry how the user wants the bot to SOUND. Everything
# else is a fact and has no business in the always-injected tone note.
_TONE_CATEGORIES = frozenset({"interaction_style", "stable_preference"})
_FIELD_RE = re.compile(r"^\s*-\s*(?P<name>[a-z_]+):\s*(?P<value>.*?)\s*$")
_GUILD_SOURCE_RE = re.compile(r"^guild (?P<guild_id>\d+)$")


class DeltaOutcome(BaseModel):
    """What one compartment's delta batch did.

    Attributes:
        created: Facts written that did not exist before.
        updated: Existing facts rewritten in place.
        deleted: Facts removed.
        dropped: Deltas refused individually (unknown section, empty body, bad id).
        rejected: Why the whole batch was refused, or "" when it was applied.
        written: Ids this batch created or updated, so a rebuild can drop the rest.
    """

    model_config = ConfigDict(frozen=True)

    created: int = Field(default=0, description="Facts written that did not exist before.")
    updated: int = Field(default=0, description="Existing facts rewritten in place.")
    deleted: int = Field(default=0, description="Facts removed.")
    dropped: int = Field(default=0, description="Deltas refused individually.")
    rejected: str = Field(default="", description="Why the batch was refused; empty when applied.")
    written: tuple[str, ...] = Field(
        default=(),
        description="Ids this batch created or updated, so a rebuild can drop the rest.",
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
    for timestamp, block in _iter_observations(text=raw_text):
        if _is_forget_request(block=block):
            continue
        compartment = (
            GLOBAL_COMPARTMENT if flavor == "server" else _compartment_for_block(block=block)
        )
        buckets.setdefault(compartment, []).append((timestamp, block))
    return {compartment: _render_entries(blocks=blocks) for compartment, blocks in buckets.items()}


def partition_forget_requests(raw_text: str, compartments: tuple[str, ...]) -> dict[str, str]:
    """Splits the forget requests in a raw batch into their own per-compartment texts.

    Deliberately separate from `partition_raw_entries` rather than a bucket alongside the
    observations, because a forget must never share a consolidation call with them. The call
    that carries one is applied with `deletes_only`, and that flag is per call: on a turn that
    both remembered and forgot something, a combined bucket would leave nothing but a prompt
    line stopping the model from writing the forget's own sentence into a compartment it was
    copied into precisely because it could not reach the fact any other way. Keeping the two
    apart costs one extra call on a mixed turn and keeps the guarantee structural.

    A request is COPIED into every compartment its speaker could read from, since the fact it
    names may be stored in any of them; `_forget_targets` decides which. An empty
    `compartments` yields nothing at all.
    """
    buckets: dict[str, list[tuple[str, str]]] = {}
    for timestamp, block in _iter_observations(text=raw_text):
        if not _is_forget_request(block=block):
            continue
        for compartment in _forget_targets(block=block, compartments=compartments):
            buckets.setdefault(compartment, []).append((timestamp, block))
    return {compartment: _render_entries(blocks=blocks) for compartment, blocks in buckets.items()}


def _is_forget_request(block: str) -> bool:
    """Whether one raw block is a forget request rather than an observation."""
    header = _OBSERVATION_HEADER_RE.match(block)
    return header is not None and header.group("category") == FORGET_REQUEST_CATEGORY


def _forget_targets(block: str, compartments: tuple[str, ...]) -> tuple[str, ...]:
    """Which compartments one forget request is copied into.

    A forget can only sensibly name a fact its speaker could see, and what they can see is
    exactly what `compartments_for_reading` injects: in a guild, the shared compartment plus
    that guild's own; in the owner's own DMs, everything, since their whole memory is readable
    there. Copying wider would let a forget spoken in one guild reach a fact stored for
    another, and copying narrower would leave the ordinary case, forgetting something the bot
    just told them, unable to reach a fact that happens to live in `global/`.
    """
    source = _fields_of(block=block).get("source", "")
    if source == "dm" or not source:
        return compartments
    match = _GUILD_SOURCE_RE.match(source)
    if match is None:
        return compartments
    guild = guild_compartment(guild_id=int(match.group("guild_id")))
    return tuple(
        compartment for compartment in compartments if compartment in {GLOBAL_COMPARTMENT, guild}
    )


def tone_evidence_from_raw(raw_text: str) -> str:
    """Returns the whole batch's tone-bearing observations, ignoring compartments.

    Tone is the one tier that is cross-server safe by construction, so it must not be
    partitioned: nearly half of all observations are `source_only`, and a bucket-gated
    tone note would simply stop updating for those conversations. Each line carries its
    `evidence_kind` and then the summary, in the order the entries were appended, and
    the prompt is explicit that this block feeds the note alone, that the ordering is
    oldest-first (its "a later stated preference wins" rule has no other clock) and
    that the tag is not to be copied into the note.

    The kind is what tells a preference the user stated apart from one inferred off
    their own behaviour, and the note is a merge of many batches, so without it every
    bullet reads alike and the note converges on whichever reading has the most bullets.
    That is not hypothetical: a user who asked in DM to be addressed respectfully, then
    trash-talked the bot across a guild for weeks, ended up with a note saying they
    wanted trash-talk back. The one stated preference lost to five inferred ones, and
    the compartment calls never had this problem because `<raw_entries>` carries the
    kind to them already.
    """
    lines: list[str] = []
    for _, block in _iter_observations(text=raw_text):
        # The category is the block's `### <category>` header, not one of its fields.
        header = _OBSERVATION_HEADER_RE.match(block)
        if header is None or header.group("category") not in _TONE_CATEGORIES:
            continue
        fields = _fields_of(block=block)
        summary = fields.get("summary_zh", "")
        if summary:
            lines.append(f"* [{fields.get('evidence_kind', 'unknown')}] {summary}")
    return "\n".join(lines)


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
    carried nothing but forget requests. It is what makes a broadcast forget structurally safe
    rather than safe by prompt: the compartments it reaches beyond the one holding the fact are
    handed a sentence that may be `source_only`, and this stops any of them writing it down
    however the model reads it.
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
        # consolidation's whole job, and the median scope holds a handful of facts, so
        # a raw-delete ceiling would refuse the common case.
        return DeltaOutcome(dropped=dropped, rejected="mass deletion")
    for fact_id in sorted(to_delete):
        delete_fact(scope=scope, compartment=compartment, fact_id=fact_id)
    for fact in to_write:
        write_fact(scope=scope, fact=fact)
    return DeltaOutcome(
        created=created,
        updated=len(to_write) - created,
        deleted=len(to_delete),
        dropped=dropped,
        written=tuple(sorted(written_ids)),
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
    """
    if delta.section not in allowed:
        logfire.warn("Memory delta names an unknown section; dropping", section=delta.section)
        return None
    named_id = delta.fact_id.strip()
    known = named_id if FACT_ID_RE.match(named_id) and named_id in existing else ""
    if delta.action == "delete":
        return (known, True) if known else None
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

    The field is model-authored free text and only `member_alias` renders it
    (`facts.py::_render_fact_line`), so on every other section a junk id costs the field
    and nothing else — where casting it unguarded cost the whole fan-out, `apply_deltas`
    raising past a broad handler that abandons the compartments still queued behind it
    (#527). An alias row that resolves to None is dropped in `_resolve_delta` instead,
    because there the id IS the row.

    The test and the cast live in one place so they cannot disagree again, and it takes
    both halves: `isdigit` accepts a "²" that `int()` refuses, and `isdecimal` alone still
    accepts a digit string longer than CPython converts.
    """
    if not delta.subject_id.isdecimal():
        return None
    try:
        return int(delta.subject_id)
    except ValueError:
        # Past the int-conversion limit, the same way `utils/amount_parsing.py` handles it.
        return None


def _delta_body(delta: MemoryFactDelta) -> str:
    """Returns the body this delta writes, which for an alias row the code renders itself.

    The model's `text` is not read for that section at all: it is asked for the member's
    name and aliases as fields instead, so the row cannot come out as a sentence with a
    personal aside attached to it (#464).
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


def sweep_stale_facts(scope: str, compartment: str, today: datetime) -> int:
    """Deletes facts the freshness rules have aged out, returning how many went.

    Two rules, both formerly prose in the consolidation prompt and now deterministic
    because the dates are code-stamped:

    * a `recent` fact expires `RECENT_CONTEXT_TTL_DAYS` after it was last confirmed;
    * a `stable` fact is displaced once it falls `STABLE_FRESHNESS_WINDOW_DAYS` behind
      the freshest stable fact in the SAME compartment, so a quiet compartment ages
      nothing and forgets nothing while a busy one self-trims.

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
        if fact.section == "recent":
            expired = today - fact.last_confirmed > timedelta(days=RECENT_CONTEXT_TTL_DAYS)
        elif fact.durability == "stable" and latest_stable is not None:
            expired = latest_stable - fact.last_confirmed > timedelta(
                days=STABLE_FRESHNESS_WINDOW_DAYS
            )
        else:
            expired = False
        if expired and delete_fact(scope=scope, compartment=compartment, fact_id=fact.fact_id):
            removed += 1
    return removed


def _compartment_for_block(block: str) -> str:
    """Routes one observation block to its compartment from its stamped fields."""
    fields = _fields_of(block=block)
    if fields.get("sharing") != "source_only":
        # `global`, and anything predating the stamped fields, is cross-server safe.
        return GLOBAL_COMPARTMENT
    source = fields.get("source", "")
    if source == "dm":
        return DM_COMPARTMENT
    match = _GUILD_SOURCE_RE.match(source)
    if match is not None:
        return guild_compartment(guild_id=int(match.group("guild_id")))
    # `source_only` with no usable source cannot be placed in a guild, and putting it
    # in `global` would publish exactly what the flag asked to confine, so it goes to
    # the owner's own DMs — visible to them alone.
    return DM_COMPARTMENT


def _fields_of(block: str) -> dict[str, str]:
    """Extracts one observation block's `- name: value` fields."""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        match = _FIELD_RE.match(line)
        if match is not None:
            fields[match.group("name")] = match.group("value")
    return fields


def _iter_observations(text: str) -> list[tuple[str, str]]:
    """Splits a raw or detail file into `(entry timestamp, observation block)` pairs."""
    pairs: list[tuple[str, str]] = []
    timestamp = ""
    current: list[str] = []

    def flush() -> None:
        block = "\n".join(current).strip()
        if block:
            pairs.append((timestamp, block))
        current.clear()

    for line in text.splitlines():
        header = _ENTRY_HEADER_RE.match(line)
        if header is not None:
            flush()
            timestamp = header.group("timestamp")
            continue
        if _OBSERVATION_HEADER_RE.match(line):
            flush()
        current.append(line)
    flush()
    return pairs


def _render_entries(blocks: list[tuple[str, str]]) -> str:
    """Re-renders bucketed observation blocks under their original entry headers."""
    rendered: list[str] = []
    previous = ""
    for timestamp, block in blocks:
        if timestamp and timestamp != previous:
            rendered.append(f"## {timestamp}")
            previous = timestamp
        rendered.append(block)
    return "\n\n".join(rendered)


def today_utc() -> datetime:
    """Returns the current UTC day boundary used by the freshness sweep."""
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
