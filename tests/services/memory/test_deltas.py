"""Tests for applying consolidation deltas: writes, deletes, forgets, aging and alias rows."""

from datetime import timedelta

import pytest

from discordbot.typings.memory import (
    MemoryOwner,
    MemorySection,
    MemoryDurability,
    MemoryDeltaAction,
)
from discordbot.cogs.gen_reply.recall import allowlist_ids_from_server_memory
from discordbot.services.memory.facts import mint_fact_id
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    read_facts,
    user_scope,
    write_fact,
    guild_compartment,
)
from discordbot.services.memory.deltas import (
    apply_deltas,
    sweep_stale_facts,
    partition_raw_entries,
    render_existing_facts,
    drop_released_evidence,
    partition_forget_requests,
)
from discordbot.services.memory.writer import MemoryFactDelta
from discordbot.services.memory.constants import (
    RECENT_CONTEXT_TTL_DAYS,
    STABLE_FRESHNESS_WINDOW_DAYS,
)
from discordbot.services.memory.raw_entries import iter_observations, render_forget_requests

from tests.helpers.memory import STAMPED_AT, make_delta
from tests.helpers.user_memory import (
    _OWNER,
    USER_ID,
    USER_SCOPE,
    _RAW_BATCH,
    _fact,
    _entry,
    _observation,
    _stored_fact,
    _forget_entry,
)
from tests.helpers.server_memory import (
    SERVER_OWNER,
    SERVER_SCOPE,
    _alias_fact,
    _server_fact,
    _server_document,
)


async def test_forget_reaches_a_fact_stored_in_another_compartment() -> None:
    """A forget spoken in a guild has to be able to delete a fact stored in `global/`.

    Routing it by its own source would file it under `g/42`, where `apply_deltas` never sees
    the global fact's id and drops the delete. The consolidation prompt would then be told to
    record the corrected state in its own compartment instead, leaving the original surfacing
    in every server with a contradiction filed beside it.
    """
    write_fact(scope=USER_SCOPE, fact=_stored_fact(fact_id="a" * 16, text="使用者住在台中"))
    write_fact(scope=USER_SCOPE, fact=_stored_fact(fact_id="b" * 16, text="使用者在新竹上班"))
    forget = render_forget_requests(notes=("使用者已經不住台中了",), source="guild 42")
    buckets = partition_forget_requests(raw_text=forget, compartments=("global", "g/42", "g/99"))
    assert sorted(buckets) == ["g/42", "global"]
    # And it is not in the observation partition at all, so it can never share a call with one.
    assert partition_raw_entries(raw_text=forget, flavor="user") == {}
    # And the compartments it reaches may only delete, never write the sentence down.
    outcome = apply_deltas(
        scope=USER_SCOPE,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(
            make_delta(
                action="delete",
                fact_id="a" * 16,
                section="fact",
                summary="住台中",
                text="使用者住在台中",
            ),
            make_delta(section="fact", summary="使用者要求忘記住處", text="使用者已經不住台中了"),
            make_delta(action="update", fact_id="b" * 16, text="使用者在新竹上班，不住台中"),
        ),
        owner=MemoryOwner(owner_id=USER_ID, owner_name="Alice"),
        allow_mass_delete=False,
        deletes_only=True,
    )
    assert outcome.deleted == 1
    assert outcome.created == 0
    assert outcome.updated == 0
    assert outcome.dropped == 2
    facts = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [fact.text for fact in facts] == ["使用者在新竹上班"]


def test_a_delete_survives_a_section_this_flavor_does_not_allow() -> None:
    """A delete is resolved by its id, so the section it names cannot cost the deletion.

    The section vocabularies are per flavor, and `member_alias` is legal on a server scope
    and not on a user one. Gating a delete on that dropped it outright — on the path every
    `<forget-memory>` runs through, where the fact survives and the bot keeps repeating what
    it was asked to drop. The fact carries its own section already; the delta's is decoration.
    """
    write_fact(scope=USER_SCOPE, fact=_stored_fact(fact_id="a" * 16, text="使用者住在台中"))
    outcome = apply_deltas(
        scope=USER_SCOPE,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(make_delta(action="delete", fact_id="a" * 16, section="member_alias"),),
        owner=MemoryOwner(owner_id=USER_ID, owner_name="Alice"),
        allow_mass_delete=False,
        deletes_only=True,
    )
    assert outcome.deleted == 1
    assert outcome.dropped == 0
    assert read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT) == []


def test_a_delete_releases_only_the_keys_no_remaining_fact_carries() -> None:
    """A key another fact still cites is that fact's evidence too, so a forget must leave it.

    The keys are model-cited and two facts can share one; releasing a shared key would take the
    surviving fact's evidence away along with the deleted one's.
    """
    write_fact(
        scope=USER_SCOPE, fact=_stored_fact(fact_id="a" * 16, keys=("fact.city", "fact.move"))
    )
    write_fact(
        scope=USER_SCOPE,
        fact=_stored_fact(fact_id="b" * 16, summary="搬家計畫", keys=("fact.move",)),
    )
    outcome = apply_deltas(
        scope=USER_SCOPE,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(make_delta(action="delete", fact_id="a" * 16),),
        owner=MemoryOwner(owner_id=USER_ID, owner_name="Alice"),
        allow_mass_delete=False,
        deletes_only=True,
    )
    assert outcome.released_keys == ("fact.city",)


def test_a_forget_releases_only_older_evidence_its_compartment_holds() -> None:
    """Only what was said before the newest forget, where the forget could reach, goes.

    Something said after it is a restatement the user chose to make, the same key filed for
    another server is not this forget's to take, and the requests themselves stay for
    `/memory regenerate` to replay. An older request does not move the cutoff back: one pass
    cannot tell which of its requests deleted which fact.
    """
    city = _observation(summary="住在台中", normalized_key="fact.city", sharing="source_only")
    job = _observation(summary="在工廠上班", normalized_key="fact.job", sharing="source_only")
    text = "\n\n".join([
        _forget_entry("2026-08-31T00:00:00+00:00"),
        _entry("2026-09-01T00:00:00+00:00", city, job),
        _entry("2026-09-01T00:00:01+00:00", city, source="guild 99"),
        _forget_entry("2026-09-02T00:00:00+00:00"),
        _entry("2026-09-03T00:00:00+00:00", city),
    ])
    forgets = partition_forget_requests(raw_text=text, compartments=("g/42", "g/99"))
    dropped = drop_released_evidence(text=text, released={"g/42": ("fact.city",)}, forgets=forgets)
    assert dropped == "\n\n".join([
        _forget_entry("2026-08-31T00:00:00+00:00"),
        _entry("2026-09-01T00:00:00+00:00", job),
        _entry("2026-09-01T00:00:01+00:00", city, source="guild 99"),
        _forget_entry("2026-09-02T00:00:00+00:00"),
        _entry("2026-09-03T00:00:00+00:00", city),
    ])
    # Nothing released, nothing rewritten: the very same text comes back.
    assert drop_released_evidence(text=text, released={"g/42": ()}, forgets=forgets) == text


def test_a_forget_keeps_a_legacy_identity_header_and_the_stamps_under_it() -> None:
    """An entry header with a ` | <identity>` suffix, as old detail files still hold, is a header.

    The store splits entries there. Read as a body line of the block before it, the header would
    leave with that block when a forget released it, and the blocks after it would take its stamp.
    """
    city = _observation(summary="住在台中", normalized_key="fact.city", sharing="source_only")
    job = _observation(summary="在工廠上班", normalized_key="fact.job", sharing="source_only")
    legacy = _entry("2026-06-05T02:23:02+00:00 | Alice (alice) [id: 1]", job)
    forget = _forget_entry("2026-09-02T00:00:00+00:00")
    text = "\n\n".join([_entry("2026-06-05T01:00:00+00:00", city), legacy, forget])
    forgets = partition_forget_requests(raw_text=text, compartments=("g/42",))
    dropped = drop_released_evidence(text=text, released={"g/42": ("fact.city",)}, forgets=forgets)
    assert dropped == f"{legacy}\n\n{forget}"


def test_raw_entries_partition_by_sharing_and_source() -> None:
    """Routing is deterministic code, off the fields code itself stamped."""
    buckets = partition_raw_entries(raw_text=_RAW_BATCH, flavor="user")
    assert set(buckets) == {"global", "g/222", "dm"}
    assert "全域偏好" in buckets["global"]
    assert "本群祕密" in buckets["g/222"]
    assert "私訊祕密" in buckets["dm"]
    assert "本群祕密" not in buckets["global"]
    # The bucket keeps the entry's own stamp and the observation whole.
    assert iter_observations(text=buckets["g/222"]) == [
        (
            "2026-07-01T00:00:00+00:00",
            "### stable_fact\n"
            "- normalized_key: fact.b\n"
            "- source: guild 222\n"
            "- sharing: source_only\n"
            "- summary_zh: 本群祕密",
        )
    ]


def test_server_evidence_all_lands_in_the_single_compartment() -> None:
    """Server observations carry no source, and a server scope has one compartment."""
    buckets = partition_raw_entries(raw_text=_RAW_BATCH, flavor="server")
    assert set(buckets) == {"global"}


def test_source_only_with_an_unusable_source_goes_to_the_owners_dm() -> None:
    """`source_only` can never fall back to global; the owner's own DM is the safe floor."""
    text = (
        "## 2026-07-01T00:00:00+00:00\n"
        "### stable_fact\n- normalized_key: fact.x\n- sharing: source_only\n- summary_zh: 祕密\n"
    )
    assert set(partition_raw_entries(raw_text=text, flavor="user")) == {"dm"}


def test_a_create_delta_writes_a_fact() -> None:
    """The ordinary path: one delta, one file, code-stamped."""
    scope = user_scope(user_id=111)
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(make_delta(from_keys=("preference.a",)),),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert outcome.applied
    assert outcome.created == 1
    stored = read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)
    assert stored[0].owner_id == 111
    assert stored[0].keys == ("preference.a",)
    assert stored[0].compartment == GLOBAL_COMPARTMENT


def test_a_create_whose_keys_already_back_a_fact_updates_it() -> None:
    """Retrying a partly-applied batch must upsert, not file a duplicate."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact(keys=("preference.a",)))
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(make_delta(summary="換個說法", text="新版本", from_keys=("preference.a",)),),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    stored = read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)
    assert outcome.created == 0
    assert len(stored) == 1
    assert stored[0].text == "新版本"


def test_an_update_keeps_the_original_creation_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """`created` is the fact's own history; only `last_confirmed` moves."""
    updated_at = STAMPED_AT + timedelta(days=1)
    monkeypatch.setattr("discordbot.services.memory.deltas.utc_now", lambda: updated_at)
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact())
    apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(make_delta(action="update", fact_id="0123456789abcdef", text="改寫"),),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    stored = read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)[0]
    assert stored.created == STAMPED_AT
    assert stored.last_confirmed == updated_at


def test_an_update_naming_a_missing_id_becomes_a_create() -> None:
    """A retry against a changed tree still lands the fact instead of dropping it."""
    scope = user_scope(user_id=111)
    apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(make_delta(action="update", fact_id="f" * 16),),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 1


def test_bad_deltas_are_dropped_without_failing_the_batch() -> None:
    """A deterministic content check that rejects the batch would freeze the scope."""
    scope = user_scope(user_id=111)
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(
            # A server-only section on a user scope.
            make_delta(section="culture"),
            make_delta(text="   "),
            make_delta(fact_id="", text="好的事實"),
        ),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert outcome.applied
    assert outcome.dropped == 2
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 1


def test_an_unusable_subject_id_costs_the_field_and_not_the_run() -> None:
    """An id cast unguarded here would raise, and raising abandons the rest of the fan-out.

    A user scope is where that bites: `member_alias` is a server-only section, so the guard
    that reads this field never fires on one and every delta reached the cast.
    """
    scope = user_scope(user_id=111)
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(
            make_delta(section="fact", summary="猜出來的 id", subject_id="阿華"),
            # `isdigit` accepts a superscript and `int()` refuses it, which is the same raise.
            make_delta(summary="上標數字", subject_id="²"),
            # And `isdecimal` accepts a digit string longer than CPython will convert.
            make_delta(summary="過長的 id", subject_id="9" * 4400),
        ),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert outcome.applied
    # Nothing outside `member_alias` renders the id, so the fact itself is still worth keeping.
    assert outcome.dropped == 0
    stored = read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)
    assert sorted(fact.summary for fact in stored) == ["上標數字", "猜出來的 id", "過長的 id"]
    assert {fact.subject_id for fact in stored} == {None}


def test_a_median_merge_is_not_mistaken_for_a_wipe() -> None:
    """Four deletes plus one create is consolidation working, not losing data."""
    scope = user_scope(user_id=111)
    ids = [f"{index:016x}" for index in range(5)]
    for fact_id in ids:
        write_fact(scope=scope, fact=_fact(fact_id=fact_id))
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(
            make_delta(summary="合併後的事實"),
            *(make_delta(action="delete", fact_id=fact_id) for fact_id in ids[:4]),
        ),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert outcome.applied
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 2


def test_a_wipe_is_refused_and_changes_nothing() -> None:
    """Outside a forget or a rebuild, a batch that only deletes is a lossy rewrite, so it never
    touches the disk.
    """
    scope = user_scope(user_id=111)
    ids = [f"{index:016x}" for index in range(10)]
    for fact_id in ids:
        write_fact(scope=scope, fact=_fact(fact_id=fact_id))
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=tuple(make_delta(action="delete", fact_id=fact_id) for fact_id in ids),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert not outcome.applied
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 10


def test_a_rebuild_may_replace_the_whole_set() -> None:
    """The regeneration path is exempt: replacing everything is what it is for."""
    scope = user_scope(user_id=111)
    ids = [f"{index:016x}" for index in range(10)]
    for fact_id in ids:
        write_fact(scope=scope, fact=_fact(fact_id=fact_id))
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=tuple(make_delta(action="delete", fact_id=fact_id) for fact_id in ids),
        owner=_OWNER,
        allow_mass_delete=True,
    )
    assert outcome.applied
    assert read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT) == []


def test_the_sweep_expires_recent_context_but_never_permanent() -> None:
    """Aging is deterministic code off the stamped dates, never the model's writing."""
    scope = user_scope(user_id=111)
    today = STAMPED_AT + timedelta(days=60)
    write_fact(
        scope=scope,
        fact=_fact(
            fact_id="a" * 16, section="recent", durability="recent", last_confirmed=STAMPED_AT
        ),
    )
    write_fact(
        scope=scope,
        fact=_fact(
            fact_id="b" * 16,
            section="permanent",
            durability="permanent",
            last_confirmed=STAMPED_AT,
        ),
    )
    assert sweep_stale_facts(scope=scope, compartment=GLOBAL_COMPARTMENT, today=today) == 1
    assert [fact.fact_id for fact in read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)] == [
        "b" * 16
    ]


def test_stable_facts_age_by_displacement_within_their_own_compartment() -> None:
    """A busy compartment self-trims; a quiet one anchored elsewhere forgets nothing."""
    scope = user_scope(user_id=111)
    quiet = guild_compartment(guild_id=222)
    write_fact(scope=scope, fact=_fact(fact_id="a" * 16, last_confirmed=STAMPED_AT))
    write_fact(
        scope=scope, fact=_fact(fact_id="b" * 16, last_confirmed=STAMPED_AT + timedelta(days=90))
    )
    write_fact(
        scope=scope, fact=_fact(fact_id="c" * 16, compartment=quiet, last_confirmed=STAMPED_AT)
    )
    today = STAMPED_AT + timedelta(days=365)
    assert sweep_stale_facts(scope=scope, compartment=GLOBAL_COMPARTMENT, today=today) == 1
    # The other compartment saw no newer activity, so nothing displaced its fact.
    assert sweep_stale_facts(scope=scope, compartment=quiet, today=today) == 0
    assert len(read_facts(scope=scope, compartment=quiet)) == 1


def test_existing_facts_render_with_the_ids_the_model_must_echo() -> None:
    """`fact_id` is the model's only handle on a stored fact, so it has to be shown."""
    rendered = render_existing_facts(facts=[_fact(keys=("preference.a",))])
    assert "[0123456789abcdef] section=preference durability=stable" in rendered
    assert "from_keys: preference.a" in rendered


def test_a_batch_reports_the_ids_it_wrote() -> None:
    """A rebuild drops what it did not re-emit, so it needs the written set, not the survivors."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact(fact_id="a" * 16))
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(make_delta(summary="新的事實"),),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert outcome.written == (mint_fact_id(compartment=GLOBAL_COMPARTMENT, summary="新的事實"),)
    # The untouched fact is still on disk; it is the CALLER's job to drop it on a rebuild.
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 2


def test_a_permanent_section_fact_never_ages_even_when_marked_stable() -> None:
    """Nothing couples `section` to `durability`, and `render_existing_facts` feeds a
    mismatched pairing back on every later update, so one slip by the model would
    otherwise displace an enforced standing directive out of memory for good.
    """
    scope = user_scope(user_id=111)
    write_fact(
        scope=scope,
        fact=_fact(
            fact_id="a" * 16, section="permanent", durability="stable", last_confirmed=STAMPED_AT
        ),
    )
    write_fact(
        scope=scope, fact=_fact(fact_id="b" * 16, last_confirmed=STAMPED_AT + timedelta(days=120))
    )
    assert (
        sweep_stale_facts(
            scope=scope, compartment=GLOBAL_COMPARTMENT, today=STAMPED_AT + timedelta(days=365)
        )
        == 0
    )
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 2


@pytest.mark.parametrize(
    ("section", "durability"),
    [("fact", "recent"), ("fact", "session"), ("preference", "volatile"), ("recent", "stable")],
)
def test_a_fact_not_displaced_as_stable_ages_on_the_ttl(
    section: MemorySection, durability: MemoryDurability
) -> None:
    """The delta schema lets the model pair any section with any durability, so a fact
    that is neither exempt nor a `stable` one outside `recent` must still age instead of
    being injected until a later pass happens to delete it. The fresh anchor puts the old
    fact past the TTL but inside the stable window, so displacement alone would keep it.
    """
    scope = user_scope(user_id=111)
    today = STAMPED_AT + timedelta(days=RECENT_CONTEXT_TTL_DAYS + 1)
    write_fact(
        scope=scope,
        fact=_fact(
            fact_id="a" * 16, section=section, durability=durability, last_confirmed=STAMPED_AT
        ),
    )
    write_fact(scope=scope, fact=_fact(fact_id="b" * 16, last_confirmed=today))
    assert sweep_stale_facts(scope=scope, compartment=GLOBAL_COMPARTMENT, today=today) == 1
    assert [fact.fact_id for fact in read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)] == [
        "b" * 16
    ]


def _alias_delta(  # noqa: PLR0913 -- test helper mirrors the alias half of the delta schema
    summary: str = "李董的社群暱稱",
    display_name: str = "小李",
    aliases: tuple[str, ...] = ("李董",),
    text: str = "",
    subject_id: str = "4242",
    action: MemoryDeltaAction = "create",
) -> MemoryFactDelta:
    """Builds one member-alias consolidation delta.

    `text` defaults to empty because the row is rendered from the two structured fields;
    a test passing one is checking that the model's prose is discarded.
    """
    return make_delta(
        action=action,
        section="member_alias",
        durability="permanent",
        summary=summary,
        text=text,
        subject_id=subject_id,
        display_name=display_name,
        aliases=aliases,
    )


def test_a_member_alias_delta_without_a_member_id_is_dropped() -> None:
    """An alias row with no id is unaskable, and a guessed one would widen the wrong memory."""
    outcome = apply_deltas(
        scope=SERVER_SCOPE,
        compartment=GLOBAL_COMPARTMENT,
        flavor="server",
        deltas=(
            _alias_delta(),
            _alias_delta(
                summary="沒有 id 的暱稱", display_name="阿明", aliases=("明哥",), subject_id=""
            ),
            _alias_delta(
                summary="猜出來的 id", display_name="阿華", aliases=("華哥",), subject_id="阿華"
            ),
            # Two ids `int()` refuses; the gate reads them the way the cast does.
            _alias_delta(
                summary="上標的 id", display_name="阿光", aliases=("光哥",), subject_id="²"
            ),
            _alias_delta(
                summary="過長的 id", display_name="阿強", aliases=("強哥",), subject_id="9" * 4400
            ),
        ),
        owner=SERVER_OWNER,
        allow_mass_delete=False,
    )
    # The batch still lands: a whole-batch rejection would freeze the scope for good.
    assert outcome.applied
    assert outcome.created == 1
    assert outcome.dropped == 4
    stored = read_facts(scope=SERVER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [fact.subject_id for fact in stored] == [4242]
    assert allowlist_ids_from_server_memory(memory=_server_document()) == {
        4242: "小李(社群暱稱:李董)"
    }


def test_an_alias_row_is_built_from_its_fields_not_the_models_prose() -> None:
    """Code renders the compact shape, since a model asked for it rarely writes it.

    Whatever the model puts in `text` — a full sentence, an id token, a personal aside that
    has no business in a nickname table — never reaches the stored row.
    """
    apply_deltas(
        scope=SERVER_SCOPE,
        compartment=GLOBAL_COMPARTMENT,
        flavor="server",
        deltas=(
            _alias_delta(
                display_name="阿明",
                aliases=("明哥", "明神"),
                text="社群習慣稱呼成員 阿明 [id: 999] 為「明哥」，他常被拿來開玩笑說投資虧錢。",
                subject_id="9001",
            ),
        ),
        owner=SERVER_OWNER,
        allow_mass_delete=False,
    )
    stored = read_facts(scope=SERVER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [fact.text for fact in stored] == ["阿明(社群暱稱:明哥、明神)"]
    assert allowlist_ids_from_server_memory(memory=_server_document()) == {
        9001: "阿明(社群暱稱:明哥、明神)"
    }


def test_an_alias_row_survives_a_missing_name_but_not_a_missing_alias() -> None:
    """The mapping the table exists for is the alias and the id; the name is a label.

    A member the evidence only ever identifies by id is an ordinary outcome, so the row is
    written without one rather than lost. A row with no alias carries nothing at all and is
    dropped like any other empty body.
    """
    outcome = apply_deltas(
        scope=SERVER_SCOPE,
        compartment=GLOBAL_COMPARTMENT,
        flavor="server",
        deltas=(
            _alias_delta(summary="只有 id 的暱稱", display_name="", subject_id="4242"),
            # The shape a model that ignores the new fields produces: prose in `text`,
            # nothing in `aliases`. That row is dropped, not written back as the sentence.
            _alias_delta(
                summary="沒有別稱",
                display_name="阿華",
                aliases=(),
                text="社群習慣稱呼成員 阿華 為「華哥」。",
                subject_id="9001",
            ),
            _alias_delta(summary="只有空白別稱", aliases=("", "  "), subject_id="9002"),
        ),
        owner=SERVER_OWNER,
        allow_mass_delete=False,
    )
    assert outcome.created == 1
    assert outcome.dropped == 2
    assert allowlist_ids_from_server_memory(memory=_server_document()) == {4242: "(社群暱稱:李董)"}


def test_an_alias_row_stays_one_clean_line() -> None:
    """The rendered table is read back line by line, and a repeat is what unioning costs."""
    apply_deltas(
        scope=SERVER_SCOPE,
        compartment=GLOBAL_COMPARTMENT,
        flavor="server",
        deltas=(
            _alias_delta(
                display_name="阿明\n（新名字）",
                aliases=("明哥", "明哥", "明  神"),
                subject_id="9001",
            ),
        ),
        owner=SERVER_OWNER,
        allow_mass_delete=False,
    )
    stored = read_facts(scope=SERVER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [fact.text for fact in stored] == ["阿明 （新名字）(社群暱稱:明哥、明 神)"]


def test_member_alias_rows_never_age_out() -> None:
    """A swept alias row silently shrinks the allowlist, so the sweep skips every one of them."""
    stale = STAMPED_AT - timedelta(days=STABLE_FRESHNESS_WINDOW_DAYS + 30)
    write_fact(
        scope=SERVER_SCOPE,
        fact=_alias_fact(
            fact_id="a" * 16, text="小李(社群暱稱:李董)", subject_id=4242, last_confirmed=stale
        ),
    )
    # Even filed as merely `stable` (the durability the prompt forbids for an alias), the
    # row is exempt on its node type alone.
    write_fact(
        scope=SERVER_SCOPE,
        fact=_alias_fact(
            fact_id="b" * 16,
            text="阿明(社群暱稱:明哥)",
            subject_id=9001,
            durability="stable",
            last_confirmed=stale,
        ),
    )
    write_fact(scope=SERVER_SCOPE, fact=_server_fact(fact_id="c" * 16, last_confirmed=stale))
    write_fact(scope=SERVER_SCOPE, fact=_server_fact(fact_id="d" * 16, text="社群近來只聊楓之谷"))
    assert (
        sweep_stale_facts(
            scope=SERVER_SCOPE,
            compartment=GLOBAL_COMPARTMENT,
            today=STAMPED_AT + timedelta(days=1),
        )
        == 1
    )
    remaining = {
        fact.fact_id for fact in read_facts(scope=SERVER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    }
    assert remaining == {"a" * 16, "b" * 16, "d" * 16}
