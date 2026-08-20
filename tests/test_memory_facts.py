"""Tests for the compartment store, the fact file format, and delta application.

These pin the mechanisms the directory boundary rests on: which directories a reading
context may open, that a fact file round-trips, that a delta batch cannot widen a fact's
reach or wipe a scope, and that aging is deterministic now that the dates are code-stamped.
"""

from typing import cast
from pathlib import Path
from datetime import UTC, datetime, timedelta

import pytest

from discordbot.typings.memory import (
    MemoryFact,
    MemoryOwner,
    MemorySection,
    MemoryDurability,
    MemoryDeltaAction,
)
from discordbot.services.memory.facts import (
    FACT_ID_RE,
    mint_fact_id,
    node_type_for,
    parse_identity,
    parse_fact_file,
    render_fact_file,
    sections_for_flavor,
    render_owner_identity,
)
from discordbot.services.memory.store import (
    DM_COMPARTMENT,
    GLOBAL_COMPARTMENT,
    read_facts,
    read_owner,
    user_scope,
    write_fact,
    delete_fact,
    iter_scopes,
    clear_memory,
    server_scope,
    scope_owner_id,
    compartment_dir,
    guild_compartment,
    list_compartments,
    prune_compartment,
    unaccounted_files,
    read_memory_document,
)
from discordbot.services.memory.deltas import (
    apply_deltas,
    sweep_stale_facts,
    partition_raw_entries,
    render_existing_facts,
    tone_evidence_from_raw,
)
from discordbot.cogs.gen_reply.memory_tool import (
    MemoryReadContext,
    compartments_for_reading,
    allowlist_ids_from_server_memory,
)
from discordbot.services.memory.extraction import MemoryFactDelta

_OWNER = MemoryOwner(owner_id=111, owner_name="Alice (alice)")
_NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


def _fact(  # noqa: PLR0913 -- test helper mirrors the stored fact's own fields
    *,
    fact_id: str = "0123456789abcdef",
    compartment: str = GLOBAL_COMPARTMENT,
    section: str = "preference",
    durability: str = "stable",
    text: str = "喜歡簡短回覆",
    last_confirmed: datetime = _NOW,
    subject_id: int | None = None,
    keys: tuple[str, ...] = (),
) -> MemoryFact:
    """Builds a stored fact with the boilerplate filled in."""
    return MemoryFact(
        fact_id=fact_id,
        summary="回覆長度偏好",
        section=cast("MemorySection", section),
        durability=cast("MemoryDurability", durability),
        text=text,
        compartment=compartment,
        owner_id=_OWNER.owner_id,
        owner_name=_OWNER.owner_name,
        subject_id=subject_id,
        node_type=node_type_for(section=cast("MemorySection", section)),
        created=_NOW,
        last_confirmed=last_confirmed,
        keys=keys,
    )


def _delta(  # noqa: PLR0913 -- test helper mirrors the delta schema
    *,
    action: str = "create",
    fact_id: str = "",
    section: str = "preference",
    durability: str = "stable",
    summary: str = "回覆長度偏好",
    text: str = "喜歡簡短回覆",
    from_keys: tuple[str, ...] = (),
    subject_id: str = "",
) -> MemoryFactDelta:
    """Builds a consolidation delta with the boilerplate filled in."""
    return MemoryFactDelta(
        action=cast("MemoryDeltaAction", action),
        fact_id=fact_id,
        section=cast("MemorySection", section),
        durability=cast("MemoryDurability", durability),
        summary=summary,
        text=text,
        from_keys=from_keys,
        subject_id=subject_id,
    )


def test_fact_file_round_trips(memory_isolated_dir: Path) -> None:
    """A written fact parses back identical, including its code-stamped fields."""
    fact = _fact(keys=("preference.reply_length",))
    parsed = parse_fact_file(text=render_fact_file(fact=fact), compartment=GLOBAL_COMPARTMENT)
    assert parsed == fact


def test_fact_file_from_the_wrong_directory_is_refused() -> None:
    """A stored compartment that disagrees with the directory is corruption, not a hint."""
    fact = _fact(compartment=guild_compartment(guild_id=222))
    assert (
        parse_fact_file(text=render_fact_file(fact=fact), compartment=GLOBAL_COMPARTMENT) is None
    )


def test_fact_file_survives_a_multiline_body_and_colons() -> None:
    """The hand-rolled header stops at its fence; the body may contain anything."""
    fact = _fact(text="第一行: 有冒號\n第二行\n---\n第三行")
    parsed = parse_fact_file(text=render_fact_file(fact=fact), compartment=GLOBAL_COMPARTMENT)
    assert parsed is not None
    assert parsed.text == "第一行: 有冒號\n第二行\n---\n第三行"


@pytest.mark.parametrize("text", ["", "no fence here", "---\nid: x\nno closing fence"])
def test_malformed_fact_files_parse_to_none(text: str) -> None:
    """An unreadable file is skipped rather than half-parsed into a fact."""
    assert parse_fact_file(text=text, compartment=GLOBAL_COMPARTMENT) is None


def test_minted_ids_are_stable_per_compartment_and_summary() -> None:
    """The id is code-owned, deterministic, and never shared across compartments."""
    first = mint_fact_id(compartment=GLOBAL_COMPARTMENT, summary="回覆長度偏好")
    again = mint_fact_id(compartment=GLOBAL_COMPARTMENT, summary="  回覆長度偏好\n")
    elsewhere = mint_fact_id(compartment=DM_COMPARTMENT, summary="回覆長度偏好")
    assert first == again
    assert first != elsewhere
    assert FACT_ID_RE.match(first)


def test_identity_line_round_trips() -> None:
    """The stored identity survives the pipeline's string round-trip."""
    owner = parse_identity(identity="Alice (alice) [id: 111]", fallback_owner_id=0)
    assert owner == MemoryOwner(owner_id=111, owner_name="Alice (alice)")
    assert render_owner_identity(owner=owner) == "Alice (alice) [id: 111]"


def test_unparseable_identity_falls_back_to_the_scope_id() -> None:
    """A job persisted before the format existed still stamps the right owner id."""
    owner = parse_identity(identity="", fallback_owner_id=111)
    assert owner.owner_id == 111


def test_compartment_dir_refuses_anything_but_the_three_shapes() -> None:
    """The single join between a compartment string and a path never takes a stray value."""
    for bad in ("../escape", "g/../..", "global/x", "", "G/1"):
        with pytest.raises(ValueError, match="invalid memory compartment"):
            compartment_dir(scope="1", compartment=bad)


def test_write_read_and_delete_one_fact(memory_isolated_dir: Path) -> None:
    """A fact lands in its compartment directory and comes back out of it."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact())
    assert [fact.fact_id for fact in read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)] == [
        "0123456789abcdef"
    ]
    assert delete_fact(scope=scope, compartment=GLOBAL_COMPARTMENT, fact_id="0123456789abcdef")
    assert read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT) == []
    assert not delete_fact(scope=scope, compartment=GLOBAL_COMPARTMENT, fact_id="0123456789abcdef")


def test_list_compartments_is_ordered_global_guilds_dm(memory_isolated_dir: Path) -> None:
    """Compartment order is deterministic so the injected prefix does not reshuffle."""
    scope = user_scope(user_id=111)
    for compartment in (
        DM_COMPARTMENT,
        guild_compartment(guild_id=900),
        guild_compartment(guild_id=100),
        GLOBAL_COMPARTMENT,
    ):
        write_fact(scope=scope, fact=_fact(compartment=compartment))
    assert list_compartments(scope=scope) == ["global", "g/100", "g/900", "dm"]


def test_reading_a_guild_never_opens_another_guild(memory_isolated_dir: Path) -> None:
    """The whole privacy boundary: a guild read joins global with that guild alone."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact(fact_id="a" * 16, text="全域事實"))
    write_fact(
        scope=scope,
        fact=_fact(fact_id="b" * 16, compartment=guild_compartment(guild_id=111), text="本群事實"),
    )
    write_fact(
        scope=scope,
        fact=_fact(fact_id="c" * 16, compartment=guild_compartment(guild_id=222), text="他群祕密"),
    )
    write_fact(
        scope=scope, fact=_fact(fact_id="d" * 16, compartment=DM_COMPARTMENT, text="私訊祕密")
    )
    document = read_memory_document(
        scope=scope,
        compartments=compartments_for_reading(
            owner_id=111, context=MemoryReadContext(guild_id=111, dm_partner_id=None)
        ),
        flavor="user",
    )
    assert "全域事實" in document
    assert "本群事實" in document
    assert "他群祕密" not in document
    assert "私訊祕密" not in document


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (MemoryReadContext(guild_id=111, dm_partner_id=None), ["global", "g/111"]),
        # A guild the owner has no memory in still reads only global; the join simply
        # names a directory that does not exist.
        (MemoryReadContext(guild_id=999, dm_partner_id=None), ["global", "g/999"]),
        # A group DM, and a third party's memory read inside a 1:1 DM, both get the
        # cross-server compartment only.
        (MemoryReadContext(guild_id=None, dm_partner_id=None), ["global"]),
        (MemoryReadContext(guild_id=None, dm_partner_id=222), ["global"]),
    ],
)
def test_compartments_for_reading_matrix(
    memory_isolated_dir: Path, context: MemoryReadContext, expected: list[str]
) -> None:
    """Each reading context resolves to exactly the directories it may open."""
    assert compartments_for_reading(owner_id=111, context=context) == expected


def test_the_owner_reads_everything_in_their_own_dm(memory_isolated_dir: Path) -> None:
    """A user's own information cannot leak to themselves, so their DM opens it all."""
    scope = user_scope(user_id=111)
    for compartment in (GLOBAL_COMPARTMENT, guild_compartment(guild_id=222), DM_COMPARTMENT):
        write_fact(scope=scope, fact=_fact(compartment=compartment))
    compartments = compartments_for_reading(
        owner_id=111, context=MemoryReadContext(guild_id=None, dm_partner_id=111)
    )
    assert set(compartments) == {"global", "g/222", "dm"}


def test_rendered_document_groups_by_section_and_dates_recent(memory_isolated_dir: Path) -> None:
    """The rendered shape is the one the reply prompts have always been handed."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact(fact_id="a" * 16, section="profile", text="愛玩遊戲的人"))
    write_fact(scope=scope, fact=_fact(fact_id="b" * 16, section="preference", text="喜歡簡短"))
    write_fact(
        scope=scope,
        fact=_fact(fact_id="c" * 16, section="recent", durability="recent", text="正在搬家"),
    )
    document = read_memory_document(scope=scope, compartments=[GLOBAL_COMPARTMENT], flavor="user")
    assert document.index("## 使用者輪廓") < document.index("## 穩定偏好")
    assert document.index("## 穩定偏好") < document.index("## 近期脈絡")
    assert "\n愛玩遊戲的人" in document
    assert "* 喜歡簡短" in document
    assert "* [2026-07-01] 正在搬家" in document


def test_member_alias_rows_carry_their_id_for_the_allowlist(memory_isolated_dir: Path) -> None:
    """The nickname table stays parseable, so allowlist widening keeps working."""
    scope = server_scope(server_id=500)
    write_fact(
        scope=scope,
        fact=_fact(
            fact_id="a" * 16,
            section="member_alias",
            durability="permanent",
            text="小明(社群暱稱:明哥)",
            subject_id=777,
        ),
    )
    document = read_memory_document(
        scope=scope, compartments=[GLOBAL_COMPARTMENT], flavor="server"
    )
    assert "## 成員稱呼" in document
    assert allowlist_ids_from_server_memory(memory=document) == {777: "小明(社群暱稱:明哥)"}


def test_the_document_is_capped_and_says_so(memory_isolated_dir: Path) -> None:
    """Past the cap the render stops and admits it, rather than silently overflowing."""
    scope = user_scope(user_id=111)
    for index in range(20):
        write_fact(scope=scope, fact=_fact(fact_id=f"{index:016x}", text="長" * 200))
    document = read_memory_document(
        scope=scope, compartments=[GLOBAL_COMPARTMENT], flavor="user", max_chars=600
    )
    assert len(document) < 1_000
    assert "記憶已達可注入上限" in document


def test_reads_are_cached_until_a_write_lands(memory_isolated_dir: Path) -> None:
    """A repeat read of an unchanged scope must not re-open every fact file."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact(text="第一版"))
    first = read_memory_document(scope=scope, compartments=[GLOBAL_COMPARTMENT], flavor="user")
    # Bypassing the store leaves the generation counter untouched, so a cached read must
    # still return the old document — which is exactly what proves the cache is live.
    path = compartment_dir(scope=scope, compartment=GLOBAL_COMPARTMENT) / "0123456789abcdef.md"
    path.write_text(path.read_text().replace("第一版", "第二版"), encoding="utf-8")
    assert (
        read_memory_document(scope=scope, compartments=[GLOBAL_COMPARTMENT], flavor="user")
        == first
    )
    write_fact(scope=scope, fact=_fact(fact_id="b" * 16, text="第三版"))
    assert "第二版" in read_memory_document(
        scope=scope, compartments=[GLOBAL_COMPARTMENT], flavor="user"
    )


def test_iter_scopes_finds_compartment_trees_and_skips_dot_dirs(memory_isolated_dir: Path) -> None:
    """The sweep sees a scope that has only fact files, and never the git directory."""
    write_fact(scope=user_scope(user_id=111), fact=_fact())
    write_fact(scope=server_scope(server_id=500), fact=_fact())
    (memory_isolated_dir / ".git").mkdir(parents=True, exist_ok=True)
    (memory_isolated_dir / ".git" / "raw.md").write_text("## 2026-01-01T00:00:00+00:00\n")
    assert iter_scopes() == ["111", "bot_memories/500"]


def test_iter_scopes_ignores_a_scope_whose_only_file_is_unreadable(
    memory_isolated_dir: Path,
) -> None:
    """A file no reader can parse must not answer for a whole scope on its own.

    Listing it instead of parsing it kept such a scope on `iter_scopes` permanently, so
    the restart sweep and the offline rebuild picked it up on every run with nothing
    either could do about it.
    """
    scope = user_scope(user_id=111)
    directory = compartment_dir(scope=scope, compartment=GLOBAL_COMPARTMENT)
    directory.mkdir(parents=True)
    (directory / f"{'b' * 16}.md").write_text("hand-edited into nonsense\n", encoding="utf-8")
    assert iter_scopes() == []


def test_prune_compartment_removes_every_fact_file_it_cannot_read(
    memory_isolated_dir: Path,
) -> None:
    """The rebuild's prune reads the directory, not the facts readable inside it.

    Both shapes `read_facts` skips are here: a header that does not parse, and a fact
    whose stored compartment disagrees with the directory holding it (the deliberate
    skip). Neither reaches a snapshot taken over `read_facts`, which is what let them
    outlive a rebuild that reports the compartment replaced.
    """
    scope = user_scope(user_id=111)
    kept = _fact()
    write_fact(scope=scope, fact=kept)
    directory = compartment_dir(scope=scope, compartment=GLOBAL_COMPARTMENT)
    broken = directory / f"{'b' * 16}.md"
    broken.write_text("hand-edited into nonsense\n", encoding="utf-8")
    misfiled = directory / f"{'c' * 16}.md"
    misfiled.write_text(
        render_fact_file(
            fact=_fact(fact_id="c" * 16, compartment=guild_compartment(guild_id=222))
        ),
        encoding="utf-8",
    )
    # The leftover of a crash between `write_fact`'s tmp write and its `os.replace`, in a
    # compartment that keeps its facts, so no `rmdir` can stand in for removing it.
    stranded = directory / f"{'d' * 16}.md.tmp"
    stranded.write_text("半途寫壞的檔案", encoding="utf-8")

    pruned = prune_compartment(scope=scope, compartment=GLOBAL_COMPARTMENT, keep={kept.fact_id})
    assert pruned.unaccounted == []
    # Both of them were content no reader could reach, so both are reported as destroyed;
    # the stranded `.md.tmp` is not, since `_fact_paths` never globbed it to begin with.
    assert pruned.unreadable == [broken.name, misfiled.name]
    assert not broken.exists()
    assert not misfiled.exists()
    assert not stranded.exists()
    assert [fact.fact_id for fact in read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)] == [
        kept.fact_id
    ]


def test_prune_compartment_reports_a_file_the_store_never_wrote(memory_isolated_dir: Path) -> None:
    """Removing a foreign file is too aggressive to do silently, so it is named instead.

    A store file is recognised by its name, not by its suffix, so an operator's own
    markdown note beside the facts is reported like anything else rather than swept up
    with the fact files.
    """
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact())
    directory = compartment_dir(scope=scope, compartment=GLOBAL_COMPARTMENT)
    for name in ("backup.txt", "notes.md"):
        (directory / name).write_text("操作者的備份", encoding="utf-8")

    pruned = prune_compartment(scope=scope, compartment=GLOBAL_COMPARTMENT, keep=set())
    assert pruned.unaccounted == ["backup.txt", "notes.md"]
    # The one fact file it removed parsed fine; a rebuild drops those by not re-emitting
    # them, which is the ordinary outcome and not a loss anyone has to be told about.
    assert pruned.unreadable == []
    assert (directory / "notes.md").exists()
    assert read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT) == []


def test_a_file_the_store_cannot_decode_never_stops_the_sweep(memory_isolated_dir: Path) -> None:
    """`iter_scopes` parses the tier now, so an undecodable file must degrade, not raise.

    A hand edit saved in the wrong encoding would otherwise abort the restart sweep and
    stop the offline rebuild starting at all — the tool for repairing exactly that store.
    """
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact())
    mojibake = compartment_dir(scope=scope, compartment=GLOBAL_COMPARTMENT) / f"{'b' * 16}.md"
    mojibake.write_bytes("喜歡簡短回覆".encode("big5"))

    assert iter_scopes() == ["111"]
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 1
    # It is still the store's own file by name, so a rebuild's prune takes it, and takes
    # it unread — `read_facts` skipped it above rather than parsing it.
    pruned = prune_compartment(scope=scope, compartment=GLOBAL_COMPARTMENT, keep=set())
    assert pruned.unreadable == [mojibake.name]
    assert not mojibake.exists()


def test_a_fact_shaped_name_carrying_a_newline_is_not_the_stores_own(
    memory_isolated_dir: Path,
) -> None:
    """`$` also matches before a trailing newline, so the name test has to be a fullmatch."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact())
    forged = compartment_dir(scope=scope, compartment=GLOBAL_COMPARTMENT) / f"{'b' * 16}\n.md"
    forged.write_text("不是我們寫的", encoding="utf-8")

    pruned = prune_compartment(scope=scope, compartment=GLOBAL_COMPARTMENT, keep=set())
    assert pruned.unaccounted == [f"{'b' * 16}\n.md"]
    assert forged.exists()


def test_a_directory_named_like_a_fact_file_never_reaches_a_reader(
    memory_isolated_dir: Path,
) -> None:
    """`iter_scopes` parses the tier now, so a reader that raises takes the sweep with it.

    `_read_text` catches only a missing file, so one hand-made directory whose name ends
    in `.md` would abort the restart sweep and every offline run.
    """
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact())
    (compartment_dir(scope=scope, compartment=GLOBAL_COMPARTMENT) / f"{'b' * 16}.md").mkdir()

    assert iter_scopes() == ["111"]
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 1
    assert unaccounted_files(scope=scope, compartment=GLOBAL_COMPARTMENT) == [f"{'b' * 16}.md"]


def test_the_guild_parent_goes_with_the_last_compartment_under_it(
    memory_isolated_dir: Path,
) -> None:
    """`delete_memory_files` removes `g/` for the same reason; a prune emptying it must too.

    `list_compartments` cannot stand in for this assertion: it reports nothing for an
    empty `g/` either way.
    """
    scope = user_scope(user_id=111)
    compartment = guild_compartment(guild_id=222)
    write_fact(scope=scope, fact=_fact(compartment=compartment))

    assert prune_compartment(scope=scope, compartment=compartment, keep=set()).unaccounted == []
    assert not (memory_isolated_dir / scope / "g").exists()


def test_a_pruned_fact_leaves_the_cached_document(memory_isolated_dir: Path) -> None:
    """The prune is a write like any other, and the render cache is keyed on that counter."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact())
    write_fact(scope=scope, fact=_fact(fact_id="a" * 16, text="另一件事"))
    assert "喜歡簡短回覆" in read_memory_document(
        scope=scope, compartments=[GLOBAL_COMPARTMENT], flavor="user"
    )

    assert (
        prune_compartment(scope=scope, compartment=GLOBAL_COMPARTMENT, keep={"a" * 16}).unaccounted
        == []
    )
    assert "喜歡簡短回覆" not in read_memory_document(
        scope=scope, compartments=[GLOBAL_COMPARTMENT], flavor="user"
    )


def test_prune_compartment_removes_a_directory_it_emptied(memory_isolated_dir: Path) -> None:
    """An emptied compartment stops being one, so it stops costing a call per rebuild.

    The stranded `.md.tmp` of a crashed `write_fact` goes too: it is the store's own, and
    leaving it behind would keep the directory alive forever.
    """
    scope = user_scope(user_id=111)
    compartment = guild_compartment(guild_id=222)
    write_fact(scope=scope, fact=_fact(compartment=compartment))
    directory = compartment_dir(scope=scope, compartment=compartment)
    (directory / f"{'d' * 16}.md.tmp").write_text("半途寫壞的檔案", encoding="utf-8")

    pruned = prune_compartment(scope=scope, compartment=compartment, keep=set())
    assert pruned.unaccounted == []
    assert pruned.unreadable == []
    assert not directory.exists()
    assert list_compartments(scope=scope) == []


def test_clear_removes_the_whole_compartment_tree(memory_isolated_dir: Path) -> None:
    """A clear has to walk the tree now, not three fixed filenames."""
    scope = user_scope(user_id=111)
    for compartment in (GLOBAL_COMPARTMENT, guild_compartment(guild_id=222), DM_COMPARTMENT):
        write_fact(scope=scope, fact=_fact(compartment=compartment))
    assert clear_memory(scope=scope)
    assert list_compartments(scope=scope) == []
    assert not (memory_isolated_dir / scope).exists()


def test_read_owner_recovers_the_stored_identity(memory_isolated_dir: Path) -> None:
    """Offline paths with no Discord context keep the last online write's label."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact())
    assert read_owner(scope=scope) == _OWNER
    assert read_owner(scope=user_scope(user_id=999)).owner_id == 999


def test_scope_owner_id_reads_both_flavors() -> None:
    """A server scope's owner id is the server id, not the bot's."""
    assert scope_owner_id(scope=user_scope(user_id=111)) == 111
    assert scope_owner_id(scope=server_scope(server_id=500)) == 500


_RAW_BATCH = """## 2026-07-01T00:00:00+00:00
### stable_preference
- normalized_key: preference.a
- source: guild 222
- sharing: global
- summary_zh: 全域偏好

### stable_fact
- normalized_key: fact.b
- source: guild 222
- sharing: source_only
- summary_zh: 本群祕密

### interaction_style
- normalized_key: style.c
- source: dm
- sharing: source_only
- summary_zh: 私訊祕密
"""


def test_raw_entries_partition_by_sharing_and_source() -> None:
    """Routing is deterministic code, off the fields code itself stamped."""
    buckets = partition_raw_entries(raw_text=_RAW_BATCH, flavor="user")
    assert set(buckets) == {"global", "g/222", "dm"}
    assert "全域偏好" in buckets["global"]
    assert "本群祕密" in buckets["g/222"]
    assert "私訊祕密" in buckets["dm"]
    assert "本群祕密" not in buckets["global"]
    assert buckets["g/222"].startswith("## 2026-07-01T00:00:00+00:00")


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


def test_tone_evidence_ignores_compartments() -> None:
    """Tone must see the whole batch, or it stops updating for half of all conversations."""
    evidence = tone_evidence_from_raw(raw_text=_RAW_BATCH)
    assert "私訊祕密" in evidence
    assert "全域偏好" in evidence


def test_tone_evidence_carries_the_evidence_kind() -> None:
    """A stated preference has to reach the note distinguishable from an inferred one.

    Carrying only the summary made both read alike, so the note converged on whichever
    reading had the most bullets: one stated "address me respectfully" lost to five
    inferred banter observations and the note came out saying the opposite.
    """
    text = (
        "## 2026-07-01T00:00:00+00:00\n"
        "### interaction_style\n"
        "- normalized_key: style.stated\n"
        "- evidence_kind: explicit_preference\n"
        "- sharing: source_only\n"
        "- summary_zh: 要求對本人使用尊敬語氣\n"
        "\n"
        "### interaction_style\n"
        "- normalized_key: style.inferred\n"
        "- evidence_kind: repeated_behavior\n"
        "- sharing: global\n"
        "- summary_zh: 會主動嗆機器人\n"
    )
    evidence = tone_evidence_from_raw(raw_text=text)
    assert "* [explicit_preference] 要求對本人使用尊敬語氣" in evidence
    assert "* [repeated_behavior] 會主動嗆機器人" in evidence


def test_a_create_delta_writes_a_fact(memory_isolated_dir: Path) -> None:
    """The ordinary path: one delta, one file, code-stamped."""
    scope = user_scope(user_id=111)
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(_delta(from_keys=("preference.a",)),),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert outcome.applied
    assert outcome.created == 1
    stored = read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)
    assert stored[0].owner_id == 111
    assert stored[0].keys == ("preference.a",)
    assert stored[0].compartment == GLOBAL_COMPARTMENT


def test_a_create_whose_keys_already_back_a_fact_updates_it(memory_isolated_dir: Path) -> None:
    """Retrying a partly-applied batch must upsert, not file a duplicate."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact(keys=("preference.a",)))
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(_delta(summary="換個說法", text="新版本", from_keys=("preference.a",)),),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    stored = read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)
    assert outcome.created == 0
    assert len(stored) == 1
    assert stored[0].text == "新版本"


def test_an_update_keeps_the_original_creation_date(memory_isolated_dir: Path) -> None:
    """`created` is the fact's own history; only `last_confirmed` moves."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact())
    apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(_delta(action="update", fact_id="0123456789abcdef", text="改寫"),),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    stored = read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)[0]
    assert stored.created == _NOW
    assert stored.last_confirmed > _NOW


def test_an_update_naming_a_missing_id_becomes_a_create(memory_isolated_dir: Path) -> None:
    """A retry against a changed tree still lands the fact instead of dropping it."""
    scope = user_scope(user_id=111)
    apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(_delta(action="update", fact_id="f" * 16),),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 1


def test_bad_deltas_are_dropped_without_failing_the_batch(memory_isolated_dir: Path) -> None:
    """A deterministic content check that rejects the batch would freeze the scope."""
    scope = user_scope(user_id=111)
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(
            # A server-only section on a user scope.
            _delta(section="culture"),
            _delta(text="   "),
            _delta(fact_id="", text="好的事實"),
        ),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert outcome.applied
    assert outcome.dropped == 2
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 1


def test_an_unusable_subject_id_costs_the_field_and_not_the_run(memory_isolated_dir: Path) -> None:
    """The id used to be cast unguarded here, and raising abandons the rest of the fan-out.

    A user scope is where that bites: `member_alias` is a server-only section, so the guard
    that reads this field never fires on one and every delta reached the cast.
    """
    scope = user_scope(user_id=111)
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(
            _delta(section="fact", summary="猜出來的 id", subject_id="阿華"),
            # `isdigit` accepts a superscript and `int()` refuses it, which is the same raise.
            _delta(summary="上標數字", subject_id="²"),
            # And `isdecimal` accepts a digit string longer than CPython will convert.
            _delta(summary="過長的 id", subject_id="9" * 4400),
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


def test_a_median_merge_is_not_mistaken_for_a_wipe(memory_isolated_dir: Path) -> None:
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
            _delta(summary="合併後的事實"),
            *(_delta(action="delete", fact_id=fact_id) for fact_id in ids[:4]),
        ),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert outcome.applied
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 2


def test_a_wipe_is_refused_and_changes_nothing(memory_isolated_dir: Path) -> None:
    """A batch that only deletes is a lossy rewrite, so it never touches the disk."""
    scope = user_scope(user_id=111)
    ids = [f"{index:016x}" for index in range(10)]
    for fact_id in ids:
        write_fact(scope=scope, fact=_fact(fact_id=fact_id))
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=tuple(_delta(action="delete", fact_id=fact_id) for fact_id in ids),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert not outcome.applied
    assert outcome.rejected == "mass deletion"
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 10


def test_a_rebuild_may_replace_the_whole_set(memory_isolated_dir: Path) -> None:
    """The regeneration path is exempt: replacing everything is what it is for."""
    scope = user_scope(user_id=111)
    ids = [f"{index:016x}" for index in range(10)]
    for fact_id in ids:
        write_fact(scope=scope, fact=_fact(fact_id=fact_id))
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=tuple(_delta(action="delete", fact_id=fact_id) for fact_id in ids),
        owner=_OWNER,
        allow_mass_delete=True,
    )
    assert outcome.applied
    assert read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT) == []


def test_the_sweep_expires_recent_context_but_never_permanent(memory_isolated_dir: Path) -> None:
    """Aging is deterministic code now that the dates are stamped rather than written."""
    scope = user_scope(user_id=111)
    today = _NOW + timedelta(days=60)
    write_fact(
        scope=scope,
        fact=_fact(fact_id="a" * 16, section="recent", durability="recent", last_confirmed=_NOW),
    )
    write_fact(
        scope=scope,
        fact=_fact(
            fact_id="b" * 16, section="permanent", durability="permanent", last_confirmed=_NOW
        ),
    )
    assert sweep_stale_facts(scope=scope, compartment=GLOBAL_COMPARTMENT, today=today) == 1
    assert [fact.fact_id for fact in read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)] == [
        "b" * 16
    ]


def test_stable_facts_age_by_displacement_within_their_own_compartment(
    memory_isolated_dir: Path,
) -> None:
    """A busy compartment self-trims; a quiet one anchored elsewhere forgets nothing."""
    scope = user_scope(user_id=111)
    quiet = guild_compartment(guild_id=222)
    write_fact(scope=scope, fact=_fact(fact_id="a" * 16, last_confirmed=_NOW))
    write_fact(scope=scope, fact=_fact(fact_id="b" * 16, last_confirmed=_NOW + timedelta(days=90)))
    write_fact(scope=scope, fact=_fact(fact_id="c" * 16, compartment=quiet, last_confirmed=_NOW))
    today = _NOW + timedelta(days=365)
    assert sweep_stale_facts(scope=scope, compartment=GLOBAL_COMPARTMENT, today=today) == 1
    # The other compartment saw no newer activity, so nothing displaced its fact.
    assert sweep_stale_facts(scope=scope, compartment=quiet, today=today) == 0
    assert len(read_facts(scope=scope, compartment=quiet)) == 1


def test_existing_facts_render_with_the_ids_the_model_must_echo(memory_isolated_dir: Path) -> None:
    """`fact_id` is the model's only handle on a stored fact, so it has to be shown."""
    rendered = render_existing_facts(facts=[_fact(keys=("preference.a",))])
    assert "[0123456789abcdef] section=preference durability=stable" in rendered
    assert "from_keys: preference.a" in rendered


def test_sections_are_flavor_scoped() -> None:
    """A user scope has no member-alias table and a server scope has no preference section."""
    assert "member_alias" in sections_for_flavor(flavor="server")
    assert "member_alias" not in sections_for_flavor(flavor="user")
    assert "preference" in sections_for_flavor(flavor="user")
    assert "preference" not in sections_for_flavor(flavor="server")


def test_the_owners_dm_renders_each_global_fact_once(memory_isolated_dir: Path) -> None:
    """The owner-DM read set must not name `global` twice.

    `list_compartments` already leads with it, and the reader concatenates whatever it is
    handed with no dedupe, so a prepended second copy renders every cross-server fact
    twice and charges it twice against the injection cap.
    """
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact(fact_id="a" * 16, text="全域事實"))
    write_fact(
        scope=scope, fact=_fact(fact_id="b" * 16, compartment=DM_COMPARTMENT, text="私訊事實")
    )
    compartments = compartments_for_reading(
        owner_id=111, context=MemoryReadContext(guild_id=None, dm_partner_id=111)
    )
    assert compartments == list(dict.fromkeys(compartments))
    document = read_memory_document(scope=scope, compartments=compartments, flavor="user")
    assert document.count("全域事實") == 1
    assert document.count("私訊事實") == 1


def test_an_owner_with_no_stored_facts_still_reads_the_shared_compartment(
    memory_isolated_dir: Path,
) -> None:
    """A scope with no directories yet must still resolve to a readable compartment list."""
    assert compartments_for_reading(
        owner_id=111, context=MemoryReadContext(guild_id=None, dm_partner_id=111)
    ) == [GLOBAL_COMPARTMENT]


def test_a_batch_reports_the_ids_it_wrote(memory_isolated_dir: Path) -> None:
    """A rebuild drops what it did not re-emit, so it needs the written set, not the survivors."""
    scope = user_scope(user_id=111)
    write_fact(scope=scope, fact=_fact(fact_id="a" * 16))
    outcome = apply_deltas(
        scope=scope,
        compartment=GLOBAL_COMPARTMENT,
        flavor="user",
        deltas=(_delta(summary="新的事實"),),
        owner=_OWNER,
        allow_mass_delete=False,
    )
    assert outcome.written == (mint_fact_id(compartment=GLOBAL_COMPARTMENT, summary="新的事實"),)
    # The untouched fact is still on disk; it is the CALLER's job to drop it on a rebuild.
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 2


def test_a_permanent_section_fact_never_ages_even_when_marked_stable(
    memory_isolated_dir: Path,
) -> None:
    """Nothing couples `section` to `durability`, and `render_existing_facts` feeds a
    mismatched pairing back on every later update, so one slip by the model would
    otherwise displace an enforced standing directive out of memory for good.
    """
    scope = user_scope(user_id=111)
    write_fact(
        scope=scope,
        fact=_fact(
            fact_id="a" * 16, section="permanent", durability="stable", last_confirmed=_NOW
        ),
    )
    write_fact(
        scope=scope, fact=_fact(fact_id="b" * 16, last_confirmed=_NOW + timedelta(days=120))
    )
    assert (
        sweep_stale_facts(
            scope=scope, compartment=GLOBAL_COMPARTMENT, today=_NOW + timedelta(days=365)
        )
        == 0
    )
    assert len(read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT)) == 2


def test_a_scope_whose_only_evidence_is_detail_is_still_a_scope(memory_isolated_dir: Path) -> None:
    """`detail.md` alone counts: it is what a rebuild reconstructs everything from, and a
    scope that has gone quiet since its last consolidation holds nothing else — which is
    the steady state for a server, not an edge case. Missing it made the migration skip
    22 of the live store's scopes, half of them server memories.
    """
    scope = user_scope(user_id=111)
    (memory_isolated_dir / scope).mkdir(parents=True)
    (memory_isolated_dir / scope / "detail.md").write_text(
        "## 2026-07-01T00:00:00+00:00\n### stable_fact\n- normalized_key: a\n", encoding="utf-8"
    )
    assert iter_scopes() == ["111"]


def test_an_alias_row_cannot_be_given_someone_elses_id(memory_isolated_dir: Path) -> None:
    """The allowlist parser takes the FIRST `[id: N]` on a row, and an alias body is
    distilled from messages anyone in the server can write. Every id token is stripped
    from the body so only the code-stamped `subject_id` can ever survive.
    """
    scope = server_scope(server_id=500)
    write_fact(
        scope=scope,
        fact=_fact(
            fact_id="a" * 16,
            section="member_alias",
            durability="permanent",
            text="小明[id: 999999999999999999](社群暱稱:明哥)",
            subject_id=777,
        ),
    )
    document = read_memory_document(
        scope=scope, compartments=[GLOBAL_COMPARTMENT], flavor="server"
    )
    assert "999999999999999999" not in document
    assert allowlist_ids_from_server_memory(memory=document) == {777: "小明(社群暱稱:明哥)"}
