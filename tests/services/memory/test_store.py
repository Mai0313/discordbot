"""Tests for the memory store: scopes, compartments, fact files and the evidence tiers."""

import re
import time
from pathlib import Path
from collections.abc import Callable

import pytest

from discordbot.cogs.gen_reply.recall import RecallContext, compartments_for_reading
from discordbot.services.memory.facts import render_fact_file
from discordbot.services.memory.store import (
    DM_COMPARTMENT,
    GLOBAL_COMPARTMENT,
    BOT_MEMORY_DIR_NAME,
    clear_raw,
    flavor_of,
    read_tone,
    read_facts,
    read_owner,
    scope_lock,
    user_scope,
    write_fact,
    write_tone,
    delete_fact,
    iter_scopes,
    mark_cleared,
    server_scope,
    append_detail,
    cleared_since,
    read_evidence,
    raw_file_bytes,
    scope_owner_id,
    compartment_dir,
    append_raw_entry,
    read_detail_tail,
    read_raw_entries,
    count_raw_entries,
    guild_compartment,
    list_compartments,
    prune_compartment,
    unaccounted_files,
    delete_memory_files,
    read_memory_document,
)

from tests.helpers.user_memory import (
    _BIG5,
    _OWNER,
    USER_ID,
    USER_SCOPE,
    _GUILD_222,
    _OLDEST_DETAIL_HEADER,
    _fact,
    _memory_text,
    _stored_fact,
)
from tests.helpers.server_memory import GUILD_ID, SERVER_SCOPE, _server_fact, _server_document


def test_a_scope_with_no_facts_renders_an_empty_document(memory_isolated_dir: Path) -> None:
    """The read path answers "" for an unknown scope rather than raising on a missing dir."""
    assert _memory_text() == ""


def test_a_written_fact_comes_back_through_the_document_read(memory_isolated_dir: Path) -> None:
    """One fact per file: the write lands atomically and the render finds it again."""
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="測試內容"))
    assert "測試內容" in _memory_text()
    leftovers = list((memory_isolated_dir / str(USER_ID) / GLOBAL_COMPARTMENT).glob("*.tmp"))
    assert leftovers == []


def test_the_store_never_clamps_a_fact_body(memory_isolated_dir: Path) -> None:
    """Growth is bounded by the consolidation compaction pass, never by a silent truncation."""
    body = "長" * 50_000
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text=body))
    stored = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [len(fact.text) for fact in stored] == [len(body)]


def test_append_raw_entry_creates_timestamped_entries(memory_isolated_dir: Path) -> None:
    append_raw_entry(scope=USER_SCOPE, entry_text="偏好訊號:\n- 喜歡簡短回覆")
    append_raw_entry(scope=USER_SCOPE, entry_text="穩定事實:\n- 慣用繁體中文")
    assert count_raw_entries(scope=USER_SCOPE) == 2
    raw_file = (memory_isolated_dir / str(USER_ID) / "raw.md").read_text(encoding="utf-8")
    stamp = r"## \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00"
    assert re.fullmatch(
        pattern=rf"{stamp}\n偏好訊號:\n- 喜歡簡短回覆\n\n{stamp}\n穩定事實:\n- 慣用繁體中文\n",
        string=raw_file,
    )


def test_append_raw_entry_evicts_oldest_on_overflow(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("discordbot.services.memory.store.RAW_FILE_MAX_BYTES", 280)
    append_raw_entry(scope=USER_SCOPE, entry_text="first entry " + "a" * 100)
    append_raw_entry(scope=USER_SCOPE, entry_text="second entry " + "b" * 100)
    raw_text = read_raw_entries(scope=USER_SCOPE)
    assert "first entry" not in raw_text
    assert "second entry" in raw_text
    assert count_raw_entries(scope=USER_SCOPE) == 1
    # The evicted entry is preserved in the detail file.
    detail_text = (memory_isolated_dir / str(USER_ID) / "detail.md").read_text(encoding="utf-8")
    assert "first entry" in detail_text


def test_append_raw_entry_truncates_single_oversized_entry(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("discordbot.services.memory.store.RAW_FILE_MAX_BYTES", 80)
    append_raw_entry(scope=USER_SCOPE, entry_text="oversized " + "c" * 200)
    assert count_raw_entries(scope=USER_SCOPE) == 1
    # The lone entry cannot be evicted, so it is truncated to honor the cap.
    assert raw_file_bytes(scope=USER_SCOPE) <= 80 + 1


def test_raw_file_bytes_missing_file_is_zero(memory_isolated_dir: Path) -> None:
    assert raw_file_bytes(scope=USER_SCOPE) == 0
    append_raw_entry(scope=USER_SCOPE, entry_text="something")
    assert raw_file_bytes(scope=USER_SCOPE) > 0


def test_clear_raw_removes_only_raw_file(memory_isolated_dir: Path) -> None:
    """Retiring a consumed batch must not touch the facts that batch just produced."""
    write_fact(scope=USER_SCOPE, fact=_stored_fact())
    append_raw_entry(scope=USER_SCOPE, entry_text="raw entry")
    clear_raw(scope=USER_SCOPE)
    assert count_raw_entries(scope=USER_SCOPE) == 0
    assert _memory_text() != ""


def test_delete_memory_files_removes_files_and_directory(memory_isolated_dir: Path) -> None:
    """Every tier goes, the emptied scope directory with them, and a repeat delete is a no-op."""
    write_fact(scope=USER_SCOPE, fact=_stored_fact())
    write_fact(scope=USER_SCOPE, fact=_stored_fact(fact_id="1" * 16, compartment=DM_COMPARTMENT))
    write_fact(scope=USER_SCOPE, fact=_stored_fact(fact_id="2" * 16, compartment=_GUILD_222))
    append_raw_entry(scope=USER_SCOPE, entry_text="raw entry")
    append_detail(scope=USER_SCOPE, text="## 2026-01-01T00:00:00 | x\n舊證據")
    assert delete_memory_files(scope=USER_SCOPE) is True
    assert _memory_text() == ""
    assert count_raw_entries(scope=USER_SCOPE) == 0
    assert list_compartments(scope=USER_SCOPE) == []
    assert not (memory_isolated_dir / str(USER_ID)).exists()
    assert delete_memory_files(scope=USER_SCOPE) is False


def test_delete_memory_files_tolerates_leftover_tmp(memory_isolated_dir: Path) -> None:
    """A crash between a tmp write and its rename must not leave the scope unclearable."""
    write_fact(scope=USER_SCOPE, fact=_stored_fact())
    append_raw_entry(scope=USER_SCOPE, entry_text="raw entry")
    user_dir = memory_isolated_dir / str(USER_ID)
    (user_dir / "raw.md.tmp").write_text(data="partial", encoding="utf-8")
    (user_dir / GLOBAL_COMPARTMENT / "deadbeefdeadbeef.md.tmp").write_text(
        data="partial", encoding="utf-8"
    )
    assert delete_memory_files(scope=USER_SCOPE) is True
    assert not user_dir.exists()


def test_mark_cleared_flags_in_flight_updates(memory_isolated_dir: Path) -> None:
    started_at = time.monotonic()
    assert cleared_since(scope=USER_SCOPE, started_at=started_at) is False
    mark_cleared(scope=USER_SCOPE)
    assert cleared_since(scope=USER_SCOPE, started_at=started_at) is True
    later = time.monotonic()
    assert cleared_since(scope=USER_SCOPE, started_at=later) is False


async def test_user_lock_is_stable_per_user(memory_isolated_dir: Path) -> None:
    lock_a = scope_lock(scope=USER_SCOPE)
    lock_b = scope_lock(scope=USER_SCOPE)
    lock_other = scope_lock(scope=user_scope(user_id=USER_ID + 1))
    assert lock_a is lock_b
    assert lock_a is not lock_other


def test_read_detail_tail_missing_file_is_empty(memory_isolated_dir: Path) -> None:
    assert read_detail_tail(scope=USER_SCOPE, max_chars=100) == ""


def test_read_detail_tail_window_aligns_to_entry_header(memory_isolated_dir: Path) -> None:
    entry_one = "## 2026-01-01T00:00:00+00:00\n第一筆細節"
    entry_two = "## 2026-02-01T00:00:00+00:00\n第二筆細節"
    user_dir = memory_isolated_dir / str(USER_ID)
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "detail.md").write_text(data=f"{entry_one}\n\n{entry_two}\n", encoding="utf-8")
    full = read_detail_tail(scope=USER_SCOPE, max_chars=10_000)
    assert "第一筆細節" in full
    assert "第二筆細節" in full
    # A window cutting into entry one drops the partial entry and starts at the
    # next header.
    windowed = read_detail_tail(scope=USER_SCOPE, max_chars=len(entry_two) + 4)
    assert windowed.startswith("## 2026-02-01")
    assert "第一筆細節" not in windowed


_NEWEST_DETAIL = "## 2026-02-01T00:00:00+00:00\n第二筆細節"
_CUT_WINDOW_CHARS = len(_NEWEST_DETAIL) + 4


def _detail_cut_inside(older: bytes, opening: int) -> bytes:
    """Returns a `detail.md` whose window at `_CUT_WINDOW_CHARS` opens `opening` bytes before `older` ends.

    The window is the file's last `4 * max_chars` bytes, so ASCII padding after `older` sizes it.
    """
    newest = f"\n\n{_NEWEST_DETAIL}\n".encode()
    return older + b"." * (4 * _CUT_WINDOW_CHARS - opening - len(newest)) + newest


@pytest.mark.parametrize("opening", [1, 2, 3])
def test_read_detail_tail_skips_the_character_its_window_cut(
    opening: int, memory_isolated_dir: Path
) -> None:
    """A window opening on the trailing bytes of a character it split reads on past them."""
    user_dir = memory_isolated_dir / str(USER_ID)
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "detail.md").write_bytes(
        _detail_cut_inside(older=_OLDEST_DETAIL_HEADER + "😀".encode(), opening=opening)
    )
    assert read_detail_tail(scope=USER_SCOPE, max_chars=_CUT_WINDOW_CHARS) == _NEWEST_DETAIL


@pytest.mark.parametrize(
    ("content", "max_chars"),
    [
        pytest.param(_OLDEST_DETAIL_HEADER + _BIG5 + b"\n", 10_000, id="re-saved-as-big5"),
        pytest.param(
            _detail_cut_inside(older=_OLDEST_DETAIL_HEADER + _BIG5, opening=len(_BIG5)),
            _CUT_WINDOW_CHARS,
            id="big5-in-a-cut-window",
        ),
        pytest.param(
            _OLDEST_DETAIL_HEADER + "第一筆".encode()[:-1] + f"\n\n{_NEWEST_DETAIL}\n".encode(),
            10_000,
            id="torn-append",
        ),
    ],
)
def test_read_detail_tail_raises_on_bytes_no_window_cut_explains(
    content: bytes, max_chars: int, memory_isolated_dir: Path
) -> None:
    """A `detail.md` that is not UTF-8 fails its read instead of coming back garbled."""
    user_dir = memory_isolated_dir / str(USER_ID)
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "detail.md").write_bytes(content)
    with pytest.raises(UnicodeDecodeError):
        read_detail_tail(scope=USER_SCOPE, max_chars=max_chars)


def test_read_evidence_puts_the_detail_tail_ahead_of_raw(memory_isolated_dir: Path) -> None:
    """The corpus is read as one oldest-first batch, and a missing tier adds no separator."""
    assert read_evidence(scope=USER_SCOPE) == ""
    append_raw_entry(scope=USER_SCOPE, entry_text="- 還沒整理的觀察")
    raw = read_raw_entries(scope=USER_SCOPE)
    assert read_evidence(scope=USER_SCOPE) == raw
    append_detail(scope=USER_SCOPE, text="## 2026-01-01T00:00:00+00:00\n已整理的觀察")
    detail = read_detail_tail(scope=USER_SCOPE, max_chars=10_000)
    assert read_evidence(scope=USER_SCOPE) == f"{detail}\n\n{raw}"


def test_append_detail_trims_oldest_past_cap(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("discordbot.services.memory.store.DETAIL_FILE_MAX_BYTES", 300)
    monkeypatch.setattr("discordbot.services.memory.store.DETAIL_FILE_TRIM_TARGET_BYTES", 200)
    for index in range(6):
        append_detail(
            scope=USER_SCOPE,
            text=f"## 2026-01-0{index + 1}T00:00:00+00:00 | x\nentry {index} " + "a" * 80,
        )
    detail_path = memory_isolated_dir / str(USER_ID) / "detail.md"
    text = detail_path.read_text(encoding="utf-8")
    # The newest entry always survives, the oldest entries are gone for good,
    # and the file honors the cap.
    assert "entry 5" in text
    assert "entry 0" not in text
    assert len(text.encode("utf-8")) <= 300 + 1
    assert not detail_path.with_suffix(".md.tmp").exists()


def test_iter_scopes_only_descends_into_the_bot_memory_directory(
    memory_isolated_dir: Path,
) -> None:
    server = server_scope(server_id=555)
    append_raw_entry(scope=server, entry_text="- s")
    # Nested memory anywhere else is not a scope, so a stray directory (or a symlink
    # to `bot_memories`) can never hand the sweep the same memory under a second name.
    (memory_isolated_dir / "999" / "555").mkdir(parents=True)
    (memory_isolated_dir / "999" / "555" / "raw.md").write_text("- s", encoding="utf-8")
    assert iter_scopes() == [server]


@pytest.mark.usefixtures("memory_isolated_dir")
def test_iter_scopes_skips_a_directory_not_named_after_an_id() -> None:
    """A backup copy holds memory but is no scope: its owner id cannot be parsed back out."""
    scopes = [user_scope(user_id=111), server_scope(server_id=333)]
    for key in [*scopes, *(f"{scope}.bak" for scope in scopes)]:
        append_raw_entry(scope=key, entry_text="- s")
    assert iter_scopes() == scopes


def test_flavor_of_distinguishes_user_and_server() -> None:
    assert flavor_of(scope=user_scope(user_id=USER_ID)) == "user"
    assert flavor_of(scope=server_scope(server_id=2)) == "server"


def test_read_tone_missing_file_returns_empty(memory_isolated_dir: Path) -> None:
    assert read_tone(scope=USER_SCOPE) == ""


def test_write_tone_roundtrip(memory_isolated_dir: Path) -> None:
    write_tone(scope=USER_SCOPE, content="## 語氣偏好\n* 偏好禮貌\n")
    assert read_tone(scope=USER_SCOPE) == "## 語氣偏好\n* 偏好禮貌"
    leftovers = list((memory_isolated_dir / str(USER_ID)).glob("*.tmp"))
    assert leftovers == []


def test_write_tone_truncates_past_byte_cap(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("discordbot.services.memory.store.TONE_FILE_MAX_BYTES", 32)
    write_tone(scope=USER_SCOPE, content="## 語氣偏好\n" + "長" * 100)
    stored = read_tone(scope=USER_SCOPE)
    assert stored.startswith("## 語氣偏好")
    assert len(stored.encode("utf-8")) <= 32


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
            owner_id=111, context=RecallContext(guild_id=111, dm_partner_id=None)
        ),
        flavor="user",
    )
    assert "全域事實" in document
    assert "本群事實" in document
    assert "他群祕密" not in document
    assert "私訊祕密" not in document


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


_BOTH_FLAVORS = (user_scope(user_id=111), server_scope(server_id=500))


@pytest.mark.parametrize(
    ("write_tier", "scopes"),
    [
        pytest.param(
            lambda scope: append_raw_entry(scope=scope, entry_text="- 觀察"),
            _BOTH_FLAVORS,
            id="raw",
        ),
        pytest.param(
            lambda scope: append_detail(
                scope=scope,
                text="## 2026-07-01T00:00:00+00:00\n### stable_fact\n- normalized_key: a",
            ),
            _BOTH_FLAVORS,
            id="detail",
        ),
        # A server scope has no tone tier.
        pytest.param(
            lambda scope: write_tone(scope=scope, content="## 語氣偏好\n* 簡短"),
            (user_scope(user_id=111),),
            id="tone",
        ),
        pytest.param(
            lambda scope: write_fact(scope=scope, fact=_fact()), _BOTH_FLAVORS, id="fact"
        ),
    ],
)
def test_iter_scopes_finds_a_scope_holding_one_tier_and_skips_dot_dirs(
    memory_isolated_dir: Path, write_tier: Callable[[str], None], scopes: tuple[str, ...]
) -> None:
    """Any one tier its flavor can hold makes a scope, and the git directory never is one.

    `detail.md` alone counts: it is what a rebuild reconstructs everything from, and a scope
    that has gone quiet since its last consolidation holds nothing else — which is the steady
    state for a server, not an edge case.
    """
    for scope in scopes:
        write_tier(scope)
    (memory_isolated_dir / ".git").mkdir(parents=True, exist_ok=True)
    (memory_isolated_dir / ".git" / "raw.md").write_text("## 2026-01-01T00:00:00+00:00\n")
    assert iter_scopes() == list(scopes)


def test_iter_scopes_ignores_a_scope_whose_only_file_is_unreadable(
    memory_isolated_dir: Path,
) -> None:
    """A file no reader can parse must not answer for a whole scope on its own.

    Listed rather than parsed, such a scope would stay on `iter_scopes` for good, so the
    restart sweep and the offline rebuild would pick it up on every run with nothing either
    could do about it.
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
    assert pruned.unaccounted == ()
    # Both of them were content no reader could reach, so both are reported as destroyed;
    # the stranded `.md.tmp` is not, since `_fact_paths` never globbed it to begin with.
    assert pruned.unreadable == (broken.name, misfiled.name)
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
    assert pruned.unaccounted == ("backup.txt", "notes.md")
    # The one fact file it removed parsed fine; a rebuild drops those by not re-emitting
    # them, which is the ordinary outcome and not a loss anyone has to be told about.
    assert pruned.unreadable == ()
    assert (directory / "notes.md").exists()
    assert read_facts(scope=scope, compartment=GLOBAL_COMPARTMENT) == []


def test_a_file_the_store_cannot_decode_never_stops_the_sweep(memory_isolated_dir: Path) -> None:
    """`iter_scopes` parses the tier, so an undecodable file must degrade, not raise.

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
    assert pruned.unreadable == (mojibake.name,)
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
    assert pruned.unaccounted == (f"{'b' * 16}\n.md",)
    assert forged.exists()


def test_a_directory_named_like_a_fact_file_never_reaches_a_reader(
    memory_isolated_dir: Path,
) -> None:
    """`iter_scopes` parses the tier, so a reader that raises takes the sweep with it.

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

    assert prune_compartment(scope=scope, compartment=compartment, keep=set()).unaccounted == ()
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
        == ()
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
    assert pruned.unaccounted == ()
    assert pruned.unreadable == ()
    assert not directory.exists()
    assert list_compartments(scope=scope) == []


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


def test_server_scope_nests_under_the_fixed_bot_directory() -> None:
    """The bot directory is fixed, so a bot account change never strands a scope."""
    assert server_scope(server_id=GUILD_ID) == f"{BOT_MEMORY_DIR_NAME}/{GUILD_ID}"


def test_user_and_server_scopes_never_collide() -> None:
    """A user scope is a bare snowflake; a server scope always carries a `/`."""
    assert "/" not in user_scope(user_id=GUILD_ID)
    assert "/" in server_scope(server_id=GUILD_ID)
    assert user_scope(user_id=GUILD_ID) != server_scope(server_id=GUILD_ID)


def test_server_scope_isolated_from_user_scope_on_disk(memory_isolated_dir: Path) -> None:
    """One snowflake read as a user and as a server keeps two separate fact trees."""
    scope = user_scope(user_id=GUILD_ID)
    write_fact(scope=scope, fact=_server_fact(fact_id="a" * 16, section="profile", text="個人"))
    write_fact(
        scope=SERVER_SCOPE, fact=_server_fact(fact_id="b" * 16, section="profile", text="社群")
    )
    user_document = read_memory_document(
        scope=scope, compartments=[GLOBAL_COMPARTMENT], flavor="user"
    )
    assert "個人" in user_document
    assert "社群" not in user_document
    # The same section key renders under the flavor's own heading.
    assert "## 使用者輪廓" in user_document
    assert "## 伺服器輪廓" in _server_document()
    assert "個人" not in _server_document()
    # Server observations carry no source to route by, so the scope never grows a
    # second compartment.
    assert list_compartments(scope=SERVER_SCOPE) == [GLOBAL_COMPARTMENT]
    assert (
        memory_isolated_dir
        / BOT_MEMORY_DIR_NAME
        / str(GUILD_ID)
        / GLOBAL_COMPARTMENT
        / f"{'b' * 16}.md"
    ).exists()
