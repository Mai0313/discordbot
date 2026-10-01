"""Tests for the offline memory regeneration script."""

from typing import TYPE_CHECKING, cast
import asyncio

import pytest
from scripts import regen_memories as regen_script

from discordbot.typings.memory import MemoryOwner
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    read_tone,
    read_facts,
    user_scope,
    write_fact,
    write_tone,
    server_scope,
    compartment_dir,
    append_raw_entry,
    read_raw_entries,
)
from discordbot.services.memory.prompts import PHASE2_PROMPT, PHASE1_EVALUATOR_PROMPT
from discordbot.services.memory.regeneration import RegenerationReport
from discordbot.services.memory.server_prompts import (
    SERVER_PHASE2_PROMPT,
    SERVER_PHASE1_EVALUATOR_PROMPT,
)

from tests.helpers.memory import make_fact

if TYPE_CHECKING:
    from discordbot.services.memory.writer import MemoryWriterAI

pytestmark = pytest.mark.usefixtures("memory_isolated_dir")

_USER = user_scope(user_id=111)
_OTHER_USER = user_scope(user_id=222)
_SERVER = server_scope(server_id=333)


def _seed(scope: str) -> None:
    """Gives a scope enough on disk for `iter_scopes` to return it."""
    append_raw_entry(scope=scope, entry_text="### stable_preference\n- summary_zh: 喜歡簡短回覆")


def test_all_reaches_user_and_server_scopes() -> None:
    """The default target covers both flavors."""
    for scope in (_USER, _OTHER_USER, _SERVER):
        _seed(scope=scope)
    every = regen_script._scopes_for_target(target="all")
    assert every == [_USER, _OTHER_USER, _SERVER]
    assert sorted(
        regen_script._scopes_for_target(target="users")
        + regen_script._scopes_for_target(target="servers")
    ) == sorted(every)


def test_users_excludes_the_bot_memories_tree() -> None:
    """`users` stays on user scopes even with server memory on disk."""
    _seed(scope=_USER)
    _seed(scope=_SERVER)
    assert regen_script._scopes_for_target(target="users") == [_USER]


def test_servers_reaches_the_bot_memories_tree() -> None:
    """`servers` reaches every server scope under the bot memory tree."""
    _seed(scope=_USER)
    _seed(scope=_SERVER)
    assert regen_script._scopes_for_target(target="servers") == [_SERVER]


@pytest.mark.parametrize("target", [_USER, _SERVER])
def test_a_scope_key_targets_only_itself(target: str) -> None:
    """Either flavor can be named by its own scope key."""
    for scope in (_USER, _OTHER_USER, _SERVER):
        _seed(scope=scope)
    assert regen_script._scopes_for_target(target=target) == [target]


def test_a_scope_with_nothing_on_disk_is_refused() -> None:
    """A mistyped id fails loudly instead of passing itself off as a skipped scope."""
    _seed(scope=_USER)
    with pytest.raises(SystemExit):
        regen_script._scopes_for_target(target=_OTHER_USER)


def test_a_collective_target_over_an_empty_store_is_not_an_error() -> None:
    """`all` over nothing is an empty run, not the single-scope refusal."""
    assert regen_script._scopes_for_target(target="all") == []


def test_parse_args_defaults_to_the_real_rebuild_over_the_whole_store() -> None:
    """The rebuild is the common case, so the preview is the flag and not the default."""
    args = regen_script._parse_args(argv=[])
    writer = RuntimeModelCatalog().memory_writer_model
    assert args.target == "all"
    assert args.dry_run is False
    assert regen_script._parse_args(argv=["--dry-run"]).dry_run is True
    assert args.model == writer.name
    assert args.effort == writer.effort


async def test_the_offline_fan_out_runs_under_its_own_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The script carries its own bound, tuned by hand rather than exposed as a flag."""
    for scope in (_USER, _OTHER_USER, _SERVER):
        _seed(scope=scope)
    monkeypatch.setattr(regen_script, "_CONCURRENCY", 2)
    in_flight = 0
    peak = 0

    async def _rebuild(scope: str, writer: object, identity: str) -> object:
        """Holds its slot long enough for every scope the bound allows to be in flight."""
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return RegenerationReport(result="no_evidence")

    monkeypatch.setattr(regen_script, "regenerate_scope_memory", _rebuild)

    await regen_script._rebuild_batch(
        writers=dict.fromkeys(("user", "server"), cast("MemoryWriterAI", None)),
        scopes=regen_script._scopes_for_target(target="all"),
    )

    assert peak == 2
    assert "concurrency" not in vars(regen_script._parse_args(argv=[]))


@pytest.mark.parametrize(
    ("target", "expects_store_line"), [("all", True), ("users", True), (_USER, False)]
)
async def test_the_stop_the_bot_warning_fires_on_every_target(
    target: str, expects_store_line: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """An out-of-process write races the bot on one scope too; only blast radius grades."""
    _seed(scope=_USER)
    await regen_script._regen_all(
        model=ModelSettings(name="test-model", effort="low"), target=target, dry_run=True
    )
    output = " ".join(capsys.readouterr().out.split())
    assert "Stop the bot before rebuilding" in output
    assert ("commit data/memories first" in output) is expects_store_line


@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_scope_with_no_evidence_is_reported_as_skipped_and_kept(
    dry_run: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rebuild never touches such a scope, so neither report may call it emptied."""
    write_fact(scope=_USER, fact=make_fact(owner=MemoryOwner(owner_id=111, owner_name="Alice")))
    write_tone(scope=_USER, content="## 語氣偏好\n- 簡短")
    monkeypatch.setattr(regen_script.console, "input", lambda *args, **kwargs: "y")
    monkeypatch.setattr(regen_script, "AsyncOpenAI", lambda **kwargs: object())

    await regen_script._regen_all(
        model=ModelSettings(name="test-model", effort="low"), target=_USER, dry_run=dry_run
    )

    output = " ".join(capsys.readouterr().out.split())
    assert "SKIPPED" in output
    assert "REBUILDS EMPTY" not in output
    assert "EMPTY GLOBAL" not in output
    # The note says the memory is kept, so the same run must have kept it.
    assert read_facts(scope=_USER, compartment=GLOBAL_COMPARTMENT)
    assert read_tone(scope=_USER) == "## 語氣偏好\n- 簡短"


async def test_the_dry_run_skips_only_the_scopes_the_rebuild_skips(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Evidence holding no observation is still rebuilt, from nothing, so the scope empties."""
    write_fact(scope=_USER, fact=make_fact(owner=MemoryOwner(owner_id=111, owner_name="Alice")))
    append_raw_entry(scope=_USER, entry_text="### forget_request\n- summary_zh: 忘記回覆偏好")

    await regen_script._regen_all(
        model=ModelSettings(name="test-model", effort="low"), target=_USER, dry_run=True
    )

    output = " ".join(capsys.readouterr().out.split())
    assert "REBUILDS EMPTY" in output
    assert "SKIPPED" not in output


async def test_the_dry_run_names_files_a_rebuild_cannot_account_for(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A rebuild never removes a file the store did not write, and logfire is off here."""
    _seed(scope=_USER)
    directory = compartment_dir(scope=_USER, compartment=GLOBAL_COMPARTMENT)
    directory.mkdir(parents=True)
    (directory / "backup.txt").write_text("備份", encoding="utf-8")

    await regen_script._regen_all(
        model=ModelSettings(name="test-model", effort="low"), target=_USER, dry_run=True
    )

    output = " ".join(capsys.readouterr().out.split())
    assert "UNACCOUNTED" in output
    assert "backup.txt" in output


async def test_a_batch_run_counts_finished_scopes_and_reports_them_in_store_order(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`_CONCURRENCY` scopes are in flight at once, so the bar tracks the batch, not one."""
    for scope in (_USER, _OTHER_USER, _SERVER):
        _seed(scope=scope)
    scopes = regen_script._scopes_for_target(target="all")

    async def _land_in_reverse(scope: str, writer: object, identity: str) -> object:
        """Finishes the batch back to front, which is what the re-keying has to survive."""
        await asyncio.sleep(0.01 * (len(scopes) - scopes.index(scope)))
        return RegenerationReport(result="no_evidence")

    monkeypatch.setattr(regen_script, "regenerate_scope_memory", _land_in_reverse)

    rows = await regen_script._rebuild_batch(
        writers=dict.fromkeys(("user", "server"), cast("MemoryWriterAI", None)), scopes=scopes
    )

    assert [row.scope for row in rows] == scopes
    assert "3/3" in " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize("answer", ["n", "", "yes", pytest.param(EOFError, id="nothing-on-stdin")])
async def test_a_run_that_is_not_confirmed_builds_no_client_and_writes_nothing(
    answer: str | type[EOFError],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Nothing is written without a typed `y`, and silence is not consent."""
    _seed(scope=_USER)

    def _answer(*args: object, **kwargs: object) -> str:
        if isinstance(answer, str):
            return answer
        raise answer

    def _refuse_client(*args: object, **kwargs: object) -> object:
        raise AssertionError("the run built a client without a confirmation")

    monkeypatch.setattr(regen_script.console, "input", _answer)
    monkeypatch.setattr(regen_script, "AsyncOpenAI", _refuse_client)

    await regen_script._regen_all(
        model=ModelSettings(name="test-model", effort="low"), target=_USER, dry_run=False
    )

    # A rebuild ends by unlinking `raw.md`, so its survival is what says nothing ran.
    assert read_raw_entries(scope=_USER)
    # A run that stops after the warnings with no word for it reads like a crash.
    assert "nothing was written" in " ".join(capsys.readouterr().out.split())


async def test_a_server_scope_rebuilds_under_the_server_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prompts ride on the writer, so a server scope needs the server-flavor one."""
    _seed(scope=_USER)
    _seed(scope=_SERVER)
    prompts: dict[str, tuple[str, str]] = {}

    async def _record(scope: str, writer: object, identity: str) -> object:
        chosen = cast("MemoryWriterAI", writer)
        prompts[scope] = (chosen.evaluator_prompt, chosen.consolidate_prompt)
        return RegenerationReport(result="no_evidence")

    monkeypatch.setattr(regen_script.console, "input", lambda *args, **kwargs: "y")
    monkeypatch.setattr(regen_script, "AsyncOpenAI", lambda **kwargs: object())
    monkeypatch.setattr(regen_script, "regenerate_scope_memory", _record)

    await regen_script._regen_all(
        model=ModelSettings(name="test-model", effort="low"), target="all", dry_run=False
    )

    assert prompts == {
        _USER: (PHASE1_EVALUATOR_PROMPT, PHASE2_PROMPT),
        _SERVER: (SERVER_PHASE1_EVALUATOR_PROMPT, SERVER_PHASE2_PROMPT),
    }


def test_the_report_says_how_many_fact_files_a_run_destroyed_unread(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """What a rebuild removed unread has no other trace offline: logfire reaches nobody."""
    regen_script._report(
        rows=[
            regen_script._ScopeRow(
                scope=_USER, result="regenerated", counts={"global": 2}, unreadable_removed=3
            )
        ]
    )

    output = " ".join(capsys.readouterr().out.split())
    assert "UNREADABLE: 3 fact file(s) removed unread" in output


async def test_a_scope_key_that_is_not_a_discord_id_becomes_one_error_row() -> None:
    """`read_owner` parses the id, so its raise must stay inside the handler, not the gather."""
    _seed(scope="111.bak")
    row = await regen_script._regen_one(
        writer=cast("MemoryWriterAI", None), scope="111.bak", semaphore=asyncio.Semaphore(1)
    )
    assert row.scope == "111.bak"
    assert row.result.startswith("error: ValueError")
    # A run that never reached the store destroyed nothing, and must not imply it did.
    assert row.unreadable_removed == 0


@pytest.mark.parametrize(
    ("result", "buckets", "expected"),
    [
        ("dry-run", {}, "REBUILDS EMPTY"),
        ("dry-run", {"global": 0}, "REBUILDS EMPTY"),
        ("regenerated", {"g/1": 3}, "EMPTY GLOBAL"),
        ("regenerated", {"global": 2, "g/1": 3}, ""),
    ],
)
def test_loss_note_flags_the_two_expected_losses(
    result: str, buckets: dict[str, int], expected: str
) -> None:
    """The dry run says which scopes rebuild empty or lose their cross-server compartment."""
    assert regen_script._loss_note(result=result, buckets=buckets).startswith(expected)


@pytest.mark.parametrize("buckets", [{}, {"global": 0}])
def test_a_rebuild_that_kept_no_fact_is_not_reported_as_having_no_evidence(
    buckets: dict[str, int],
) -> None:
    """Evidence that yields no fact is still evidence; only a skipped scope had none."""
    note = regen_script._loss_note(result="regenerated", buckets=buckets)
    assert note.startswith("REBUILDS EMPTY")
    assert "no evidence" not in note


@pytest.mark.parametrize(
    "result", ["failed", "error: ValueError: invalid literal for int() with base 10: '111.bak'"]
)
def test_a_rebuild_that_stopped_short_is_not_reported_as_rebuilt_empty(result: str) -> None:
    """It may have stopped before distilling anything, so its empty counts prove nothing."""
    assert regen_script._loss_note(result=result, buckets={}) == ""
