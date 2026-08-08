"""Tests for the offline memory regeneration script."""

from typing import TYPE_CHECKING, cast
import asyncio

import pytest
from scripts import regen_memories as regen_script

from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.services.memory.store import user_scope, write_tone, server_scope, append_raw_entry
from discordbot.services.memory.constants import MEMORY_GLOBAL_CONCURRENCY

if TYPE_CHECKING:
    from discordbot.services.memory.extraction import MemoryExtractorAI

pytestmark = pytest.mark.usefixtures("memory_isolated_dir")

_USER = user_scope(user_id=111)
_OTHER_USER = user_scope(user_id=222)
_SERVER = server_scope(server_id=333)


def _seed(scope: str) -> None:
    """Gives a scope enough on disk for `iter_scopes` to return it."""
    append_raw_entry(scope=scope, entry_text="### stable_preference\n- summary_zh: 喜歡簡短回覆")


def test_all_reaches_user_and_server_scopes() -> None:
    """The default target covers both flavors, which neither old script did alone."""
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
    """The scope gap this script used to have: server memory was unreachable."""
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
    """A mistyped id fails loudly instead of reporting itself as rebuilding empty."""
    _seed(scope=_USER)
    with pytest.raises(SystemExit):
        regen_script._scopes_for_target(target=_OTHER_USER)


def test_a_collective_target_over_an_empty_store_is_not_an_error() -> None:
    """`all` over nothing is an empty run, not the single-scope refusal."""
    assert regen_script._scopes_for_target(target="all") == []


def test_parse_args_defaults_to_the_whole_store_and_the_writer_tier() -> None:
    """A bare invocation is a dry run over every scope with the runtime writer model."""
    args = regen_script._parse_args(argv=[])
    writer = RuntimeModelCatalog().memory_writer_model
    assert args.target == "all"
    assert args.apply is False
    assert args.model == writer.name
    assert args.effort == writer.effort


def test_the_offline_fan_out_does_not_reuse_the_live_bots_concurrency_cap() -> None:
    """The issue asks for the script's own bound, in the 8-10 band, with no flag."""
    assert regen_script._CONCURRENCY != MEMORY_GLOBAL_CONCURRENCY
    assert 8 <= regen_script._CONCURRENCY <= 10
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
        model=ModelSettings(name="test-model", effort="low"), target=target, apply=False
    )
    output = " ".join(capsys.readouterr().out.split())
    assert "Stop the bot before --apply" in output
    assert ("commit data/memories first" in output) is expects_store_line


async def test_the_dry_run_flags_a_scope_with_nothing_left_to_rebuild_from(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The loud note has to reach the preview, not only the run that already rewrote."""
    write_tone(scope=_USER, content="## 語氣偏好\n- 簡短")
    await regen_script._regen_all(
        model=ModelSettings(name="test-model", effort="low"), target=_USER, apply=False
    )
    output = " ".join(capsys.readouterr().out.split())
    assert "REBUILDS EMPTY" in output
    assert "EMPTY GLOBAL" not in output


async def test_a_scope_key_that_is_not_a_discord_id_becomes_one_error_row() -> None:
    """`read_owner` parses the id, and it used to raise past the handler into the gather."""
    _seed(scope="111.bak")
    scope, result, _ = await regen_script._regen_one(
        extractor=cast("MemoryExtractorAI", None), scope="111.bak", semaphore=asyncio.Semaphore(1)
    )
    assert scope == "111.bak"
    assert result.startswith("error: ValueError")


@pytest.mark.parametrize(
    ("result", "buckets", "expected"),
    [
        ("no_evidence", {}, "REBUILDS EMPTY"),
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
