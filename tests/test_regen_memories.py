"""Tests for the offline memory regeneration script."""

import pytest
from scripts import regen_memories as regen_script

from discordbot.services.memory.store import user_scope, server_scope, append_raw_entry

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
    assert regen_script._scopes_for_target(target="all") == [_USER, _OTHER_USER, _SERVER]


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
    assert args.target == "all"
    assert args.apply is False
    assert args.model


@pytest.mark.parametrize(
    ("result", "buckets", "expected"),
    [
        ("no_evidence", {}, "REBUILDS EMPTY"),
        ("regenerated", {"g/1": 3}, "EMPTY GLOBAL"),
        ("regenerated", {"global": 2, "g/1": 3}, ""),
    ],
)
def test_loss_note_flags_the_two_expected_losses(
    result: str, buckets: dict[str, int], expected: str
) -> None:
    """The dry run says which scopes rebuild empty or lose their cross-server compartment."""
    assert regen_script._loss_note(result=result, buckets=buckets).startswith(expected)
