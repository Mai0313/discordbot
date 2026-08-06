"""Pins the memory store's best-effort git history (`services/memory/git_history.py`).

That service swallows and counts every failure, and nothing upstream branches on whether a
commit happened, so a committer that silently stopped working looks exactly like a healthy one
from the outside. These tests are the only thing that would notice. They pin the shape of the
history (one commit per enqueued scope, subject `chore(memory): <reason> <scope>`, a clean
working tree afterwards), the status guard that makes an unchanged scope a no-op instead of a
`git add` that exits 128, the two ways the service turns itself off (no repository when `start`
runs, then a run of failures), the drop of anything enqueued before `start`, and the committer
identity riding on every invocation.

The clear test is the odd one out: it asserts the erased content is STILL readable from the
previous commit. That is the recorded decision on #408 for a private, unpushed, single-operator
store, so it is pinned as behavior rather than left as a comment someone could helpfully "fix".

Every test builds its own repository under `tmp_path` and points the store there, so the live
`data/memories` repository is never a target — the same rule `media_cleanup` follows for its
serve dir. The service is fire-and-forget behind one worker, so the assertions wait on the
repository through `_wait_for` rather than on the queue.
"""

import shutil
import asyncio
from pathlib import Path
import subprocess
from collections.abc import Callable

import pytest

from discordbot.services.memory.store import user_scope
from discordbot.services.memory.git_history import MemoryGitService


def _git(repository: Path, *args: str) -> str:
    """Runs one git command in a scratch repository.

    `check=True` because every call here is setup or an assertion: a non-zero exit is a broken
    test, never something a caller branches on.

    Returns:
        The command's stdout.
    """
    return subprocess.run(  # noqa: S603 -- fixed argv against a tmp_path repository
        ["git", "-C", str(repository), *args],  # noqa: S607 -- git is resolved from PATH, like every other dev tool here
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def memory_repository(memory_isolated_dir: Path) -> Path:
    """Initializes the isolated memory dir as a git repository with one commit.

    The baseline commit gives every test a HEAD to diff against, and git refuses to make one
    with nothing staged, which is what the `.gitignore` is for; ignoring `*.tmp` also matches
    the `.md.tmp` siblings the store's atomic writes leave behind mid-write. The local
    `user.email` / `user.name` stand in for the gitconfig a developer's machine has and a
    container does not — `test_commits_carry_their_own_committer_identity` strips them again.

    Returns:
        The repository root, which is also the store root the service commits into.
    """
    memory_isolated_dir.mkdir(parents=True, exist_ok=True)
    _git(memory_isolated_dir, "init", "-q", "-b", "main")
    _git(memory_isolated_dir, "config", "user.email", "test@example.com")
    _git(memory_isolated_dir, "config", "user.name", "test")
    (memory_isolated_dir / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    _git(memory_isolated_dir, "add", "-A")
    _git(memory_isolated_dir, "commit", "-q", "-m", "baseline")
    return memory_isolated_dir


async def _wait_for(check: Callable[[], bool]) -> None:
    """Waits for the single worker to produce an observable outcome.

    Polls the outcome rather than the queue: the queue empties when the worker picks a
    request up, not when it has finished the two subprocesses that request runs.

    Raises:
        AssertionError: The outcome never appeared within roughly four seconds, which for git
            commands against a `tmp_path` repository means the worker never got there.
    """
    for _ in range(200):
        if check():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the git worker never reached the expected state")


async def test_a_scope_change_becomes_one_commit(memory_repository: Path) -> None:
    """A consolidation's files reach history under a message naming what happened."""
    scope = user_scope(user_id=111)
    (memory_repository / scope / "global").mkdir(parents=True)
    (memory_repository / scope / "global" / "a.md").write_text("fact", encoding="utf-8")
    service = MemoryGitService()
    service.start()
    service.enqueue(scope=scope, reason="update")
    await _wait_for(check=lambda: "update 111" in _git(memory_repository, "log", "--format=%s"))
    await service.stop()
    assert "chore(memory): update 111" in _git(memory_repository, "log", "--format=%s")
    assert _git(memory_repository, "status", "--porcelain") == ""


async def test_an_unchanged_scope_makes_no_commit(memory_repository: Path) -> None:
    """A scope with nothing on disk commits nothing and leaves the service enabled.

    The status guard is required rather than an optimization: `git add` on a never-tracked,
    now-absent path exits 128, and an empty commit would fail too, so without the guard this
    request would count as a failure and walk the service towards disabling itself.
    """
    service = MemoryGitService()
    service.start()
    service.enqueue(scope=user_scope(user_id=999), reason="update")
    # Nothing observable happens, so give the worker real time to do the wrong thing.
    await asyncio.sleep(0.3)
    await service.stop()
    assert _git(memory_repository, "log", "--format=%s").strip() == "baseline"
    assert service.enabled


async def test_a_clear_commits_the_deletion(memory_repository: Path) -> None:
    """A clear leaves a clean working tree — and, deliberately, the earlier commits."""
    scope = user_scope(user_id=111)
    (memory_repository / scope / "global").mkdir(parents=True)
    (memory_repository / scope / "global" / "a.md").write_text("secret", encoding="utf-8")
    service = MemoryGitService()
    service.start()
    service.enqueue(scope=scope, reason="update")
    await _wait_for(check=lambda: "update 111" in _git(memory_repository, "log", "--format=%s"))
    (memory_repository / scope / "global" / "a.md").unlink()
    service.enqueue(scope=scope, reason="clear")
    await _wait_for(check=lambda: "clear 111" in _git(memory_repository, "log", "--format=%s"))
    await service.stop()
    assert "chore(memory): clear 111" in _git(memory_repository, "log", "--format=%s")
    assert _git(memory_repository, "status", "--porcelain") == ""
    # Recorded, not accidental: the content is still reachable from the earlier commit,
    # which is why the clear cannot promise history is gone (#408).
    assert "secret" in _git(memory_repository, "show", "HEAD~1:111/global/a.md")


async def test_a_store_that_is_not_a_repository_disables_the_service(
    memory_isolated_dir: Path,
) -> None:
    """`start` against a store with no `.git` turns the service off for the whole process.

    The bot never runs `git init`; an absent repository is simply not a deployment that wanted
    history, so it is left alone instead of being rechecked on every reconnect.
    """
    memory_isolated_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 -- one mkdir under tmp_path, not blocking IO
    service = MemoryGitService()
    service.start()
    assert not service.enabled
    # An enqueue after that is a silent no-op rather than an unbounded queue.
    service.enqueue(scope=user_scope(user_id=111), reason="update")
    await service.stop()


async def test_an_unstarted_service_drops_requests(memory_isolated_dir: Path) -> None:
    """Enqueueing before `start` is a silent drop, not a queue bound to the calling loop."""
    service = MemoryGitService()
    service.enqueue(scope=user_scope(user_id=111), reason="update")
    await service.stop()


async def test_repeated_failures_disable_the_service(memory_repository: Path) -> None:
    """A permanently broken repository stops being retried instead of filling the log."""
    scope = user_scope(user_id=111)
    (memory_repository / scope / "global").mkdir(parents=True)
    (memory_repository / scope / "global" / "a.md").write_text("fact", encoding="utf-8")
    service = MemoryGitService()
    service.start()
    # Removing the repository after start makes every git invocation fail without
    # touching the stored files, which is the shape of a genuinely broken deployment.
    shutil.rmtree(memory_repository / ".git")
    for _ in range(5):
        service.enqueue(scope=scope, reason="update")
    await _wait_for(check=lambda: not service.enabled)
    await service.stop()
    assert not service.enabled


async def test_commits_carry_their_own_committer_identity(memory_repository: Path) -> None:
    """A repository carrying no configured ident still gets commits, authored by the bot.

    The container runs as a user with no gitconfig, and a container hostname has no domain, so
    git refuses its own auto-detected ident and every commit exits 128. The identity has to
    ride on the invocation; an operator's host-side `git init` cannot supply it.
    """
    scope = user_scope(user_id=111)
    (memory_repository / scope / "global").mkdir(parents=True)
    (memory_repository / scope / "global" / "a.md").write_text("fact", encoding="utf-8")
    # Strip every fallback the fixture set up, leaving the repository exactly as a fresh
    # `git init` inside the image would: no user.name, no user.email.
    _git(memory_repository, "config", "--unset", "user.email")
    _git(memory_repository, "config", "--unset", "user.name")
    service = MemoryGitService()
    service.start()
    service.enqueue(scope=scope, reason="update")
    await _wait_for(check=lambda: "update 111" in _git(memory_repository, "log", "--format=%s"))
    await service.stop()
    assert service.enabled
    assert "discordbot" in _git(memory_repository, "log", "-1", "--format=%an")
