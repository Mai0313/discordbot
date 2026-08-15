"""Tests for the memory store's best-effort git history.

Every test builds its own repository under `tmp_path` and points the store there, so the
live `data/memories` repository is never a target — the same rule `media_cleanup` follows
for its serve dir.
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
    """Runs one git command in a scratch repository, returning stdout."""
    return subprocess.run(  # noqa: S603 -- fixed argv against a tmp_path repository
        ["git", "-C", str(repository), *args],  # noqa: S607 -- git is resolved from PATH, like every other dev tool here
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def memory_repository(memory_isolated_dir: Path) -> Path:
    """Initializes the isolated memory dir as a git repository with one commit."""
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
    request up, not when it has finished the git commands that request runs.
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
    """The status guard is required, not an optimization: `git add` on a never-tracked,
    now-absent path exits 128, and an empty commit would fail too.
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
    """The bot never runs `git init`; an absent repository is simply not a deployment
    that wanted history.
    """
    memory_isolated_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 -- one mkdir under tmp_path, not blocking IO
    service = MemoryGitService()
    service.start()
    assert not service.enabled
    # An enqueue after that is a silent no-op rather than an unbounded queue.
    service.enqueue(scope=user_scope(user_id=111), reason="update")
    await service.stop()


async def test_an_unstarted_service_drops_requests(memory_isolated_dir: Path) -> None:
    """Enqueueing before `start` must not bind a queue to whichever loop got there first."""
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
    """The container runs as a user with no gitconfig, and a container hostname has no
    domain, so git refuses its own auto-detected ident and every commit exits 128. The
    identity has to ride on the invocation; an operator's host-side `git init` cannot
    supply it.
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
