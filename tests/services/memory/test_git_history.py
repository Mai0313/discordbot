"""Tests for the memory store's best-effort git history.

Every test points the store at `memory_isolated_dir`, so the live `data/memories` repository
is never a target — the same rule `media_cleanup` follows for its serve dir.
"""

import shutil
import asyncio
from pathlib import Path
import subprocess

import pytest

from discordbot.typings.config import clear_bot_config_cache
from discordbot.services.memory.store import user_scope
from discordbot.services.memory.git_history import MemoryGitService, _GitRequest

from tests.helpers.logfire_capture import capture_logs


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


@pytest.fixture
def handled(monkeypatch: pytest.MonkeyPatch) -> asyncio.Queue[str]:
    """Receives each request's scope once the worker has finished its git commands.

    The worker's own queue empties when it picks a request up, not when that request's
    commit has landed, so a test waits on this instead.
    """
    finished: asyncio.Queue[str] = asyncio.Queue()
    real_commit = MemoryGitService._commit

    async def commit_then_report(self: MemoryGitService, request: _GitRequest) -> None:
        await real_commit(self, request=request)
        finished.put_nowait(request.scope)

    monkeypatch.setattr(MemoryGitService, "_commit", commit_then_report)
    return finished


async def _wait_for(handled: asyncio.Queue[str], requests: int = 1) -> None:
    """Waits until the worker has finished `requests` more requests."""
    async with asyncio.timeout(delay=30):
        for _ in range(requests):
            await handled.get()


def _subjects(repository: Path) -> list[str]:
    """The repository's commit subjects, newest first."""
    return _git(repository, "log", "--format=%s").splitlines()


async def test_a_scope_change_becomes_one_commit(
    memory_repository: Path, handled: asyncio.Queue[str]
) -> None:
    """A consolidation's files reach history under a message naming what happened."""
    scope = user_scope(user_id=111)
    (memory_repository / scope / "global").mkdir(parents=True)
    (memory_repository / scope / "global" / "a.md").write_text("fact", encoding="utf-8")
    service = MemoryGitService()
    service.start()
    service.enqueue(scope=scope, reason="update")
    await _wait_for(handled=handled)
    await service.stop()
    newest, baseline = _subjects(repository=memory_repository)
    assert baseline == "baseline"
    assert "update" in newest
    assert scope in newest
    assert _git(memory_repository, "status", "--porcelain") == ""


async def test_an_unchanged_scope_makes_no_commit(
    memory_repository: Path, handled: asyncio.Queue[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status guard is required, not an optimization: `git add` on a never-tracked,
    now-absent path exits 128, and an empty commit would fail too.
    """
    warned = capture_logs(monkeypatch=monkeypatch, level="warn")
    changed = user_scope(user_id=111)
    (memory_repository / changed / "global").mkdir(parents=True)
    (memory_repository / changed / "global" / "a.md").write_text("fact", encoding="utf-8")
    service = MemoryGitService()
    service.start()
    service.enqueue(scope=user_scope(user_id=999), reason="update")
    service.enqueue(scope=changed, reason="update")
    await _wait_for(handled=handled, requests=2)
    await service.stop()
    newest, baseline = _subjects(repository=memory_repository)
    assert baseline == "baseline"
    assert changed in newest
    # Skipped by the guard, not by a failed `git add`, which would leave no commit either.
    assert warned == []
    assert service.enabled


async def test_a_clear_commits_the_deletion(
    memory_repository: Path, handled: asyncio.Queue[str]
) -> None:
    """A clear commits the deletion, leaving a clean working tree."""
    scope = user_scope(user_id=111)
    (memory_repository / scope / "global").mkdir(parents=True)
    (memory_repository / scope / "global" / "a.md").write_text("secret", encoding="utf-8")
    service = MemoryGitService()
    service.start()
    service.enqueue(scope=scope, reason="update")
    await _wait_for(handled=handled)
    (memory_repository / scope / "global" / "a.md").unlink()
    service.enqueue(scope=scope, reason="clear")
    await _wait_for(handled=handled)
    await service.stop()
    newest = _subjects(repository=memory_repository)[0]
    assert "clear" in newest
    assert scope in newest
    assert _git(memory_repository, "status", "--porcelain") == ""


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


async def test_enqueue_lazily_starts_when_loop_is_running(
    memory_repository: Path, handled: asyncio.Queue[str]
) -> None:
    """Enqueueing inside a running event loop lazily starts the worker and commits."""
    scope = user_scope(user_id=111)
    (memory_repository / scope / "global").mkdir(parents=True)
    (memory_repository / scope / "global" / "a.md").write_text("fact", encoding="utf-8")
    service = MemoryGitService()
    assert service._worker is None

    service.enqueue(scope=scope, reason="update")
    await _wait_for(handled=handled)
    await service.stop()

    newest, baseline = _subjects(repository=memory_repository)
    assert baseline == "baseline"
    assert scope in newest


def test_enqueue_outside_an_event_loop_drops_requests(memory_repository: Path) -> None:
    """Synchronous enqueueing with no running loop drops the request without binding a queue."""
    del memory_repository
    service = MemoryGitService()
    service.enqueue(scope=user_scope(user_id=111), reason="update")
    assert service._queue is None
    assert service._worker is None


async def test_start_evaluates_config_at_start_not_import(
    memory_repository: Path, bot_config_isolated: Path
) -> None:
    """`[global.memory].git_enabled = false` disables git history when `start()` runs."""
    del memory_repository
    bot_config_isolated.write_text(data="[global.memory]\ngit_enabled = false\n", encoding="utf-8")  # noqa: ASYNC240 -- a tmp_path entry, not blocking IO
    clear_bot_config_cache()
    service = MemoryGitService()
    service.start()
    assert not service.enabled
    assert not service.is_running
    await service.stop()


async def test_start_is_idempotent_on_the_same_loop(memory_repository: Path) -> None:
    """Calling `start()` repeatedly on the same loop keeps the existing worker and queue."""
    del memory_repository
    service = MemoryGitService()
    service.start()
    first_worker = service._worker
    first_queue = service._queue
    service.start()
    assert service._worker is first_worker
    assert service._queue is first_queue
    assert service.is_running
    await service.stop()
    assert not service.is_running


def test_service_rebinds_cleanly_across_event_loops(
    memory_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A service started in an earlier loop cleanly rebinds queue and worker in a new loop."""
    scope_a = user_scope(user_id=111)
    scope_b = user_scope(user_id=222)
    (memory_repository / scope_a / "global").mkdir(parents=True)
    (memory_repository / scope_a / "global" / "a.md").write_text("a", encoding="utf-8")
    (memory_repository / scope_b / "global").mkdir(parents=True)
    (memory_repository / scope_b / "global" / "b.md").write_text("b", encoding="utf-8")
    service = MemoryGitService()
    real_commit = MemoryGitService._commit

    async def run_one(scope: str) -> None:
        done: asyncio.Queue[str] = asyncio.Queue()

        async def commit_then_report(self: MemoryGitService, request: _GitRequest) -> None:
            await real_commit(self, request=request)
            done.put_nowait(request.scope)

        monkeypatch.setattr(MemoryGitService, "_commit", commit_then_report)
        service.start()
        service.enqueue(scope=scope, reason="update")
        await _wait_for(handled=done)

    # Intentionally skip `stop()` on the first loop to exercise stale-loop rebinding too.
    asyncio.run(run_one(scope=scope_a))
    asyncio.run(run_one(scope=scope_b))

    subjects = _subjects(repository=memory_repository)
    assert any(scope_a in subject for subject in subjects)
    assert any(scope_b in subject for subject in subjects)


async def test_repeated_failures_disable_the_service(
    memory_repository: Path, handled: asyncio.Queue[str]
) -> None:
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
        await _wait_for(handled=handled)
        if not service.enabled:
            break
    await service.stop()
    assert not service.enabled


async def test_commits_carry_their_own_committer_identity(
    memory_repository: Path, handled: asyncio.Queue[str]
) -> None:
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
    await _wait_for(handled=handled)
    await service.stop()
    assert service.enabled
    assert "discordbot" in _git(memory_repository, "log", "-1", "--format=%an")


async def test_commit_is_scoped_and_leaves_other_staged_paths_alone(
    memory_repository: Path, handled: asyncio.Queue[str]
) -> None:
    """Committing one scope must not swallow changes left staged in another scope."""
    scope_a = user_scope(user_id=111)
    scope_b = user_scope(user_id=222)
    (memory_repository / scope_a / "global").mkdir(parents=True)
    (memory_repository / scope_a / "global" / "a.md").write_text("fact a", encoding="utf-8")
    (memory_repository / scope_b / "global").mkdir(parents=True)
    (memory_repository / scope_b / "global" / "b.md").write_text("fact b", encoding="utf-8")
    _git(memory_repository, "add", "-A", "--", scope_b)

    service = MemoryGitService()
    service.start()
    service.enqueue(scope=scope_a, reason="update")
    await _wait_for(handled=handled)
    await service.stop()

    committed = _git(memory_repository, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert committed == [f"{scope_a}/global/a.md"]
    assert f"A  {scope_b}/global/b.md" in _git(memory_repository, "status", "--porcelain")


async def test_hung_git_command_times_out_and_kills_child_process(
    memory_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled git subprocess times out, kills the child, and releases the scope lock."""
    monkeypatch.setattr(
        "discordbot.services.memory.git_history._GIT_COMMAND_TIMEOUT_SECONDS", 0.05
    )
    killed = False
    waited = False

    class _HangingProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(5)
            return b"", b""

        def kill(self) -> None:
            nonlocal killed
            killed = True
            self.returncode = -9

        async def wait(self) -> int:
            nonlocal waited
            waited = True
            return -9

    async def spawn_hanging(*args: object, **kwargs: object) -> _HangingProcess:
        del args, kwargs
        return _HangingProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_hanging)
    service = MemoryGitService()

    with pytest.raises(TimeoutError):
        await service._git("status")

    assert killed is True
    assert waited is True


async def test_stop_cancels_worker_and_kills_live_git_subprocess(
    memory_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stop()` terminates an in-flight git subprocess cleanly while cancelling the worker."""
    scope = user_scope(user_id=111)
    (memory_repository / scope / "global").mkdir(parents=True)
    (memory_repository / scope / "global" / "a.md").write_text("fact", encoding="utf-8")
    started = asyncio.Event()
    killed = False
    waited = False

    class _HangingProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            started.set()
            await asyncio.sleep(10)
            return b"", b""

        def kill(self) -> None:
            nonlocal killed
            killed = True
            self.returncode = -9

        async def wait(self) -> int:
            nonlocal waited
            waited = True
            return -9

    async def spawn_hanging(*args: object, **kwargs: object) -> _HangingProcess:
        del args, kwargs
        return _HangingProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_hanging)
    service = MemoryGitService()
    service.start()
    service.enqueue(scope=scope, reason="update")
    await started.wait()

    await service.stop()

    assert killed is True
    assert waited is True
    assert not service.is_running


async def test_enqueue_during_stop_does_not_spawn_a_second_worker(
    memory_repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requests arriving while `stop()` awaits a dying subprocess drop instead of starting a second worker."""
    scope = user_scope(user_id=111)
    (memory_repository / scope / "global").mkdir(parents=True)
    (memory_repository / scope / "global" / "a.md").write_text("fact", encoding="utf-8")
    started = asyncio.Event()
    wait_entered = asyncio.Event()
    release_wait = asyncio.Event()

    class _SlowExitProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            started.set()
            await asyncio.sleep(10)
            return b"", b""

        def kill(self) -> None:
            return None

        async def wait(self) -> int:
            wait_entered.set()
            await release_wait.wait()
            self.returncode = -9
            return -9

    async def spawn_slow(*args: object, **kwargs: object) -> _SlowExitProcess:
        del args, kwargs
        return _SlowExitProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_slow)
    service = MemoryGitService()
    service.start()
    service.enqueue(scope=scope, reason="update")
    await started.wait()

    stop_task = asyncio.create_task(service.stop())
    await wait_entered.wait()
    # While `stop()` is awaiting the killed subprocess, `enqueue()` must not start a second worker.
    service.enqueue(scope=scope, reason="update")
    assert service._worker is None
    assert not service.is_running

    release_wait.set()
    await stop_task
    assert not service._stopping


async def test_stop_kills_tracked_active_process_fallback_without_worker(
    memory_repository: Path,
) -> None:
    """`stop()` still kills and reaps a tracked `_active_process` even if `_worker` is already cleared."""
    del memory_repository
    killed = False
    waited = False

    class _StrandedProcess:
        returncode: int | None = None

        def kill(self) -> None:
            nonlocal killed
            killed = True

        async def wait(self) -> int:
            nonlocal waited
            waited = True
            self.returncode = -9
            return -9

    service = MemoryGitService()
    service._loop = asyncio.get_running_loop()
    service._active_process = _StrandedProcess()  # ty: ignore[invalid-assignment] -- stub subprocess

    await service.stop()

    assert killed is True
    assert waited is True
    assert service._active_process is None
