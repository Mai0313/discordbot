"""Best-effort local git history for `data/memories`.

An unpushed local repository initialized out of band by the operator (`git init`); a missing
`.git` directory silently disables the service.

* **Single worker:** one queue and task serialize all `git` invocations against `.git/index.lock`.
* **Scope-locked:** commits acquire `scope_lock` so multi-file delta batches commit atomically.
* **Best-effort:** failures are swallowed and counted; repeated failures disable the service for
  the rest of the process.

A `/memory clear` commits the file deletion while earlier git history remains intact.
"""

import asyncio
import contextlib

import logfire
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr

from discordbot.typings.config import load_bot_config
from discordbot.typings.timeouts import YTDLP_SOCKET_TIMEOUT_SECONDS
from discordbot.services.memory.store import scope_lock, memory_root

# Reuses a 30s bound from `typings/timeouts.py` so a hung git subprocess cannot deadlock scope_lock.
_GIT_COMMAND_TIMEOUT_SECONDS: float = YTDLP_SOCKET_TIMEOUT_SECONDS


class _GitRequest(BaseModel):
    """One queued commit of a single scope's directory."""

    model_config = ConfigDict(frozen=True)

    scope: str = Field(..., description="Scope directory to stage, relative to the store root.")
    reason: str = Field(
        ..., description="What changed, for the commit subject.", examples=["update"]
    )


class MemoryGitService(BaseModel):
    """Serialized, best-effort committer for the memory store.

    Call `start()` inside a running event loop to bind the queue and worker task; `enqueue()`
    also starts the worker lazily when called on a running loop. Rebinds cleanly when the active
    event loop changes across tests or restarts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = Field(default=True, description="Whether commits are attempted at all.")

    _queue: asyncio.Queue[_GitRequest] | None = PrivateAttr(default=None)
    _worker: asyncio.Task[None] | None = PrivateAttr(default=None)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)
    _active_process: asyncio.subprocess.Process | None = PrivateAttr(default=None)
    _stopping: bool = PrivateAttr(default=False)
    _failures: int = PrivateAttr(default=0)

    def _reset_stale_loop_state(self, loop: asyncio.AbstractEventLoop) -> None:
        """Drops queue and worker handles left over from a previous event loop."""
        if self._loop is not None and self._loop is not loop:
            self._queue = None
            self._worker = None
            self._loop = None
            self._active_process = None
            self._stopping = False
            self._failures = 0

    @property
    def is_running(self) -> bool:
        """Returns whether the git worker is currently active on the running event loop."""
        if self._worker is None or self._worker.done():
            return False
        try:
            return self._loop is asyncio.get_running_loop()
        except RuntimeError:
            return False

    def start(self) -> None:
        """Starts the single worker if git history is enabled and a repository exists.

        Idempotent on the active event loop; evaluates `[global.memory].git_enabled` here
        rather than at import time and rebinds cleanly if the event loop changed.
        """
        if not self.enabled or self._stopping:
            return
        loop = asyncio.get_running_loop()
        self._reset_stale_loop_state(loop=loop)
        if self._worker is not None and not self._worker.done():
            return
        if not load_bot_config().global_config.memory.git_enabled:
            logfire.info("Memory git history disabled by configuration")
            self.enabled = False
            return
        if not (memory_root() / ".git").is_dir():
            logfire.info("Memory git history disabled: the store is not a git repository")
            self.enabled = False
            return
        self._failures = 0
        self._queue = asyncio.Queue()
        self._loop = loop
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancels the worker and terminates any live git subprocess, leaving queued commits undone."""
        self._stopping = True
        worker = self._worker
        worker_loop = self._loop
        process = self._active_process
        self._worker = None
        self._queue = None
        self._loop = None
        self._active_process = None
        try:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            same_loop = worker_loop is current_loop and current_loop is not None
            if worker is not None and same_loop:
                worker.cancel()
                # Shutdown cancels the worker; any uncommitted change folds into the next commit.
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
            if process is not None and process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                if same_loop:
                    await process.wait()
        finally:
            self._stopping = False

    def enqueue(self, scope: str, reason: str) -> None:
        """Queues a commit of `scope` without blocking, raising, or awaiting.

        Lazily starts the worker when a loop is running; drops requests when disabled, stopping,
        or outside any running event loop.

        Args:
            scope (str): Scope directory to stage, relative to the store root.
            reason (str): Commit subject tag (for example, `'update'` or `'clear'`).
        """
        if not self.enabled or self._stopping:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._reset_stale_loop_state(loop=loop)
        if self._worker is None or self._worker.done():
            self.start()
        queue = self._queue
        if queue is None or not self.enabled:
            return
        queue.put_nowait(_GitRequest(scope=scope, reason=reason))

    async def _run(self) -> None:
        """Drains the queue one request at a time, stopping once disabled."""
        queue = self._queue
        if queue is None:
            return
        while self.enabled:
            request = await queue.get()
            if not self.enabled:
                return
            await self._commit(request=request)

    async def _commit(self, request: _GitRequest) -> None:
        """Stages and commits one scope, swallowing and counting any failure."""
        try:
            async with scope_lock(scope=request.scope):
                if not await self._has_changes(scope=request.scope):
                    # Guard required: `git add` on an untracked, deleted path exits 128.
                    return
                await self._git("add", "-A", "--", request.scope)
                await self._git(
                    "commit",
                    "-m",
                    f"chore(memory): {request.reason} {request.scope}",
                    "--",
                    request.scope,
                )
        except Exception as error:
            # Broad on purpose: this is a background best-effort path, and every git
            # failure mode (missing binary, index lock, full disk, hook rejection)
            # arrives as a different exception with the same correct response.
            self._failures += 1
            logfire.warn(
                "Memory git commit failed",
                scope=request.scope,
                failures=self._failures,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            # Disable after consecutive failures so a broken or locked repo does not spam logs.
            max_consecutive_failures = 5
            if self._failures >= max_consecutive_failures:
                logfire.warn("Memory git history disabled after repeated failures")
                self.enabled = False
            return
        self._failures = 0

    async def _has_changes(self, scope: str) -> bool:
        """Whether the scope's directory differs from HEAD."""
        return bool(await self._git("status", "--porcelain", "--", scope))

    async def _git(self, *args: str) -> str:
        """Runs one git command in the store and returns stdout, raising `RuntimeError` on non-zero exit."""
        root = memory_root()
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(root),
            # Avoid git dubious-ownership errors on host bind mounts with a different uid.
            "-c",
            f"safe.directory={root.resolve()}",
            # Container has no gitconfig and a domainless hostname (git commit exits 128);
            # pass identity per invocation so an operator's repo needs only `git init`.
            "-c",
            "user.name=discordbot",
            "-c",
            "user.email=discordbot@localhost",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._active_process = process
        try:
            async with asyncio.timeout(delay=_GIT_COMMAND_TIMEOUT_SECONDS):
                stdout, stderr = await process.communicate()
        except BaseException:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
            raise
        finally:
            if self._active_process is process:
                self._active_process = None
        if process.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} exited {process.returncode}: "
                f"{stderr.decode(errors='replace').strip()}"
            )
        return stdout.decode(errors="replace")


# Process-wide singleton mirroring the single store repository; config is evaluated at `start()`.
memory_git = MemoryGitService()
