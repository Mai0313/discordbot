"""Best-effort git history for the memory store.

``data/memories`` is a local git repository (never pushed; the parent repo ignores
``/data``), so a consolidation that goes wrong can be read back and compared. The bot
never creates it: an operator runs ``git init`` once, and a missing repository disables
this quietly rather than filling an ignored directory with one nobody asked for.

Three properties are what make committing from inside a running bot safe:

* **One worker.** Every invocation goes through a single queue and one process-wide
  lock. ``MEMORY_GLOBAL_CONCURRENCY`` is 24, and ``git commit`` takes ``.git/index.lock``,
  so unserialised commits would start failing exactly when the store is busiest.
* **Under the scope lock.** A delta batch is N renames, not one atomic replace, so a
  commit taken mid-batch would record a tree that never existed. The worker takes the
  same ``scope_lock`` the writer used; it is background work, so waiting costs nothing.
* **Never load-bearing.** Every failure is swallowed and counted, and a run of them
  disables the service for the rest of the process. Nothing upstream branches on whether
  a commit happened.

Note what this does NOT do: a ``/memory clear`` commits the deletion, but every earlier
commit still holds the content, and ``gc`` only drops *unreachable* objects. Local
history therefore outlives a clear. That is a deliberate, recorded decision on #408, not
an oversight — the store is a private, unpushed, single-operator repository.

The bot holds exactly one service, ``memory_git`` at the bottom of this module, mirroring
the one repository it commits to. ``gen_reply``'s ``on_ready`` calls ``start`` on it so the
queue binds to the gateway's loop, and ``services/memory/pipeline.py`` is the only caller of
``enqueue`` — once per applied consolidation, per regeneration, and per clear. Tests build
their own service against a ``tmp_path`` repository; nothing in the running bot should.
"""

import asyncio
import contextlib

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.memory import MemoryConfig
from discordbot.utils.asyncio_locks import LoopLocalLock
from discordbot.services.memory.store import scope_lock, memory_root

# Consecutive failures before the service stops trying. A repository that is missing,
# locked by an operator, or out of disk fails every time, and a background task that
# retries forever just fills the log.
_MAX_CONSECUTIVE_FAILURES = 5

# Committer identity for the store's own history. Passed per invocation rather than
# written into the repository so the bot never edits an operator's config, and so a
# repository created by hand needs no setup beyond `git init`.
_COMMITTER_NAME = "discordbot"
_COMMITTER_EMAIL = "discordbot@localhost"


class _GitRequest(BaseModel):
    """One queued commit of a single scope's directory."""

    model_config = ConfigDict(frozen=True)

    scope: str = Field(..., description="Scope directory to stage, relative to the store root.")
    reason: str = Field(
        ..., description="What changed, for the commit subject.", examples=["update"]
    )


class MemoryGitService(BaseModel):
    """Serialized, best-effort committer for the memory store."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = Field(default=True, description="Whether commits are attempted at all.")

    def __init__(self, **data: object) -> None:
        """Initializes the service with no worker; `start` binds it to a loop.

        Nothing loop-bound is built here, because the process-wide singleton is constructed at
        import time when there is no running loop for a queue to attach to.

        Args:
            **data (object): Field values forwarded to pydantic, i.e. `enabled`.
        """
        super().__init__(**data)
        self._queue: asyncio.Queue[_GitRequest] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._failures = 0
        self._lock = LoopLocalLock()

    def start(self) -> None:
        """Starts the single worker, if git history is enabled and a repository exists.

        Called from a cog's `on_ready`, so the queue is created on the running loop.
        Deliberately not lazy: an unstarted service drops every request instead of
        binding a queue to whichever loop happened to enqueue first. Idempotent, since
        `on_ready` fires again on every reconnect. A store that is not a repository
        disables the service for the rest of the process rather than being rechecked.
        """
        if not self.enabled or self._worker is not None:
            return
        if not (memory_root() / ".git").is_dir():
            logfire.info("Memory git history disabled: the store is not a git repository")
            self.enabled = False
            return
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancels the worker, leaving any queued commits undone."""
        worker = self._worker
        self._worker = None
        self._queue = None
        if worker is None:
            return
        worker.cancel()
        # Expected: shutdown cancels the worker, and an uncommitted change is picked up
        # by the next commit of the same scope.
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    def enqueue(self, scope: str, reason: str) -> None:
        """Requests a commit of one scope. Never blocks, never raises, never awaits.

        A request made before `start`, or after the service disabled itself, is dropped
        silently, so no caller has to know whether history is on in this deployment.

        Args:
            scope (str): Scope directory to stage, relative to the store root.
            reason (str): What changed, used as the commit subject.
        """
        queue = self._queue
        if queue is None or not self.enabled:
            return
        queue.put_nowait(_GitRequest(scope=scope, reason=reason))

    async def _run(self) -> None:
        """Drains the queue one request at a time for as long as the service is enabled.

        A request already queued when the service disables itself is taken off and dropped, so
        a broken deployment stops committing without leaving work parked behind it.
        """
        queue = self._queue
        if queue is None:
            return
        while True:
            request = await queue.get()
            if not self.enabled:
                continue
            await self._commit(request=request)

    async def _commit(self, request: _GitRequest) -> None:
        """Stages and commits one scope, swallowing and counting any failure.

        Takes the process-wide lock and then that scope's write lock, in that order and for the
        reasons in the module docstring; waiting on either costs nothing, since this is
        background work. A run of `_MAX_CONSECUTIVE_FAILURES` failures disables the service for
        the rest of the process, and any success resets the count.

        Args:
            request (_GitRequest): The scope to stage and the reason for its commit subject.
        """
        try:
            async with self._lock.get(), scope_lock(scope=request.scope):
                if not await self._has_changes(scope=request.scope):
                    # `git add` on a path that was never tracked and no longer exists
                    # exits 128, so this guard is required rather than an optimization.
                    return
                await self._git("add", "-A", "--", request.scope)
                await self._git("commit", "-m", f"chore(memory): {request.reason} {request.scope}")
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
            if self._failures >= _MAX_CONSECUTIVE_FAILURES:
                logfire.warn("Memory git history disabled after repeated failures")
                self.enabled = False
            return
        self._failures = 0

    async def _has_changes(self, scope: str) -> bool:
        """Whether the scope's directory differs from HEAD.

        Args:
            scope (str): Scope directory to inspect, relative to the store root.

        Returns:
            True when `git status` reports anything under it, tracked or not.
        """
        return bool(await self._git("status", "--porcelain", "--", scope))

    async def _git(self, *args: str) -> str:
        """Runs one git command in the store, returning stdout.

        Args:
            *args (str): The subcommand and its arguments, appended after the flags every
                invocation carries.

        Returns:
            The command's stdout, decoded with undecodable bytes replaced.

        Raises:
            RuntimeError: The command exited non-zero. The caller turns every failure
                into a counted warning, so the message only has to be readable.
        """
        root = memory_root()
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(root),
            # The store is owned by the bot process but may be bind-mounted from a host
            # with a different uid, which git otherwise refuses as dubious ownership.
            "-c",
            f"safe.directory={root.resolve()}",
            # The container runs as an unprivileged user with no gitconfig, and a
            # container hostname carries no domain, so git rejects its own auto-detected
            # `app@<id>.(none)` ident and every commit exits 128. An operator's host-side
            # `git init` cannot supply this: it writes no identity, and a global config
            # on the host is not visible inside the image.
            "-c",
            f"user.name={_COMMITTER_NAME}",
            "-c",
            f"user.email={_COMMITTER_EMAIL}",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} exited {process.returncode}: "
                f"{stderr.decode(errors='replace').strip()}"
            )
        return stdout.decode(errors="replace")


# One service per process, mirroring the single repository it commits to.
memory_git = MemoryGitService(enabled=MemoryConfig().git_history_enabled)
