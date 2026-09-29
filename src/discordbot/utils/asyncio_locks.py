"""Loop-local asyncio primitives that rebind when the running event loop changes.

An `asyncio.Lock` / `Semaphore` binds to the loop of the first call that actually waits on
it rather than to the loop it was built on, so constructing one lazily does not help: the
next loop to reach it raises `is bound to a different event loop`. Every test case runs on
a fresh loop, so a primitive held at module level or on a process-wide singleton hits that
on the second test. Hold one instance per call site; each accessor rebinds to the current
loop, rebuilding (or clearing) state bound to a stale loop.

`spawn_tracked` starts a fire-and-forget task and holds it until it finishes.
"""

from typing import Any
import asyncio
from functools import partial
from contextlib import asynccontextmanager
from collections.abc import Callable, Coroutine, AsyncIterator

import logfire
from pydantic import Field, BaseModel, PrivateAttr


class LoopLocalLock(BaseModel):
    """An asyncio.Lock rebuilt whenever the running event loop changes."""

    _lock: asyncio.Lock | None = PrivateAttr(default=None)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    def get(self) -> asyncio.Lock:
        """Returns the lock bound to the current event loop, rebuilding it on a loop change."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
        return self._lock


class LoopLocalSemaphore(BaseModel):
    """An asyncio.Semaphore rebuilt whenever the running event loop changes.

    The capacity is read from `capacity_provider` each time the semaphore is (re)built,
    not at construction, so a test that monkeypatches the cap constant before the first
    use still takes effect.
    """

    capacity_provider: Callable[[], int] = Field(
        ...,
        description="Returns the concurrency cap, read fresh each time the semaphore is rebuilt.",
    )
    _semaphore: asyncio.Semaphore | None = PrivateAttr(default=None)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    def get(self) -> asyncio.Semaphore:
        """Returns the semaphore bound to the current event loop, rebuilding on a loop change."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(value=self.capacity_provider())
            self._loop = loop
        return self._semaphore


class LoopLocalRegistry[K, V](BaseModel):
    """A process-local dict rebuilt (cleared) whenever the running event loop changes.

    Every access rebinds to the current loop first, dropping entries left over from a
    stale loop.
    """

    _items: dict[K, V] = PrivateAttr(default_factory=dict)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    def _bind(self) -> dict[K, V]:
        """Returns the current loop's dict, clearing it when the loop changed."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._items = {}
            self._loop = loop
        return self._items

    def setdefault(self, key: K, default: V) -> V:
        """Returns the value for `key`, inserting `default` when absent."""
        return self._bind().setdefault(key, default)

    def get(self, key: K) -> V | None:
        """Returns the value for `key`, or None when absent."""
        return self._bind().get(key)

    def set(self, key: K, value: V) -> None:
        """Stores `value` under `key`."""
        self._bind()[key] = value

    def pop(self, key: K) -> V | None:
        """Removes and returns `key`'s value, or None when absent."""
        return self._bind().pop(key, None)


class KeyedLockManager[K](BaseModel):
    """Refcounted per-key asyncio locks, rebuilt when the running event loop changes.

    Serializes work per key while keeping the maps bounded: a key's lock and refcount are
    dropped once the last holder releases, so an idle key leaves no residue.
    """

    _locks: dict[K, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _refcounts: dict[K, int] = PrivateAttr(default_factory=dict)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    def _bind(self) -> None:
        """Clears the per-key maps when the running loop changed."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._locks = {}
            self._refcounts = {}
            self._loop = loop

    @asynccontextmanager
    async def hold(self, key: K) -> AsyncIterator[None]:
        """Holds the per-key lock for the duration of the context, refcounting the key."""
        self._bind()
        lock = self._locks.setdefault(key, asyncio.Lock())
        self._refcounts[key] = self._refcounts.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._refcounts[key] -= 1
            if self._refcounts[key] <= 0:
                self._refcounts.pop(key, None)
                self._locks.pop(key, None)


def spawn_tracked(
    *, coro: Coroutine[Any, Any, None], tasks: set[asyncio.Task[None]], name: str
) -> None:
    """Runs `coro` as a fire-and-forget task, held in `tasks` until it finishes.

    The event loop keeps only a weak reference to a task, so one nothing else holds can be
    garbage-collected mid-flight; `tasks` is the strong reference, and a caller that needs the
    work to have landed can await what it holds. `coro` is expected to report its own failures:
    one that escapes anyway is logged here, since asyncio would print it only to `sys.stderr`,
    which `./data/logs` never sees.

    Args:
        coro: The work to run.
        tasks: The owner's set of running tasks; the task leaves it when done.
        name: The task's name, which the failure log carries.
    """
    task = asyncio.create_task(coro=coro, name=name)
    tasks.add(task)
    task.add_done_callback(partial(_release_tracked, tasks=tasks))


def _release_tracked(task: asyncio.Task[None], *, tasks: set[asyncio.Task[None]]) -> None:
    """Drops a finished task from its owner's set and logs a failure it let escape."""
    tasks.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logfire.error(
            "Background task failed",
            task=task.get_name(),
            error_type=type(error).__name__,
            _exc_info=error,
        )
