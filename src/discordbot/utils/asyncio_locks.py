"""Loop-local asyncio primitives that rebind on event-loop changes, plus tracked task spawning.

Hold one instance per call site; each accessor rebinds to the running loop and rebuilds or
clears state bound to a stale loop.
"""

from typing import Any
import asyncio
from functools import partial
from contextlib import asynccontextmanager
from collections.abc import Callable, Coroutine, AsyncGenerator

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

    Reads the capacity from `capacity_provider` on each build rather than at construction.
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
    """A process-local dict cleared whenever the running event loop changes."""

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
        """Returns the value for `key`, inserting `default` when absent.

        Args:
            key (K): Entry key to look up.
            default (V): Value inserted and returned when `key` is absent.
        """
        return self._bind().setdefault(key, default)

    def get(self, key: K) -> V | None:
        """Returns the value for `key`, or None when absent.

        Args:
            key (K): Entry key to look up.
        """
        return self._bind().get(key)

    def set(self, key: K, value: V) -> None:
        """Stores `value` under `key`.

        Args:
            key (K): Entry key to store under.
            value (V): Value to store.
        """
        self._bind()[key] = value

    def pop(self, key: K) -> V | None:
        """Returns and removes `key`'s value, or None when absent.

        Args:
            key (K): Entry key to remove.
        """
        return self._bind().pop(key, None)


class KeyedLockManager[K](BaseModel):
    """Refcounted per-key asyncio locks, cleared when the running event loop changes.

    Drops a key's lock and refcount once its last holder releases.
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
    async def hold(self, key: K) -> AsyncGenerator[None]:
        """Holds the per-key lock for the duration of the context, refcounting the key.

        Args:
            key (K): Key whose lock to acquire.
        """
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
    coro: Coroutine[Any, Any, None], tasks: set[asyncio.Task[None]], name: str
) -> None:
    """Runs `coro` as a background task, retaining a strong reference in `tasks` until done.

    Logs any unhandled exception escaping `coro` via logfire.

    Args:
        coro (Coroutine[Any, Any, None]): Coroutine to schedule.
        tasks (set[asyncio.Task[None]]): Caller-owned set holding running tasks; the finished
            task is removed automatically.
        name (str): Task name recorded on unhandled-failure logs.
    """
    task = asyncio.create_task(coro=coro, name=name)
    tasks.add(task)
    task.add_done_callback(partial(_release_tracked, tasks=tasks))


def _release_tracked(task: asyncio.Task[None], tasks: set[asyncio.Task[None]]) -> None:
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
