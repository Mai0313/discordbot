"""Loop-local asyncio primitives that rebind when the running event loop changes.

Module-level locks / semaphores / registries must rebind per loop: the test suite runs
each case on a fresh event loop (and swaps the module-level engines tests monkeypatch), so
a primitive bound to a closed loop would be unusable. Hold one instance per call site; each
accessor rebinds to the current loop, rebuilding (or clearing) state bound to a stale loop.
"""

import asyncio
from contextlib import asynccontextmanager
from collections.abc import Callable, AsyncIterator

from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation


class LoopLocalLock(BaseModel):
    """An asyncio.Lock rebuilt whenever the running event loop changes.

    Call `get()` to obtain the lock bound to the current loop; it rebuilds the lock on a
    loop change and otherwise returns the same instance.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    capacity_provider: SkipValidation[Callable[[], int]] = Field(
        description="Returns the concurrency cap, read fresh each time the semaphore is rebuilt."
    )
    _semaphore: asyncio.Semaphore | None = PrivateAttr(default=None)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    def get(self) -> asyncio.Semaphore:
        """Returns the semaphore bound to the current event loop, rebuilding on a loop change."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self.capacity_provider())
            self._loop = loop
        return self._semaphore


class LoopLocalRegistry[K, V](BaseModel):
    """A process-local dict rebuilt (cleared) whenever the running event loop changes.

    Backs per-scope lock tables and per-scope task slots. Every access rebinds to the
    current loop first, dropping entries left over from a stale loop.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    def snapshot(self) -> dict[K, V]:
        """Returns a shallow copy of the current loop's entries (mainly for tests)."""
        return dict(self._bind())


class KeyedLockManager[K](BaseModel):
    """Refcounted per-key asyncio locks, rebuilt when the running event loop changes.

    Serializes work per key (user / symbol) while keeping the maps bounded: a key's lock
    and refcount are dropped once the last holder releases, so an idle key leaves no
    residue (the empty-map invariant some tests assert via `is_empty`).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    @property
    def is_empty(self) -> bool:
        """Whether no per-key lock or refcount remains (no held or pending keys)."""
        return not self._locks and not self._refcounts
