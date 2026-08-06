"""Loop-local asyncio primitives: a lock, a semaphore, a registry, and a per-key lock table.

An `asyncio.Lock` or `Semaphore` binds to the loop of the first call that actually waits on it,
not to the loop that constructed it, and anywhere else afterwards it raises `is bound to a
different event loop`. That is what makes a module-level primitive, or one living on a
process-wide singleton, unusable under the test suite: every test runs on its own fresh loop, so
the second test to reach the primitive inherits the first one's dead loop. Constructing it
lazily does not help, since the binding happens at the first wait rather than at construction.

Each type here holds the primitive alongside the loop it was built for, and every accessor that
hands one out compares that loop against `asyncio.get_running_loop()` first, rebuilding on a
change. The rebuild DISCARDS the previous loop's state instead of migrating it: a registry is
emptied and a lock is replaced unheld, which is only sound because the old loop is no longer
running and none of its tasks can still be inside. So these carry state whose life is one
loop's; anything that must outlive a loop change (a monotonic timestamp, a payload cache) stays
an ordinary module-level dict, and `services/memory/store.py` keeps both kinds side by side.

What they deliberately do not promise: thread safety (an asyncio primitive is not thread-safe
either), any cross-loop continuity, and any usefulness outside a running loop — every accessor
that hands out a primitive or an entry needs one. `KeyedLockManager.is_empty` is the single
exception: it reads its maps without rebinding, so it answers with no loop running at all, and
right after a loop change it answers about the stale loop. Hold ONE instance per call site,
normally module-level beside the engine or cache it guards, and take the primitive fresh at each
use rather than caching what an accessor returned.

It sits in `utils/` because every layer above needs it — `services/` (economy, stock, memory),
several cog databases, and `utils/douyin.py` — and `services/` may never reach into a cog for it.
"""

import asyncio
from contextlib import asynccontextmanager
from collections.abc import Callable, AsyncIterator

from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation


class LoopLocalLock(BaseModel):
    """An asyncio.Lock rebuilt whenever the running event loop changes.

    Guards a process-wide critical section that lives for one loop: schema creation on each
    database module, loan acceptance, stock news generation, the memory git worker.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _lock: asyncio.Lock | None = PrivateAttr(default=None)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    def get(self) -> asyncio.Lock:
        """Returns the lock bound to the current event loop, rebuilding it on a loop change.

        Within one loop every call returns the SAME lock object, which is what makes the guarded
        section mutually exclusive at all; only a loop change mints a new one. That replacement
        is unheld, so a rebuild silently drops whatever the stale loop was serializing. Take the
        lock through this accessor at each use: one cached across a loop change is exactly the
        binding error the class exists to avoid.

        Returns:
            The lock belonging to the running loop, the same instance as the previous call on it.
        """
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
        return self._lock


class LoopLocalSemaphore(BaseModel):
    """An asyncio.Semaphore rebuilt whenever the running event loop changes.

    Caps concurrent outbound work that must stay bounded process-wide: memory pipeline jobs,
    Douyin and Bilibili fetches, link-media uploads, the stock news provider.

    The capacity is read from `capacity_provider` each time the semaphore is (re)built, not at
    construction, so a test that monkeypatches the cap constant before the first use still takes
    effect.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    capacity_provider: SkipValidation[Callable[[], int]] = Field(
        ...,
        description="Returns the concurrency cap, read fresh each time the semaphore is rebuilt.",
    )
    _semaphore: asyncio.Semaphore | None = PrivateAttr(default=None)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    def get(self) -> asyncio.Semaphore:
        """Returns the semaphore bound to the current event loop, rebuilding on a loop change.

        Within one loop every call returns the SAME semaphore, so its callers share one permit
        pool rather than each taking a private cap. A rebuild re-reads the capacity and starts at
        full permits, so anything still holding a permit from the stale loop is forgotten rather
        than counted.

        Returns:
            The semaphore belonging to the running loop, sized by `capacity_provider` and the
            same instance as the previous call on that loop.
        """
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self.capacity_provider())
            self._loop = loop
        return self._semaphore


class LoopLocalRegistry[K, V](BaseModel):
    """A process-local dict rebuilt (cleared) whenever the running event loop changes.

    Backs per-scope lock tables (`services/memory/store.py::scope_lock`) and per-scope task slots
    (the memory pipeline's in-flight regenerations) — values that are themselves loop-bound and
    so must not survive their loop. Every accessor rebinds first, so a stale loop's entries are
    never handed out; they are dropped, not cancelled or closed, since the loop that could have
    run them is gone. Anything that has to persist across loops belongs in a plain dict instead.

    It never grows a lock of its own: a rebind and a dict operation both run without awaiting, so
    two tasks on one loop cannot interleave inside them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _items: dict[K, V] = PrivateAttr(default_factory=dict)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    def _bind(self) -> dict[K, V]:
        """Returns the current loop's dict, clearing it when the loop changed.

        Every public accessor goes through here, which is what makes the drop total: there is no
        path that reads an entry belonging to a loop that is no longer running.

        Returns:
            The dict holding the running loop's entries.
        """
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._items = {}
            self._loop = loop
        return self._items

    def setdefault(self, key: K, default: V) -> V:
        """Returns the value for `key`, inserting `default` when absent.

        `default` is built by the caller before the lookup, so an already-present key discards a
        freshly constructed value; keep it cheap (an unheld `asyncio.Lock`, a bare slot).

        Args:
            key (K): The entry to look up.
            default (V): Stored, and returned, only when the key is absent.

        Returns:
            The value now held under `key`, existing or just inserted.
        """
        return self._bind().setdefault(key, default)

    def get(self, key: K) -> V | None:
        """Returns the value for `key`, or None when absent.

        Args:
            key (K): The entry to look up.

        Returns:
            The stored value, or None when the key is absent or was left by a stale loop.
        """
        return self._bind().get(key)

    def set(self, key: K, value: V) -> None:
        """Stores `value` under `key`, replacing whatever was there.

        Args:
            key (K): The entry to write.
            value (V): The value to store.
        """
        self._bind()[key] = value

    def pop(self, key: K) -> V | None:
        """Removes `key` and returns what it held.

        An absent key is not an error, so a caller clearing a slot it may never have filled (a
        finished task removing itself) needs no guard.

        Args:
            key (K): The entry to remove.

        Returns:
            The removed value, or None when the key was absent.
        """
        return self._bind().pop(key, None)

    def snapshot(self) -> dict[K, V]:
        """Returns a shallow copy of the current loop's entries.

        Not a production accessor. It is here for tests and debugging, and has no call site in
        `src/` or `tests/` today. The copy is shallow: the values are the live ones, not copies
        of their own.

        Returns:
            A new dict holding the running loop's entries.
        """
        return dict(self._bind())


class KeyedLockManager[K](BaseModel):
    """Refcounted per-key asyncio locks, rebuilt when the running event loop changes.

    Serializes work per key — one user's operations on one stock (a `(user_id, symbol)` tuple,
    so the same user's two symbols do NOT wait on each other), one symbol's market advance, one
    angler, one Douyin URL, one memory scope's staging writes — while keeping the maps bounded
    against a key space that is effectively unbounded. A key's lock and refcount are dropped
    once the last holder leaves, so an idle key leaves no residue; `is_empty` is that invariant,
    asserted by `tests/test_stock.py` after concurrent trades.

    Correctness rests on the refcount counting waiters, not just the holder: see `hold`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _locks: dict[K, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _refcounts: dict[K, int] = PrivateAttr(default_factory=dict)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    def _bind(self) -> None:
        """Clears the per-key maps when the running loop changed.

        Runs at the top of `hold` and nowhere else, before the key's lock or refcount is touched,
        so the two maps are always cleared together and can never disagree about a key.
        """
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._locks = {}
            self._refcounts = {}
            self._loop = loop

    @asynccontextmanager
    async def hold(self, key: K) -> AsyncIterator[None]:
        """Holds the per-key lock for the duration of the context, refcounting the key.

        The refcount is raised BEFORE the lock is awaited, so it counts waiters as well as the
        holder: a caller queued behind the holder keeps the entry alive, and the release path
        cannot drop a lock somebody is still blocked on and hand the next arrival a fresh,
        unheld one. Lock and count are removed together once the last of them leaves, which is
        what `is_empty` reports.

        Not reentrant: re-entering the same key from inside the context deadlocks.

        Args:
            key (K): The key whose work is serialized.

        Yields:
            None, with the key's lock held for the body of the context.
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

    @property
    def is_empty(self) -> bool:
        """Whether no per-key lock or refcount remains (no held or pending keys).

        True is the resting state, so an entry left behind after everything settled means some
        `hold` never reached its `finally`. Read without rebinding, unlike every other path here,
        so right after a loop change it still reports the stale loop's leftovers until the next
        `hold` clears them.

        Returns:
            True when no key is held or waited on.
        """
        return not self._locks and not self._refcounts
