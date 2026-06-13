"""A loop-local asyncio lock that rebinds when the running event loop changes."""

import asyncio

from pydantic import BaseModel, ConfigDict, PrivateAttr


class LoopLocalLock(BaseModel):
    """An asyncio.Lock rebuilt whenever the running event loop changes.

    Module-level locks must rebind per loop: the test suite runs each case on a fresh
    event loop (and swaps the module-level engines tests monkeypatch), so a lock bound to
    a closed loop would be unusable. Hold one instance per call site and call `get()` to
    obtain the lock bound to the current loop; it rebuilds the lock on a loop change and
    otherwise returns the same instance.
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
