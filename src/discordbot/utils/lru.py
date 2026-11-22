"""Bounded least-recently-used in-memory cache."""

from collections import OrderedDict

from pydantic import Field, BaseModel, PrivateAttr


class BoundedLRU[K, V](BaseModel):
    """Fixed-capacity LRU cache evicting the least recently used entry past `max_entries`.

    Not thread-safe on its own; callers sharing an instance across worker threads guard access
    with their own lock.
    """

    max_entries: int = Field(
        default=128,
        ge=1,
        description="Maximum number of entries retained before evicting the least recently used.",
    )
    _items: OrderedDict[K, V] = PrivateAttr(default_factory=OrderedDict)

    def get(self, key: K) -> V | None:
        """Returns the value for `key` and marks it most recently used, or None when absent.

        Args:
            key (K): Entry key to look up.
        """
        if key not in self._items:
            return None
        self._items.move_to_end(key=key)
        return self._items[key]

    def peek(self, key: K) -> V | None:
        """Returns the value for `key` without updating recency, or None when absent.

        Args:
            key (K): Entry key to look up.
        """
        return self._items.get(key)

    def touch(self, key: K) -> None:
        """Marks `key` as most recently used when present.

        Args:
            key (K): Entry key to refresh.
        """
        if key in self._items:
            self._items.move_to_end(key=key)

    def set(self, key: K, value: V) -> None:
        """Stores `value` under `key`, marking it most recently used and evicting past `max_entries`.

        Args:
            key (K): Entry key to insert or overwrite.
            value (V): Value to store.
        """
        self._items[key] = value
        self._items.move_to_end(key=key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def pop(self, key: K) -> V | None:
        """Removes and returns `key`'s value, or None when absent.

        Args:
            key (K): Entry key to remove.
        """
        return self._items.pop(key, None)

    def clear(self) -> None:
        """Removes every entry from the cache."""
        self._items.clear()

    def __contains__(self, key: K) -> bool:
        """Returns whether `key` is stored without updating recency."""
        return key in self._items

    def __len__(self) -> int:
        """Returns the number of entries currently stored."""
        return len(self._items)
