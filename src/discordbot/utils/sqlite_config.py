"""Shared SQLite connection configuration for the project's DB engines.

Economy, stock, message-log, and cleanup storage all open SQLite with the same
WAL / synchronous / busy_timeout PRAGMA trade-off. This helper centralizes that
setup so every engine configures connections the same way. `StoredInteger`
engines additionally register the integer-aware UDFs.
"""

from typing import Any
import contextlib
from collections.abc import Callable

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine

from discordbot.utils.stored_integer import configure_sqlite_stored_integer_functions


def configure_sqlite_connection(
    dbapi_connection: Any,  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    enable_foreign_keys: bool = False,
    register_stored_integer: bool = True,
) -> None:
    """Applies the project's standard PRAGMA setup to a new SQLite connection.

    WAL flips the read/write lock so readers never block on writes;
    `synchronous=NORMAL` is the right durability trade-off in WAL; a tolerant
    `busy_timeout` gives writers time to wait under contention.

    Args:
        dbapi_connection: The freshly opened DBAPI connection.
        enable_foreign_keys: Whether to turn on `PRAGMA foreign_keys` for the connection.
        register_stored_integer: Whether to register the integer-aware UDFs used by `StoredInteger`.
    """
    with contextlib.closing(dbapi_connection.cursor()) as cursor:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        if enable_foreign_keys:
            cursor.execute("PRAGMA foreign_keys=ON")
    if register_stored_integer:
        configure_sqlite_stored_integer_functions(dbapi_connection=dbapi_connection)


def ensure_sqlite_hooks(
    engine: AsyncEngine, on_connect_fn: Callable[..., None], on_checkout_fn: Callable[..., None]
) -> None:
    """Installs the connect and checkout listeners on an engine exactly once.

    Session factories call this on every open because tests swap the module-level
    engines; `event.contains` keeps repeat calls from stacking duplicate listeners.

    Args:
        engine: The async engine whose sync engine receives the listeners.
        on_connect_fn: The module's `connect` event callback.
        on_checkout_fn: The module's `checkout` event callback.
    """
    if not event.contains(target=engine.sync_engine, identifier="connect", fn=on_connect_fn):
        event.listen(target=engine.sync_engine, identifier="connect", fn=on_connect_fn)
    if not event.contains(target=engine.sync_engine, identifier="checkout", fn=on_checkout_fn):
        event.listen(target=engine.sync_engine, identifier="checkout", fn=on_checkout_fn)
