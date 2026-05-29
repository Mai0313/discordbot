"""Shared SQLite connection configuration for the project's DB engines.

Economy, stock, message-log, and cleanup storage all open SQLite with the same
WAL / synchronous / busy_timeout PRAGMA trade-off. This helper centralizes that
setup so every engine configures connections the same way. `StoredInteger`
engines additionally register the integer-aware UDFs.
"""

from typing import Any

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
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    if enable_foreign_keys:
        cursor.execute("PRAGMA foreign_keys=ON")
    if register_stored_integer:
        configure_sqlite_stored_integer_functions(dbapi_connection=dbapi_connection)
    cursor.close()
