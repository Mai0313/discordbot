"""One SQLite connection setup, so every engine in the project opens its connections alike.

Economy, stock, games, fishing, memory, research, the message log and the message-cleanup table
each own a separate engine — several of them over a shared file, games / fishing / message-cleanup
on `games.db` and memory / research on `reply.db` — and all of them want the same PRAGMA
trade-off: WAL, `synchronous=NORMAL`, and a tolerant `busy_timeout`.
`configure_sqlite_connection` is that setup, with the two things that genuinely differ left as
flags — foreign keys (economy alone asks for them) and the `StoredInteger` UDFs (on by default,
turned off by the two stores that hold no `StoredInteger` column, `log_msg` and
`message_cleanup`).

`ensure_sqlite_hooks` is the other half, and exists because of how those listeners are
registered. A module-level `@event.listens_for(_engine.sync_engine, "connect")` binds to
whichever engine existed at import time, so a test that monkeypatches the module's `_engine`
onto a `tmp_path` gets an engine carrying no listener and therefore no PRAGMAs at all. Session
factories re-install on every open instead, and this helper is what keeps that idempotent.

It owns no engine, no path and no schema: which file an engine opens, and when its tables are
created, stays with the module that owns the data. It sits in `utils/` because both `services/`
engines and cog-owned engines configure connections through it, and neither layer may reach
into the other to share one.
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
    """Applies the project's standard PRAGMA setup to one SQLite connection.

    WAL flips the read/write lock so readers never block on a writer, and `synchronous=NORMAL`
    is the durability trade WAL makes safe. The `busy_timeout` is what lets a contended writer
    wait instead of surfacing SQLITE_BUSY, which is why the economy engine's
    SELECT-then-conditional-UPDATE loops can keep a small retry budget. Everything here except
    `journal_mode` is per-connection state, so this has to run for every connection the pool opens
    rather than once per file.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
        enable_foreign_keys (bool): Whether to turn on `PRAGMA foreign_keys` for the connection.
        register_stored_integer (bool): Whether to register the integer-aware UDFs `StoredInteger`
            columns compare and add with. A store with no `StoredInteger` column passes False.
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
    """Installs the connect and checkout listeners on an engine, at most once each.

    Session factories call this on every open, because a test that swaps the module-level engine
    produces one the import-time `@event.listens_for` never reached; `event.contains` is what
    stops those repeat calls stacking another copy of each listener on every open.

    Both listeners are needed. `connect` fires only when the pool opens a NEW connection, so a
    connection the swapped engine already pooled (the test fixtures create the schema before they
    monkeypatch) would stay unconfigured forever; `checkout` catches that one on its way out of
    the pool.

    Args:
        engine (AsyncEngine): The async engine whose sync engine receives the listeners.
        on_connect_fn (Callable[..., None]): The module's `connect` event callback.
        on_checkout_fn (Callable[..., None]): The module's `checkout` event callback.
    """
    if not event.contains(target=engine.sync_engine, identifier="connect", fn=on_connect_fn):
        event.listen(target=engine.sync_engine, identifier="connect", fn=on_connect_fn)
    if not event.contains(target=engine.sync_engine, identifier="checkout", fn=on_checkout_fn):
        event.listen(target=engine.sync_engine, identifier="checkout", fn=on_checkout_fn)
