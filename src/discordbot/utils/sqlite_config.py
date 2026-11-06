"""Shared SQLite connection configuration for the project's DB engines.

Every SQLite engine in the project configures its connections here: the same WAL /
synchronous / busy_timeout PRAGMA trade-off, plus the integer-aware UDFs `StoredInteger`
needs.

`SqliteBootstrap` owns the layer above that: the connect and checkout listeners and the session
factory. The schema itself comes only from the Alembic migrations under `migrations/`.
"""

from typing import Any
import contextlib
from collections.abc import AsyncGenerator

from pydantic import Field, BaseModel
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from discordbot.typings.timeouts import SQLITE_BUSY_TIMEOUT_SECONDS
from discordbot.utils.stored_integer import configure_sqlite_stored_integer_functions


def configure_sqlite_connection(
    dbapi_connection: Any,  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    enable_foreign_keys: bool = False,
) -> None:
    """Applies the project's standard PRAGMA setup to a new SQLite connection.

    WAL flips the read/write lock so readers never block on writes.
    `synchronous=NORMAL` is the right durability trade-off under it: every commit still
    fsyncs the WAL frame, and only the main file waits for a checkpoint rather than every
    write. A tolerant `busy_timeout` gives writers time to wait under contention.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
        enable_foreign_keys (bool): Whether to turn on `PRAGMA foreign_keys` for the connection.
    """
    with contextlib.closing(dbapi_connection.cursor()) as cursor:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute(f"PRAGMA busy_timeout={int(SQLITE_BUSY_TIMEOUT_SECONDS * 1000)}")
        if enable_foreign_keys:
            cursor.execute("PRAGMA foreign_keys=ON")
    configure_sqlite_stored_integer_functions(dbapi_connection=dbapi_connection)


class SqliteBootstrap(BaseModel):
    """The hooks and session factory one async SQLite database needs.

    What it deliberately does not hold is the engine. That stays a module-level `_engine`
    binding per module, because a `cached_property` engine would leak the pool, the dialect
    and the inspector cache, and because tests monkeypatch it by that name. So every method
    here takes the caller's current engine instead, which is what makes a swapped `_engine`
    take effect on the very next call rather than on the next process.
    """

    enable_foreign_keys: bool = Field(
        default=False, description="Whether new connections turn on `PRAGMA foreign_keys`."
    )

    def _on_connect(self, dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
        """Configures a newly opened SQLite connection."""
        configure_sqlite_connection(
            dbapi_connection=dbapi_connection, enable_foreign_keys=self.enable_foreign_keys
        )

    def _on_checkout(
        self, dbapi_connection: object, _connection_record: object, _connection_proxy: object
    ) -> None:
        """Re-applies the connection setup each time a connection leaves the pool."""
        configure_sqlite_connection(
            dbapi_connection=dbapi_connection, enable_foreign_keys=self.enable_foreign_keys
        )

    def install_hooks(self, engine: AsyncEngine) -> None:
        """Installs the connect and checkout listeners on an engine exactly once.

        Called on every open, because tests swap the module engines; `event.contains` keeps
        repeat calls from stacking duplicate listeners. The per-open call is required rather
        than defensive: an engine handed over with a connection already in its pool has one
        that never saw `connect`, and `checkout` is the only listener that can still register
        the `StoredInteger` UDFs onto it. Without it that connection raises
        `no such function: discordbot_int_add_text`.

        Bound methods are safe to hand it, which is not obvious: SQLAlchemy keys a plain
        function on `id(fn)`, which a freshly bound method would defeat, and special-cases
        `MethodType` to `(id(fn.__func__), id(fn.__self__))` instead.

        Args:
            engine (AsyncEngine): The async engine whose sync engine receives the listeners.
        """
        listeners = (("connect", self._on_connect), ("checkout", self._on_checkout))
        for identifier, listener in listeners:
            if not event.contains(target=engine.sync_engine, identifier=identifier, fn=listener):
                event.listen(target=engine.sync_engine, identifier=identifier, fn=listener)

    @contextlib.asynccontextmanager
    async def open_session(self, engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
        """Yields an `AsyncSession` bound to `engine`, its connection hooks installed first.

        Args:
            engine (AsyncEngine): The module's current engine.
        """
        self.install_hooks(engine=engine)
        async with AsyncSession(bind=engine, expire_on_commit=False) as session:
            yield session
