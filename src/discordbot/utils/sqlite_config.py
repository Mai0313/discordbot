"""Shared SQLite connection configuration for the project's DB engines.

Every SQLite engine in the project opens connections with the same WAL /
synchronous / busy_timeout PRAGMA trade-off. This helper centralizes that setup
so every engine configures connections the same way. `StoredInteger` engines
additionally register the integer-aware UDFs.

`SqliteBootstrap` owns the layer above that for the async engines: the connect and
checkout listeners, the lazy schema creation and the session factory, which were
copy-pasted into six modules before #608.
"""

from typing import Any, Protocol, runtime_checkable
import contextlib
from collections.abc import Awaitable

from pydantic import Field, BaseModel, ConfigDict, PrivateAttr
from sqlalchemy import MetaData, event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, AsyncConnection

from discordbot.typings.timeouts import SQLITE_BUSY_TIMEOUT_SECONDS
from discordbot.utils.asyncio_locks import LoopLocalLock
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
        # The one surface that wants milliseconds; the shared bound is in seconds like the rest.
        cursor.execute(f"PRAGMA busy_timeout={int(SQLITE_BUSY_TIMEOUT_SECONDS * 1000)}")
        if enable_foreign_keys:
            cursor.execute("PRAGMA foreign_keys=ON")
    if register_stored_integer:
        configure_sqlite_stored_integer_functions(dbapi_connection=dbapi_connection)


@runtime_checkable
class SqliteSchemaHook(Protocol):
    """Extra bootstrap a database runs inside its own `create_all` transaction."""

    def __call__(self, *, conn: AsyncConnection) -> Awaitable[None]:
        """Runs the extra bootstrap against the open connection."""
        ...


class SqliteBootstrap(BaseModel):
    """The hooks, lazy schema and session factory one async SQLite database needs.

    What it deliberately does NOT hold is the engine. That stays a module-level `_engine`
    binding per module, because a `cached_property` engine would leak the pool, the dialect
    and the inspector cache, and because tests monkeypatch it by that name. So every method
    here takes the caller's current engine instead, which is what makes a swapped `_engine`
    take effect on the very next call rather than on the next process.

    Attributes:
        metadata: The module's declarative metadata, created once per engine.
        enable_foreign_keys: Whether new connections turn on `PRAGMA foreign_keys`.
        after_create: Extra bootstrap run inside the same transaction as `create_all`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: MetaData = Field(
        ..., description="The module's declarative metadata, created once per engine."
    )
    enable_foreign_keys: bool = Field(
        default=False, description="Whether new connections turn on `PRAGMA foreign_keys`."
    )
    after_create: SqliteSchemaHook | None = Field(
        default=None,
        description="Extra bootstrap run inside the same transaction as `create_all`.",
    )

    # The engine identity rather than a bool, so pointing `_engine` at another file (a test
    # using a temp path) forces another schema check on its own.
    _ready_for: AsyncEngine | None = PrivateAttr(default=None)
    # SQLAlchemy's SQLite `create_all(checkfirst=True)` still has a check-then-create race
    # under concurrent first use, so schema creation is serialized per event loop.
    _schema_lock: LoopLocalLock = PrivateAttr(default_factory=LoopLocalLock)

    def _on_connect(self, dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
        """Configures a newly opened SQLite connection."""
        configure_sqlite_connection(
            dbapi_connection=dbapi_connection, enable_foreign_keys=self.enable_foreign_keys
        )

    def _on_checkout(
        self, dbapi_connection: object, _connection_record: object, _connection_proxy: object
    ) -> None:
        """Configures pooled connections from test-swapped engines."""
        configure_sqlite_connection(
            dbapi_connection=dbapi_connection, enable_foreign_keys=self.enable_foreign_keys
        )

    def install_hooks(self, engine: AsyncEngine) -> None:
        """Installs the connect and checkout listeners on an engine exactly once.

        Called once beside the module's own engine and again on every open, because tests
        swap those engines; `event.contains` keeps repeat calls from stacking duplicate
        listeners. The per-open call is required rather than defensive: a test that runs
        `create_all` on its fresh engine before handing it over leaves a connection already
        in the pool, and `checkout` is the only listener that can still register the
        `StoredInteger` UDFs onto it. Dropping it fails the economy suite outright on `no
        such function: discordbot_int_add_text`.

        Bound methods are safe to hand it, which is not obvious: SQLAlchemy keys a plain
        function on `id(fn)`, which a freshly bound method would defeat, and special-cases
        `MethodType` to `(id(fn.__func__), id(fn.__self__))` instead.

        Args:
            engine: The async engine whose sync engine receives the listeners.
        """
        listeners = (("connect", self._on_connect), ("checkout", self._on_checkout))
        for identifier, listener in listeners:
            if not event.contains(target=engine.sync_engine, identifier=identifier, fn=listener):
                event.listen(target=engine.sync_engine, identifier=identifier, fn=listener)

    async def ensure_schema(self, engine: AsyncEngine) -> None:
        """Creates this database's tables once per engine, hooks installed first.

        Args:
            engine: The module's current engine.
        """
        self.install_hooks(engine=engine)
        if self._ready_for is engine:
            return
        async with self._schema_lock.get():
            if self._ready_for is engine:
                return
            async with engine.begin() as conn:
                await conn.run_sync(self.metadata.create_all)
                if self.after_create is not None:
                    await self.after_create(conn=conn)
            self._ready_for = engine

    def open_session(self, engine: AsyncEngine) -> AsyncSession:
        """Creates an async session bound to the module's current engine.

        Args:
            engine: The module's current engine.

        Returns:
            An `AsyncSession` bound to it, with the connection hooks installed.
        """
        self.install_hooks(engine=engine)
        return AsyncSession(bind=engine, expire_on_commit=False)
