"""Shared SQLite connection configuration for the project's DB engines.

Every SQLite engine in the project configures its connections here: the same WAL /
synchronous / busy_timeout PRAGMA trade-off, plus the integer-aware UDFs `StoredInteger`
needs.

`SqliteBootstrap` owns the layer above that: the connect and checkout listeners and the session
factory. The schema itself comes only from the Alembic migrations under `migrations/`.
"""

from typing import Any
from pathlib import Path
import contextlib
from collections.abc import AsyncGenerator

from alembic import command
from pydantic import Field, BaseModel
from sqlalchemy import event
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from discordbot.typings.timeouts import SQLITE_BUSY_TIMEOUT_SECONDS
from discordbot.utils.stored_integer import configure_sqlite_stored_integer_functions

_upgraded_sections: set[tuple[Path, str]] = set()


def ensure_database_upgraded(section: str) -> None:
    """Upgrades the Alembic `section` database to head once per working directory and section.

    Creates `./data/database` before running Alembic so fresh checkouts and containers boot
    without pre-seeded directories. Safe to call from multiple cogs sharing one database.

    Args:
        section (str): Section name in `alembic.ini` to upgrade to `"head"`.
    """
    key = (Path.cwd().resolve(), section)
    if key in _upgraded_sections:
        return
    Path("./data/database").mkdir(parents=True, exist_ok=True)
    command.upgrade(config=Config(file_="alembic.ini", ini_section=section), revision="head")
    _upgraded_sections.add(key)


def configure_sqlite_connection(
    dbapi_connection: Any,  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    enable_foreign_keys: bool = False,
) -> None:
    """Applies standard SQLite PRAGMAs (`WAL`, `synchronous=NORMAL`, `busy_timeout`) and UDFs.

    Args:
        dbapi_connection (Any): Freshly opened DBAPI connection.
        enable_foreign_keys (bool): Whether to enable `PRAGMA foreign_keys=ON`.
    """
    with contextlib.closing(dbapi_connection.cursor()) as cursor:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute(f"PRAGMA busy_timeout={int(SQLITE_BUSY_TIMEOUT_SECONDS * 1000)}")
        if enable_foreign_keys:
            cursor.execute("PRAGMA foreign_keys=ON")
    configure_sqlite_stored_integer_functions(dbapi_connection=dbapi_connection)


class SqliteBootstrap(BaseModel):
    """Connect/checkout hooks and session factory for an async SQLite database.

    Does not hold an engine; callers pass their module-level `_engine` on each open so engine
    swaps take effect immediately without leaking pool or inspector state.
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
        """Installs `connect` and `checkout` listeners on `engine` idempotently.

        Registers `checkout` as well as `connect` so pre-existing pooled connections also receive
        `StoredInteger` UDFs.

        Args:
            engine (AsyncEngine): Async engine whose sync engine receives the listeners.
        """
        listeners = (("connect", self._on_connect), ("checkout", self._on_checkout))
        for identifier, listener in listeners:
            # SQLAlchemy keys bound methods on (id(__func__), id(__self__)), deduplicating safely.
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
