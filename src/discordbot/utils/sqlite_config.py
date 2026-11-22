"""Shared SQLite connection configuration and URL helpers for the project's DB engines.

Every SQLite engine in the project builds its URL through `sqlite_url` and configures its
connections here: the same WAL / synchronous / busy_timeout PRAGMA trade-off, plus the
integer-aware UDFs `StoredInteger` needs.

`SqliteBootstrap` owns the layer above that: a one-shot `checkout` listener and session factory.
The shared `reply.db` stores all open sessions against `reply_db_engine` via `open_reply_session`
with `enable_foreign_keys=True`. The schema itself comes only from the Alembic migrations under
`migrations/`.
"""

from typing import Any, Final
from pathlib import Path
import contextlib
from collections.abc import AsyncGenerator

from alembic import command
from pydantic import Field, BaseModel
from sqlalchemy import event
from alembic.config import Config
from sqlalchemy.pool import ConnectionPoolEntry
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from discordbot.typings.timeouts import SQLITE_BUSY_TIMEOUT_SECONDS
from discordbot.utils.stored_integer import configure_sqlite_stored_integer_functions

DATABASE_DIR: Final[Path] = Path("./data/database")

_upgraded_sections: set[tuple[Path, str]] = set()


def sqlite_url(filename: str) -> str:
    """Returns the async SQLite URL for `filename` under `DATABASE_DIR`.

    Args:
        filename (str): SQLite filename, such as `"reply.db"`.
    """
    return f"sqlite+aiosqlite:///{(DATABASE_DIR / filename).as_posix()}"


def ensure_database_upgraded(section: str) -> None:
    """Upgrades the Alembic `section` database to head once per working directory and section.

    Creates `DATABASE_DIR` before running Alembic so fresh checkouts and containers boot without
    pre-seeded directories. Safe to call from multiple cogs sharing one database.

    Args:
        section (str): Section name in `alembic.ini` to upgrade to `"head"`.
    """
    key = (Path.cwd().resolve(), section)
    if key in _upgraded_sections:
        return
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
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
    """One-shot checkout hook and session factory for an async SQLite database.

    Does not hold an engine; callers pass their current `AsyncEngine` on each open so engine swaps
    take effect immediately without leaking pool or inspector state.
    """

    enable_foreign_keys: bool = Field(
        default=False, description="Whether new connections turn on `PRAGMA foreign_keys`."
    )

    def _on_checkout(
        self,
        dbapi_connection: object,
        connection_record: ConnectionPoolEntry,
        _connection_proxy: object,
    ) -> None:
        """Configures a DBAPI connection once per pooled connection lifetime."""
        if connection_record.info.get("sqlite_configured") is self.enable_foreign_keys:
            return
        configure_sqlite_connection(
            dbapi_connection=dbapi_connection, enable_foreign_keys=self.enable_foreign_keys
        )
        connection_record.info["sqlite_configured"] = self.enable_foreign_keys

    def install_hooks(self, engine: AsyncEngine) -> None:
        """Installs the `checkout` listener on `engine` idempotently.

        Listening on `checkout` covers both newly opened connections and connections already in the
        pool when the hook attaches, while `connection_record.info` keeps configuration one-shot
        per DBAPI connection lifetime.

        Args:
            engine (AsyncEngine): Async engine whose sync engine receives the listener.
        """
        # SQLAlchemy keys bound methods on (id(__func__), id(__self__)), deduplicating safely.
        if not event.contains(
            target=engine.sync_engine, identifier="checkout", fn=self._on_checkout
        ):
            event.listen(target=engine.sync_engine, identifier="checkout", fn=self._on_checkout)

    @contextlib.asynccontextmanager
    async def open_session(self, engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
        """Yields an `AsyncSession` bound to `engine`, its connection hooks installed first.

        Args:
            engine (AsyncEngine): The module's current engine.
        """
        self.install_hooks(engine=engine)
        async with AsyncSession(bind=engine, expire_on_commit=False) as session:
            yield session


reply_db_engine: AsyncEngine = create_async_engine(url=sqlite_url(filename="reply.db"))
_reply_database = SqliteBootstrap(enable_foreign_keys=True)


def current_reply_db_engine() -> AsyncEngine:
    """Returns the active shared `reply.db` engine (reflects test monkeypatches)."""
    return reply_db_engine


def open_reply_session(
    engine: AsyncEngine | None = None,
) -> contextlib.AbstractAsyncContextManager[AsyncSession]:
    """Returns an async session context manager on the shared `reply.db` engine.

    Args:
        engine (AsyncEngine | None): Explicit engine to open on, or None for `reply_db_engine`.
    """
    return _reply_database.open_session(engine=reply_db_engine if engine is None else engine)
