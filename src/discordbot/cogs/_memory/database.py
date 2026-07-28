"""Persistent phase-1 memory extraction inbox (`data/database/reply.db`).

One row per scope (a user or a bot-per-server). The row durably stages a phase-1
extraction turn so a bot restart resumes the work the in-memory pipeline had not
yet flushed to `raw.md`. Success is *recorded* (`status='done'`, `transcript`
cleared) rather than the row deleted, so the table doubles as an inspectable
per-scope processing state; an LLM failure parks the row at `status='failed'`
with its transcript kept, so the restart sweep retries it without any timeout
tuning. A user-requested memory clear is the one caller that deletes a row
outright (`delete_job`), since the staged transcript is exactly the content the
clear erases and a surviving row would resume it after the next restart.

Engine, PRAGMA hooks, and the schema bootstrap follow `cogs/_research/database.py`
exactly: a module-level `AsyncEngine` singleton on the shared `reply.db` (a
per-instance `cached_property` engine would leak the pool / dialect cache), with
this module owning its own `Base` and the `memory_job` table, distinct from
research's `research` table in the same file. No money columns, so no
`StoredInteger`. Like research it avoids `from __future__ import annotations`:
SQLAlchemy resolves the `Mapped[datetime]` columns at class-definition time. The
one addition is `_rekey_legacy_server_scopes`, a self-retiring rename of server
rows still keyed on the bot id, run under the same one-shot bootstrap.

The version / ordering token is `time.time_ns()` (an INTEGER), not a `DateTime`:
`database_now()` is Asia/Taipei wall-clock with microsecond collisions and
round-trips tz-naive under aiosqlite, so it cannot back a per-turn newest-wins
guard; an integer nanosecond clock is per-scope-unique in practice (two turns for
one scope are a whole reply cycle apart) and stays comparable across restarts.
The newest-wins upsert and the terminal updates are guarded on this token, so a
stale turn's write no-ops once a newer turn has overwritten the scope's row.
"""

from typing import Any, Literal, cast
from datetime import datetime

import logfire
from pydantic import Field, BaseModel
from sqlalchemy import Text, String, Integer, DateTime, CursorResult, event, delete, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, AsyncConnection, create_async_engine
from sqlalchemy.dialects.sqlite import insert

from discordbot.utils.timezone import database_now as _database_now
from discordbot.cogs._memory.store import BOT_MEMORY_DIR_NAME
from discordbot.utils.asyncio_locks import LoopLocalLock
from discordbot.utils.sqlite_config import ensure_sqlite_hooks, configure_sqlite_connection

# Memory flavor stored per row so the restart sweep rebuilds the matching extractor.
MemoryJobFlavor = Literal["user", "server"]
# Lifecycle of a persisted extraction turn, stored in the `status` column.
MemoryJobStatus = Literal["pending", "done", "failed"]
# `last_error` is a bounded blurb, not a full traceback.
_MAX_ERROR_CHARS = 500

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/reply.db")


def _configure_sqlite_connection(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Applies the project's standard PRAGMA setup to a new reply.db connection."""
    configure_sqlite_connection(dbapi_connection=dbapi_connection)


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Configures a newly opened SQLite connection."""
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _configure_sqlite_on_checkout(
    dbapi_connection: object, _connection_record: object, _connection_proxy: object
) -> None:
    """Configures pooled connections from test-swapped engines."""
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


class Base(DeclarativeBase):
    """Base class for the memory_job ORM model (its own metadata, not research's)."""

    pass


class MemoryJobRow(Base):
    """One scope's persisted phase-1 extraction turn.

    Attributes:
        scope: Opaque memory scope (``<user_id>`` or ``bot_memories/<server_id>``); primary key.
        flavor: ``user`` or ``server`` so the restart sweep picks the matching extractor.
        subject: The phase-1 directive naming the target (``target_user_id: <id>`` etc.).
        transcript: The rendered phase-1 input; set to NULL once the turn is ``done``.
        identity: Single-line identity stamped into main.md, persisted so resume needs no Discord context.
        status: Lifecycle status (see ``MemoryJobStatus``).
        token: ``time.time_ns()`` version / ordering token; guards newest-wins and the terminal update.
        last_error: Bounded failure blurb when ``status='failed'``.
        created_at: First-write timestamp.
        updated_at: Latest-write timestamp.
    """

    __tablename__ = "memory_job"

    scope: Mapped[str] = mapped_column(String(length=128), primary_key=True)
    flavor: Mapped[str] = mapped_column(String(length=16), nullable=False)
    subject: Mapped[str] = mapped_column(String(length=128), nullable=False)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    identity: Mapped[str] = mapped_column(String(length=256), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(length=16), default="pending", nullable=False)
    token: Mapped[int] = mapped_column(Integer, nullable=False)
    last_error: Mapped[str | None] = mapped_column(String(length=512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


class MemoryJob(BaseModel):
    """A memory_job row read back from `reply.db`."""

    scope: str = Field(..., description="Opaque memory scope; primary key.")
    flavor: MemoryJobFlavor = Field(..., description="User or server flavor of the scope.")
    subject: str = Field(..., description="The phase-1 directive naming the extraction target.")
    transcript: str | None = Field(
        ..., description="The rendered phase-1 input, or None once the turn is done."
    )
    identity: str = Field(..., description="Single-line identity stamped into main.md.")
    status: MemoryJobStatus = Field(..., description="Lifecycle status of the turn.")
    token: int = Field(..., description="time.time_ns() version / ordering token.")
    last_error: str | None = Field(..., description="Bounded failure blurb when failed.")


_schema_ready_for: AsyncEngine | None = None
_schema_lock = LoopLocalLock()


async def _rekey_legacy_server_scopes(conn: AsyncConnection) -> None:
    """Moves server rows still keyed on the bot id onto the `bot_memories/` scope.

    The scope column is the store path, so a row staged before the memories
    directory was renamed points at a path the running code no longer writes: the
    restart sweep would resume that transcript into a phantom `<bot_id>/<server_id>`
    directory instead of the guild's real memory. Re-keying is a plain rename of a
    handful of rows (one guild can have one per bot id it was staged under),
    idempotent, and self-retiring once no legacy row is left; a legacy row whose
    target key is already taken is dropped rather than updated, both to keep the
    primary key intact and because the row holding it is the newer turn the token
    guard would have kept anyway — a `bot_memories/` row can only have been written
    after the rename, and the newest-token ordering below is what makes that true
    for a guild whose rows are spread over two bot ids as well.
    """
    # Read every server row and partition in Python rather than filtering by a LIKE
    # prefix: there is one such row per guild and bot id, and `bot_memories` carries
    # a `_`, which LIKE would read as a wildcard unless escaped.
    result = await conn.execute(
        statement=select(MemoryJobRow.scope)
        .where(MemoryJobRow.flavor == "server")
        .order_by(MemoryJobRow.token.desc())
    )
    server_scopes = result.scalars().all()
    prefix = f"{BOT_MEMORY_DIR_NAME}/"
    legacy_scopes = [
        scope for scope in server_scopes if "/" in scope and not scope.startswith(prefix)
    ]
    if not legacy_scopes:
        return
    taken = {scope for scope in server_scopes if scope.startswith(prefix)}
    for scope in legacy_scopes:
        _, _, server_id = scope.partition("/")
        target = f"{BOT_MEMORY_DIR_NAME}/{server_id}"
        if target in taken:
            await conn.execute(statement=delete(MemoryJobRow).where(MemoryJobRow.scope == scope))
            continue
        await conn.execute(
            statement=update(MemoryJobRow).where(MemoryJobRow.scope == scope).values(scope=target)
        )
        taken.add(target)
    logfire.info("re-keyed legacy server memory jobs", count=len(legacy_scopes))


async def _ensure_schema() -> None:
    """Bootstraps the `memory_job` table once per engine (loop-local-locked)."""
    global _schema_ready_for  # noqa: PLW0603 -- module-level cache by engine identity
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    if _schema_ready_for is _engine:
        return
    async with _schema_lock.get():
        if _schema_ready_for is _engine:
            return
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await _rekey_legacy_server_scopes(conn=conn)
        _schema_ready_for = _engine


def open_session() -> AsyncSession:
    """Creates an async session bound to the current reply.db engine."""
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    return AsyncSession(bind=_engine, expire_on_commit=False)


def cast_flavor(value: str) -> MemoryJobFlavor:
    """Narrows a stored flavor string, defaulting odd values to user."""
    return "server" if value == "server" else "user"


def cast_status(value: str) -> MemoryJobStatus:
    """Narrows a stored status string, defaulting odd values to pending."""
    if value in ("pending", "done", "failed"):
        return cast("MemoryJobStatus", value)
    return "pending"


def _row_to_model(row: MemoryJobRow) -> MemoryJob:
    """Maps an ORM row to its pydantic snapshot."""
    return MemoryJob(
        scope=row.scope,
        flavor=cast_flavor(value=row.flavor),
        subject=row.subject,
        transcript=row.transcript,
        identity=row.identity,
        status=cast_status(value=row.status),
        token=row.token,
        last_error=row.last_error,
    )


async def upsert_pending(  # noqa: PLR0913 -- one row's columns are all per-call inputs
    *,
    scope: str,
    flavor: MemoryJobFlavor,
    subject: str,
    transcript: str,
    identity: str,
    token: int,
) -> None:
    """Records (newest-wins) a pending extraction turn for a scope.

    On conflict the row is overwritten only when the new `token` is strictly
    newer than the stored one, so an older turn's write can never clobber a newer
    turn's row (the guard that keeps two interleaved turns consistent).
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        stmt = insert(MemoryJobRow).values(
            scope=scope,
            flavor=flavor,
            subject=subject,
            transcript=transcript,
            identity=identity,
            status="pending",
            token=token,
            last_error=None,
            created_at=now,
            updated_at=now,
        )
        await session.execute(
            statement=stmt.on_conflict_do_update(
                index_elements=["scope"],
                set_={
                    "flavor": flavor,
                    "subject": subject,
                    "transcript": transcript,
                    "identity": identity,
                    "status": "pending",
                    "token": token,
                    "last_error": None,
                    "updated_at": now,
                },
                where=MemoryJobRow.token < token,
            )
        )
        await session.commit()


async def mark_done(*, scope: str, token: int) -> None:
    """Marks a turn done and drops its now-consumed transcript (token-guarded)."""
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        await session.execute(
            statement=update(MemoryJobRow)
            .where(MemoryJobRow.scope == scope, MemoryJobRow.token == token)
            .values(status="done", transcript=None, last_error=None, updated_at=now)
        )
        await session.commit()


async def mark_failed(*, scope: str, token: int, error: str) -> None:
    """Parks a turn at failed, keeping its transcript for a restart retry (token-guarded)."""
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        await session.execute(
            statement=update(MemoryJobRow)
            .where(MemoryJobRow.scope == scope, MemoryJobRow.token == token)
            .values(status="failed", last_error=error[:_MAX_ERROR_CHARS], updated_at=now)
        )
        await session.commit()


async def delete_job(*, scope: str) -> bool:
    """Drops a scope's row entirely, for a user-requested memory clear.

    Deletion rather than a `mark_done` transition, for two reasons: the row's
    `transcript` holds the very conversation the user asked to erase, and the
    terminal transitions are token-guarded while a clear holds no token. Removing
    the row is also what keeps the clear whole, since `list_resumable` would
    otherwise hand the restart sweep a turn that rebuilds the wiped memory.

    Returns:
        True when a row existed and was removed.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = cast(
            "CursorResult[Any]",
            await session.execute(
                statement=delete(MemoryJobRow).where(MemoryJobRow.scope == scope)
            ),
        )
        await session.commit()
        return bool(result.rowcount and result.rowcount > 0)


async def list_resumable() -> list[MemoryJob]:
    """Returns every non-`done` row, for the restart resume sweep."""
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(MemoryJobRow).where(MemoryJobRow.status != "done")
        )
        return [_row_to_model(row=row) for row in result.scalars().all()]


async def get_job(*, scope: str) -> MemoryJob | None:
    """Reads one scope's row, or `None` when it is not tracked."""
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(MemoryJobRow).where(MemoryJobRow.scope == scope)
        )
        row = result.scalars().one_or_none()
        return _row_to_model(row=row) if row is not None else None
