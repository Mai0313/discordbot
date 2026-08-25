"""Persistent phase-1 memory extraction inbox (`data/database/reply.db`).

One row per scope (a user or a bot-per-server). The row durably stages a phase-1
extraction turn so a bot restart resumes the work the in-memory pipeline had not
yet flushed to `raw.md`. Success is *recorded* (`status='done'`, `transcript`
cleared) rather than the row deleted, so the table doubles as an inspectable
per-scope processing state; an LLM failure parks the row at `status='failed'`
with its transcript kept, so the restart sweep retries it without any timeout
tuning. A user-requested memory clear replaces the row with a transcript-free
`cleared` tombstone. Its ordering token prevents a staging write captured before
the clear from recreating the erased transcript after the tombstone commits, and
a row already newer than the clear makes `clear_job` refuse rather than report an
empty scope, so the caller never erases the files behind a tombstone that no-opped.

Engine, PRAGMA hooks, and the schema bootstrap follow `cogs/research/database.py`
exactly: a module-level `AsyncEngine` singleton on the shared `reply.db` (a
per-instance `cached_property` engine would leak the pool / dialect cache), with
this module owning its own `Base` and the `memory_job` table, distinct from
research's `research` table in the same file. No money columns, so no
`StoredInteger`. Like research it avoids `from __future__ import annotations`:
SQLAlchemy resolves the `Mapped[datetime]` columns at class-definition time.

The version / ordering token is a logical INTEGER, not a wall clock. Each process
reserves one range from `memory_token_clock`, above both the prior watermark and
every legacy token already in `memory_job`, then assigns turns and clears in
capture order inside that range. This keeps newest-wins comparable across
restarts without letting an NTP clock correction make a clear older than the
transcript it must erase. The upsert and terminal updates are guarded on this
token, so a stale turn's write no-ops once a newer turn has overwritten the
scope's row.
"""

from typing import Any, Literal, cast
from datetime import datetime
from itertools import count
from threading import Lock

from pydantic import Field, BaseModel
from sqlalchemy import Text, String, Integer, DateTime, func, text, event, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.dialects.sqlite import insert

from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.asyncio_locks import LoopLocalLock
from discordbot.utils.sqlite_config import ensure_sqlite_hooks, configure_sqlite_connection

# Memory flavor stored per row so the restart sweep rebuilds the matching writer.
MemoryJobFlavor = Literal["user", "server"]
# Lifecycle of a persisted extraction turn, stored in the `status` column.
MemoryJobStatus = Literal["pending", "done", "failed", "cleared"]
# `last_error` is a bounded blurb, not a full traceback.
_MAX_ERROR_CHARS = 500
# One process would need a trillion captured memory events to exhaust its range.
_TOKEN_BLOCK_SIZE = 1_000_000_000_000
_SQLITE_MAX_INTEGER = (1 << 63) - 1

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/reply.db")

# Negative values are process-local placeholders. The first database operation
# maps them into a range reserved durably from SQLite; the same placeholder then
# resolves to the same positive token for every later terminal update.
_token_sequence = count(start=1)
_token_block_bases: dict[AsyncEngine, int] = {}
_token_state_lock = Lock()


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
    """Base class for the memory ORM models (their own metadata, not research's)."""

    pass


class MemoryJobRow(Base):
    """One scope's persisted phase-1 extraction turn.

    Attributes:
        scope: Opaque memory scope (``<user_id>`` or ``bot_memories/<server_id>``); primary key.
        flavor: ``user`` or ``server`` so the restart sweep picks the matching writer.
        subject: The phase-1 directive naming the target (``target_user_id: <id>`` etc.).
        transcript: The rendered phase-1 input; set to NULL once the turn is ``done``.
        identity: Single-line identity ``parse_identity`` splits into the ``owner_id`` /
            ``owner_name`` stamped on every fact this scope writes; persisted so a resume
            needs no Discord context.
        status: Lifecycle status (see ``MemoryJobStatus``).
        token: Logical version / ordering token; guards newest-wins and the terminal update.
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


class MemoryTokenClockRow(Base):
    """Singleton high watermark for process-level logical token reservations."""

    __tablename__ = "memory_token_clock"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    high_watermark: Mapped[int] = mapped_column(Integer, nullable=False)


class MemoryJob(BaseModel):
    """A memory_job row read back from `reply.db`."""

    scope: str = Field(..., description="Opaque memory scope; primary key.")
    flavor: MemoryJobFlavor = Field(..., description="User or server flavor of the scope.")
    subject: str = Field(..., description="The phase-1 directive naming the extraction target.")
    transcript: str | None = Field(
        ..., description="The rendered phase-1 input, or None once the turn is done."
    )
    identity: str = Field(..., description="Single-line identity stamped onto the scope's facts.")
    status: MemoryJobStatus = Field(..., description="Lifecycle status of the turn.")
    token: int = Field(..., description="Logical version / ordering token.")
    last_error: str | None = Field(..., description="Bounded failure blurb when failed.")


_schema_ready_for: AsyncEngine | None = None
_schema_lock = LoopLocalLock()


async def _ensure_schema() -> None:
    """Bootstraps this module's tables once per engine (loop-local-locked)."""
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
    if value in ("pending", "done", "failed", "cleared"):
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


def new_token() -> int:
    """Returns a process-local token placeholder in strict capture order."""
    with _token_state_lock:
        sequence = next(_token_sequence)
    if sequence > _TOKEN_BLOCK_SIZE:
        raise RuntimeError("memory token block exhausted")
    return -sequence


async def _reserve_token_block(*, engine: AsyncEngine) -> int:
    """Atomically reserves a token range and returns its exclusive lower bound."""
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(statement=text("BEGIN IMMEDIATE"))
        clock_high = await session.scalar(
            statement=select(MemoryTokenClockRow.high_watermark).where(MemoryTokenClockRow.id == 1)
        )
        job_high = await session.scalar(
            statement=select(func.max(MemoryJobRow.token)).where(MemoryJobRow.token > 0)
        )
        block_base = max(clock_high or 0, job_high or 0)
        if block_base > _SQLITE_MAX_INTEGER - _TOKEN_BLOCK_SIZE:
            raise RuntimeError("memory token space exhausted")
        high_watermark = block_base + _TOKEN_BLOCK_SIZE
        stmt = insert(MemoryTokenClockRow).values(id=1, high_watermark=high_watermark)
        await session.execute(
            statement=stmt.on_conflict_do_update(
                index_elements=["id"], set_={"high_watermark": high_watermark}
            )
        )
        await session.commit()
        return block_base


async def _resolve_token(*, token: int) -> int:
    """Maps a local placeholder to this process's durable token range."""
    if token >= 0:
        return token
    sequence = -token
    if sequence > _TOKEN_BLOCK_SIZE:
        raise RuntimeError("memory token placeholder is outside the reserved block")
    engine = _engine
    with _token_state_lock:
        block_base = _token_block_bases.get(engine)
    if block_base is None:
        reserved_base = await _reserve_token_block(engine=engine)
        # Two event loops can race to reserve ranges for the same engine. Both
        # reservations are durable and disjoint; the first cached range wins and
        # the other is simply an unused gap.
        with _token_state_lock:
            block_base = _token_block_bases.setdefault(engine, reserved_base)
    return block_base + sequence


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
    token = await _resolve_token(token=token)
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
    token = await _resolve_token(token=token)
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
    token = await _resolve_token(token=token)
    now = _database_now()
    async with open_session() as session:
        await session.execute(
            statement=update(MemoryJobRow)
            .where(MemoryJobRow.scope == scope, MemoryJobRow.token == token)
            .values(status="failed", last_error=error[:_MAX_ERROR_CHARS], updated_at=now)
        )
        await session.commit()


async def clear_job(*, scope: str, flavor: MemoryJobFlavor, token: int) -> bool:
    """Scrubs a scope's row and leaves a token-guarded clear tombstone.

    The tombstone closes both possible commit orderings with a staging write. A
    pending row that committed first is overwritten here; a stale upsert that
    commits later loses the existing newest-token guard. `BEGIN IMMEDIATE` makes
    the existence check and tombstone upsert one serialized write transaction, so
    the caller can still distinguish an empty scope without reopening that race.

    Returns:
        True when a non-cleared row existed and was scrubbed.

    Raises:
        RuntimeError: When the scope already carries a row newer than this clear, so
            the guarded upsert below would no-op. Reporting that as an ordinary
            "nothing to scrub" let the caller delete the files anyway and leave the
            pre-clear transcript in `reply.db` for the restart sweep to resume — the
            exact failure the tombstone exists to close, reached through the other
            door. Raising rolls this transaction back before the caller's file pass,
            so every tier stays in place. A retry is harmless but keeps refusing:
            `_resolve_token` reserves ONE block per process, so the clear's token
            only rises above the stray row after a restart picks a base above it.
            That same reservation is why this cannot fire while one process owns the
            block; the guard is what keeps a second writer against the same
            `reply.db` loud instead of silent.
    """
    await _ensure_schema()
    token = await _resolve_token(token=token)
    now = _database_now()
    async with open_session() as session:
        await session.execute(statement=text("BEGIN IMMEDIATE"))
        result = await session.execute(
            statement=select(MemoryJobRow.status, MemoryJobRow.token).where(
                MemoryJobRow.scope == scope
            )
        )
        previous = result.one_or_none()
        if previous is not None and previous.token > token:
            raise RuntimeError(
                f"memory_job token {previous.token} is newer than the clear's {token}; "
                "refusing to erase behind an unwritable tombstone"
            )
        stmt = insert(MemoryJobRow).values(
            scope=scope,
            flavor=flavor,
            subject="",
            transcript=None,
            identity="",
            status="cleared",
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
                    "subject": "",
                    "transcript": None,
                    "identity": "",
                    "status": "cleared",
                    "token": token,
                    "last_error": None,
                    "updated_at": now,
                },
                where=MemoryJobRow.token <= token,
            )
        )
        await session.commit()
        return previous is not None and previous.status != "cleared"


async def list_resumable() -> list[MemoryJob]:
    """Returns pending and failed rows for the restart resume sweep."""
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(MemoryJobRow).where(MemoryJobRow.status.in_(("pending", "failed")))
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
