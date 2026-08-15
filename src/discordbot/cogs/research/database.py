"""Persistent deep-research session store (`data/database/reply.db`).

One row per launched research thread. The row lets a bot restart resume an
in-flight research: the Gemini interaction runs server-side with `store=True`,
so after a restart the cog reloads the rows still in `researching` and re-attaches
a live stream to each `interaction_id`. It is also the per-user concurrency
guard (one active research per owner).

The engine is a module-level `AsyncEngine` singleton, exactly like
`services/economy/database.py`: a per-instance `cached_property` engine would leak
the connection pool / dialect cache for every interaction. `reply.db` is the
shared file for reply-side persistence (this table plus the memory pipeline's
phase-1 inbox in `services/memory/database.py`, which keeps its own engine on the
same file); it has no money columns, so no `StoredInteger`. Each call opens an `AsyncSession`
bound to the current `_engine`, so tests can monkeypatch `_engine` per-test.

This module deliberately avoids `from __future__ import annotations`: SQLAlchemy
resolves the `Mapped[datetime]` column annotations at class-definition time, and
postponed evaluation breaks that.
"""

from typing import Any, Literal, cast
from datetime import datetime

from pydantic import Field, BaseModel
from sqlalchemy import String, Integer, DateTime, event, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.dialects.sqlite import insert

from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.asyncio_locks import LoopLocalLock
from discordbot.utils.sqlite_config import ensure_sqlite_hooks, configure_sqlite_connection

# Lifecycle of a research session, persisted in the `phase` column. A row left `planning` by the
# removed escalation tiers is not migrated: nothing selects that value any more, so it is inert.
ResearchPhase = Literal["researching", "done", "failed", "cancelled"]

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
    """Base class for reply.db ORM models."""

    pass


class ResearchSessionRow(Base):
    """One launched deep-research thread.

    Attributes:
        thread_id: Discord thread ID; primary key.
        owner_id: Discord user ID that launched the research.
        channel_id: The thread's parent channel ID.
        guild_id: Guild ID. Nullable in the schema, but never written `None`: a launch outside a
            guild text channel is refused before any row exists.
        source_message_id: The message the thread was anchored to.
        agent: The Gemini agent string currently running this session.
        interaction_id: The running interaction's id; `None` before it starts.
        brief: The research brief. Nothing reads it back, but it is the row's only human-readable
            identity, and the column is `NOT NULL` with no server default, so dropping it from the
            model would break every INSERT against an already-deployed `reply.db`.
        phase: Lifecycle phase (see `ResearchPhase`).
        created_at: First-write timestamp.
        updated_at: Latest-write timestamp.
    """

    __tablename__ = "research"

    thread_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(Integer, nullable=False)
    channel_id: Mapped[int] = mapped_column(Integer, nullable=False)
    guild_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    agent: Mapped[str] = mapped_column(String(length=64), nullable=False)
    interaction_id: Mapped[str | None] = mapped_column(String(length=256), nullable=True)
    brief: Mapped[str] = mapped_column(String(length=16384), default="", nullable=False)
    phase: Mapped[str] = mapped_column(String(length=16), default="researching", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


class PersistentResearchSession(BaseModel):
    """A research session row read back from `reply.db`."""

    thread_id: int = Field(..., description="Discord thread ID; primary key.")
    owner_id: int = Field(..., description="Discord user ID that launched the research.")
    channel_id: int = Field(..., description="Parent channel ID the thread hangs off.")
    guild_id: int | None = Field(
        ..., description="Guild ID; nullable in the schema but never written None."
    )
    source_message_id: int = Field(..., description="The message the thread was anchored to.")
    agent: str = Field(..., description="The Gemini agent string currently running this session.")
    interaction_id: str | None = Field(
        ..., description="The running interaction's id; None before it starts."
    )
    brief: str = Field(..., description="The research brief the session was launched with.")
    phase: ResearchPhase = Field(..., description="Lifecycle phase of the session.")


_schema_ready_for: AsyncEngine | None = None
_schema_lock = LoopLocalLock()


async def _ensure_schema() -> None:
    """Bootstraps the `research` table once per engine (loop-local-locked)."""
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


def _row_to_model(row: ResearchSessionRow) -> PersistentResearchSession:
    """Maps an ORM row to its pydantic snapshot."""
    return PersistentResearchSession(
        thread_id=row.thread_id,
        owner_id=row.owner_id,
        channel_id=row.channel_id,
        guild_id=row.guild_id,
        source_message_id=row.source_message_id,
        agent=row.agent,
        interaction_id=row.interaction_id,
        brief=row.brief,
        phase=cast_phase(value=row.phase),
    )


def cast_phase(value: str) -> ResearchPhase:
    """Narrows a stored phase string to `ResearchPhase`, defaulting odd values to failed."""
    if value in ("researching", "done", "failed", "cancelled"):
        return cast("ResearchPhase", value)
    return "failed"


async def upsert_session(  # noqa: PLR0913 -- one row's columns are all per-call inputs
    *,
    thread_id: int,
    owner_id: int,
    channel_id: int,
    guild_id: int | None,
    source_message_id: int,
    agent: str,
    interaction_id: str | None,
    brief: str,
    phase: ResearchPhase,
) -> None:
    """Creates or overwrites the session row for a thread."""
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        stmt = insert(ResearchSessionRow).values(
            thread_id=thread_id,
            owner_id=owner_id,
            channel_id=channel_id,
            guild_id=guild_id,
            source_message_id=source_message_id,
            agent=agent,
            interaction_id=interaction_id,
            brief=brief,
            phase=phase,
            created_at=now,
            updated_at=now,
        )
        await session.execute(
            statement=stmt.on_conflict_do_update(
                index_elements=["thread_id"],
                set_={
                    "owner_id": owner_id,
                    "channel_id": channel_id,
                    "guild_id": guild_id,
                    "source_message_id": source_message_id,
                    "agent": agent,
                    "interaction_id": interaction_id,
                    "brief": brief,
                    "phase": phase,
                    "updated_at": now,
                },
            )
        )
        await session.commit()


async def set_interaction(
    *, thread_id: int, interaction_id: str, agent: str, phase: ResearchPhase
) -> None:
    """Updates the running interaction id / agent / phase for a thread."""
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        await session.execute(
            statement=update(ResearchSessionRow)
            .where(ResearchSessionRow.thread_id == thread_id)
            .values(interaction_id=interaction_id, agent=agent, phase=phase, updated_at=now)
        )
        await session.commit()


async def set_phase(*, thread_id: int, phase: ResearchPhase) -> None:
    """Transitions a session to a new lifecycle phase."""
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        await session.execute(
            statement=update(ResearchSessionRow)
            .where(ResearchSessionRow.thread_id == thread_id)
            .values(phase=phase, updated_at=now)
        )
        await session.commit()


async def list_resumable() -> list[PersistentResearchSession]:
    """Returns sessions still `researching`, for the restart resume sweep."""
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(ResearchSessionRow).where(ResearchSessionRow.phase == "researching")
        )
        return [_row_to_model(row=row) for row in result.scalars().all()]


async def active_thread_for_owner(*, owner_id: int) -> int | None:
    """Returns an owner's in-flight research thread id, or `None` when they have none.

    The concurrency guard: an owner may only have one `researching` session at a time, so a new
    launch is refused while one is active.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(ResearchSessionRow.thread_id).where(
                ResearchSessionRow.owner_id == owner_id, ResearchSessionRow.phase == "researching"
            )
        )
        return result.scalars().first()
