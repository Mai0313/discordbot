"""Persistent deep-research session store (`data/database/reply.db`).

One row per launched research thread, written by `cogs/research/cog.py` and by nothing else. The
row is what lets a bot restart resume an in-flight research: the Gemini interaction runs
server-side with `store=True`, so after a restart the cog reloads the rows still in `researching`
and re-attaches to each stored `interaction_id`. That same `researching` predicate is the
per-owner concurrency guard, so leaving the phase both frees the owner's one slot and drops the
row out of the resume sweep. Nothing is ever deleted here: a terminal phase is recorded in place,
so the table doubles as the history of every research the bot has run.

It sits beside `cog.py` rather than inside it so the schema can be imported without the Discord
surface — `tests/conftest.py` imports `Base` to build a per-test `reply.db` on a `tmp_path`, and
importing the cog module for that would drag nextcord and everything the cog constructs along
with it. Reads hand back `PersistentResearchSession` snapshots rather than ORM rows, so nothing
outside this module can read a row after its session closed. The resume takes only `thread_id`,
`owner_id`, `agent` and `interaction_id` off a snapshot, and `phase` only as a SQL predicate; the
rest of the row is written for the record (see `ResearchSessionRow.brief` for why an unread column
still cannot be dropped).

The engine is a module-level `AsyncEngine` singleton, exactly like `services/economy/database.py`:
a per-instance `cached_property` engine would leak the connection pool / dialect cache for every
interaction. `reply.db` is the shared file for reply-side persistence, so this module owns its own
`Base` and the `research` table while `services/memory/database.py` owns `memory_job` in the same
file, neither seeing the other's metadata. There are no money columns, so no `StoredInteger`, and
no migration mechanism either: `Base.metadata.create_all` creates missing tables and nothing else,
so the model and an already-deployed file have to keep agreeing on the columns. Each call opens an
`AsyncSession` bound to the current `_engine`, so tests can monkeypatch `_engine` per-test.

This module deliberately avoids `from __future__ import annotations`: SQLAlchemy resolves the
`Mapped[datetime]` column annotations at class-definition time, and postponed evaluation breaks
that.
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
    """Applies the project's standard PRAGMA setup to a new reply.db connection.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
    """
    configure_sqlite_connection(dbapi_connection=dbapi_connection)


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Configures a newly opened SQLite connection.

    The decorator binds this to whichever engine existed at import time, so a test that swaps
    `_engine` gets one carrying no listener; `_ensure_schema` and `open_session` re-install it
    through `ensure_sqlite_hooks` for exactly that reason.

    Args:
        dbapi_connection (Any): The connection the pool just opened.
        _connection_record (Any): The pool's bookkeeping record, unused.
    """
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _configure_sqlite_on_checkout(
    dbapi_connection: object, _connection_record: object, _connection_proxy: object
) -> None:
    """Configures pooled connections from test-swapped engines.

    `connect` fires only when the pool opens a NEW connection, so one an engine had already
    pooled before a test swapped it in would stay unconfigured forever; this catches it on its
    way back out.

    Args:
        dbapi_connection (object): The connection being handed to a caller.
        _connection_record (object): The pool's bookkeeping record, unused.
        _connection_proxy (object): The proxy wrapping the checked-out connection, unused.
    """
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


class Base(DeclarativeBase):
    """Base class for the research ORM model (its own metadata, not memory's)."""

    pass


class ResearchSessionRow(Base):
    """One launched deep-research thread.

    Attributes:
        thread_id: Discord thread ID; primary key.
        owner_id: Discord user ID that launched the research.
        channel_id: The text channel the thread hangs off.
        guild_id: Guild the thread lives in. Nullable in the schema, but nothing writes `None`:
            a launch anywhere a nested thread cannot exist is refused before a row is written.
        source_message_id: The message the thread was anchored to.
        agent: The Gemini agent string currently running this session.
        interaction_id: The running interaction's id; `None` between the launch and the
            interaction reporting created, which is the one window a restart cannot resume.
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
    """A research session row read back from `reply.db`.

    Handed to callers instead of the ORM instance, so a read survives its session closing.
    """

    thread_id: int = Field(..., description="Discord thread ID; primary key.")
    owner_id: int = Field(..., description="Discord user ID that launched the research.")
    channel_id: int = Field(..., description="The text channel the research thread hangs off.")
    guild_id: int | None = Field(..., description="Guild the research thread lives in.")
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
    """Bootstraps the `research` table once per engine, and re-arms its connection hooks.

    The hook install runs on every call, ahead of the already-bootstrapped fast path: a test that
    monkeypatches `_engine` gets an engine the import-time `@event.listens_for` never reached,
    which would otherwise open every connection with no PRAGMAs at all. The cache key is engine
    identity rather than a bool for the same reason. The re-check inside the loop-local lock is
    what stops two concurrent first calls both running `create_all`.
    """
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
    """Creates an async session bound to the current reply.db engine.

    Reads `_engine` per call rather than caching a session factory, so a test that swaps the
    module engine is served by the next open; `ensure_sqlite_hooks` keeps the repeat installs
    from stacking listeners. It does NOT create the schema, so a caller that may run before any
    `_ensure_schema` must await that first.

    Returns:
        A session bound to whichever engine the module currently holds.
    """
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    return AsyncSession(bind=_engine, expire_on_commit=False)


def _row_to_model(row: ResearchSessionRow) -> PersistentResearchSession:
    """Maps an ORM row to its pydantic snapshot.

    Args:
        row (ResearchSessionRow): The row just read.

    Returns:
        The snapshot handed back to callers.
    """
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
    """Narrows a stored phase string to `ResearchPhase`, defaulting odd values to failed.

    The column is a plain `String`, so a hand-written or retired value (a row the removed
    escalation tiers left in `planning`) still has to narrow to something, and `failed` is the
    one landing that claims nothing is in flight. Only the typed snapshot is affected: both
    queries filter on the stored column in SQL, so narrowing here never widens what the restart
    sweep resumes or what holds an owner's slot.

    Args:
        value (str): The stored `phase` column.

    Returns:
        The narrowed phase.
    """
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
    """Creates or overwrites the session row for a thread.

    Written once at launch, under the cog's per-owner lock and after the thread exists, so the
    row claims the owner's slot the moment it lands. The conflict branch rewrites every column
    but `created_at`, leaving a re-launched thread its original creation stamp. `interaction_id`
    is `None` on this first write; `set_interaction` fills it in once the run reports created.

    Args:
        thread_id (int): Discord thread ID; the row's primary key.
        owner_id (int): Discord user ID whose one active-research slot this row claims.
        channel_id (int): The text channel the thread hangs off.
        guild_id (int | None): Guild the thread lives in.
        source_message_id (int): The message the thread was anchored to.
        agent (str): The Gemini agent string this session runs on.
        interaction_id (str | None): The running interaction's id, or None before it starts.
        brief (str): The research brief the session was launched with.
        phase (ResearchPhase): Lifecycle phase to record.
    """
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
    """Updates the running interaction id / agent / phase for a thread.

    Called from the agent's `on_created` callback, before the minutes-long wait rather than
    after it: the id is the only handle a restart has, so a run whose id is not persisted by
    then can only be reported as unresumable. A thread with no row absorbs this silently, since
    an UPDATE that matches nothing is not an error.

    Args:
        thread_id (int): Discord thread ID of the session to update.
        interaction_id (str): The interaction id a restart re-attaches to.
        agent (str): The Gemini agent string the run actually started on.
        phase (ResearchPhase): Lifecycle phase to record alongside the id.
    """
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
    """Transitions a session to a new lifecycle phase.

    Moving off `researching` is what frees the owner's one slot and takes the row out of the
    restart sweep, so every terminal path in the cog has to reach this even when the delivery
    itself failed. A thread with no row absorbs the update silently.

    Args:
        thread_id (int): Discord thread ID of the session to transition.
        phase (ResearchPhase): The phase to record.
    """
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
    """Returns sessions still `researching`, for the restart resume sweep.

    The filter is on the stored string in SQL, so a row parked in a phase this store no longer
    knows is never picked up. There is no single-row reader: the sweep wants the whole set, and
    the concurrency guard reads one column, so nothing else needs one.

    Returns:
        Every in-flight session, in whatever order SQLite hands the rows back.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(ResearchSessionRow).where(ResearchSessionRow.phase == "researching")
        )
        return [_row_to_model(row=row) for row in result.scalars().all()]


async def active_thread_for_owner(*, owner_id: int) -> int | None:
    """Returns an owner's in-flight research thread id, or `None` when they have none.

    The concurrency guard: an owner may only have one `researching` session at a time, so a new
    launch is refused while one is active. Selects the id alone rather than the row, since the
    caller only needs a thread to point the refusal at. The check and the claiming write are not
    atomic here; the cog holds a per-owner lock across both.

    Args:
        owner_id (int): Discord user ID to look up.

    Returns:
        The thread id of the owner's in-flight research, or None when they have none.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(ResearchSessionRow.thread_id).where(
                ResearchSessionRow.owner_id == owner_id, ResearchSessionRow.phase == "researching"
            )
        )
        return result.scalars().first()
