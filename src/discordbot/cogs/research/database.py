"""Persistent deep-research session store (`data/database/reply.db`).

One row per launched research thread. The row lets a bot restart resume an
in-flight research: the Gemini interaction runs server-side with `store=True`,
so after a restart the cog reloads the rows still in `researching` and re-attaches
a live stream to each `interaction_id`. It is also the per-user concurrency
guard (one active research per owner).

`reply.db` is shared with other reply-side tables. Each call opens an `AsyncSession`
bound to the current `_engine`, so tests can monkeypatch `_engine` per-test.

This module deliberately avoids `from __future__ import annotations`: SQLAlchemy
resolves the `Mapped[datetime]` column annotations when the class is built, and under
postponed evaluation ruff's TC rules move the annotation-only `datetime` import under
`TYPE_CHECKING`, where the mapper cannot resolve it and the module fails to import.
"""

from typing import Literal
from datetime import datetime
from contextlib import AbstractAsyncContextManager

from pydantic import Field, BaseModel
from sqlalchemy import String, Integer, DateTime, insert, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from discordbot.utils.timezone import as_taipei as _as_taipei
from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.sqlite_config import SqliteBootstrap

# Lifecycle of a research session, persisted in the `phase` column. Any stored value outside
# this set is inert: nothing selects it.
ResearchPhase = Literal["researching", "done", "failed", "cancelled"]

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/reply.db")


class Base(DeclarativeBase):
    """Base class for reply.db ORM models."""

    pass


class ResearchSessionRow(Base):
    """One launched deep-research thread.

    Attributes:
        thread_id (Mapped[int]): Discord thread ID; primary key.
        owner_id (Mapped[int]): Discord user ID that launched the research.
        channel_id (Mapped[int]): The thread's parent channel ID.
        guild_id (Mapped[int | None]): Guild ID; nullable in the schema, never written `None`.
        source_message_id (Mapped[int]): The message the thread was anchored to.
        agent (Mapped[str]): The Gemini agent string running this session.
        interaction_id (Mapped[str | None]): Running interaction ID, or `None` before creation.
        brief (Mapped[str]): Research brief; kept `NOT NULL` for existing `reply.db` tables.
        phase (Mapped[str]): Lifecycle phase (`ResearchPhase`).
        created_at (Mapped[datetime]): First-write timestamp.
        updated_at (Mapped[datetime]): Latest-write timestamp.
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
    """A research session row read back from `reply.db` for restart resume."""

    thread_id: int = Field(..., description="Discord thread ID; primary key.")
    owner_id: int = Field(..., description="Discord user ID that launched the research.")
    agent: str = Field(..., description="The Gemini agent string currently running this session.")
    interaction_id: str | None = Field(
        ..., description="The running interaction's id; None before it starts."
    )


_database = SqliteBootstrap(metadata=Base.metadata)


def open_session() -> AbstractAsyncContextManager[AsyncSession]:
    """Returns an async session context manager on `_engine` with the `research` table ensured."""
    return _database.open_session(engine=_engine)


def _row_to_model(row: ResearchSessionRow) -> PersistentResearchSession:
    """Maps an ORM row to its pydantic snapshot."""
    return PersistentResearchSession(
        thread_id=row.thread_id,
        owner_id=row.owner_id,
        agent=row.agent,
        interaction_id=row.interaction_id,
    )


async def insert_session(  # noqa: PLR0913 -- one row's columns are all per-call inputs
    thread_id: int,
    owner_id: int,
    channel_id: int,
    guild_id: int | None,
    source_message_id: int,
    agent: str,
    brief: str,
) -> None:
    """Records a newly opened thread as `researching` before its interaction ID exists.

    Args:
        thread_id (int): Discord thread ID; primary key.
        owner_id (int): Discord user ID that launched the research.
        channel_id (int): Parent channel ID of the thread.
        guild_id (int | None): Discord guild ID.
        source_message_id (int): Anchor message ID.
        agent (str): Gemini agent identifier.
        brief (str): Research brief text.
    """
    now = _database_now()
    async with open_session() as session:
        await session.execute(
            statement=insert(ResearchSessionRow).values(
                thread_id=thread_id,
                owner_id=owner_id,
                channel_id=channel_id,
                guild_id=guild_id,
                source_message_id=source_message_id,
                agent=agent,
                interaction_id=None,
                brief=brief,
                phase="researching",
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()


async def set_interaction(thread_id: int, interaction_id: str) -> None:
    """Records the running `interaction_id` for `thread_id`.

    Args:
        thread_id (int): Discord thread ID of the session.
        interaction_id (str): Gemini interaction ID to persist.
    """
    now = _database_now()
    async with open_session() as session:
        await session.execute(
            statement=update(ResearchSessionRow)
            .where(ResearchSessionRow.thread_id == thread_id)
            .values(interaction_id=interaction_id, updated_at=now)
        )
        await session.commit()


async def set_phase(thread_id: int, phase: ResearchPhase) -> None:
    """Transitions `thread_id`'s session to `phase`.

    Args:
        thread_id (int): Discord thread ID of the session.
        phase (ResearchPhase): New lifecycle phase.
    """
    now = _database_now()
    async with open_session() as session:
        await session.execute(
            statement=update(ResearchSessionRow)
            .where(ResearchSessionRow.thread_id == thread_id)
            .values(phase=phase, updated_at=now)
        )
        await session.commit()


async def list_resumable(created_before: datetime) -> list[PersistentResearchSession]:
    """Returns sessions launched before `created_before` that are still `researching`.

    Args:
        created_before (datetime): Process start timestamp; excludes sessions launched by the
            current process before `on_ready`.
    """
    async with open_session() as session:
        result = await session.execute(
            statement=select(ResearchSessionRow).where(
                ResearchSessionRow.phase == "researching",
                # SQLite keeps the Taipei wall clock without its offset.
                ResearchSessionRow.created_at < _as_taipei(dt=created_before),
            )
        )
        return [_row_to_model(row=row) for row in result.scalars().all()]


async def active_thread_for_owner(owner_id: int) -> int | None:
    """Returns `owner_id`'s in-flight research thread ID, or `None` when none is active.

    Args:
        owner_id (int): Discord user ID to check.
    """
    async with open_session() as session:
        result = await session.execute(
            statement=select(ResearchSessionRow.thread_id).where(
                ResearchSessionRow.owner_id == owner_id, ResearchSessionRow.phase == "researching"
            )
        )
        return result.scalars().first()
