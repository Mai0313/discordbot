"""Tests for `SqliteBootstrap`, the connection hooks every SQLite database shares.

Each runs on a throwaway schema and engine of its own, so no project database's tables or
seed rows decide the outcome.
"""

from pathlib import Path

from sqlalchemy import insert, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.utils.sqlite_config import SqliteBootstrap
from discordbot.utils.stored_integer import StoredInteger


class _Base(DeclarativeBase):
    """The throwaway schema's own metadata."""


class _Counter(_Base):
    """One integer stored as decimal text, which is what needs the registered functions."""

    __tablename__ = "counter"

    counter_id: Mapped[int] = mapped_column(primary_key=True)
    amount: Mapped[int] = mapped_column(StoredInteger(), nullable=False)


async def test_a_connection_pooled_before_the_hooks_still_gets_the_integer_functions(
    tmp_path: Path,
) -> None:
    """An engine handed over with a connection already in its pool still runs integer SQL.

    That connection never saw `connect`, so only the `checkout` listener can register the
    `StoredInteger` functions on it; without it the update below, which adds through one,
    raises `no such function: discordbot_int_add_text`.
    """
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{tmp_path / 'pooled.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)
        await conn.execute(statement=insert(_Counter).values(counter_id=1, amount=1_000))

    async with SqliteBootstrap().open_session(engine=engine) as session:
        await session.execute(statement=update(_Counter).values(amount=_Counter.amount + 5))
        await session.commit()
        assert await session.scalar(statement=select(_Counter.amount)) == 1_005
    await engine.dispose()
