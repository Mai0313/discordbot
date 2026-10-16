"""Tests for `SqliteBootstrap`, the hooks and lazy schema every SQLite database shares.

Each runs on a throwaway schema and engine of its own, so no project database's tables or
seed rows decide the outcome.
"""

import asyncio
from pathlib import Path

from sqlalchemy import text, insert, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from discordbot.utils.sqlite_config import SqliteBootstrap
from discordbot.utils.stored_integer import StoredInteger


class _Base(DeclarativeBase):
    """The throwaway schema's own metadata."""


class _Counter(_Base):
    """One integer stored as decimal text, which is what needs the registered functions."""

    __tablename__ = "counter"

    counter_id: Mapped[int] = mapped_column(primary_key=True)
    amount: Mapped[int] = mapped_column(StoredInteger(), nullable=False)


async def _seed_counter(conn: AsyncConnection) -> None:
    """Seeds the one counter row; a second run on the same file breaks its primary key."""
    await conn.execute(statement=insert(_Counter).values(counter_id=1, amount=1_000))


def _bootstrap() -> SqliteBootstrap:
    """A fresh bootstrap for the throwaway schema, ready for no engine yet."""
    return SqliteBootstrap(metadata=_Base.metadata, after_create=_seed_counter)


async def test_a_connection_pooled_before_the_hooks_still_gets_the_integer_functions(
    tmp_path: Path,
) -> None:
    """An engine handed over with a connection already in its pool still runs integer SQL.

    That connection never saw `connect`, so only the `checkout` listener can register the
    `StoredInteger` functions on it; without it the update below, which adds through one,
    raises `no such function: discordbot_int_add_text`.
    """
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{tmp_path / 'pooled.db'}")
    async with engine.connect() as conn:
        await conn.execute(statement=text(text="SELECT 1"))

    async with _bootstrap().open_session(engine=engine) as session:
        await session.execute(statement=update(_Counter).values(amount=_Counter.amount + 5))
        await session.commit()
        assert await session.scalar(statement=select(_Counter.amount)) == 1_005
    await engine.dispose()


async def test_concurrent_first_use_bootstraps_the_schema_once(tmp_path: Path) -> None:
    """Concurrent first sessions create the tables and run the seed exactly once.

    Every one of them finds the engine not yet ready, and `create_all(checkfirst=True)` is a
    check-then-create, so unserialized they race SQLite's `CREATE TABLE` and repeat the seed.
    """
    engine = create_async_engine(
        url=f"sqlite+aiosqlite:///{tmp_path / 'concurrent.db'}", poolclass=NullPool
    )
    bootstrap = _bootstrap()

    async def read_counter() -> int | None:
        """Reads the seeded counter through a first-use session."""
        async with bootstrap.open_session(engine=engine) as session:
            return await session.scalar(statement=select(_Counter.amount))

    assert await asyncio.gather(*(read_counter() for _ in range(20))) == [1_000] * 20


async def test_a_swapped_engine_gets_its_own_schema(tmp_path: Path) -> None:
    """Readiness follows the engine, so the first session on another one bootstraps it.

    Readiness that ignored which engine it was recorded for would skip the second file's
    schema and its seed row.
    """
    bootstrap = _bootstrap()
    for name in ("first.db", "second.db"):
        engine = create_async_engine(
            url=f"sqlite+aiosqlite:///{tmp_path / name}", poolclass=NullPool
        )
        async with bootstrap.open_session(engine=engine) as session:
            assert await session.scalar(statement=select(_Counter.amount)) == 1_000, name
