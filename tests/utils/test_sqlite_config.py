"""Tests for `SqliteBootstrap`, the connection hooks every SQLite database shares.

Each runs on a throwaway schema and engine of its own, so no project database's tables or
seed rows decide the outcome.
"""

from pathlib import Path

import pytest
from sqlalchemy import insert, select, update
from alembic.config import Config
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.utils import sqlite_config
from discordbot.utils.sqlite_config import SqliteBootstrap, ensure_database_upgraded
from discordbot.utils.stored_integer import StoredInteger

from tests.helpers.source_tree import REPO_ROOT


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


def test_ensure_database_upgraded_creates_dir_and_runs_once_per_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ensure_database_upgraded` creates `data/database`, upgrades once, and is idempotent."""
    (tmp_path / "alembic.ini").symlink_to(target=REPO_ROOT / "alembic.ini")
    (tmp_path / "migrations").symlink_to(target=REPO_ROOT / "migrations")
    monkeypatch.chdir(path=tmp_path)

    calls: list[str] = []
    real_upgrade = sqlite_config.command.upgrade

    def counting_upgrade(config: Config, revision: str) -> None:
        calls.append(config.config_ini_section)
        real_upgrade(config=config, revision=revision)

    monkeypatch.setattr(target=sqlite_config.command, name="upgrade", value=counting_upgrade)

    ensure_database_upgraded(section="cleanup")
    ensure_database_upgraded(section="cleanup")
    ensure_database_upgraded(section="economy")

    assert (tmp_path / "data" / "database" / "cleanup.db").is_file()
    assert (tmp_path / "data" / "database" / "economy.db").is_file()
    assert calls == ["cleanup", "economy"]
