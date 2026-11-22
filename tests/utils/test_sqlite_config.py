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

from discordbot.utils import sqlite_config, message_cleanup, expansion_placeholder
from discordbot.cogs.games import database as games_database
from discordbot.cogs.log_msg import cog as log_msg
from discordbot.cogs.research import database as research_database
from discordbot.cogs.gen_reply import ask_store, feedback_store
from discordbot.services.memory import database as memory_database
from discordbot.services.economy import database as economy_database
from discordbot.utils.sqlite_config import SqliteBootstrap, ensure_database_upgraded
from discordbot.utils.stored_integer import StoredInteger

from tests.helpers.source_tree import REPO_ROOT

# Captured at collection time, before function-scoped `conftest.py` isolation swaps `_engine`.
_PRODUCTION_ENGINE_URLS: dict[str, str] = {
    "cleanup": str(message_cleanup._engine.url),
    "economy": str(economy_database._engine.url),
    "games": str(games_database._engine.url),
    "messages": str(log_msg._engine.url),
    "ask": str(ask_store._engine.url),
    "feedback": str(feedback_store._engine.url),
    "research": str(research_database._engine.url),
    "expansion": str(expansion_placeholder._engine.url),
    "memory": str(memory_database._engine.url),
}


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


def test_every_database_engine_points_at_its_expected_sqlite_file() -> None:
    """Pins each store's production SQLite URL before moving URL construction to one helper."""
    assert _PRODUCTION_ENGINE_URLS == {
        "cleanup": "sqlite+aiosqlite:///data/database/cleanup.db",
        "economy": "sqlite+aiosqlite:///data/database/economy.db",
        "games": "sqlite+aiosqlite:///data/database/games.db",
        "messages": "sqlite+aiosqlite:///data/database/messages.db",
        "ask": "sqlite+aiosqlite:///data/database/reply.db",
        "feedback": "sqlite+aiosqlite:///data/database/reply.db",
        "research": "sqlite+aiosqlite:///data/database/reply.db",
        "expansion": "sqlite+aiosqlite:///data/database/reply.db",
        "memory": "sqlite+aiosqlite:///data/database/reply.db",
    }
