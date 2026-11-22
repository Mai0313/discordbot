"""Tests for `SqliteBootstrap`, the connection hooks every SQLite database shares.

Each runs on a throwaway schema and engine of its own, so no project database's tables or
seed rows decide the outcome.
"""

from pathlib import Path
from collections import Counter

import pytest
from sqlalchemy import text, insert, select, update
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
from discordbot.utils.sqlite_config import (
    DATABASE_DIR,
    SqliteBootstrap,
    sqlite_url,
    open_reply_session,
    ensure_database_upgraded,
)
from discordbot.utils.stored_integer import StoredInteger

from tests.helpers.source_tree import REPO_ROOT

# Captured at collection time, before function-scoped `conftest.py` isolation swaps engines.
_PRODUCTION_ENGINE_URLS: dict[str, str] = {
    "cleanup": str(message_cleanup._engine.url),
    "economy": str(economy_database._engine.url),
    "games": str(games_database._engine.url),
    "messages": str(log_msg._engine.url),
    "reply": str(sqlite_config.reply_db_engine.url),
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
    """Pins each database engine's production URL and verifies reply.db stores share one engine."""
    assert Path("./data/database") == DATABASE_DIR
    assert sqlite_url(filename="reply.db") == "sqlite+aiosqlite:///data/database/reply.db"
    assert _PRODUCTION_ENGINE_URLS == {
        "cleanup": "sqlite+aiosqlite:///data/database/cleanup.db",
        "economy": "sqlite+aiosqlite:///data/database/economy.db",
        "games": "sqlite+aiosqlite:///data/database/games.db",
        "messages": "sqlite+aiosqlite:///data/database/messages.db",
        "reply": "sqlite+aiosqlite:///data/database/reply.db",
    }
    for module in (
        ask_store,
        feedback_store,
        research_database,
        expansion_placeholder,
        memory_database,
    ):
        assert not hasattr(module, "_engine")


async def test_checkout_configures_each_pooled_connection_once_and_reconfigures_after_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One-shot checkout skips repeat PRAGMAs on pool reuse and re-applies after invalidation."""
    calls: list[bool] = []
    real_configure = sqlite_config.configure_sqlite_connection

    def counting_configure(dbapi_connection: object, enable_foreign_keys: bool = False) -> None:
        calls.append(enable_foreign_keys)
        real_configure(dbapi_connection=dbapi_connection, enable_foreign_keys=enable_foreign_keys)

    monkeypatch.setattr(
        target=sqlite_config, name="configure_sqlite_connection", value=counting_configure
    )
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{tmp_path / 'oneshot.db'}")
    bootstrap = SqliteBootstrap(enable_foreign_keys=True)

    try:
        async with bootstrap.open_session(engine=engine) as session:
            assert await session.scalar(statement=text(text="PRAGMA foreign_keys")) == 1
        async with bootstrap.open_session(engine=engine) as session:
            assert await session.scalar(statement=text(text="PRAGMA foreign_keys")) == 1
        assert Counter(calls) == Counter({True: 1})

        async with bootstrap.open_session(engine=engine) as session:
            conn = await session.connection()
            await conn.invalidate()
        async with bootstrap.open_session(engine=engine) as session:
            assert await session.scalar(statement=text(text="PRAGMA foreign_keys")) == 1
        assert Counter(calls) == Counter({True: 2})
    finally:
        await engine.dispose()


async def test_open_reply_session_enables_foreign_keys() -> None:
    """The unified `reply.db` engine enforces foreign keys for `feedback_store` constraints."""
    async with open_reply_session() as session:
        assert await session.scalar(statement=text(text="PRAGMA foreign_keys")) == 1
