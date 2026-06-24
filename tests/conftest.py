"""Shared pytest fixtures.

`economy_isolated_db` lives here instead of being copy-pasted into every
test module that exercises the economy DB.
"""

from pathlib import Path
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.cogs._economy.database import Base
from discordbot.cogs._fishing.database import Base as FishingBase
from discordbot.cogs._research.database import Base as ResearchBase


@pytest.fixture
async def economy_isolated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Per-test SQLite file with the full economy schema."""
    economy_db_path = tmp_path / "economy.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{economy_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr("discordbot.cogs._economy.database._engine", engine)
    yield
    await engine.dispose()


@pytest.fixture
async def research_isolated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Per-test SQLite file with the research schema (reply.db)."""
    research_db_path = tmp_path / "reply.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{research_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(ResearchBase.metadata.create_all)
    monkeypatch.setattr("discordbot.cogs._research.database._engine", engine)
    monkeypatch.setattr("discordbot.cogs._research.database._schema_ready_for", None)
    yield
    await engine.dispose()


@pytest.fixture
def memory_isolated_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test memory dir + isolated memory_job DB with reset process-local state."""
    memories_dir = tmp_path / "memories"
    monkeypatch.setattr("discordbot.cogs._memory.store._MEMORY_DIR", memories_dir)
    monkeypatch.setattr("discordbot.cogs._memory.store._cleared_at", {})
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._inflight_tasks", {})
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._pending_updates", {})
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._inflight_loop", None)
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._last_consolidation", {})
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._last_regeneration", {})
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._db_tasks", set())
    # Point the memory_job engine at a throwaway reply.db so no test ever writes the
    # real file: every schedule_memory_update now persists, and those writes are
    # swallowed best-effort, so a missing swap would pass green while polluting the
    # real DB. NullPool closes each connection on return (no async dispose needed in
    # this sync fixture); the schema bootstraps lazily on the first helper call.
    memory_db_engine = create_async_engine(
        url=f"sqlite+aiosqlite:///{tmp_path / 'memory_reply.db'}", poolclass=NullPool
    )
    monkeypatch.setattr("discordbot.cogs._memory.database._engine", memory_db_engine)
    monkeypatch.setattr("discordbot.cogs._memory.database._schema_ready_for", None)
    # _scope_locks, _regeneration_tasks, and the memory semaphore are loop-local
    # helpers that rebuild on the per-test event loop, so they need no manual reset.
    return memories_dir


@pytest.fixture
async def fishing_isolated_db(
    economy_isolated_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Per-test SQLite file with the full fishing schema and an isolated economy DB."""
    fishing_db_path = tmp_path / "fishing.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{fishing_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(FishingBase.metadata.create_all)
    monkeypatch.setattr("discordbot.cogs._fishing.database._engine", engine)
    monkeypatch.setattr("discordbot.cogs._fishing.database._schema_ready_for", None)
    yield
    await engine.dispose()
