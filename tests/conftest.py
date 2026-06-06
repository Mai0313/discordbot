"""Shared pytest fixtures.

`economy_isolated_db` lives here instead of being copy-pasted into every
test module that exercises the economy DB.
"""

from pathlib import Path
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.cogs._economy.database import Base
from discordbot.cogs._fishing.database import Base as FishingBase


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
def memory_isolated_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test memory directory with reset process-local memory state."""
    memories_dir = tmp_path / "memories"
    monkeypatch.setattr("discordbot.cogs._memory.store._MEMORY_DIR", memories_dir)
    monkeypatch.setattr("discordbot.cogs._memory.store._user_locks", {})
    monkeypatch.setattr("discordbot.cogs._memory.store._user_locks_loop", None)
    monkeypatch.setattr("discordbot.cogs._memory.store._cleared_at", {})
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._inflight_tasks", {})
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._pending_updates", {})
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._inflight_loop", None)
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._last_consolidation", {})
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._memory_semaphore_obj", None)
    monkeypatch.setattr("discordbot.cogs._memory.pipeline._memory_semaphore_loop", None)
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
