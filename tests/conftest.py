"""Shared pytest fixtures.

`economy_isolated_db` lives here instead of being copy-pasted into every
test module that exercises the economy DB.
"""

from pathlib import Path
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.cogs._economy.database import Base, GlobalStateBase
from discordbot.cogs._games.fishing_database import Base as FishingBase


@pytest.fixture
async def economy_isolated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Per-test SQLite file with the full economy schema."""
    economy_db_path = tmp_path / "economy.db"
    global_state_db_path = tmp_path / "global_state.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{economy_db_path}")
    global_state_engine = create_async_engine(url=f"sqlite+aiosqlite:///{global_state_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with global_state_engine.begin() as conn:
        await conn.run_sync(GlobalStateBase.metadata.create_all)
    monkeypatch.setattr("discordbot.cogs._economy.database._engine", engine)
    monkeypatch.setattr(
        "discordbot.cogs._economy.database._global_state_engine", global_state_engine
    )
    yield
    await engine.dispose()
    await global_state_engine.dispose()


@pytest.fixture
async def fishing_isolated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, economy_isolated_db: None
) -> AsyncIterator[None]:
    """Per-test SQLite file with the fishing schema, layered on the economy DB.

    Depends on `economy_isolated_db` so cross-engine wallet writes (rod/bait
    purchases and fish sales) also hit isolated storage.
    """
    fishing_db_path = tmp_path / "fishing.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{fishing_db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(FishingBase.metadata.create_all)
    monkeypatch.setattr("discordbot.cogs._games.fishing_database._engine", engine)
    monkeypatch.setattr("discordbot.cogs._games.fishing_database._schema_ready_for", None)
    yield
    await engine.dispose()
