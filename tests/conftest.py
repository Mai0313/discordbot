"""Shared pytest fixtures.

`economy_isolated_db` lives here instead of being copy-pasted into every
test module that exercises the economy DB.
"""

from pathlib import Path
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.cogs._economy.database import Base, GlobalStateBase


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
