"""Tests for the manual balance adjustment script."""

from pathlib import Path
from collections.abc import AsyncIterator

import pytest
from scripts import modify_balance as modify_balance_script
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.cogs._economy import database


@pytest.fixture(autouse=True)
async def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Replaces the economy database with a per-test SQLite file."""
    db_path = tmp_path / "economy.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(database.Base.metadata.create_all)
    monkeypatch.setattr(target=database, name="_engine", value=engine)
    yield
    await engine.dispose()


def test_parse_args_accepts_all_target() -> None:
    """The CLI accepts `all` instead of a numeric Discord user ID."""
    args = modify_balance_script._parse_args(argv=["all", "50000"])

    assert args.target == "all"
    assert args.delta == 50_000


async def test_modify_all_balances_updates_existing_accounts_only() -> None:
    """Bulk adjustment updates only accounts already present in the DB."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    await database.add_balance(user_id=2, name="bob", amount=200)

    result = await modify_balance_script.modify_all_balances(delta=50_000)

    assert len(result.changes) == 2
    assert result.applied_delta == 100_000
    assert all(not change.created for change in result.changes)
    assert await database.get_account(user_id=3) is None

    alice = await database.get_account(user_id=1)
    bob = await database.get_account(user_id=2)
    assert alice is not None
    assert bob is not None
    assert alice[1] == 50_100
    assert bob[1] == 50_200
