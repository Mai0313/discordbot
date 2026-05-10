"""Tests for the economy persistence layer."""

from pathlib import Path
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine

from discordbot.cogs._economy import database


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replaces the module-level engine with a per-test SQLite file.

    Each test gets a fresh DB so writes never leak between tests, and the
    real ``data/economy.db`` is left alone.
    """
    db_path = tmp_path / "economy.db"
    engine = create_engine(url=f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    database._Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(target=database, name="_engine", value=engine)
    yield
    engine.dispose()


async def test_add_balance_creates_user() -> None:
    """First write upserts the row and returns the new balance."""
    new = await database.add_balance(user_id=42, name="alice", amount=100)
    assert new == 100
    assert await database.get_balance(user_id=42) == 100


async def test_add_balance_accumulates() -> None:
    """Repeated adds increment the running balance."""
    await database.add_balance(user_id=42, name="alice", amount=100)
    new = await database.add_balance(user_id=42, name="alice", amount=50)
    assert new == 150


async def test_add_balance_zero_is_noop() -> None:
    """Zero or negative amounts must not change the balance."""
    await database.add_balance(user_id=42, name="alice", amount=100)
    assert await database.add_balance(user_id=42, name="alice", amount=0) == 100
    assert await database.add_balance(user_id=42, name="alice", amount=-5) == 100


async def test_add_balance_refreshes_name() -> None:
    """Subsequent writes refresh the cached display name."""
    await database.add_balance(user_id=42, name="alice", amount=10)
    await database.add_balance(user_id=42, name="alice_renamed", amount=10)
    rows = await database.top_n(limit=1)
    assert rows[0][1] == "alice_renamed"


async def test_settle_game_clamps_at_zero() -> None:
    """A loss larger than the balance must clamp the balance at zero."""
    await database.add_balance(user_id=42, name="alice", amount=10)
    new = await database.settle_game(user_id=42, name="alice", delta=-1000)
    assert new == 0


async def test_settle_game_positive_pays_out() -> None:
    """Positive delta credits the account and increments total_earned."""
    await database.add_balance(user_id=42, name="alice", amount=10)
    new = await database.settle_game(user_id=42, name="alice", delta=50)
    assert new == 60


async def test_get_balance_unknown_user_returns_zero() -> None:
    """Reading a never-seen user returns zero, not an error."""
    assert await database.get_balance(user_id=999) == 0


async def test_transfer_moves_points_between_users() -> None:
    """Successful transfer debits sender and credits receiver atomically."""
    await database.add_balance(user_id=1, name="alice", amount=200)
    ok = await database.transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=80
    )
    assert ok is True
    assert await database.get_balance(user_id=1) == 120
    assert await database.get_balance(user_id=2) == 80


async def test_transfer_rejects_self() -> None:
    """Transfers to oneself must be rejected."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    ok = await database.transfer(
        sender_id=1, sender_name="alice", receiver_id=1, receiver_name="alice", amount=10
    )
    assert ok is False
    assert await database.get_balance(user_id=1) == 100


async def test_transfer_rejects_insufficient_balance() -> None:
    """Transfers exceeding the sender's balance must be rejected."""
    await database.add_balance(user_id=1, name="alice", amount=10)
    ok = await database.transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=100
    )
    assert ok is False
    assert await database.get_balance(user_id=1) == 10
    assert await database.get_balance(user_id=2) == 0


@pytest.mark.parametrize(argnames="amount", argvalues=[0, -1, -1000])
async def test_transfer_rejects_non_positive(amount: int) -> None:
    """Transfers with non-positive amounts must be rejected."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    ok = await database.transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=amount
    )
    assert ok is False


async def test_top_n_orders_by_balance_descending() -> None:
    """Leaderboard returns the top accounts ordered by balance."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    await database.add_balance(user_id=2, name="bob", amount=300)
    await database.add_balance(user_id=3, name="carol", amount=50)
    rows = await database.top_n(limit=2)
    assert rows == [(2, "bob", 300), (1, "alice", 100)]


async def test_top_n_excludes_specified_users() -> None:
    """Excluded user IDs (e.g. the bot's house ledger) must not appear in the result."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    await database.add_balance(user_id=2, name="bob", amount=300)
    await database.add_balance(user_id=99, name="house", amount=999)
    rows = await database.top_n(limit=10, exclude_user_ids=(99,))
    assert (99, "house", 999) not in rows
    assert rows[0] == (2, "bob", 300)


async def test_house_settle_allows_negative_balance() -> None:
    """House ledger keeps a true running net even when the dealer is down."""
    await database.house_settle(user_id=99, name="house", delta=-500)
    assert await database.get_balance(user_id=99) == -500


async def test_house_settle_accumulates_gross_flows() -> None:
    """Wins and losses both accumulate gross totals, not just the net balance."""
    await database.house_settle(user_id=99, name="house", delta=200)
    await database.house_settle(user_id=99, name="house", delta=-300)
    account = await database.get_account(user_id=99)
    assert account is not None
    name, balance, total_earned, total_spent = account
    assert name == "house"
    assert balance == -100
    assert total_earned == 200
    assert total_spent == 300


async def test_get_account_returns_none_for_unseen_user() -> None:
    """Unknown users return None instead of a synthetic zero row."""
    assert await database.get_account(user_id=12345) is None
