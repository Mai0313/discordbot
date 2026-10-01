"""Tests for the economy persistence layer."""

from types import SimpleNamespace
from typing import Any, cast
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from collections.abc import Callable, Awaitable

import pytest
from sqlalchemy import text, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from discordbot.utils.timezone import TAIWAN_TIMEZONE, as_taipei, database_now
from discordbot.typings.economy import (
    TRANSFER_TAX_BPS,
    VIP_PURCHASE_COST,
    AccountSnapshot,
    JackpotSnapshot,
    LeaderboardEntry,
    LossLeaderboardEntry,
    BalanceAdjustmentResult,
    JackpotSettlementRequest,
    apply_vip_blackjack_bonus,
)
from discordbot.services.economy import database as economy_database
from discordbot.services.economy.database import (
    UserWallet,
    JackpotPool,
    UserAccount,
    CasinoLedger,
    CasinoAccount,
    CentralBankLedger,
    top_n,
    buy_vip,
    get_vip,
    transfer,
    get_admin,
    set_admin,
    top_losers,
    get_account,
    get_balance,
    open_session,
    adjust_balance,
    _taipei_midnight,
    get_casino_ledger,
    call_personal_loans,
    list_loan_contracts,
    accept_loan_proposal,
    get_jackpot_snapshot,
    repay_personal_loans,
    _commit_balance_write,
    credit_with_repayment,
    call_central_bank_loans,
    get_central_bank_status,
    apply_jackpot_settlement,
    repay_central_bank_loans,
    apply_blackjack_settlement,
    create_personal_loan_request,
    apply_jackpot_settlement_batch,
    _apply_jackpot_delta_in_session,
    create_central_bank_loan_request,
    _apply_daily_casino_delta_in_session,
    invalidate_economy_leaderboard_cache,
)

from tests.helpers.economy import (
    LENDING_GUILD,
    seed_balance,
    approve_as_admin,
    get_jackpot_pool,
    seed_participant,
    open_personal_loan,
    hide_from_leaderboard,
)
from tests.helpers.economy_invariants import (
    assert_wallet_consistent,
    assert_daily_casino_stats,
    assert_casino_ledger_consistent,
)


async def _stored_avatar_url(user_id: int) -> str:
    """Reads the cached avatar URL for one account."""
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.avatar_url).where(UserAccount.user_id == user_id)
        )
        return result.scalar_one()


async def _stored_wallet_name(user_id: int) -> str:
    """Reads the denormalized wallet name for one account."""
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserWallet.name).where(UserWallet.user_id == user_id)
        )
        return result.scalar_one()


async def _casino_account_ids() -> list[int]:
    """Every user id holding a daily casino counter row, sorted."""
    async with open_session() as session:
        result = await session.execute(statement=select(CasinoAccount.user_id))
        return sorted(row[0] for row in result.all())


async def _economy_schema_details() -> tuple[
    set[str], set[str], set[str], dict[str, set[str]], dict[str, dict[str, str]]
]:
    """Reads current economy schema metadata."""
    async with open_session() as session:
        result = await session.execute(
            statement=text(
                text="SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        economy_tables = {row[0] for row in result.all()}
        result = await session.execute(statement=text(text="PRAGMA index_list(user_wallet)"))
        wallet_index_names = {row[1] for row in result.all()}
        result = await session.execute(statement=text(text="PRAGMA index_list(casino_account)"))
        casino_index_names = {row[1] for row in result.all()}
        column_queries = {
            "user_account": "PRAGMA table_info(user_account)",
            "user_wallet": "PRAGMA table_info(user_wallet)",
            "loan_proposal": "PRAGMA table_info(loan_proposal)",
            "loan_contract": "PRAGMA table_info(loan_contract)",
            "casino_account": "PRAGMA table_info(casino_account)",
            "guild_participant": "PRAGMA table_info(guild_participant)",
            "casino_ledger": "PRAGMA table_info(casino_ledger)",
            "central_bank_ledger": "PRAGMA table_info(central_bank_ledger)",
        }
        table_columns: dict[str, set[str]] = {}
        table_column_types: dict[str, dict[str, str]] = {}
        for table_name, query in column_queries.items():
            result = await session.execute(statement=text(text=query))
            table_info = result.all()
            table_columns[table_name] = {row[1] for row in table_info}
            table_column_types[table_name] = {row[1]: row[2] for row in table_info}
    return (
        economy_tables,
        wallet_index_names,
        casino_index_names,
        table_columns,
        table_column_types,
    )


async def _jackpot_schema_details() -> tuple[tuple[int, int, int, int, int], dict[str, str]]:
    """Reads the seeded jackpot row and its column types from the economy DB."""
    async with open_session() as session:
        result = await session.execute(
            statement=select(
                JackpotPool.pool_balance,
                JackpotPool.total_contributed,
                JackpotPool.total_claimed,
                JackpotPool.seeded_amount,
                JackpotPool.generation,
            ).where(JackpotPool.game_id == "dragon_gate")
        )
        jackpot_row = result.one()
        result = await session.execute(statement=text(text="PRAGMA table_info(jackpot_pool)"))
        jackpot_column_types = {row[1]: row[2] for row in result.all()}
    return (cast("tuple[int, int, int, int, int]", tuple(jackpot_row)), jackpot_column_types)


def _assert_money_columns_are_text(
    table_column_types: dict[str, dict[str, str]], jackpot_column_types: dict[str, str]
) -> None:
    """Checks all decimal-string money columns use SQLite TEXT affinity."""
    economy_money_columns = {
        "user_wallet": ("balance", "total_earned", "total_spent"),
        "loan_proposal": ("amount", "escrow_amount"),
        "loan_contract": (
            "original_principal",
            "principal_remaining",
            "interest_due",
            "total_interest_paid",
            "total_principal_paid",
        ),
        "casino_account": ("daily_loss", "daily_win", "daily_net"),
        "casino_ledger": ("balance", "total_earned", "total_spent"),
        "central_bank_ledger": ("balance", "total_earned"),
    }
    for table_name, column_names in economy_money_columns.items():
        for column_name in column_names:
            assert table_column_types[table_name][column_name] == "TEXT"
    for column_name in ("pool_balance", "total_contributed", "total_claimed", "seeded_amount"):
        assert jackpot_column_types[column_name] == "TEXT"


async def test_adjust_balance_zero_is_noop() -> None:
    """Zero deltas do not change balance or lifetime totals."""
    await seed_balance(user_id=42, name="alice", amount=100)
    result = await adjust_balance(user_id=42, name="alice", delta=0)
    assert result == BalanceAdjustmentResult(new_balance=100, applied_delta=0)
    account = await get_account(user_id=42)
    assert account == AccountSnapshot(name="alice", balance=100, total_earned=100, total_spent=0)


async def test_adjust_balance_positive_updates_total_earned() -> None:
    """Positive manual adjustments are counted as earned points."""
    result = await adjust_balance(user_id=42, name="alice", delta=100)
    assert result == BalanceAdjustmentResult(new_balance=100, applied_delta=100)
    account = await get_account(user_id=42)
    assert account == AccountSnapshot(name="alice", balance=100, total_earned=100, total_spent=0)


async def test_adjust_balance_clamps_at_zero() -> None:
    """Negative manual adjustment clamps at zero by default and spends only what it applied."""
    await seed_balance(user_id=42, name="alice", amount=10)
    result = await adjust_balance(user_id=42, name="alice", delta=-1_000)
    assert result == BalanceAdjustmentResult(new_balance=0, applied_delta=-10)
    account = await get_account(user_id=42)
    assert account == AccountSnapshot(name="alice", balance=0, total_earned=10, total_spent=10)


async def test_adjust_balance_negative_missing_user_does_not_create_row() -> None:
    """Clamped negative adjustments to absent users stay no-op reads."""
    result = await adjust_balance(user_id=42, name="alice", delta=-1_000)

    assert result == BalanceAdjustmentResult(new_balance=0, applied_delta=0)
    assert await get_account(user_id=42) is None


async def test_adjust_balance_allows_negative_when_requested() -> None:
    """Manual tooling can explicitly allow a negative resulting balance."""
    await seed_balance(user_id=42, name="alice", amount=10)
    result = await adjust_balance(user_id=42, name="alice", delta=-500, allow_negative=True)
    assert result == BalanceAdjustmentResult(new_balance=-490, applied_delta=-500)


async def test_adjust_balance_refreshes_name() -> None:
    """Subsequent writes refresh the cached display name."""
    await seed_balance(user_id=42, name="alice", amount=10)
    await seed_balance(user_id=42, name="alice_renamed", amount=10)
    rows = await top_n(limit=1)
    assert rows[0].name == "alice_renamed"
    assert rows[0].avatar_url == ""
    assert await _stored_wallet_name(user_id=42) == "alice_renamed"


async def test_adjust_balance_stores_and_refreshes_avatar_url() -> None:
    """Subsequent writes refresh the cached avatar URL."""
    await seed_balance(user_id=42, name="alice", amount=10, avatar_url="https://cdn.example/a.png")
    assert await _stored_avatar_url(user_id=42) == "https://cdn.example/a.png"

    await seed_balance(user_id=42, name="alice", amount=10, avatar_url="https://cdn.example/b.png")
    assert await _stored_avatar_url(user_id=42) == "https://cdn.example/b.png"


async def test_admin_flag_defaults_to_false() -> None:
    """Unknown users and normal accounts are not economy admins."""
    assert await get_admin(user_id=42) is False
    await seed_balance(user_id=42, name="alice", amount=10)
    assert await get_admin(user_id=42) is False


async def test_leaderboard_hidden_flag_defaults_to_false() -> None:
    """New accounts are visible on public leaderboards by default."""
    await seed_balance(user_id=42, name="alice", amount=10)
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.hide_from_leaderboard).where(UserAccount.user_id == 42)
        )
    assert result.scalar_one() is False


async def test_set_admin_creates_user() -> None:
    """Granting admin creates a zero-balance account row."""
    applied = await set_admin(user_id=42, name="alice", is_admin=True)
    assert applied is True
    assert await get_admin(user_id=42) is True
    assert await get_balance(user_id=42) == 0


async def test_set_admin_revokes_existing_user() -> None:
    """Revoking admin clears the flag on an existing account."""
    await set_admin(user_id=42, name="alice", is_admin=True)
    applied = await set_admin(user_id=42, name="alice", is_admin=False)
    assert applied is True
    assert await get_admin(user_id=42) is False


async def test_set_admin_revoke_missing_user_noops() -> None:
    """Revoking a missing user does not create an account row."""
    applied = await set_admin(user_id=42, name="alice", is_admin=False)
    assert applied is False
    assert await get_account(user_id=42) is None


async def test_write_timestamps_use_taiwan_local_time() -> None:
    """Account timestamps are persisted as Taiwan-local wall time."""
    before = datetime.now(tz=TAIWAN_TIMEZONE).replace(tzinfo=None)
    await credit_with_repayment(user_id=42, name="alice", amount=10)
    after = datetime.now(tz=TAIWAN_TIMEZONE).replace(tzinfo=None)

    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.updated_at).where(UserAccount.user_id == 42)
        )
        updated_at = result.scalar_one()

    assert before <= updated_at <= after


def test_every_test_gets_its_own_ledger(tmp_path: Path) -> None:
    """A test that asks for no isolation still cannot reach the deployed `economy.db`.

    It requests nothing but `tmp_path`, so dropping the autouse from `economy_isolated_db`
    fails here, before any ledger call, instead of letting the next unpatched one write the
    live file. The engine is read off the module because the fixture swaps it per test.
    """
    assert economy_database._engine.url.database == str(tmp_path / "economy.db")


async def test_ensure_schema_bootstraps_current_databases() -> None:
    """A clean startup's first session creates only the current economy tables."""
    (
        economy_tables,
        wallet_index_names,
        casino_index_names,
        table_columns,
        table_column_types,
    ) = await _economy_schema_details()
    jackpot_row, jackpot_column_types = await _jackpot_schema_details()
    assert economy_tables == {
        "user_account",
        "user_wallet",
        "guild_participant",
        "loan_proposal",
        "loan_contract",
        "casino_account",
        "jackpot_pool",
        "casino_ledger",
        "central_bank_ledger",
    }
    assert table_columns["guild_participant"] == {"guild_id", "user_id", "updated_at"}
    assert {"user_id", "name", "is_central_banker"} <= table_columns["user_account"]
    assert {"user_id", "name", "balance", "total_earned", "total_spent"} <= table_columns[
        "user_wallet"
    ]
    assert {"balance", "total_earned", "total_spent"}.isdisjoint(table_columns["user_account"])
    assert {"borrower_id", "borrower_name", "lender_id", "lender_name"} <= table_columns[
        "loan_proposal"
    ]
    assert {"borrower_id", "borrower_name", "lender_type"} <= table_columns["loan_contract"]
    _assert_money_columns_are_text(
        table_column_types=table_column_types, jackpot_column_types=jackpot_column_types
    )
    assert "ix_user_wallet_balance" in wallet_index_names
    assert "ix_casino_account_day_loss" in casino_index_names
    assert jackpot_row == (1_000, 0, 0, 1_000, 0)

    await seed_balance(
        user_id=42, name="alice", amount=5, avatar_url="https://cdn.example/avatar.png"
    )
    assert await _stored_avatar_url(user_id=42) == "https://cdn.example/avatar.png"
    assert await _stored_wallet_name(user_id=42) == "alice"
    account = await get_account(user_id=42)
    assert account == AccountSnapshot(name="alice", balance=5, total_earned=5, total_spent=0)


async def test_a_connection_pooled_before_the_hooks_still_gets_the_integer_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine handed over with a connection already in its pool still settles money.

    That connection never saw `connect`, so only the `checkout` listener can register the
    `StoredInteger` functions on it; without it the settlement below, which folds into the
    daily casino counters through one, raises `no such function: discordbot_int_add_text`.
    """
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{tmp_path / 'pooled-economy.db'}")
    async with engine.connect() as conn:
        await conn.execute(statement=text(text="SELECT 1"))
    monkeypatch.setattr("discordbot.services.economy.database._engine", engine)

    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=5, casino_delta=-5
    )

    assert await get_balance(user_id=1) == 5
    await engine.dispose()


async def test_ensure_schema_serializes_concurrent_first_use() -> None:
    """Concurrent first-use schema bootstrap does not race SQLite CREATE TABLE."""
    await asyncio.gather(*(get_balance(user_id=42) for _ in range(20)))

    async with open_session() as session:
        result = await session.execute(
            statement=text(
                text="SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'loan_proposal'"
            )
        )
        assert result.scalar_one_or_none() == "loan_proposal"
        result = await session.execute(
            statement=select(JackpotPool.pool_balance).where(JackpotPool.game_id == "dragon_gate")
        )
        assert result.scalar_one() == 1_000


async def test_a_swapped_engine_gets_its_own_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readiness follows the engine, so the first session on a swapped-in one bootstraps it.

    The fixture's engine is bootstrapped before the swap, so readiness that ignored which engine
    it was recorded for would skip the new file's schema and its seed rows.
    """
    await seed_balance(user_id=42, name="alice", amount=5)
    swapped = create_async_engine(url=f"sqlite+aiosqlite:///{tmp_path / 'swapped-economy.db'}")
    monkeypatch.setattr("discordbot.services.economy.database._engine", swapped)

    assert await get_balance(user_id=42) == 0
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000
    await swapped.dispose()


async def test_get_balance_unknown_user_returns_zero() -> None:
    """Reading a never-seen user returns zero, not an error."""
    assert await get_balance(user_id=999) == 0


@pytest.mark.parametrize(
    argnames=("sender_start", "amount"), argvalues=[(200, 80), (1_000, 1_000), (10_000, 2_500)]
)
async def test_transfer_taxes_and_preserves_invariant(sender_start: int, amount: int) -> None:
    """A transfer debits the sender in full, burns the tax, and credits the taxed net.

    The burned tax is derived from TRANSFER_TAX_BPS rather than hardcoded, and both wallets keep
    the balance == total_earned - total_spent identity, so the burn truly leaves circulation.
    """
    await seed_balance(user_id=1, name="alice", amount=sender_start)
    result = await transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=amount
    )
    assert result is not None
    expected_tax = amount * TRANSFER_TAX_BPS // 10_000
    assert result.tax_amount == expected_tax
    assert result.received_amount == amount - expected_tax

    sender = await assert_wallet_consistent(user_id=1, expected_balance=sender_start - amount)
    receiver = await assert_wallet_consistent(user_id=2, expected_balance=amount - expected_tax)
    # Sender spent the full amount; receiver earned only the net, so the tax is out of circulation.
    assert sender.total_spent == amount
    assert receiver.total_earned == amount - expected_tax


@pytest.mark.parametrize(
    argnames=("sender_start", "receiver_id", "amount"),
    argvalues=[(100, 1, 10), (10, 2, 100)],
    ids=["self-transfer", "insufficient-balance"],
)
async def test_transfer_rejects_invalid(sender_start: int, receiver_id: int, amount: int) -> None:
    """A self-transfer or an over-balance transfer is rejected and leaves the sender untouched."""
    await seed_balance(user_id=1, name="alice", amount=sender_start)
    result = await transfer(
        sender_id=1,
        sender_name="alice",
        receiver_id=receiver_id,
        receiver_name="bob",
        amount=amount,
    )
    assert result is None
    assert await get_balance(user_id=1) == sender_start


async def test_transfer_prevents_concurrent_double_spend() -> None:
    """Concurrent transfers from one sender cannot reuse the same points."""
    await seed_balance(user_id=1, name="alice", amount=100)
    results = await asyncio.gather(
        transfer(sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=80),
        transfer(
            sender_id=1, sender_name="alice", receiver_id=3, receiver_name="carol", amount=80
        ),
    )
    assert sum(result is not None for result in results) == 1
    assert results.count(None) == 1
    assert await get_balance(user_id=1) == 20
    # Whichever transfer won, the receiver nets 80 minus the 5% (4) tax burn.
    assert await get_balance(user_id=2) + await get_balance(user_id=3) == 76


async def test_transfer_concurrent_credits_accumulate() -> None:
    """Concurrent transfers into one receiver must not lose either credit."""
    await seed_balance(user_id=1, name="alice", amount=100)
    await seed_balance(user_id=2, name="bob", amount=100)
    results = await asyncio.gather(
        transfer(
            sender_id=1, sender_name="alice", receiver_id=3, receiver_name="carol", amount=80
        ),
        transfer(sender_id=2, sender_name="bob", receiver_id=3, receiver_name="carol", amount=70),
    )
    assert all(result is not None for result in results)
    assert {result.sender_balance for result in results if result is not None} == {20, 30}
    # 80 nets 76 and 70 nets 67 after the 5% burn; both credits accumulate to 143.
    assert max(result.receiver_balance for result in results if result is not None) == 143
    assert await get_balance(user_id=3) == 143


@pytest.mark.parametrize(argnames="amount", argvalues=[0, -1, -1000])
async def test_transfer_rejects_non_positive(amount: int) -> None:
    """Transfers with non-positive amounts must be rejected."""
    await seed_balance(user_id=1, name="alice", amount=100)
    result = await transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=amount
    )
    assert result is None


async def test_top_n_orders_by_balance_descending() -> None:
    """Leaderboard returns the top accounts ordered by balance."""
    await seed_balance(user_id=1, name="alice", amount=100, avatar_url="https://cdn/a.png")
    await seed_balance(user_id=2, name="bob", amount=300, avatar_url="https://cdn/b.png")
    await seed_balance(user_id=3, name="carol", amount=50)
    rows = await top_n(limit=2)
    assert rows == [
        LeaderboardEntry(user_id=2, name="bob", balance=300, avatar_url="https://cdn/b.png"),
        LeaderboardEntry(user_id=1, name="alice", balance=100, avatar_url="https://cdn/a.png"),
    ]


async def test_top_n_excludes_leaderboard_hidden_accounts_by_default() -> None:
    """Accounts marked hidden do not appear on the public balance leaderboard."""
    await seed_balance(user_id=1, name="alice", amount=100)
    await seed_balance(user_id=2, name="bob", amount=300)
    await seed_balance(user_id=3, name="carol", amount=200)
    await hide_from_leaderboard(user_id=2)

    rows = await top_n(limit=2)
    assert rows == [
        LeaderboardEntry(user_id=3, name="carol", balance=200, avatar_url=""),
        LeaderboardEntry(user_id=1, name="alice", balance=100, avatar_url=""),
    ]


async def test_top_n_can_include_leaderboard_hidden_accounts() -> None:
    """Maintenance callers can still enumerate hidden accounts when needed."""
    await seed_balance(user_id=1, name="alice", amount=100)
    await seed_balance(user_id=2, name="bob", amount=300)
    await hide_from_leaderboard(user_id=2)

    rows = await top_n(limit=2, include_hidden=True)
    assert rows[0] == LeaderboardEntry(user_id=2, name="bob", balance=300, avatar_url="")


async def test_top_n_none_limit_returns_all_matching_accounts() -> None:
    """Maintenance callers can request every matching account without a sentinel limit."""
    await seed_balance(user_id=1, name="alice", amount=100)
    await seed_balance(user_id=2, name="bob", amount=300)
    await seed_balance(user_id=3, name="carol", amount=200)

    rows = await top_n(limit=None)
    assert [row.user_id for row in rows] == [2, 3, 1]


async def test_top_n_db_order_handles_large_zero_and_negative_balances() -> None:
    """DB-side ordering keeps decimal-text balances in numeric order."""
    await adjust_balance(user_id=1, name="huge", delta=10**30)
    await adjust_balance(user_id=2, name="small", delta=999)
    await adjust_balance(user_id=3, name="zero", delta=0)
    await adjust_balance(user_id=4, name="minus_one", delta=-1, allow_negative=True)
    await adjust_balance(user_id=5, name="minus_ten", delta=-10, allow_negative=True)
    await adjust_balance(user_id=6, name="minus_two", delta=-2, allow_negative=True)

    rows = await top_n(limit=None)

    assert [(row.user_id, row.balance) for row in rows] == [
        (1, 10**30),
        (2, 999),
        (4, -1),
        (6, -2),
        (5, -10),
    ]


async def test_top_n_short_cache_hit_and_manual_invalidation() -> None:
    """Repeated leaderboard reads use cached rows until explicitly invalidated."""
    await seed_balance(user_id=1, name="alice", amount=100)
    await seed_balance(user_id=2, name="bob", amount=50)

    assert [row.user_id for row in await top_n(limit=1)] == [1]
    async with open_session() as session:
        await session.execute(
            statement=update(UserWallet)
            .where(UserWallet.user_id == 2)
            .values(balance=1_000, total_earned=1_000, total_spent=0)
        )
        await session.commit()

    assert [row.user_id for row in await top_n(limit=1)] == [1]
    invalidate_economy_leaderboard_cache()
    assert [row.user_id for row in await top_n(limit=1)] == [2]


async def test_a_balance_write_clears_the_leaderboard_caches_only_once_committed() -> None:
    """A clear before the commit lets a read in between cache the rows the write replaces."""
    await top_n(limit=None)
    cached_at_commit: list[bool] = []

    async def commit() -> None:
        """Notes whether the leaderboard rows were still cached when the write committed."""
        cached_at_commit.append(bool(economy_database._top_n_cache))

    await _commit_balance_write(session=cast("AsyncSession", SimpleNamespace(commit=commit)))

    assert cached_at_commit == [True]
    assert economy_database._top_n_cache == {}


async def _ledger_every_write_path_can_touch() -> int:
    """Seeds what every leaderboard write below needs and returns a pending request's id.

    alice (1) can afford VIP and owes both bob (2) and the central bank; bob has asked
    alice for a loan she has not answered yet.
    """
    await seed_participant(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    await seed_balance(user_id=2, name="bob", amount=1_000)
    await open_personal_loan(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=100
    )
    central = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=100
    )
    assert central is not None
    assert (
        await approve_as_admin(proposal_id=central.proposal_id, actor_id=99, name="banker")
        is not None
    )
    pending = await create_personal_loan_request(
        borrower_id=2, borrower_name="bob", lender_id=1, lender_name="alice", amount=100
    )
    assert pending is not None
    return pending.proposal_id


@pytest.mark.parametrize(
    argnames="write",
    argvalues=[
        pytest.param(
            lambda _: credit_with_repayment(user_id=2, name="bob", amount=10),
            id="credit_with_repayment",
        ),
        pytest.param(
            lambda _: adjust_balance(user_id=2, name="bob", delta=10), id="adjust_balance"
        ),
        pytest.param(
            lambda _: apply_blackjack_settlement(
                player_id=2, player_account_name="bob", player_delta=10, casino_delta=-10
            ),
            id="apply_blackjack_settlement",
        ),
        pytest.param(
            lambda _: apply_jackpot_settlement(
                player_id=2, player_account_name="bob", player_delta=-10, game_id="dragon_gate"
            ),
            id="apply_jackpot_settlement",
        ),
        pytest.param(lambda _: buy_vip(user_id=1, name="alice"), id="buy_vip"),
        pytest.param(
            lambda _: transfer(
                sender_id=2, sender_name="bob", receiver_id=1, receiver_name="alice", amount=100
            ),
            id="transfer",
        ),
        pytest.param(
            lambda proposal_id: accept_loan_proposal(
                proposal_id=proposal_id, actor_id=1, actor_name="alice"
            ),
            id="accept_loan_proposal",
        ),
        pytest.param(
            lambda _: repay_personal_loans(
                borrower_id=1, borrower_name="alice", lender_id=2, amount=10
            ),
            id="repay_personal_loans",
        ),
        pytest.param(
            lambda _: call_personal_loans(
                lender_id=2, borrower_id=1, borrower_name="alice", amount=10
            ),
            id="call_personal_loans",
        ),
        pytest.param(
            lambda _: repay_central_bank_loans(borrower_id=1, borrower_name="alice", amount=10),
            id="repay_central_bank_loans",
        ),
        pytest.param(
            lambda _: call_central_bank_loans(
                guild_id=LENDING_GUILD, borrower_id=1, borrower_name="alice", amount=10
            ),
            id="call_central_bank_loans",
        ),
    ],
)
async def test_every_balance_write_invalidates_the_leaderboard_cache(
    write: Callable[[int], Awaitable[object]],
) -> None:
    """A leaderboard read right after any public balance write shows the write."""
    pending_proposal_id = await _ledger_every_write_path_can_touch()
    cached = await top_n(limit=None)

    await write(pending_proposal_id)
    after = await top_n(limit=None)
    invalidate_economy_leaderboard_cache()

    assert after == await top_n(limit=None)
    # Otherwise the write moved no balance and the check above proves nothing.
    assert after != cached


async def test_apply_blackjack_settlement_casino_accumulates_gross_flows() -> None:
    """Wins and losses both accumulate gross totals, not just the net balance."""
    await seed_balance(user_id=1, name="alice", amount=200)
    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-200, casino_delta=200
    )
    await apply_blackjack_settlement(
        player_id=2, player_account_name="bob", player_delta=300, casino_delta=-300
    )
    ledger = await get_casino_ledger()
    assert ledger.balance == -100
    assert ledger.total_earned == 200
    assert ledger.total_spent == 300


async def test_get_account_returns_none_for_unseen_user() -> None:
    """Unknown users return None instead of a synthetic zero row."""
    assert await get_account(user_id=12345) is None


async def test_adjust_balance_concurrent_credits_accumulate() -> None:
    """Verifies that concurrent credits on the same user do not lose updates."""
    await seed_balance(user_id=42, name="alice", amount=100)
    await asyncio.gather(*[seed_balance(user_id=42, name="alice", amount=10) for _ in range(20)])
    assert await get_balance(user_id=42) == 300


async def test_adjust_balance_concurrent_first_sight_does_not_raise() -> None:
    """Verifies that concurrent first-sight credits merge instead of racing."""
    results = await asyncio.gather(*[
        seed_balance(user_id=42, name="alice", amount=10) for _ in range(8)
    ])
    assert all(isinstance(value, int) for value in results)
    assert await get_balance(user_id=42) == 80


async def test_apply_blackjack_settlement_concurrent_credits_accumulate() -> None:
    """Concurrent positive settlements on the same user must not lose updates."""
    await seed_balance(user_id=42, name="alice", amount=100)
    await asyncio.gather(*[
        apply_blackjack_settlement(
            player_id=42, player_account_name="alice", player_delta=10, casino_delta=-10
        )
        for _ in range(10)
    ])
    assert await get_balance(user_id=42) == 200


async def test_apply_blackjack_settlement_concurrent_casino_updates_accumulate() -> None:
    """Verifies that concurrent casino ledger settlements accumulate."""
    for user_id in range(10):
        await seed_balance(user_id=user_id, name=f"player{user_id}", amount=10)
    await asyncio.gather(*[
        apply_blackjack_settlement(
            player_id=user_id,
            player_account_name=f"player{user_id}",
            player_delta=-10,
            casino_delta=10,
        )
        for user_id in range(10)
    ])
    ledger = await get_casino_ledger()
    assert ledger.balance == 100
    assert ledger.total_earned == 100
    assert ledger.total_spent == 0


async def test_apply_blackjack_settlement_is_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A casino mirror that fails takes the player's side of the round down with it.

    The player is written first, so only one shared transaction stops that write from
    committing on its own when the house side raises.
    """
    await seed_balance(user_id=1, name="alice", amount=100)

    async def failing_casino_mirror(**_kwargs: object) -> int:
        """Fails the house side after the player side has been written."""
        raise RuntimeError("forced casino failure")

    monkeypatch.setattr(
        "discordbot.services.economy.database._apply_casino_ledger_delta_in_session",
        failing_casino_mirror,
    )

    with pytest.raises(expected_exception=RuntimeError, match="forced casino failure"):
        await apply_blackjack_settlement(
            player_id=1, player_account_name="alice", player_delta=40, casino_delta=-40
        )

    await assert_wallet_consistent(user_id=1, expected_balance=100)
    await assert_casino_ledger_consistent(expected_balance=0)
    assert await _casino_account_ids() == []


async def test_apply_blackjack_settlement_loss_debits_player_and_casino() -> None:
    """A loss debits the player and credits the casino, and both sides book it in their totals."""
    await seed_balance(user_id=1, name="alice", amount=100)

    result = await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-40, casino_delta=40
    )
    assert result.player_balance == 60
    assert result.casino_balance == 40
    account = await get_account(user_id=1)
    ledger = await get_casino_ledger()
    assert account is not None
    assert (account.balance, account.total_earned, account.total_spent) == (60, 100, 40)
    assert (ledger.balance, ledger.total_earned, ledger.total_spent) == (40, 40, 0)


async def test_apply_blackjack_settlement_loss_clamps_player_and_casino_to_available_balance() -> (
    None
):
    """Deferred settlement stops at zero and only credits the casino with actual debit."""
    await seed_balance(user_id=1, name="alice", amount=25)

    result = await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-40, casino_delta=40
    )

    assert result.player_balance == 0
    assert result.casino_balance == 25
    assert result.applied_player_delta == -25
    account = await get_account(user_id=1)
    assert account is not None
    assert account.total_spent == 25


async def test_a_positive_adjustment_is_never_clamped() -> None:
    """A credit always applies in full, on a new account and on an existing one.

    The admin panel reads `applied_delta != delta` as "the collection hit the balance
    floor", which is only honest while a credit cannot differ from what was asked. Put a
    ceiling on balances and that footnote starts appearing on refunds.
    """
    first = await adjust_balance(user_id=1, name="alice", delta=100)
    second = await adjust_balance(user_id=1, name="alice", delta=250)

    assert first.applied_delta == 100
    assert second.applied_delta == 250
    assert second.new_balance == 350


async def test_apply_blackjack_settlement_books_the_whole_take_when_the_loss_collects() -> None:
    """A system-funded bonus rides inside `player_delta` and must not shrink the house's take.

    The bonus is already added back into the player's net, so capping the ledger at that net
    deducts money the casino never paid out — on a round where the loss collected in full.
    """
    await seed_balance(user_id=1, name="alice", amount=500)

    result = await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-50, casino_delta=150
    )

    assert result.player_balance == 450
    assert result.casino_balance == 150


async def test_apply_blackjack_settlement_books_the_bonus_even_when_the_loss_is_short() -> None:
    """The two rules meet here, and this is the only case where the arithmetic can differ.

    The wallet cannot cover the loss AND a system-funded bonus rides in the player's net, so
    the ledger must lose the shortfall and keep the bonus.
    """
    await seed_balance(user_id=1, name="alice", amount=20)

    result = await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-50, casino_delta=150
    )

    assert result.player_balance == 0
    assert result.casino_balance == 120


async def test_apply_blackjack_settlement_updates_daily_casino_counters() -> None:
    """Blackjack-style player settlements persist gross loss, gross win, and net."""
    await seed_balance(user_id=1, name="alice", amount=1_000)

    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-300, casino_delta=300
    )
    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=500, casino_delta=-500
    )

    await assert_daily_casino_stats(user_id=1, loss=300, win=500, net=200)
    async with open_session() as session:
        result = await session.execute(
            statement=select(CasinoAccount.day_started_at).where(CasinoAccount.user_id == 1)
        )
        day_started_at = result.scalar_one()
    assert day_started_at is not None
    assert as_taipei(dt=day_started_at) == _taipei_midnight(now=database_now())


async def test_daily_casino_counters_store_large_values_as_text() -> None:
    """Casino counters can exceed SQLite's INTEGER range without becoming REAL."""
    await seed_balance(user_id=1, name="alice", amount=1)
    large_loss = 10**20

    async with open_session() as session:
        now = database_now()
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=1, name="alice", delta=-large_loss, now=now
        )
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=1, name="alice", delta=-7, now=now
        )
        await session.commit()

    async with open_session() as session:
        result = await session.execute(
            statement=text(
                text="""
                SELECT daily_loss, typeof(daily_loss), daily_win, typeof(daily_win), daily_net, typeof(daily_net)
                  FROM casino_account
                 WHERE user_id = 1
                """
            )
        )
        counter_row = result.one()

    rows = await top_losers(limit=10)

    assert counter_row == (
        str(large_loss + 7),
        "text",
        "0",
        "text",
        str(-(large_loss + 7)),
        "text",
    )
    assert rows == [
        LossLeaderboardEntry(user_id=1, name="alice", loss_amount=large_loss + 7, avatar_url="")
    ]


async def test_wallet_and_jackpot_store_large_values_as_text() -> None:
    """Core wallet and jackpot money columns can exceed SQLite's INTEGER range."""
    large_amount = 10**20

    await adjust_balance(user_id=1, name="alice", delta=large_amount)
    result = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=-large_amount, game_id="dragon_gate"
    )

    async with open_session() as session:
        wallet_result = await session.execute(
            statement=text(
                text="""
                SELECT balance, typeof(balance), total_earned, typeof(total_earned), total_spent, typeof(total_spent)
                  FROM user_wallet
                 WHERE user_id = 1
                """
            )
        )
        wallet_row = wallet_result.one()
        jackpot_result = await session.execute(
            statement=text(
                text="""
                SELECT pool_balance, typeof(pool_balance), total_contributed, typeof(total_contributed)
                  FROM jackpot_pool
                 WHERE game_id = 'dragon_gate'
                """
            )
        )
        jackpot_row = jackpot_result.one()

    assert result.player_balance == 0
    assert result.applied_player_delta == -large_amount
    assert wallet_row == ("0", "text", str(large_amount), "text", str(large_amount), "text")
    assert jackpot_row == (str(1_000 + large_amount), "text", str(large_amount), "text")


async def test_daily_casino_counters_skip_push_and_house_ledger() -> None:
    """Zero deltas and dealer ledger mirrors do not enter player loss counters."""
    await seed_balance(user_id=1, name="alice", amount=100)

    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=0, casino_delta=0
    )
    assert await _casino_account_ids() == []

    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-40, casino_delta=40
    )
    # The player's own loss lands. The casino's mirrored +40 goes to `casino_ledger`, so it
    # neither becomes a win here nor opens a counter row under some house id of its own.
    await assert_daily_casino_stats(user_id=1, loss=40, win=0, net=-40)
    assert await _casino_account_ids() == [1]


# credit_with_repayment -----------------------------------------------------


async def test_credit_with_repayment_zero_amount_is_noop() -> None:
    """Non-positive reward calls do not create phantom income."""
    await seed_balance(user_id=1, name="alice", amount=50)

    result = await credit_with_repayment(user_id=1, name="alice", amount=0)

    assert result.new_balance == 50
    account = await get_account(user_id=1)
    assert account is not None
    assert account.total_earned == 50


async def test_credit_with_repayment_first_sight_creates_row() -> None:
    """A first reward creates the user account row."""
    result = await credit_with_repayment(user_id=1, name="alice", amount=200)

    assert result.new_balance == 200
    assert await get_balance(user_id=1) == 200


async def test_credit_with_repayment_concurrent_credits_accumulate() -> None:
    """Concurrent reward writes add up instead of losing one update."""
    await asyncio.gather(
        *(credit_with_repayment(user_id=1, name="alice", amount=10) for _ in range(20))
    )

    assert await get_balance(user_id=1) == 200


async def test_credit_with_repayment_does_not_touch_long_term_debt() -> None:
    """Passive income does not auto-repay explicit long-term loan contracts."""
    await seed_balance(user_id=2, name="bob", amount=1_000)
    await open_personal_loan(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=500
    )

    result = await credit_with_repayment(user_id=1, name="alice", amount=100)
    contracts = await list_loan_contracts(user_id=1)

    assert result.new_balance == 600
    assert len(contracts) == 1
    assert contracts[0].principal_remaining == 500


# VIP purchase --------------------------------------------------------------


@pytest.mark.parametrize(
    argnames=("delta", "is_vip", "expected"),
    argvalues=[
        (100, False, 100),
        (100, True, 120),
        (101, True, 121),
        (1, True, 1),
        (0, True, 0),
        (-50, True, -50),
    ],
)
def test_apply_vip_blackjack_bonus(delta: int, is_vip: bool, expected: int) -> None:
    """VIP bonus applies only to positive winnings and floors fractional fifths."""
    assert apply_vip_blackjack_bonus(delta=delta, is_vip=is_vip) == expected


async def test_buy_vip_sets_flag_and_debits_balance() -> None:
    """A successful purchase costs `VIP_PURCHASE_COST` and flips `is_vip`."""
    await seed_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST + 100)
    result = await buy_vip(user_id=1, name="alice")
    assert result is not None
    assert result.new_balance == 100
    assert result.cost == VIP_PURCHASE_COST
    assert await get_vip(user_id=1) is True


async def test_buy_vip_rejects_insufficient_balance() -> None:
    """Users without enough points cannot purchase VIP."""
    await seed_balance(user_id=1, name="alice", amount=100)
    result = await buy_vip(user_id=1, name="alice")
    assert result is None
    assert await get_vip(user_id=1) is False


async def test_buy_vip_rejects_existing_vip() -> None:
    """A second purchase by an existing VIP returns None and does not re-debit."""
    await seed_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST * 2)
    first = await buy_vip(user_id=1, name="alice")
    assert first is not None
    second = await buy_vip(user_id=1, name="alice")
    assert second is None
    assert await get_balance(user_id=1) == VIP_PURCHASE_COST


async def test_buy_vip_rejects_unseen_user() -> None:
    """A user without a row cannot purchase (no balance to debit)."""
    assert await buy_vip(user_id=999, name="ghost") is None


async def test_buy_vip_updates_lifetime_spent() -> None:
    """A successful purchase counts as spent points."""
    await seed_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    await buy_vip(user_id=1, name="alice")
    account = await get_account(user_id=1)
    assert account == AccountSnapshot(
        name="alice", balance=0, total_earned=VIP_PURCHASE_COST, total_spent=VIP_PURCHASE_COST
    )


async def test_get_vip_unknown_user_returns_false() -> None:
    """Unknown users report no VIP perk rather than raising."""
    assert await get_vip(user_id=12345) is False


# Loss leaderboard ----------------------------------------------------------


async def test_top_losers_uses_gross_loss_not_net() -> None:
    """Winning later does not erase a player's gross loss leaderboard amount."""
    await seed_balance(user_id=1, name="alice", amount=1_000)
    await seed_balance(user_id=2, name="bob", amount=1_000)
    await seed_balance(user_id=3, name="carol", amount=1_000)
    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-300, casino_delta=300
    )
    await apply_blackjack_settlement(
        player_id=2, player_account_name="bob", player_delta=200, casino_delta=-200
    )
    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=500, casino_delta=-500
    )
    await apply_blackjack_settlement(
        player_id=3, player_account_name="carol", player_delta=-200, casino_delta=200
    )
    rows = await top_losers(limit=10)
    assert rows == [
        LossLeaderboardEntry(user_id=1, name="alice", loss_amount=300, avatar_url=""),
        LossLeaderboardEntry(user_id=3, name="carol", loss_amount=200, avatar_url=""),
    ]


async def test_top_losers_orders_by_loss_magnitude() -> None:
    """The leaderboard sorts from biggest loss to smallest."""
    for user_id, name, loss in [(1, "alice", 100), (2, "bob", 500), (3, "carol", 250)]:
        await seed_balance(user_id=user_id, name=name, amount=loss)
        await apply_blackjack_settlement(
            player_id=user_id, player_account_name=name, player_delta=-loss, casino_delta=loss
        )
    rows = await top_losers(limit=10)
    assert [(row.user_id, row.loss_amount) for row in rows] == [(2, 500), (3, 250), (1, 100)]


async def test_top_losers_excludes_leaderboard_hidden_accounts_by_default() -> None:
    """Hidden accounts do not appear on the public daily loss leaderboard."""
    await seed_balance(user_id=1, name="alice", amount=500)
    await seed_balance(user_id=2, name="bob", amount=400)
    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-500, casino_delta=500
    )
    await apply_blackjack_settlement(
        player_id=2, player_account_name="bob", player_delta=-400, casino_delta=400
    )
    await hide_from_leaderboard(user_id=1)

    rows = await top_losers(limit=10)
    assert rows == [LossLeaderboardEntry(user_id=2, name="bob", loss_amount=400, avatar_url="")]


async def test_top_losers_can_include_leaderboard_hidden_accounts() -> None:
    """Maintenance callers can include hidden accounts in daily loss queries."""
    await seed_balance(user_id=1, name="alice", amount=500)
    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-500, casino_delta=500
    )
    await hide_from_leaderboard(user_id=1)

    rows = await top_losers(limit=10, include_hidden=True)
    assert rows == [LossLeaderboardEntry(user_id=1, name="alice", loss_amount=500, avatar_url="")]


async def test_top_losers_ignores_counters_before_today() -> None:
    """Stale account counters from an older Taipei day do not count."""
    await seed_balance(user_id=1, name="alice", amount=500)
    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-500, casino_delta=500
    )
    past = datetime.now(tz=TAIWAN_TIMEZONE) - timedelta(days=2)
    async with open_session() as session:
        await session.execute(
            statement=update(CasinoAccount)
            .where(CasinoAccount.user_id == 1)
            .values(day_started_at=_taipei_midnight(now=past))
        )
        await session.commit()
    assert await top_losers(limit=10) == []


async def test_top_losers_empty_when_no_casino_activity() -> None:
    """Without any daily casino loss counters the leaderboard is empty."""
    await seed_balance(user_id=1, name="alice", amount=100)
    assert await top_losers(limit=10) == []


async def test_top_losers_ignores_manual_adjustments() -> None:
    """Manual admin debits do not count as casino losses."""
    await adjust_balance(user_id=1, name="alice", delta=-100, allow_negative=True)
    assert await top_losers(limit=10) == []


async def test_apply_jackpot_settlement_credits_player_and_drains_pool() -> None:
    """Player wins pull points out of the jackpot row in one atomic step."""
    await seed_balance(user_id=1, name="alice", amount=10_000)
    # The schema bootstrap already seeded the dragon_gate pool at 1_000.
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000

    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=200, game_id="dragon_gate"
    )

    assert settlement.player_balance == 10_200
    assert settlement.jackpot_balance == 800
    assert settlement.applied_player_delta == 200
    assert settlement.jackpot_depleted is False
    assert await get_jackpot_pool(game_id="dragon_gate") == 800
    await assert_daily_casino_stats(user_id=1, loss=0, win=200, net=200)


async def test_apply_jackpot_settlement_replenishes_drained_seed_pool() -> None:
    """A seeded jackpot restores itself after a player wins the whole pool."""
    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=1_000, game_id="dragon_gate"
    )

    assert settlement.player_balance == 1_000
    assert settlement.jackpot_balance == 1_000
    assert settlement.applied_player_delta == 1_000
    assert settlement.jackpot_depleted is True
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000
    async with open_session() as session:
        result = await session.execute(
            statement=select(JackpotPool.seeded_amount, JackpotPool.total_claimed).where(
                JackpotPool.game_id == "dragon_gate"
            )
        )
        seeded_amount, total_claimed = result.one()
    assert seeded_amount == 2_000
    assert total_claimed == 1_000


async def test_apply_jackpot_settlement_clamps_loss_and_grows_pool_by_actual_debit() -> None:
    """Player losses stop at zero and feed the jackpot with the actual debit."""
    await seed_balance(user_id=1, name="alice", amount=15_000)

    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=-25_000, game_id="dragon_gate"
    )

    assert settlement.player_balance == 0
    assert settlement.jackpot_balance == 16_000
    assert settlement.applied_player_delta == -15_000
    account = await get_account(user_id=1)
    assert account == AccountSnapshot(
        name="alice", balance=0, total_earned=15_000, total_spent=15_000
    )
    await assert_daily_casino_stats(user_id=1, loss=15_000, win=0, net=-15_000)


async def test_apply_jackpot_settlement_concurrent_clamped_losses_count_actual_debit() -> None:
    """Concurrent clamped jackpot losses cannot over-credit the pool."""
    await seed_balance(user_id=1, name="alice", amount=100)

    first, second = await asyncio.gather(
        apply_jackpot_settlement(
            player_id=1, player_account_name="alice", player_delta=-80, game_id="dragon_gate"
        ),
        apply_jackpot_settlement(
            player_id=1, player_account_name="alice", player_delta=-80, game_id="dragon_gate"
        ),
    )

    applied_total = first.applied_player_delta + second.applied_player_delta
    assert applied_total == -100
    assert await get_balance(user_id=1) == 0
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_100
    account = await get_account(user_id=1)
    assert account == AccountSnapshot(name="alice", balance=0, total_earned=100, total_spent=100)
    await assert_daily_casino_stats(user_id=1, loss=100, win=0, net=-100)


async def test_apply_jackpot_settlement_caps_win_to_live_pool() -> None:
    """A stale oversized jackpot win only pays the live pool amount."""
    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=150_000, game_id="dragon_gate"
    )

    assert settlement.player_balance == 1_000
    assert settlement.applied_player_delta == 1_000
    assert settlement.jackpot_balance == 1_000
    assert settlement.jackpot_depleted is True


async def test_apply_jackpot_settlement_concurrent_wins_do_not_double_claim_pool() -> None:
    """Concurrent whole-pool wins cannot both claim the same jackpot generation."""
    snapshot = await get_jackpot_snapshot(game_id="dragon_gate")
    first, second = await asyncio.gather(
        apply_jackpot_settlement(
            player_id=1,
            player_account_name="alice",
            player_delta=1_000,
            game_id="dragon_gate",
            expected_jackpot_generation=snapshot.generation,
        ),
        apply_jackpot_settlement(
            player_id=2,
            player_account_name="bob",
            player_delta=1_000,
            game_id="dragon_gate",
            expected_jackpot_generation=snapshot.generation,
        ),
    )

    applied_total = first.applied_player_delta + second.applied_player_delta
    assert applied_total == 1_000
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000


async def test_apply_jackpot_settlement_batch_charges_multiple_players_atomically() -> None:
    """Batch jackpot settlements share one transaction and one final snapshot."""
    await seed_balance(user_id=1, name="alice", amount=10_000)
    await seed_balance(user_id=2, name="bob", amount=10_000)

    result = await apply_jackpot_settlement_batch(
        game_id="dragon_gate",
        settlements=(
            JackpotSettlementRequest(
                player_id=1, player_account_name="alice", player_delta=-5_000
            ),
            JackpotSettlementRequest(player_id=2, player_account_name="bob", player_delta=-7_000),
        ),
    )

    assert result.player_balances == {1: 5_000, 2: 3_000}
    assert result.applied_player_deltas == {1: -5_000, 2: -7_000}
    assert result.jackpot_balance == 13_000
    assert await get_jackpot_pool(game_id="dragon_gate") == 13_000


async def test_apply_jackpot_settlement_batch_rejects_required_full_debit() -> None:
    """Ante-style full-debit batches reject without partially charging anyone."""
    await seed_balance(user_id=1, name="alice", amount=10_000)
    await seed_balance(user_id=2, name="bob", amount=3_000)

    result = await apply_jackpot_settlement_batch(
        game_id="dragon_gate",
        settlements=(
            JackpotSettlementRequest(
                player_id=1,
                player_account_name="alice",
                player_delta=-5_000,
                require_full_debit=True,
            ),
            JackpotSettlementRequest(
                player_id=2,
                player_account_name="bob",
                player_delta=-5_000,
                require_full_debit=True,
            ),
        ),
    )

    assert result.rejected_player_ids == (2,)
    assert result.player_balances == {}
    assert result.applied_player_deltas == {}
    assert await get_balance(user_id=1) == 10_000
    assert await get_balance(user_id=2) == 3_000
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000


async def test_apply_jackpot_settlement_batch_rolls_back_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed batch ante settlement cannot partially charge players."""
    await seed_balance(user_id=1, name="alice", amount=10_000)
    await seed_balance(user_id=2, name="bob", amount=10_000)
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000

    calls = 0
    original_apply = _apply_jackpot_delta_in_session

    async def flaky_apply_jackpot_delta_in_session(
        **kwargs: Any,  # noqa: ANN401 -- test double accepts heterogeneous kwargs
    ) -> tuple[JackpotSnapshot, bool]:
        """Fails on the second jackpot write to test batch rollback."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced batch failure")
        return await original_apply(**kwargs)

    monkeypatch.setattr(
        "discordbot.services.economy.database._apply_jackpot_delta_in_session",
        flaky_apply_jackpot_delta_in_session,
    )

    with pytest.raises(expected_exception=RuntimeError, match="forced batch failure"):
        await apply_jackpot_settlement_batch(
            game_id="dragon_gate",
            settlements=(
                JackpotSettlementRequest(
                    player_id=1, player_account_name="alice", player_delta=-5_000
                ),
                JackpotSettlementRequest(
                    player_id=2, player_account_name="bob", player_delta=-7_000
                ),
            ),
        )

    assert await get_balance(user_id=1) == 10_000
    assert await get_balance(user_id=2) == 10_000
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000


async def test_apply_jackpot_settlement_skips_vip_blackjack_bonus() -> None:
    """射龍門 winnings stay at face value even for VIP accounts."""
    await seed_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    purchase = await buy_vip(user_id=1, name="alice")
    assert purchase is not None

    player_balance_before = await get_balance(user_id=1)
    pool_before = await get_jackpot_pool(game_id="dragon_gate")
    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=100, game_id="dragon_gate"
    )

    assert settlement.player_balance == player_balance_before + 100
    assert settlement.jackpot_balance == pool_before - 100
    assert settlement.applied_player_delta == 100


async def test_get_jackpot_pool_returns_zero_for_missing_game() -> None:
    """Unseeded game ids surface as 0 instead of raising."""
    assert await get_jackpot_pool(game_id="never_registered") == 0


async def test_get_jackpot_pool_replenishes_drained_seed_pool() -> None:
    """Reading a seeded jackpot replenishes a zero-balance row."""
    async with open_session() as session:
        await session.execute(
            statement=update(JackpotPool)
            .where(JackpotPool.game_id == "dragon_gate")
            .values(pool_balance=0)
        )
        await session.commit()

    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000
    snapshot = await get_jackpot_snapshot(game_id="dragon_gate")
    assert snapshot.generation == 1


async def test_ensure_schema_rerun_keeps_every_seeded_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart keeps the jackpot pool and both ledgers where play left them.

    Each row is moved off its seed first, since a bootstrap that reset them would otherwise
    look exactly like one that left them alone.
    """
    db_path = tmp_path / "seed-economy.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setattr("discordbot.services.economy.database._engine", engine)
    async with open_session() as session:
        await session.execute(
            statement=update(JackpotPool)
            .where(JackpotPool.game_id == "dragon_gate")
            .values(pool_balance=1_234, generation=3)
        )
        await session.execute(
            statement=update(CasinoLedger).values(balance=-55, total_earned=5, total_spent=60)
        )
        await session.execute(
            statement=update(CentralBankLedger).values(balance=77, total_earned=77)
        )
        await session.commit()
    await engine.dispose()

    # A second engine on the same file is what makes the bootstrap run again (readiness is
    # tracked by engine identity), and it is also what a restart looks like.
    restarted = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setattr("discordbot.services.economy.database._engine", restarted)

    jackpot = await get_jackpot_snapshot(game_id="dragon_gate")
    ledger = await get_casino_ledger()
    central_bank = await get_central_bank_status(guild_id=1)
    assert (jackpot.balance, jackpot.generation) == (1_234, 3)
    assert (ledger.balance, ledger.total_earned, ledger.total_spent) == (-55, 5, 60)
    assert central_bank.ledger_balance == 77
    await restarted.dispose()
