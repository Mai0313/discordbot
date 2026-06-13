"""Accounting-invariant assertions for economy and casino state.

These read production state through the real database helpers and assert the
structural identities the economy guarantees, instead of each test re-summing
magic numbers by hand. They return the snapshot so a caller can layer a focused
exact check (a settlement delta that must equal a computed value) on top.
"""

from discordbot.typings.economy import AccountSnapshot, CasinoDailyStats, CasinoLedgerSnapshot
from discordbot.cogs._economy.database import (
    get_account,
    get_casino_ledger,
    get_casino_daily_stats,
)


async def assert_wallet_consistent(
    user_id: int, expected_balance: int | None = None
) -> AccountSnapshot:
    """Asserts the wallet identity ``balance == total_earned - total_spent``.

    When ``expected_balance`` is given it must match too. Returns the snapshot.
    """
    account = await get_account(user_id=user_id)
    assert account is not None, f"no account for user {user_id}"
    assert account.balance == account.total_earned - account.total_spent, (
        f"wallet identity broken for user {user_id}: "
        f"{account.balance} != {account.total_earned} - {account.total_spent}"
    )
    if expected_balance is not None:
        assert account.balance == expected_balance, (
            f"balance {account.balance} != expected {expected_balance}"
        )
    return account


async def assert_casino_ledger_consistent(
    expected_balance: int | None = None,
) -> CasinoLedgerSnapshot:
    """Asserts the casino ledger identity ``balance == total_earned - total_spent``.

    When ``expected_balance`` is given it must match too. Returns the snapshot.
    """
    ledger = await get_casino_ledger()
    assert ledger.balance == ledger.total_earned - ledger.total_spent, (
        f"casino ledger identity broken: "
        f"{ledger.balance} != {ledger.total_earned} - {ledger.total_spent}"
    )
    if expected_balance is not None:
        assert ledger.balance == expected_balance, (
            f"casino balance {ledger.balance} != expected {expected_balance}"
        )
    return ledger


async def assert_daily_casino_stats(
    user_id: int, loss: int, win: int, net: int
) -> CasinoDailyStats:
    """Asserts a user's daily casino counters and that ``net == win - loss``."""
    stats = await get_casino_daily_stats(user_id=user_id)
    assert stats.daily_net == stats.daily_win - stats.daily_loss, (
        f"daily net identity broken for user {user_id}: "
        f"{stats.daily_net} != {stats.daily_win} - {stats.daily_loss}"
    )
    assert stats.daily_loss == loss, f"daily_loss {stats.daily_loss} != {loss}"
    assert stats.daily_win == win, f"daily_win {stats.daily_win} != {win}"
    assert stats.daily_net == net, f"daily_net {stats.daily_net} != {net}"
    return stats
