"""Accounting-invariant assertions for economy and casino state.

These read production state through the real database helpers and assert the structural
identities the economy guarantees, instead of each test re-summing magic numbers by hand. They
return the snapshot so a caller can layer a focused exact check (a settlement delta that must
equal a computed value) on top.

The identities are worth pinning because they are the ledger's only self-check: there is no
transaction table, so a write path that moves `balance` without bumping `total_earned` /
`total_spent` by the same amount leaves nothing else to catch it. The three helpers cover the
three places money is counted: `assert_wallet_consistent` the per-user wallet,
`assert_casino_ledger_consistent` the single house ledger row (which is deliberately NOT the
bot's own wallet), and `assert_daily_casino_stats` the per-user daily casino counters.

Every one of them queries the live module-level engine, so a caller needs the
`economy_isolated_db` fixture from `tests/conftest.py` or it reads the real `economy.db`.
"""

from discordbot.typings.economy import AccountSnapshot, CasinoDailyStats, CasinoLedgerSnapshot
from discordbot.services.economy.database import (
    get_account,
    get_casino_ledger,
    get_casino_daily_stats,
)


async def assert_wallet_consistent(
    user_id: int, expected_balance: int | None = None
) -> AccountSnapshot:
    """Asserts one user's wallet identity `balance == total_earned - total_spent`.

    A user with no account row fails rather than passing vacuously, since the caller reached here
    expecting a write that evidently never happened.

    Args:
        user_id (int): Discord user ID whose wallet is checked.
        expected_balance (int | None): Exact balance to require as well; None checks only the
            identity.

    Returns:
        The account snapshot, so a caller can go on to check `total_earned` / `total_spent`.
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
    """Asserts the casino ledger identity `balance == total_earned - total_spent`.

    The house ledger is one row rather than an account, so there is no user to name. A database
    with no ledger row yet reads back as all-zero, which is what lets this be called as a
    pre-settlement baseline instead of only after a round.

    Args:
        expected_balance (int | None): Exact ledger balance to require as well; None checks only
            the identity.

    Returns:
        The casino ledger snapshot, so a caller can go on to check the lifetime gross flows.
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
    """Asserts a user's daily casino counters and the identity `net == win - loss`.

    The counters are gross and scoped to the current Taipei day, and the read reports a stale day
    as all-zero without rewriting it, so an all-zero expectation passes equally for a user who
    never played and for one whose row belongs to yesterday.

    Args:
        user_id (int): Discord user ID whose counters are checked.
        loss (int): Expected gross loss for the current day.
        win (int): Expected gross win for the current day.
        net (int): Expected signed net for the current day.

    Returns:
        The daily stats snapshot.
    """
    stats = await get_casino_daily_stats(user_id=user_id)
    assert stats.daily_net == stats.daily_win - stats.daily_loss, (
        f"daily net identity broken for user {user_id}: "
        f"{stats.daily_net} != {stats.daily_win} - {stats.daily_loss}"
    )
    assert stats.daily_loss == loss, f"daily_loss {stats.daily_loss} != {loss}"
    assert stats.daily_win == win, f"daily_win {stats.daily_win} != {win}"
    assert stats.daily_net == net, f"daily_net {stats.daily_net} != {net}"
    return stats
