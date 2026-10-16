"""Tests for the invariant asserts in `tests/helpers/economy_invariants.py`.

The redesigned suite leans on these asserts, so they are pinned here against the real
database helpers.
"""

import pytest
from sqlalchemy import Update, update

from discordbot.services.economy.database import (
    UserWallet,
    CasinoLedger,
    CasinoAccount,
    open_session,
    adjust_balance,
    apply_blackjack_settlement,
)

from tests.helpers.economy_invariants import (
    assert_wallet_consistent,
    assert_daily_casino_stats,
    assert_casino_ledger_consistent,
)


async def _write(statement: Update) -> None:
    """Writes straight to the ledger, past every helper that keeps its identities."""
    async with open_session() as session:
        await session.execute(statement=statement)
        await session.commit()


async def test_assert_wallet_consistent_fails_a_broken_wallet() -> None:
    """A clean credit passes, and totals that no longer add up to the balance fail."""
    await adjust_balance(user_id=1, name="alice", delta=250)
    await assert_wallet_consistent(user_id=1, expected_balance=250)

    await _write(statement=update(UserWallet).values(total_spent=1))

    with pytest.raises(AssertionError, match="wallet identity broken"):
        await assert_wallet_consistent(user_id=1)


async def test_the_casino_asserts_fail_a_broken_counter() -> None:
    """Settled play passes, and a ledger or daily counter off its identity fails.

    Each broken call passes the stored values as expected, so only the identity can fail it.
    """
    await adjust_balance(user_id=1, name="alice", delta=100)
    await apply_blackjack_settlement(
        player_id=1, player_account_name="alice", player_delta=-10, casino_delta=10
    )
    await assert_casino_ledger_consistent(expected_balance=10)
    await assert_daily_casino_stats(user_id=1, loss=10, win=0, net=-10)

    await _write(statement=update(CasinoLedger).values(total_spent=1))
    await _write(statement=update(CasinoAccount).values(daily_net=0))

    with pytest.raises(AssertionError, match="casino ledger identity broken"):
        await assert_casino_ledger_consistent(expected_balance=10)
    with pytest.raises(AssertionError, match="daily net identity broken"):
        await assert_daily_casino_stats(user_id=1, loss=10, win=0, net=0)
