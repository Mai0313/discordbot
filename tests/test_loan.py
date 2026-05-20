"""Tests for the loan layer: credit limit, borrow / repay flows, and daily reset."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from discordbot.typings.economy import TransactionKind
from discordbot.cogs._economy.database import (
    TAIWAN_TIMEZONE,
    LoanAccount,
    repay,
    borrow,
    transfer,
    get_account,
    get_balance,
    credit_limit,
    open_session,
    _database_now,
    get_loan_view,
    _ensure_schema,
    adjust_balance,
    _build_credit_upsert,
    credit_with_repayment,
    apply_round_settlement,
    apply_vip_blackjack_bonus,
)

if TYPE_CHECKING:
    from nextcord import User

pytestmark = pytest.mark.usefixtures("economy_isolated_db")


def _user_with_age(days_old: int) -> User:
    """Returns a minimal stand-in for a Discord user with a known creation date."""
    return cast(
        "User", SimpleNamespace(created_at=datetime.now(tz=UTC) - timedelta(days=days_old))
    )


async def _add_balance(user_id: int, name: str, amount: int, avatar_url: str = "") -> int:
    """Seeds a positive balance without loan side effects."""
    await _ensure_schema()
    if amount <= 0:
        return await get_balance(user_id=user_id)
    now = _database_now()
    async with open_session() as session:
        result = await session.execute(
            statement=_build_credit_upsert(
                user_id=user_id, name=name, amount=amount, avatar_url=avatar_url, now=now
            )
        )
        await session.commit()
        return result.scalar_one()


async def _backdate_loan_opened_at(user_id: int, days_ago: int) -> None:
    """Test helper: simulates time passing by pushing loan_opened_at back."""
    past = datetime.now(tz=TAIWAN_TIMEZONE) - timedelta(days=days_ago)
    async with open_session() as session:
        await session.execute(
            statement=update(LoanAccount)
            .where(LoanAccount.user_id == user_id)
            .values(opened_at=past)
        )
        await session.commit()


# Pure functions ------------------------------------------------------------


@pytest.mark.parametrize(
    argnames=("days", "expected"),
    argvalues=[
        (0, 1_000),
        (29, 1_000),
        (30, 10_000),
        (179, 10_000),
        (180, 50_000),
        (364, 50_000),
        (365, 200_000),
        (365 * 3 - 1, 200_000),
        (365 * 3, 500_000),
        (365 * 10, 500_000),
    ],
)
def test_credit_limit_tier_boundaries(days: int, expected: int) -> None:
    """Each tier boundary returns the expected cap (non-VIP)."""
    user = _user_with_age(days_old=days)
    assert credit_limit(user=user) == expected


def test_credit_limit_doubles_for_vip() -> None:
    """VIP perk doubles the credit limit regardless of account-age tier."""
    user = _user_with_age(days_old=365 * 5)
    assert credit_limit(user=user, is_vip=False) == 500_000
    assert credit_limit(user=user, is_vip=True) == 1_000_000


# VIP blackjack bonus -------------------------------------------------------


@pytest.mark.parametrize(
    argnames=("delta", "is_vip", "expected"),
    argvalues=[
        (100, False, 100),
        (100, True, 150),
        (101, True, 151),
        (0, True, 0),
        (-50, True, -50),
        (1, True, 1),
    ],
)
def test_apply_vip_blackjack_bonus(delta: int, is_vip: bool, expected: int) -> None:
    """VIP 1.5x multiplier only fires for positive deltas."""
    assert apply_vip_blackjack_bonus(delta=delta, is_vip=is_vip) == expected


# Borrow --------------------------------------------------------------------


async def test_borrow_first_time_creates_row_and_disburses() -> None:
    """A first-time borrow creates the row, sets opened_at, and credits balance."""
    result = await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    assert result is not None
    assert result.new_balance == 500
    assert result.principal == 500
    assert result.borrowed_amount == 500

    view = await get_loan_view(user_id=1)
    assert view is not None
    assert view.principal == 500
    assert view.total_borrowed == 500
    assert view.opened_at is not None


async def test_borrow_accumulates_principal_on_existing_loan() -> None:
    """A second borrow adds to existing principal without resetting opened_at."""
    first = await borrow(user_id=1, name="alice", amount=300, credit_limit_value=1_000)
    assert first is not None
    first_view = await get_loan_view(user_id=1)
    assert first_view is not None
    first_opened_at = first_view.opened_at

    result = await borrow(user_id=1, name="alice", amount=400, credit_limit_value=1_000)
    assert result is not None
    assert result.new_balance == 700
    assert result.principal == 700
    assert result.borrowed_amount == 400
    view = await get_loan_view(user_id=1)
    assert view is not None
    assert view.opened_at == first_opened_at
    assert view.total_borrowed == 700


async def test_borrow_over_limit_clamps_to_remaining_credit() -> None:
    """A borrow that would exceed the cap disburses the remaining daily credit."""
    await borrow(user_id=1, name="alice", amount=800, credit_limit_value=1_000)
    result = await borrow(user_id=1, name="alice", amount=300, credit_limit_value=1_000)
    assert result is not None
    assert result.new_balance == 1_000
    assert result.principal == 1_000
    assert result.borrowed_amount == 200
    view = await get_loan_view(user_id=1)
    assert view is not None
    assert view.principal == 1_000


async def test_borrow_first_request_over_limit_clamps_to_limit() -> None:
    """A first borrow request above the cap borrows the full cap."""
    result = await borrow(user_id=1, name="alice", amount=1_500, credit_limit_value=1_000)
    assert result is not None
    assert result.new_balance == 1_000
    assert result.principal == 1_000
    assert result.borrowed_amount == 1_000


async def test_borrow_rejects_when_remaining_credit_is_zero() -> None:
    """Once principal reaches the cap, additional borrow requests are rejected."""
    await borrow(user_id=1, name="alice", amount=1_500, credit_limit_value=1_000)
    result = await borrow(user_id=1, name="alice", amount=1, credit_limit_value=1_000)
    assert result is None


async def test_borrow_concurrent_requests_cannot_exceed_limit() -> None:
    """Concurrent borrows must not both pass against the same observed principal."""
    results = await asyncio.gather(
        borrow(user_id=1, name="alice", amount=800, credit_limit_value=1_000),
        borrow(user_id=1, name="alice", amount=800, credit_limit_value=1_000),
    )
    assert all(result is not None for result in results)
    view = await get_loan_view(user_id=1)
    assert view is not None
    assert view.principal == 1_000
    assert await get_balance(user_id=1) == 1_000
    assert sum(result.borrowed_amount for result in results if result is not None) == 1_000


async def test_borrow_counts_balance_increase_as_earnings() -> None:
    """Borrowed money increases balance, so it also bumps total_earned."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    account = await get_account(user_id=1)
    assert account is not None
    assert account.balance == 500
    assert account.total_earned == 500


async def test_borrow_on_existing_account_extends_total_earned() -> None:
    """Borrowing on top of an existing balance keeps totals aligned with balance."""
    await _add_balance(user_id=1, name="alice", amount=100)
    result = await borrow(user_id=1, name="alice", amount=400, credit_limit_value=1_000)
    assert result is not None
    assert result.new_balance == 500
    account = await get_account(user_id=1)
    assert account is not None
    assert account.balance == 500
    assert account.total_earned == 500


@pytest.mark.parametrize(argnames="amount", argvalues=[0, -1, -1000])
async def test_borrow_rejects_non_positive_amount(amount: int) -> None:
    """Non-positive borrow amounts are rejected."""
    result = await borrow(user_id=1, name="alice", amount=amount, credit_limit_value=1_000)
    assert result is None


# Repay ---------------------------------------------------------------------


async def test_repay_pays_principal_only() -> None:
    """Repayment debits the requested amount from principal in one shot."""
    await borrow(user_id=1, name="alice", amount=1_000, credit_limit_value=10_000)
    result = await repay(user_id=1, name="alice", amount=300)
    assert result is not None
    assert result.principal_repaid == 300
    assert result.remaining_debt == 700
    assert result.new_balance == 700


async def test_repay_clamps_to_debt_total() -> None:
    """Over-request only repays up to the user's debt."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=10_000)
    result = await repay(user_id=1, name="alice", amount=10_000)
    assert result is not None
    assert result.principal_repaid == 500
    assert result.remaining_debt == 0
    assert result.new_balance == 0


async def test_repay_clamps_to_balance_when_balance_smaller() -> None:
    """When balance < debt, repayment caps at balance."""
    await borrow(user_id=1, name="alice", amount=1_000, credit_limit_value=10_000)
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-600,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=600,
    )
    result = await repay(user_id=1, name="alice", amount=10_000)
    assert result is not None
    assert result.principal_repaid == 400
    assert result.new_balance == 0
    assert result.remaining_debt == 600


async def test_repay_no_debt_returns_none() -> None:
    """No outstanding debt → repayment rejected."""
    await _add_balance(user_id=1, name="alice", amount=100)
    assert await repay(user_id=1, name="alice", amount=50) is None


async def test_repay_zero_balance_returns_none() -> None:
    """Zero balance → no funds to repay with, even when debt remains."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-500,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=500,
    )
    assert await repay(user_id=1, name="alice", amount=100) is None


async def test_repay_negative_balance_returns_none() -> None:
    """Negative balance from manual maintenance is not repayable cash."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await adjust_balance(user_id=1, name="alice", delta=-600, allow_negative=True)
    assert await repay(user_id=1, name="alice", amount=100) is None


async def test_repay_counts_balance_decrease_as_spent() -> None:
    """Repaying a loan decreases balance, so it bumps total_spent."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await repay(user_id=1, name="alice", amount=200)
    account = await get_account(user_id=1)
    assert account is not None
    assert account.total_spent == 200
    assert account.total_earned - account.total_spent == account.balance


@pytest.mark.parametrize(argnames="amount", argvalues=[0, -1])
async def test_repay_rejects_non_positive_amount(amount: int) -> None:
    """Non-positive repay amounts are rejected."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    assert await repay(user_id=1, name="alice", amount=amount) is None


# credit_with_repayment ----------------------------------------------------


async def test_credit_with_repayment_full_credit_when_no_debt() -> None:
    """Without debt, the full amount lands in balance."""
    result = await credit_with_repayment(
        user_id=1, name="alice", amount=100, kind=TransactionKind.CHAT_REWARD
    )
    assert result.new_balance == 100
    assert result.credited_amount == 100
    assert result.principal_repaid == 0
    assert result.remaining_debt == 0


async def test_credit_with_repayment_leaves_principal_untouched_at_zero_ratio() -> None:
    """With auto-repay disabled (ratio=0), positive income lands fully in balance."""
    await borrow(user_id=1, name="alice", amount=1_000, credit_limit_value=10_000)
    result = await credit_with_repayment(
        user_id=1, name="alice", amount=200, kind=TransactionKind.CHAT_REWARD
    )
    assert result.principal_repaid == 0
    assert result.credited_amount == 200
    assert result.remaining_debt == 1_000


async def test_credit_with_repayment_full_credit_even_with_outstanding_principal() -> None:
    """Outstanding debt is not auto-deducted at ratio=0, even on a large payout."""
    await borrow(user_id=1, name="alice", amount=50, credit_limit_value=1_000)
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-50,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=50,
    )
    result = await credit_with_repayment(
        user_id=1, name="alice", amount=1_000, kind=TransactionKind.CHAT_REWARD
    )
    assert result.principal_repaid == 0
    assert result.credited_amount == 1_000
    assert result.remaining_debt == 50
    assert result.new_balance == 1_000


async def test_credit_with_repayment_odd_amount_lands_whole_at_zero_ratio() -> None:
    """Odd-amount income still credits fully when auto-repay ratio is 0."""
    await borrow(user_id=1, name="alice", amount=100, credit_limit_value=1_000)
    result = await credit_with_repayment(
        user_id=1, name="alice", amount=11, kind=TransactionKind.CHAT_REWARD
    )
    assert result.principal_repaid == 0
    assert result.credited_amount == 11


async def test_credit_with_repayment_zero_amount_is_noop() -> None:
    """Zero amount returns current balance without mutating totals."""
    await _add_balance(user_id=1, name="alice", amount=100)
    result = await credit_with_repayment(
        user_id=1, name="alice", amount=0, kind=TransactionKind.CHAT_REWARD
    )
    assert result.new_balance == 100
    assert result.credited_amount == 0
    account = await get_account(user_id=1)
    assert account is not None
    assert account.total_earned == 100
    assert account.total_spent == 0


async def test_credit_with_repayment_first_sight_creates_row() -> None:
    """An unknown user receiving credit gets a fresh row."""
    result = await credit_with_repayment(
        user_id=42, name="newcomer", amount=200, kind=TransactionKind.CHAT_REWARD
    )
    assert result.new_balance == 200
    assert result.credited_amount == 200
    assert await get_balance(user_id=42) == 200


async def test_credit_with_repayment_concurrent_credits_accumulate() -> None:
    """Concurrent credits on the same user must not lose updates."""
    await _add_balance(user_id=1, name="alice", amount=0)
    await asyncio.gather(*[
        credit_with_repayment(user_id=1, name="alice", amount=10, kind=TransactionKind.CHAT_REWARD)
        for _ in range(10)
    ])
    assert await get_balance(user_id=1) == 100


# Daily loan reset ---------------------------------------------------------


async def test_loan_expires_at_next_taipei_midnight() -> None:
    """A loan opened before today's Taipei midnight is wiped on next access."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await _backdate_loan_opened_at(user_id=1, days_ago=2)
    view = await get_loan_view(user_id=1)
    assert view is not None
    assert view.principal == 0
    assert view.opened_at is None


async def test_loan_stays_active_within_the_same_taipei_day() -> None:
    """A loan opened earlier the same Taipei day is still outstanding."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    view = await get_loan_view(user_id=1)
    assert view is not None
    assert view.principal == 500
    assert view.opened_at is not None


async def test_borrow_after_expiry_re_arms_daily_window() -> None:
    """A new borrow after expiry creates a fresh daily window."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await _backdate_loan_opened_at(user_id=1, days_ago=3)

    second = await borrow(user_id=1, name="alice", amount=400, credit_limit_value=1_000)
    assert second is not None
    assert second.principal == 400
    view = await get_loan_view(user_id=1)
    assert view is not None
    assert view.opened_at is not None


async def test_expired_loan_does_not_reduce_balance() -> None:
    """The daily reset wipes principal but does not claw back borrowed funds."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await _backdate_loan_opened_at(user_id=1, days_ago=2)
    assert await get_balance(user_id=1) == 500


async def test_credit_with_repayment_after_expiry_credits_full_amount() -> None:
    """After the daily reset, expired debt is gone and the full credit lands."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await _backdate_loan_opened_at(user_id=1, days_ago=2)
    result = await credit_with_repayment(
        user_id=1, name="alice", amount=200, kind=TransactionKind.CHAT_REWARD
    )
    assert result.credited_amount == 200
    assert result.principal_repaid == 0
    assert result.remaining_debt == 0


# get_loan_view ------------------------------------------------------------


async def test_get_loan_view_unknown_user_returns_none() -> None:
    """An unknown user has no row to project."""
    assert await get_loan_view(user_id=999) is None


async def test_get_loan_view_account_without_loan_returns_zero_state() -> None:
    """A user with a balance row but no loan returns a fresh-zero snapshot."""
    await _add_balance(user_id=1, name="alice", amount=100)
    view = await get_loan_view(user_id=1)
    assert view is not None
    assert view.principal == 0
    assert view.opened_at is None
    assert view.total_borrowed == 0
    assert view.total_repaid == 0


async def test_get_loan_view_after_borrow_and_repay_tracks_totals() -> None:
    """Gross flows persist in total_borrowed / total_repaid across operations."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await repay(user_id=1, name="alice", amount=200)
    view = await get_loan_view(user_id=1)
    assert view is not None
    assert view.principal == 300
    assert view.total_borrowed == 500
    assert view.total_repaid == 200


# Account totals -------------------------------------------------------------


async def test_borrow_and_repay_keep_account_totals_aligned() -> None:
    """Borrow and repay maintain the account balance invariant."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await repay(user_id=1, name="alice", amount=200)
    account = await get_account(user_id=1)
    assert account is not None
    assert account.balance == 300
    assert account.total_earned == 500
    assert account.total_spent == 200
    assert account.total_earned - account.total_spent == account.balance


async def test_chat_reward_with_debt_updates_earned_only_at_zero_ratio() -> None:
    """With auto-repay disabled, income lands fully in balance and principal stays."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await credit_with_repayment(
        user_id=1, name="alice", amount=100, kind=TransactionKind.CHAT_REWARD
    )
    account = await get_account(user_id=1)
    loan = await get_loan_view(user_id=1)
    assert account is not None
    assert loan is not None
    assert account.balance == 600
    assert account.total_earned == 600
    assert account.total_spent == 0
    assert loan.principal == 500


async def test_transfer_updates_sender_and_receiver_totals() -> None:
    """Transfer debits sender spent total and credits receiver earned total."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await transfer(sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=40)
    sender = await get_account(user_id=1)
    receiver = await get_account(user_id=2)
    assert sender is not None
    assert receiver is not None
    assert (sender.balance, sender.total_earned, sender.total_spent) == (60, 100, 40)
    assert (receiver.balance, receiver.total_earned, receiver.total_spent) == (40, 40, 0)


async def test_apply_round_settlement_updates_player_and_house_totals() -> None:
    """Casino settlement stores actual applied deltas in account totals."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-40,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=40,
    )
    player = await get_account(user_id=1)
    house = await get_account(user_id=99)
    assert player is not None
    assert house is not None
    assert (player.balance, player.total_earned, player.total_spent) == (60, 100, 40)
    assert (house.balance, house.total_earned, house.total_spent) == (40, 40, 0)


async def test_adjust_balance_counts_applied_delta_not_requested_delta() -> None:
    """A clamped manual adjustment spends only the applied balance delta."""
    await _add_balance(user_id=1, name="alice", amount=10)
    await adjust_balance(user_id=1, name="alice", delta=-1000)
    account = await get_account(user_id=1)
    assert account is not None
    assert (account.balance, account.total_earned, account.total_spent) == (0, 10, 10)
