"""Tests for the loan layer: credit limit, borrow / repay flows, auto-repay, daily reset, audit log."""

from __future__ import annotations

from types import SimpleNamespace
import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update

from discordbot.typings.economy import TransactionKind
from discordbot.cogs._economy.database import (
    TAIWAN_TIMEZONE,
    UserAccount,
    PointTransaction,
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

pytestmark = pytest.mark.usefixtures("economy_isolated_db")


def _user_with_age(days_old: int) -> SimpleNamespace:
    """Returns a minimal stand-in for a Discord user with a known creation date."""
    return SimpleNamespace(created_at=datetime.now(tz=UTC) - timedelta(days=days_old))


async def _list_transactions(user_id: int) -> list[tuple[str, int, int]]:
    """Returns ``(kind, delta, debt_after)`` rows for a user, oldest first."""
    async with open_session() as session:
        result = await session.execute(
            statement=select(
                PointTransaction.kind, PointTransaction.delta, PointTransaction.debt_after
            )
            .where(PointTransaction.user_id == user_id)
            .order_by(PointTransaction.id)
        )
        return [(row[0], row[1], row[2]) for row in result.all()]


async def _add_balance(user_id: int, name: str, amount: int, avatar_url: str = "") -> int:
    """Seeds a positive balance without writing audit rows."""
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
            statement=update(UserAccount)
            .where(UserAccount.user_id == user_id)
            .values(loan_opened_at=past)
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


async def test_borrow_treats_borrowed_money_as_debt_not_earnings() -> None:
    """Borrowed money goes into balance but does not bump total_earned."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    account = await get_account(user_id=1)
    assert account is not None
    _, balance, total_earned, _ = account
    assert balance == 500
    assert total_earned == 0


async def test_borrow_on_existing_account_preserves_total_earned() -> None:
    """Borrowing on top of an existing balance doesn't disturb earnings history."""
    await _add_balance(user_id=1, name="alice", amount=100)
    result = await borrow(user_id=1, name="alice", amount=400, credit_limit_value=1_000)
    assert result is not None
    assert result.new_balance == 500
    account = await get_account(user_id=1)
    assert account is not None
    _, balance, total_earned, _ = account
    assert balance == 500
    assert total_earned == 100


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
    """Negative casino balance is not repayable cash."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-600,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=600,
    )
    assert await repay(user_id=1, name="alice", amount=100) is None


async def test_repay_does_not_bump_total_spent() -> None:
    """Repaying a loan must not pollute the gameplay-spent counter."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await repay(user_id=1, name="alice", amount=200)
    account = await get_account(user_id=1)
    assert account is not None
    _, _, _, total_spent = account
    assert total_spent == 0


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


async def test_credit_with_repayment_diverts_50_percent_to_principal() -> None:
    """With debt, 50% of income pays down principal."""
    await borrow(user_id=1, name="alice", amount=1_000, credit_limit_value=10_000)
    result = await credit_with_repayment(
        user_id=1, name="alice", amount=200, kind=TransactionKind.CHAT_REWARD
    )
    assert result.principal_repaid == 100
    assert result.credited_amount == 100
    assert result.remaining_debt == 900


async def test_credit_with_repayment_caps_at_principal_total() -> None:
    """Repayment clamps to total debt when 50% exceeds outstanding."""
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
    assert result.principal_repaid == 50
    assert result.credited_amount == 950
    assert result.remaining_debt == 0
    assert result.new_balance == 950


async def test_credit_with_repayment_floors_odd_amount() -> None:
    """Floor-division of odd amount: 11 // 2 = 5 to repayment, 6 to balance."""
    await borrow(user_id=1, name="alice", amount=100, credit_limit_value=1_000)
    result = await credit_with_repayment(
        user_id=1, name="alice", amount=11, kind=TransactionKind.CHAT_REWARD
    )
    assert result.principal_repaid == 5
    assert result.credited_amount == 6


async def test_credit_with_repayment_zero_amount_is_noop() -> None:
    """Zero amount returns current balance without writing or logging."""
    await _add_balance(user_id=1, name="alice", amount=100)
    result = await credit_with_repayment(
        user_id=1, name="alice", amount=0, kind=TransactionKind.CHAT_REWARD
    )
    assert result.new_balance == 100
    assert result.credited_amount == 0
    rows = await _list_transactions(user_id=1)
    assert rows == []


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
    """After the daily reset the 50% auto-repay no longer applies."""
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


# Audit log ----------------------------------------------------------------


async def test_borrow_logs_audit_row() -> None:
    """Borrow writes one BORROW row with positive delta and post-state debt."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    rows = await _list_transactions(user_id=1)
    assert rows == [(TransactionKind.BORROW.value, 500, 500)]


async def test_repay_logs_audit_row() -> None:
    """Repay writes one REPAY row with negative delta and reduced debt."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await repay(user_id=1, name="alice", amount=200)
    rows = await _list_transactions(user_id=1)
    assert rows == [
        (TransactionKind.BORROW.value, 500, 500),
        (TransactionKind.REPAY.value, -200, 300),
    ]


async def test_chat_reward_logs_credited_slice_with_debt_context() -> None:
    """credit_with_repayment logs delta=credited and debt_after=post-repay debt."""
    await borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await credit_with_repayment(
        user_id=1, name="alice", amount=100, kind=TransactionKind.CHAT_REWARD
    )
    rows = await _list_transactions(user_id=1)
    assert rows == [
        (TransactionKind.BORROW.value, 500, 500),
        (TransactionKind.CHAT_REWARD.value, 50, 450),
    ]


async def test_transfer_logs_both_sides() -> None:
    """Transfer writes TRANSFER_OUT for sender and TRANSFER_IN for receiver."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await transfer(sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=40)
    assert await _list_transactions(user_id=1) == [(TransactionKind.TRANSFER_OUT.value, -40, 0)]
    assert await _list_transactions(user_id=2) == [(TransactionKind.TRANSFER_IN.value, 40, 0)]


async def test_transfer_log_note_captures_counterparty() -> None:
    """Transfer audit rows carry the counterparty identity in ``note``."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await transfer(sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=40)
    async with open_session() as session:
        result = await session.execute(
            statement=select(PointTransaction.user_id, PointTransaction.note).order_by(
                PointTransaction.id
            )
        )
        notes = result.all()
    assert (1, "to bob (2)") in notes
    assert (2, "from alice (1)") in notes


async def test_apply_round_settlement_logs_casino_bet() -> None:
    """A negative player settlement writes a CASINO_BET row."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-40,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=40,
    )
    assert await _list_transactions(user_id=1) == [(TransactionKind.CASINO_BET.value, -40, 0)]


async def test_apply_round_settlement_logs_payout_and_house_settle() -> None:
    """Player and dealer sides of a round each produce one audit row."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=40,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=-40,
    )
    assert await _list_transactions(user_id=1) == [(TransactionKind.CASINO_PAYOUT.value, 40, 0)]
    assert await _list_transactions(user_id=99) == [(TransactionKind.HOUSE_SETTLE.value, -40, 0)]


async def test_apply_round_settlement_skips_zero_delta_house_log() -> None:
    """A push (dealer_delta=0) does not produce a house audit row."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=0,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=0,
    )
    assert await _list_transactions(user_id=99) == []


async def test_adjust_balance_logs_applied_delta_not_requested_delta() -> None:
    """A clamped manual adjustment logs only the applied balance delta."""
    await _add_balance(user_id=1, name="alice", amount=10)
    await adjust_balance(user_id=1, name="alice", delta=-1000)
    rows = await _list_transactions(user_id=1)
    assert rows == [(TransactionKind.MANUAL_ADJUSTMENT.value, -10, 0)]
