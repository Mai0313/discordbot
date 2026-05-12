"""Tests for the loan layer: credit limit, interest accrual, borrow / repay flows, auto-repay, and audit logging."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.cogs._economy import database
from discordbot.typings.economy import TransactionKind

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import AsyncIterator


@pytest.fixture(autouse=True)
async def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Per-test SQLite file with the full loan-aware schema."""
    db_path = tmp_path / "economy.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(database.Base.metadata.create_all)
    monkeypatch.setattr(target=database, name="_engine", value=engine)
    yield
    await engine.dispose()


def _user_with_age(*, days_old: int) -> SimpleNamespace:
    """Returns a minimal stand-in for a Discord user with a known creation date."""
    return SimpleNamespace(created_at=datetime.now(tz=UTC) - timedelta(days=days_old))


async def _list_transactions(*, user_id: int) -> list[tuple[str, int, int]]:
    """Returns ``(kind, delta, debt_after)`` rows for a user, oldest first."""
    async with database.open_session() as session:
        result = await session.execute(
            statement=select(
                database.PointTransaction.kind,
                database.PointTransaction.delta,
                database.PointTransaction.debt_after,
            )
            .where(database.PointTransaction.user_id == user_id)
            .order_by(database.PointTransaction.id)
        )
        return [(row[0], row[1], row[2]) for row in result.all()]


async def _set_loan_interest(*, user_id: int, interest: int) -> None:
    """Test helper: backdoors a stored interest value onto a user's loan row."""
    async with database.open_session() as session:
        await session.execute(
            statement=update(database.UserAccount)
            .where(database.UserAccount.user_id == user_id)
            .values(loan_interest=interest)
        )
        await session.commit()


async def _backdate_last_accrual(*, user_id: int, days_ago: int) -> None:
    """Test helper: simulates time passing by pushing last_accrual_at back."""
    past = datetime.now(tz=database.TAIWAN_TIMEZONE) - timedelta(days=days_ago)
    async with database.open_session() as session:
        await session.execute(
            statement=update(database.UserAccount)
            .where(database.UserAccount.user_id == user_id)
            .values(loan_last_accrual_at=past)
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
    """Each tier boundary returns the expected cap."""
    user = _user_with_age(days_old=days)
    assert database.credit_limit(user=user) == expected


def test_accrual_delta_zero_principal_returns_zero() -> None:
    """Zero principal accrues no interest regardless of elapsed time."""
    now = datetime.now(tz=UTC)
    delta = database.accrual_delta(principal=0, last_accrual_at=now - timedelta(days=10), now=now)
    assert delta == 0


def test_accrual_delta_clock_skew_returns_zero() -> None:
    """Now < last_accrual_at must not produce negative interest."""
    now = datetime.now(tz=UTC)
    delta = database.accrual_delta(
        principal=10_000, last_accrual_at=now + timedelta(hours=1), now=now
    )
    assert delta == 0


@pytest.mark.parametrize(
    argnames=("principal", "days", "expected"),
    argvalues=[
        (10_000, 1, 100),
        (10_000, 7, 700),
        (10_000, 30, 3_000),
        (500_000, 1, 5_000),
        (500_000, 30, 150_000),
        (100, 0.001, 0),
    ],
)
def test_accrual_delta_simple_interest(principal: int, days: float, expected: int) -> None:
    """1% per day simple interest, integer-floored."""
    now = datetime.now(tz=UTC)
    delta = database.accrual_delta(
        principal=principal, last_accrual_at=now - timedelta(days=days), now=now
    )
    assert delta == expected


# Borrow --------------------------------------------------------------------


async def test_borrow_first_time_creates_row_and_disburses() -> None:
    """A first-time borrow creates the row, sets opened_at, and credits balance."""
    result = await database.borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    assert result is not None
    assert result.new_balance == 500
    assert result.principal == 500
    assert result.interest == 0

    view = await database.get_loan_view(user_id=1)
    assert view is not None
    assert view.principal == 500
    assert view.total_borrowed == 500
    assert view.opened_at is not None
    assert view.last_accrual_at is not None


async def test_borrow_accumulates_principal_on_existing_loan() -> None:
    """A second borrow adds to existing principal without resetting opened_at."""
    first = await database.borrow(user_id=1, name="alice", amount=300, credit_limit_value=1_000)
    assert first is not None
    first_view = await database.get_loan_view(user_id=1)
    assert first_view is not None
    first_opened_at = first_view.opened_at

    result = await database.borrow(user_id=1, name="alice", amount=400, credit_limit_value=1_000)
    assert result is not None
    assert result.new_balance == 700
    assert result.principal == 700
    view = await database.get_loan_view(user_id=1)
    assert view is not None
    assert view.opened_at == first_opened_at
    assert view.total_borrowed == 700


async def test_borrow_over_limit_rejected() -> None:
    """A borrow that would exceed the cap is rejected without side effects."""
    await database.borrow(user_id=1, name="alice", amount=800, credit_limit_value=1_000)
    result = await database.borrow(user_id=1, name="alice", amount=300, credit_limit_value=1_000)
    assert result is None
    view = await database.get_loan_view(user_id=1)
    assert view is not None
    assert view.principal == 800


async def test_borrow_treats_borrowed_money_as_debt_not_earnings() -> None:
    """Borrowed money goes into balance but does not bump total_earned."""
    await database.borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    account = await database.get_account(user_id=1)
    assert account is not None
    _, balance, total_earned, _ = account
    assert balance == 500
    assert total_earned == 0


async def test_borrow_on_existing_account_preserves_total_earned() -> None:
    """Borrowing on top of an existing balance doesn't disturb earnings history."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    result = await database.borrow(user_id=1, name="alice", amount=400, credit_limit_value=1_000)
    assert result is not None
    assert result.new_balance == 500
    account = await database.get_account(user_id=1)
    assert account is not None
    _, balance, total_earned, _ = account
    assert balance == 500
    assert total_earned == 100


@pytest.mark.parametrize(argnames="amount", argvalues=[0, -1, -1000])
async def test_borrow_rejects_non_positive_amount(amount: int) -> None:
    """Non-positive borrow amounts are rejected."""
    result = await database.borrow(
        user_id=1, name="alice", amount=amount, credit_limit_value=1_000
    )
    assert result is None


# Repay ---------------------------------------------------------------------


async def test_repay_pays_interest_before_principal() -> None:
    """Interest is paid down before principal."""
    await database.borrow(user_id=1, name="alice", amount=1_000, credit_limit_value=10_000)
    await _set_loan_interest(user_id=1, interest=200)
    result = await database.repay(user_id=1, name="alice", amount=300)
    assert result is not None
    assert result.interest_repaid == 200
    assert result.principal_repaid == 100
    assert result.remaining_debt == 900


async def test_repay_clamps_to_debt_total() -> None:
    """Over-request only repays up to the user's debt."""
    await database.borrow(user_id=1, name="alice", amount=500, credit_limit_value=10_000)
    result = await database.repay(user_id=1, name="alice", amount=10_000)
    assert result is not None
    assert result.principal_repaid == 500
    assert result.interest_repaid == 0
    assert result.remaining_debt == 0
    assert result.new_balance == 0


async def test_repay_clamps_to_balance_when_balance_smaller() -> None:
    """When balance < debt, repayment caps at balance."""
    await database.borrow(user_id=1, name="alice", amount=1_000, credit_limit_value=10_000)
    placed = await database.place_bet(user_id=1, name="alice", requested_bet=600)
    assert placed is not None
    result = await database.repay(user_id=1, name="alice", amount=10_000)
    assert result is not None
    assert result.principal_repaid + result.interest_repaid == 400
    assert result.new_balance == 0
    assert result.remaining_debt == 600


async def test_repay_no_debt_returns_none() -> None:
    """No outstanding debt → repayment rejected."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    assert await database.repay(user_id=1, name="alice", amount=50) is None


async def test_repay_zero_balance_returns_none() -> None:
    """Zero balance → no funds to repay with, even when debt remains."""
    await database.borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    placed = await database.place_bet(user_id=1, name="alice", requested_bet=500)
    assert placed is not None
    assert await database.repay(user_id=1, name="alice", amount=100) is None


async def test_repay_negative_balance_returns_none() -> None:
    """Negative casino balance is not repayable cash."""
    await database.borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await database.apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-600,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=600,
    )
    assert await database.repay(user_id=1, name="alice", amount=100) is None


async def test_repay_does_not_bump_total_spent() -> None:
    """Repaying a loan must not pollute the gameplay-spent counter."""
    await database.borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await database.repay(user_id=1, name="alice", amount=200)
    account = await database.get_account(user_id=1)
    assert account is not None
    _, _, _, total_spent = account
    assert total_spent == 0


@pytest.mark.parametrize(argnames="amount", argvalues=[0, -1])
async def test_repay_rejects_non_positive_amount(amount: int) -> None:
    """Non-positive repay amounts are rejected."""
    await database.borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    assert await database.repay(user_id=1, name="alice", amount=amount) is None


# credit_with_repayment ----------------------------------------------------


async def test_credit_with_repayment_full_credit_when_no_debt() -> None:
    """Without debt, the full amount lands in balance."""
    result = await database.credit_with_repayment(
        user_id=1, name="alice", amount=100, kind=TransactionKind.CHAT_REWARD
    )
    assert result.new_balance == 100
    assert result.credited_amount == 100
    assert result.interest_repaid == 0
    assert result.principal_repaid == 0
    assert result.remaining_debt == 0


async def test_credit_with_repayment_diverts_50_percent_interest_first() -> None:
    """With debt, 50% repays interest before touching principal."""
    await database.borrow(user_id=1, name="alice", amount=1_000, credit_limit_value=10_000)
    await _set_loan_interest(user_id=1, interest=200)
    result = await database.credit_with_repayment(
        user_id=1, name="alice", amount=200, kind=TransactionKind.CHAT_REWARD
    )
    # 50% of 200 = 100 → all 100 paid to interest (200 outstanding)
    assert result.interest_repaid == 100
    assert result.principal_repaid == 0
    assert result.credited_amount == 100
    assert result.remaining_debt == 100 + 1_000


async def test_credit_with_repayment_spills_to_principal_after_interest() -> None:
    """If 50% exceeds interest, the rest pays down principal."""
    await database.borrow(user_id=1, name="alice", amount=1_000, credit_limit_value=10_000)
    await _set_loan_interest(user_id=1, interest=30)
    result = await database.credit_with_repayment(
        user_id=1, name="alice", amount=200, kind=TransactionKind.CHAT_REWARD
    )
    # 50% of 200 = 100; 30 to interest, 70 to principal
    assert result.interest_repaid == 30
    assert result.principal_repaid == 70
    assert result.credited_amount == 100
    assert result.remaining_debt == 1_000 - 70


async def test_credit_with_repayment_caps_at_debt_total() -> None:
    """Repayment clamps to total debt when 50% exceeds outstanding."""
    await database.borrow(user_id=1, name="alice", amount=50, credit_limit_value=1_000)
    placed = await database.place_bet(user_id=1, name="alice", requested_bet=50)
    assert placed is not None
    result = await database.credit_with_repayment(
        user_id=1, name="alice", amount=1_000, kind=TransactionKind.CHAT_REWARD
    )
    assert result.principal_repaid == 50
    assert result.interest_repaid == 0
    assert result.credited_amount == 950
    assert result.remaining_debt == 0
    assert result.new_balance == 950


async def test_credit_with_repayment_floors_odd_amount() -> None:
    """Floor-division of odd amount: 11 // 2 = 5 to repayment, 6 to balance."""
    await database.borrow(user_id=1, name="alice", amount=100, credit_limit_value=1_000)
    result = await database.credit_with_repayment(
        user_id=1, name="alice", amount=11, kind=TransactionKind.CHAT_REWARD
    )
    assert result.principal_repaid == 5
    assert result.credited_amount == 6


async def test_credit_with_repayment_zero_amount_is_noop() -> None:
    """Zero amount returns current balance without writing or logging."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    result = await database.credit_with_repayment(
        user_id=1, name="alice", amount=0, kind=TransactionKind.CHAT_REWARD
    )
    assert result.new_balance == 100
    assert result.credited_amount == 0
    rows = await _list_transactions(user_id=1)
    assert rows == []


async def test_credit_with_repayment_first_sight_creates_row() -> None:
    """An unknown user receiving credit gets a fresh row."""
    result = await database.credit_with_repayment(
        user_id=42, name="newcomer", amount=200, kind=TransactionKind.CHAT_REWARD
    )
    assert result.new_balance == 200
    assert result.credited_amount == 200
    assert await database.get_balance(user_id=42) == 200


async def test_credit_with_repayment_concurrent_credits_accumulate() -> None:
    """Concurrent credits on the same user must not lose updates."""
    await database.add_balance(user_id=1, name="alice", amount=0)
    await asyncio.gather(*[
        database.credit_with_repayment(
            user_id=1, name="alice", amount=10, kind=TransactionKind.CHAT_REWARD
        )
        for _ in range(10)
    ])
    assert await database.get_balance(user_id=1) == 100


# Interest accrual integration ---------------------------------------------


async def test_borrow_accrues_pending_interest_before_limit_check() -> None:
    """A follow-up borrow includes pending interest in the limit check."""
    await database.borrow(user_id=1, name="alice", amount=1_000, credit_limit_value=10_000)
    await _backdate_last_accrual(user_id=1, days_ago=7)
    result = await database.borrow(user_id=1, name="alice", amount=100, credit_limit_value=10_000)
    assert result is not None
    assert result.interest == 70


async def test_repay_accrues_pending_interest_first() -> None:
    """A repay applied after time has passed pays down freshly accrued interest."""
    await database.borrow(user_id=1, name="alice", amount=1_000, credit_limit_value=10_000)
    await _backdate_last_accrual(user_id=1, days_ago=10)
    result = await database.repay(user_id=1, name="alice", amount=50)
    assert result is not None
    assert result.interest_repaid == 50
    assert result.principal_repaid == 0
    assert result.remaining_debt == (1_000 + 100) - 50


# get_loan_view ------------------------------------------------------------


async def test_get_loan_view_unknown_user_returns_none() -> None:
    """An unknown user has no row to project."""
    assert await database.get_loan_view(user_id=999) is None


async def test_get_loan_view_account_without_loan_returns_zero_state() -> None:
    """A user with a balance row but no loan returns a fresh-zero snapshot."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    view = await database.get_loan_view(user_id=1)
    assert view is not None
    assert view.principal == 0
    assert view.interest_stored == 0
    assert view.last_accrual_at is None
    assert view.opened_at is None
    assert view.total_borrowed == 0
    assert view.total_repaid == 0


async def test_get_loan_view_after_borrow_and_repay_tracks_totals() -> None:
    """Gross flows persist in total_borrowed / total_repaid across operations."""
    await database.borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await database.repay(user_id=1, name="alice", amount=200)
    view = await database.get_loan_view(user_id=1)
    assert view is not None
    assert view.principal == 300
    assert view.total_borrowed == 500
    assert view.total_repaid == 200


# Audit log ----------------------------------------------------------------


async def test_borrow_logs_audit_row() -> None:
    """Borrow writes one BORROW row with positive delta and post-state debt."""
    await database.borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    rows = await _list_transactions(user_id=1)
    assert rows == [(TransactionKind.BORROW.value, 500, 500)]


async def test_repay_logs_audit_row() -> None:
    """Repay writes one REPAY row with negative delta and reduced debt."""
    await database.borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await database.repay(user_id=1, name="alice", amount=200)
    rows = await _list_transactions(user_id=1)
    assert rows == [
        (TransactionKind.BORROW.value, 500, 500),
        (TransactionKind.REPAY.value, -200, 300),
    ]


async def test_chat_reward_logs_credited_slice_with_debt_context() -> None:
    """credit_with_repayment logs delta=credited and debt_after=post-repay debt."""
    await database.borrow(user_id=1, name="alice", amount=500, credit_limit_value=1_000)
    await database.credit_with_repayment(
        user_id=1, name="alice", amount=100, kind=TransactionKind.CHAT_REWARD
    )
    rows = await _list_transactions(user_id=1)
    # 50 paid to principal, 50 credited to balance; post-state debt 450
    assert rows == [
        (TransactionKind.BORROW.value, 500, 500),
        (TransactionKind.CHAT_REWARD.value, 50, 450),
    ]


async def test_transfer_logs_both_sides() -> None:
    """Transfer writes TRANSFER_OUT for sender and TRANSFER_IN for receiver."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    await database.transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=40
    )
    assert await _list_transactions(user_id=1) == [(TransactionKind.TRANSFER_OUT.value, -40, 0)]
    assert await _list_transactions(user_id=2) == [(TransactionKind.TRANSFER_IN.value, 40, 0)]


async def test_transfer_log_note_captures_counterparty() -> None:
    """Transfer audit rows carry the counterparty identity in ``note``."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    await database.transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=40
    )
    async with database.open_session() as session:
        result = await session.execute(
            statement=select(
                database.PointTransaction.user_id, database.PointTransaction.note
            ).order_by(database.PointTransaction.id)
        )
        notes = result.all()
    assert (1, "to bob (2)") in notes
    assert (2, "from alice (1)") in notes


async def test_place_bet_logs_casino_bet() -> None:
    """place_bet writes a CASINO_BET row with negative delta."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    await database.place_bet(user_id=1, name="alice", requested_bet=40)
    assert await _list_transactions(user_id=1) == [(TransactionKind.CASINO_BET.value, -40, 0)]


async def test_apply_round_settlement_logs_payout_and_house_settle() -> None:
    """Player and dealer sides of a round each produce one audit row."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    await database.apply_round_settlement(
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
    await database.add_balance(user_id=1, name="alice", amount=100)
    await database.apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=0,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=0,
    )
    assert await _list_transactions(user_id=99) == []


async def test_add_balance_does_not_log() -> None:
    """add_balance is a low-level primitive and stays out of the audit log."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    assert await _list_transactions(user_id=1) == []


async def test_settle_game_logs_applied_delta_not_requested_delta() -> None:
    """A loss bigger than the balance logs only what was actually clamped."""
    await database.add_balance(user_id=1, name="alice", amount=10)
    await database.settle_game(user_id=1, name="alice", delta=-1000)
    rows = await _list_transactions(user_id=1)
    # Clamp at 0 means only 10 actually left the account
    assert rows == [(TransactionKind.CASINO_PAYOUT.value, -10, 0)]
