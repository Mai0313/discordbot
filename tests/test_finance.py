"""Tests for long-term lending, central bank lending, and player stocks."""

from datetime import timedelta

import pytest
from sqlalchemy import update

from discordbot.typings.economy import STOCK_ISSUE_MIN_NET_WORTH
from discordbot.cogs._economy.database import (
    LoanContract,
    buy_stock,
    get_account,
    get_balance,
    issue_stock,
    open_session,
    _database_now,
    adjust_balance,
    pay_stock_dividend,
    set_central_banker,
    accept_loan_proposal,
    repay_personal_loans,
    call_central_bank_loans,
    get_central_bank_status,
    create_personal_loan_request,
    create_central_bank_loan_request,
)

pytestmark = pytest.mark.usefixtures("economy_isolated_db")


async def _add_balance(user_id: int, name: str, amount: int) -> int:
    """Seeds spendable balance through the public adjustment path."""
    result = await adjust_balance(user_id=user_id, name=name, delta=amount)
    return result.new_balance


async def _backdate_contract(contract_id: int, days: int) -> None:
    """Moves a loan contract's accrual timestamp into the past."""
    async with open_session() as session:
        await session.execute(
            statement=update(LoanContract)
            .where(LoanContract.id == contract_id)
            .values(last_interest_accrued_at=_database_now() - timedelta(days=days))
        )
        await session.commit()


async def test_personal_loan_request_accepts_and_repay_allocates_interest_first() -> None:
    """Accepted personal request debits lender, credits borrower, and repays interest first."""
    await _add_balance(user_id=2, name="bob", amount=1_000)

    proposal = await create_personal_loan_request(
        borrower_id=1,
        borrower_name="alice",
        lender_id=2,
        lender_name="bob",
        amount=500,
        monthly_rate_bps=300,
    )
    assert proposal is not None
    accepted = await accept_loan_proposal(
        proposal_id=proposal.proposal_id, actor_id=2, actor_name="bob"
    )
    assert accepted is not None
    assert accepted.borrower_balance == 500
    assert accepted.lender_balance == 500

    await _backdate_contract(contract_id=accepted.contract.contract_id, days=30)
    result = await repay_personal_loans(
        borrower_id=1, borrower_name="alice", lender_id=2, amount=100
    )

    assert result is not None
    assert result.paid_amount == 100
    assert result.interest_paid == 15
    assert result.principal_paid == 85
    assert result.remaining_principal == 415
    assert result.remaining_interest == 0
    assert await get_balance(user_id=1) == 400
    assert await get_balance(user_id=2) == 600


async def test_central_bank_loan_approves_against_cap_and_call_clamps_to_balance() -> None:
    """Central bank loans mint on approval and forced collection never drives balance negative."""
    await _add_balance(user_id=10, name="capital", amount=1_000)
    assert await set_central_banker(user_id=99, name="banker", is_central_banker=True)
    proposal = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=500, monthly_rate_bps=300
    )
    assert proposal is not None

    accepted = await accept_loan_proposal(
        proposal_id=proposal.proposal_id, actor_id=99, actor_name="banker", is_central_banker=True
    )
    assert accepted is not None
    assert accepted.borrower_balance == 500
    await _backdate_contract(contract_id=accepted.contract.contract_id, days=30)

    result = await call_central_bank_loans(borrower_id=1, borrower_name="alice", amount=None)
    status = await get_central_bank_status()

    assert result is not None
    assert result.paid_amount == 500
    assert result.interest_paid == 15
    assert result.principal_paid == 485
    assert result.remaining_principal == 15
    assert result.borrower_balance == 0
    assert status.outstanding_principal == 15


async def test_central_bank_capacity_decreases_after_approval() -> None:
    """Central bank loans cannot reuse minted balances as fresh lending capacity."""
    await _add_balance(user_id=10, name="capital", amount=1_000)
    first = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=600
    )
    assert first is not None
    accepted = await accept_loan_proposal(
        proposal_id=first.proposal_id, actor_id=99, actor_name="banker", is_central_banker=True
    )
    assert accepted is not None
    assert accepted.central_bank_available_credit == 400

    too_large = await create_central_bank_loan_request(
        borrower_id=2, borrower_name="bob", amount=500
    )
    assert too_large is not None
    rejected = await accept_loan_proposal(
        proposal_id=too_large.proposal_id, actor_id=99, actor_name="banker", is_central_banker=True
    )
    assert rejected is None


async def test_central_bank_self_approval_requires_explicit_flag() -> None:
    """Central bank self-approval stays blocked unless the caller explicitly opts in."""
    await _add_balance(user_id=10, name="capital", amount=1_000)
    blocked = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=100
    )
    assert blocked is not None
    assert (
        await accept_loan_proposal(
            proposal_id=blocked.proposal_id, actor_id=1, actor_name="alice", is_central_banker=True
        )
        is None
    )

    allowed = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=100
    )
    assert allowed is not None
    accepted = await accept_loan_proposal(
        proposal_id=allowed.proposal_id,
        actor_id=1,
        actor_name="alice",
        is_central_banker=True,
        allow_central_bank_self_approval=True,
    )
    assert accepted is not None
    assert accepted.borrower_balance == 100


async def test_forced_collection_without_amount_includes_accrued_interest() -> None:
    """Calling all owed accrues interest before deciding the collection amount."""
    await _add_balance(user_id=10, name="capital", amount=1_000)
    proposal = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=500, monthly_rate_bps=300
    )
    assert proposal is not None
    accepted = await accept_loan_proposal(
        proposal_id=proposal.proposal_id, actor_id=99, actor_name="banker", is_central_banker=True
    )
    assert accepted is not None
    await _add_balance(user_id=1, name="alice", amount=100)
    await _backdate_contract(contract_id=accepted.contract.contract_id, days=30)

    result = await call_central_bank_loans(borrower_id=1, borrower_name="alice", amount=None)

    assert result is not None
    assert result.paid_amount == 515
    assert result.interest_paid == 15
    assert result.principal_paid == 500
    assert result.closed_contract_ids == (accepted.contract.contract_id,)
    assert await get_balance(user_id=1) == 85


async def test_stock_issue_buy_and_dividend_keep_account_totals_aligned() -> None:
    """Stock purchase finances issuer and dividend pays sold-share holders."""
    await _add_balance(user_id=3, name="almost", amount=STOCK_ISSUE_MIN_NET_WORTH)
    assert await issue_stock(issuer_id=3, issuer_name="almost", shares=100, price=10) is None

    await _add_balance(user_id=1, name="issuer", amount=STOCK_ISSUE_MIN_NET_WORTH + 1)
    await _add_balance(user_id=2, name="buyer", amount=1_000)

    profile = await issue_stock(issuer_id=1, issuer_name="issuer", shares=100, price=10)
    assert profile is not None
    purchase = await buy_stock(buyer_id=2, buyer_name="buyer", issuer_id=1, shares=5)
    assert purchase is not None
    assert purchase.total_cost == 50
    assert purchase.treasury_shares == 95

    dividend = await pay_stock_dividend(issuer_id=1, issuer_name="issuer", amount=100)
    assert dividend is not None
    assert dividend.distributed_amount == 100
    assert dividend.recipient_count == 1
    issuer = await get_account(user_id=1)
    buyer = await get_account(user_id=2)
    assert issuer is not None
    assert buyer is not None
    assert issuer.total_earned - issuer.total_spent == issuer.balance
    assert buyer.total_earned - buyer.total_spent == buyer.balance
    assert buyer.balance == 1_050
