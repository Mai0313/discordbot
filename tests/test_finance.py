"""Tests for the long-term lending half of `services/economy/database.py`.

Personal loans and central-bank loans share one proposal -> contract -> payment path, and what
separates them is where the principal comes from: a personal loan DEBITS the lender at acceptance,
a central-bank loan MINTS with no lender side at all. That asymmetry is why the central-bank cases
dominate this file. The capacity read taken under the acceptance lock is the only thing between an
approval and unbounded minting, so the tests pin that a minted balance never comes back as fresh
capacity, that two concurrent approvals cannot both spend the same free credit, and that a banker
approving their own request stays refused unless the caller explicitly opts in.

The rest pins the arithmetic a borrower feels: acceptance prepays `MIN_INTEREST_DAYS` of interest
so an instant repayment still costs something, a payment clears interest before principal, and a
forced collection with no amount named takes principal plus everything owed in interest, clamped
at the borrower's balance instead of driving it negative. Interest is charged lazily against
wall-clock time and only for whole elapsed days, so `_backdate_contract` ages a contract in place
rather than the suite waiting a month, and `_backdate_proposal` does the same for the
`LOAN_PROPOSAL_TIMEOUT_SECONDS` decision window.

One test reaches past the ORM and asks SQLite for `typeof()` on the raw loan columns: loan money is
`StoredInteger` decimal text, and only a `text` typeof proves a 10**20 principal is not silently
riding SQLite's 64-bit INTEGER ceiling. Everything here runs against the throwaway ledger the
`economy_isolated_db` fixture installs; `tests/test_loan.py` covers the income and settlement
helpers that deliberately leave these contracts alone.
"""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import text, select, update

from discordbot.typings.economy import (
    MIN_INTEREST_DAYS,
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
    LoanProposalStatus,
)
from discordbot.services.economy.database import (
    LoanContract,
    LoanProposal,
    get_balance,
    open_session,
    _database_now,
    adjust_balance,
    set_central_banker,
    accept_loan_proposal,
    repay_personal_loans,
    call_central_bank_loans,
    get_central_bank_status,
    create_personal_loan_request,
    reject_expired_loan_proposal,
    create_central_bank_loan_request,
)

pytestmark = pytest.mark.usefixtures("economy_isolated_db")


async def _add_balance(user_id: int, name: str, amount: int) -> int:
    """Seeds spendable balance through the public adjustment path.

    Returns:
        The wallet balance after the credit.
    """
    result = await adjust_balance(user_id=user_id, name=name, delta=amount)
    return result.new_balance


async def _backdate_contract(contract_id: int, days: int) -> None:
    """Ages a loan contract by `days`, keeping the MIN_INTEREST_DAYS prepaid window aligned.

    Interest accrues from wall-clock time, so a test cannot wait for it. Anchoring last accrual to
    the end of the prepaid window leaves a 30-day backdate owing exactly what acceptance prepaid;
    only the days past `MIN_INTEREST_DAYS` accrue on top of it.
    """
    now = _database_now()
    opened_at = now - timedelta(days=days)
    last_accrued_at = opened_at + timedelta(days=MIN_INTEREST_DAYS)
    async with open_session() as session:
        await session.execute(
            statement=update(LoanContract)
            .where(LoanContract.id == contract_id)
            .values(opened_at=opened_at, last_interest_accrued_at=last_accrued_at)
        )
        await session.commit()


async def _backdate_proposal(proposal_id: int, seconds: int) -> None:
    """Moves a loan proposal's creation timestamp into the past.

    Expiry is evaluated lazily off `created_at` whenever a proposal is touched, so this is how a
    test crosses the decision window without sleeping out `LOAN_PROPOSAL_TIMEOUT_SECONDS`.
    """
    async with open_session() as session:
        await session.execute(
            statement=update(LoanProposal)
            .where(LoanProposal.id == proposal_id)
            .values(
                created_at=_database_now() - timedelta(seconds=seconds),
                updated_at=_database_now() - timedelta(seconds=seconds),
            )
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


async def test_personal_loan_money_columns_store_large_values_as_text() -> None:
    """Loan money columns hold a principal past SQLite's INTEGER range as decimal text."""
    large_amount = 10**20
    await _add_balance(user_id=10, name="lender", amount=large_amount)
    proposal = await create_personal_loan_request(
        borrower_id=20,
        borrower_name="borrower",
        lender_id=10,
        lender_name="lender",
        amount=large_amount,
    )
    assert proposal is not None

    accepted = await accept_loan_proposal(
        proposal_id=proposal.proposal_id, actor_id=10, actor_name="lender"
    )
    assert accepted is not None
    assert accepted.borrower_balance == large_amount
    assert accepted.lender_balance == 0
    assert accepted.contract.principal_remaining == large_amount

    async with open_session() as session:
        proposal_result = await session.execute(
            statement=text(
                text="""
                SELECT amount, typeof(amount), escrow_amount, typeof(escrow_amount)
                  FROM loan_proposal
                 WHERE id = :proposal_id
                """
            ),
            params={"proposal_id": proposal.proposal_id},
        )
        contract_result = await session.execute(
            statement=text(
                text="""
                SELECT original_principal, typeof(original_principal),
                       principal_remaining, typeof(principal_remaining),
                       interest_due, typeof(interest_due)
                  FROM loan_contract
                 WHERE id = :contract_id
                """
            ),
            params={"contract_id": accepted.contract.contract_id},
        )

    prepaid_interest = large_amount * 300 * MIN_INTEREST_DAYS // (10_000 * 30)
    assert proposal_result.one() == (str(large_amount), "text", "0", "text")
    assert contract_result.one() == (
        str(large_amount),
        "text",
        str(large_amount),
        "text",
        str(prepaid_interest),
        "text",
    )


async def test_expired_loan_request_rejects_without_debiting_lender() -> None:
    """An expired request rejects, refuses a later acceptance, and never debits the lender."""
    await _add_balance(user_id=2, name="bob", amount=1_000)
    proposal = await create_personal_loan_request(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=500
    )
    assert proposal is not None
    await _backdate_proposal(
        proposal_id=proposal.proposal_id, seconds=LOAN_PROPOSAL_TIMEOUT_SECONDS
    )

    expired = await reject_expired_loan_proposal(proposal_id=proposal.proposal_id)
    accepted = await accept_loan_proposal(
        proposal_id=proposal.proposal_id, actor_id=2, actor_name="bob"
    )

    async with open_session() as session:
        result = await session.execute(
            statement=select(LoanProposal.status).where(LoanProposal.id == proposal.proposal_id)
        )
        stored_status = result.scalar_one()

    assert expired is not None
    assert expired.status == LoanProposalStatus.REJECTED
    assert accepted is None
    assert stored_status == LoanProposalStatus.REJECTED
    assert await get_balance(user_id=1) == 0
    assert await get_balance(user_id=2) == 1_000


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


async def test_central_bank_concurrent_approvals_do_not_exceed_capacity() -> None:
    """Two concurrent central-bank approvals cannot both spend the same free credit."""
    await _add_balance(user_id=10, name="capital", amount=1_000)
    first = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=800
    )
    second = await create_central_bank_loan_request(borrower_id=2, borrower_name="bob", amount=800)
    assert first is not None
    assert second is not None

    first_result, second_result = await asyncio.gather(
        accept_loan_proposal(
            proposal_id=first.proposal_id, actor_id=99, actor_name="banker", is_central_banker=True
        ),
        accept_loan_proposal(
            proposal_id=second.proposal_id,
            actor_id=98,
            actor_name="banker2",
            is_central_banker=True,
        ),
    )
    accepted_results = [result for result in (first_result, second_result) if result is not None]
    status = await get_central_bank_status()

    assert len(accepted_results) == 1
    assert status.outstanding_principal == 800
    assert status.available_credit == 200


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
    """Collecting with no amount takes principal plus interest owed and closes the contract."""
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
