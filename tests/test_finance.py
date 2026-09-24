"""Tests for long-term lending and central bank lending."""

from types import SimpleNamespace
import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import text, select, update

from discordbot.typings.economy import (
    MIN_INTEREST_DAYS,
    CENTRAL_BANK_BASE_CAPACITY,
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
    LoanProposalStatus,
    LoanProposalAcceptResult,
)
from discordbot.cogs.economy.cog import EconomyCogs
from discordbot.services.economy.database import (
    LoanContract,
    LoanProposal,
    transfer,
    get_balance,
    open_session,
    _database_now,
    adjust_balance,
    get_credit_ceiling,
    accept_loan_proposal,
    repay_personal_loans,
    call_central_bank_loans,
    get_central_bank_status,
    record_guild_participant,
    create_personal_loan_request,
    reject_expired_loan_proposal,
    create_central_bank_loan_request,
)

from tests.helpers.casting import as_bot, as_interaction
from tests.helpers.discord_mocks import FakeUser, FakeInteraction

pytestmark = pytest.mark.usefixtures("economy_isolated_db")

# The guild every central-bank test lends in. Capacity is per guild now, so a borrower who
# takes part in none of them has no pool to draw on and no administrator who may approve.
GUILD = 555
OTHER_GUILD = 777


async def _add_balance(user_id: int, name: str, amount: int) -> int:
    """Seeds spendable balance through the public adjustment path."""
    result = await adjust_balance(user_id=user_id, name=name, delta=amount)
    return result.new_balance


async def _join(user_id: int, name: str, amount: int, guild_id: int = GUILD) -> int:
    """Seeds a balance and records the user as taking part in `guild_id`."""
    balance = await _add_balance(user_id=user_id, name=name, amount=amount)
    await record_guild_participant(guild_id=guild_id, user_id=user_id)
    return balance


async def _approve(
    proposal_id: int, actor_id: int, name: str, allow_self_approval: bool = False
) -> LoanProposalAcceptResult | None:
    """Approves a proposal as a server administrator of `GUILD`."""
    return await accept_loan_proposal(
        proposal_id=proposal_id,
        actor_id=actor_id,
        actor_name=name,
        approver_is_guild_admin=True,
        guild_id=GUILD,
        allow_central_bank_self_approval=allow_self_approval,
    )


async def _backdate_contract(contract_id: int, days: int) -> None:
    """Ages a loan contract by `days`, keeping the MIN_INTEREST_DAYS prepaid window aligned."""
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
    """Moves a loan proposal's creation timestamp into the past."""
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
    """Loan proposal and contract money columns can exceed SQLite's INTEGER range."""
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
    """Expired pending requests become rejected and cannot be accepted later."""
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
    await _join(user_id=1, name="alice", amount=1_000)
    proposal = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=500, monthly_rate_bps=300
    )
    assert proposal is not None

    accepted = await _approve(proposal_id=proposal.proposal_id, actor_id=99, name="banker")
    assert accepted is not None
    assert accepted.borrower_balance == 1_500
    # Spent back down, so the collection below meets a balance smaller than the debt.
    await adjust_balance(user_id=1, name="alice", delta=-1_000)
    await _backdate_contract(contract_id=accepted.contract.contract_id, days=30)

    result = await call_central_bank_loans(
        guild_id=GUILD, borrower_id=1, borrower_name="alice", amount=None
    )
    status = await get_central_bank_status(guild_id=GUILD)

    assert result is not None
    assert result.paid_amount == 500
    assert result.interest_paid == 15
    assert result.principal_paid == 485
    assert result.remaining_principal == 15
    assert result.borrower_balance == 0
    assert status.outstanding_principal == 15


async def test_central_bank_capacity_decreases_after_approval() -> None:
    """Central bank loans cannot reuse minted balances as fresh lending capacity.

    Sized past `CENTRAL_BANK_BASE_CAPACITY` on purpose: below it the flat floor every
    guild carries is larger than the participants' own money, so the pool never binds
    and the refusal being checked here would come from the borrower's ceiling instead.
    """
    await _join(user_id=1, name="alice", amount=6_000_000)
    first = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=6_000_000
    )
    assert first is not None
    accepted = await _approve(proposal_id=first.proposal_id, actor_id=99, name="banker")
    assert accepted is not None
    assert accepted.central_bank_available_credit == CENTRAL_BANK_BASE_CAPACITY

    # Inside alice's own remaining ceiling of 6,000,000, so only the pool can refuse it.
    too_large = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=CENTRAL_BANK_BASE_CAPACITY + 1
    )
    assert too_large is not None
    assert await get_credit_ceiling(user_id=1) > CENTRAL_BANK_BASE_CAPACITY
    rejected = await _approve(proposal_id=too_large.proposal_id, actor_id=99, name="banker")
    assert rejected is None


async def test_central_bank_concurrent_approvals_do_not_exceed_capacity() -> None:
    """Concurrent central-bank approvals serialize capacity consumption."""
    await _join(user_id=1, name="alice", amount=6_000_000)
    first = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=8_000_000
    )
    second = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=8_000_000
    )
    assert first is not None
    assert second is not None

    first_result, second_result = await asyncio.gather(
        _approve(proposal_id=first.proposal_id, actor_id=99, name="banker"),
        _approve(proposal_id=second.proposal_id, actor_id=98, name="banker2"),
    )
    accepted_results = [result for result in (first_result, second_result) if result is not None]
    status = await get_central_bank_status(guild_id=GUILD)

    assert len(accepted_results) == 1
    assert status.outstanding_principal == 8_000_000
    # 14,000,000 of balance once the loan is counted, less the 8,000,000 it minted, plus the
    # base capacity, less the same 8,000,000 as capacity already spent.
    assert status.available_credit == 6_000_000 + CENTRAL_BANK_BASE_CAPACITY - 8_000_000


async def test_central_bank_self_approval_requires_explicit_flag() -> None:
    """Central bank self-approval stays blocked unless the caller explicitly opts in."""
    await _join(user_id=1, name="alice", amount=1_000)
    blocked = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=100
    )
    assert blocked is not None
    assert await _approve(proposal_id=blocked.proposal_id, actor_id=1, name="alice") is None

    allowed = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=100
    )
    assert allowed is not None
    accepted = await _approve(
        proposal_id=allowed.proposal_id, actor_id=1, name="alice", allow_self_approval=True
    )
    assert accepted is not None
    assert accepted.borrower_balance == 1_100


async def test_forced_collection_without_amount_includes_accrued_interest() -> None:
    """Calling all owed accrues interest before deciding the collection amount."""
    await _join(user_id=1, name="alice", amount=1_000)
    proposal = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=500, monthly_rate_bps=300
    )
    assert proposal is not None
    accepted = await _approve(proposal_id=proposal.proposal_id, actor_id=99, name="banker")
    assert accepted is not None
    await _backdate_contract(contract_id=accepted.contract.contract_id, days=30)

    result = await call_central_bank_loans(
        guild_id=GUILD, borrower_id=1, borrower_name="alice", amount=None
    )

    assert result is not None
    assert result.paid_amount == 515
    assert result.interest_paid == 15
    assert result.principal_paid == 500
    assert result.closed_contract_ids == (accepted.contract.contract_id,)
    assert await get_balance(user_id=1) == 985


async def test_a_borrower_cannot_owe_more_than_their_own_ceiling() -> None:
    """The per-borrower ceiling is what bounds minting, not the guild pool.

    Borrowing lowers it by exactly what was borrowed, so it cannot be walked upward by
    re-borrowing against the balance the previous loan minted.
    """
    await _join(user_id=1, name="alice", amount=1_000)
    assert await get_credit_ceiling(user_id=1) == 2_000

    first = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=2_000, monthly_rate_bps=0
    )
    assert first is not None
    assert await _approve(proposal_id=first.proposal_id, actor_id=99, name="banker") is not None
    # Balance is now 3,000 against 2,000 of debt, and the ceiling is spent rather than tripled.
    assert await get_balance(user_id=1) == 3_000
    assert await get_credit_ceiling(user_id=1) == 0

    second = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=1, monthly_rate_bps=0
    )
    assert second is not None
    assert await _approve(proposal_id=second.proposal_id, actor_id=99, name="banker") is None


async def test_handing_a_minted_balance_to_a_second_account_runs_the_pool_down() -> None:
    """The pool is what bounds a pair taking turns, and it has to reach zero to do it.

    A borrower who gives their balance away has a ceiling of zero either way, so the
    `max(..., 0)` stops charging them for the debt they still owe and their partner's own
    ceiling is untouched by it. Only the pool still counts that debt. Adding the base
    capacity outside the pool's subtraction pins it at a floor it can never fall through,
    which takes it out of the bounding job entirely: measured, 1,000 became 301,314 in
    eight rounds of borrow-then-`/give` and was still accelerating.
    """
    await _join(user_id=1, name="alice", amount=1_000)
    await _join(user_id=2, name="bob", amount=0)

    holder, partner = 1, 2
    for _ in range(40):
        ceiling = await get_credit_ceiling(user_id=holder)
        if ceiling > 0:
            proposal = await create_central_bank_loan_request(
                borrower_id=holder, borrower_name=str(holder), amount=ceiling, monthly_rate_bps=0
            )
            assert proposal is not None
            if await _approve(proposal_id=proposal.proposal_id, actor_id=99, name="admin") is None:
                break
        balance = await get_balance(user_id=holder)
        if balance > 0:
            await transfer(
                sender_id=holder,
                sender_name=str(holder),
                receiver_id=partner,
                receiver_name=str(partner),
                amount=balance,
            )
        holder, partner = partner, holder
    else:  # pragma: no cover -- only reached if the pool never refuses
        pytest.fail("the lending pool never ran out")

    status = await get_central_bank_status(guild_id=GUILD)
    assert status.available_credit < CENTRAL_BANK_BASE_CAPACITY
    minted = await get_balance(user_id=1) + await get_balance(user_id=2)
    assert minted < CENTRAL_BANK_BASE_CAPACITY


async def test_minting_is_bounded_when_each_account_holds_its_own_guild() -> None:
    """Three accounts in three guilds is where a per-guild debt term stops bounding anything.

    Balance crosses servers and debt does not follow it, so money minted in one guild and
    handed to somebody who takes part in another arrives as collateral with nothing owed
    against it. Charging each guild only its own participants' debt therefore never runs
    any pool down: measured on that version, 1,000 became 686,826,650,532 over these forty
    rounds and was still accelerating. The whole bank's outstanding principal is what has
    to be subtracted.
    """
    accounts = (1, 2, 3)
    await _add_balance(user_id=1, name="1", amount=1_000)
    for user_id in accounts:  # each account takes part in its own guild and no other
        await record_guild_participant(guild_id=user_id, user_id=user_id)

    holder = 1
    for _ in range(40):
        status = await get_central_bank_status(guild_id=holder)
        wanted = min(await get_credit_ceiling(user_id=holder), status.available_credit)
        if wanted > 0:
            proposal = await create_central_bank_loan_request(
                borrower_id=holder, borrower_name=str(holder), amount=wanted, monthly_rate_bps=0
            )
            assert proposal is not None
            await accept_loan_proposal(
                proposal_id=proposal.proposal_id,
                actor_id=99,
                actor_name="admin",
                approver_is_guild_admin=True,
                guild_id=holder,
            )
        balance = await get_balance(user_id=holder)
        following = holder % len(accounts) + 1
        if balance > 0:
            await transfer(
                sender_id=holder,
                sender_name=str(holder),
                receiver_id=following,
                receiver_name=str(following),
                amount=balance,
            )
        holder = following

    circulating = sum([await get_balance(user_id=user_id) for user_id in accounts])
    assert circulating < CENTRAL_BANK_BASE_CAPACITY


async def test_a_fully_leveraged_guild_has_no_capacity_left() -> None:
    """The base capacity is spent by lending rather than standing under it."""
    await _join(user_id=1, name="alice", amount=CENTRAL_BANK_BASE_CAPACITY * 2)
    opening = await get_central_bank_status(guild_id=GUILD)
    assert opening.available_credit == CENTRAL_BANK_BASE_CAPACITY * 3

    proposal = await create_central_bank_loan_request(
        borrower_id=1,
        borrower_name="alice",
        amount=CENTRAL_BANK_BASE_CAPACITY * 3,
        monthly_rate_bps=0,
    )
    assert proposal is not None
    assert await _approve(proposal_id=proposal.proposal_id, actor_id=99, name="admin") is not None

    assert (await get_central_bank_status(guild_id=GUILD)).available_credit == 0


async def test_an_untaxed_personal_loan_cannot_refill_the_ceiling() -> None:
    """A 0% personal loan moves balance without tax, so the ceiling has to count it.

    Counting only central-bank debt would leave this pair taking turns: each hop hands the
    minted balance to an account whose own ceiling looks untouched, and the pair doubles
    what they can mint every round at no cost.
    """
    await _join(user_id=1, name="alice", amount=1_000)
    await _join(user_id=2, name="bob", amount=0)

    minted = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=2_000, monthly_rate_bps=0
    )
    assert minted is not None
    assert await _approve(proposal_id=minted.proposal_id, actor_id=99, name="banker") is not None

    handover = await create_personal_loan_request(
        borrower_id=2,
        borrower_name="bob",
        lender_id=1,
        lender_name="alice",
        amount=3_000,
        monthly_rate_bps=0,
    )
    assert handover is not None
    accepted = await accept_loan_proposal(
        proposal_id=handover.proposal_id, actor_id=1, actor_name="alice"
    )
    assert accepted is not None

    # Bob holds every coin alice had, and none of it is free equity: he owes it all.
    assert await get_balance(user_id=2) == 3_000
    assert await get_credit_ceiling(user_id=2) == 0
    assert await get_credit_ceiling(user_id=1) == 0


async def test_central_bank_keeps_its_interest_and_lends_it_again() -> None:
    """Repaid interest is kept and adds to what the bank can lend; principal still disappears."""
    await _join(user_id=1, name="alice", amount=1_000)
    proposal = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=500, monthly_rate_bps=300
    )
    assert proposal is not None
    accepted = await _approve(proposal_id=proposal.proposal_id, actor_id=99, name="banker")
    assert accepted is not None
    assert (await get_central_bank_status(guild_id=GUILD)).ledger_balance == 0

    result = await call_central_bank_loans(
        guild_id=GUILD, borrower_id=1, borrower_name="alice", amount=None
    )

    assert result is not None
    assert result.interest_paid == 15
    assert (await get_central_bank_status(guild_id=GUILD)).ledger_balance == 15
    # A guild nobody takes part in lends on the bank's own capital alone, which now
    # includes the interest it just kept.
    nobody_here = await get_central_bank_status(guild_id=999)
    assert nobody_here.available_credit == CENTRAL_BANK_BASE_CAPACITY + 15


async def test_forced_collection_refuses_a_borrower_from_another_guild() -> None:
    """Collection is scoped to the calling guild's own participants.

    Approval is a server administrator's now and anyone can create a server to become one,
    so without this scope an administrator anywhere could sweep any borrower's balance.
    """
    await _join(user_id=1, name="alice", amount=1_000)
    proposal = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=500, monthly_rate_bps=300
    )
    assert proposal is not None
    accepted = await _approve(proposal_id=proposal.proposal_id, actor_id=99, name="banker")
    assert accepted is not None
    await _backdate_contract(contract_id=accepted.contract.contract_id, days=30)

    assert (
        await call_central_bank_loans(
            guild_id=OTHER_GUILD, borrower_id=1, borrower_name="alice", amount=None
        )
        is None
    )
    assert await get_balance(user_id=1) == 1_500


async def test_approval_needs_a_guild_and_an_administrator() -> None:
    """Neither half of the approval gate is optional, and a DM has neither."""
    await _join(user_id=1, name="alice", amount=1_000)
    without_guild = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=100
    )
    assert without_guild is not None
    assert (
        await accept_loan_proposal(
            proposal_id=without_guild.proposal_id,
            actor_id=99,
            actor_name="banker",
            approver_is_guild_admin=True,
            guild_id=None,
        )
        is None
    )
    assert (
        await accept_loan_proposal(
            proposal_id=without_guild.proposal_id,
            actor_id=99,
            actor_name="banker",
            approver_is_guild_admin=False,
            guild_id=GUILD,
        )
        is None
    )


async def test_a_request_over_the_ceiling_is_refused_before_anyone_is_asked() -> None:
    """The ceiling is checked at request time too, so no unanswerable panel is posted."""
    await _join(user_id=1, name="alice", amount=1_000)
    assert await get_credit_ceiling(user_id=1) == 2_000

    cog = EconomyCogs(bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999))))
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), guild_id=GUILD)
    await EconomyCogs.central_bank_borrow.callback(
        cog, as_interaction(fake=interaction), amount="2001", monthly_rate_percent=0.0
    )

    embed = interaction.followup.sent[0]["embed"]
    assert embed.title == "🏛️ 超過個人信用上限"
    assert "view" not in interaction.followup.sent[0]


@pytest.mark.parametrize(
    ("command", "kwargs", "title"),
    [
        ("central_bank_borrow", {"amount": "100", "monthly_rate_percent": 0.0}, "央行借款失敗"),
        ("central_bank_status", {}, "央行狀態"),
    ],
)
async def test_the_central_bank_refuses_a_direct_message(
    command: str, kwargs: dict[str, object], title: str
) -> None:
    """Outside a guild there is no pool to lend from and nobody who could approve.

    Refused up front rather than at the button, or borrowing in a DM would post a
    request that can be neither approved nor rejected and simply times out.
    """
    cog = EconomyCogs(bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999))))
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), in_guild=False)

    await getattr(EconomyCogs, command).callback(cog, as_interaction(fake=interaction), **kwargs)

    assert interaction.response.sent[0]["ephemeral"] is True
    assert interaction.response.sent[0]["embed"].title == title


async def test_each_guild_lends_against_its_own_participants() -> None:
    """One wallet backs every guild its owner takes part in, and no others."""
    await _join(user_id=1, name="alice", amount=6_000_000)
    await _join(user_id=2, name="bob", amount=4_000_000, guild_id=OTHER_GUILD)

    here = await get_central_bank_status(guild_id=GUILD)
    there = await get_central_bank_status(guild_id=OTHER_GUILD)
    nowhere = await get_central_bank_status(guild_id=999)

    assert here.participant_count == 1
    assert here.total_positive_user_balance == 6_000_000
    assert there.total_positive_user_balance == 4_000_000
    assert nowhere.participant_count == 0
    assert nowhere.available_credit == CENTRAL_BANK_BASE_CAPACITY
