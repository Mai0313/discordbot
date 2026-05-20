"""Tests for shared economy income, transfer, and settlement helpers."""

import asyncio

import pytest

from discordbot.typings.economy import TransactionKind
from discordbot.cogs._economy.database import (
    transfer,
    get_account,
    get_balance,
    adjust_balance,
    list_loan_contracts,
    accept_loan_proposal,
    credit_with_repayment,
    apply_round_settlement,
    apply_vip_blackjack_bonus,
    create_personal_loan_request,
)

pytestmark = pytest.mark.usefixtures("economy_isolated_db")


async def _add_balance(user_id: int, name: str, amount: int) -> int:
    """Seeds a positive balance through the public adjustment path."""
    result = await adjust_balance(user_id=user_id, name=name, delta=amount)
    return result.new_balance


# VIP blackjack bonus -------------------------------------------------------


@pytest.mark.parametrize(
    argnames=("delta", "is_vip", "expected"),
    argvalues=[
        (100, False, 100),
        (100, True, 150),
        (101, True, 151),
        (1, True, 1),
        (0, True, 0),
        (-50, True, -50),
    ],
)
def test_apply_vip_blackjack_bonus(delta: int, is_vip: bool, expected: int) -> None:
    """VIP bonus applies only to positive winnings and floors fractional halves."""
    assert apply_vip_blackjack_bonus(delta=delta, is_vip=is_vip) == expected


# credit_with_repayment ----------------------------------------------------


async def test_credit_with_repayment_full_credit() -> None:
    """Income credits the full amount and leaves repayment fields at zero."""
    result = await credit_with_repayment(
        user_id=1, name="alice", amount=100, kind=TransactionKind.CHAT_REWARD
    )

    assert result.new_balance == 100
    assert result.credited_amount == 100
    assert result.principal_repaid == 0
    assert result.remaining_debt == 0


async def test_credit_with_repayment_zero_amount_is_noop() -> None:
    """Non-positive reward calls do not create phantom income."""
    await _add_balance(user_id=1, name="alice", amount=50)

    result = await credit_with_repayment(
        user_id=1, name="alice", amount=0, kind=TransactionKind.CHAT_REWARD
    )

    assert result.new_balance == 50
    assert result.credited_amount == 0
    assert result.principal_repaid == 0
    assert result.remaining_debt == 0


async def test_credit_with_repayment_first_sight_creates_row() -> None:
    """A first reward creates the user account row."""
    result = await credit_with_repayment(
        user_id=1, name="alice", amount=200, kind=TransactionKind.MESSAGE_REWARD
    )

    assert result.credited_amount == 200
    assert await get_balance(user_id=1) == 200


async def test_credit_with_repayment_concurrent_credits_accumulate() -> None:
    """Concurrent reward writes add up instead of losing one update."""
    await asyncio.gather(
        *(
            credit_with_repayment(
                user_id=1, name="alice", amount=10, kind=TransactionKind.CHAT_REWARD
            )
            for _ in range(20)
        )
    )

    assert await get_balance(user_id=1) == 200


async def test_credit_with_repayment_does_not_touch_long_term_debt() -> None:
    """Passive income does not auto-repay explicit long-term loan contracts."""
    await _add_balance(user_id=2, name="bob", amount=1_000)
    proposal = await create_personal_loan_request(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=500
    )
    assert proposal is not None
    accepted = await accept_loan_proposal(
        proposal_id=proposal.proposal_id, actor_id=2, actor_name="bob"
    )
    assert accepted is not None

    result = await credit_with_repayment(
        user_id=1, name="alice", amount=100, kind=TransactionKind.CHAT_REWARD
    )
    contracts = await list_loan_contracts(user_id=1)

    assert result.new_balance == 600
    assert result.credited_amount == 100
    assert result.principal_repaid == 0
    assert len(contracts) == 1
    assert contracts[0].principal_remaining == 500


# Account totals -------------------------------------------------------------


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

    await adjust_balance(user_id=1, name="alice", delta=-1_000)

    account = await get_account(user_id=1)
    assert account is not None
    assert (account.balance, account.total_earned, account.total_spent) == (0, 10, 10)
