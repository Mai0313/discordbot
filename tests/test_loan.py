"""Pins the shared income facade and the lifetime totals every economy write path has to keep.

`services/economy/database.py` carries no transaction table, so `balance == total_earned -
total_spent` per side is the only self-check the ledger owns, and each path exercised here is one
where what the caller asked for and what was applied deliberately differ. `transfer` debits the
sender the gross amount but credits the receiver only the taxed net, so the burn has to leave
circulation on both halves of the identity; `apply_round_settlement` mirrors into the casino
ledger only the debit it actually collected, so a loss clamped at a player's balance cannot book
house income nobody paid; `adjust_balance` spends the clamped applied delta rather than the
requested one. A path that moved a balance and bumped the wrong lifetime total leaves nothing else
to catch it, which is why these three assert the whole triple rather than the balance alone.

`credit_with_repayment` is the single income facade — the per-message reward, `/checkin`, casino
payouts and fishing payouts all land in it — so the tests around it pin the full credit, the
zero-amount call that must not mint phantom income, the first-sight row creation, and twenty
concurrent credits accumulating instead of losing an update.

The one loan test is what the file is named after and the reason it stays separate from the
ledger's own suite: income used to auto-repay long-term debt and no longer does, so a real
proposal is opened and accepted purely to assert that a later credit lands entirely in the
borrower's balance and leaves `principal_remaining` untouched. The function still carries
`repayment` in its name, which is exactly what makes the negative worth a test instead of a
comment. The contracts themselves — capacity, interest, collection — belong to
`tests/test_finance.py`.

`apply_vip_blackjack_bonus` needs no ledger at all and is parametrized instead: the 1.2x perk
fires only on a win, and it floors, so it never mints the fraction of a point a 1-unit win would
otherwise round up to. Everything else runs against the throwaway ledger `economy_isolated_db`
installs.
"""

import asyncio

import pytest

from discordbot.services.economy.database import (
    transfer,
    get_account,
    get_balance,
    adjust_balance,
    get_casino_ledger,
    list_loan_contracts,
    accept_loan_proposal,
    credit_with_repayment,
    apply_round_settlement,
    apply_vip_blackjack_bonus,
    create_personal_loan_request,
)

pytestmark = pytest.mark.usefixtures("economy_isolated_db")


async def _add_balance(user_id: int, name: str, amount: int) -> int:
    """Seeds a positive balance through the public adjustment path.

    Going through `adjust_balance` rather than writing the row keeps the seeded `total_earned`
    consistent with the balance, so a test can assert the accounting identity on top of it.

    Returns:
        The post-adjustment balance.
    """
    result = await adjust_balance(user_id=user_id, name=name, delta=amount)
    return result.new_balance


# VIP blackjack bonus -------------------------------------------------------


@pytest.mark.parametrize(
    argnames=("delta", "is_vip", "expected"),
    argvalues=[
        (100, False, 100),
        (100, True, 120),
        (101, True, 121),
        (1, True, 1),
        (0, True, 0),
        (-50, True, -50),
    ],
)
def test_apply_vip_blackjack_bonus(delta: int, is_vip: bool, expected: int) -> None:
    """The VIP bonus lifts only a winning delta and floors the fifth it adds."""
    assert apply_vip_blackjack_bonus(delta=delta, is_vip=is_vip) == expected


# credit_with_repayment ----------------------------------------------------


async def test_credit_with_repayment_full_credit() -> None:
    """Income credits the full amount and reports no repayment against it."""
    result = await credit_with_repayment(user_id=1, name="alice", amount=100)

    assert result.new_balance == 100
    assert result.credited_amount == 100
    assert result.principal_repaid == 0
    assert result.remaining_debt == 0


async def test_credit_with_repayment_zero_amount_is_noop() -> None:
    """A zero-amount credit reports the standing balance and mints nothing."""
    await _add_balance(user_id=1, name="alice", amount=50)

    result = await credit_with_repayment(user_id=1, name="alice", amount=0)

    assert result.new_balance == 50
    assert result.credited_amount == 0
    assert result.principal_repaid == 0
    assert result.remaining_debt == 0


async def test_credit_with_repayment_first_sight_creates_row() -> None:
    """A credit for a never-seen user creates the wallet row rather than dropping the income."""
    result = await credit_with_repayment(user_id=1, name="alice", amount=200)

    assert result.credited_amount == 200
    assert await get_balance(user_id=1) == 200


async def test_credit_with_repayment_concurrent_credits_accumulate() -> None:
    """Concurrent credits all land instead of losing an update to a read-modify-write race."""
    await asyncio.gather(
        *(credit_with_repayment(user_id=1, name="alice", amount=10) for _ in range(20))
    )

    assert await get_balance(user_id=1) == 200


async def test_credit_with_repayment_does_not_touch_long_term_debt() -> None:
    """Income never auto-repays an accepted long-term loan contract."""
    await _add_balance(user_id=2, name="bob", amount=1_000)
    proposal = await create_personal_loan_request(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=500
    )
    assert proposal is not None
    accepted = await accept_loan_proposal(
        proposal_id=proposal.proposal_id, actor_id=2, actor_name="bob"
    )
    assert accepted is not None

    result = await credit_with_repayment(user_id=1, name="alice", amount=100)
    contracts = await list_loan_contracts(user_id=1)

    assert result.new_balance == 600
    assert result.credited_amount == 100
    assert result.principal_repaid == 0
    assert len(contracts) == 1
    assert contracts[0].principal_remaining == 500


# Account totals -------------------------------------------------------------


async def test_transfer_updates_sender_and_receiver_totals() -> None:
    """A transfer spends the sender the gross amount and earns the receiver only the taxed net."""
    await _add_balance(user_id=1, name="alice", amount=100)

    await transfer(sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=40)

    sender = await get_account(user_id=1)
    receiver = await get_account(user_id=2)
    assert sender is not None
    assert receiver is not None
    # 40 transferred, 5% (2) burned; receiver nets 38.
    assert (sender.balance, sender.total_earned, sender.total_spent) == (60, 100, 40)
    assert (receiver.balance, receiver.total_earned, receiver.total_spent) == (38, 38, 0)


async def test_apply_round_settlement_updates_player_and_casino_totals() -> None:
    """Casino settlement books the applied delta into both the player's and the ledger's totals."""
    await _add_balance(user_id=1, name="alice", amount=100)

    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-40, casino_delta=40
    )

    player = await get_account(user_id=1)
    ledger = await get_casino_ledger()
    assert player is not None
    assert (player.balance, player.total_earned, player.total_spent) == (60, 100, 40)
    assert (ledger.balance, ledger.total_earned, ledger.total_spent) == (40, 40, 0)


async def test_adjust_balance_counts_applied_delta_not_requested_delta() -> None:
    """A clamped manual debit spends only the delta it applied, not the one it was handed."""
    await _add_balance(user_id=1, name="alice", amount=10)

    await adjust_balance(user_id=1, name="alice", delta=-1_000)

    account = await get_account(user_id=1)
    assert account is not None
    assert (account.balance, account.total_earned, account.total_spent) == (0, 10, 10)
