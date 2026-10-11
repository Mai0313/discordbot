"""What a started Blackjack round holds of a wallet (`services/economy/holds.py`)."""

from types import SimpleNamespace
import asyncio
from collections.abc import Callable, Coroutine

import pytest
from pydantic import Field, BaseModel

from discordbot.typings.economy import VIP_PURCHASE_COST
from discordbot.cogs.economy.cog import EconomyCogs
from discordbot.cogs.games.seats import claim_seat, release_seats, hand_over_seats
from discordbot.cogs.games.blackjack import max_round_loss
from discordbot.services.economy.holds import (
    spending,
    spendable,
    hold_stake,
    held_amount,
    release_stake,
    wait_for_spending,
)
from discordbot.services.economy.database import (
    buy_vip,
    transfer,
    get_balance,
    adjust_balance,
    call_personal_loans,
    accept_loan_proposal,
    repay_personal_loans,
    get_spendable_balance,
    create_personal_loan_request,
)

from tests.helpers.casting import as_bot
from tests.helpers.economy import seed_balance, open_personal_loan
from tests.helpers.discord_mocks import FakeUser, FakeInteraction


class _Holder(BaseModel):
    """A stand-in for a lobby or table that seats players."""

    live: bool = Field(default=True, description="Whether the seats it claimed still count.")

    def holds_seats(self) -> bool:
        """Returns whether this holder's seats still count."""
        return self.live


async def test_holds_add_up_and_a_whole_wallet_hold_leaves_nothing() -> None:
    """Amounts sum, a whole-wallet hold leaves nothing spendable, and releasing restores it."""
    table, lobby = _Holder(), _Holder()
    hold_stake(user_id=1, holder=table, amount=300)

    assert held_amount(user_id=1) == 300
    assert spendable(user_id=1, balance=1_000) == 700
    assert spendable(user_id=1, balance=200) == 0
    hold_stake(user_id=1, holder=lobby, amount=None)
    assert held_amount(user_id=1) is None
    assert spendable(user_id=1, balance=1_000) == 0
    release_stake(user_id=1, holder=lobby)
    release_stake(user_id=1, holder=table)
    assert held_amount(user_id=1) == 0


async def test_a_seat_holds_the_wallet_until_its_table_holds_the_stake() -> None:
    """The lobby's claim holds everything, the table then holds the stake, its release nothing."""
    lobby, table = _Holder(), _Holder()

    claim_seat(user_id=1, holder=lobby)
    assert held_amount(user_id=1) is None
    hand_over_seats(stakes={1: 400}, from_holder=lobby, to_holder=table)
    assert held_amount(user_id=1) == 400
    release_seats(user_ids=[1], holder=lobby)
    assert held_amount(user_id=1) == 400
    release_seats(user_ids=[1], holder=table)
    assert held_amount(user_id=1) == 0


async def test_a_transfer_moves_only_what_no_table_holds() -> None:
    """Over the spendable part, the transfer is refused and nothing moves."""
    await seed_balance(user_id=1, name="alice", amount=1_000)
    hold_stake(user_id=1, holder=_Holder(), amount=800)

    refused = await transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=300
    )
    allowed = await transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=200
    )

    assert refused is None
    assert allowed is not None
    assert await get_balance(user_id=1) == 800


async def test_a_transfer_is_refused_while_the_whole_wallet_is_held() -> None:
    """During a lobby's start press nothing leaves the wallet."""
    await seed_balance(user_id=1, name="alice", amount=1_000)
    hold_stake(user_id=1, holder=_Holder(), amount=None)

    assert (
        await transfer(
            sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=1
        )
        is None
    )


async def test_vip_needs_its_cost_outside_the_table() -> None:
    """A wallet that covers VIP only with what a table holds cannot buy it."""
    await seed_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST + 100)
    table = _Holder()
    hold_stake(user_id=1, holder=table, amount=200)

    assert await buy_vip(user_id=1, name="alice") is None
    release_stake(user_id=1, holder=table)
    assert await buy_vip(user_id=1, name="alice") is not None


async def test_a_lender_cannot_fund_a_loan_from_the_table() -> None:
    """Approving a personal loan debits only what no table holds."""
    await seed_balance(user_id=2, name="bob", amount=1_000)
    proposal = await create_personal_loan_request(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=200
    )
    assert proposal is not None
    hold_stake(user_id=2, holder=_Holder(), amount=900)

    accepted = await accept_loan_proposal(
        proposal_id=proposal.proposal_id, actor_id=2, actor_name="bob"
    )

    assert accepted is None
    assert await get_balance(user_id=2) == 1_000


async def test_repaying_and_collecting_take_only_what_no_table_holds() -> None:
    """A repayment stops at the spendable part, and a forced collection takes nothing held."""
    await seed_balance(user_id=2, name="bob", amount=1_000)
    await open_personal_loan(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=500
    )
    hold_stake(user_id=1, holder=_Holder(), amount=400)

    repaid = await repay_personal_loans(
        borrower_id=1, borrower_name="alice", lender_id=2, amount=300
    )
    collected = await call_personal_loans(
        lender_id=2, borrower_id=1, borrower_name="alice", amount=None
    )

    assert repaid is not None
    assert repaid.paid_amount == 100
    assert collected is None
    assert await get_balance(user_id=1) == 400


async def test_money_earned_during_the_round_stays_spendable() -> None:
    """The hold is what the player sat down with; anything credited since is theirs to use."""
    await seed_balance(user_id=1, name="alice", amount=1_000)
    hold_stake(user_id=1, holder=_Holder(), amount=1_000)
    assert await get_spendable_balance(user_id=1) == 0

    await adjust_balance(user_id=1, name="alice", delta=250)

    assert await get_spendable_balance(user_id=1) == 250


async def test_a_seat_risks_at_most_two_and_a_half_stakes_within_its_balance() -> None:
    """A Double or a Split doubles the stake, insurance adds half, all within the start balance."""
    assert max_round_loss(bet=10, balance_at_start=100) == 25
    assert max_round_loss(bet=100, balance_at_start=150) == 150


async def test_a_hold_whose_table_is_gone_holds_nothing() -> None:
    """A table that stopped holding seats frees its hold even if it never released it."""
    table = _Holder()
    hold_stake(user_id=1, holder=table, amount=500)
    table.live = False

    assert held_amount(user_id=1) == 0


async def test_a_claim_waits_out_a_debit_already_in_flight() -> None:
    """Waiting for the user's spending returns only once the debit that held it has finished."""
    entered, release = asyncio.Event(), asyncio.Event()

    async def debit_in_flight() -> None:
        async with spending(user_id=1):
            entered.set()
            await release.wait()

    debit = asyncio.create_task(debit_in_flight())
    await entered.wait()
    waiter = asyncio.create_task(wait_for_spending(user_ids=[1]))
    await asyncio.sleep(0)

    assert not waiter.done()
    release.set()
    await debit
    await waiter


async def test_a_transfer_runs_under_the_senders_spending_lock() -> None:
    """A transfer waits for the sender's lock, so a table's claim and a debit never interleave."""
    await seed_balance(user_id=1, name="alice", amount=1_000)
    release = asyncio.Event()

    async def claim_holding_the_lock() -> None:
        async with spending(user_id=1):
            await release.wait()

    claim = asyncio.create_task(claim_holding_the_lock())
    await asyncio.sleep(0)
    sending = asyncio.create_task(
        transfer(sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=100)
    )
    await asyncio.sleep(0.05)

    assert not sending.done()
    release.set()
    await claim
    assert await sending is not None


@pytest.mark.parametrize(
    argnames="debit",
    argvalues=[
        pytest.param(lambda: buy_vip(user_id=1, name="alice"), id="buy_vip"),
        pytest.param(
            lambda: repay_personal_loans(
                borrower_id=1, borrower_name="alice", lender_id=2, amount=10
            ),
            id="repay_personal_loans",
        ),
        pytest.param(
            lambda: call_personal_loans(
                lender_id=2, borrower_id=1, borrower_name="alice", amount=10
            ),
            id="call_personal_loans",
        ),
    ],
)
async def test_every_voluntary_debit_waits_for_the_users_spending_lock(
    debit: Callable[[], Coroutine[object, object, object]],
) -> None:
    """Each one queues behind a table's claim on the same wallet."""
    await seed_balance(user_id=2, name="bob", amount=1_000)
    await open_personal_loan(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=500
    )
    release = asyncio.Event()

    async def claim_holding_the_lock() -> None:
        async with spending(user_id=1):
            await release.wait()

    claim = asyncio.create_task(claim_holding_the_lock())
    await asyncio.sleep(0)
    pending = asyncio.create_task(debit())
    await asyncio.sleep(0.05)

    assert not pending.done()
    release.set()
    await claim
    await pending


async def test_funding_a_loan_waits_for_the_lenders_spending_lock() -> None:
    """Approving a personal loan queues behind a table's claim on the lender's wallet."""
    await seed_balance(user_id=2, name="bob", amount=1_000)
    proposal = await create_personal_loan_request(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=200
    )
    assert proposal is not None
    release = asyncio.Event()

    async def claim_holding_the_lock() -> None:
        async with spending(user_id=2):
            await release.wait()

    claim = asyncio.create_task(claim_holding_the_lock())
    await asyncio.sleep(0)
    pending = asyncio.create_task(
        accept_loan_proposal(proposal_id=proposal.proposal_id, actor_id=2, actor_name="bob")
    )
    await asyncio.sleep(0.05)

    assert not pending.done()
    release.set()
    await claim
    assert await pending is not None


@pytest.mark.parametrize(
    argnames=("held", "heading"),
    argvalues=[(1_000, "### 沒有可還的餘額"), (400, "### 沒有可還給 Bob 的有效個人借款")],
    ids=["table-holds-it-all", "no-such-loan"],
)
async def test_a_refused_repayment_blames_the_table_only_when_it_holds_everything(
    held: int, heading: str
) -> None:
    """With money to spend, a refused repayment means there is no such loan."""
    await seed_balance(user_id=1, name="alice", amount=1_000)
    hold_stake(user_id=1, holder=_Holder(), amount=held)
    cog = EconomyCogs(bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999))))
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"))

    await EconomyCogs.credit_repay.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob", display_name="Bob"), amount="50"
    )

    (sent,) = interaction.followup.sent
    description = sent["embed"].description
    assert description is not None
    assert description.startswith(heading)
