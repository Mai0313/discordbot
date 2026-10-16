"""Tests for the economy cog's slash commands.

Most replace the ledger facade where the cog imports it: what those cover is which facade call
each command makes, whether it is acknowledged before that write, and who gets to see the answer.
The `/vip` and central-bank ceiling tests run against the throwaway ledger.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
import asyncio
from datetime import UTC, datetime
from unittest.mock import ANY

import pytest

from discordbot.cogs.economy import cog as economy
from discordbot.cogs.economy import views
from discordbot.typings.economy import (
    VIP_PURCHASE_COST,
    PortfolioView,
    TransferResult,
    AccountSnapshot,
    LeaderboardEntry,
    LoanContractView,
    LoanProposalKind,
    LoanProposalView,
    CentralBankStatus,
    LoanPaymentResult,
    VipPurchaseResult,
    CasinoLedgerSnapshot,
    LossLeaderboardEntry,
    BalanceAdjustmentResult,
)
from discordbot.cogs.economy.cog import EconomyCogs
from discordbot.cogs.economy.views import CreditLoanDecisionView, CentralBankLoanDecisionView
from discordbot.cogs.economy.embeds import (
    BORROW_COLOR,
    LEADERBOARD_COLOR,
    LEADERBOARD_TITLE,
    LOSS_LEADERBOARD_COLOR,
    LOSS_LEADERBOARD_TITLE,
    build_vip_already_embed,
    build_vip_success_embed,
    build_vip_insufficient_embed,
)
from discordbot.utils.message_cleanup import list_pending_public_messages
from discordbot.services.economy.database import (
    buy_vip,
    get_vip,
    get_account,
    get_balance,
    get_credit_ceiling,
)

from tests.helpers.casting import as_bot, as_interaction
from tests.helpers.economy import (
    LENDING_GUILD,
    seed_balance,
    seed_participant,
    fake_loan_proposal,
    personal_loan_contract,
)
from tests.helpers.discord_mocks import FakeUser, FakeGuild, FakeInteraction, FakeDiscordMessage
from tests.helpers.message_cleanup import record_scheduled_deletes

if TYPE_CHECKING:
    from typing import Unpack
    from collections.abc import Callable, Awaitable

    from nextcord.ext import commands

    from tests.helpers.discord_mocks import DiscordPayload


def _bot() -> commands.Bot:
    """A stub bot whose own account is the dealer, id 999."""
    return as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))


async def fake_get_balance(user_id: int) -> int:
    """Returns a stable fake balance."""
    return 150


async def fake_get_portfolio(user_id: int) -> PortfolioView:
    """Returns a stable fake portfolio."""
    return PortfolioView(
        user_id=user_id,
        name="alice",
        balance=150,
        debt_principal=30,
        debt_interest=5,
        net_worth=115,
    )


async def fake_get_vip(user_id: int) -> bool:
    """Returns non-VIP status."""
    return False


async def fake_get_admin(user_id: int) -> bool:
    """Returns economy admin status."""
    return True


async def fake_top_n(limit: int | None, include_hidden: bool = False) -> list[LeaderboardEntry]:
    """Returns one fake leaderboard row."""
    return [
        LeaderboardEntry(
            user_id=1, name="alice", balance=150, avatar_url="https://cdn.example/alice.png"
        )
    ]


async def fake_top_losers(limit: int) -> list[LossLeaderboardEntry]:
    """Returns one fake loss leaderboard row."""
    return [
        LossLeaderboardEntry(
            user_id=1, name="alice", loss_amount=500, avatar_url="https://cdn.example/alice.png"
        )
    ]


async def fake_get_account(user_id: int) -> AccountSnapshot:
    """Returns a fake bot wallet account."""
    return AccountSnapshot(name="Bot", balance=-50, total_earned=100, total_spent=150)


async def fake_get_casino_ledger() -> CasinoLedgerSnapshot:
    """Returns a fake casino ledger snapshot."""
    return CasinoLedgerSnapshot(
        balance=-50, total_earned=100, total_spent=150, updated_at=datetime.now(tz=UTC)
    )


async def fake_adjust_balance(
    user_id: int, name: str, delta: int, allow_negative: bool = False, avatar_url: str = ""
) -> BalanceAdjustmentResult:
    """Returns a successful fake manual adjustment result."""
    return BalanceAdjustmentResult(new_balance=150 + delta, applied_delta=delta)


async def fake_create_loan_request(**_kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
    """Returns a fake personal request."""
    return fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)


async def fake_create_central_bank_request(**_kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
    """Returns a fake central-bank request."""
    return fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)


async def fake_list_loan_contracts(user_id: int) -> list[LoanContractView]:
    """Returns one active loan contract."""
    return [personal_loan_contract(borrower_id=user_id, interest_due=3)]


async def fake_loan_payment(**_kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
    """Returns a fake repayment result."""
    return LoanPaymentResult(
        paid_amount=50,
        interest_paid=5,
        principal_paid=45,
        borrower_balance=100,
        lender_balance=200,
        remaining_principal=55,
        remaining_interest=0,
    )


async def fake_get_credit_ceiling(user_id: int) -> int:
    """Returns a ceiling high enough that only the amount parser can refuse a request."""
    del user_id
    return 10**40


async def fake_record_guild_participant(guild_id: int, user_id: int) -> None:
    """Swallows the participation upsert the commands make after acknowledging."""
    del guild_id, user_id


async def fake_get_central_bank_status(guild_id: int) -> CentralBankStatus:
    """Returns fake central-bank capacity."""
    return CentralBankStatus(
        participant_count=3,
        total_positive_user_balance=1_000,
        outstanding_principal=100,
        available_credit=900,
        ledger_balance=42,
    )


async def fake_buy_vip(user_id: int, name: str, avatar_url: str) -> VipPurchaseResult:
    """Returns a successful fake VIP purchase result."""
    return VipPurchaseResult(new_balance=500_000, cost=VIP_PURCHASE_COST)


def _record_transfers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, int | str]]:
    """Replaces the cog's transfer facade with a recorder; returns each call's arguments."""
    transfers: list[dict[str, int | str]] = []

    async def record_transfer(  # noqa: PLR0913 -- mirrors transfer signature
        sender_id: int,
        sender_name: str,
        receiver_id: int,
        receiver_name: str,
        amount: int,
        sender_avatar_url: str = "",
        receiver_avatar_url: str = "",
    ) -> TransferResult:
        """Records the transfer's identity and avatar payloads."""
        transfers.append({
            "sender_id": sender_id,
            "sender_name": sender_name,
            "receiver_id": receiver_id,
            "receiver_name": receiver_name,
            "amount": amount,
            "sender_avatar_url": sender_avatar_url,
            "receiver_avatar_url": receiver_avatar_url,
        })
        return TransferResult(
            sender_balance=50, receiver_balance=100, received_amount=100, tax_amount=0
        )

    monkeypatch.setattr(economy, "transfer", record_transfer)
    return transfers


# The four loan settlements, each with the ledger call that decides it. A repayment of zero is
# refused before the command ever acks, so these carry a real amount; a collection of zero means
# "everything owed" and is the ordinary call.
_LOAN_SETTLEMENTS: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("credit_repay", "repay_personal_loans", {"member": FakeUser(user_id=2), "amount": "5"}),
    ("credit_call", "call_personal_loans", {"member": FakeUser(user_id=2), "amount": "0"}),
    ("central_bank_repay", "repay_central_bank_loans", {"amount": "5"}),
    (
        "central_bank_call",
        "call_central_bank_loans",
        {"member": FakeUser(user_id=2), "amount": "0"},
    ),
)


async def test_a_failed_money_command_stays_private(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acking before the write must not turn a private failure into a public one.

    Having no matching loan is an ordinary mistake, not an edge case. The failure fills the
    placeholder the defer posted, so the defer's flag decides both whether the channel sees that
    the attempt happened and whether it sees what came of it.

    All four, because the ack is written out once per command and only a sweep catches the one
    someone changes on its own.
    """

    async def nothing_to_settle(**_kwargs: object) -> None:
        """Stands in for a write that matched no loan."""
        return

    monkeypatch.setattr(economy, "record_guild_participant", fake_record_guild_participant)
    for command, ledger_call, kwargs in _LOAN_SETTLEMENTS:
        monkeypatch.setattr(economy, ledger_call, nothing_to_settle)
        cog = EconomyCogs(bot=as_bot(fake=SimpleNamespace()))
        # Administrator, or `central_bank_call` returns from its permission branch and this
        # stops covering the failure path it was written for — while still passing, because
        # that branch defers and follows up ephemerally just like the one under test.
        interaction = FakeInteraction(
            user=FakeUser(user_id=1, display_name="Alice"), administrator=True, slash_command=True
        )

        await getattr(EconomyCogs, command).callback(
            cog, as_interaction(fake=interaction), **kwargs
        )

        # This flag says an ack happened, not when — the ordering is the structural test's job.
        assert interaction.response.deferred is True, f"{command} never acknowledged at all"
        assert interaction.response.deferred_ephemeral is True, (
            f"{command} announced the attempt to the channel"
        )
        assert interaction.followup.sent[-1]["ephemeral"] is True, (
            f"{command} announced the failure to the channel"
        )


async def test_a_settled_loan_is_posted_to_the_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repayment or collection is a settlement, so the other party sees it in the channel.

    The defer stays ephemeral for the failure's sake, and the first followup after it fills that
    placeholder with its flag, so a settlement sent as that followup reaches only the caller.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    monkeypatch.setattr(economy, "record_guild_participant", fake_record_guild_participant)
    for command, ledger_call, kwargs in _LOAN_SETTLEMENTS:
        monkeypatch.setattr(economy, ledger_call, fake_loan_payment)
        interaction = FakeInteraction(
            user=FakeUser(user_id=1, display_name="Alice"), administrator=True, slash_command=True
        )

        await getattr(EconomyCogs, command).callback(
            EconomyCogs(bot=as_bot(fake=SimpleNamespace())),
            as_interaction(fake=interaction),
            **kwargs,
        )

        assert interaction.response.deferred_ephemeral is True, (
            f"{command} announced the attempt to the channel"
        )
        assert interaction.followup.sent[-1].get("ephemeral") is not True, (
            f"{command} posted its settlement only to the caller"
        )
        assert interaction.original_deleted is True, f"{command} left the caller a second copy"
    assert len(scheduled.messages) == len(_LOAN_SETTLEMENTS)


async def test_economy_commands_use_database_facade(  # noqa: PLR0915 -- command smoke exercises one facade surface
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies economy slash commands call the database facade and send embeds."""
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    monkeypatch.setattr(economy, "get_spendable_balance", fake_get_balance)
    monkeypatch.setattr(economy, "get_vip", fake_get_vip)
    monkeypatch.setattr(economy, "get_admin", fake_get_admin)
    monkeypatch.setattr(economy, "top_n", fake_top_n)
    monkeypatch.setattr(economy, "top_losers", fake_top_losers)
    monkeypatch.setattr(economy, "get_account", fake_get_account)
    monkeypatch.setattr(economy, "get_casino_ledger", fake_get_casino_ledger)
    _record_transfers(monkeypatch=monkeypatch)
    monkeypatch.setattr(economy, "adjust_balance", fake_adjust_balance)
    monkeypatch.setattr(economy, "get_portfolio", fake_get_portfolio)
    monkeypatch.setattr(economy, "create_personal_loan_request", fake_create_loan_request)
    monkeypatch.setattr(economy, "repay_personal_loans", fake_loan_payment)
    monkeypatch.setattr(economy, "call_personal_loans", fake_loan_payment)
    monkeypatch.setattr(
        economy, "create_central_bank_loan_request", fake_create_central_bank_request
    )
    monkeypatch.setattr(economy, "list_loan_contracts", fake_list_loan_contracts)
    monkeypatch.setattr(economy, "get_central_bank_status", fake_get_central_bank_status)
    monkeypatch.setattr(economy, "repay_central_bank_loans", fake_loan_payment)
    monkeypatch.setattr(economy, "call_central_bank_loans", fake_loan_payment)
    monkeypatch.setattr(economy, "get_credit_ceiling", fake_get_credit_ceiling)
    monkeypatch.setattr(economy, "record_guild_participant", fake_record_guild_participant)
    monkeypatch.setattr(economy, "buy_vip", fake_buy_vip)
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1), administrator=True, slash_command=True)
    await EconomyCogs.balance.callback(cog, interaction, member=None)
    await EconomyCogs.leaderboard.callback(cog, interaction)
    await EconomyCogs.loss_leaderboard.callback(cog, interaction)
    await EconomyCogs.casino.callback(cog, interaction)
    await EconomyCogs.pocat.callback(cog, interaction)
    await EconomyCogs.admin_refund_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="100"
    )
    await EconomyCogs.admin_collect_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="50"
    )
    await EconomyCogs.give.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="100"
    )
    await EconomyCogs.credit_borrow.callback(
        cog,
        interaction,
        member=FakeUser(user_id=2, name="bob"),
        amount="100",
        monthly_rate_percent=3.0,
    )
    await EconomyCogs.credit_repay.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="50"
    )
    await EconomyCogs.credit_call.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="0"
    )
    await EconomyCogs.credit_status.callback(cog, interaction)
    await EconomyCogs.central_bank_borrow.callback(
        cog, interaction, amount="100", monthly_rate_percent=3.0
    )
    await EconomyCogs.central_bank_repay.callback(cog, interaction, amount="50")
    await EconomyCogs.central_bank_call.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="0"
    )
    await EconomyCogs.central_bank_status.callback(cog, interaction)
    await EconomyCogs.vip_command.callback(cog, interaction)
    assert len(interaction.followup.sent) == 21
    assert len(scheduled.messages) == 12
    assert interaction.followup.sent[0].get("ephemeral") is True
    assert "view" not in interaction.followup.sent[1]
    assert interaction.followup.sent[1]["files"][0].filename == "economy_leaderboard.png"
    assert interaction.followup.sent[2]["files"][0].filename == "economy_loss_leaderboard.png"
    assert "view" not in interaction.followup.sent[3]
    assert "view" not in interaction.followup.sent[4]
    assert interaction.followup.sent[5].get("ephemeral") is not True
    assert interaction.followup.sent[6].get("ephemeral") is not True
    assert interaction.followup.sent[7].get("ephemeral") is not True
    assert interaction.followup.sent[8].get("ephemeral") is not True
    assert interaction.followup.sent[10].get("ephemeral") is not True
    assert interaction.followup.sent[12].get("ephemeral") is not True
    assert interaction.followup.sent[13].get("ephemeral") is True
    assert interaction.followup.sent[16].get("ephemeral") is not True
    assert interaction.followup.sent[18].get("ephemeral") is not True
    assert interaction.followup.sent[19].get("ephemeral") is not True
    assert interaction.followup.sent[-1].get("ephemeral") is True
    balance_embed = interaction.followup.sent[0]["embed"]
    # Assert the financial summary's structure and the facade values it surfaces, not the exact
    # localized title/labels: cash 150, debt principal 30, net worth 115.
    assert (balance_embed.title or "").startswith("💰")
    assert "115" in (balance_embed.description or "")
    balance_fields = {field.name: field.value or "" for field in balance_embed.fields}
    assert "150" in balance_fields["現金"]
    assert "30" in balance_fields["債務"]
    borrow_embed = interaction.followup.sent[8]["embed"]
    # The footer explains the loan-decision timeout; assert the behavioral 180s, not the copy.
    assert "180" in (borrow_embed.footer.text or "")
    borrow_view = interaction.followup.sent[8]["view"]
    assert isinstance(borrow_view, CreditLoanDecisionView)
    assert borrow_view.message is not None
    central_bank_payload = interaction.followup.sent[14]
    central_bank_view = central_bank_payload["view"]
    assert isinstance(central_bank_view, CentralBankLoanDecisionView)
    assert central_bank_view.message is not None

    inspected_member = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.balance.callback(
        cog, inspected_member, member=FakeUser(user_id=2, name="bob", display_name="Bob")
    )
    inspected_description = inspected_member.followup.sent[0]["embed"].description
    assert inspected_description is not None
    assert "Bob" in inspected_description


@pytest.mark.parametrize(
    argnames=("command", "kwargs"),
    argvalues=[
        (
            "credit_borrow",
            {
                "member": FakeUser(user_id=2, name="bob"),
                "amount": "100",
                "monthly_rate_percent": 3.0,
            },
        ),
        ("central_bank_borrow", {"amount": "100", "monthly_rate_percent": 3.0}),
    ],
    ids=["credit", "central_bank"],
)
async def test_a_loan_request_panel_is_recorded_for_the_restart_sweep(
    monkeypatch: pytest.MonkeyPatch, command: str, kwargs: dict[str, object]
) -> None:
    """Its buttons die with the process, so a restart's sweep must be able to take it down."""
    monkeypatch.setattr(economy, "create_personal_loan_request", fake_create_loan_request)
    monkeypatch.setattr(
        economy, "create_central_bank_loan_request", fake_create_central_bank_request
    )
    monkeypatch.setattr(economy, "get_credit_ceiling", fake_get_credit_ceiling)
    monkeypatch.setattr(economy, "record_guild_participant", fake_record_guild_participant)
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), slash_command=True)
    send_followup = interaction.followup.send

    async def send_with_an_id_of_its_own(**payload: Unpack[DiscordPayload]) -> FakeDiscordMessage:
        """Numbers each followup, since every fake message is otherwise id 1."""
        message = await send_followup(**payload)
        message.id = 100 + len(interaction.followup.sent)
        return message

    monkeypatch.setattr(interaction.followup, "send", send_with_an_id_of_its_own)

    await getattr(EconomyCogs, command).callback(
        EconomyCogs(bot=_bot()), as_interaction(fake=interaction), **kwargs
    )

    view = interaction.followup.sent[-1]["view"]
    assert isinstance(view, views.LoanDecisionViewBase)
    assert view.message is not None
    assert [record.message_id for record in await list_pending_public_messages()] == [
        view.message.id
    ]


async def test_economy_admin_rejects_non_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin economy commands must check the DB admin flag before mutating balance."""
    called = False

    async def fake_get_admin_false(user_id: int) -> bool:
        """Returns a non-admin status."""
        return False

    async def fake_adjust_balance_guard(**_kwargs: Any) -> BalanceAdjustmentResult:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Fails the test if a non-admin reaches the mutation path."""
        nonlocal called
        called = True
        return BalanceAdjustmentResult(new_balance=0, applied_delta=0)

    monkeypatch.setattr(economy, "get_admin", fake_get_admin_false)
    monkeypatch.setattr(economy, "adjust_balance", fake_adjust_balance_guard)
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.admin_refund_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="100"
    )

    assert called is False
    assert interaction.followup.sent[0].get("ephemeral") is True
    admin_rejection_title = interaction.followup.sent[0]["embed"].title
    assert admin_rejection_title is not None
    assert "權限不足" in admin_rejection_title


async def test_a_positive_amount_refuses_zero() -> None:
    """Zero is a well-formed amount, but no command that takes a positive one accepts it."""
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    amount = await economy._amount_or_refuse(
        interaction=as_interaction(fake=interaction), raw_amount="0", title="轉帳失敗"
    )

    assert amount is None
    assert interaction.response.sent[0]["ephemeral"] is True


async def test_economy_admin_tax_allows_bot_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin tax commands may adjust the bot account."""
    captured_targets: list[tuple[int, str, int]] = []

    async def record_adjust_balance(
        user_id: int, name: str, delta: int, allow_negative: bool = False, avatar_url: str = ""
    ) -> BalanceAdjustmentResult:
        """Records target accounts and parsed adjustment deltas."""
        del allow_negative, avatar_url
        captured_targets.append((user_id, name, delta))
        return BalanceAdjustmentResult(new_balance=150 + delta, applied_delta=delta)

    monkeypatch.setattr(economy, "get_admin", fake_get_admin)
    monkeypatch.setattr(economy, "adjust_balance", record_adjust_balance)
    record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1))
    bot_member = FakeUser(user_id=999, name="discordbot", display_name="Dealer", bot=True)

    await EconomyCogs.admin_refund_tax.callback(cog, interaction, member=bot_member, amount="100")
    await EconomyCogs.admin_collect_tax.callback(cog, interaction, member=bot_member, amount="50")

    # order-contract: each awaited command completes its balance adjustment before returning.
    assert captured_targets == [(999, "discordbot", 100), (999, "discordbot", -50)]
    assert interaction.followup.sent[0].get("ephemeral") is not True
    assert interaction.followup.sent[1].get("ephemeral") is not True


async def test_give_passes_guild_avatar_urls_to_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transfer writes should cache guild avatars instead of only global avatars."""
    sender = FakeUser(user_id=1, name="alice")
    receiver = FakeUser(user_id=2, name="bob")
    guild = FakeGuild(
        members=[
            FakeUser(
                user_id=1, name="alice", guild_avatar_url="https://example.test/alice-server.png"
            ),
            FakeUser(
                user_id=2, name="bob", guild_avatar_url="https://example.test/bob-server.png"
            ),
        ],
        cached=True,
    )
    interaction = FakeInteraction(user=sender)
    interaction.guild = guild
    transfers = _record_transfers(monkeypatch=monkeypatch)
    record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = EconomyCogs(bot=_bot())

    await EconomyCogs.give.callback(cog, interaction, member=receiver, amount="100")

    assert transfers[0]["sender_avatar_url"] == "https://example.test/alice-server.png"
    assert transfers[0]["receiver_avatar_url"] == "https://example.test/bob-server.png"
    assert guild.fetch_count == 0


async def test_give_allows_bot_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Players may transfer balance to the bot account."""
    sender = FakeUser(user_id=1, name="alice")
    bot_receiver = FakeUser(user_id=999, name="discordbot", display_name="Dealer", bot=True)
    interaction = FakeInteraction(user=sender)
    transfers = _record_transfers(monkeypatch=monkeypatch)
    record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = EconomyCogs(bot=as_bot(fake=SimpleNamespace(user=bot_receiver)))

    await EconomyCogs.give.callback(cog, interaction, member=bot_receiver, amount="100")

    assert transfers == [
        {
            "sender_id": 1,
            "sender_name": "alice",
            "receiver_id": 999,
            "receiver_name": "discordbot",
            "amount": 100,
            # Which avatar each side stores is the guild-avatar test's business.
            "sender_avatar_url": ANY,
            "receiver_avatar_url": ANY,
        }
    ]
    give_bot_receiver_title = interaction.followup.sent[0]["embed"].title
    assert give_bot_receiver_title is not None
    assert "轉帳完成" in give_bot_receiver_title


@pytest.mark.parametrize(
    argnames=("receiver_id", "reason"),
    argvalues=[(1, "### 不能轉給自己"), (2, "### 餘額不足")],
    ids=["self-transfer", "insufficient-balance"],
)
async def test_a_refused_transfer_is_a_public_expiring_embed(
    monkeypatch: pytest.MonkeyPatch, receiver_id: int, reason: str
) -> None:
    """`/give` defers publicly before it can know the transfer fails, so its refusal is public too.

    The channel already saw the attempt, so the refusal lands there and is cleaned up with it.
    """

    async def refuse_transfer(**_kwargs: object) -> None:
        """Stands in for a sender who cannot cover the amount."""
        return

    monkeypatch.setattr(economy, "transfer", refuse_transfer)
    monkeypatch.setattr(economy, "get_spendable_balance", fake_get_balance)
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"))

    await EconomyCogs.give.callback(
        EconomyCogs(bot=_bot()),
        as_interaction(fake=interaction),
        member=FakeUser(user_id=receiver_id, name="bob"),
        amount="1000",
    )

    assert interaction.response.deferred is True
    assert interaction.response.deferred_ephemeral is False
    assert interaction.followup.sent[0].get("ephemeral") is not True
    assert reason in (interaction.followup.sent[0]["embed"].description or "")
    assert len(scheduled.messages) == 1


@pytest.mark.parametrize(
    argnames=("command", "kwargs"),
    argvalues=[
        ("central_bank_borrow", {"amount": "100", "monthly_rate_percent": 3.0}),
        ("central_bank_repay", {"amount": "50"}),
        ("central_bank_call", {"member": FakeUser(user_id=2, name="bob"), "amount": "0"}),
        ("central_bank_status", {}),
    ],
    ids=["borrow", "repay", "call", "status"],
)
async def test_only_the_caller_is_recorded_as_taking_part_in_the_guild(
    monkeypatch: pytest.MonkeyPatch, command: str, kwargs: dict[str, object]
) -> None:
    """A `/central_bank` caller takes part in the guild; a `member:` target never does.

    Recording the target would hand their whole balance to this server's lending pool without
    their knowledge.
    """
    recorded: list[tuple[int, int]] = []

    async def record_participant(guild_id: int, user_id: int) -> None:
        """Records who was enrolled in which guild."""
        recorded.append((guild_id, user_id))

    monkeypatch.setattr(economy, "record_guild_participant", record_participant)
    monkeypatch.setattr(economy, "get_credit_ceiling", fake_get_credit_ceiling)
    monkeypatch.setattr(
        economy, "create_central_bank_loan_request", fake_create_central_bank_request
    )
    monkeypatch.setattr(economy, "repay_central_bank_loans", fake_loan_payment)
    monkeypatch.setattr(economy, "call_central_bank_loans", fake_loan_payment)
    monkeypatch.setattr(economy, "get_central_bank_status", fake_get_central_bank_status)
    record_scheduled_deletes(monkeypatch=monkeypatch)
    interaction = FakeInteraction(user=FakeUser(user_id=1), guild_id=321, administrator=True)

    await getattr(EconomyCogs, command).callback(
        EconomyCogs(bot=_bot()), as_interaction(fake=interaction), **kwargs
    )

    assert recorded == [(321, 1)]


@pytest.mark.parametrize(
    argnames=("command", "kwargs", "title"),
    argvalues=[
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
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), in_guild=False)

    await getattr(EconomyCogs, command).callback(cog, as_interaction(fake=interaction), **kwargs)

    assert interaction.response.sent[0]["ephemeral"] is True
    assert interaction.response.sent[0]["embed"].title == title


async def test_economy_money_commands_accept_large_string_amounts(  # noqa: PLR0915 -- one sweep over every command that parses an amount
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loan, transfer, collection and tax amounts parse beyond Discord integer option limits."""
    big_amount = 9_007_199_254_740_993
    captured: dict[str, int | None] = {}
    adjusted: list[int] = []

    async def record_transfer(**kwargs: Any) -> TransferResult:  # noqa: ANN401 -- command facade double
        captured["give"] = kwargs["amount"]
        return TransferResult(
            sender_balance=0, receiver_balance=0, received_amount=0, tax_amount=0
        )

    async def record_create_personal(**kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
        captured["credit_borrow"] = kwargs["amount"]
        return fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)

    async def record_create_central(**kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
        captured["central_bank_borrow"] = kwargs["amount"]
        return fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)

    async def record_repay_personal(**kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
        captured["credit_repay"] = kwargs["amount"]
        return await fake_loan_payment()

    async def record_repay_central(**kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
        captured["central_bank_repay"] = kwargs["amount"]
        return await fake_loan_payment()

    async def record_call_personal(**kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
        captured["credit_call"] = kwargs["amount"]
        return await fake_loan_payment()

    async def record_call_central(**kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
        captured["central_bank_call"] = kwargs["amount"]
        return await fake_loan_payment()

    async def record_adjust_balance(**kwargs: Any) -> BalanceAdjustmentResult:  # noqa: ANN401 -- command facade double
        adjusted.append(kwargs["delta"])
        return BalanceAdjustmentResult(new_balance=0, applied_delta=kwargs["delta"])

    monkeypatch.setattr(economy, "transfer", record_transfer)
    monkeypatch.setattr(economy, "create_personal_loan_request", record_create_personal)
    monkeypatch.setattr(economy, "create_central_bank_loan_request", record_create_central)
    monkeypatch.setattr(economy, "repay_personal_loans", record_repay_personal)
    monkeypatch.setattr(economy, "repay_central_bank_loans", record_repay_central)
    monkeypatch.setattr(economy, "call_personal_loans", record_call_personal)
    monkeypatch.setattr(economy, "call_central_bank_loans", record_call_central)
    monkeypatch.setattr(economy, "get_admin", fake_get_admin)
    monkeypatch.setattr(economy, "adjust_balance", record_adjust_balance)
    monkeypatch.setattr(economy, "get_credit_ceiling", fake_get_credit_ceiling)
    monkeypatch.setattr(economy, "record_guild_participant", fake_record_guild_participant)
    record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), administrator=True)
    big_text = "9,007,199,254,740,993"
    member = FakeUser(user_id=2, name="bob")

    await EconomyCogs.give.callback(cog, interaction, member=member, amount=big_text)
    await EconomyCogs.credit_borrow.callback(
        cog, interaction, member=member, amount=big_text, monthly_rate_percent=3.0
    )
    await EconomyCogs.credit_repay.callback(cog, interaction, member=member, amount=big_text)
    await EconomyCogs.credit_call.callback(cog, interaction, member=member, amount=big_text)
    await EconomyCogs.central_bank_borrow.callback(
        cog, interaction, amount=big_text, monthly_rate_percent=3.0
    )
    await EconomyCogs.central_bank_repay.callback(cog, interaction, amount=big_text)
    await EconomyCogs.central_bank_call.callback(cog, interaction, member=member, amount=big_text)
    await EconomyCogs.admin_refund_tax.callback(cog, interaction, member=member, amount=big_text)
    await EconomyCogs.admin_collect_tax.callback(cog, interaction, member=member, amount=big_text)

    # order-contract: each awaited command completes its balance adjustment before returning.
    assert adjusted == [big_amount, -big_amount]
    assert captured == {
        "give": big_amount,
        "credit_borrow": big_amount,
        "credit_repay": big_amount,
        "credit_call": big_amount,
        "central_bank_borrow": big_amount,
        "central_bank_repay": big_amount,
        "central_bank_call": big_amount,
    }

    await EconomyCogs.credit_call.callback(cog, interaction, member=member, amount="0")
    await EconomyCogs.central_bank_call.callback(cog, interaction, member=member, amount="")
    assert captured["credit_call"] is None
    assert captured["central_bank_call"] is None


async def test_economy_money_commands_reject_invalid_amount_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed amount text is rejected before any balance, loan, collection or tax mutation."""
    mutated: list[str] = []

    async def guard_transfer(**kwargs: Any) -> TransferResult:  # noqa: ANN401 -- command facade double
        del kwargs
        mutated.append("transfer")
        return TransferResult(
            sender_balance=0, receiver_balance=0, received_amount=0, tax_amount=0
        )

    async def guard_create_personal(**kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
        del kwargs
        mutated.append("create_personal")
        return fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)

    async def guard_create_central(**kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
        del kwargs
        mutated.append("create_central")
        return fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)

    async def guard_payment(**kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
        del kwargs
        mutated.append("payment")
        return await fake_loan_payment()

    async def guard_adjust_balance(**kwargs: Any) -> BalanceAdjustmentResult:  # noqa: ANN401 -- command facade double
        del kwargs
        mutated.append("adjust_balance")
        return BalanceAdjustmentResult(new_balance=0, applied_delta=0)

    monkeypatch.setattr(economy, "transfer", guard_transfer)
    monkeypatch.setattr(economy, "create_personal_loan_request", guard_create_personal)
    monkeypatch.setattr(economy, "create_central_bank_loan_request", guard_create_central)
    monkeypatch.setattr(economy, "repay_personal_loans", guard_payment)
    monkeypatch.setattr(economy, "repay_central_bank_loans", guard_payment)
    monkeypatch.setattr(economy, "call_personal_loans", guard_payment)
    monkeypatch.setattr(economy, "call_central_bank_loans", guard_payment)
    monkeypatch.setattr(economy, "adjust_balance", guard_adjust_balance)
    monkeypatch.setattr(economy, "get_credit_ceiling", fake_get_credit_ceiling)
    monkeypatch.setattr(economy, "record_guild_participant", fake_record_guild_participant)
    cog = EconomyCogs(bot=_bot())
    member = FakeUser(user_id=2, name="bob")

    def assert_rejected(interaction: FakeInteraction, expected_title: str) -> None:
        """Asserts an ephemeral malformed-amount rejection with no mutation followup."""
        assert interaction.response.sent[0]["ephemeral"] is True
        assert interaction.response.sent[0]["embed"].title == expected_title
        rejection_description = interaction.response.sent[0]["embed"].description
        assert rejection_description is not None
        assert "金額格式錯誤" in rejection_description
        assert interaction.followup.sent == []

    rejections: list[tuple[str, Callable[[FakeInteraction], Awaitable[None]]]] = [
        ("轉帳失敗", lambda i: EconomyCogs.give.callback(cog, i, member=member, amount="x")),
        (
            "借款失敗",
            lambda i: EconomyCogs.credit_borrow.callback(
                cog, i, member=member, amount="x", monthly_rate_percent=3.0
            ),
        ),
        (
            "還款失敗",
            lambda i: EconomyCogs.credit_repay.callback(cog, i, member=member, amount="x"),
        ),
        (
            "催收失敗",
            lambda i: EconomyCogs.credit_call.callback(cog, i, member=member, amount="x"),
        ),
        (
            "央行借款失敗",
            lambda i: EconomyCogs.central_bank_borrow.callback(
                cog, i, amount="x", monthly_rate_percent=3.0
            ),
        ),
        ("央行還款失敗", lambda i: EconomyCogs.central_bank_repay.callback(cog, i, amount="x")),
        (
            "央行催收失敗",
            lambda i: EconomyCogs.central_bank_call.callback(cog, i, member=member, amount="x"),
        ),
        (
            "退稅失敗",
            lambda i: EconomyCogs.admin_refund_tax.callback(cog, i, member=member, amount="x"),
        ),
        (
            "收稅失敗",
            lambda i: EconomyCogs.admin_collect_tax.callback(cog, i, member=member, amount="x"),
        ),
    ]
    for expected_title, invoke in rejections:
        interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), administrator=True)
        await invoke(interaction)
        assert_rejected(interaction, expected_title)

    assert mutated == []


async def test_loss_leaderboard_uses_daily_loss_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loss leaderboard embed describes gross daily loss, not net P&L."""

    async def daily_losses(limit: int) -> list[LossLeaderboardEntry]:
        """Returns fake daily gross loss rows."""
        return [
            LossLeaderboardEntry(user_id=1, name="alice", loss_amount=500, avatar_url=""),
            LossLeaderboardEntry(user_id=2, name="bob", loss_amount=200, avatar_url=""),
        ]

    monkeypatch.setattr(economy, "top_losers", daily_losses)
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.loss_leaderboard.callback(cog, interaction)

    embed = interaction.followup.sent[0]["embed"]
    assert embed.title is not None
    assert "今日輸局累計" in embed.title
    assert embed.description is not None
    assert "累計輸" in embed.description
    assert interaction.followup.sent[0]["files"][0].filename == "economy_loss_leaderboard.png"
    assert embed.footer.text is not None
    assert "贏回來不抵扣" in embed.footer.text
    assert len(scheduled.messages) == 1


async def test_loss_leaderboard_empty_state_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loss leaderboard empty state stays explicit about today's loss rows."""

    async def no_daily_losses(limit: int) -> list[LossLeaderboardEntry]:
        """Returns an empty daily loss board."""
        return []

    monkeypatch.setattr(economy, "top_losers", no_daily_losses)
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.loss_leaderboard.callback(cog, interaction)

    embed = interaction.followup.sent[0]["embed"]
    assert embed.title is not None
    assert "今日輸局累計" in embed.title
    assert embed.description is not None
    assert "今天還沒有人輸錢" in embed.description
    assert len(scheduled.messages) == 1


async def test_an_empty_board_or_credit_list_answers_with_its_own_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing to rank or list, each command still answers with its titled, coloured panel."""

    async def nothing(**_kwargs: object) -> list[object]:
        """Answers an empty board or contract list."""
        return []

    for facade in ("top_n", "top_losers", "list_loan_contracts"):
        monkeypatch.setattr(economy, facade, nothing)
    record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = EconomyCogs(bot=_bot())
    panels: list[tuple[str | None, str | None, int | None, bool]] = []
    for command in (
        EconomyCogs.leaderboard,
        EconomyCogs.loss_leaderboard,
        EconomyCogs.credit_status,
    ):
        interaction = FakeInteraction(user=FakeUser(user_id=1))
        await command.callback(cog, interaction)
        sent = interaction.followup.sent[0]
        embed = sent["embed"]
        panels.append((
            embed.title,
            embed.description,
            embed.colour.value if embed.colour else None,
            sent.get("ephemeral") is True,
        ))

    assert panels == [
        (
            LEADERBOARD_TITLE,
            "### 尚未開張\n/games blackjack 或 /games dragon_gate 開局就會上榜",
            LEADERBOARD_COLOR,
            False,
        ),
        (
            LOSS_LEADERBOARD_TITLE,
            "### 今天還沒有人輸錢\n/games blackjack 或 /games dragon_gate 開局就可能進榜",
            LOSS_LEADERBOARD_COLOR,
            False,
        ),
        ("信貸狀態", "### 目前沒有有效信貸", BORROW_COLOR, True),
    ]


async def test_a_request_over_the_ceiling_is_refused_before_anyone_is_asked() -> None:
    """The ceiling is checked at request time too, so no unanswerable panel is posted."""
    await seed_participant(user_id=1, name="alice", amount=1_000)
    assert await get_credit_ceiling(user_id=1) == 2_000

    cog = EconomyCogs(bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999))))
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), guild_id=LENDING_GUILD)
    await EconomyCogs.central_bank_borrow.callback(
        cog, as_interaction(fake=interaction), amount="2001", monthly_rate_percent=0.0
    )

    embed = interaction.followup.sent[0]["embed"]
    assert embed.title == "🏛️ 超過個人信用上限"
    assert "view" not in interaction.followup.sent[0]


@pytest.mark.parametrize("balance", [VIP_PURCHASE_COST, 120_000])
async def test_overlapping_vip_purchases_report_one_purchase(
    monkeypatch: pytest.MonkeyPatch, balance: int
) -> None:
    await seed_balance(user_id=1, name="alice", amount=balance)
    admitted = asyncio.Barrier(parties=2)
    purchase_lock = asyncio.Lock()

    async def overlapping_purchase(
        user_id: int, name: str, avatar_url: str
    ) -> VipPurchaseResult | None:
        # Both commands must reach the purchase before either can become VIP.
        await admitted.wait()
        async with purchase_lock:
            return await buy_vip(user_id=user_id, name=name, avatar_url=avatar_url)

    monkeypatch.setattr("discordbot.cogs.economy.cog.buy_vip", overlapping_purchase)
    cog = EconomyCogs(bot=as_bot(fake=SimpleNamespace()))
    interactions = [
        FakeInteraction(user=FakeUser(user_id=1, name="alice"), slash_command=True)
        for _ in range(2)
    ]
    async with asyncio.timeout(delay=5):
        await asyncio.gather(
            *(
                EconomyCogs.vip_command.callback(
                    self=cog, interaction=as_interaction(fake=interaction)
                )
                for interaction in interactions
            )
        )

    expected_titles = {
        build_vip_already_embed(actor_name="alice", avatar_url="").title,
        build_vip_success_embed(
            actor_name="alice",
            avatar_url="",
            result=VipPurchaseResult(
                new_balance=balance - VIP_PURCHASE_COST, cost=VIP_PURCHASE_COST
            ),
        ).title,
    }
    assert {
        interaction.followup.sent[0]["embed"].title for interaction in interactions
    } == expected_titles
    for interaction in interactions:
        assert interaction.response.deferred_ephemeral is True
        assert len(interaction.followup.sent) == 1
        assert interaction.followup.sent[0]["ephemeral"] is True
    assert await get_vip(user_id=1) is True
    account = await get_account(user_id=1)
    assert account is not None
    assert account.balance == balance - VIP_PURCHASE_COST
    assert account.total_spent == VIP_PURCHASE_COST
    assert account.total_earned - account.total_spent == account.balance


@pytest.mark.parametrize("balance", [0, VIP_PURCHASE_COST - 1])
async def test_vip_purchase_reports_an_insufficient_balance(balance: int) -> None:
    await seed_balance(user_id=1, name="alice", amount=balance)
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), slash_command=True)

    await EconomyCogs.vip_command.callback(
        self=EconomyCogs(bot=as_bot(fake=SimpleNamespace())),
        interaction=as_interaction(fake=interaction),
    )

    assert (
        interaction.followup.sent[0]["embed"].title
        == build_vip_insufficient_embed(
            actor_name="alice", avatar_url="", balance_now=balance
        ).title
    )
    assert await get_vip(user_id=1) is False
    assert await get_balance(user_id=1) == balance


async def test_existing_vip_is_not_charged_again() -> None:
    await seed_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    assert await buy_vip(user_id=1, name="alice") is not None
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), slash_command=True)

    await EconomyCogs.vip_command.callback(
        self=EconomyCogs(bot=as_bot(fake=SimpleNamespace())),
        interaction=as_interaction(fake=interaction),
    )

    assert (
        interaction.followup.sent[0]["embed"].title
        == build_vip_already_embed(actor_name="alice", avatar_url="").title
    )
    assert await get_balance(user_id=1) == 0


async def test_exhausted_vip_purchase_does_not_claim_insufficient_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await seed_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    monkeypatch.setattr("discordbot.services.economy.database._CONDITIONAL_WRITE_MAX_RETRIES", 0)
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), slash_command=True)

    await EconomyCogs.vip_command.callback(
        self=EconomyCogs(bot=as_bot(fake=SimpleNamespace())),
        interaction=as_interaction(fake=interaction),
    )

    assert interaction.followup.sent[0]["embed"].description == (
        "The purchase could not be completed. Please try again."
    )
    assert await get_vip(user_id=1) is False
    assert await get_balance(user_id=1) == VIP_PURCHASE_COST
