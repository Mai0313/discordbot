"""Tests for the economy cog's slash commands and loan-decision buttons.

The ledger facade is replaced where the cog and its views import it, so nothing here writes the
economy database: what these cover is which facade call each command makes, whether it is
acknowledged before that write, and who gets to see the answer.
"""

from __future__ import annotations

import ast
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any
from pathlib import Path
from datetime import UTC, datetime
from unittest.mock import ANY

import pytest
import logfire

from discordbot.utils import interaction_responses as interactions
from discordbot.cogs.economy import cog as economy
from discordbot.cogs.economy import views
from discordbot.typings.economy import (
    VIP_PURCHASE_COST,
    PortfolioView,
    LoanLenderType,
    TransferResult,
    AccountSnapshot,
    LeaderboardEntry,
    LoanContractView,
    LoanProposalKind,
    LoanProposalView,
    CentralBankStatus,
    LoanPaymentResult,
    VipPurchaseResult,
    LoanContractStatus,
    LoanProposalStatus,
    CasinoLedgerSnapshot,
    LossLeaderboardEntry,
    BalanceAdjustmentResult,
    LoanProposalAcceptResult,
)
from discordbot.cogs.economy.cog import EconomyCogs
from discordbot.cogs.economy.views import CreditLoanDecisionView, CentralBankLoanDecisionView

from tests.helpers.casting import (
    as_bot,
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
)
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

    from nextcord.ext import commands


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


async def fake_top_losers(limit: int, include_hidden: bool = False) -> list[LossLeaderboardEntry]:
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


async def fake_transfer(  # noqa: PLR0913 -- mirrors transfer signature
    sender_id: int,
    sender_name: str,
    receiver_id: int,
    receiver_name: str,
    amount: int,
    sender_avatar_url: str = "",
    receiver_avatar_url: str = "",
) -> TransferResult | None:
    """Returns a successful fake transfer result."""
    return TransferResult(
        sender_balance=50, receiver_balance=100, received_amount=100, tax_amount=0
    )


async def fake_adjust_balance(
    user_id: int, name: str, delta: int, allow_negative: bool = False, avatar_url: str = ""
) -> BalanceAdjustmentResult:
    """Returns a successful fake manual adjustment result."""
    return BalanceAdjustmentResult(new_balance=150 + delta, applied_delta=delta)


def _fake_loan_proposal(kind: LoanProposalKind) -> LoanProposalView:
    """Builds a fake loan proposal view."""
    return LoanProposalView(
        proposal_id=1,
        kind=kind,
        status=LoanProposalStatus.PENDING,
        lender_type=LoanLenderType.CENTRAL_BANK
        if kind == LoanProposalKind.CENTRAL_BANK_REQUEST
        else LoanLenderType.USER,
        borrower_id=1,
        borrower_name="alice",
        lender_id=None if kind == LoanProposalKind.CENTRAL_BANK_REQUEST else 2,
        lender_name="bob",
        amount=100,
        monthly_rate_bps=300,
        created_at=datetime.now(tz=UTC),
    )


async def fake_create_loan_request(**_kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
    """Returns a fake personal request."""
    return _fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)


async def fake_create_central_bank_request(**_kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
    """Returns a fake central-bank request."""
    return _fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)


async def fake_reject_loan_proposal(
    proposal_id: int, actor_id: int, approver_is_guild_admin: bool = False
) -> LoanProposalView:
    """Returns a rejected fake proposal."""
    proposal = _fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)
    return proposal.model_copy(update={"status": LoanProposalStatus.REJECTED})


async def fake_cancel_loan_proposal(proposal_id: int, actor_id: int) -> LoanProposalView:
    """Returns a canceled fake proposal."""
    proposal = _fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)
    return proposal.model_copy(update={"status": LoanProposalStatus.CANCELED})


async def fake_accept_loan_proposal(**_kwargs: Any) -> LoanProposalAcceptResult:  # noqa: ANN401 -- command facade double
    """Returns a fake accepted proposal result."""
    contract = LoanContractView(
        contract_id=1,
        lender_type=LoanLenderType.USER,
        lender_id=2,
        lender_name="bob",
        borrower_id=1,
        borrower_name="alice",
        principal_remaining=100,
        interest_due=0,
        monthly_rate_bps=300,
        opened_at=datetime.now(tz=UTC),
        last_interest_accrued_at=datetime.now(tz=UTC),
        status=LoanContractStatus.ACTIVE,
    )
    return LoanProposalAcceptResult(
        contract=contract,
        borrower_balance=250,
        lender_balance=100,
        central_bank_available_credit=1_000,
    )


async def fake_list_loan_contracts(user_id: int) -> list[LoanContractView]:
    """Returns one active loan contract."""
    return [
        LoanContractView(
            contract_id=1,
            lender_type=LoanLenderType.USER,
            lender_id=2,
            lender_name="bob",
            borrower_id=user_id,
            borrower_name="alice",
            principal_remaining=100,
            interest_due=3,
            monthly_rate_bps=300,
            opened_at=datetime.now(tz=UTC),
            last_interest_accrued_at=datetime.now(tz=UTC),
            status=LoanContractStatus.ACTIVE,
        )
    ]


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


async def fake_get_central_bank_status(
    guild_id: int, exclude_user_ids: tuple[int, ...] = ()
) -> CentralBankStatus:
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


def ignore_scheduled_public_message(
    message: FakeDiscordMessage, delay: float = 180, user_name: str | None = None
) -> None:
    """Ignores cleanup scheduling in command smoke tests."""
    return


def _record_scheduled(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> list[FakeDiscordMessage]:
    """Replaces `module`'s public-message cleanup with a recorder; returns what it scheduled."""
    scheduled: list[FakeDiscordMessage] = []

    def record_scheduled(
        message: FakeDiscordMessage, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records the message handed over for cleanup."""
        del delay, user_name
        scheduled.append(message)

    monkeypatch.setattr(module, "schedule_public_message_delete", record_scheduled)
    return scheduled


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


# Everything `cogs/economy/` imports from the ledger. Split rather than listed: a new import
# fails the sweep below until someone decides which half it belongs in, which is the only thing
# that keeps this from silently missing the next money-moving call.
_LEDGER_MUTATIONS = frozenset({
    "adjust_balance",
    "buy_vip",
    "call_central_bank_loans",
    "call_personal_loans",
    "create_central_bank_loan_request",
    "create_personal_loan_request",
    "repay_central_bank_loans",
    "repay_personal_loans",
    "transfer",
    # Both accrue and persist interest before returning, so they take the same write lock.
    "get_portfolio",
    "list_loan_contracts",
    "accept_loan_proposal",
    "cancel_loan_proposal",
    "reject_expired_loan_proposal",
    "reject_loan_proposal",
    # An upsert of its own row, so it takes the write lock like any other.
    "record_guild_participant",
})


_LEDGER_READS = frozenset({
    "get_account",
    "get_admin",
    "get_balance",
    "get_casino_ledger",
    "get_central_bank_status",
    "get_credit_ceiling",
    "get_vip",
    "top_losers",
    "top_n",
})


# The one writer in these modules with no interaction to acknowledge: a view's expiry fires on
# its own timer. Exempt by name rather than by skipping the file it sits in, so the buttons
# beside it stay swept.
_ACK_SWEEP_EXEMPT = frozenset({("views.py", "on_timeout")})


def _first_unavoidable_ack(node: ast.AST) -> ast.Call | None:
    """The first acknowledgement a write inside this function cannot go around.

    Recurses into `try` and `with` bodies, which always run, and deliberately not into `if`,
    `for` or `while`: a permission guard that acks and returns sits inside an `if`, and an ack
    found there would vouch for a write further down that never passes through it.
    """
    for statement in getattr(node, "body", []):
        if isinstance(statement, ast.Try | ast.With | ast.AsyncWith):
            found = _first_unavoidable_ack(statement)
            if found is not None:
                return found
            continue
        # `AsyncFunctionDef` is not a subclass of `FunctionDef`, and in an async cog a nested
        # def is far more likely to be the async one — an ack inside a closure may never run.
        if isinstance(
            statement, ast.If | ast.For | ast.While | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        for inner in ast.walk(statement):
            if not isinstance(inner, ast.Call):
                continue
            name = ast.unparse(inner.func)
            if (
                name.endswith("response.defer")
                or name.endswith("response.send_message")
                or name.endswith("send_ephemeral_response")
            ):
                return inner
    return None


def _edits_the_original_without_deferring(node: ast.AsyncFunctionDef, ack: ast.Call | None) -> str:
    """Why an ack that is not a `defer` breaks a handler answering by editing its own message.

    `edit_original_message` targets whatever the interaction answered with, so a handler that
    replied with `send_message` first edits THAT — the settlement embed lands in an ephemeral
    notice and the panel keeps its live buttons over a decided loan. It is silent, where the
    `response.edit_message` this replaced would have raised `InteractionResponded`.
    """
    edits = any(
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "edit_response_embed"
        for inner in ast.walk(node)
    )
    if not edits or (ack is not None and ast.unparse(ack.func).endswith("response.defer")):
        return ""
    return f"{node.name} edits the original response without having deferred it"


def _writes_before(node: ast.AsyncFunctionDef, ack: ast.Call | None) -> list[str]:
    """Ledger writes in this function that run before it has acknowledged anything."""
    acked_at = ack.lineno if ack is not None else None
    found: list[str] = []
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call) or not isinstance(inner.func, ast.Name):
            continue
        if inner.func.id in _LEDGER_MUTATIONS and (acked_at is None or inner.lineno < acked_at):
            found.append(f"{node.name} writes at line {inner.lineno} before any ack")
    return found


async def test_a_failed_money_command_stays_private(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acking before the write must not turn a private failure into a public one.

    Having no matching loan is an ordinary mistake, not an edge case. The placeholder's flag
    and the followup's are independent and both matter: the first decides whether the channel
    sees that the attempt happened, the second whether it sees what came of it.

    All four, because the ack is written out once per command and only a sweep catches the one
    someone changes on its own.
    """

    async def nothing_to_settle(**_kwargs: object) -> None:
        """Stands in for a write that matched no loan."""
        return

    # A repayment of zero is refused before the command ever acks, so these have to carry a
    # real amount; a collection of zero means "everything owed" and is the ordinary call.
    collect: dict[str, object] = {"member": FakeUser(user_id=2), "amount": "0"}
    commands_under_test = (
        ("credit_repay", "repay_personal_loans", {"member": FakeUser(user_id=2), "amount": "5"}),
        ("credit_call", "call_personal_loans", collect),
        ("central_bank_repay", "repay_central_bank_loans", {"amount": "5"}),
        ("central_bank_call", "call_central_bank_loans", collect),
    )

    monkeypatch.setattr(economy, "record_guild_participant", fake_record_guild_participant)
    for command, ledger_call, kwargs in commands_under_test:
        monkeypatch.setattr(economy, ledger_call, nothing_to_settle)
        cog = EconomyCogs(bot=as_bot(fake=SimpleNamespace()))
        # Administrator, or `central_bank_call` returns from its permission branch and this
        # stops covering the failure path it was written for — while still passing, because
        # that branch defers and follows up ephemerally just like the one under test.
        interaction = FakeInteraction(
            user=FakeUser(user_id=1, display_name="Alice"), administrator=True
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


def test_every_money_command_acknowledges_before_it_mutates() -> None:
    """Discord kills the token three seconds after dispatch; a SQLite writer may wait longer.

    A command that writes first and acks on the result can therefore commit someone's money
    and then fail to say so, reaching them as Discord's "the application did not respond" with
    no statement of what happened.

    Structural because the race needs a contended write lock to observe, and because this is
    the kind of ordering someone reintroduces by moving an `if`.

    Covers the buttons as well as the commands: approving a loan is where the lender is
    actually debited, and its fix is a different one (a component defers into an edit of the
    panel, not into a followup), so it is exactly the pair someone changes one half of.
    """
    economy_dir = Path(__file__).resolve().parents[1] / "src/discordbot/cogs/economy"
    scanned: list[str] = []
    exempted: set[tuple[str, str]] = set()
    offenders: list[str] = []
    for path in sorted(economy_dir.glob("*.py")):
        scanned.append(path.name)
        for node in ast.walk(ast.parse(source=path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            if (path.name, node.name) in _ACK_SWEEP_EXEMPT:
                exempted.add((path.name, node.name))
                continue
            ack = _first_unavoidable_ack(node)
            offenders.extend(_writes_before(node=node, ack=ack))
            offenders.extend(filter(None, [_edits_the_original_without_deferring(node, ack)]))

    assert {"cog.py", "views.py"} <= set(scanned), "the sweep missed a module it must cover"
    # An exemption is the other way a writer leaves this guard while the guard stays green.
    assert exempted == _ACK_SWEEP_EXEMPT, f"exemption never matched anything: {_ACK_SWEEP_EXEMPT}"
    assert not offenders, f"money handlers whose acknowledgement is wrong: {sorted(offenders)}"


def test_the_ack_sweep_accounts_for_every_ledger_name_the_cogs_import() -> None:
    """A new import has to be classified before it can be missed.

    The sweep can only see writes it was told are writes, and a hand-kept list of those goes
    stale the first time someone imports one more.
    """
    economy_dir = Path(__file__).resolve().parents[1] / "src/discordbot/cogs/economy"
    imported: set[str] = set()
    for path in sorted(economy_dir.glob("*.py")):
        for node in ast.walk(ast.parse(source=path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                "economy.database"
            ):
                imported.update(alias.name for alias in node.names)

    unclassified = imported - _LEDGER_MUTATIONS - _LEDGER_READS
    assert imported, "no ledger imports found, so this guard is watching nothing"
    assert not unclassified, (
        f"classify these as a ledger write or a read before the ack sweep can cover them: "
        f"{sorted(unclassified)}"
    )


async def test_economy_commands_use_database_facade(  # noqa: PLR0915 -- command smoke exercises one facade surface
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies economy slash commands call the database facade and send embeds."""
    scheduled = _record_scheduled(monkeypatch=monkeypatch, module=interactions)
    monkeypatch.setattr(economy, "get_balance", fake_get_balance)
    monkeypatch.setattr(economy, "get_vip", fake_get_vip)
    monkeypatch.setattr(economy, "get_admin", fake_get_admin)
    monkeypatch.setattr(economy, "top_n", fake_top_n)
    monkeypatch.setattr(economy, "top_losers", fake_top_losers)
    monkeypatch.setattr(economy, "get_account", fake_get_account)
    monkeypatch.setattr(economy, "get_casino_ledger", fake_get_casino_ledger)
    monkeypatch.setattr(economy, "transfer", fake_transfer)
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
    interaction = FakeInteraction(user=FakeUser(user_id=1), administrator=True)
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
    assert len(interaction.followup.sent) == 17
    assert len(scheduled) == 12
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
    assert interaction.followup.sent[9].get("ephemeral") is not True
    assert interaction.followup.sent[10].get("ephemeral") is not True
    assert interaction.followup.sent[11].get("ephemeral") is True
    assert interaction.followup.sent[13].get("ephemeral") is not True
    assert interaction.followup.sent[14].get("ephemeral") is not True
    assert interaction.followup.sent[15].get("ephemeral") is not True
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
    central_bank_payload = interaction.followup.sent[12]
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


async def test_central_bank_decision_buttons_require_admin_and_allow_self_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Central bank buttons are gated on the server admin permission Discord sent.

    Read off `Interaction.permissions` rather than looked up, which is also what refuses a
    DM: outside a guild Discord resolves no permissions at all.
    """
    captured_accept_kwargs: dict[str, Any] = {}
    captured_cancel_kwargs: dict[str, int] = {}

    async def fake_accept_for_button(**kwargs: Any) -> LoanProposalAcceptResult:  # noqa: ANN401 -- command facade double
        """Records approval arguments and returns a fake accepted proposal."""
        captured_accept_kwargs.update(kwargs)
        return await fake_accept_loan_proposal()

    async def fake_cancel_for_button(proposal_id: int, actor_id: int) -> LoanProposalView:
        """Records cancellation arguments and returns a fake canceled proposal."""
        captured_cancel_kwargs.update({"proposal_id": proposal_id, "actor_id": actor_id})
        return await fake_cancel_loan_proposal(proposal_id=proposal_id, actor_id=actor_id)

    monkeypatch.setattr(views, "accept_loan_proposal", fake_accept_for_button)
    monkeypatch.setattr(views, "cancel_loan_proposal", fake_cancel_for_button)
    view = CentralBankLoanDecisionView(
        bot=_bot(), proposal_id=42, creator_id=1, allow_self_approval=True
    )
    approve_button = next(
        child
        for child in view.children
        if getattr(child, "custom_id", "") == "central_bank:approve"
    )

    denied = FakeInteraction(user=FakeUser(user_id=2, name="bob"), administrator=False)
    await approve_button.callback(as_interaction(fake=denied))
    assert denied.followup.sent[0]["ephemeral"] is True
    assert captured_accept_kwargs == {}

    in_a_dm = FakeInteraction(
        user=FakeUser(user_id=1, name="alice"), in_guild=False, administrator=False
    )
    await approve_button.callback(as_interaction(fake=in_a_dm))
    assert in_a_dm.followup.sent[0]["ephemeral"] is True
    assert captured_accept_kwargs == {}

    allowed = FakeInteraction(
        user=FakeUser(user_id=1, name="alice"), guild_id=321, administrator=True
    )
    await approve_button.callback(as_interaction(fake=allowed))
    assert captured_accept_kwargs["proposal_id"] == 42
    assert captured_accept_kwargs["actor_id"] == 1
    assert captured_accept_kwargs["guild_id"] == 321
    assert captured_accept_kwargs["approver_is_guild_admin"] is True
    assert captured_accept_kwargs["allow_central_bank_self_approval"] is True
    assert allowed.edits[0]["view"] is None

    cancel_view = CentralBankLoanDecisionView(bot=_bot(), proposal_id=43, creator_id=1)
    cancel_button = next(
        child
        for child in cancel_view.children
        if getattr(child, "custom_id", "") == "central_bank:cancel"
    )
    denied_cancel = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await cancel_button.callback(as_interaction(fake=denied_cancel))
    assert denied_cancel.followup.sent[0]["ephemeral"] is True
    assert captured_cancel_kwargs == {}

    allowed_cancel = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    await cancel_button.callback(as_interaction(fake=allowed_cancel))
    assert captured_cancel_kwargs == {"proposal_id": 43, "actor_id": 1}
    assert allowed_cancel.edits[0]["view"] is None


async def test_credit_decision_buttons_gate_lender_and_creator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Personal credit request buttons are lender-gated, while cancel is creator-gated."""
    captured_accept_kwargs: dict[str, Any] = {}
    captured_reject_kwargs: dict[str, int] = {}
    captured_cancel_kwargs: dict[str, int] = {}

    async def fake_accept_for_button(**kwargs: Any) -> LoanProposalAcceptResult:  # noqa: ANN401 -- command facade double
        """Records approval arguments and returns a fake accepted proposal."""
        captured_accept_kwargs.update(kwargs)
        return await fake_accept_loan_proposal()

    async def fake_reject_for_button(
        proposal_id: int, actor_id: int, approver_is_guild_admin: bool = False
    ) -> LoanProposalView:
        """Records rejection arguments and returns a fake rejected proposal."""
        captured_reject_kwargs.update({"proposal_id": proposal_id, "actor_id": actor_id})
        return await fake_reject_loan_proposal(proposal_id=proposal_id, actor_id=actor_id)

    async def fake_cancel_for_button(proposal_id: int, actor_id: int) -> LoanProposalView:
        """Records cancellation arguments and returns a fake canceled proposal."""
        captured_cancel_kwargs.update({"proposal_id": proposal_id, "actor_id": actor_id})
        return await fake_cancel_loan_proposal(proposal_id=proposal_id, actor_id=actor_id)

    monkeypatch.setattr(views, "accept_loan_proposal", fake_accept_for_button)
    monkeypatch.setattr(views, "reject_loan_proposal", fake_reject_for_button)
    monkeypatch.setattr(views, "cancel_loan_proposal", fake_cancel_for_button)
    view = CreditLoanDecisionView(proposal_id=42, lender_id=2, creator_id=1)
    approve_button = next(
        child for child in view.children if getattr(child, "custom_id", "") == "credit:approve"
    )

    denied_approve = FakeInteraction(user=FakeUser(user_id=3, name="charlie"))
    await approve_button.callback(as_interaction(fake=denied_approve))
    assert denied_approve.followup.sent[0]["ephemeral"] is True
    assert captured_accept_kwargs == {}

    allowed_approve = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await approve_button.callback(as_interaction(fake=allowed_approve))
    assert captured_accept_kwargs["proposal_id"] == 42
    assert captured_accept_kwargs["actor_id"] == 2
    assert allowed_approve.edits[0]["view"] is None

    reject_view = CreditLoanDecisionView(proposal_id=43, lender_id=2, creator_id=1)
    reject_button = next(
        child
        for child in reject_view.children
        if getattr(child, "custom_id", "") == "credit:reject"
    )
    denied_reject = FakeInteraction(user=FakeUser(user_id=3, name="charlie"))
    await reject_button.callback(as_interaction(fake=denied_reject))
    assert denied_reject.followup.sent[0]["ephemeral"] is True
    assert captured_reject_kwargs == {}

    allowed_reject = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await reject_button.callback(as_interaction(fake=allowed_reject))
    assert captured_reject_kwargs == {"proposal_id": 43, "actor_id": 2}
    assert allowed_reject.edits[0]["view"] is None

    cancel_view = CreditLoanDecisionView(proposal_id=44, lender_id=2, creator_id=1)
    cancel_button = next(
        child
        for child in cancel_view.children
        if getattr(child, "custom_id", "") == "credit:cancel"
    )
    denied_cancel = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await cancel_button.callback(as_interaction(fake=denied_cancel))
    assert denied_cancel.followup.sent[0]["ephemeral"] is True
    assert captured_cancel_kwargs == {}

    allowed_cancel = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    await cancel_button.callback(as_interaction(fake=allowed_cancel))
    assert captured_cancel_kwargs == {"proposal_id": 44, "actor_id": 1}
    assert allowed_cancel.edits[0]["view"] is None


async def test_a_loan_button_is_acknowledged_before_it_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approving is where a lender is actually debited, so the ack cannot wait for the result.

    The sweep above reads the order off the source. This one observes it, so a defer that is
    present but unreachable — moved behind a guard, or into one branch — still fails here.

    Every writer, because the panel is the surface someone edits one button of.
    """
    clicked: list[tuple[str, FakeInteraction]] = []
    acked_at_write: dict[str, bool] = {}

    async def accept_and_note(**_kwargs: Any) -> LoanProposalAcceptResult:  # noqa: ANN401 -- command facade double
        """Records whether the click was acknowledged before the ledger was reached."""
        custom_id, interaction = clicked[-1]
        acked_at_write[custom_id] = interaction.response.deferred
        return await fake_accept_loan_proposal()

    async def resolve_and_note(**_kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
        """Same, for the three writes that close a proposal without moving money."""
        custom_id, interaction = clicked[-1]
        acked_at_write[custom_id] = interaction.response.deferred
        return await fake_cancel_loan_proposal(proposal_id=1, actor_id=1)

    monkeypatch.setattr(views, "accept_loan_proposal", accept_and_note)
    monkeypatch.setattr(views, "reject_loan_proposal", resolve_and_note)
    monkeypatch.setattr(views, "cancel_loan_proposal", resolve_and_note)

    central = CentralBankLoanDecisionView(
        bot=_bot(), proposal_id=42, creator_id=1, allow_self_approval=True
    )
    credit = CreditLoanDecisionView(proposal_id=43, lender_id=1, creator_id=1)
    buttons = [
        (central, "central_bank:approve"),
        (central, "central_bank:reject"),
        (credit, "credit:approve"),
        (credit, "credit:reject"),
        (credit, "credit:cancel"),
    ]

    for view, custom_id in buttons:
        button = next(c for c in view.children if getattr(c, "custom_id", "") == custom_id)
        interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), administrator=True)
        clicked.append((custom_id, interaction))
        await button.callback(as_interaction(fake=interaction))
        assert interaction.edits, f"{custom_id} never edited the panel it was clicked on"

    # Keyed by button rather than collected in order: which of them acked first says nothing,
    # while a missing key is a button that reached no write at all and so proved nothing.
    assert acked_at_write == {custom_id: True for _, custom_id in buttons}, (
        f"a loan button wrote before acknowledging: {acked_at_write}"
    )


async def test_loan_decision_timeout_rejects_and_schedules_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loan request views reject stale proposals and remove buttons on timeout."""
    rejected: list[int] = []

    async def fake_reject_expired_loan_proposal(proposal_id: int) -> LoanProposalView:
        """Records the expired proposal rejection."""
        rejected.append(proposal_id)
        return _fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST).model_copy(
            update={"proposal_id": proposal_id, "status": LoanProposalStatus.REJECTED}
        )

    monkeypatch.setattr(views, "reject_expired_loan_proposal", fake_reject_expired_loan_proposal)
    scheduled = _record_scheduled(monkeypatch=monkeypatch, module=views)

    credit_message = FakeDiscordMessage()
    credit_view = CreditLoanDecisionView(proposal_id=42, lender_id=2, creator_id=1)
    credit_view.message = as_message(fake=credit_message)
    await credit_view.on_timeout()

    central_message = FakeDiscordMessage()
    central_view = CentralBankLoanDecisionView(bot=_bot(), proposal_id=43, creator_id=1)
    central_view.message = as_message(fake=central_message)
    await central_view.on_timeout()

    # order-contract: each `on_timeout` is awaited to completion before the next view exists.
    assert rejected == [42, 43]
    # order-contract: same sequential awaits, so cleanup is scheduled in construction order.
    assert scheduled == [credit_message, central_message]
    assert credit_message.edits[0]["view"] is None
    assert central_message.edits[0]["view"] is None
    credit_timeout_title = credit_message.edits[0]["embed"].title
    assert credit_timeout_title is not None
    assert "逾時" in credit_timeout_title
    central_timeout_title = central_message.edits[0]["embed"].title
    assert central_timeout_title is not None
    assert "逾時" in central_timeout_title


@pytest.mark.parametrize(
    argnames=("failure", "level", "traceback"),
    argvalues=[
        (make_forbidden(message="Missing Access"), "warn", False),
        (make_not_found(message="Unknown Message"), "info", False),
        (make_server_error(), "warn", True),
    ],
    ids=["refused", "message_gone", "broke"],
)
async def test_a_loan_timeout_edit_that_fails_is_reported_and_still_cleaned_up(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, level: str, traceback: bool
) -> None:
    """Nothing awaits a timeout, so its failure is logged here at the level its cause earns."""

    async def fake_reject_expired_loan_proposal(proposal_id: int) -> LoanProposalView:
        """Answers the rejection the timeout asks for."""
        return _fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST).model_copy(
            update={"proposal_id": proposal_id, "status": LoanProposalStatus.REJECTED}
        )

    monkeypatch.setattr(views, "reject_expired_loan_proposal", fake_reject_expired_loan_proposal)
    scheduled = _record_scheduled(monkeypatch=monkeypatch, module=views)
    reports: list[tuple[str, dict[str, object]]] = []
    for name in ("info", "warn"):
        monkeypatch.setattr(
            target=logfire,
            name=name,
            value=lambda _message, name=name, **fields: reports.append((name, fields)),
        )
    message = FakeDiscordMessage()
    message.edit_failure = failure
    view = CreditLoanDecisionView(proposal_id=42, lender_id=2, creator_id=1)
    view.message = as_message(fake=message)

    await view.on_timeout()

    assert [(name, "_exc_info" in fields) for name, fields in reports] == [(level, traceback)]
    assert scheduled == [message]


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


def test_a_positive_amount_refuses_zero() -> None:
    """Zero is a well-formed amount, but no command that takes a positive one accepts it."""
    assert economy._parse_positive_amount(raw_amount="0") is None


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
    monkeypatch.setattr(
        interactions, "schedule_public_message_delete", ignore_scheduled_public_message
    )
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
    cached_sender = FakeUser(user_id=1, name="alice")
    cached_sender.__dict__["guild_avatar"] = SimpleNamespace(
        url="https://example.test/alice-server.png"
    )
    cached_receiver = FakeUser(user_id=2, name="bob")
    cached_receiver.__dict__["guild_avatar"] = SimpleNamespace(
        url="https://example.test/bob-server.png"
    )
    members = {cached_sender.id: cached_sender, cached_receiver.id: cached_receiver}

    async def fail_fetch_member(user_id: int) -> FakeUser:
        """Fails if the helper ignores the cached member path."""
        raise AssertionError(f"unexpected fetch_member({user_id})")

    guild = SimpleNamespace(get_member=members.get, fetch_member=fail_fetch_member)
    interaction = FakeInteraction(user=sender)
    interaction.guild = guild
    transfers = _record_transfers(monkeypatch=monkeypatch)
    monkeypatch.setattr(
        interactions, "schedule_public_message_delete", ignore_scheduled_public_message
    )
    cog = EconomyCogs(bot=_bot())

    await EconomyCogs.give.callback(cog, interaction, member=receiver, amount="100")

    assert transfers[0]["sender_avatar_url"] == "https://example.test/alice-server.png"
    assert transfers[0]["receiver_avatar_url"] == "https://example.test/bob-server.png"


async def test_give_allows_bot_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Players may transfer balance to the bot account."""
    sender = FakeUser(user_id=1, name="alice")
    bot_receiver = FakeUser(user_id=999, name="discordbot", display_name="Dealer", bot=True)
    interaction = FakeInteraction(user=sender)
    transfers = _record_transfers(monkeypatch=monkeypatch)
    monkeypatch.setattr(
        interactions, "schedule_public_message_delete", ignore_scheduled_public_message
    )
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
    monkeypatch.setattr(economy, "get_balance", fake_get_balance)
    scheduled = _record_scheduled(monkeypatch=monkeypatch, module=interactions)
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
    assert len(scheduled) == 1


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
    monkeypatch.setattr(
        interactions, "schedule_public_message_delete", ignore_scheduled_public_message
    )
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
        return _fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)

    async def record_create_central(**kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
        captured["central_bank_borrow"] = kwargs["amount"]
        return _fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)

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
    monkeypatch.setattr(
        interactions, "schedule_public_message_delete", ignore_scheduled_public_message
    )
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
        return _fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)

    async def guard_create_central(**kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
        del kwargs
        mutated.append("create_central")
        return _fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)

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

    async def daily_losses(limit: int, include_hidden: bool = False) -> list[LossLeaderboardEntry]:
        """Returns fake daily gross loss rows."""
        return [
            LossLeaderboardEntry(user_id=1, name="alice", loss_amount=500, avatar_url=""),
            LossLeaderboardEntry(user_id=2, name="bob", loss_amount=200, avatar_url=""),
        ]

    monkeypatch.setattr(economy, "top_losers", daily_losses)
    scheduled = _record_scheduled(monkeypatch=monkeypatch, module=interactions)
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
    assert len(scheduled) == 1


async def test_loss_leaderboard_empty_state_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loss leaderboard empty state stays explicit about today's loss rows."""

    async def no_daily_losses(
        limit: int, include_hidden: bool = False
    ) -> list[LossLeaderboardEntry]:
        """Returns an empty daily loss board."""
        return []

    monkeypatch.setattr(economy, "top_losers", no_daily_losses)
    scheduled = _record_scheduled(monkeypatch=monkeypatch, module=interactions)
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.loss_leaderboard.callback(cog, interaction)

    embed = interaction.followup.sent[0]["embed"]
    assert embed.title is not None
    assert "今日輸局累計" in embed.title
    assert embed.description is not None
    assert "今天還沒有人輸錢" in embed.description
    assert len(scheduled) == 1
