"""Tests for the loan-decision panels: who may press each button, and how a panel closes."""

from typing import Any
from collections.abc import Callable, Awaitable

import pytest
from sqlalchemy import select

from discordbot.cogs.economy import views
from discordbot.typings.economy import (
    CENTRAL_BANK_BASE_CAPACITY,
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
    LoanProposalStatus,
)
from discordbot.cogs.economy.views import CreditLoanDecisionView, CentralBankLoanDecisionView
from discordbot.services.economy.database import (
    LoanProposal,
    get_balance,
    open_session,
    create_personal_loan_request,
    create_central_bank_loan_request,
)

from tests.helpers.games import attached_button
from tests.helpers.casting import (
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
)
from tests.helpers.economy import (
    LENDING_GUILD,
    seed_balance,
    approve_as_admin,
    seed_participant,
    backdate_proposal,
)
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_levels
from tests.helpers.message_cleanup import record_scheduled_deletes


async def _status(proposal_id: int) -> str:
    """Reads one loan request's stored status, which compares equal to its `LoanProposalStatus`."""
    async with open_session() as session:
        result = await session.execute(
            statement=select(LoanProposal.status).where(LoanProposal.id == proposal_id)
        )
        return result.scalar_one()


async def _expired_credit_request() -> int:
    """Files alice's request to bob and ages it past its decision window; returns its id."""
    proposal = await create_personal_loan_request(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=100
    )
    await backdate_proposal(
        proposal_id=proposal.proposal_id, seconds=LOAN_PROPOSAL_TIMEOUT_SECONDS
    )
    return proposal.proposal_id


async def test_central_bank_decision_buttons_require_admin_and_allow_self_approval() -> None:
    """Central bank buttons are gated on the server admin permission Discord sent.

    Read off `Interaction.permissions` rather than looked up, which is also what refuses a
    DM: outside a guild Discord resolves no permissions at all.
    """
    await seed_participant(user_id=1, name="alice", amount=CENTRAL_BANK_BASE_CAPACITY)
    # Past the base capacity every guild lends on, so only this guild's own pool can fund it.
    amount = CENTRAL_BANK_BASE_CAPACITY + 1
    proposal = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=amount, monthly_rate_bps=0
    )
    view = CentralBankLoanDecisionView(
        proposal_id=proposal.proposal_id, creator_id=1, allow_self_approval=True
    )
    approve_button = attached_button(view=view, custom_id="central_bank:approve")

    denied = FakeInteraction(
        user=FakeUser(user_id=2, name="bob"), guild_id=LENDING_GUILD, administrator=False
    )
    await approve_button.callback(as_interaction(fake=denied))
    assert denied.followup.sent[0]["ephemeral"] is True

    in_a_dm = FakeInteraction(
        user=FakeUser(user_id=1, name="alice"), in_guild=False, administrator=False
    )
    await approve_button.callback(as_interaction(fake=in_a_dm))
    assert in_a_dm.followup.sent[0]["ephemeral"] is True
    assert await _status(proposal_id=proposal.proposal_id) == LoanProposalStatus.PENDING

    # The borrower approving her own request in her own server: it goes through only if the
    # panel hands the ledger the self-approval flag, the admin's word and this guild's pool.
    allowed = FakeInteraction(
        user=FakeUser(user_id=1, name="alice"), guild_id=LENDING_GUILD, administrator=True
    )
    await approve_button.callback(as_interaction(fake=allowed))
    assert allowed.edits[0]["view"] is None
    assert await get_balance(user_id=1) == CENTRAL_BANK_BASE_CAPACITY + amount

    withdrawn = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=100
    )
    cancel_view = CentralBankLoanDecisionView(proposal_id=withdrawn.proposal_id, creator_id=1)
    cancel_button = attached_button(view=cancel_view, custom_id="central_bank:cancel")
    denied_cancel = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await cancel_button.callback(as_interaction(fake=denied_cancel))
    assert denied_cancel.followup.sent[0]["ephemeral"] is True
    assert (
        denied_cancel.followup.sent[0]["embed"].description
        == CentralBankLoanDecisionView.CANCEL_DENIED_NOTICE
    )
    assert await _status(proposal_id=withdrawn.proposal_id) == LoanProposalStatus.PENDING

    allowed_cancel = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    await cancel_button.callback(as_interaction(fake=allowed_cancel))
    assert await _status(proposal_id=withdrawn.proposal_id) == LoanProposalStatus.CANCELED
    assert allowed_cancel.edits[0]["view"] is None


async def test_credit_decision_buttons_gate_lender_and_creator() -> None:
    """Personal credit request buttons are lender-gated, while cancel is creator-gated."""
    await seed_balance(user_id=2, name="bob", amount=1_000)
    requests = [
        await create_personal_loan_request(
            borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=100
        )
        for _ in range(3)
    ]
    approved, rejected, withdrawn = (request.proposal_id for request in requests)

    view = CreditLoanDecisionView(proposal_id=approved, lender_id=2, creator_id=1)
    approve_button = attached_button(view=view, custom_id="credit:approve")
    # The ledger turns a stranger away too; the panel must do it first, saying why.
    denied_approve = FakeInteraction(user=FakeUser(user_id=3, name="charlie"))
    await approve_button.callback(as_interaction(fake=denied_approve))
    assert denied_approve.followup.sent[0]["ephemeral"] is True
    assert (
        denied_approve.followup.sent[0]["embed"].description
        != CreditLoanDecisionView.APPROVE_FAILED_NOTICE
    )
    assert await _status(proposal_id=approved) == LoanProposalStatus.PENDING

    allowed_approve = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await approve_button.callback(as_interaction(fake=allowed_approve))
    assert await get_balance(user_id=1) == 100
    assert allowed_approve.edits[0]["view"] is None

    reject_view = CreditLoanDecisionView(proposal_id=rejected, lender_id=2, creator_id=1)
    reject_button = attached_button(view=reject_view, custom_id="credit:reject")
    denied_reject = FakeInteraction(user=FakeUser(user_id=3, name="charlie"))
    await reject_button.callback(as_interaction(fake=denied_reject))
    assert denied_reject.followup.sent[0]["ephemeral"] is True
    assert (
        denied_reject.followup.sent[0]["embed"].description
        != CreditLoanDecisionView.REJECT_FAILED_NOTICE
    )
    assert await _status(proposal_id=rejected) == LoanProposalStatus.PENDING

    allowed_reject = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await reject_button.callback(as_interaction(fake=allowed_reject))
    assert await _status(proposal_id=rejected) == LoanProposalStatus.REJECTED
    assert allowed_reject.edits[0]["view"] is None

    cancel_view = CreditLoanDecisionView(proposal_id=withdrawn, lender_id=2, creator_id=1)
    cancel_button = attached_button(view=cancel_view, custom_id="credit:cancel")
    denied_cancel = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await cancel_button.callback(as_interaction(fake=denied_cancel))
    assert denied_cancel.followup.sent[0]["ephemeral"] is True
    assert (
        denied_cancel.followup.sent[0]["embed"].description
        == CreditLoanDecisionView.CANCEL_DENIED_NOTICE
    )
    assert await _status(proposal_id=withdrawn) == LoanProposalStatus.PENDING

    allowed_cancel = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    await cancel_button.callback(as_interaction(fake=allowed_cancel))
    assert await _status(proposal_id=withdrawn) == LoanProposalStatus.CANCELED
    assert allowed_cancel.edits[0]["view"] is None


async def test_a_loan_button_is_acknowledged_before_it_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approving is where a lender is actually debited, so the ack cannot wait for the result.

    The sweep in `tests/guards/test_ack_before_write.py` reads the order off the source. This one
    observes it, so a defer that is present but unreachable — moved behind a guard, or into one
    branch — still fails here.

    Every writer, because the panel is the surface someone edits one button of.
    """
    clicked: list[tuple[str, FakeInteraction]] = []
    acked_at_write: dict[str, bool] = {}

    def noting_the_ack(
        write: Callable[..., Awaitable[object]],
    ) -> Callable[..., Awaitable[object]]:
        """Wraps a ledger write to note whether the click was acknowledged before it ran."""

        async def note_then_write(**kwargs: Any) -> object:  # noqa: ANN401 -- passes the write's own arguments through
            custom_id, interaction = clicked[-1]
            acked_at_write[custom_id] = interaction.response.deferred
            return await write(**kwargs)

        return note_then_write

    for write in ("accept_loan_proposal", "reject_loan_proposal", "cancel_loan_proposal"):
        monkeypatch.setattr(views, write, noting_the_ack(write=getattr(views, write)))

    # alice can fund every request below and decides them; bob asked her for the personal ones.
    await seed_participant(user_id=1, name="alice", amount=1_000)
    buttons: list[tuple[views.LoanDecisionViewBase, str, int]] = []
    for custom_id in ("central_bank:approve", "central_bank:reject", "central_bank:cancel"):
        central = await create_central_bank_loan_request(
            borrower_id=1, borrower_name="alice", amount=100
        )
        view = CentralBankLoanDecisionView(
            proposal_id=central.proposal_id, creator_id=1, allow_self_approval=True
        )
        buttons.append((view, custom_id, 1))
    for custom_id, presser_id in (
        ("credit:approve", 1),
        ("credit:reject", 1),
        ("credit:cancel", 2),
    ):
        credit = await create_personal_loan_request(
            borrower_id=2, borrower_name="bob", lender_id=1, lender_name="alice", amount=100
        )
        view = CreditLoanDecisionView(proposal_id=credit.proposal_id, lender_id=1, creator_id=2)
        buttons.append((view, custom_id, presser_id))

    for view, custom_id, presser_id in buttons:
        button = attached_button(view=view, custom_id=custom_id)
        interaction = FakeInteraction(
            user=FakeUser(user_id=presser_id), guild_id=LENDING_GUILD, administrator=True
        )
        clicked.append((custom_id, interaction))
        await button.callback(as_interaction(fake=interaction))
        assert interaction.edits, f"{custom_id} never edited the panel it was clicked on"

    # Keyed by button rather than collected in order: which of them acked first says nothing,
    # while a missing key is a button that reached no write at all and so proved nothing.
    assert acked_at_write == {custom_id: True for _, custom_id, _ in buttons}, (
        f"a loan button wrote before acknowledging: {acked_at_write}"
    )


async def test_loan_decision_timeout_rejects_and_schedules_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loan request views reject stale proposals and remove buttons on timeout."""
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    credit_id = await _expired_credit_request()
    central = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=100
    )
    await backdate_proposal(proposal_id=central.proposal_id, seconds=LOAN_PROPOSAL_TIMEOUT_SECONDS)

    credit_message = FakeDiscordMessage()
    credit_view = CreditLoanDecisionView(proposal_id=credit_id, lender_id=2, creator_id=1)
    credit_view.message = as_message(fake=credit_message)
    await credit_view.on_timeout()

    central_message = FakeDiscordMessage()
    central_view = CentralBankLoanDecisionView(proposal_id=central.proposal_id, creator_id=1)
    central_view.message = as_message(fake=central_message)
    await central_view.on_timeout()

    assert await _status(proposal_id=credit_id) == LoanProposalStatus.REJECTED
    assert await _status(proposal_id=central.proposal_id) == LoanProposalStatus.REJECTED
    # order-contract: same sequential awaits, so cleanup is scheduled in construction order.
    assert scheduled.messages == [credit_message, central_message]
    assert credit_message.edits[0]["view"] is None
    assert central_message.edits[0]["view"] is None
    assert credit_message.edits[0]["embed"].title == CreditLoanDecisionView.TIMEOUT_TITLE
    assert central_message.edits[0]["embed"].title == CentralBankLoanDecisionView.TIMEOUT_TITLE


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
    proposal_id = await _expired_credit_request()
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    reports = capture_levels(monkeypatch=monkeypatch, levels=("info", "warn"))
    message = FakeDiscordMessage()
    message.edit_failure = failure
    view = CreditLoanDecisionView(proposal_id=proposal_id, lender_id=2, creator_id=1)
    view.message = as_message(fake=message)

    await view.on_timeout()

    assert [(name, "_exc_info" in fields) for name, _message, fields in reports] == [
        (level, traceback)
    ]
    assert scheduled.messages == [message]


@pytest.mark.parametrize(
    argnames=("custom_id", "user_id"),
    argvalues=[("credit:approve", 3), ("credit:reject", 3), ("credit:cancel", 2)],
    ids=["bystander_approves", "bystander_rejects", "lender_cancels"],
)
async def test_a_loan_panel_kept_open_by_a_refused_press_closes_through_it(
    monkeypatch: pytest.MonkeyPatch, custom_id: str, user_id: int
) -> None:
    """A refused press restarts the panel's timer, so the timeout closes the panel through it.

    By then the slash command's own token, the only other way to the panel, can be past its life.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    proposal = await create_personal_loan_request(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=100
    )
    followup = FakeDiscordMessage()
    followup.edit_failure = make_not_found(message="Unknown Webhook")
    view = CreditLoanDecisionView(proposal_id=proposal.proposal_id, lender_id=2, creator_id=1)
    view.message = as_message(fake=followup)
    panel = FakeDiscordMessage()
    press = FakeInteraction(user=FakeUser(user_id=user_id, name="charlie"), message=panel)
    button = attached_button(view=view, custom_id=custom_id)

    await button.callback(as_interaction(fake=press))
    await backdate_proposal(
        proposal_id=proposal.proposal_id, seconds=LOAN_PROPOSAL_TIMEOUT_SECONDS
    )
    await view.on_timeout()

    assert [(edit["embed"].title, edit["view"]) for edit in press.edits] == [
        (CreditLoanDecisionView.TIMEOUT_TITLE, None)
    ]
    assert (scheduled.messages, scheduled.interactions) == ([panel], [press])
    assert [sent["ephemeral"] for sent in press.followup.sent] == [True]


async def test_a_decided_loan_panel_is_deleted_through_the_deciding_press(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The delete runs three minutes after the deciding press, while that press's token lives.

    Refused presses can have kept the panel open until the slash command's own token is near
    its end, so the delete cannot ride that one.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    proposal = await create_personal_loan_request(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=100
    )
    view = CreditLoanDecisionView(proposal_id=proposal.proposal_id, lender_id=2, creator_id=1)
    view.message = as_message(fake=FakeDiscordMessage())
    panel = FakeDiscordMessage()
    press = FakeInteraction(user=FakeUser(user_id=1, name="alice"), message=panel)
    button = attached_button(view=view, custom_id="credit:cancel")

    await button.callback(as_interaction(fake=press))

    assert press.edits[0]["embed"].title == CreditLoanDecisionView.CANCEL_TITLE
    assert (scheduled.messages, scheduled.interactions) == ([panel], [press])


@pytest.mark.parametrize(
    argnames=("custom_id", "presser_id"),
    argvalues=[("credit:approve", 2), ("credit:reject", 2), ("credit:cancel", 1)],
    ids=["lender_approves", "lender_rejects", "borrower_cancels"],
)
async def test_a_press_past_the_window_closes_the_panel_as_expired(
    monkeypatch: pytest.MonkeyPatch, custom_id: str, presser_id: int
) -> None:
    """The press that finds a request expired closes its panel, as the view's timeout would.

    Every press restarts the view's own timer, so the view can outlive the request; once a
    press has expired it, the timeout finds nothing left to expire and leaves the buttons up.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    await seed_balance(user_id=2, name="bob", amount=1_000)
    proposal = await create_personal_loan_request(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=500
    )
    await backdate_proposal(
        proposal_id=proposal.proposal_id, seconds=LOAN_PROPOSAL_TIMEOUT_SECONDS
    )
    panel = FakeDiscordMessage()
    view = CreditLoanDecisionView(proposal_id=proposal.proposal_id, lender_id=2, creator_id=1)
    view.message = as_message(fake=panel)
    button = attached_button(view=view, custom_id=custom_id)
    presser = FakeInteraction(user=FakeUser(user_id=presser_id, name="presser"), message=panel)

    await button.callback(as_interaction(fake=presser))

    assert presser.followup.sent == []
    assert presser.edits[0]["embed"].title == CreditLoanDecisionView.TIMEOUT_TITLE
    assert presser.edits[0]["view"] is None
    assert view.is_finished()
    assert scheduled.messages == [panel]
    assert await get_balance(user_id=2) == 1_000


async def test_an_administrator_rejects_a_central_bank_request_from_its_panel() -> None:
    """The ledger takes the panel's word that the clicker administers this server.

    Without it no central-bank request could be turned down, since nobody is named to decide one.
    """
    await seed_participant(user_id=1, name="alice", amount=1_000)
    proposal = await create_central_bank_loan_request(
        borrower_id=1, borrower_name="alice", amount=100
    )
    assert proposal is not None
    view = CentralBankLoanDecisionView(proposal_id=proposal.proposal_id, creator_id=1)
    reject_button = attached_button(view=view, custom_id="central_bank:reject")
    admin = FakeInteraction(
        user=FakeUser(user_id=99, name="banker"), guild_id=LENDING_GUILD, administrator=True
    )

    await reject_button.callback(as_interaction(fake=admin))

    assert admin.followup.sent == []
    assert admin.edits[0]["embed"].title == CentralBankLoanDecisionView.REJECT_TITLE
    assert (
        await approve_as_admin(proposal_id=proposal.proposal_id, actor_id=99, name="banker")
        is None
    )
    assert await get_balance(user_id=1) == 1_000
