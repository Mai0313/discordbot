"""Tests for the loan-decision panels: who may press each button, and how a panel closes."""

from typing import Any

import pytest

from discordbot.cogs.economy import views
from discordbot.typings.economy import (
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
    LoanProposalKind,
    LoanProposalView,
    LoanProposalStatus,
    LoanProposalAcceptResult,
)
from discordbot.cogs.economy.views import CreditLoanDecisionView, CentralBankLoanDecisionView
from discordbot.services.economy.database import (
    get_balance,
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
    fake_loan_proposal,
    personal_loan_contract,
)
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_levels
from tests.helpers.message_cleanup import record_scheduled_deletes


async def fake_reject_loan_proposal(
    proposal_id: int, actor_id: int, approver_is_guild_admin: bool = False
) -> LoanProposalView:
    """Returns a rejected fake proposal."""
    proposal = fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)
    return proposal.model_copy(update={"status": LoanProposalStatus.REJECTED})


async def fake_cancel_loan_proposal(proposal_id: int, actor_id: int) -> LoanProposalView:
    """Returns a canceled fake proposal."""
    proposal = fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)
    return proposal.model_copy(update={"status": LoanProposalStatus.CANCELED})


async def fake_accept_loan_proposal(**_kwargs: Any) -> LoanProposalAcceptResult:  # noqa: ANN401 -- command facade double
    """Returns a fake accepted proposal result."""
    return LoanProposalAcceptResult(
        contract=personal_loan_contract(),
        borrower_balance=250,
        lender_balance=100,
        central_bank_available_credit=1_000,
    )


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
    view = CentralBankLoanDecisionView(proposal_id=42, creator_id=1, allow_self_approval=True)
    approve_button = attached_button(view=view, custom_id="central_bank:approve")

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

    cancel_view = CentralBankLoanDecisionView(proposal_id=43, creator_id=1)
    cancel_button = attached_button(view=cancel_view, custom_id="central_bank:cancel")
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
    approve_button = attached_button(view=view, custom_id="credit:approve")

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
    reject_button = attached_button(view=reject_view, custom_id="credit:reject")
    denied_reject = FakeInteraction(user=FakeUser(user_id=3, name="charlie"))
    await reject_button.callback(as_interaction(fake=denied_reject))
    assert denied_reject.followup.sent[0]["ephemeral"] is True
    assert captured_reject_kwargs == {}

    allowed_reject = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await reject_button.callback(as_interaction(fake=allowed_reject))
    assert captured_reject_kwargs == {"proposal_id": 43, "actor_id": 2}
    assert allowed_reject.edits[0]["view"] is None

    cancel_view = CreditLoanDecisionView(proposal_id=44, lender_id=2, creator_id=1)
    cancel_button = attached_button(view=cancel_view, custom_id="credit:cancel")
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

    The sweep in `tests/guards/test_ack_before_write.py` reads the order off the source. This one observes it, so a defer that is
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
        """Same, for the writes that close a proposal without moving money."""
        custom_id, interaction = clicked[-1]
        acked_at_write[custom_id] = interaction.response.deferred
        return await fake_cancel_loan_proposal(proposal_id=1, actor_id=1)

    monkeypatch.setattr(views, "accept_loan_proposal", accept_and_note)
    monkeypatch.setattr(views, "reject_loan_proposal", resolve_and_note)
    monkeypatch.setattr(views, "cancel_loan_proposal", resolve_and_note)

    central = CentralBankLoanDecisionView(proposal_id=42, creator_id=1, allow_self_approval=True)
    credit = CreditLoanDecisionView(proposal_id=43, lender_id=1, creator_id=1)
    buttons = [
        (central, "central_bank:approve"),
        (central, "central_bank:reject"),
        (central, "central_bank:cancel"),
        (credit, "credit:approve"),
        (credit, "credit:reject"),
        (credit, "credit:cancel"),
    ]

    for view, custom_id in buttons:
        button = attached_button(view=view, custom_id=custom_id)
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
        return fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST).model_copy(
            update={"proposal_id": proposal_id, "status": LoanProposalStatus.REJECTED}
        )

    monkeypatch.setattr(views, "reject_expired_loan_proposal", fake_reject_expired_loan_proposal)
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    credit_message = FakeDiscordMessage()
    credit_view = CreditLoanDecisionView(proposal_id=42, lender_id=2, creator_id=1)
    credit_view.message = as_message(fake=credit_message)
    await credit_view.on_timeout()

    central_message = FakeDiscordMessage()
    central_view = CentralBankLoanDecisionView(proposal_id=43, creator_id=1)
    central_view.message = as_message(fake=central_message)
    await central_view.on_timeout()

    # order-contract: each `on_timeout` is awaited to completion before the next view exists.
    assert rejected == [42, 43]
    # order-contract: same sequential awaits, so cleanup is scheduled in construction order.
    assert scheduled.messages == [credit_message, central_message]
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
        return fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST).model_copy(
            update={"proposal_id": proposal_id, "status": LoanProposalStatus.REJECTED}
        )

    monkeypatch.setattr(views, "reject_expired_loan_proposal", fake_reject_expired_loan_proposal)
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    reports = capture_levels(monkeypatch=monkeypatch, levels=("info", "warn"))
    message = FakeDiscordMessage()
    message.edit_failure = failure
    view = CreditLoanDecisionView(proposal_id=42, lender_id=2, creator_id=1)
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

    async def fake_reject_expired_loan_proposal(proposal_id: int) -> LoanProposalView:
        """Answers the rejection the timeout asks for."""
        return fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST).model_copy(
            update={"proposal_id": proposal_id, "status": LoanProposalStatus.REJECTED}
        )

    monkeypatch.setattr(views, "reject_expired_loan_proposal", fake_reject_expired_loan_proposal)
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    followup = FakeDiscordMessage()
    followup.edit_failure = make_not_found(message="Unknown Webhook")
    view = CreditLoanDecisionView(proposal_id=42, lender_id=2, creator_id=1)
    view.message = as_message(fake=followup)
    panel = FakeDiscordMessage()
    press = FakeInteraction(user=FakeUser(user_id=user_id, name="charlie"), message=panel)
    button = attached_button(view=view, custom_id=custom_id)

    await button.callback(as_interaction(fake=press))
    await view.on_timeout()

    assert [(edit["embed"].title, edit["view"]) for edit in press.edits] == [
        ("信貸申請已逾時", None)
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
    monkeypatch.setattr(views, "cancel_loan_proposal", fake_cancel_loan_proposal)
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    view = CreditLoanDecisionView(proposal_id=42, lender_id=2, creator_id=1)
    view.message = as_message(fake=FakeDiscordMessage())
    panel = FakeDiscordMessage()
    press = FakeInteraction(user=FakeUser(user_id=1, name="alice"), message=panel)
    button = attached_button(view=view, custom_id="credit:cancel")

    await button.callback(as_interaction(fake=press))

    assert press.edits[0]["embed"].title == "信貸申請已取消"
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
    assert presser.edits[0]["embed"].title == "信貸申請已逾時"
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
    assert admin.edits[0]["embed"].title == "🏛️ 央行申請已拒絕"
    assert (
        await approve_as_admin(proposal_id=proposal.proposal_id, actor_id=99, name="banker")
        is None
    )
    assert await get_balance(user_id=1) == 1_000
