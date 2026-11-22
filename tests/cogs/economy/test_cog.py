"""Tests for the economy cog's slash commands, run against the throwaway ledger.

What these cover is what each command does to the ledger, whether it acknowledges the caller
before that write, and who gets to see the answer.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
import asyncio

import pytest
from sqlalchemy import select

from discordbot.cogs.economy import cog as economy
from discordbot.cogs.economy import views
from discordbot.typings.economy import (
    VIP_PURCHASE_COST,
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
    VipPurchaseResult,
)
from discordbot.cogs.economy.cog import EconomyCogs
from discordbot.cogs.economy.embeds import (
    LEADERBOARD_TITLE,
    LOSS_LEADERBOARD_TITLE,
    build_vip_already_embed,
    build_vip_success_embed,
    build_credit_status_embed,
    build_invalid_amount_embed,
    build_vip_insufficient_embed,
    build_empty_leaderboard_embed,
    build_transfer_insufficient_embed,
    build_empty_loss_leaderboard_embed,
)
from discordbot.utils.message_cleanup import list_pending_public_messages
from discordbot.services.economy.database import (
    CreditResult,
    LoanProposal,
    top_n,
    buy_vip,
    get_vip,
    set_admin,
    get_account,
    get_balance,
    open_session,
    get_credit_ceiling,
    list_loan_contracts,
    accept_loan_proposal,
    get_central_bank_status,
    apply_blackjack_settlement,
    create_personal_loan_request,
    create_central_bank_loan_request,
)

from tests.helpers.casting import as_message, as_interaction
from tests.helpers.economy import (
    LENDING_GUILD,
    seed_balance,
    approve_as_admin,
    seed_participant,
    open_personal_loan,
)
from tests.helpers.discord_mocks import (
    BOT_USER_ID,
    FakeUser,
    FakeGuild,
    FakeInteraction,
    FakeDiscordMessage,
    stub_bot,
)
from tests.helpers.message_cleanup import record_scheduled_deletes

if TYPE_CHECKING:
    from typing import Unpack

    from nextcord.ext import commands

    from tests.helpers.discord_mocks import DiscordPayload


def _bot() -> commands.Bot:
    """A stub bot whose own account is the dealer, id 999."""
    return stub_bot(user=FakeUser(user_id=BOT_USER_ID, display_name="Dealer"))


def _alice_in_the_lending_guild() -> FakeInteraction:
    """A slash command from alice (1), an administrator of `LENDING_GUILD`."""
    return FakeInteraction(
        user=FakeUser(user_id=1, name="alice", display_name="Alice"),
        guild_id=LENDING_GUILD,
        administrator=True,
        slash_command=True,
    )


async def _requested_amounts() -> list[int]:
    """The amount of every loan request the ledger holds, oldest first."""
    async with open_session() as session:
        result = await session.execute(
            statement=select(LoanProposal.amount).order_by(LoanProposal.id)
        )
        return list(result.scalars())


async def _ledger_every_money_command_moves(amount: int) -> None:
    """Seeds a ledger on which every money command can move `amount` in full.

    alice (1) is an economy admin. She and bob (2) both take part in `LENDING_GUILD`, hold
    ten times `amount`, and owe each other and the central bank twice `amount` each.
    """
    for user_id, name in ((1, "alice"), (2, "bob")):
        await seed_participant(user_id=user_id, name=name, amount=10 * amount)
    await set_admin(user_id=1, name="alice", is_admin=True)
    await open_personal_loan(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=2 * amount
    )
    await open_personal_loan(
        borrower_id=2, borrower_name="bob", lender_id=1, lender_name="alice", amount=2 * amount
    )
    for user_id, name in ((1, "alice"), (2, "bob")):
        central = await create_central_bank_loan_request(
            borrower_id=user_id, borrower_name=name, amount=2 * amount
        )
        assert await approve_as_admin(proposal_id=central.proposal_id, actor_id=99, name="banker")


# The four loan settlements. A repayment of zero is refused before the command ever acks, so
# these carry a real amount; a collection of zero means "everything owed" and is the ordinary
# call.
_LOAN_SETTLEMENTS: tuple[tuple[str, dict[str, object]], ...] = (
    ("credit_repay", {"member": FakeUser(user_id=2), "amount": "5"}),
    ("credit_call", {"member": FakeUser(user_id=2), "amount": "0"}),
    ("central_bank_repay", {"amount": "5"}),
    ("central_bank_call", {"member": FakeUser(user_id=2), "amount": "0"}),
)


async def test_a_failed_money_command_stays_private() -> None:
    """Acking before the write must not turn a private failure into a public one.

    Having no matching loan is an ordinary mistake, not an edge case. The failure fills the
    placeholder the defer posted, so the defer's flag decides both whether the channel sees that
    the attempt happened and whether it sees what came of it.

    All four, because the ack is written out once per command and only a sweep catches the one
    someone changes on its own.
    """
    for command, kwargs in _LOAN_SETTLEMENTS:
        cog = EconomyCogs(bot=stub_bot())
        # Administrator, or `central_bank_call` returns from its permission branch and this
        # stops covering the failure path it was written for — while still passing, because
        # that branch defers and follows up ephemerally just like the one under test.
        interaction = _alice_in_the_lending_guild()

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
    await _ledger_every_money_command_moves(amount=100)
    for command, kwargs in _LOAN_SETTLEMENTS:
        interaction = _alice_in_the_lending_guild()

        await getattr(EconomyCogs, command).callback(
            EconomyCogs(bot=stub_bot()), as_interaction(fake=interaction), **kwargs
        )

        assert interaction.response.deferred_ephemeral is True, (
            f"{command} announced the attempt to the channel"
        )
        assert interaction.followup.sent[-1].get("ephemeral") is not True, (
            f"{command} posted its settlement only to the caller"
        )
        assert interaction.original_deleted is True, f"{command} left the caller a second copy"
    assert len(scheduled.messages) == len(_LOAN_SETTLEMENTS)


@pytest.mark.parametrize(
    argnames=("command", "kwargs", "audience"),
    argvalues=[
        ("balance", {"member": None}, "caller"),
        ("leaderboard", {}, "channel"),
        ("loss_leaderboard", {}, "channel"),
        ("casino", {}, "channel"),
        ("pocat", {}, "channel"),
        (
            "admin_refund_tax",
            {"member": FakeUser(user_id=2, name="bob"), "amount": "100"},
            "channel",
        ),
        (
            "admin_collect_tax",
            {"member": FakeUser(user_id=2, name="bob"), "amount": "50"},
            "channel",
        ),
        ("give", {"member": FakeUser(user_id=2, name="bob"), "amount": "100"}, "channel"),
        (
            "credit_borrow",
            {
                "member": FakeUser(user_id=2, name="bob"),
                "amount": "100",
                "monthly_rate_percent": 3.0,
            },
            "panel",
        ),
        ("credit_status", {}, "caller"),
        ("central_bank_borrow", {"amount": "100", "monthly_rate_percent": 3.0}, "panel"),
        ("central_bank_status", {}, "channel"),
        ("vip_command", {}, "caller"),
    ],
)
async def test_each_command_answers_the_audience_its_result_is_for(
    monkeypatch: pytest.MonkeyPatch, command: str, kwargs: dict[str, object], audience: str
) -> None:
    """Personal state reaches only the caller; shared events go to the channel and expire.

    A loan request is a panel the channel answers with its buttons, so it expires only once it
    is decided.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    await _ledger_every_money_command_moves(amount=100)
    interaction = _alice_in_the_lending_guild()

    await getattr(EconomyCogs, command).callback(
        EconomyCogs(bot=_bot()), as_interaction(fake=interaction), **kwargs
    )

    answer = interaction.followup.sent[-1]
    assert (answer.get("ephemeral") is True) is (audience == "caller")
    assert len(scheduled.messages) == (1 if audience == "channel" else 0)
    assert ("view" in answer) is (audience == "panel")
    if audience == "panel":
        view = answer["view"]
        assert isinstance(view, views.LoanDecisionViewBase)
        assert view.message is not None
        assert str(LOAN_PROPOSAL_TIMEOUT_SECONDS) in (answer["embed"].footer.text or "")


async def test_balance_shows_cash_debt_and_net_worth() -> None:
    """`/balance` reports the cash, the debt and the net worth, for the caller or a member."""
    await seed_balance(user_id=1, name="alice", amount=555)
    await seed_balance(user_id=2, name="bob", amount=222)
    request = await create_personal_loan_request(
        borrower_id=1,
        borrower_name="alice",
        lender_id=2,
        lender_name="bob",
        amount=222,
        monthly_rate_bps=0,
    )
    assert await accept_loan_proposal(
        proposal_id=request.proposal_id, actor_id=2, actor_name="bob"
    )
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1), slash_command=True)

    await EconomyCogs.balance.callback(cog, as_interaction(fake=interaction), member=None)

    embed = interaction.followup.sent[0]["embed"]
    field_values = [field.value or "" for field in embed.fields]
    assert any("777" in value for value in field_values)
    assert any("222" in value for value in field_values)
    assert "555" in (embed.description or "")

    inspected_member = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.balance.callback(
        cog,
        as_interaction(fake=inspected_member),
        member=FakeUser(user_id=2, name="bob", display_name="Bob"),
    )
    assert "Bob" in (inspected_member.followup.sent[0]["embed"].description or "")


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
    await seed_participant(user_id=1, name="alice", amount=1_000)
    interaction = _alice_in_the_lending_guild()
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


async def test_economy_admin_rejects_non_admin() -> None:
    """Admin economy commands must check the DB admin flag before mutating balance."""
    interaction = FakeInteraction(user=FakeUser(user_id=1), slash_command=True)

    await EconomyCogs.admin_refund_tax.callback(
        EconomyCogs(bot=_bot()),
        as_interaction(fake=interaction),
        member=FakeUser(user_id=2, name="bob"),
        amount="100",
    )

    assert await get_account(user_id=2) is None
    assert interaction.followup.sent[0].get("ephemeral") is True


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
    await set_admin(user_id=1, name="alice", is_admin=True)
    record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = EconomyCogs(bot=_bot())
    bot_member = FakeUser(user_id=999, name="discordbot", display_name="Dealer", bot=True)

    for command, amount in (
        (EconomyCogs.admin_refund_tax, "100"),
        (EconomyCogs.admin_collect_tax, "50"),
    ):
        interaction = FakeInteraction(user=FakeUser(user_id=1), slash_command=True)
        await command.callback(
            cog, as_interaction(fake=interaction), member=bot_member, amount=amount
        )
        assert interaction.response.deferred is True
        assert interaction.response.deferred_ephemeral is True
        assert interaction.original_deleted is True
        assert interaction.followup.sent[-1].get("ephemeral") is not True

    account = await get_account(user_id=999)
    assert account is not None
    assert (account.name, account.balance) == ("discordbot", 50)


async def test_give_passes_guild_avatar_urls_to_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transfer writes should cache guild avatars instead of only global avatars."""
    await seed_balance(user_id=1, name="alice", amount=1_000)
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
    record_scheduled_deletes(monkeypatch=monkeypatch)

    await EconomyCogs.give.callback(
        EconomyCogs(bot=_bot()), as_interaction(fake=interaction), member=receiver, amount="100"
    )

    stored = {row.user_id: row.avatar_url for row in await top_n(limit=None)}
    assert stored == {
        1: "https://example.test/alice-server.png",
        2: "https://example.test/bob-server.png",
    }
    assert guild.fetch_count == 0


async def test_give_allows_bot_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Players may transfer balance to the bot account."""
    await seed_balance(user_id=1, name="alice", amount=1_000)
    bot_receiver = FakeUser(user_id=999, name="discordbot", display_name="Dealer", bot=True)
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = EconomyCogs(bot=stub_bot(user=bot_receiver))

    await EconomyCogs.give.callback(
        cog, as_interaction(fake=interaction), member=bot_receiver, amount="100"
    )

    assert await get_balance(user_id=1) == 900
    # What the bot nets after the transfer tax is the ledger's business.
    assert await get_balance(user_id=999) > 0


@pytest.mark.parametrize(
    argnames=("receiver_id", "amount", "short"),
    argvalues=[(1, 100, False), (2, 1_000, True)],
    ids=["self-transfer", "insufficient-balance"],
)
async def test_a_refused_transfer_is_a_public_expiring_embed(
    monkeypatch: pytest.MonkeyPatch, receiver_id: int, amount: int, short: bool
) -> None:
    """`/give` defers publicly before it can know the transfer fails, so its refusal is public too.

    The channel already saw the attempt, so the refusal lands there and is cleaned up with it.
    A self-transfer the sender could afford is refused as one, not as a shortfall.
    """
    await seed_balance(user_id=1, name="alice", amount=500)
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), slash_command=True)

    await EconomyCogs.give.callback(
        EconomyCogs(bot=_bot()),
        as_interaction(fake=interaction),
        member=FakeUser(user_id=receiver_id, name="bob"),
        amount=str(amount),
    )

    assert await get_balance(user_id=1) == 500
    shortfall = build_transfer_insufficient_embed(
        sender_name="Alice", sender_avatar_url="", balance_now=500, amount=amount
    )
    assert (interaction.followup.sent[0]["embed"].description == shortfall.description) is short
    assert interaction.response.deferred is True
    assert interaction.response.deferred_ephemeral is False
    assert interaction.followup.sent[0].get("ephemeral") is not True
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
    await seed_balance(user_id=1, name="alice", amount=1_000)
    await seed_balance(user_id=2, name="bob", amount=7_000)
    record_scheduled_deletes(monkeypatch=monkeypatch)
    interaction = FakeInteraction(user=FakeUser(user_id=1), guild_id=321, administrator=True)

    await getattr(EconomyCogs, command).callback(
        EconomyCogs(bot=_bot()), as_interaction(fake=interaction), **kwargs
    )

    status = await get_central_bank_status(guild_id=321)
    assert (status.participant_count, status.total_positive_user_balance) == (1, 1_000)


@pytest.mark.parametrize(
    argnames=("command", "kwargs"),
    argvalues=[
        ("central_bank_borrow", {"amount": "100", "monthly_rate_percent": 0.0}),
        ("central_bank_status", {}),
    ],
    ids=["borrow", "status"],
)
async def test_the_central_bank_refuses_a_direct_message(
    command: str, kwargs: dict[str, object]
) -> None:
    """Outside a guild there is no pool to lend from and nobody who could approve.

    Refused up front rather than at the button, or borrowing in a DM would post a
    request that can be neither approved nor rejected and simply times out.
    """
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), in_guild=False)

    await getattr(EconomyCogs, command).callback(cog, as_interaction(fake=interaction), **kwargs)

    assert interaction.response.sent[0]["ephemeral"] is True
    assert interaction.followup.sent == []


async def test_economy_money_commands_accept_large_string_amounts() -> None:
    """Loan, transfer, collection and tax amounts parse beyond Discord integer option limits.

    Past 2**53, where a float would round the amount, so each command must move it exactly.
    """
    big_amount = 9_007_199_254_740_993
    await _ledger_every_money_command_moves(amount=big_amount)
    cog = EconomyCogs(bot=_bot())
    big_text = "9,007,199,254,740,993"
    bob = FakeUser(user_id=2, name="bob")

    moves = [
        (EconomyCogs.give, {"member": bob}, 1, -big_amount),
        (EconomyCogs.credit_repay, {"member": bob}, 1, -big_amount),
        (EconomyCogs.credit_call, {"member": bob}, 1, big_amount),
        (EconomyCogs.central_bank_repay, {}, 1, -big_amount),
        (EconomyCogs.central_bank_call, {"member": bob}, 2, -big_amount),
        (EconomyCogs.admin_refund_tax, {"member": bob}, 2, big_amount),
        (EconomyCogs.admin_collect_tax, {"member": bob}, 2, -big_amount),
    ]
    for command, kwargs, user_id, delta in moves:
        before = await get_balance(user_id=user_id)
        await command.callback(
            cog, as_interaction(fake=_alice_in_the_lending_guild()), amount=big_text, **kwargs
        )
        assert await get_balance(user_id=user_id) - before == delta, command.name

    await EconomyCogs.credit_borrow.callback(
        cog,
        as_interaction(fake=_alice_in_the_lending_guild()),
        member=bob,
        amount=big_text,
        monthly_rate_percent=3.0,
    )
    await EconomyCogs.central_bank_borrow.callback(
        cog,
        as_interaction(fake=_alice_in_the_lending_guild()),
        amount=big_text,
        monthly_rate_percent=3.0,
    )
    assert (await _requested_amounts())[-2:] == [big_amount, big_amount]

    # A blank or zero collection takes everything bob still owes, to alice and to the bank.
    await EconomyCogs.credit_call.callback(
        cog, as_interaction(fake=_alice_in_the_lending_guild()), member=bob, amount="0"
    )
    await EconomyCogs.central_bank_call.callback(
        cog, as_interaction(fake=_alice_in_the_lending_guild()), member=bob, amount=""
    )
    assert [
        contract for contract in await list_loan_contracts(user_id=2) if contract.borrower_id == 2
    ] == []


async def test_economy_money_commands_reject_invalid_amount_text() -> None:
    """Malformed amount text is rejected before any balance, loan, collection or tax mutation."""
    await _ledger_every_money_command_moves(amount=100)
    ledger_before = (
        await get_account(user_id=1),
        await get_account(user_id=2),
        await _requested_amounts(),
    )
    cog = EconomyCogs(bot=_bot())
    member = FakeUser(user_id=2, name="bob")
    rejections = [
        (EconomyCogs.give, {"member": member}),
        (EconomyCogs.credit_borrow, {"member": member, "monthly_rate_percent": 3.0}),
        (EconomyCogs.credit_repay, {"member": member}),
        (EconomyCogs.credit_call, {"member": member}),
        (EconomyCogs.central_bank_borrow, {"monthly_rate_percent": 3.0}),
        (EconomyCogs.central_bank_repay, {}),
        (EconomyCogs.central_bank_call, {"member": member}),
        (EconomyCogs.admin_refund_tax, {"member": member}),
        (EconomyCogs.admin_collect_tax, {"member": member}),
    ]
    for command, kwargs in rejections:
        interaction = _alice_in_the_lending_guild()
        await command.callback(cog, as_interaction(fake=interaction), amount="x", **kwargs)

        refusal = interaction.response.sent[0]
        assert refusal["ephemeral"] is True, command.name
        assert refusal["embed"].description == build_invalid_amount_embed(title="").description
        assert interaction.followup.sent == [], command.name

    assert (
        await get_account(user_id=1),
        await get_account(user_id=2),
        await _requested_amounts(),
    ) == ledger_before


@pytest.mark.parametrize(
    argnames=("command", "title"),
    argvalues=[("leaderboard", LEADERBOARD_TITLE), ("loss_leaderboard", LOSS_LEADERBOARD_TITLE)],
)
async def test_a_board_goes_to_the_channel_with_its_png(
    monkeypatch: pytest.MonkeyPatch, command: str, title: str
) -> None:
    """Each board is posted for the channel with the PNG its embed shows, and cleaned up with it."""
    for user_id, name, loss in ((1, "alice", 500), (2, "bob", 200)):
        await seed_balance(user_id=user_id, name=name, amount=2 * loss)
        await apply_blackjack_settlement(
            player_id=user_id, player_account_name=name, player_delta=-loss, casino_delta=loss
        )
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    interaction = FakeInteraction(user=FakeUser(user_id=1), slash_command=True)

    await getattr(EconomyCogs, command).callback(
        EconomyCogs(bot=_bot()), as_interaction(fake=interaction)
    )

    sent = interaction.followup.sent[0]
    assert sent["embed"].title == title
    # The PNG shows only if the embed points at the very file sent with it.
    assert sent["embed"].image.url == f"attachment://{sent['files'][0].filename}"
    assert sent.get("ephemeral") is not True
    assert len(scheduled.messages) == 1


async def test_an_empty_board_or_credit_list_answers_with_its_own_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing to rank or list, each command still answers with its titled, coloured panel.

    The boards are public and cleaned up like any other; the credit list stays the caller's.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = EconomyCogs(bot=_bot())
    for command, panel, private in (
        (EconomyCogs.leaderboard, build_empty_leaderboard_embed(), False),
        (EconomyCogs.loss_leaderboard, build_empty_loss_leaderboard_embed(), False),
        (EconomyCogs.credit_status, build_credit_status_embed(contracts=[], viewer_id=1), True),
    ):
        expired_before = len(scheduled.messages)
        interaction = FakeInteraction(user=FakeUser(user_id=1), slash_command=True)

        await command.callback(cog, as_interaction(fake=interaction))

        sent = interaction.followup.sent[0]
        embed = sent["embed"]
        assert (embed.title, embed.description, embed.colour) == (
            panel.title,
            panel.description,
            panel.colour,
        )
        assert (sent.get("ephemeral") is True) is private, command.name
        assert len(scheduled.messages) - expired_before == (0 if private else 1), command.name


async def test_a_request_over_the_ceiling_is_refused_before_anyone_is_asked() -> None:
    """The ceiling is checked at request time too, so no unanswerable panel is posted."""
    await seed_participant(user_id=1, name="alice", amount=1_000)
    assert await get_credit_ceiling(user_id=1) == 2_000

    cog = EconomyCogs(bot=stub_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), guild_id=LENDING_GUILD)
    await EconomyCogs.central_bank_borrow.callback(
        cog, as_interaction(fake=interaction), amount="2001", monthly_rate_percent=0.0
    )

    assert await _requested_amounts() == []
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
    cog = EconomyCogs(bot=stub_bot())
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
        self=EconomyCogs(bot=stub_bot()), interaction=as_interaction(fake=interaction)
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
        self=EconomyCogs(bot=stub_bot()), interaction=as_interaction(fake=interaction)
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
        self=EconomyCogs(bot=stub_bot()), interaction=as_interaction(fake=interaction)
    )

    # Whatever it says, it is none of the answers that would misstate what happened.
    assert interaction.followup.sent[0]["embed"].title not in {
        build_vip_success_embed(
            actor_name="alice",
            avatar_url="",
            result=VipPurchaseResult(new_balance=0, cost=VIP_PURCHASE_COST),
        ).title,
        build_vip_already_embed(actor_name="alice", avatar_url="").title,
        build_vip_insufficient_embed(
            actor_name="alice", avatar_url="", balance_now=VIP_PURCHASE_COST
        ).title,
    }
    assert interaction.followup.sent[0]["embed"].title == "VIP 購買未完成"
    assert await get_vip(user_id=1) is False
    assert await get_balance(user_id=1) == VIP_PURCHASE_COST


async def test_economy_commands_defer_before_avatar_and_admin_lookups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admin adjustments and loan repayments/calls defer before avatar or admin DB lookups."""
    await set_admin(user_id=1, name="alice", is_admin=True)
    await seed_balance(user_id=1, name="alice", amount=1_000)
    await seed_balance(user_id=2, name="bob", amount=1_000)
    record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = EconomyCogs(bot=_bot())
    observed_deferred: list[bool] = []
    real_avatar = economy.guild_avatar_url
    real_get_admin = economy.get_admin

    def install_spies(interaction: FakeInteraction) -> None:
        async def spy_avatar(*args: Any, **kwargs: Any) -> str:  # noqa: ANN401 -- test spy
            observed_deferred.append(interaction.response.deferred)
            return await real_avatar(*args, **kwargs)

        async def spy_get_admin(*args: Any, **kwargs: Any) -> bool:  # noqa: ANN401 -- test spy
            observed_deferred.append(interaction.response.deferred)
            return await real_get_admin(*args, **kwargs)

        monkeypatch.setattr(economy, "guild_avatar_url", spy_avatar)
        monkeypatch.setattr(economy, "get_admin", spy_get_admin)

    target = FakeUser(user_id=2, name="bob", display_name="Bob")
    cases = [
        (EconomyCogs.admin_refund_tax, {"member": target, "amount": "50"}, False),
        (EconomyCogs.credit_repay, {"member": target, "amount": "50"}, False),
        (EconomyCogs.credit_call, {"member": target, "amount": "50"}, False),
        (EconomyCogs.central_bank_repay, {"amount": "50"}, True),
        (EconomyCogs.central_bank_call, {"member": target, "amount": "50"}, True),
    ]
    for command, kwargs, needs_admin in cases:
        observed_deferred.clear()
        interaction = FakeInteraction(
            user=FakeUser(user_id=1, name="alice"),
            guild_id=LENDING_GUILD,
            administrator=needs_admin,
            slash_command=True,
        )
        install_spies(interaction=interaction)
        await command.callback(cog, as_interaction(fake=interaction), **kwargs)
        assert observed_deferred, f"{command.name} never ran its lookup"
        assert all(observed_deferred), f"{command.name} ran a lookup before deferring"


async def test_central_bank_call_refuses_non_admin_without_deferring() -> None:
    """Pre-DB permission refusal on `/central_bank call` replies ephemerally without deferring."""
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(
        user=FakeUser(user_id=1, name="alice"),
        guild_id=LENDING_GUILD,
        administrator=False,
        slash_command=True,
    )

    await EconomyCogs.central_bank_call.callback(
        cog,
        as_interaction(fake=interaction),
        member=FakeUser(user_id=2, name="bob", display_name="Bob"),
        amount="50",
    )

    assert interaction.response.deferred is False
    assert len(interaction.response.sent) == 1
    assert interaction.response.sent[0]["ephemeral"] is True
    assert interaction.followup.sent == []


async def test_balance_uses_portfolio_vip_flag_without_second_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/balance` reads VIP status from `PortfolioView.is_vip` instead of calling `get_vip`."""
    await seed_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST + 500)
    assert await buy_vip(user_id=1, name="alice") is not None

    async def fail_get_vip(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("get_vip should not be called by /balance")

    monkeypatch.setattr(economy, "get_vip", fail_get_vip)
    cog = EconomyCogs(bot=_bot())
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), slash_command=True)

    await EconomyCogs.balance.callback(cog, as_interaction(fake=interaction), member=None)

    assert len(interaction.followup.sent) == 1
    assert "VIP" in str(interaction.followup.sent[0]["embed"].footer.text)


@pytest.fixture
def message_rewards(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every credit the economy message reward asks for, recorded instead of paid."""
    recorded: list[dict[str, Any]] = []

    async def record_reward(**kwargs: Any) -> CreditResult:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Records base reward arguments and returns a fake credit result."""
        recorded.append(kwargs)
        return CreditResult(new_balance=10)

    monkeypatch.setattr(target=economy, name="credit_with_repayment", value=record_reward)
    return recorded


async def test_economy_message_reward_pays_a_member_and_never_the_bot(
    message_rewards: list[dict[str, Any]],
) -> None:
    """A member's message earns the base reward; the bot's own message earns nothing."""
    cog = EconomyCogs(bot=_bot())
    user_message = SimpleNamespace(author=FakeUser(user_id=1, bot=False), guild=None)
    await cog.on_message(message=as_message(fake=user_message))
    assert message_rewards[0]["amount"] == economy.BASE_MESSAGE_REWARD_AMOUNT
    await cog.on_message(message=as_message(fake=SimpleNamespace(author=cog.bot.user, guild=None)))
    assert len(message_rewards) == 1


async def test_economy_message_reward_cooldown_suppresses_rapid_repeat(
    message_rewards: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second message within the cooldown earns nothing; a later one earns again."""
    clock = [1_000.0]
    monkeypatch.setattr(target=economy, name="monotonic", value=lambda: clock[0])
    cog = EconomyCogs(bot=_bot())
    message = SimpleNamespace(author=FakeUser(user_id=1, bot=False), guild=None)

    await cog.on_message(message=as_message(fake=message))
    clock[0] += economy.MESSAGE_REWARD_COOLDOWN_SECONDS - 1
    await cog.on_message(message=as_message(fake=message))
    assert len(message_rewards) == 1

    clock[0] += 2
    await cog.on_message(message=as_message(fake=message))
    assert len(message_rewards) == 2


async def test_economy_message_reward_cooldown_prunes_expired_users(
    message_rewards: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expired per-user cooldown slots are dropped lazily on later messages."""
    monkeypatch.setattr(target=economy, name="monotonic", value=lambda: 1_000.0)
    cooldown = economy.MESSAGE_REWARD_COOLDOWN_SECONDS
    cog = EconomyCogs(bot=_bot())
    cog._message_reward_at = {1: 1_000.0 - cooldown - 1, 2: 1_000.0 - cooldown / 2}

    await cog.on_message(
        message=as_message(fake=SimpleNamespace(author=FakeUser(user_id=3, bot=False), guild=None))
    )

    assert 1 not in cog._message_reward_at
    assert cog._message_reward_at[2] == 1_000.0 - cooldown / 2
    assert cog._message_reward_at[3] == 1_000.0
    assert len(message_rewards) == 1


async def test_economy_message_reward_cooldown_rolls_back_on_credit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed credit must not leave the user on cooldown for the next message."""
    attempts = 0

    async def flaky_reward(**kwargs: Any) -> CreditResult:  # noqa: ANN401 -- command facade double
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient DB failure")
        return CreditResult(new_balance=10)

    monkeypatch.setattr(target=economy, name="credit_with_repayment", value=flaky_reward)
    cog = EconomyCogs(bot=_bot())
    message = SimpleNamespace(author=FakeUser(user_id=1, bot=False), guild=None)

    await cog.on_message(message=as_message(fake=message))
    # The first credit failed, so the slot is rolled back and the next message retries.
    assert 1 not in cog._message_reward_at
    await cog.on_message(message=as_message(fake=message))
    assert attempts == 2
    assert cog._message_reward_at.get(1) is not None


async def test_economy_message_reward_stores_guild_avatar(
    message_rewards: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Base message rewards pass the guild avatar into the economy DB facade."""
    recorded_participation: list[tuple[int, int]] = []

    async def fake_record_guild_participant(guild_id: int, user_id: int) -> None:
        """Records the participation upsert instead of writing to the live economy DB."""
        recorded_participation.append((guild_id, user_id))

    monkeypatch.setattr(economy, "record_guild_participant", fake_record_guild_participant)
    author = FakeUser(user_id=7, avatar_url="https://cdn.test/global.png")
    cached_member = FakeUser(
        user_id=7,
        avatar_url="https://cdn.test/global.png",
        guild_avatar_url="https://cdn.test/server.png",
    )
    guild = FakeGuild(members=[cached_member], cached=True)
    message = SimpleNamespace(author=author, guild=guild)

    await EconomyCogs(bot=_bot()).on_message(message=as_message(fake=message))

    assert [reward["avatar_url"] for reward in message_rewards] == ["https://cdn.test/server.png"]
    # The faucet is the only bulk source of central-bank participation, and it rides the
    # reward rather than every message.
    assert recorded_participation == [(100, 7)]


async def test_economy_message_reward_rollback_does_not_clobber_a_newer_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late failure from an earlier message must not erase a newer message's cooldown slot."""
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    clock = [1_000.0]
    monkeypatch.setattr(target=economy, name="monotonic", value=lambda: clock[0])

    async def racing_reward(**_kwargs: Any) -> CreditResult:  # noqa: ANN401 -- facade double
        if not first_entered.is_set():
            first_entered.set()
            await release_first.wait()
            raise RuntimeError("late failure")
        return CreditResult(new_balance=10)

    monkeypatch.setattr(target=economy, name="credit_with_repayment", value=racing_reward)
    cog = EconomyCogs(bot=_bot())
    message = SimpleNamespace(author=FakeUser(user_id=1, bot=False), guild=None)

    first_task = asyncio.create_task(coro=cog.on_message(message=as_message(fake=message)))
    await first_entered.wait()

    # Cooldown elapses while the first credit is stuck; a newer message succeeds at t=2000.
    clock[0] = 2_000.0
    await cog.on_message(message=as_message(fake=message))
    assert cog._message_reward_at[1] == 2_000.0

    # First credit now fails; it must leave the newer 2000.0 timestamp intact.
    release_first.set()
    await first_task
    assert cog._message_reward_at[1] == 2_000.0


async def test_economy_on_ready_runs_stale_loan_sweep_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`on_ready` spawns the stale-loan sweep once across reconnects."""
    calls: list[object] = []

    async def record_sweep(cutoff: object) -> None:
        calls.append(cutoff)

    monkeypatch.setattr(
        target=economy, name="reject_loan_proposals_created_before", value=record_sweep
    )
    cog = EconomyCogs(bot=_bot())

    await cog.on_ready()
    await cog.on_ready()
    await asyncio.gather(*list(cog._startup_tasks))

    assert calls == [cog._started_at]
