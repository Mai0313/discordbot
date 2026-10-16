"""Tests for the game lobbies: joining, starting, the ante batch, and closing on a timeout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn
import sqlite3
import contextlib

import pytest
import nextcord
from nextcord import Embed, Interaction, HTTPException
from sqlalchemy.exc import OperationalError

from discordbot.typings.economy import (
    JackpotSnapshot,
    JackpotSettlementRequest,
    JackpotSettlementBatchResult,
)
from discordbot.cogs.games.dragon_gate import ANTE, GAME_ID
from discordbot.services.economy.database import (
    top_losers,
    get_jackpot_snapshot,
    apply_jackpot_settlement,
    apply_jackpot_settlement_batch,
)
from discordbot.cogs.games.dragon_gate_views import DragonGateView, DragonGateLobbyView

from tests.helpers.games import (
    card,
    seat,
    joins_as,
    lobby_button,
    everyone_stays,
    scripted_blackjack_lobby,
)
from tests.helpers.casting import (
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
)
from tests.helpers.economy import seed_balance, get_jackpot_pool
from tests.helpers.dragon_gate import (
    RiggedRandom,
    dragon_gate_seat,
    funded_dragon_gate_seat,
    record_jackpot_settlements,
)
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_levels
from tests.helpers.message_cleanup import record_scheduled_deletes
from tests.helpers.economy_invariants import assert_wallet_consistent, assert_daily_casino_stats

if TYPE_CHECKING:
    from collections.abc import Sequence

    from discordbot.typings.games import GameParticipant, RefreshParticipantsResult


@pytest.mark.parametrize(
    argnames=("failure", "level"),
    argvalues=[
        (make_forbidden(message="Missing Access"), "warn"),
        (make_not_found(message="Unknown Message"), "info"),
    ],
    ids=["refused", "message_gone"],
)
async def test_a_blackjack_start_discord_refuses_reopens_the_lobby_and_tells_the_owner(
    monkeypatch: pytest.MonkeyPatch, failure: HTTPException, level: str
) -> None:
    """A lobby whose table edit Discord refuses goes back to taking presses and its timeout.

    The refusal, or a lobby someone deleted: the type and the ids are the whole finding, so it is
    logged without a traceback, and a deleted lobby is a routine outcome rather than a degraded
    one.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    reports = capture_levels(monkeypatch=monkeypatch, levels=("info", "warn"))
    # Nothing but fives: no natural and no insurance, so the deal leaves a table to show.
    lobby = scripted_blackjack_lobby(dealt=[])
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)
    owner_interaction = FakeInteraction(user=FakeUser(user_id=1), message=message)
    owner_interaction.edit_failure = failure
    start_button = lobby_button(view=lobby, label="開始")

    await start_button.callback(as_interaction(fake=owner_interaction))

    assert len(owner_interaction.followup.sent) == 1
    assert owner_interaction.followup.sent[0]["ephemeral"] is True
    assert [(name, fields) for name, _, fields in reports] == [
        (level, {"channel_id": 200, "message_id": 1, "code": failure.code})
    ]
    assert not lobby.is_finished()
    await lobby.on_timeout()
    assert scheduled.messages == [message]


@pytest.mark.parametrize(
    argnames=("failure", "level", "traceback"),
    argvalues=[
        (make_forbidden(message="Missing Access"), "warn", False),
        (make_not_found(message="Unknown Message"), "info", False),
        (make_server_error(), "warn", True),
    ],
    ids=["refused", "message_gone", "broke"],
)
async def test_a_lobby_timeout_edit_that_fails_is_reported_and_still_cleaned_up(
    monkeypatch: pytest.MonkeyPatch, failure: HTTPException, level: str, traceback: bool
) -> None:
    """Nothing awaits a timeout, so its failure is logged here at the level its cause earns."""
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    reports = capture_levels(monkeypatch=monkeypatch, levels=("info", "warn"))
    lobby = scripted_blackjack_lobby(dealt=[])
    message = FakeDiscordMessage()
    message.edit_failure = failure
    lobby.message = as_message(fake=message)

    await lobby.on_timeout()

    assert [(name, "_exc_info" in fields) for name, _, fields in reports] == [(level, traceback)]
    assert scheduled.messages == [message]


@pytest.mark.parametrize(argnames="expired", argvalues=[False, True], ids=["live", "expired"])
async def test_a_lobby_someone_joined_times_out_through_the_join_in_a_shut_out_channel(
    monkeypatch: pytest.MonkeyPatch, expired: bool
) -> None:
    """The join rebound the lobby to the message it pressed, which only the join's token reaches.

    So the timeout closes it and schedules its delete through that token while it lives.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    lobby = scripted_blackjack_lobby(dealt=[])
    lobby.prepare_participant = joins_as(
        participant=seat(user_id=2, display_name="Bob", bet=10, balance_at_start=100)
    )
    message = FakeDiscordMessage()
    message.edit_failure = make_forbidden(message="Missing Access")
    lobby.message = as_message(fake=message)
    join = FakeInteraction(
        user=FakeUser(user_id=2, name="bob", display_name="Bob"), message=message
    )
    join_button = lobby_button(view=lobby, label="加入")
    await join_button.callback(as_interaction(fake=join))
    join.expired = expired

    await lobby.on_timeout()

    # The join's own refresh, then the timeout's edit while the token lives.
    assert len(join.edits) == (1 if expired else 2)
    if not expired:
        joined, closed = (edit["embed"].description for edit in join.edits)
        assert closed, "the timeout's edit says the lobby closed"
        assert closed != joined
    assert scheduled.interactions == [join]


@pytest.mark.parametrize(argnames="presser", argvalues=["owner", "broke"])
async def test_a_lobby_kept_open_by_refused_presses_closes_through_the_newest_one(
    monkeypatch: pytest.MonkeyPatch, presser: str
) -> None:
    """Every press restarts a lobby's timer, a refused one included, so the lobby closes through it.

    Nobody joined, so the only other way to the lobby is the slash command's own token, which
    the presses can have kept the lobby open past.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    async def turned_away(interaction: Interaction[Any]) -> GameParticipant | None:
        """Answers a press the way the cog does for a balance that cannot cover the bet."""
        del interaction

    lobby = scripted_blackjack_lobby(dealt=[])
    lobby.prepare_participant = turned_away
    followup = FakeDiscordMessage()
    followup.edit_failure = make_not_found(message="Unknown Webhook")
    lobby.message = as_message(fake=followup)
    message = FakeDiscordMessage()
    user = FakeUser(user_id=1) if presser == "owner" else FakeUser(user_id=2, name="bob")
    press = FakeInteraction(user=user, message=message)

    await lobby_button(view=lobby, label="加入").callback(as_interaction(fake=press))
    await lobby.on_timeout()

    assert len(press.edits) == 1, "the refused press edited nothing; the timeout's edit did"
    assert press.edits[0]["embed"].description, "the timeout's edit says the lobby closed"
    assert (scheduled.messages, scheduled.interactions) == ([message], [press])
    assert [sent["ephemeral"] for sent in press.followup.sent] == (
        [True] if presser == "owner" else []
    ), "a refusal is a followup, so the press's own token stays on the lobby"


@pytest.mark.parametrize(argnames="label", argvalues=["加入", "開始"], ids=["join", "start"])
async def test_a_lobby_press_that_fails_after_its_acknowledgement_still_closes_the_lobby(
    monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    """The press restarted the lobby's timer before its callback failed, so it is kept too."""
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    async def wallet_unreadable(interaction: Interaction[Any]) -> GameParticipant | None:
        """Fails the join's balance read."""
        raise RuntimeError(interaction)

    async def balances_unreadable(
        participants: list[GameParticipant],
    ) -> RefreshParticipantsResult:
        """Fails the start's balance re-check."""
        raise RuntimeError(participants)

    lobby = scripted_blackjack_lobby(dealt=[])
    lobby.prepare_participant = wallet_unreadable
    lobby.refresh_participants = balances_unreadable
    followup = FakeDiscordMessage()
    followup.edit_failure = make_not_found(message="Unknown Webhook")
    lobby.message = as_message(fake=followup)
    message = FakeDiscordMessage()
    user = FakeUser(user_id=2, name="bob") if label == "加入" else FakeUser(user_id=1)
    press = FakeInteraction(user=user, message=message)

    with pytest.raises(RuntimeError):
        await lobby_button(view=lobby, label=label).callback(as_interaction(fake=press))
    await lobby.on_timeout()

    assert len(press.edits) == 1, "the failed press edited nothing; the timeout's edit did"
    assert press.edits[0]["embed"].description, "the timeout's edit says the lobby closed"
    assert (scheduled.messages, scheduled.interactions) == ([message], [press])


async def test_a_refusal_after_the_blackjack_table_is_up_is_not_a_failed_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bot's first move edits the table right after it lands; that refusal is the table's.

    Reported as a failed start, the owner would be told no table opened while one is up.
    """

    class _RefusedAfterTheTable(FakeInteraction):
        async def edit_original_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- Discord kwargs
            await super().edit_original_message(**kwargs)
            self.edit_failure = make_forbidden(message="Missing Access")

    reports = capture_levels(monkeypatch=monkeypatch, levels=("info", "warn"))
    bot = seat(user_id=999, display_name="Dealer", bet=10, balance_at_start=100)
    # Alice 5 5, the bot 5 5, the dealer's hole 5 and an ace up: the bot owes an insurance call.
    lobby = scripted_blackjack_lobby(dealt=[card(rank="5")] * 5 + [card(rank="A")], bot=bot)
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)
    owner_interaction = _RefusedAfterTheTable(user=FakeUser(user_id=1), message=message)
    start_button = lobby_button(view=lobby, label="開始")

    with pytest.raises(nextcord.Forbidden):
        await start_button.callback(as_interaction(fake=owner_interaction))

    assert len(message.edits) == 1, "the table landed before the refusal"
    assert owner_interaction.followup.sent == []
    assert reports == []


async def test_dragon_gate_lobby_join_leave_and_owner_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lobby buttons mutate participants, and the owner's start charges the ante."""
    owner = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    bob = dragon_gate_seat(user_id=2, display_name="Bob")
    message = FakeDiscordMessage()
    settlements = record_jackpot_settlements(monkeypatch=monkeypatch)
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        prepare_participant=joins_as(participant=bob),
        refresh_participants=everyone_stays,
        initial_jackpot=pool_before,
    )
    view.message = as_message(fake=message)

    join_button = lobby_button(view=view, label="加入")
    await join_button.callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )
    assert view.participants == [owner, bob]
    join_embed = message.edits[-1]["embed"]
    assert isinstance(join_embed, Embed)
    assert isinstance(join_embed.description, str)

    leave_button = lobby_button(view=view, label="離開")
    await leave_button.callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )
    assert view.participants == [owner]

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1), message=message)
    await lobby_button(view=view, label="開始").callback(as_interaction(fake=owner_interaction))
    assert isinstance(message.edits[-1]["view"], DragonGateView)
    assert settlements == [
        JackpotSettlementRequest(
            player_id=owner.user_id,
            player_account_name=owner.account_name,
            player_avatar_url=owner.avatar_url,
            player_delta=-ANTE,
            require_full_debit=True,
        )
    ]
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before + ANTE


async def test_dragon_gate_lobby_ante_rejection_keeps_lobby_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the ledger rejects a non-owner's ante, nobody is charged and the lobby stays startable.

    Bob's wallet is empty, so the full debit the antes require refuses him.
    """
    owner = await funded_dragon_gate_seat(user_id=1, display_name="Alice")
    bob = dragon_gate_seat(user_id=2, display_name="Bob")
    message = FakeDiscordMessage()
    record_scheduled_deletes(monkeypatch=monkeypatch)
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        prepare_participant=joins_as(participant=bob),
        refresh_participants=everyone_stays,
        initial_jackpot=pool_before,
    )
    view.message = as_message(fake=message)

    join_button = lobby_button(view=view, label="加入")
    await join_button.callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )
    start_button = lobby_button(view=view, label="開始")
    await start_button.callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=1), message=message))
    )

    assert view.participants == [owner]
    assert view._started is False
    assert isinstance(message.edits[-1]["view"], DragonGateLobbyView)
    embed = message.edits[-1]["embed"]
    assert isinstance(embed, Embed)
    assert "Bob" in str(embed.description), "the lobby says who it dropped"
    await assert_wallet_consistent(user_id=1, expected_balance=owner.balance_at_start)
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before


@pytest.mark.parametrize(
    argnames="failure",
    argvalues=[
        make_forbidden(message="Missing Access"),
        make_not_found(message="Unknown Message"),
        make_server_error(),
    ],
    ids=["refused", "message_gone", "discord_failing"],
)
async def test_a_start_whose_table_never_lands_returns_the_antes_and_reopens_the_lobby(
    monkeypatch: pytest.MonkeyPatch, failure: HTTPException
) -> None:
    """The antes are charged before the table edit, so a table that never appears hands them back.

    Left marked started, the lobby would refuse every press and skip its own timeout cleanup.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    batches: list[list[JackpotSettlementRequest]] = []

    async def recording_batch(
        game_id: str, settlements: Sequence[JackpotSettlementRequest]
    ) -> JackpotSettlementBatchResult:
        """Records each batch and applies it to the isolated ledger."""
        batches.append(list(settlements))
        return await apply_jackpot_settlement_batch(game_id=game_id, settlements=settlements)

    monkeypatch.setattr(
        "discordbot.cogs.games.lobby.apply_jackpot_settlement_batch", recording_batch
    )
    owner = dragon_gate_seat(user_id=1, display_name="Alice")
    bob = dragon_gate_seat(user_id=2, display_name="Bob")
    for participant in (owner, bob):
        await seed_balance(
            user_id=participant.user_id, name=participant.account_name, amount=1_000
        )
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    message = FakeDiscordMessage()
    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        prepare_participant=joins_as(participant=bob),
        refresh_participants=everyone_stays,
        initial_jackpot=pool_before,
    )
    view.message = as_message(fake=message)
    join_button = lobby_button(view=view, label="加入")
    await join_button.callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )

    owner_start = FakeInteraction(user=FakeUser(user_id=1), message=message)
    owner_start.edit_failure = failure
    start_button = lobby_button(view=view, label="開始")
    # Only a refusal is answered in place; any other failure still reaches the view's on_error.
    with contextlib.suppress(HTTPException):
        await start_button.callback(as_interaction(fake=owner_start))

    await assert_wallet_consistent(user_id=1, expected_balance=1_000)
    await assert_wallet_consistent(user_id=2, expected_balance=1_000)
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before
    for user_id in (1, 2):
        await assert_daily_casino_stats(user_id=user_id, loss=0, win=0, net=0)
    assert await top_losers() == []
    assert len(batches) == 2, "the antes go back in one transaction, as they were charged"
    # order-contract: the refund is awaited after the ante batch it reverses.
    assert {request.player_id: request.player_delta for request in batches[-1]} == {
        1: ANTE,
        2: ANTE,
    }
    assert not view.is_finished()
    await view.on_timeout()
    assert scheduled.messages == [message]


async def test_a_start_whose_table_never_lands_returns_every_whole_ante_from_a_drained_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another table's win can leave the shared pool short of one ante before the refund lands."""
    record_scheduled_deletes(monkeypatch=monkeypatch)

    async def win_elsewhere_before_the_refund(
        game_id: str, settlements: Sequence[JackpotSettlementRequest]
    ) -> JackpotSettlementBatchResult:
        """Leaves the pool holding 5 just before the antes go back."""
        if settlements[0].player_delta > 0:
            pool = await get_jackpot_pool(game_id=game_id)
            await apply_jackpot_settlement(
                player_id=3, player_account_name="carol", player_delta=pool - 5, game_id=game_id
            )
        return await apply_jackpot_settlement_batch(game_id=game_id, settlements=settlements)

    monkeypatch.setattr(
        "discordbot.cogs.games.lobby.apply_jackpot_settlement_batch",
        win_elsewhere_before_the_refund,
    )
    owner = await funded_dragon_gate_seat(user_id=1, display_name="Alice", balance=1_000)
    bob = await funded_dragon_gate_seat(user_id=2, display_name="Bob", balance=1_000)
    seed = await get_jackpot_snapshot(game_id=GAME_ID)

    message = FakeDiscordMessage()
    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        prepare_participant=joins_as(participant=bob),
        refresh_participants=everyone_stays,
        initial_jackpot=seed.balance,
    )
    view.message = as_message(fake=message)
    await lobby_button(view=view, label="加入").callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )
    owner_start = FakeInteraction(user=FakeUser(user_id=1), message=message)
    owner_start.edit_failure = make_server_error()
    with pytest.raises(HTTPException):
        await lobby_button(view=view, label="開始").callback(as_interaction(fake=owner_start))

    await assert_wallet_consistent(user_id=1, expected_balance=1_000)
    await assert_wallet_consistent(user_id=2, expected_balance=1_000)
    # The first refund overdraws the 5 left, so the pool is topped back up before the second.
    assert await get_jackpot_snapshot(game_id=GAME_ID) == JackpotSnapshot(
        balance=seed.balance - ANTE, generation=seed.generation + 1
    )


@pytest.mark.parametrize(
    argnames=("failing_step", "error"),
    argvalues=[
        (
            "discordbot.cogs.games.lobby.apply_jackpot_settlement_batch",
            OperationalError("ante", None, sqlite3.OperationalError("database is locked")),
        ),
        (
            "discordbot.cogs.games.dragon_gate_views.DragonGateView.in_progress_embeds",
            ValueError("table"),
        ),
    ],
    ids=["ante_charge_fails", "table_build_fails"],
)
async def test_a_start_that_raises_before_its_table_is_up_charges_nothing_and_reopens_the_lobby(
    monkeypatch: pytest.MonkeyPatch, failing_step: str, error: Exception
) -> None:
    """Whichever step of the start raises, the antes end where they were and the lobby reopens.

    Left marked started, the lobby would refuse every press and skip its own timeout cleanup.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    def failing(*_args: object, **_kwargs: object) -> NoReturn:
        raise error

    monkeypatch.setattr(failing_step, failing)
    owner = await funded_dragon_gate_seat(user_id=1, display_name="Alice", balance=1_000)
    bob = await funded_dragon_gate_seat(user_id=2, display_name="Bob", balance=1_000)
    pool_before = await get_jackpot_pool(game_id=GAME_ID)

    message = FakeDiscordMessage()
    view = DragonGateLobbyView(
        owner=owner,
        rng=RiggedRandom(choices=("3", "♠", "9", "♥")),
        prepare_participant=joins_as(participant=bob),
        refresh_participants=everyone_stays,
        initial_jackpot=pool_before,
    )
    view.message = as_message(fake=message)
    await lobby_button(view=view, label="加入").callback(
        as_interaction(fake=FakeInteraction(user=FakeUser(user_id=2), message=message))
    )
    # Called directly, the press skips the view's on_error, so the raise reaches the test.
    with pytest.raises(type(error)):
        await lobby_button(view=view, label="開始").callback(
            as_interaction(fake=FakeInteraction(user=FakeUser(user_id=1), message=message))
        )

    await assert_wallet_consistent(user_id=1, expected_balance=1_000)
    await assert_wallet_consistent(user_id=2, expected_balance=1_000)
    assert await get_jackpot_pool(game_id=GAME_ID) == pool_before
    assert not view.is_finished()
    await view.on_timeout()
    assert scheduled.messages == [message]
