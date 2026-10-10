"""One started Blackjack round per human player (`cogs/games/seats.py`)."""

from types import SimpleNamespace
from random import Random
from typing import Any
import asyncio

import pytest
from pydantic import Field, BaseModel

from discordbot.cogs.games import cog as games
from discordbot.cogs.games import blackjack_views
from discordbot.typings.games import GameParticipant, RefreshParticipantsResult
from discordbot.cogs.games.cog import GamesCogs
from discordbot.cogs.games.shoe import BlackjackShoeStore
from discordbot.cogs.games.seats import (
    claim_seat,
    release_seats,
    hand_over_seats,
    seated_elsewhere,
)
from discordbot.cogs.games.blackjack_views import BlackjackView, BlackjackLobbyView

from tests.helpers.games import card, seat, lobby_button, everyone_stays, blackjack_round
from tests.helpers.casting import as_bot, as_message, as_interaction, make_forbidden
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.message_cleanup import record_scheduled_deletes

BOB = seat(user_id=2, display_name="Bob", bet=10, balance_at_start=100)
BOT = seat(user_id=999, display_name="Dealer", bet=10, balance_at_start=100)


class _Holder(BaseModel):
    """A stand-in for a lobby or table elsewhere that seats players."""

    live: bool = Field(default=True, description="Whether the seats it claimed still count.")

    def holds_seats(self) -> bool:
        """Returns whether this holder's seats still count."""
        return self.live


def _lobby(
    extra: list[GameParticipant] | None = None, bot_user_id: int | None = None
) -> BlackjackLobbyView:
    """Builds Alice's lobby, plus `extra`, whose round deals nothing but fives."""

    async def nobody_joins(interaction: object) -> None:
        raise AssertionError(interaction)

    return BlackjackLobbyView(
        owner=seat(bet=10, balance_at_start=100),
        requested_bet=10,
        rng=Random(x=0),  # noqa: S311 -- the scripted shoe decides the deal
        prepare_participant=nobody_joins,
        refresh_participants=everyone_stays,
        bot_user_id=bot_user_id,
        extra_initial_participants=extra,
        shoe_store=BlackjackShoeStore(shoes={7: [card(rank="5") for _ in range(100)]}),
        channel_id=7,
    )


async def _press_start(lobby: BlackjackLobbyView) -> FakeInteraction:
    """Presses 開始 as Alice and returns her interaction."""
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)
    press = FakeInteraction(user=FakeUser(user_id=1), message=message)
    await lobby_button(view=lobby, label="開始").callback(as_interaction(fake=press))
    return press


async def test_a_live_seat_blocks_others_and_a_stale_one_does_not() -> None:
    """A holder's seat counts only while it holds seats, and only against other holders."""
    here, there = _Holder(), _Holder()
    claim_seat(user_id=1, holder=there)

    assert seated_elsewhere(user_id=1, holder=here)
    assert seated_elsewhere(user_id=1)
    assert not seated_elsewhere(user_id=1, holder=there)
    there.live = False
    assert not seated_elsewhere(user_id=1, holder=here)


async def test_seats_move_and_free_only_for_their_own_holder() -> None:
    """Handing over or releasing never touches a seat another holder has."""
    lobby, table, other = _Holder(), _Holder(), _Holder()
    claim_seat(user_id=1, holder=lobby)
    claim_seat(user_id=2, holder=other)

    hand_over_seats(user_ids=[1, 2], from_holder=lobby, to_holder=table)
    release_seats(user_ids=[1, 2], holder=lobby)

    assert seated_elsewhere(user_id=1, holder=other)
    assert not seated_elsewhere(user_id=1, holder=table)
    assert seated_elsewhere(user_id=2, holder=table)
    release_seats(user_ids=[1], holder=table)
    assert not seated_elsewhere(user_id=1)


async def test_a_start_drops_a_player_another_round_seats_and_seats_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bob sits at another started round, so this one starts without him and holds Alice."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    elsewhere = _Holder()
    claim_seat(user_id=2, holder=elsewhere)
    lobby = _lobby(extra=[BOB])

    press = await _press_start(lobby=lobby)

    assert press.followup.sent == [{"content": "已在另一桌, 已移出: Bob", "ephemeral": True}]
    assert [participant.user_id for participant in lobby.participants] == [1]
    assert seated_elsewhere(user_id=1, holder=lobby)
    assert seated_elsewhere(user_id=2, holder=lobby)
    elsewhere.live = False
    assert not seated_elsewhere(user_id=2)


async def test_an_owner_another_round_seats_cannot_start_until_it_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lobby stays open with everyone on it, seats nobody, and starts once that round ends."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    elsewhere = _Holder()
    claim_seat(user_id=1, holder=elsewhere)
    lobby = _lobby(extra=[BOB])

    press = await _press_start(lobby=lobby)

    assert press.followup.sent == [{"content": "你正在另一桌, 打完才能開始", "ephemeral": True}]
    assert lobby._started is False
    assert [participant.user_id for participant in lobby.participants] == [1, 2]
    assert not seated_elsewhere(user_id=2)
    elsewhere.live = False
    await _press_start(lobby=lobby)
    assert lobby._started is True


async def test_an_owner_who_cannot_cover_the_stake_keeps_the_lobby_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A balance refusal drops nobody and holds no seat, so the owner can start again later."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    lobby = _lobby(extra=[BOB])

    async def owner_is_broke(participants: list[GameParticipant]) -> RefreshParticipantsResult:
        return RefreshParticipantsResult(
            participants=[participant for participant in participants if participant.user_id != 1],
            dropped_names=["Alice"],
        )

    lobby.refresh_participants = owner_is_broke

    press = await _press_start(lobby=lobby)

    assert press.followup.sent == [{"content": "你的餘額不足, 不能開始", "ephemeral": True}]
    assert [participant.user_id for participant in lobby.participants] == [1, 2]
    assert not seated_elsewhere(user_id=1)
    assert not seated_elsewhere(user_id=2)


async def test_a_player_dropped_for_balance_gets_their_seat_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The start seats Bob before reading balances, then frees him once the re-check drops him."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    lobby = _lobby(extra=[BOB])

    async def bob_is_broke(participants: list[GameParticipant]) -> RefreshParticipantsResult:
        assert seated_elsewhere(user_id=2)
        return RefreshParticipantsResult(
            participants=[participant for participant in participants if participant.user_id != 2],
            dropped_names=["Bob"],
        )

    lobby.refresh_participants = bob_is_broke

    await _press_start(lobby=lobby)

    assert lobby._started is True
    assert not seated_elsewhere(user_id=2)
    assert seated_elsewhere(user_id=1)


async def test_the_bot_player_is_never_seated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bot joins every table on purpose, so a seat elsewhere never drops it."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    claim_seat(user_id=999, holder=_Holder())
    lobby = _lobby(extra=[BOT], bot_user_id=999)

    press = await _press_start(lobby=lobby)

    assert press.followup.sent == []
    assert [participant.user_id for participant in lobby.participants] == [1, 999]


async def test_a_start_that_fails_gives_every_seat_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lobby that reopens after a failed start holds nobody, so nobody is locked out."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    lobby = _lobby(extra=[BOB])
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)
    start = FakeInteraction(user=FakeUser(user_id=1), message=message)
    start.edit_failure = make_forbidden(message="Missing Access")

    await lobby_button(view=lobby, label="開始").callback(as_interaction(fake=start))

    assert lobby._started is False
    assert not seated_elsewhere(user_id=1)
    assert not seated_elsewhere(user_id=2)


async def test_a_table_frees_its_seats_after_settling_even_when_settling_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seats stay held until settlement has been attempted, and are freed when finalize raises."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="9", suit="♥")]],
        dealer=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
        seats=[seat(bet=10, balance_at_start=100)],
        finished=True,
    )
    view = BlackjackView(round_state=round_state, owner=seat())
    claim_seat(user_id=1, holder=view)
    seen_while_settling: list[bool] = []

    async def refuse_settlement(**_kwargs: object) -> None:
        seen_while_settling.append(seated_elsewhere(user_id=1))
        raise RuntimeError("settlement stubbed out")

    def fail_render(**_kwargs: object) -> None:
        raise RuntimeError("render stubbed out")

    monkeypatch.setattr(blackjack_views, "settle_blackjack_player", refuse_settlement)
    # Raises past the per-seat handler, so only the finalize's own `finally` frees the seats.
    monkeypatch.setattr(blackjack_views, "build_final_embeds", fail_render)

    with pytest.raises(RuntimeError, match="render stubbed out"):
        await view.finalize(message=as_message(fake=FakeDiscordMessage()), interaction=None)

    assert seen_while_settling == [True]
    assert not view.holds_seats()
    assert not seated_elsewhere(user_id=1)


async def test_a_join_from_a_player_another_round_seats_is_refused() -> None:
    """The Join press answers privately and seats nobody."""
    claim_seat(user_id=2, holder=_Holder())
    cog = GamesCogs(bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999))))
    press = FakeInteraction(user=FakeUser(user_id=2))

    participant = await cog._prepare_blackjack_participant(
        interaction=as_interaction(fake=press), wager=10
    )

    assert participant is None
    assert [sent["embed"].title for sent in press.followup.sent] == ["你正在另一桌"]


async def test_opening_a_table_while_seated_elsewhere_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/games blackjack answers privately before deferring and opens no lobby."""
    record_scheduled_deletes(monkeypatch=monkeypatch)

    async def balance(user_id: int) -> int:
        del user_id
        return 100

    monkeypatch.setattr(games, "get_balance", balance)
    claim_seat(user_id=1, holder=_Holder())
    cog = GamesCogs(bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999))))
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await GamesCogs.blackjack.callback(cog, interaction, bet="10")

    assert interaction.response.sent[0]["ephemeral"] is True
    assert interaction.response.sent[0]["embed"].title == "你正在另一桌"
    assert interaction.response.deferred is False
    assert interaction.followup.sent == []


async def test_a_second_start_press_leaves_the_first_ones_seats_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A press that finds the table already starting gives back nothing the dealing press holds."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    lobby = _lobby(extra=[BOB])
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)
    first = FakeInteraction(user=FakeUser(user_id=1), message=message)
    second = FakeInteraction(user=FakeUser(user_id=1), message=message)
    table_edit_reached, release_table_edit = asyncio.Event(), asyncio.Event()
    record_edit = first.edit_original_message

    async def slow_table_edit(**kwargs: Any) -> None:  # noqa: ANN401 -- forwards the edit payload
        table_edit_reached.set()
        await release_table_edit.wait()
        await record_edit(**kwargs)

    monkeypatch.setattr(first, "edit_original_message", slow_table_edit)
    start = lobby_button(view=lobby, label="開始")

    dealing = asyncio.create_task(start.callback(as_interaction(fake=first)))
    await table_edit_reached.wait()
    await start.callback(as_interaction(fake=second))
    release_table_edit.set()
    await dealing

    assert second.followup.sent == [{"content": "這桌已經開始了", "ephemeral": True}]
    assert seated_elsewhere(user_id=1, holder=lobby)
    assert seated_elsewhere(user_id=2, holder=lobby)
