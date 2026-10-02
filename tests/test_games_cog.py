"""Tests for the `/games` cog: its commands, how it seats players, and its startup cleanup."""

from types import SimpleNamespace
from random import Random
from typing import Any, NoReturn, cast
from pathlib import Path
import contextlib

import pytest
import nextcord
from nextcord import Embed, Interaction, HTTPException

from discordbot.utils import message_cleanup as cleanup_module
from discordbot.cogs.games import cog as games
from discordbot.cogs.games import blackjack_views
from discordbot.typings.games import GameParticipant, RefreshParticipantsResult
from discordbot.cogs.games.cog import GamesCogs
from discordbot.cogs.games.shoe import BlackjackShoeStore
from discordbot.typings.economy import MAX_SINGLE_BET, JackpotSnapshot
from discordbot.cogs.games.lobby import BaseGameLobbyView
from discordbot.cogs.games.blackjack import Card
from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME, embed_spacer_url
from discordbot.utils.message_cleanup import PendingPublicMessage, list_pending_public_messages
from discordbot.cogs.games.blackjack_views import BlackjackView, BlackjackLobbyView
from discordbot.cogs.games.dragon_gate_views import DragonGateLobbyView

from tests.helpers.games import card, seat, joins_as, lobby_button, everyone_stays, attached_button
from tests.helpers.casting import (
    as_bot,
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
)
from tests.helpers.economy import seed_balance
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_levels
from tests.helpers.message_cleanup import record_scheduled_deletes


def _cog() -> GamesCogs:
    """Builds the cog around a bot whose own account is user 999, the table's bot player."""
    return GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )


async def fake_game_balance(user_id: int) -> int:
    """Returns a small fake game balance for anyone but the bot, whose empty wallet keeps it out."""
    return 0 if user_id == 999 else 100


async def _empty_game_balance(user_id: int) -> int:
    """Returns no spendable game balance."""
    return 0


async def _wealthy_game_balance(user_id: int) -> int:
    """Returns a fake balance large enough for the Dragon Gate ante."""
    del user_id
    return 1_000_000


async def fake_dragon_gate_jackpot_snapshot(game_id: str) -> JackpotSnapshot:
    """Returns a stable fake Dragon Gate jackpot snapshot."""
    del game_id
    return JackpotSnapshot(balance=100_000)


async def test_bot_blackjack_participant_spreads_bet_by_true_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bot's Kelly wager rises with a favorable channel true count."""
    cog = _cog()

    async def fake_get_balance(user_id: int) -> int:
        return 1_000_000

    async def fake_avatar(user: object, guild: object = None) -> str:
        return ""

    monkeypatch.setattr(games, "get_balance", fake_get_balance)
    monkeypatch.setattr(games, "guild_avatar_url", fake_avatar)

    neutral = await cog._bot_blackjack_participant(guild=None, table_bet=100, channel_id=1)
    # A ten-rich stored shoe above the reshuffle threshold gives channel 2 a strongly
    # positive true count.
    cog._blackjack_shoes.save_shoe(channel_id=2, cards=[card(rank="10") for _ in range(120)])
    favorable = await cog._bot_blackjack_participant(guild=None, table_bet=100, channel_id=2)

    assert neutral is not None
    assert favorable is not None
    assert favorable.bet > neutral.bet


def test_games_commands_are_grouped_under_games() -> None:
    """Verifies casino games are registered as /games subcommands."""
    assert GamesCogs.games.name == "games"
    assert GamesCogs.games.name_localizations[nextcord.Locale.zh_TW] == "小遊戲"
    assert set(GamesCogs.games.children) == {"blackjack", "blackjack_history", "dragon_gate"}
    assert GamesCogs.blackjack.name == "blackjack"
    assert GamesCogs.blackjack.name_localizations[nextcord.Locale.zh_TW] == "二十一點"
    assert GamesCogs.blackjack_history.name == "blackjack_history"
    assert GamesCogs.blackjack_history.name_localizations[nextcord.Locale.zh_TW] == "二十一點紀錄"
    assert GamesCogs.dragon_gate.name == "dragon_gate"
    assert GamesCogs.dragon_gate.name_localizations[nextcord.Locale.zh_TW] == "射龍門"


async def test_blackjack_history_missing_user_sends_notice() -> None:
    """A missing interaction user gets feedback instead of an empty deferred response."""
    cog = _cog()
    interaction = FakeInteraction()
    cast("Any", interaction).user = None

    await GamesCogs.blackjack_history.callback(cog, interaction, member=None, count=10)

    assert interaction.response.deferred is False
    assert interaction.response.sent[0]["ephemeral"] is True
    content = interaction.response.sent[0]["content"]
    assert isinstance(content, str)
    assert "無法辨識使用者" in content
    assert interaction.followup.sent == []


def test_every_test_gets_its_own_cleanup_store(tmp_path: Path) -> None:
    """A test that asks for no isolation still cannot reach the deployed `games.db`.

    The lobby tests below record their public messages for deletion without patching the
    store, so this requests nothing but `tmp_path` and fails if the autouse swap is dropped.
    It lives here rather than beside the store's own tests so that a module-local swap there
    cannot satisfy it.
    """
    assert cleanup_module._engine.url.database == str(tmp_path / "game_cleanup.db")


async def test_games_commands_open_their_lobbies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each game command opens its lobby as a public followup carrying the width spacer."""
    monkeypatch.setattr(games, "get_balance", fake_game_balance)

    cog = _cog()

    blackjack_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, blackjack_interaction, bet="10")
    assert blackjack_interaction.followup.sent[0]["wait"] is True
    assert isinstance(blackjack_interaction.followup.sent[0]["view"], BlackjackLobbyView)
    assert (
        blackjack_interaction.followup.sent[0]["files"][0].filename
        == DEFAULT_EMBED_SPACER_FILENAME
    )
    assert blackjack_interaction.followup.sent[0]["embed"].image.url == embed_spacer_url()

    monkeypatch.setattr(games, "get_jackpot_snapshot", fake_dragon_gate_jackpot_snapshot)
    monkeypatch.setattr(games, "get_balance", _wealthy_game_balance)
    dragon_gate_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, dragon_gate_interaction)
    assert dragon_gate_interaction.followup.sent[-1]["wait"] is True
    assert isinstance(dragon_gate_interaction.followup.sent[-1]["view"], DragonGateLobbyView)
    assert (
        dragon_gate_interaction.followup.sent[-1]["files"][0].filename
        == DEFAULT_EMBED_SPACER_FILENAME
    )
    assert dragon_gate_interaction.followup.sent[-1]["embed"].image.url == embed_spacer_url()


@pytest.mark.parametrize(argnames="game", argvalues=["blackjack", "dragon_gate"])
async def test_an_opened_lobby_is_recorded_for_the_restart_sweep(
    monkeypatch: pytest.MonkeyPatch, game: str
) -> None:
    """A lobby the bot restarts under is deleted by the next start's sweep, from this record."""
    monkeypatch.setattr(games, "get_jackpot_snapshot", fake_dragon_gate_jackpot_snapshot)
    cog = _cog()
    interaction = FakeInteraction(user=FakeUser(user_id=1))
    if game == "blackjack":
        monkeypatch.setattr(games, "get_balance", fake_game_balance)
        await GamesCogs.blackjack.callback(cog, interaction, bet="10")
    else:
        monkeypatch.setattr(games, "get_balance", _wealthy_game_balance)
        await GamesCogs.dragon_gate.callback(cog, interaction)

    lobby = interaction.followup.sent[-1]["view"]
    assert isinstance(lobby, BaseGameLobbyView)
    assert lobby.message is not None
    assert await list_pending_public_messages() == [
        PendingPublicMessage(
            channel_id=lobby.message.channel.id, message_id=lobby.message.id, user_name="alice"
        )
    ]


async def test_a_lobby_start_is_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anyone but the owner pressing 開始 is told so privately, and the lobby stays open."""
    monkeypatch.setattr(games, "get_balance", fake_game_balance)

    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="10")
    lobby_view = owner_interaction.followup.sent[0]["view"]
    assert isinstance(lobby_view, BlackjackLobbyView)

    start_button = lobby_button(view=lobby_view, label="開始")
    other_interaction = FakeInteraction(user=FakeUser(user_id=2, name="bob", display_name="Bob"))
    await start_button.callback(as_interaction(fake=other_interaction))

    assert other_interaction.followup.sent == [{"content": "只有房主可以開始", "ephemeral": True}]
    assert lobby_view._started is False


async def _nobody_joins(interaction: Interaction[Any]) -> GameParticipant | None:
    """Lobby join hook for a table nobody joins."""
    raise AssertionError(interaction)


def _scripted_blackjack_lobby(
    dealt: list[Card], bot: GameParticipant | None = None
) -> BlackjackLobbyView:
    """Builds Alice's Blackjack lobby whose round deals `dealt` first, then fives."""
    shoe = dealt + [card(rank="5") for _ in range(100)]
    return BlackjackLobbyView(
        owner=seat(bet=10, balance_at_start=100),
        requested_bet=10,
        rng=Random(x=0),  # noqa: S311 -- the scripted shoe decides the deal
        prepare_participant=_nobody_joins,
        refresh_participants=everyone_stays,
        bot_user_id=None if bot is None else bot.user_id,
        extra_initial_participants=None if bot is None else [bot],
        shoe_store=BlackjackShoeStore(shoes={7: shoe}),
        channel_id=7,
    )


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
    lobby = _scripted_blackjack_lobby(dealt=[])
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
    argnames="failure",
    argvalues=[
        make_forbidden(message="Missing Access"),
        make_not_found(message="Unknown Message"),
        make_server_error(),
    ],
    ids=["refused", "message_gone", "discord_failing"],
)
async def test_a_blackjack_start_whose_table_never_lands_keeps_the_channel_shoe(
    monkeypatch: pytest.MonkeyPatch, failure: HTTPException
) -> None:
    """No card of the failed deal was shown, so the next start deals from the shoe the bot counted."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    lobby = _scripted_blackjack_lobby(dealt=[])
    store = lobby._shoe_store
    assert store is not None
    before = list(store.shoes[7])
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)
    owner_interaction = FakeInteraction(user=FakeUser(user_id=1), message=message)
    owner_interaction.edit_failure = failure

    # Only a refusal is answered in place; any other failure still reaches the view's on_error.
    with contextlib.suppress(HTTPException):
        await lobby_button(view=lobby, label="開始").callback(
            as_interaction(fake=owner_interaction)
        )

    assert store.shoes.get(7) == before


@pytest.mark.parametrize(
    argnames="failing_step", argvalues=["build_in_progress_embeds", "table_edit_kwargs"]
)
async def test_a_blackjack_start_that_raises_before_its_table_is_up_reopens_and_keeps_the_shoe(
    monkeypatch: pytest.MonkeyPatch, failing_step: str
) -> None:
    """A start whose table is never sent reopens the lobby, whichever step raised.

    Left marked started, the lobby would refuse every press and skip its own timeout cleanup.
    """
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    def failing(**_kwargs: object) -> NoReturn:
        raise ValueError(failing_step)

    monkeypatch.setattr(target=blackjack_views, name=failing_step, value=failing)
    lobby = _scripted_blackjack_lobby(dealt=[])
    store = lobby._shoe_store
    assert store is not None
    before = list(store.shoes[7])
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)

    # Called directly, the press skips the view's on_error, so the raise reaches the test.
    with pytest.raises(ValueError, match=failing_step):
        await lobby_button(view=lobby, label="開始").callback(
            as_interaction(fake=FakeInteraction(user=FakeUser(user_id=1), message=message))
        )

    assert store.shoes.get(7) == before
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
    lobby = _scripted_blackjack_lobby(dealt=[])
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
    lobby = _scripted_blackjack_lobby(dealt=[])
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

    embed = join.edits[-1]["embed"]
    assert (embed.description == "Lobby 已逾時") is not expired
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

    lobby = _scripted_blackjack_lobby(dealt=[])
    lobby.prepare_participant = turned_away
    followup = FakeDiscordMessage()
    followup.edit_failure = make_not_found(message="Unknown Webhook")
    lobby.message = as_message(fake=followup)
    message = FakeDiscordMessage()
    user = FakeUser(user_id=1) if presser == "owner" else FakeUser(user_id=2, name="bob")
    press = FakeInteraction(user=user, message=message)

    await lobby_button(view=lobby, label="加入").callback(as_interaction(fake=press))
    await lobby.on_timeout()

    assert [edit["embed"].description for edit in press.edits] == ["Lobby 已逾時"]
    assert (scheduled.messages, scheduled.interactions) == ([message], [press])
    assert press.followup.sent == (
        [{"content": "你已經在這桌了", "ephemeral": True}] if presser == "owner" else []
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

    lobby = _scripted_blackjack_lobby(dealt=[])
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

    assert [edit["embed"].description for edit in press.edits] == ["Lobby 已逾時"]
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
    lobby = _scripted_blackjack_lobby(dealt=[card(rank="5")] * 5 + [card(rank="A")], bot=bot)
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)
    owner_interaction = _RefusedAfterTheTable(user=FakeUser(user_id=1), message=message)
    start_button = lobby_button(view=lobby, label="開始")

    with pytest.raises(nextcord.Forbidden):
        await start_button.callback(as_interaction(fake=owner_interaction))

    assert len(message.edits) == 1, "the table landed before the refusal"
    assert owner_interaction.followup.sent == []
    assert reports == []


async def _start_in_a_shut_out_channel(
    monkeypatch: pytest.MonkeyPatch, dealt: list[Card], bot: GameParticipant | None = None
) -> FakeDiscordMessage:
    """Presses 開始 on Alice's scripted lobby in a channel that refuses every edit.

    The lobby went up on the slash command's token, so it shows in a channel the server shut the
    bot out of afterwards. Returns the lobby message, whose `edits` hold only what landed.
    """
    record_scheduled_deletes(monkeypatch=monkeypatch)
    monkeypatch.setattr(blackjack_views, "PEEK_REVEAL_DELAY_SECONDS", 0)
    monkeypatch.setattr(blackjack_views, "BOT_TURN_EDIT_DELAY_SECONDS", 0)
    await seed_balance(user_id=1, name="alice", amount=100)
    if bot is not None:
        await seed_balance(user_id=bot.user_id, name=bot.account_name, amount=100)
    lobby = _scripted_blackjack_lobby(dealt=dealt, bot=bot)
    message = FakeDiscordMessage()
    message.edit_failure = make_forbidden(message="Missing Access")
    lobby.message = as_message(fake=message)
    owner_start = FakeInteraction(user=FakeUser(user_id=1), message=message)
    start_button = lobby_button(view=lobby, label="開始")

    await start_button.callback(as_interaction(fake=owner_start))

    assert owner_start.followup.sent == []
    return message


def _alice_press(message: FakeDiscordMessage) -> FakeInteraction:
    """Builds one press by Alice on the table message."""
    return FakeInteraction(user=FakeUser(user_id=1), message=message)


async def test_a_blackjack_lobby_in_a_channel_the_bot_was_shut_out_of_still_deals_and_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The start and each table press edit through their own token, which no channel can refuse.

    The settling press's token also carries the table's scheduled delete.
    """
    # Nothing but fives: no natural and no insurance, so the deal leaves a table to show.
    message = await _start_in_a_shut_out_channel(monkeypatch=monkeypatch, dealt=[])
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)
    stale_double = attached_button(view=table, custom_id="bj:double")
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    await attached_button(view=table, custom_id="bj:hit").callback(
        as_interaction(fake=_alice_press(message=message))
    )
    stale_press = _alice_press(message=message)
    await stale_double.callback(as_interaction(fake=stale_press))

    assert len(stale_press.followup.sent) == 1
    assert len(message.edits) == 3, "the stale press refreshed the table it was pressed on"

    stand = _alice_press(message=message)
    await attached_button(view=table, custom_id="bj:stand").callback(as_interaction(fake=stand))
    await table.wait_for_background_tasks()

    assert {"view": table} in message.edits, "the controls went dead before the dealer played"
    assert message.edits[-1]["view"] is None, "the settled table replaced the live one"
    assert scheduled.interactions == [stand]


@pytest.mark.parametrize(argnames="expired", argvalues=[False, True], ids=["live", "expired"])
@pytest.mark.parametrize(argnames="last", argvalues=["start", "hit"])
async def test_a_blackjack_table_left_to_time_out_in_a_shut_out_channel_closes_through_its_last_press(
    monkeypatch: pytest.MonkeyPatch, last: str, expired: bool
) -> None:
    """A timeout has no press of its own, so it renders and deletes through the last one.

    That is the start itself when nobody pressed after it. Once that press's token has expired
    only the channel is left, whose refusal is expected and so carries no traceback.
    """
    # Nothing but fives: no natural and no insurance, so the deal leaves a table to show.
    message = await _start_in_a_shut_out_channel(monkeypatch=monkeypatch, dealt=[])
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)
    if last == "start":
        press = cast("FakeInteraction", table.last_press)
        assert press.edits[0]["view"] is table, "the start press dealt the table"
    else:
        press = _alice_press(message=message)
        await attached_button(view=table, custom_id="bj:hit").callback(as_interaction(fake=press))
    press.expired = expired
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    reports = capture_levels(monkeypatch=monkeypatch, levels=("info", "warn"))

    await table.on_timeout()
    await table.wait_for_background_tasks()

    assert (press.edits[-1]["view"] is None) is not expired, "the settled table landed via it"
    assert (scheduled.messages, scheduled.interactions) == ([message], [press]), (
        "the delete rides the same press"
    )
    assert [(level, "_exc_info" in fields) for level, _, fields in reports] == (
        [("warn", False), ("warn", False)] if expired else []
    )


async def test_a_blackjack_table_kept_open_by_a_refused_press_closes_through_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A press refused out of turn restarts the table's timer, so the timeout closes through it.

    By then the press that last moved the table can be past its life, and in a channel the bot
    was shut out of nothing else reaches the table to show its result or delete it.
    """
    # Nothing but fives: no natural and no insurance, so the deal leaves a table to show.
    message = await _start_in_a_shut_out_channel(monkeypatch=monkeypatch, dealt=[])
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)
    cast("FakeInteraction", table.last_press).expired = True
    refused = FakeInteraction(user=FakeUser(user_id=2, name="bob"), message=message)
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)

    assert await table.interaction_check(interaction=as_interaction(fake=refused)) is False
    await table.on_timeout()
    await table.wait_for_background_tasks()

    assert [edit["view"] for edit in refused.edits[-1:]] == [None], "the result landed through it"
    assert (scheduled.messages, scheduled.interactions) == ([message], [refused])
    assert refused.followup.sent == [{"content": "現在輪到 Alice", "ephemeral": True}]


async def test_a_blackjack_natural_at_the_deal_settles_inside_the_start_press(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dealer natural under a ten ends the round at the deal, so the start press shows it all."""
    # Alice 5 5, the dealer's hole an ace under a king.
    message = await _start_in_a_shut_out_channel(
        monkeypatch=monkeypatch, dealt=[card(rank="5")] * 2 + [card(rank="A"), card(rank="K")]
    )
    table = message.edits[0]["view"]
    assert isinstance(table, BlackjackView)
    await table.wait_for_background_tasks()

    # The controls going dead, the peek's two frames, and the settled table.
    assert len(message.edits) == 4
    hidden, revealed = (cast("str", edit["embeds"][0].description) for edit in message.edits[1:3])
    assert "🂠" in hidden, "the peek shows the hole card face down first"
    assert "🂠" not in revealed
    assert message.edits[-1]["view"] is None


async def test_a_blackjack_natural_under_an_ace_settles_inside_the_insurance_press(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last insurance call closes the phase on a dealer natural, so its press shows the end."""
    # Alice 5 5, the dealer's hole a king under an ace.
    message = await _start_in_a_shut_out_channel(
        monkeypatch=monkeypatch, dealt=[card(rank="5")] * 2 + [card(rank="K"), card(rank="A")]
    )
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)

    await attached_button(view=table, custom_id="bj:insure_no").callback(
        as_interaction(fake=_alice_press(message=message))
    )
    await table.wait_for_background_tasks()

    # The table, the controls going dead, the peek's two frames, and the settled table.
    assert len(message.edits) == 5
    assert message.edits[-1]["view"] is None


async def test_an_ace_up_peek_without_a_natural_keeps_the_hole_face_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dealer peeks under an ace for a natural only; finding none, the hole stays secret.

    The bot reads the hole as its private edge, so a peek showing it to the table would hand
    every player the same card.
    """
    # Alice 5 5, the dealer's hole a 6 under an ace: insurance, then no natural.
    message = await _start_in_a_shut_out_channel(
        monkeypatch=monkeypatch, dealt=[card(rank="5")] * 2 + [card(rank="6"), card(rank="A")]
    )
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)

    await attached_button(view=table, custom_id="bj:insure_no").callback(
        as_interaction(fake=_alice_press(message=message))
    )

    dealer_seats = [cast("str", edit["embeds"][0].description) for edit in message.edits]
    assert all("🂠" in dealer_seat for dealer_seat in dealer_seats), "the hole stays face down"
    assert len(dealer_seats) == 2, "the table, then the table after the insurance call"
    assert table.round_state.phase == "player_actions"


async def test_the_bot_plays_its_seat_through_the_press_that_handed_it_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bot sits at every table its wallet can fund, and it has no press of its own.

    Every move it makes runs inside a human press still holding the round lock, so it edits the
    table through that press; through the channel, a shut-out table stalls on its first move.
    """
    bot = seat(user_id=999, display_name="Dealer", bet=10, balance_at_start=100)
    # Alice 5 5, the bot 5 5, the dealer's hole a 6 under an ace: insurance, no natural, and a
    # dealer bound for 17 on a shoe of fives, so the bot draws to 20 before it stands.
    message = await _start_in_a_shut_out_channel(
        monkeypatch=monkeypatch,
        dealt=[card(rank="5")] * 4 + [card(rank="6"), card(rank="A")],
        bot=bot,
    )
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)
    assert len(message.edits) == 2, "the table, then the bot's insurance call on the start press"

    await attached_button(view=table, custom_id="bj:insure_no").callback(
        as_interaction(fake=_alice_press(message=message))
    )
    assert len(message.edits) == 3, "no natural under the ace, so only the table follows the call"

    await attached_button(view=table, custom_id="bj:stand").callback(
        as_interaction(fake=_alice_press(message=message))
    )
    await table.wait_for_background_tasks()

    bot_hand = table.round_state.players[1].hands[0]
    assert len(bot_hand.cards) == 4, "the bot drew twice on the press that handed it the turn"
    assert message.edits[-1]["view"] is None, "the bot played its hand out and the table settled"


async def test_blackjack_owner_overbet_sets_table_bet_to_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies owner over-betting clamps the shared Blackjack lobby bet."""

    async def balance_by_user(user_id: int) -> int:
        """Returns distinct balances for the owner and the joining player; the bot has none."""
        return {1: 300, 2: 50_000_000, 999: 0}[user_id]

    monkeypatch.setattr(games, "get_balance", balance_by_user)

    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="1,000,000")
    lobby_view = owner_interaction.followup.sent[0]["view"]
    assert isinstance(lobby_view, BlackjackLobbyView)
    assert lobby_view.requested_bet == 300
    assert lobby_view.participants[0].bet == 300
    assert lobby_view.participants[0].is_allin is True

    join_button = lobby_button(view=lobby_view, label="加入")
    join_interaction = FakeInteraction(user=FakeUser(user_id=2, name="bob", display_name="Bob"))
    join_interaction.message = as_message(fake=FakeDiscordMessage())
    await join_button.callback(as_interaction(fake=join_interaction))

    bob = lobby_view.participants[1]
    assert bob.display_name == "Bob"
    assert bob.bet == 300
    assert bob.balance_at_start == 50_000_000
    assert bob.is_allin is False


async def test_refresh_participants_preserves_existing_blackjack_wagers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies start-time balance refresh keeps per-seat Blackjack wagers."""

    async def balance_by_user(user_id: int) -> int:
        """Returns enough balance for the owner and bot to keep their queued bets."""
        return {1: 500, 999: 1_000}[user_id]

    monkeypatch.setattr(games, "get_balance", balance_by_user)
    cog = _cog()
    owner = seat(bet=300, balance_at_start=300)
    bot_player = seat(user_id=999, display_name="Dealer", bet=125, balance_at_start=1_000)

    refreshed = await cog._refresh_participants(participants=[owner, bot_player], mode="clamp")

    assert [participant.bet for participant in refreshed.participants] == [300, 125]
    assert [participant.balance_at_start for participant in refreshed.participants] == [500, 1_000]


async def test_blackjack_string_bet_accepts_large_formatted_amount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large formatted bet is parsed without error but caps at MAX_SINGLE_BET."""

    async def balance_by_user(user_id: int) -> int:
        """Returns enough balance to cover a large formatted wager."""
        del user_id
        return 10_000_000_000_000_000

    monkeypatch.setattr(games, "get_balance", balance_by_user)

    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="9,007,199,254,740,993")
    lobby_view = owner_interaction.followup.sent[0]["view"]

    assert isinstance(lobby_view, BlackjackLobbyView)
    # The wager is parsed and the lobby is created, but the single-bet cap applies.
    assert lobby_view.requested_bet == MAX_SINGLE_BET
    assert lobby_view.participants[0].bet == MAX_SINGLE_BET


async def test_blackjack_string_bet_rejects_invalid_text() -> None:
    """Verifies invalid text is rejected before wager preparation."""
    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="not a number")

    assert owner_interaction.response.sent[0]["ephemeral"] is True
    assert owner_interaction.response.sent[0]["embed"].title == "下注格式錯誤"
    assert owner_interaction.response.sent[0]["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert owner_interaction.response.sent[0]["embed"].image.url == embed_spacer_url()
    assert owner_interaction.followup.sent == []
    assert owner_interaction.response.deferred is False


async def test_blackjack_owner_zero_bet_caps_all_in_at_max_single_bet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bet zero means all in, but a huge balance still caps at MAX_SINGLE_BET."""

    async def balance_by_user(user_id: int) -> int:
        """Returns a large owner balance that exceeds the single-bet cap; the bot has none."""
        return {1: 300_000_000_000_000, 999: 0}[user_id]

    monkeypatch.setattr(games, "get_balance", balance_by_user)

    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="0")
    lobby_view = owner_interaction.followup.sent[0]["view"]
    assert isinstance(lobby_view, BlackjackLobbyView)
    # All-in caps at the single-bet ceiling, so it is no longer a true all-in.
    assert lobby_view.requested_bet == MAX_SINGLE_BET
    assert lobby_view.participants[0].bet == MAX_SINGLE_BET
    assert lobby_view.participants[0].is_allin is False


async def test_blackjack_zero_bet_rejects_empty_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies zero means all in, not a zero-stake table."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    monkeypatch.setattr(games, "get_balance", _empty_game_balance)

    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="0")

    assert owner_interaction.followup.sent[0]["wait"] is True
    assert "view" not in owner_interaction.followup.sent[0]
    assert owner_interaction.followup.sent[0]["embed"].title == "餘額不足"
    assert owner_interaction.followup.sent[0]["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert owner_interaction.followup.sent[0]["embed"].image.url == embed_spacer_url()


async def test_dragon_gate_rejects_empty_balance_with_spacer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies the Dragon Gate insufficient-balance response keeps uniform width."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    monkeypatch.setattr(games, "get_balance", _empty_game_balance)

    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, owner_interaction)

    assert owner_interaction.followup.sent[0]["wait"] is True
    assert "view" not in owner_interaction.followup.sent[0]
    assert owner_interaction.followup.sent[0]["embed"].title == "餘額不足"
    assert owner_interaction.followup.sent[0]["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert owner_interaction.followup.sent[0]["embed"].image.url == embed_spacer_url()


async def test_prepare_participant_insufficient_balance_applies_embed_spacer() -> None:
    """Insufficient-balance lobby join reply carries the shared embed spacer."""
    interaction = FakeInteraction(user=FakeUser(user_id=7))

    async def fake_participant_from_user(**_kwargs: object) -> SimpleNamespace:
        """Stands in for a balance check that rejects the wager."""
        return SimpleNamespace(participant=None, balance=0)

    stub_self = SimpleNamespace(_participant_from_user=fake_participant_from_user)

    await GamesCogs._prepare_participant(
        cast("Any", stub_self),
        interaction=as_interaction(fake=interaction),
        wager=100,
        mode="clamp",
        insufficient_embed_builder=lambda balance: Embed(
            title="餘額不足", description=str(balance)
        ),
    )

    assert len(interaction.followup.sent) == 1
    sent = interaction.followup.sent[0]
    assert sent["ephemeral"] is True
    assert sent["embed"].image.url == embed_spacer_url()
    assert sent["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
