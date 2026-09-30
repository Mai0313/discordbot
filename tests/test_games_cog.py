"""Tests for the `/games` cog: its commands, how it seats players, and its startup cleanup."""

from types import SimpleNamespace
from random import Random
from typing import Any, cast
from pathlib import Path

import pytest
import logfire
import nextcord
from nextcord import Embed, Interaction, HTTPException

from discordbot.utils import message_cleanup as cleanup_module
from discordbot.utils import interaction_responses
from discordbot.cogs.games import cog as games
from discordbot.cogs.games import blackjack_views
from discordbot.typings.games import GameParticipant, RefreshParticipantsResult
from discordbot.cogs.games.cog import GamesCogs
from discordbot.cogs.games.shoe import BlackjackShoeStore
from discordbot.typings.economy import JackpotSnapshot
from discordbot.cogs.games.blackjack import Card
from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME, embed_spacer_url
from discordbot.cogs.games.blackjack_views import BlackjackView, BlackjackLobbyView
from discordbot.cogs.games.dragon_gate_views import DragonGateLobbyView

from tests.helpers.games import attached_button
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


def _cog() -> GamesCogs:
    """Builds the cog around a bot whose own account is user 999, the table's bot player."""
    return GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )


def ignore_scheduled_public_message(
    message: FakeDiscordMessage, delay: float = 180, user_name: str | None = None
) -> None:
    """Ignores cleanup scheduling in command smoke tests."""
    return


async def fake_game_balance(user_id: int) -> int:
    """Returns a small fake game balance for anyone.

    Never asked about the bot's own id: `_bot_blackjack_participant` reads `get_account`, so
    the empty isolated economy DB is what makes the bot skip its seat, not a balance.
    """
    del user_id
    return 100


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

    async def fake_get_account(*, user_id: int) -> object:
        return SimpleNamespace(balance=1_000_000, total_earned=0, total_spent=0)

    async def fake_avatar(*, user: object, guild: object = None) -> str:
        return ""

    monkeypatch.setattr(games, "get_account", fake_get_account)
    monkeypatch.setattr(games, "guild_avatar_url", fake_avatar)

    neutral = await cog._bot_blackjack_participant(guild=None, table_bet=100, channel_id=1)
    # A ten-rich stored shoe above the reshuffle threshold gives channel 2 a strongly
    # positive true count.
    cog._blackjack_shoes.save_shoe(
        channel_id=2, cards=[Card(rank="10", suit="♠") for _ in range(120)]
    )
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


async def test_games_commands_run_with_patched_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies game commands create lobby views with patched dependencies."""
    monkeypatch.setattr(
        interaction_responses, "schedule_public_message_delete", ignore_scheduled_public_message
    )
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


async def test_blackjack_lobby_start_is_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies only the Blackjack lobby owner can press Start."""
    monkeypatch.setattr(games, "get_balance", fake_game_balance)

    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="10")
    lobby_view = owner_interaction.followup.sent[0]["view"]
    assert isinstance(lobby_view, BlackjackLobbyView)

    start_button = next(
        child for child in lobby_view.children if getattr(child, "label", "") == "開始"
    )
    other_interaction = FakeInteraction(user=FakeUser(user_id=2, name="bob", display_name="Bob"))
    await start_button.callback(as_interaction(fake=other_interaction))

    assert other_interaction.response.sent
    assert isinstance(other_interaction.response.sent[0]["content"], str)


async def _nobody_joins(interaction: Interaction[Any]) -> GameParticipant | None:
    """Lobby join hook for a table nobody joins."""
    raise AssertionError(interaction)


async def _everyone_stays(participants: list[GameParticipant]) -> RefreshParticipantsResult:
    """Lobby start hook that leaves every participant seated."""
    return RefreshParticipantsResult(participants=participants)


def _scripted_blackjack_lobby(
    dealt: list[Card], bot: GameParticipant | None = None
) -> BlackjackLobbyView:
    """Builds Alice's Blackjack lobby whose round deals `dealt` first, then fives."""
    shoe = dealt + [Card(rank="5", suit="♠") for _ in range(100)]
    return BlackjackLobbyView(
        owner=GameParticipant(
            user_id=1,
            account_name="alice",
            display_name="Alice",
            bet=10,
            balance_at_start=100,
            is_allin=False,
        ),
        requested_bet=10,
        rng=Random(x=0),  # noqa: S311 -- the scripted shoe decides the deal
        prepare_participant=_nobody_joins,
        refresh_participants=_everyone_stays,
        bot_user_id=None if bot is None else bot.user_id,
        extra_initial_participants=None if bot is None else [bot],
        shoe_store=BlackjackShoeStore(shoes={7: shoe}),
        channel_id=7,
    )


def _recorded_reports(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, object]]]:
    """Captures `(level, fields)` for every `info` and `warn` the lobby writes."""
    reports: list[tuple[str, dict[str, object]]] = []
    for level in ("info", "warn"):
        monkeypatch.setattr(
            target=logfire,
            name=level,
            value=lambda message, level=level, **fields: reports.append((level, fields)),
        )
    return reports


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
    scheduled: list[object] = []
    monkeypatch.setattr(
        "discordbot.cogs.games.lobby.schedule_public_message_delete",
        lambda message, delay=180, user_name=None: scheduled.append(message),
    )
    reports = _recorded_reports(monkeypatch=monkeypatch)
    # Nothing but fives: no natural and no insurance, so the deal leaves a table to show.
    lobby = _scripted_blackjack_lobby(dealt=[])
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)
    owner_interaction = FakeInteraction(user=FakeUser(user_id=1), message=message)
    owner_interaction.edit_failure = failure
    start_button = next(child for child in lobby.children if getattr(child, "label", "") == "開始")

    await start_button.callback(as_interaction(fake=owner_interaction))

    assert len(owner_interaction.followup.sent) == 1
    assert owner_interaction.followup.sent[0]["ephemeral"] is True
    assert reports == [(level, {"channel_id": 200, "message_id": 1, "code": failure.code})]
    assert not lobby.is_finished()
    await lobby.on_timeout()
    assert scheduled == [message]


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
    scheduled: list[object] = []
    monkeypatch.setattr(
        "discordbot.cogs.games.lobby.schedule_public_message_delete",
        lambda message, delay=180, user_name=None: scheduled.append(message),
    )
    reports = _recorded_reports(monkeypatch=monkeypatch)
    lobby = _scripted_blackjack_lobby(dealt=[])
    message = FakeDiscordMessage()
    message.edit_failure = failure
    lobby.message = as_message(fake=message)

    await lobby.on_timeout()

    assert [(name, "_exc_info" in fields) for name, fields in reports] == [(level, traceback)]
    assert scheduled == [message]


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

    reports = _recorded_reports(monkeypatch=monkeypatch)
    bot = GameParticipant(
        user_id=999,
        account_name="dealer",
        display_name="Dealer",
        bet=10,
        balance_at_start=100,
        is_allin=False,
    )
    # Alice 5 5, the bot 5 5, the dealer's hole 5 and an ace up: the bot owes an insurance call.
    lobby = _scripted_blackjack_lobby(
        dealt=[Card(rank="5", suit="♠")] * 5 + [Card(rank="A", suit="♠")], bot=bot
    )
    message = FakeDiscordMessage()
    lobby.message = as_message(fake=message)
    owner_interaction = _RefusedAfterTheTable(user=FakeUser(user_id=1), message=message)
    start_button = next(child for child in lobby.children if getattr(child, "label", "") == "開始")

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
    monkeypatch.setattr(
        "discordbot.cogs.games.interactions.schedule_public_message_delete",
        lambda message, delay=180, user_name=None: None,
    )
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
    start_button = next(child for child in lobby.children if getattr(child, "label", "") == "開始")

    await start_button.callback(as_interaction(fake=owner_start))

    assert owner_start.followup.sent == []
    return message


def _alice_press(message: FakeDiscordMessage) -> FakeInteraction:
    """Builds one press by Alice on the table message."""
    return FakeInteraction(user=FakeUser(user_id=1), message=message)


async def test_a_blackjack_lobby_in_a_channel_the_bot_was_shut_out_of_still_deals_and_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The start and each table press edit through their own token, which no channel can refuse."""
    # Nothing but fives: no natural and no insurance, so the deal leaves a table to show.
    message = await _start_in_a_shut_out_channel(monkeypatch=monkeypatch, dealt=[])
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)
    stale_double = attached_button(view=table, custom_id="bj:double")

    await attached_button(view=table, custom_id="bj:hit").callback(
        as_interaction(fake=_alice_press(message=message))
    )
    stale_press = _alice_press(message=message)
    await stale_double.callback(as_interaction(fake=stale_press))

    assert len(stale_press.followup.sent) == 1
    assert len(message.edits) == 3, "the stale press refreshed the table it was pressed on"

    await attached_button(view=table, custom_id="bj:stand").callback(
        as_interaction(fake=_alice_press(message=message))
    )
    await table.wait_for_background_tasks()

    assert {"view": table} in message.edits, "the controls went dead before the dealer played"
    assert message.edits[-1]["view"] is None, "the settled table replaced the live one"


async def test_a_blackjack_natural_at_the_deal_settles_inside_the_start_press(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dealer natural under a ten ends the round at the deal, so the start press shows it all."""
    # Alice 5 5, the dealer's hole an ace under a king.
    message = await _start_in_a_shut_out_channel(
        monkeypatch=monkeypatch,
        dealt=[Card(rank="5", suit="♠")] * 2
        + [Card(rank="A", suit="♠"), Card(rank="K", suit="♠")],
    )
    table = message.edits[0]["view"]
    assert isinstance(table, BlackjackView)
    await table.wait_for_background_tasks()

    # The controls going dead, the peek's two frames, and the settled table.
    assert len(message.edits) == 4
    assert message.edits[-1]["view"] is None


async def test_a_blackjack_natural_under_an_ace_settles_inside_the_insurance_press(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last insurance call closes the phase on a dealer natural, so its press shows the end."""
    # Alice 5 5, the dealer's hole a king under an ace.
    message = await _start_in_a_shut_out_channel(
        monkeypatch=monkeypatch,
        dealt=[Card(rank="5", suit="♠")] * 2
        + [Card(rank="K", suit="♠"), Card(rank="A", suit="♠")],
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


async def test_the_bot_plays_its_seat_through_the_press_that_handed_it_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bot sits at every table its wallet can fund, and it has no press of its own.

    Every move it makes runs inside a human press still holding the round lock, so it edits the
    table through that press; through the channel, a shut-out table stalls on its first move.
    """
    bot = GameParticipant(
        user_id=999,
        account_name="dealer",
        display_name="Dealer",
        bet=10,
        balance_at_start=100,
        is_allin=False,
    )
    # Alice 5 5, the bot 5 5, the dealer's hole a 6 under an ace: insurance, no natural, and a
    # dealer bound for 17 on a shoe of fives, so the bot draws to 20 before it stands.
    message = await _start_in_a_shut_out_channel(
        monkeypatch=monkeypatch,
        dealt=[Card(rank="5", suit="♠")] * 4
        + [Card(rank="6", suit="♠"), Card(rank="A", suit="♠")],
        bot=bot,
    )
    table = message.edits[-1]["view"]
    assert isinstance(table, BlackjackView)
    assert len(message.edits) == 2, "the table, then the bot's insurance call on the start press"

    await attached_button(view=table, custom_id="bj:insure_no").callback(
        as_interaction(fake=_alice_press(message=message))
    )
    assert len(message.edits) == 5, "the peek's two frames, then the table after it"

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
        """Returns distinct balances for owner, joining player, and the bot."""
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

    join_button = next(
        child for child in lobby_view.children if getattr(child, "label", "") == "加入"
    )
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
    owner = GameParticipant(
        user_id=1,
        account_name="alice",
        display_name="Alice",
        bet=300,
        balance_at_start=300,
        is_allin=True,
    )
    bot_player = GameParticipant(
        user_id=999,
        account_name="dealer",
        display_name="Dealer",
        bet=125,
        balance_at_start=1_000,
        is_allin=False,
    )

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
    assert lobby_view.requested_bet == 1_000_000
    assert lobby_view.participants[0].bet == 1_000_000


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
        """Returns a large owner balance that exceeds the single-bet cap."""
        return {1: 300_000_000_000_000, 2: 500_000_000_000_000, 999: 0}[user_id]

    monkeypatch.setattr(games, "get_balance", balance_by_user)

    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="0")
    lobby_view = owner_interaction.followup.sent[0]["view"]
    assert isinstance(lobby_view, BlackjackLobbyView)
    # All-in caps at the single-bet ceiling, so it is no longer a true all-in.
    assert lobby_view.requested_bet == 1_000_000
    assert lobby_view.participants[0].bet == 1_000_000
    assert lobby_view.participants[0].is_allin is False


async def test_blackjack_zero_bet_rejects_empty_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies zero means all in, not a zero-stake table."""
    monkeypatch.setattr(
        interaction_responses, "schedule_public_message_delete", ignore_scheduled_public_message
    )
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
    monkeypatch.setattr(
        interaction_responses, "schedule_public_message_delete", ignore_scheduled_public_message
    )
    monkeypatch.setattr(games, "get_balance", _empty_game_balance)

    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, owner_interaction)

    assert owner_interaction.followup.sent[0]["wait"] is True
    assert "view" not in owner_interaction.followup.sent[0]
    assert owner_interaction.followup.sent[0]["embed"].title == "餘額不足"
    assert owner_interaction.followup.sent[0]["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert owner_interaction.followup.sent[0]["embed"].image.url == embed_spacer_url()


async def test_dragon_gate_lobby_start_is_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies only the Dragon Gate lobby owner can press Start."""
    monkeypatch.setattr(games, "get_balance", _wealthy_game_balance)
    monkeypatch.setattr(games, "get_jackpot_snapshot", fake_dragon_gate_jackpot_snapshot)

    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, owner_interaction)
    lobby_view = owner_interaction.followup.sent[-1]["view"]
    assert isinstance(lobby_view, DragonGateLobbyView)

    start_button = next(
        child for child in lobby_view.children if getattr(child, "label", "") == "開始"
    )
    other_interaction = FakeInteraction(user=FakeUser(user_id=2, name="bob", display_name="Bob"))
    await start_button.callback(as_interaction(fake=other_interaction))

    assert other_interaction.response.sent
    assert isinstance(other_interaction.response.sent[0]["content"], str)


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
