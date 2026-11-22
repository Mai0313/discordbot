"""Tests for the `/games` cog: its commands, how it seats players, and its startup cleanup."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import nextcord
from nextcord import Embed

from discordbot.cogs.games.cog import GamesCogs
from discordbot.typings.economy import MAX_SINGLE_BET
from discordbot.cogs.games.lobby import BaseGameLobbyView
from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME, embed_spacer_url
from discordbot.utils.message_cleanup import PendingPublicMessage, list_pending_public_messages
from discordbot.cogs.games.dragon_gate import GAME_ID
from discordbot.cogs.games.presentation import lobby_participant_line
from discordbot.cogs.games.blackjack_views import BlackjackLobbyView
from discordbot.cogs.games.dragon_gate_views import DragonGateLobbyView
from discordbot.services.economy.presentation import amount_code

from tests.helpers.games import card, seat, lobby_button
from tests.helpers.casting import as_message, as_interaction
from tests.helpers.economy import seed_balance, get_jackpot_pool
from tests.helpers.discord_mocks import (
    BOT_USER_ID,
    FakeUser,
    FakeInteraction,
    FakeDiscordMessage,
    stub_bot,
)
from tests.helpers.message_cleanup import record_scheduled_deletes


def _cog() -> GamesCogs:
    """Builds the cog around a bot whose own account is user 999, the table's bot player."""
    return GamesCogs(bot=stub_bot(user=FakeUser(user_id=BOT_USER_ID, display_name="Dealer")))


async def test_bot_blackjack_participant_spreads_bet_by_true_count() -> None:
    """The bot's Kelly wager rises with a favorable channel true count."""
    cog = _cog()
    await seed_balance(user_id=999, name="dealer", amount=1_000_000)

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
    assert interaction.response.sent[0]["content"]
    assert interaction.followup.sent == []


async def test_games_commands_open_their_lobbies() -> None:
    """Each game command opens its lobby as a public followup carrying the width spacer.

    The lobby lists its owner, with the stake at Blackjack and beside the pool at 射龍門.
    """
    await seed_balance(user_id=1, name="alice", amount=100)
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
    assert [field.value for field in blackjack_interaction.followup.sent[0]["embed"].fields] == [
        lobby_participant_line(index=1, display_name="Alice", bet=10)
    ]

    dragon_gate_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, dragon_gate_interaction)
    assert dragon_gate_interaction.followup.sent[-1]["wait"] is True
    assert isinstance(dragon_gate_interaction.followup.sent[-1]["view"], DragonGateLobbyView)
    assert (
        dragon_gate_interaction.followup.sent[-1]["files"][0].filename
        == DEFAULT_EMBED_SPACER_FILENAME
    )
    assert dragon_gate_interaction.followup.sent[-1]["embed"].image.url == embed_spacer_url()
    pool = await get_jackpot_pool(game_id=GAME_ID)
    assert [
        field.value for field in dragon_gate_interaction.followup.sent[-1]["embed"].fields
    ] == [
        lobby_participant_line(index=1, display_name="Alice"),
        amount_code(amount=pool, compact=True),
    ]


@pytest.mark.parametrize(argnames="game", argvalues=["blackjack", "dragon_gate"])
async def test_an_opened_lobby_is_recorded_for_the_restart_sweep(game: str) -> None:
    """A lobby the bot restarts under is deleted by the next start's sweep, from this record."""
    await seed_balance(user_id=1, name="alice", amount=100)
    cog = _cog()
    interaction = FakeInteraction(user=FakeUser(user_id=1))
    if game == "blackjack":
        await GamesCogs.blackjack.callback(cog, interaction, bet="10")
    else:
        await GamesCogs.dragon_gate.callback(cog, interaction)

    lobby = interaction.followup.sent[-1]["view"]
    assert isinstance(lobby, BaseGameLobbyView)
    assert lobby.message is not None
    assert await list_pending_public_messages() == [
        PendingPublicMessage(
            channel_id=lobby.message.channel.id, message_id=lobby.message.id, user_name="alice"
        )
    ]


async def test_a_lobby_start_is_owner_only() -> None:
    """Anyone but the owner pressing 開始 is told so privately, and the lobby stays open."""
    await seed_balance(user_id=1, name="alice", amount=100)
    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="10")
    lobby_view = owner_interaction.followup.sent[0]["view"]
    assert isinstance(lobby_view, BlackjackLobbyView)

    start_button = lobby_button(view=lobby_view, label="開始")
    other_interaction = FakeInteraction(user=FakeUser(user_id=2, name="bob", display_name="Bob"))
    await start_button.callback(as_interaction(fake=other_interaction))

    assert [sent["ephemeral"] for sent in other_interaction.followup.sent] == [True]
    assert lobby_view._started is False


async def test_blackjack_owner_overbet_sets_table_bet_to_balance() -> None:
    """Verifies owner over-betting clamps the shared Blackjack lobby bet."""
    await seed_balance(user_id=1, name="alice", amount=300)
    await seed_balance(user_id=2, name="bob", amount=50_000_000)
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


async def test_refresh_participants_preserves_existing_blackjack_wagers() -> None:
    """Verifies start-time balance refresh keeps per-seat Blackjack wagers."""
    await seed_balance(user_id=1, name="alice", amount=500)
    await seed_balance(user_id=999, name="dealer", amount=1_000)
    cog = _cog()
    owner = seat(bet=300, balance_at_start=300)
    bot_player = seat(user_id=999, display_name="Dealer", bet=125, balance_at_start=1_000)

    refreshed = await cog._refresh_participants(participants=[owner, bot_player], mode="clamp")

    assert [participant.bet for participant in refreshed.participants] == [300, 125]
    assert [participant.balance_at_start for participant in refreshed.participants] == [500, 1_000]


async def test_blackjack_string_bet_accepts_large_formatted_amount() -> None:
    """A large formatted bet is parsed without error but caps at MAX_SINGLE_BET."""
    await seed_balance(user_id=1, name="alice", amount=10_000_000_000_000_000)
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
    assert (
        owner_interaction.response.sent[0]["embed"].title == GamesCogs._invalid_bet_embed().title
    )
    assert owner_interaction.response.sent[0]["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert owner_interaction.response.sent[0]["embed"].image.url == embed_spacer_url()
    assert owner_interaction.followup.sent == []
    assert owner_interaction.response.deferred is False


async def test_blackjack_owner_zero_bet_caps_all_in_at_max_single_bet() -> None:
    """Bet zero means all in, but a huge balance still caps at MAX_SINGLE_BET."""
    await seed_balance(user_id=1, name="alice", amount=300_000_000_000_000)
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
    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="0")

    refusal = owner_interaction.followup.sent[0]["embed"]
    expected = cog._insufficient_balance_embed(balance=0)
    assert owner_interaction.followup.sent[0]["wait"] is True
    assert "view" not in owner_interaction.followup.sent[0]
    assert (refusal.title, refusal.description) == (expected.title, expected.description)
    assert owner_interaction.followup.sent[0]["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert owner_interaction.followup.sent[0]["embed"].image.url == embed_spacer_url()


async def test_dragon_gate_rejects_empty_balance_with_spacer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies the Dragon Gate insufficient-balance response keeps uniform width."""
    record_scheduled_deletes(monkeypatch=monkeypatch)
    cog = _cog()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, owner_interaction)

    refusal = owner_interaction.followup.sent[0]["embed"]
    expected = cog._dragon_gate_insufficient_balance_embed(balance=0)
    assert owner_interaction.followup.sent[0]["wait"] is True
    assert "view" not in owner_interaction.followup.sent[0]
    assert (refusal.title, refusal.description) == (expected.title, expected.description)
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


async def test_refresh_participants_batches_balance_lookups_into_one_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lobby start re-checks every participant's balance with one `get_balances` call."""
    await seed_balance(user_id=1, name="alice", amount=500)
    await seed_balance(user_id=2, name="bob", amount=50)
    await seed_balance(user_id=3, name="carol", amount=300)
    cog = _cog()
    batch_calls: list[list[int]] = []
    single_calls: list[int] = []
    real_get_balances = GamesCogs._refresh_participants.__globals__["get_balances"]

    async def spy_get_balances(user_ids: Any) -> dict[int, int]:  # noqa: ANN401 -- test spy
        batch_calls.append(list(user_ids))
        return await real_get_balances(user_ids=user_ids)

    async def spy_get_balance(user_id: int) -> int:
        single_calls.append(user_id)
        return 0

    monkeypatch.setattr("discordbot.cogs.games.cog.get_balances", spy_get_balances)
    monkeypatch.setattr("discordbot.cogs.games.cog.get_balance", spy_get_balance)

    participants = [
        seat(user_id=1, display_name="Alice", bet=100, balance_at_start=500),
        seat(user_id=2, display_name="Bob", bet=100, balance_at_start=100),
        seat(user_id=3, display_name="Carol", bet=100, balance_at_start=300),
    ]
    result = await cog._refresh_participants(participants=participants, mode="exact")

    assert batch_calls == [[1, 2, 3]]
    assert single_calls == []
    assert [participant.user_id for participant in result.participants] == [1, 3]
    assert result.dropped_names == ["Bob"]
