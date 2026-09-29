"""Tests for the `/games` cog: its commands, how it seats players, and its startup cleanup."""

from types import SimpleNamespace
from typing import Any, cast
from pathlib import Path

import pytest
import nextcord
from nextcord import Embed

from discordbot.utils import message_cleanup as cleanup_module
from discordbot.cogs.games import cog as games
from discordbot.typings.games import GameParticipant
from discordbot.cogs.games.cog import GamesCogs
from discordbot.typings.economy import JackpotSnapshot
from discordbot.cogs.games.blackjack import Card
from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME, embed_spacer_url
from discordbot.cogs.games.blackjack_views import BlackjackLobbyView
from discordbot.cogs.games.dragon_gate_views import DragonGateLobbyView

from tests.helpers.casting import as_bot, as_message, as_interaction
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


async def fake_dragon_gate_jackpot_snapshot() -> JackpotSnapshot:
    """Returns a stable fake Dragon Gate jackpot snapshot."""
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
    assert tmp_path / "game_cleanup.db" == cleanup_module._PENDING_PUBLIC_MESSAGE_DB_PATH


async def test_games_commands_run_with_patched_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies game commands create lobby views with patched dependencies."""
    monkeypatch.setattr(games, "schedule_public_message_delete", ignore_scheduled_public_message)
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

    monkeypatch.setattr(
        games, "fetch_dragon_gate_jackpot_snapshot", fake_dragon_gate_jackpot_snapshot
    )
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
    monkeypatch.setattr(games, "schedule_public_message_delete", ignore_scheduled_public_message)
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
    monkeypatch.setattr(games, "schedule_public_message_delete", ignore_scheduled_public_message)
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
    monkeypatch.setattr(
        games, "fetch_dragon_gate_jackpot_snapshot", fake_dragon_gate_jackpot_snapshot
    )

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


async def test_games_on_ready_cleans_stale_messages_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies startup cleanup runs once per GamesCogs instance."""
    bot = SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))
    calls: list[SimpleNamespace] = []

    async def record_cleanup(bot: SimpleNamespace) -> None:
        """Records the bot passed to startup cleanup."""
        calls.append(bot)

    monkeypatch.setattr(games, "delete_tracked_public_messages", record_cleanup)
    cog = GamesCogs(bot=as_bot(fake=bot))

    await cog.on_ready()
    await cog.on_ready()

    assert calls == [bot]


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
