from types import SimpleNamespace
from typing import Self
from pathlib import Path
from datetime import UTC, datetime, timedelta
from collections.abc import AsyncIterator

import pytest
import nextcord
from nextcord import Embed
from nextcord.ext import commands

from discordbot import cli
from discordbot.cogs import games, video, economy, template, auto_unmute, parse_threads
from discordbot.cogs.games import GamesCogs
from discordbot.cogs.video import VideoCogs
from discordbot.cogs.economy import EconomyCogs
from discordbot.cogs._economy import database
from discordbot.cogs.template import TemplateCogs
from discordbot.utils.threads import ThreadsOutput
from discordbot.cogs.auto_unmute import AutoUnmuteCogs
from discordbot.cogs.parse_threads import ThreadsCogs
from discordbot.cogs._games.blackjack import Card, BlackjackHand


class FakeResponse:
    def __init__(self) -> None:
        self.deferred = False
        self.sent: list[dict[str, object]] = []

    async def defer(self) -> None:
        self.deferred = True

    async def send_message(self, **kwargs: object) -> None:
        self.sent.append(kwargs)


class FakeFollowup:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send(self, **kwargs: object) -> object:
        self.sent.append(kwargs)
        return FakeDiscordMessage()


class FakeInteraction:
    def __init__(self, *, user: object | None = None) -> None:
        self.user = user or FakeUser()
        self.response = FakeResponse()
        self.followup = FakeFollowup()
        self.edits: list[dict[str, object]] = []

    async def edit_original_message(self, **kwargs: object) -> None:
        self.edits.append(kwargs)


class FakeUser:
    def __init__(
        self, *, user_id: int = 1, name: str = "alice", display_name: str = "Alice", bot: bool = False
    ) -> None:
        self.id = user_id
        self.name = name
        self.display_name = display_name
        self.bot = bot
        self.mention = f"<@{user_id}>"
        self.display_avatar = SimpleNamespace(url="https://example.test/avatar.png")


class FakeDiscordMessage:
    def __init__(self) -> None:
        self.edits: list[dict[str, object]] = []
        self.reactions: list[str] = []
        self.removed: list[tuple[str, object]] = []
        self.replies: list[dict[str, object]] = []
        self.deleted = False
        self.suppressed = False

    async def edit(self, **kwargs: object) -> None:
        if "suppress" in kwargs:
            self.suppressed = bool(kwargs["suppress"])
        self.edits.append(kwargs)

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)

    async def remove_reaction(self, *, emoji: str, member: object) -> None:
        self.removed.append((emoji, member))

    async def reply(self, **kwargs: object) -> None:
        self.replies.append(kwargs)

    async def delete(self) -> None:
        self.deleted = True


class DownloadResultStub:
    def __init__(self, *, filename: Path) -> None:
        self.filename = filename

    def __enter__(self) -> Self:
        """Returns the fake download result."""
        return self

    def __exit__(self, *args: object) -> None:
        """Leaves the fake downloaded file on disk for assertions."""
        return


class DownloaderStub:
    def __init__(self, *, results: list[DownloadResultStub]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def download(self, **kwargs: object) -> DownloadResultStub:
        self.calls.append(kwargs)
        return self.results.pop(0)


class ParseResultStub:
    def __init__(self, *, results: list[ThreadsOutput] | BaseException) -> None:
        self.results = results

    def __enter__(self) -> list[ThreadsOutput]:
        """Returns parsed posts or raises the configured parsing error."""
        if isinstance(self.results, BaseException):
            raise self.results
        return self.results

    def __exit__(self, *args: object) -> None:
        """Keeps fake parsed outputs available after context exit."""
        return


class ThreadsDownloaderStub:
    def __init__(self, *, results: list[ThreadsOutput] | BaseException) -> None:
        self.results = results

    def parse(self, url: str) -> ParseResultStub:
        return ParseResultStub(results=self.results)


def _thread_output(
    *,
    text: str = "hello",
    image_urls: list[str] | None = None,
    video_paths: list[Path] | None = None,
    video_urls: list[str] | None = None,
) -> ThreadsOutput:
    return ThreadsOutput(
        text=text,
        url="https://www.threads.net/@alice/post/abc",
        image_urls=image_urls or [],
        video_urls=video_urls or [],
        video_paths=video_paths or [],
        author_name="alice",
        author_icon_url="https://example.test/avatar.png",
        like_count=1,
        reply_count=2,
        repost_count=3,
        quote_count=4,
        reshare_count=5,
        taken_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


async def test_template_on_message_and_ping() -> None:
    cog = TemplateCogs(bot=SimpleNamespace(latency=0.123))
    message = FakeDiscordMessage()
    message.author = FakeUser(bot=False)
    message.content = "debug"
    await cog.on_message(message=message)
    assert message.reactions == ["🤬"]

    bot_message = FakeDiscordMessage()
    bot_message.author = FakeUser(bot=True)
    bot_message.content = "debug"
    await cog.on_message(message=bot_message)
    assert bot_message.reactions == []

    interaction = FakeInteraction(user=FakeUser(display_name="Alice"))
    await TemplateCogs.ping.callback(cog, interaction)
    embed = interaction.followup.sent[0]["embed"]
    assert isinstance(embed, Embed)
    assert embed.title == ":ping_pong: Pong!"


async def test_video_deliver_and_download_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cog = VideoCogs(bot=SimpleNamespace())
    small = tmp_path / "small.mp4"
    small.write_bytes(data=b"0" * 100)
    big = tmp_path / "big.mp4"
    big.write_bytes(data=b"0" * (video._DISCORD_FILE_LIMIT_BYTES + 1))
    low = tmp_path / "low.mp4"
    low.write_bytes(data=b"1" * 100)

    interaction = FakeInteraction()
    await cog._deliver(
        interaction=interaction, file_size_mb=1.25, file_path=str(small), file_name=small.name
    )
    assert interaction.followup.sent[0]["content"].startswith("✅ 下載成功")
    assert interaction.edits[-1]["content"] == "✅"

    downloader = DownloaderStub(
        results=[DownloadResultStub(filename=big), DownloadResultStub(filename=low)]
    )
    monkeypatch.setattr(video, "VideoDownloader", lambda output_folder: downloader)
    retry_interaction = FakeInteraction()
    await VideoCogs.download_video.callback(cog, retry_interaction, url="https://x.test", quality="best")
    assert [call["quality"] for call in downloader.calls] == ["best", "low"]
    assert retry_interaction.followup.sent[-1]["file"] is not None

    fail_interaction = FakeInteraction()
    monkeypatch.setattr(
        video,
        "VideoDownloader",
        lambda output_folder: DownloaderStub(results=[DownloadResultStub(filename=big)]),
    )
    await VideoCogs.download_video.callback(cog, fail_interaction, url="https://x.test", quality="low")
    assert "檔案大小超過" in fail_interaction.edits[-1]["content"]

    monkeypatch.setattr(video, "VideoDownloader", lambda output_folder: _RaiseDownloader())
    error_interaction = FakeInteraction()
    await VideoCogs.download_video.callback(
        cog, error_interaction, url="https://x.test", quality="best"
    )
    assert "檔案無法下載" in error_interaction.edits[-1]["content"]


class _RaiseDownloader:
    def download(self, **kwargs: object) -> DownloadResultStub:
        raise RuntimeError("download failed")


async def test_threads_cog_builds_embeds_and_handles_messages(tmp_path: Path) -> None:
    bot = SimpleNamespace(user=SimpleNamespace(id=999))
    cog = ThreadsCogs(bot=bot)
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(data=b"123")

    parent = _thread_output(text="parent", video_urls=["https://example.test/video.mp4"])
    target = _thread_output(image_urls=["https://example.test/1.png", "https://example.test/2.png"])
    embeds = cog._build_embeds(results=[parent, target])
    assert len(embeds) == 3
    assert "點此觀看影片" in embeds[0].description
    assert ThreadsCogs._gradient_color(index=0, total=1) == nextcord.Color.default()

    no_match = FakeDiscordMessage()
    no_match.author = FakeUser(bot=False)
    no_match.content = "hello"
    await cog.on_message(message=no_match)
    assert no_match.reactions == []

    success_message = FakeDiscordMessage()
    success_message.author = FakeUser(bot=False)
    success_message.content = "https://www.threads.net/@alice/post/abc"
    success_message.guild = SimpleNamespace(filesize_limit=25 * 1024 * 1024)
    cog.downloader = ThreadsDownloaderStub(
        results=[_thread_output(video_paths=[video_file], image_urls=[])]
    )
    await cog.on_message(message=success_message)
    assert success_message.suppressed
    assert success_message.replies[0]["files"]
    assert success_message.reactions[-1] == "🆗"

    warning_message = FakeDiscordMessage()
    warning_message.author = FakeUser(bot=False)
    warning_message.content = "https://www.threads.net/@alice/post/abc"
    warning_message.guild = None
    cog.downloader = ThreadsDownloaderStub(results=[])
    await cog.on_message(message=warning_message)
    assert warning_message.reactions[-1] == "⚠️"

    error_message = FakeDiscordMessage()
    error_message.author = FakeUser(bot=False)
    error_message.content = "https://www.threads.net/@alice/post/abc"
    error_message.guild = None
    cog.downloader = ThreadsDownloaderStub(results=RuntimeError("parse failed"))
    await cog.on_message(message=error_message)
    assert error_message.reactions[-1] == "❌"


async def test_auto_unmute_tracks_audit_and_generates_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    sent: list[str] = []
    channel = SimpleNamespace(send=lambda content: _append_async(sent, content))
    bot_user = FakeUser(user_id=999, name="bot", display_name="Bot")
    bot = SimpleNamespace(user=bot_user)
    cog = AutoUnmuteCogs(bot=bot)

    guild = SimpleNamespace(
        id=123,
        name="Guild",
        get_channel=lambda channel_id: channel,
        system_channel=None,
        audit_logs=lambda **kwargs: _audit_entries(bot_user),
    )
    await cog.on_message(
        message=SimpleNamespace(
            guild=guild, author=FakeUser(bot=False), channel=SimpleNamespace(id=456)
        )
    )
    assert cog._last_active_channel == {123: 456}

    monkeypatch.setattr(auto_unmute.nextcord.abc, "Messageable", object)
    assert cog._resolve_channel(guild=guild) is channel
    moderator, reason = await cog._lookup_audit(guild=guild)
    assert moderator.name == "moderator"
    assert reason == "testing"

    cog.__dict__["client"] = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: _response_async("not today"))
    )
    reply = await cog._generate_reply(
        guild_name="Guild",
        moderator=moderator,
        reason=reason,
        until=datetime.now(tz=UTC) + timedelta(minutes=10),
    )
    assert reply == "not today"

    member = SimpleNamespace(
        id=999,
        guild=guild,
        communication_disabled_until=datetime.now(tz=UTC) + timedelta(minutes=5),
        edit=lambda **kwargs: _async_none(),
    )
    await cog._handle_self_timeout(member=member, until=member.communication_disabled_until)
    assert sent == ["not today"]

    before = SimpleNamespace(communication_disabled_until=None)
    after = member
    handled: list[object] = []
    monkeypatch.setattr(cog, "_handle_self_timeout", lambda **kwargs: _append_async(handled, kwargs))
    await cog.on_member_update(before=before, after=after)
    assert handled


async def _audit_entries(bot_user: FakeUser) -> AsyncIterator[object]:
    yield SimpleNamespace(
        target=SimpleNamespace(id=111),
        changes=SimpleNamespace(after=SimpleNamespace(communication_disabled_until=True)),
        user=FakeUser(name="wrong"),
        reason="wrong",
    )
    yield SimpleNamespace(
        target=SimpleNamespace(id=bot_user.id),
        changes=SimpleNamespace(after=SimpleNamespace(communication_disabled_until=True)),
        user=FakeUser(name="moderator"),
        reason="testing",
    )


async def _response_async(output_text: str) -> object:
    return SimpleNamespace(output_text=output_text)


async def _append_async(container: list[object], item: object) -> None:
    container.append(item)


async def _async_none() -> None:
    return None


async def test_economy_commands_use_database_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduled: list[object] = []
    monkeypatch.setattr(
        economy, "schedule_game_message_delete", lambda **kwargs: scheduled.append(kwargs["message"])
    )
    monkeypatch.setattr(economy, "get_balance", lambda user_id: _value_async(150))
    monkeypatch.setattr(economy, "top_n", lambda **kwargs: _value_async([(1, "alice", 150)]))
    monkeypatch.setattr(economy, "get_account", lambda user_id: _value_async(("Bot", -50, 100, 150)))
    monkeypatch.setattr(
        economy,
        "transfer",
        lambda **kwargs: _value_async(database.TransferResult(sender_balance=50, receiver_balance=100)),
    )
    bot = SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))
    cog = EconomyCogs(bot=bot)
    interaction = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.balance.callback(cog, interaction)
    await EconomyCogs.leaderboard.callback(cog, interaction)
    await EconomyCogs.house.callback(cog, interaction)
    await EconomyCogs.give.callback(cog, interaction, member=FakeUser(user_id=2, name="bob"), amount=100)
    assert len(interaction.followup.sent) == 4
    assert scheduled

    bot_receiver = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.give.callback(
        cog, bot_receiver, member=FakeUser(user_id=3, name="bot", bot=True), amount=1
    )
    assert "不能把" in bot_receiver.followup.sent[0]["embed"].description


async def _value_async(value: object) -> object:
    return value


async def test_games_commands_run_with_patched_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    monkeypatch.setattr(games.asyncio, "sleep", lambda *args, **kwargs: _async_none())
    monkeypatch.setattr(games, "schedule_game_message_delete", lambda message: None)
    monkeypatch.setattr(
        games,
        "place_bet",
        lambda **kwargs: _value_async(database.PlacedBet(amount=10, balance_after=90, is_allin=False)),
    )
    monkeypatch.setattr(games, "get_balance", lambda user_id: _value_async(0))
    monkeypatch.setattr(
        games,
        "settle_wager",
        lambda **kwargs: _value_async(SimpleNamespace(delta=10, new_balance=110, house_balance=-10)),
    )
    monkeypatch.setattr(
        games,
        "settle_blackjack_round",
        lambda **kwargs: _value_async(
            SimpleNamespace(
                outcome="win", delta=15, new_balance=115, house_balance=-15, detail="natural"
            )
        ),
    )
    hand = BlackjackHand(rng=games.SystemRandom(), bet=10)
    hand.player = [Card(rank="A", suit="♠"), Card(rank="K", suit="♣")]
    hand.dealer = [Card(rank="9", suit="♠"), Card(rank="8", suit="♣")]
    hand.finished = True

    def fake_deal_initial() -> None:
        return None

    hand.deal_initial = fake_deal_initial
    monkeypatch.setattr(games, "BlackjackHand", lambda **kwargs: hand)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["dealer"] = SimpleNamespace(
        taunt_bet=lambda **kwargs: _value_async("taunt"),
        settle=lambda **kwargs: _value_async("settled"),
    )

    dice_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dice.callback(cog, dice_interaction, bet=10)
    assert dice_interaction.followup.sent[0]["wait"] is True

    dragon_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, dragon_interaction, bet=10)
    assert dragon_interaction.followup.sent[0]["wait"] is True

    blackjack_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, blackjack_interaction, bet=10)
    assert blackjack_interaction.followup.sent[0]["wait"] is True


def test_setup_functions_register_cogs(monkeypatch: pytest.MonkeyPatch) -> None:
    added: list[tuple[object, bool | None]] = []
    bot = SimpleNamespace(add_cog=lambda cog, override=None: added.append((cog, override)))
    for module in [video, games, economy, template, parse_threads, auto_unmute]:
        monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
        monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
        module.setup(bot=bot)
    assert [type(item[0]) for item in added] == [
        VideoCogs,
        GamesCogs,
        EconomyCogs,
        TemplateCogs,
        ThreadsCogs,
        AutoUnmuteCogs,
    ]


def test_cli_loads_cogs_and_handles_command_errors(tmp_path: Path) -> None:
    loaded: list[object] = []
    bot = SimpleNamespace(
        load_extensions=lambda modules, stop_at_error: loaded.append((modules, stop_at_error))
    )
    cli.DiscordBot._load_cogs_sync(bot)
    assert loaded[0][1] is True
    assert "discordbot.cogs.template" in loaded[0][0]


async def test_cli_message_and_command_error_branches() -> None:
    processed: list[object] = []
    bot = SimpleNamespace(
        user=FakeUser(user_id=999, bot=True),
        process_commands=lambda message: _append_async(processed, message),
    )
    user_message = SimpleNamespace(author=FakeUser(user_id=1, bot=False))
    await cli.DiscordBot.on_message(bot, message=user_message)
    assert processed == [user_message]
    await cli.DiscordBot.on_message(bot, message=SimpleNamespace(author=bot.user))
    assert len(processed) == 1

    sent: list[dict[str, object]] = []
    context = SimpleNamespace(
        send=lambda **kwargs: _append_async(sent, kwargs),
        guild=SimpleNamespace(name="Guild", id=1),
        author=FakeUser(user_id=1),
    )
    await cli.DiscordBot.on_command_error(bot, context, commands.NotOwner())
    await cli.DiscordBot.on_command_error(
        bot, context, commands.MissingPermissions(missing_permissions=["kick_members"])
    )
    await cli.DiscordBot.on_command_error(
        bot, context, commands.BotMissingPermissions(missing_permissions=["send_messages"])
    )
    await cli.DiscordBot.on_command_error(bot, context, commands.CommandNotFound("nope"))
    assert len(sent) == 4
