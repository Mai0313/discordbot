from __future__ import annotations

from types import TracebackType, SimpleNamespace
from typing import TYPE_CHECKING, Self, Unpack, TypedDict
from datetime import UTC, datetime, timedelta

import nextcord
from nextcord import File, Embed
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

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import AsyncIterator

    import pytest


class DiscordPayload(TypedDict, total=False):
    content: str | None
    embed: Embed
    embeds: list[Embed]
    file: File
    files: list[File]
    view: nextcord.ui.View
    wait: bool
    ephemeral: bool
    suppress: bool


class OriginalEditPayload(TypedDict, total=False):
    content: str


class SelfTimeoutCall(TypedDict):
    member: SimpleNamespace
    until: datetime


class FakeResponse:
    def __init__(self) -> None:
        self.deferred = False
        self.sent: list[DiscordPayload] = []

    async def defer(self) -> None:
        self.deferred = True

    async def send_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        self.sent.append(kwargs)


class FakeFollowup:
    def __init__(self) -> None:
        self.sent: list[DiscordPayload] = []

    async def send(self, **kwargs: Unpack[DiscordPayload]) -> FakeDiscordMessage:
        self.sent.append(kwargs)
        return FakeDiscordMessage()


class FakeInteraction:
    def __init__(self, *, user: FakeUser | None = None) -> None:
        self.user = user or FakeUser()
        self.response = FakeResponse()
        self.followup = FakeFollowup()
        self.edits: list[OriginalEditPayload] = []

    async def edit_original_message(self, **kwargs: Unpack[OriginalEditPayload]) -> None:
        self.edits.append(kwargs)


class FakeUser:
    def __init__(
        self,
        *,
        user_id: int = 1,
        name: str = "alice",
        display_name: str = "Alice",
        bot: bool = False,
    ) -> None:
        self.id = user_id
        self.name = name
        self.display_name = display_name
        self.bot = bot
        self.mention = f"<@{user_id}>"
        self.display_avatar = SimpleNamespace(url="https://example.test/avatar.png")
        # /balance and /borrow read user.created_at via the snowflake-derived
        # timestamp; pin it well into the past so the credit_limit tier lookup
        # exercises the high-tier branch.
        self.created_at = datetime.now(tz=UTC) - timedelta(days=365 * 5)


class FakeDiscordMessage:
    def __init__(self) -> None:
        self.edits: list[DiscordPayload] = []
        self.reactions: list[str] = []
        self.removed: list[tuple[str, FakeUser]] = []
        self.replies: list[DiscordPayload] = []
        self.deleted = False
        self.suppressed = False

    async def edit(self, **kwargs: Unpack[DiscordPayload]) -> None:
        if "suppress" in kwargs:
            self.suppressed = bool(kwargs["suppress"])
        self.edits.append(kwargs)

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)

    async def remove_reaction(self, *, emoji: str, member: FakeUser) -> None:
        self.removed.append((emoji, member))

    async def reply(self, **kwargs: Unpack[DiscordPayload]) -> None:
        self.replies.append(kwargs)

    async def delete(self) -> None:
        self.deleted = True


class DownloadResultStub:
    def __init__(self, *, filename: Path) -> None:
        self.filename = filename

    def __enter__(self) -> Self:
        """Returns the fake download result."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leaves the fake downloaded file on disk for assertions."""
        return


class DownloaderStub:
    def __init__(self, *, results: list[DownloadResultStub]) -> None:
        self.results = results
        self.calls: list[dict[str, str | bool]] = []

    def download(self, *, url: str, quality: str, dry_run: bool = False) -> DownloadResultStub:
        kwargs: dict[str, str | bool] = {"url": url, "quality": quality, "dry_run": dry_run}
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Keeps fake parsed outputs available after context exit."""
        return


class ThreadsDownloaderStub:
    def __init__(self, *, results: list[ThreadsOutput] | BaseException) -> None:
        self.results = results

    def parse(self, url: str) -> ParseResultStub:
        return ParseResultStub(results=self.results)


class FakeSendChannel:
    def __init__(self, sent: list[str]) -> None:
        self.sent = sent

    async def send(self, *, content: str) -> None:
        self.sent.append(content)


class FakeAuditEntry:
    def __init__(self, *, target_id: int, user: FakeUser, reason: str) -> None:
        self.target = SimpleNamespace(id=target_id)
        self.changes = SimpleNamespace(after=SimpleNamespace(communication_disabled_until=True))
        self.user = user
        self.reason = reason


class FakeGeneratedResponse:
    def __init__(self, *, output_text: str) -> None:
        self.output_text = output_text


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
    success_content = interaction.followup.sent[0]["content"]
    assert isinstance(success_content, str)
    assert success_content.startswith("✅ 下載成功")
    assert interaction.edits[-1]["content"] == "✅"

    downloader = DownloaderStub(
        results=[DownloadResultStub(filename=big), DownloadResultStub(filename=low)]
    )
    monkeypatch.setattr(video, "VideoDownloader", lambda output_folder: downloader)
    retry_interaction = FakeInteraction()
    await VideoCogs.download_video.callback(
        cog, retry_interaction, url="https://x.test", quality="best"
    )
    assert [call["quality"] for call in downloader.calls] == ["best", "low"]
    assert retry_interaction.followup.sent[-1]["file"] is not None

    fail_interaction = FakeInteraction()
    monkeypatch.setattr(
        video,
        "VideoDownloader",
        lambda output_folder: DownloaderStub(results=[DownloadResultStub(filename=big)]),
    )
    await VideoCogs.download_video.callback(
        cog, fail_interaction, url="https://x.test", quality="low"
    )
    assert "檔案大小超過" in fail_interaction.edits[-1]["content"]

    monkeypatch.setattr(video, "VideoDownloader", lambda output_folder: _RaiseDownloader())
    error_interaction = FakeInteraction()
    await VideoCogs.download_video.callback(
        cog, error_interaction, url="https://x.test", quality="best"
    )
    assert "檔案無法下載" in error_interaction.edits[-1]["content"]


class _RaiseDownloader:
    def download(self, *, url: str, quality: str, dry_run: bool = False) -> DownloadResultStub:
        raise RuntimeError("download failed")


async def test_threads_cog_builds_embeds_and_handles_messages(tmp_path: Path) -> None:
    bot = SimpleNamespace(user=SimpleNamespace(id=999))
    cog = ThreadsCogs(bot=bot)
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(data=b"123")

    parent = _thread_output(text="parent", video_urls=["https://example.test/video.mp4"])
    target = _thread_output(
        image_urls=["https://example.test/1.png", "https://example.test/2.png"]
    )
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
    channel = FakeSendChannel(sent=sent)
    bot_user = FakeUser(user_id=999, name="bot", display_name="Bot")
    bot = SimpleNamespace(user=bot_user)
    cog = AutoUnmuteCogs(bot=bot)

    guild = SimpleNamespace(
        id=123,
        name="Guild",
        get_channel=lambda channel_id: channel,
        system_channel=None,
        audit_logs=lambda action, limit: _audit_entries(bot_user),
    )
    await cog.on_message(
        message=SimpleNamespace(
            guild=guild, author=FakeUser(bot=False), channel=SimpleNamespace(id=456)
        )
    )
    assert cog._last_active_channel == {123: 456}

    monkeypatch.setattr(auto_unmute.nextcord.abc, "Messageable", FakeSendChannel)
    assert cog._resolve_channel(guild=guild) is channel
    moderator, reason = await cog._lookup_audit(guild=guild)
    assert moderator.name == "moderator"
    assert reason == "testing"

    cog.__dict__["client"] = SimpleNamespace(
        responses=SimpleNamespace(create=_create_auto_unmute_response)
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
    handled: list[SelfTimeoutCall] = []

    async def record_self_timeout(*, member: SimpleNamespace, until: datetime) -> None:
        handled.append({"member": member, "until": until})

    monkeypatch.setattr(cog, "_handle_self_timeout", record_self_timeout)
    await cog.on_member_update(before=before, after=after)
    assert handled


async def _audit_entries(bot_user: FakeUser) -> AsyncIterator[FakeAuditEntry]:
    yield FakeAuditEntry(target_id=111, user=FakeUser(name="wrong"), reason="wrong")
    yield FakeAuditEntry(target_id=bot_user.id, user=FakeUser(name="moderator"), reason="testing")


async def _create_auto_unmute_response(  # noqa: PLR0913 -- mirrors Responses API call shape
    model: str,
    instructions: str,
    input: list[dict[str, str]],  # noqa: A002 -- OpenAI SDK parameter name
    reasoning: dict[str, str],
    service_tier: str,
    extra_headers: dict[str, str],
    extra_body: dict[str, bool],
) -> FakeGeneratedResponse:
    return FakeGeneratedResponse(output_text="not today")


async def _append_async[T](container: list[T], item: T) -> None:
    container.append(item)


async def _async_none() -> None:
    return None


async def test_economy_commands_use_database_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduled: list[FakeDiscordMessage] = []

    def record_scheduled(*, message: FakeDiscordMessage) -> None:
        scheduled.append(message)

    monkeypatch.setattr(economy, "schedule_game_message_delete", record_scheduled)
    monkeypatch.setattr(economy, "get_balance", fake_get_balance)
    monkeypatch.setattr(economy, "top_n", fake_top_n)
    monkeypatch.setattr(economy, "get_account", fake_get_account)
    monkeypatch.setattr(economy, "transfer", fake_transfer)
    monkeypatch.setattr(economy, "get_loan_view", fake_get_loan_view)
    bot = SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))
    cog = EconomyCogs(bot=bot)
    interaction = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.balance.callback(cog, interaction)
    await EconomyCogs.leaderboard.callback(cog, interaction)
    await EconomyCogs.house.callback(cog, interaction)
    await EconomyCogs.give.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount=100
    )
    assert len(interaction.followup.sent) == 4
    assert scheduled

    bot_receiver = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.give.callback(
        cog, bot_receiver, member=FakeUser(user_id=3, name="bot", bot=True), amount=1
    )
    assert "不能把" in bot_receiver.followup.sent[0]["embed"].description


async def fake_get_balance(user_id: int) -> int:
    return 150


async def fake_get_loan_view(*, user_id: int) -> None:
    return None


async def fake_top_n(
    limit: int, exclude_user_ids: tuple[int, ...] = ()
) -> list[tuple[int, str, int]]:
    return [(1, "alice", 150)]


async def fake_get_account(user_id: int) -> tuple[str, int, int, int]:
    return ("Bot", -50, 100, 150)


async def fake_transfer(
    sender_id: int, sender_name: str, receiver_id: int, receiver_name: str, amount: int
) -> database.TransferResult:
    return database.TransferResult(sender_balance=50, receiver_balance=100)


async def fake_sleep(delay: float) -> None:
    return


def ignore_scheduled_game_message(message: FakeDiscordMessage) -> None:
    return


async def fake_place_bet(user_id: int, name: str, requested_bet: int) -> database.PlacedBet:
    return database.PlacedBet(amount=10, balance_after=90, is_allin=False)


async def fake_zero_balance(user_id: int) -> int:
    return 0


async def fake_settle_wager(  # noqa: PLR0913 -- mirrors settlement helper signature
    player_id: int,
    player_account_name: str,
    dealer_id: int,
    dealer_name: str,
    bet: int,
    delta: int,
) -> SimpleNamespace:
    return SimpleNamespace(delta=10, new_balance=110, house_balance=-10)


async def fake_settle_blackjack_round(
    hand: BlackjackHand, player_id: int, player_account_name: str, dealer_id: int, dealer_name: str
) -> SimpleNamespace:
    return SimpleNamespace(
        outcome="win", delta=15, new_balance=115, house_balance=-15, detail="natural"
    )


class FakeDealer:
    async def taunt_bet(
        self, author_name: str, player_name: str, balance_after_bet: int, bet: int, game: str
    ) -> str:
        return "taunt"

    async def settle(  # noqa: PLR0913 -- mirrors DealerAI.settle signature
        self,
        author_name: str,
        player_name: str,
        outcome: str,
        bet: int,
        delta: int,
        new_balance: int,
        game: str,
        detail: str,
    ) -> str:
        return "settled"


async def test_games_commands_run_with_patched_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    monkeypatch.setattr(games.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(games, "schedule_game_message_delete", ignore_scheduled_game_message)
    monkeypatch.setattr(games, "place_bet", fake_place_bet)
    monkeypatch.setattr(games, "get_balance", fake_zero_balance)
    monkeypatch.setattr(games, "settle_wager", fake_settle_wager)
    monkeypatch.setattr(games, "settle_blackjack_round", fake_settle_blackjack_round)
    hand = BlackjackHand(rng=games.SystemRandom(), bet=10)
    hand.player = [Card(rank="A", suit="♠"), Card(rank="K", suit="♣")]
    hand.dealer = [Card(rank="9", suit="♠"), Card(rank="8", suit="♣")]
    hand.finished = True

    monkeypatch.setattr(target=BlackjackHand, name="deal_initial", value=lambda self: None)

    def fake_blackjack_hand(rng: games.SystemRandom, bet: int) -> BlackjackHand:
        return hand

    monkeypatch.setattr(games, "BlackjackHand", fake_blackjack_hand)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["dealer"] = FakeDealer()

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
    added: list[
        tuple[
            VideoCogs | GamesCogs | EconomyCogs | TemplateCogs | ThreadsCogs | AutoUnmuteCogs,
            bool | None,
        ]
    ] = []

    def record_cog(
        cog: VideoCogs | GamesCogs | EconomyCogs | TemplateCogs | ThreadsCogs | AutoUnmuteCogs,
        override: bool | None = None,
    ) -> None:
        added.append((cog, override))

    bot = SimpleNamespace(add_cog=record_cog)
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
    loaded: list[tuple[list[str], bool]] = []

    def record_load_extensions(modules: list[str], stop_at_error: bool) -> None:
        loaded.append((modules, stop_at_error))

    bot = SimpleNamespace(load_extensions=record_load_extensions)
    cli.DiscordBot._load_cogs_sync(bot)
    assert loaded[0][1] is True
    assert "discordbot.cogs.template" in loaded[0][0]


async def test_cli_message_and_command_error_branches() -> None:
    processed: list[SimpleNamespace] = []

    async def record_processed(message: SimpleNamespace) -> None:
        processed.append(message)

    bot = SimpleNamespace(user=FakeUser(user_id=999, bot=True), process_commands=record_processed)
    user_message = SimpleNamespace(author=FakeUser(user_id=1, bot=False))
    await cli.DiscordBot.on_message(bot, message=user_message)
    assert processed == [user_message]
    await cli.DiscordBot.on_message(bot, message=SimpleNamespace(author=bot.user))
    assert len(processed) == 1

    sent: list[DiscordPayload] = []

    async def record_context_send(**kwargs: Unpack[DiscordPayload]) -> None:
        sent.append(kwargs)

    context = SimpleNamespace(
        send=record_context_send,
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
