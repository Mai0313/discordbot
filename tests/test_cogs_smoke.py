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
from discordbot.cogs._games.views import BlackjackLobbyView
from discordbot.cogs.parse_threads import ThreadsCogs
from discordbot.cogs._games.dragon_gate_views import DragonGateLobbyView

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
        self.deferred_ephemeral = False
        self.sent: list[DiscordPayload] = []

    async def defer(self, *, ephemeral: bool = False) -> None:
        self.deferred = True
        self.deferred_ephemeral = ephemeral

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
        self.message: FakeDiscordMessage | None = None
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
    monkeypatch.setattr(economy, "get_vip", fake_get_vip)
    monkeypatch.setattr(economy, "top_n", fake_top_n)
    monkeypatch.setattr(economy, "top_losers", fake_top_losers)
    monkeypatch.setattr(economy, "get_account", fake_get_account)
    monkeypatch.setattr(economy, "transfer", fake_transfer)
    monkeypatch.setattr(economy, "get_loan_view", fake_get_loan_view)
    monkeypatch.setattr(economy, "checkin", fake_checkin)
    monkeypatch.setattr(economy, "buy_vip", fake_buy_vip)
    bot = SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))
    cog = EconomyCogs(bot=bot)
    interaction = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.balance.callback(cog, interaction)
    await EconomyCogs.leaderboard.callback(cog, interaction)
    await EconomyCogs.loss_leaderboard.callback(cog, interaction)
    await EconomyCogs.house.callback(cog, interaction)
    await EconomyCogs.give.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount=100
    )
    await EconomyCogs.checkin_command.callback(cog, interaction)
    await EconomyCogs.vip_command.callback(cog, interaction)
    assert len(interaction.followup.sent) == 7
    assert scheduled
    # The /checkin reply must be ephemeral so only the caller sees it.
    assert interaction.followup.sent[5].get("ephemeral") is True

    bot_receiver = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.give.callback(
        cog, bot_receiver, member=FakeUser(user_id=3, name="bot", bot=True), amount=1
    )
    assert "不能" in bot_receiver.followup.sent[0]["embed"].description


async def fake_get_balance(user_id: int) -> int:
    return 150


async def fake_get_loan_view(*, user_id: int) -> None:
    return None


async def fake_get_vip(user_id: int) -> bool:
    return False


async def fake_top_n(
    limit: int, exclude_user_ids: tuple[int, ...] = ()
) -> list[tuple[int, str, int, str]]:
    return [(1, "alice", 150, "https://cdn.example/alice.png")]


async def fake_top_losers(
    limit: int, exclude_user_ids: tuple[int, ...] = ()
) -> list[tuple[int, str, int, str]]:
    return [(1, "alice", 500, "https://cdn.example/alice.png")]


async def fake_get_account(user_id: int) -> tuple[str, int, int, int]:
    return ("Bot", -50, 100, 150)


async def fake_transfer(  # noqa: PLR0913 -- mirrors database.transfer signature
    sender_id: int,
    sender_name: str,
    sender_avatar_url: str,
    receiver_id: int,
    receiver_name: str,
    receiver_avatar_url: str,
    amount: int,
) -> database.TransferResult:
    return database.TransferResult(sender_balance=50, receiver_balance=100)


async def fake_checkin(*, user_id: int, name: str, avatar_url: str) -> database.CheckinResult:
    return database.CheckinResult(new_balance=600_000, amount=150_000, streak=2, is_vip=False)


async def fake_buy_vip(*, user_id: int, name: str, avatar_url: str) -> database.VipPurchaseResult:
    return database.VipPurchaseResult(new_balance=500_000, cost=database.VIP_PURCHASE_COST)


def ignore_scheduled_game_message(message: FakeDiscordMessage) -> None:
    return


async def fake_game_balance(user_id: int) -> int:
    return 100


class FakeDealer:
    async def taunt_bet(
        self, author_name: str, player_name: str, balance_at_start: int, bet: int, game: str
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
    monkeypatch.setattr(games, "schedule_game_message_delete", ignore_scheduled_game_message)
    monkeypatch.setattr(games, "get_balance", fake_game_balance)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["dealer"] = FakeDealer()

    blackjack_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, blackjack_interaction, bet=10)
    assert blackjack_interaction.followup.sent[0]["wait"] is True
    assert isinstance(blackjack_interaction.followup.sent[0]["view"], BlackjackLobbyView)

    dragon_gate_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, dragon_gate_interaction, ante=10)
    assert dragon_gate_interaction.followup.sent[0]["wait"] is True
    assert isinstance(dragon_gate_interaction.followup.sent[0]["view"], DragonGateLobbyView)


async def test_blackjack_lobby_start_is_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    monkeypatch.setattr(games, "get_balance", fake_game_balance)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["dealer"] = FakeDealer()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet=10)
    lobby_view = owner_interaction.followup.sent[0]["view"]
    assert isinstance(lobby_view, BlackjackLobbyView)

    start_button = next(
        child for child in lobby_view.children if getattr(child, "label", "") == "開始"
    )
    other_interaction = FakeInteraction(user=FakeUser(user_id=2, name="bob", display_name="Bob"))
    await start_button.callback(other_interaction)

    assert other_interaction.followup.sent[0]["content"] == "只有發起者可以開始"


async def test_blackjack_owner_all_in_sets_table_bet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")

    async def balance_by_user(user_id: int) -> int:
        return {1: 300, 2: 50_000_000}[user_id]

    monkeypatch.setattr(games, "get_balance", balance_by_user)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["dealer"] = FakeDealer()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet=1_000_000)
    lobby_view = owner_interaction.followup.sent[0]["view"]
    assert isinstance(lobby_view, BlackjackLobbyView)
    assert lobby_view.requested_bet == 300
    assert lobby_view.participants[0].bet == 300
    assert lobby_view.participants[0].is_allin is True

    join_button = next(
        child for child in lobby_view.children if getattr(child, "label", "") == "加入"
    )
    join_interaction = FakeInteraction(user=FakeUser(user_id=2, name="bob", display_name="Bob"))
    join_interaction.message = FakeDiscordMessage()
    await join_button.callback(join_interaction)

    bob = lobby_view.participants[1]
    assert bob.display_name == "Bob"
    assert bob.bet == 300
    assert bob.balance_at_start == 50_000_000
    assert bob.is_allin is False


async def test_dragon_gate_lobby_start_is_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    monkeypatch.setattr(games, "get_balance", fake_game_balance)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["dealer"] = FakeDealer()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, owner_interaction, ante=10)
    lobby_view = owner_interaction.followup.sent[0]["view"]
    assert isinstance(lobby_view, DragonGateLobbyView)

    start_button = next(
        child for child in lobby_view.children if getattr(child, "label", "") == "開始"
    )
    other_interaction = FakeInteraction(user=FakeUser(user_id=2, name="bob", display_name="Bob"))
    await start_button.callback(other_interaction)

    assert other_interaction.followup.sent[0]["content"] == "只有房主可以開始"


async def test_games_on_ready_cleans_stale_messages_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    bot = SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))
    calls: list[SimpleNamespace] = []

    async def record_cleanup(*, bot: SimpleNamespace) -> None:
        calls.append(bot)

    monkeypatch.setattr(games, "delete_tracked_game_messages", record_cleanup)
    cog = GamesCogs(bot=bot)

    await cog.on_ready()
    await cog.on_ready()

    assert calls == [bot]


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


async def test_cli_message_and_command_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    processed: list[SimpleNamespace] = []
    rewards: list[dict[str, object]] = []

    async def record_processed(message: SimpleNamespace) -> None:
        processed.append(message)

    async def record_reward(**kwargs: object) -> database.CreditResult:
        rewards.append(kwargs)
        return database.CreditResult(
            new_balance=5_000, credited_amount=5_000, principal_repaid=0, remaining_debt=0
        )

    monkeypatch.setattr(target=cli, name="credit_with_repayment", value=record_reward)
    bot = SimpleNamespace(
        user=FakeUser(user_id=999, bot=True),
        process_commands=record_processed,
        _award_base_message_points=cli.DiscordBot._award_base_message_points,
    )
    user_message = SimpleNamespace(author=FakeUser(user_id=1, bot=False))
    await cli.DiscordBot.on_message(bot, message=user_message)
    assert processed == [user_message]
    assert rewards[0]["amount"] == cli.BASE_MESSAGE_REWARD_AMOUNT
    assert rewards[0]["kind"] == cli.TransactionKind.MESSAGE_REWARD
    await cli.DiscordBot.on_message(bot, message=SimpleNamespace(author=bot.user))
    assert len(processed) == 1
    assert len(rewards) == 1

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
