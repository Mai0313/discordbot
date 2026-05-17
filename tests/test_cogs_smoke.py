"""Smoke tests for cogs, setup hooks, and high-level Discord command branches."""

from __future__ import annotations

from types import TracebackType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, Unpack, TypedDict, cast
import asyncio
from datetime import UTC, datetime, timedelta

import nextcord
from nextcord import File, Embed
from nextcord.ext import commands

from discordbot import cli
from discordbot.cogs import games, video, economy, template, auto_unmute, parse_threads
from discordbot.cogs.games import GamesCogs
from discordbot.cogs.video import VideoCogs
from discordbot.cogs.economy import EconomyCogs
from discordbot.cogs.template import TemplateCogs
from discordbot.typings.games import BlackjackDealerDecision
from discordbot.utils.threads import ThreadsOutput
from discordbot.typings.models import ModelSettings
from discordbot.cogs.auto_unmute import AutoUnmuteCogs
from discordbot.cogs._games.dealer import DealerAI
from discordbot.cogs.parse_threads import ThreadsCogs
from discordbot.cogs._economy.database import (
    VIP_PURCHASE_COST,
    RepayResult,
    BorrowResult,
    CreditResult,
    CheckinResult,
    TransferResult,
    VipPurchaseResult,
    BalanceAdjustmentResult,
)
from discordbot.cogs._games.blackjack_views import BlackjackLobbyView
from discordbot.cogs._games.dragon_gate_views import DragonGateLobbyView

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import AsyncIterator

    from openai import AsyncOpenAI
    import pytest


class DiscordPayload(TypedDict, total=False):
    """Payload captured from fake Discord message and followup sends."""

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
    """Payload captured from fake original interaction edits."""

    content: str


class SelfTimeoutCall(TypedDict):
    """Recorded auto-unmute timeout handling call."""

    member: SimpleNamespace
    until: datetime


class FakeResponse:
    """Minimal interaction response stub that records sends and deferral."""

    def __init__(self) -> None:
        """Initializes response state records."""
        self.deferred = False
        self.deferred_ephemeral = False
        self.sent: list[DiscordPayload] = []

    async def defer(self, ephemeral: bool = False) -> None:
        """Records that the interaction response was deferred."""
        self.deferred = True
        self.deferred_ephemeral = ephemeral

    async def send_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an interaction response message."""
        self.sent.append(kwargs)

    def is_done(self) -> bool:
        """Returns whether the fake response has already been used."""
        return self.deferred or bool(self.sent)


class FakeFollowup:
    """Minimal interaction followup stub."""

    def __init__(self) -> None:
        """Initializes recorded followup sends."""
        self.sent: list[DiscordPayload] = []

    async def send(self, **kwargs: Unpack[DiscordPayload]) -> FakeDiscordMessage:
        """Records the followup payload and returns a fake message."""
        self.sent.append(kwargs)
        return FakeDiscordMessage()


class FakeInteraction:
    """Minimal interaction stub shared by cog command tests."""

    def __init__(self, user: FakeUser | None = None) -> None:
        """Initializes user, response, followup, and edit records."""
        self.user = user or FakeUser()
        self.message: FakeDiscordMessage | None = None
        self.response = FakeResponse()
        self.followup = FakeFollowup()
        self.edits: list[OriginalEditPayload] = []

    async def edit_original_message(self, **kwargs: Unpack[OriginalEditPayload]) -> None:
        """Records an edit to the deferred original response."""
        self.edits.append(kwargs)


class FakeUser:
    """Minimal Discord user/member stub."""

    def __init__(
        self, user_id: int = 1, name: str = "alice", display_name: str = "Alice", bot: bool = False
    ) -> None:
        """Initializes identity, avatar, bot flag, and account age fields."""
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
    """Minimal Discord message stub that records mutations."""

    def __init__(self) -> None:
        """Initializes message mutation records."""
        self.edits: list[DiscordPayload] = []
        self.reactions: list[str] = []
        self.removed: list[tuple[str, FakeUser]] = []
        self.replies: list[DiscordPayload] = []
        self.deleted = False
        self.suppressed = False

    async def edit(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an edit payload and suppress flag."""
        if "suppress" in kwargs:
            self.suppressed = bool(kwargs["suppress"])
        self.edits.append(kwargs)

    async def add_reaction(self, emoji: str) -> None:
        """Records an added reaction."""
        self.reactions.append(emoji)

    async def remove_reaction(self, emoji: str, member: FakeUser) -> None:
        """Records a removed reaction."""
        self.removed.append((emoji, member))

    async def reply(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records a message reply payload."""
        self.replies.append(kwargs)

    async def delete(self) -> None:
        """Records message deletion."""
        self.deleted = True


class HangingResponses:
    """Fake Responses API resource that sleeps longer than dealer timeout."""

    async def create(self, **_kwargs: Any) -> SimpleNamespace:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Sleeps before returning a late response."""
        await asyncio.sleep(delay=10)
        return SimpleNamespace(output_text="late")

    async def parse(self, **_kwargs: Any) -> SimpleNamespace:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Sleeps before returning a late parsed response."""
        await asyncio.sleep(delay=10)
        return SimpleNamespace(output_parsed=None)


class HangingClient:
    """Fake OpenAI client containing the hanging Responses resource."""

    def __init__(self) -> None:
        """Initializes the hanging responses resource."""
        self.responses = HangingResponses()


class ParsedDecisionResponses:
    """Fake Responses API resource for Blackjack dealer decision parsing."""

    def __init__(self, output_parsed: BlackjackDealerDecision | None) -> None:
        """Stores the parsed output to return."""
        self.output_parsed = output_parsed
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Records parse arguments and returns the configured parsed output."""
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.output_parsed)


class DownloadResultStub:
    """Context manager stub for a downloaded video file."""

    def __init__(self, filename: Path) -> None:
        """Stores the fake downloaded filename."""
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
    """Fake downloader that returns queued download results."""

    def __init__(self, results: list[DownloadResultStub]) -> None:
        """Initializes queued results and recorded calls."""
        self.results = results
        self.calls: list[dict[str, str | bool]] = []

    def download(self, url: str, quality: str, dry_run: bool = False) -> DownloadResultStub:
        """Records the download request and returns the next queued result."""
        kwargs: dict[str, str | bool] = {"url": url, "quality": quality, "dry_run": dry_run}
        self.calls.append(kwargs)
        return self.results.pop(0)


class ParseResultStub:
    """Context manager stub for Threads parse results."""

    def __init__(self, results: list[ThreadsOutput] | BaseException) -> None:
        """Stores parsed results or the error to raise on entry."""
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
    """Fake Threads downloader returning a configured parse context manager."""

    def __init__(self, results: list[ThreadsOutput] | BaseException) -> None:
        """Stores parsed results or parse failure."""
        self.results = results

    def parse(self, url: str) -> ParseResultStub:
        """Returns a fake parse context manager."""
        return ParseResultStub(results=self.results)


class FakeSendChannel:
    """Minimal messageable channel stub."""

    def __init__(self, sent: list[str]) -> None:
        """Stores the shared sent-message list."""
        self.sent = sent

    async def send(self, content: str) -> None:
        """Records sent content."""
        self.sent.append(content)


class FakeAuditEntry:
    """Minimal audit log entry for timeout lookup tests."""

    def __init__(self, target_id: int, user: FakeUser, reason: str) -> None:
        """Initializes target, changed field, moderator, and reason."""
        self.target = SimpleNamespace(id=target_id)
        self.changes = SimpleNamespace(after=SimpleNamespace(communication_disabled_until=True))
        self.user = user
        self.reason = reason


class FakeGeneratedResponse:
    """Fake non-streaming Responses API result."""

    def __init__(self, output_text: str) -> None:
        """Stores generated output text."""
        self.output_text = output_text


def _thread_output(
    text: str = "hello",
    image_urls: list[str] | None = None,
    video_paths: list[Path] | None = None,
    video_urls: list[str] | None = None,
) -> ThreadsOutput:
    """Builds a parsed Threads output fixture."""
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
    """Verifies template debug reaction and ping command response."""
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
    """Verifies video delivery, retry, oversize, and download error branches."""
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
    """Downloader stub that always fails."""

    def download(self, url: str, quality: str, dry_run: bool = False) -> DownloadResultStub:
        """Raises a deterministic download failure."""
        raise RuntimeError("download failed")


async def test_threads_cog_builds_embeds_and_handles_messages(tmp_path: Path) -> None:
    """Verifies Threads embed building and on_message success/warning/error paths."""
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
    """Verifies auto-unmute audit lookup, reply generation, and member update handling."""
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

    async def record_self_timeout(member: SimpleNamespace, until: datetime) -> None:
        """Records the self-timeout callback arguments."""
        handled.append({"member": member, "until": until})

    monkeypatch.setattr(cog, "_handle_self_timeout", record_self_timeout)
    await cog.on_member_update(before=before, after=after)
    assert handled


async def _audit_entries(bot_user: FakeUser) -> AsyncIterator[FakeAuditEntry]:
    """Yields unrelated and matching audit entries for lookup filtering."""
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
    """Returns a deterministic auto-unmute response."""
    return FakeGeneratedResponse(output_text="not today")


async def _append_async[T](container: list[T], item: T) -> None:
    """Appends an item through an awaitable callback."""
    container.append(item)


async def _async_none() -> None:
    """Async no-op used by fake callbacks."""


async def test_economy_commands_use_database_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies economy slash commands call the database facade and send embeds."""
    scheduled: list[FakeDiscordMessage] = []

    def record_scheduled(message: FakeDiscordMessage) -> None:
        """Records game cleanup scheduling from economy commands."""
        scheduled.append(message)

    monkeypatch.setattr(economy, "schedule_game_message_delete", record_scheduled)
    monkeypatch.setattr(economy, "get_balance", fake_get_balance)
    monkeypatch.setattr(economy, "get_vip", fake_get_vip)
    monkeypatch.setattr(economy, "get_admin", fake_get_admin)
    monkeypatch.setattr(economy, "top_n", fake_top_n)
    monkeypatch.setattr(economy, "top_losers", fake_top_losers)
    monkeypatch.setattr(economy, "get_account", fake_get_account)
    monkeypatch.setattr(economy, "transfer", fake_transfer)
    monkeypatch.setattr(economy, "adjust_balance", fake_adjust_balance)
    monkeypatch.setattr(economy, "get_loan_view", fake_get_loan_view)
    monkeypatch.setattr(economy, "borrow", fake_borrow)
    monkeypatch.setattr(economy, "repay", fake_repay)
    monkeypatch.setattr(economy, "checkin", fake_checkin)
    monkeypatch.setattr(economy, "buy_vip", fake_buy_vip)
    bot = SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))
    cog = EconomyCogs(bot=bot)
    interaction = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.balance.callback(cog, interaction)
    await EconomyCogs.leaderboard.callback(cog, interaction)
    await EconomyCogs.loss_leaderboard.callback(cog, interaction)
    await EconomyCogs.house.callback(cog, interaction)
    await EconomyCogs.admin_refund_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount=100
    )
    await EconomyCogs.admin_collect_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount=50
    )
    await EconomyCogs.give.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount=100
    )
    await EconomyCogs.borrow_loan.callback(cog, interaction, amount=100)
    await EconomyCogs.repay_loan.callback(cog, interaction, amount=50)
    await EconomyCogs.checkin_command.callback(cog, interaction)
    await EconomyCogs.vip_command.callback(cog, interaction)
    assert len(interaction.followup.sent) == 11
    assert len(scheduled) == 6
    assert interaction.followup.sent[0].get("ephemeral") is True
    assert interaction.followup.sent[4].get("ephemeral") is not True
    assert interaction.followup.sent[5].get("ephemeral") is not True
    assert interaction.followup.sent[6].get("ephemeral") is not True
    assert interaction.followup.sent[7].get("ephemeral") is True
    assert interaction.followup.sent[8].get("ephemeral") is True
    assert interaction.followup.sent[9].get("ephemeral") is True
    assert interaction.followup.sent[10].get("ephemeral") is True

    bot_receiver = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.give.callback(
        cog, bot_receiver, member=FakeUser(user_id=3, name="bot", bot=True), amount=1
    )
    assert "不能" in bot_receiver.followup.sent[0]["embed"].description


async def test_economy_admin_rejects_non_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin economy commands must check the DB admin flag before mutating balance."""
    called = False

    async def fake_get_admin_false(user_id: int) -> bool:
        """Returns a non-admin status."""
        return False

    async def fake_adjust_balance_guard(**_kwargs: Any) -> BalanceAdjustmentResult:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Fails the test if a non-admin reaches the mutation path."""
        nonlocal called
        called = True
        return BalanceAdjustmentResult(new_balance=0, applied_delta=0)

    monkeypatch.setattr(economy, "get_admin", fake_get_admin_false)
    monkeypatch.setattr(economy, "adjust_balance", fake_adjust_balance_guard)
    cog = EconomyCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.admin_refund_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount=100
    )

    assert called is False
    assert interaction.followup.sent[0].get("ephemeral") is True
    assert "權限不足" in interaction.followup.sent[0]["embed"].title


async def fake_get_balance(user_id: int) -> int:
    """Returns a stable fake balance."""
    return 150


async def fake_get_loan_view(user_id: int) -> None:
    """Returns no active loan state."""


async def fake_get_vip(user_id: int) -> bool:
    """Returns non-VIP status."""
    return False


async def fake_get_admin(user_id: int) -> bool:
    """Returns economy admin status."""
    return True


async def fake_top_n(
    limit: int, exclude_user_ids: tuple[int, ...] = ()
) -> list[tuple[int, str, int, str]]:
    """Returns one fake leaderboard row."""
    return [(1, "alice", 150, "https://cdn.example/alice.png")]


async def fake_top_losers(
    limit: int, exclude_user_ids: tuple[int, ...] = ()
) -> list[tuple[int, str, int, str]]:
    """Returns one fake loss leaderboard row."""
    return [(1, "alice", 500, "https://cdn.example/alice.png")]


async def fake_get_account(user_id: int) -> tuple[str, int, int, int]:
    """Returns a fake house ledger account."""
    return ("Bot", -50, 100, 150)


async def fake_transfer(  # noqa: PLR0913 -- mirrors transfer signature
    sender_id: int,
    sender_name: str,
    receiver_id: int,
    receiver_name: str,
    amount: int,
    sender_avatar_url: str = "",
    receiver_avatar_url: str = "",
) -> TransferResult | None:
    """Returns a successful fake transfer result."""
    return TransferResult(sender_balance=50, receiver_balance=100)


async def fake_adjust_balance(  # noqa: PLR0913 -- mirrors adjust_balance signature
    user_id: int,
    name: str,
    delta: int,
    allow_negative: bool = False,
    avatar_url: str = "",
    note: str | None = None,
) -> BalanceAdjustmentResult:
    """Returns a successful fake manual adjustment result."""
    return BalanceAdjustmentResult(new_balance=150 + delta, applied_delta=delta)


async def fake_borrow(
    user_id: int, name: str, amount: int, credit_limit_value: int, avatar_url: str = ""
) -> BorrowResult:
    """Returns a successful fake borrow result."""
    return BorrowResult(new_balance=250, principal=amount, borrowed_amount=amount)


async def fake_repay(user_id: int, name: str, amount: int, avatar_url: str = "") -> RepayResult:
    """Returns a successful fake repay result."""
    return RepayResult(new_balance=100, principal_repaid=amount, remaining_debt=0)


async def fake_checkin(user_id: int, name: str, avatar_url: str) -> CheckinResult:
    """Returns a successful fake daily check-in result."""
    return CheckinResult(new_balance=600_000, amount=150_000, streak=2, is_vip=False)


async def fake_buy_vip(user_id: int, name: str, avatar_url: str) -> VipPurchaseResult:
    """Returns a successful fake VIP purchase result."""
    return VipPurchaseResult(new_balance=500_000, cost=VIP_PURCHASE_COST)


def ignore_scheduled_game_message(message: FakeDiscordMessage) -> None:
    """Ignores cleanup scheduling in command smoke tests."""
    return


async def fake_game_balance(user_id: int) -> int:
    """Returns a small fake game balance."""
    return 100


async def _wealthy_game_balance(user_id: int) -> int:
    """Returns a fake balance large enough for Dragon Gate ante."""
    return 1_000_000


async def fake_dragon_gate_jackpot_snapshot() -> tuple[int, int]:
    """Returns a stable fake Dragon Gate jackpot snapshot."""
    return 100_000, 0


class FakeDealer:
    """Fake casino dealer that returns deterministic banter."""

    async def taunt_bet(
        self, author_name: str, player_name: str, balance_at_start: int, bet: int, game: str
    ) -> str:
        """Returns deterministic opening banter."""
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
        """Returns deterministic settlement banter."""
        return "settled"


async def test_games_commands_run_with_patched_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies game commands create lobby views with patched dependencies."""
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

    monkeypatch.setattr(
        games, "fetch_dragon_gate_jackpot_snapshot", fake_dragon_gate_jackpot_snapshot
    )
    monkeypatch.setattr(games, "get_balance", _wealthy_game_balance)
    dragon_gate_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, dragon_gate_interaction)
    assert dragon_gate_interaction.followup.sent[-1]["wait"] is True
    assert isinstance(dragon_gate_interaction.followup.sent[-1]["view"], DragonGateLobbyView)


async def test_blackjack_lobby_start_is_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies only the Blackjack lobby owner can press Start."""
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

    assert other_interaction.response.sent
    assert isinstance(other_interaction.response.sent[0]["content"], str)


async def test_blackjack_owner_all_in_sets_table_bet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies owner all-in clamps the shared Blackjack lobby bet."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")

    async def balance_by_user(user_id: int) -> int:
        """Returns distinct balances for owner and joining player."""
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
    """Verifies only the Dragon Gate lobby owner can press Start."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    monkeypatch.setattr(games, "get_balance", _wealthy_game_balance)
    monkeypatch.setattr(
        games, "fetch_dragon_gate_jackpot_snapshot", fake_dragon_gate_jackpot_snapshot
    )

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["dealer"] = FakeDealer()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, owner_interaction)
    lobby_view = owner_interaction.followup.sent[-1]["view"]
    assert isinstance(lobby_view, DragonGateLobbyView)

    start_button = next(
        child for child in lobby_view.children if getattr(child, "label", "") == "開始"
    )
    other_interaction = FakeInteraction(user=FakeUser(user_id=2, name="bob", display_name="Bob"))
    await start_button.callback(other_interaction)

    assert other_interaction.response.sent
    assert isinstance(other_interaction.response.sent[0]["content"], str)


async def test_dealer_ai_times_out_to_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies DealerAI returns fallback banter when the LLM call times out."""
    monkeypatch.setattr("discordbot.cogs._games.dealer.DEALER_AI_TIMEOUT_SECONDS", 0.01)
    dealer = DealerAI(
        client=cast("AsyncOpenAI", HangingClient()),
        model=ModelSettings(name="gemini-flash-latest", effort="none"),
    )

    line = await dealer.taunt_bet(
        author_name="alice", player_name="Alice", balance_at_start=100, bet=10, game="dragon_gate"
    )

    assert line == "下好離手, 不要等下哭"


async def test_dealer_ai_parses_blackjack_decision() -> None:
    """Verifies Blackjack dealer decisions use Responses API structured parsing."""
    responses = ParsedDecisionResponses(
        output_parsed=BlackjackDealerDecision(action="hit", reason="追過最高玩家")
    )
    client = SimpleNamespace(responses=responses)
    dealer = DealerAI(
        client=cast("AsyncOpenAI", client),
        model=ModelSettings(name="gemini-flash-latest", effort="none"),
    )

    decision = await dealer.decide_blackjack_action(
        author_name="alice", table_state="莊家總點數: 13\n玩家: Alice = 18", dealer_total=13
    )

    assert decision.action == "hit"
    assert responses.calls[0]["text_format"] is BlackjackDealerDecision


async def test_dealer_ai_empty_blackjack_decision_uses_basic_rule() -> None:
    """Verifies an empty parsed decision falls back to deterministic dealer rules."""
    responses = ParsedDecisionResponses(output_parsed=None)
    client = SimpleNamespace(responses=responses)
    dealer = DealerAI(
        client=cast("AsyncOpenAI", client),
        model=ModelSettings(name="gemini-flash-latest", effort="none"),
    )

    decision = await dealer.decide_blackjack_action(
        author_name="alice", table_state="莊家總點數: 18\n玩家: Alice = 17", dealer_total=18
    )

    assert decision == BlackjackDealerDecision(action="stand", reason="basic rule: 已達 17 點")


async def test_dealer_ai_blackjack_decision_times_out_to_basic_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies slow Blackjack decisions use the dedicated decision timeout."""
    monkeypatch.setattr(
        "discordbot.cogs._games.dealer.DEALER_BLACKJACK_DECISION_TIMEOUT_SECONDS", 0.01
    )
    dealer = DealerAI(
        client=cast("AsyncOpenAI", HangingClient()),
        model=ModelSettings(name="gemini-flash-latest", effort="medium"),
    )

    decision = await dealer.decide_blackjack_action(
        author_name="alice", table_state="莊家總點數: 16\n玩家: Alice = 18", dealer_total=16
    )

    assert decision == BlackjackDealerDecision(action="hit", reason="basic rule: 未滿 17 點")


async def test_games_on_ready_cleans_stale_messages_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies startup cleanup runs once per GamesCogs instance."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    bot = SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))
    calls: list[SimpleNamespace] = []

    async def record_cleanup(bot: SimpleNamespace) -> None:
        """Records the bot passed to startup cleanup."""
        calls.append(bot)

    monkeypatch.setattr(games, "delete_tracked_game_messages", record_cleanup)
    cog = GamesCogs(bot=bot)

    await cog.on_ready()
    await cog.on_ready()

    assert calls == [bot]


def test_setup_functions_register_cogs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies every cog setup function registers the expected cog type."""
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
        """Records the cog instance and override flag passed to add_cog."""
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
    """Verifies synchronous cog loading discovers the expected cog modules."""
    loaded: list[tuple[list[str], bool]] = []

    def record_load_extensions(modules: list[str], stop_at_error: bool) -> None:
        """Records modules passed to load_extensions."""
        loaded.append((modules, stop_at_error))

    bot = SimpleNamespace(load_extensions=record_load_extensions)
    cli.DiscordBot._load_cogs_sync(bot)
    assert loaded[0][1] is True
    assert "discordbot.cogs.template" in loaded[0][0]


async def test_cli_message_and_command_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies base message rewards and common command error embeds."""
    processed: list[SimpleNamespace] = []
    rewards: list[dict[str, Any]] = []

    async def record_processed(message: SimpleNamespace) -> None:
        """Records messages passed to process_commands."""
        processed.append(message)

    async def record_reward(**kwargs: Any) -> CreditResult:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Records base reward arguments and returns a fake credit result."""
        rewards.append(kwargs)
        return CreditResult(
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
        """Records command error responses sent through the context."""
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
