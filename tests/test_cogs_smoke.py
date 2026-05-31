"""Smoke tests for cogs, setup hooks, and high-level Discord command branches."""

from __future__ import annotations

from types import TracebackType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, Unpack, TypedDict, cast
import asyncio
from datetime import UTC, datetime, timedelta

import nextcord
from nextcord import File, Embed, AllowedMentions
from nextcord.ext import commands

from discordbot import cli
from discordbot.cogs import games, video, economy, template, auto_unmute, parse_threads
from discordbot.cogs.games import GamesCogs
from discordbot.cogs.video import VideoCogs
from discordbot.cogs.economy import EconomyCogs
from discordbot.cogs._economy import views, interactions
from discordbot.cogs.template import TemplateCogs
from discordbot.typings.games import GameParticipant
from discordbot.typings.stock import StockPortfolioView, StockPortfolioHolding
from discordbot.utils.threads import ThreadsOutput
from discordbot.typings.models import ModelSettings
from discordbot.typings.economy import (
    PortfolioView,
    LoanLenderType,
    AccountSnapshot,
    JackpotSnapshot,
    LeaderboardEntry,
    LoanContractView,
    LoanProposalKind,
    LoanProposalView,
    CentralBankStatus,
    LoanPaymentResult,
    LoanContractStatus,
    LoanProposalStatus,
    CasinoLedgerSnapshot,
    LossLeaderboardEntry,
    LoanProposalAcceptResult,
)
from discordbot.cogs.auto_unmute import AutoUnmuteCogs
from discordbot.cogs._games.dealer import SystemNarrator
from discordbot.cogs._games.wagers import parse_wager_amount
from discordbot.cogs.parse_threads import ThreadsCogs
from discordbot.cogs._economy.views import CreditLoanDecisionView, CentralBankLoanDecisionView
from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME, embed_spacer_url
from discordbot.cogs._games.blackjack import Card
from discordbot.cogs._economy.database import (
    VIP_PURCHASE_COST,
    CreditResult,
    CheckinResult,
    TransferResult,
    VipPurchaseResult,
    BalanceAdjustmentResult,
)
from discordbot.cogs._games.blackjack_views import BlackjackLobbyView
from discordbot.cogs._games.dragon_gate_views import DragonGateLobbyView

TEST_DEALER_MODEL = "test-dealer-llm-model"

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable, Awaitable, AsyncIterator

    from openai import AsyncOpenAI
    import pytest
    from nextcord.ui import View


class DiscordPayload(TypedDict, total=False):
    """Payload captured from fake Discord message and followup sends."""

    content: str | None
    embed: Embed
    embeds: list[Embed]
    file: File
    files: list[File]
    view: View | None
    wait: bool
    ephemeral: bool
    suppress: bool
    allowed_mentions: AllowedMentions


class OriginalEditPayload(TypedDict, total=False):
    """Payload captured from fake original interaction edits."""

    content: str
    file: File
    allowed_mentions: AllowedMentions


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
        self.edited: list[DiscordPayload] = []

    async def defer(self, ephemeral: bool = False) -> None:
        """Records that the interaction response was deferred."""
        self.deferred = True
        self.deferred_ephemeral = ephemeral

    async def send_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an interaction response message."""
        self.sent.append(kwargs)

    async def edit_message(self, **kwargs: Unpack[DiscordPayload]) -> None:
        """Records an interaction response edit."""
        self.edited.append(kwargs)

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
        # /balance displays the snowflake-derived account age; pin it well into
        # the past so freezegun does not make the value surprising.
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
        interaction=interaction,
        file_size_mb=1.25,
        file_path=small,
        url="https://source.test/video",
    )
    success_content = interaction.edits[-1]["content"]
    assert isinstance(success_content, str)
    assert success_content == "-# 檔案大小: 1.2MB\n-# 來源: <https://source.test/video>"
    assert interaction.edits[-1]["file"] is not None
    assert interaction.followup.sent == []

    downloader = DownloaderStub(
        results=[DownloadResultStub(filename=big), DownloadResultStub(filename=low)]
    )
    monkeypatch.setattr(video, "VideoDownloader", lambda output_folder: downloader)
    retry_interaction = FakeInteraction()
    await VideoCogs.download_video.callback(
        cog, retry_interaction, url="https://x.test", quality="best"
    )
    assert [call["quality"] for call in downloader.calls] == ["best", "low"]
    assert retry_interaction.edits[-1]["file"] is not None
    assert "來源: <https://x.test>" in retry_interaction.edits[-1]["content"]
    assert retry_interaction.followup.sent == []

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


async def test_economy_commands_use_database_facade(  # noqa: PLR0915 -- command smoke exercises one facade surface
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies economy slash commands call the database facade and send embeds."""
    scheduled: list[FakeDiscordMessage] = []

    def record_scheduled(
        message: FakeDiscordMessage, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records public cleanup scheduling from economy commands."""
        scheduled.append(message)

    monkeypatch.setattr(interactions, "schedule_public_message_delete", record_scheduled)
    monkeypatch.setattr(economy, "get_balance", fake_get_balance)
    monkeypatch.setattr(economy, "get_vip", fake_get_vip)
    monkeypatch.setattr(economy, "get_admin", fake_get_admin)
    monkeypatch.setattr(economy, "top_n", fake_top_n)
    monkeypatch.setattr(economy, "top_losers", fake_top_losers)
    monkeypatch.setattr(economy, "get_account", fake_get_account)
    monkeypatch.setattr(economy, "get_casino_ledger", fake_get_casino_ledger)
    monkeypatch.setattr(economy, "transfer", fake_transfer)
    monkeypatch.setattr(economy, "adjust_balance", fake_adjust_balance)
    monkeypatch.setattr(economy, "get_portfolio", fake_get_portfolio)
    monkeypatch.setattr(economy, "get_stock_portfolio", fake_get_stock_portfolio)
    monkeypatch.setattr(economy, "create_personal_loan_request", fake_create_loan_request)
    monkeypatch.setattr(economy, "repay_personal_loans", fake_loan_payment)
    monkeypatch.setattr(economy, "call_personal_loans", fake_call_personal_loans)
    monkeypatch.setattr(
        economy, "create_central_bank_loan_request", fake_create_central_bank_request
    )
    monkeypatch.setattr(economy, "get_central_banker", fake_get_central_banker)
    monkeypatch.setattr(economy, "list_loan_contracts", fake_list_loan_contracts)
    monkeypatch.setattr(economy, "get_central_bank_status", fake_get_central_bank_status)
    monkeypatch.setattr(economy, "repay_central_bank_loans", fake_loan_payment)
    monkeypatch.setattr(economy, "call_central_bank_loans", fake_call_central_bank_loans)
    monkeypatch.setattr(economy, "checkin", fake_checkin)
    monkeypatch.setattr(economy, "buy_vip", fake_buy_vip)
    bot = SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))
    cog = EconomyCogs(bot=bot)
    interaction = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.balance.callback(cog, interaction, member=None)
    await EconomyCogs.leaderboard.callback(cog, interaction)
    await EconomyCogs.loss_leaderboard.callback(cog, interaction)
    await EconomyCogs.casino.callback(cog, interaction)
    await EconomyCogs.pocat.callback(cog, interaction)
    await EconomyCogs.admin_refund_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="100"
    )
    await EconomyCogs.admin_collect_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="50"
    )
    await EconomyCogs.give.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="100"
    )
    await EconomyCogs.credit_borrow.callback(
        cog,
        interaction,
        member=FakeUser(user_id=2, name="bob"),
        amount="100",
        monthly_rate_percent=3.0,
    )
    await EconomyCogs.credit_repay.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="50"
    )
    await EconomyCogs.credit_call.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="0"
    )
    await EconomyCogs.credit_status.callback(cog, interaction)
    await EconomyCogs.central_bank_borrow.callback(
        cog, interaction, amount="100", monthly_rate_percent=3.0
    )
    await EconomyCogs.central_bank_repay.callback(cog, interaction, amount="50")
    await EconomyCogs.central_bank_call.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="0"
    )
    await EconomyCogs.central_bank_status.callback(cog, interaction)
    await EconomyCogs.checkin_command.callback(cog, interaction)
    await EconomyCogs.vip_command.callback(cog, interaction)
    assert len(interaction.followup.sent) == 18
    assert len(scheduled) == 12
    assert interaction.followup.sent[0].get("ephemeral") is True
    assert "view" not in interaction.followup.sent[1]
    assert interaction.followup.sent[1]["files"][0].filename == "economy_leaderboard.png"
    assert interaction.followup.sent[2]["files"][0].filename == "economy_loss_leaderboard.png"
    assert "view" not in interaction.followup.sent[3]
    assert "view" not in interaction.followup.sent[4]
    assert interaction.followup.sent[5].get("ephemeral") is not True
    assert interaction.followup.sent[6].get("ephemeral") is not True
    assert interaction.followup.sent[7].get("ephemeral") is not True
    assert interaction.followup.sent[8].get("ephemeral") is not True
    assert interaction.followup.sent[9].get("ephemeral") is not True
    assert interaction.followup.sent[10].get("ephemeral") is not True
    assert interaction.followup.sent[11].get("ephemeral") is True
    assert interaction.followup.sent[13].get("ephemeral") is not True
    assert interaction.followup.sent[14].get("ephemeral") is not True
    assert interaction.followup.sent[15].get("ephemeral") is not True
    assert interaction.followup.sent[-1].get("ephemeral") is True
    balance_embed = interaction.followup.sent[0]["embed"]
    assert balance_embed.title == "💰 財務總覽"
    assert "315 虛擬歡樂豆" in balance_embed.description
    assert any(field.name == "現金" and "`150`" in field.value for field in balance_embed.fields)
    assert any(
        field.name == "債務" and "本金 `30`" in field.value for field in balance_embed.fields
    )
    assert any(
        field.name == "股票淨值" and "`200`" in field.value for field in balance_embed.fields
    )
    assert any(
        field.name == "股票部位" and "BCAT" in field.value for field in balance_embed.fields
    )
    borrow_embed = interaction.followup.sent[8]["embed"]
    assert borrow_embed.footer.text == "貸方可用下方按鈕批准或拒絕，發起者可取消，180 秒後自動拒絕"
    borrow_view = interaction.followup.sent[8]["view"]
    assert isinstance(borrow_view, CreditLoanDecisionView)
    assert borrow_view.message is not None
    central_bank_payload = interaction.followup.sent[12]
    central_bank_view = central_bank_payload["view"]
    assert isinstance(central_bank_view, CentralBankLoanDecisionView)
    assert central_bank_view.message is not None

    inspected_member = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.balance.callback(
        cog, inspected_member, member=FakeUser(user_id=2, name="bob", display_name="Bob")
    )
    assert "Bob" in inspected_member.followup.sent[0]["embed"].description

    bot_receiver = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.give.callback(
        cog, bot_receiver, member=FakeUser(user_id=3, name="bot", bot=True), amount="1"
    )
    assert "轉帳完成" in bot_receiver.followup.sent[0]["embed"].title


async def test_central_bank_decision_buttons_require_banker_and_allow_self_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Central bank request buttons are banker-gated and pass the self-approval flag."""
    captured_accept_kwargs: dict[str, Any] = {}
    captured_cancel_kwargs: dict[str, int] = {}

    async def fake_get_central_banker_for_button(user_id: int) -> bool:
        """Only user 1 is a central banker."""
        return user_id == 1

    async def fake_accept_for_button(**kwargs: Any) -> LoanProposalAcceptResult:  # noqa: ANN401 -- command facade double
        """Records approval arguments and returns a fake accepted proposal."""
        captured_accept_kwargs.update(kwargs)
        return await fake_accept_loan_proposal()

    async def fake_cancel_for_button(proposal_id: int, actor_id: int) -> LoanProposalView:
        """Records cancellation arguments and returns a fake canceled proposal."""
        captured_cancel_kwargs.update({"proposal_id": proposal_id, "actor_id": actor_id})
        return await fake_cancel_loan_proposal(proposal_id=proposal_id, actor_id=actor_id)

    monkeypatch.setattr(views, "get_central_banker", fake_get_central_banker_for_button)
    monkeypatch.setattr(views, "accept_loan_proposal", fake_accept_for_button)
    monkeypatch.setattr(views, "cancel_loan_proposal", fake_cancel_for_button)
    view = CentralBankLoanDecisionView(
        bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")),
        proposal_id=42,
        creator_id=1,
        allow_self_approval=True,
    )
    approve_button = next(
        child
        for child in view.children
        if getattr(child, "custom_id", "") == "central_bank:approve"
    )

    denied = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await approve_button.callback(denied)
    assert denied.response.sent[0]["ephemeral"] is True
    assert captured_accept_kwargs == {}

    allowed = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    await approve_button.callback(allowed)
    assert captured_accept_kwargs["proposal_id"] == 42
    assert captured_accept_kwargs["actor_id"] == 1
    assert captured_accept_kwargs["allow_central_bank_self_approval"] is True
    assert allowed.response.edited[0]["view"] is None

    cancel_view = CentralBankLoanDecisionView(
        bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")),
        proposal_id=43,
        creator_id=1,
    )
    cancel_button = next(
        child
        for child in cancel_view.children
        if getattr(child, "custom_id", "") == "central_bank:cancel"
    )
    denied_cancel = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await cancel_button.callback(denied_cancel)
    assert denied_cancel.response.sent[0]["ephemeral"] is True
    assert captured_cancel_kwargs == {}

    allowed_cancel = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    await cancel_button.callback(allowed_cancel)
    assert captured_cancel_kwargs == {"proposal_id": 43, "actor_id": 1}
    assert allowed_cancel.response.edited[0]["view"] is None


async def test_credit_decision_buttons_gate_lender_and_creator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Personal credit request buttons are lender-gated, while cancel is creator-gated."""
    captured_accept_kwargs: dict[str, Any] = {}
    captured_reject_kwargs: dict[str, int] = {}
    captured_cancel_kwargs: dict[str, int] = {}

    async def fake_accept_for_button(**kwargs: Any) -> LoanProposalAcceptResult:  # noqa: ANN401 -- command facade double
        """Records approval arguments and returns a fake accepted proposal."""
        captured_accept_kwargs.update(kwargs)
        return await fake_accept_loan_proposal()

    async def fake_reject_for_button(proposal_id: int, actor_id: int) -> LoanProposalView:
        """Records rejection arguments and returns a fake rejected proposal."""
        captured_reject_kwargs.update({"proposal_id": proposal_id, "actor_id": actor_id})
        return await fake_reject_loan_proposal(proposal_id=proposal_id, actor_id=actor_id)

    async def fake_cancel_for_button(proposal_id: int, actor_id: int) -> LoanProposalView:
        """Records cancellation arguments and returns a fake canceled proposal."""
        captured_cancel_kwargs.update({"proposal_id": proposal_id, "actor_id": actor_id})
        return await fake_cancel_loan_proposal(proposal_id=proposal_id, actor_id=actor_id)

    monkeypatch.setattr(views, "accept_loan_proposal", fake_accept_for_button)
    monkeypatch.setattr(views, "reject_loan_proposal", fake_reject_for_button)
    monkeypatch.setattr(views, "cancel_loan_proposal", fake_cancel_for_button)
    view = CreditLoanDecisionView(proposal_id=42, lender_id=2, creator_id=1)
    approve_button = next(
        child for child in view.children if getattr(child, "custom_id", "") == "credit:approve"
    )

    denied_approve = FakeInteraction(user=FakeUser(user_id=3, name="charlie"))
    await approve_button.callback(denied_approve)
    assert denied_approve.response.sent[0]["ephemeral"] is True
    assert captured_accept_kwargs == {}

    allowed_approve = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await approve_button.callback(allowed_approve)
    assert captured_accept_kwargs["proposal_id"] == 42
    assert captured_accept_kwargs["actor_id"] == 2
    assert allowed_approve.response.edited[0]["view"] is None

    reject_view = CreditLoanDecisionView(proposal_id=43, lender_id=2, creator_id=1)
    reject_button = next(
        child
        for child in reject_view.children
        if getattr(child, "custom_id", "") == "credit:reject"
    )
    denied_reject = FakeInteraction(user=FakeUser(user_id=3, name="charlie"))
    await reject_button.callback(denied_reject)
    assert denied_reject.response.sent[0]["ephemeral"] is True
    assert captured_reject_kwargs == {}

    allowed_reject = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await reject_button.callback(allowed_reject)
    assert captured_reject_kwargs == {"proposal_id": 43, "actor_id": 2}
    assert allowed_reject.response.edited[0]["view"] is None

    cancel_view = CreditLoanDecisionView(proposal_id=44, lender_id=2, creator_id=1)
    cancel_button = next(
        child
        for child in cancel_view.children
        if getattr(child, "custom_id", "") == "credit:cancel"
    )
    denied_cancel = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await cancel_button.callback(denied_cancel)
    assert denied_cancel.response.sent[0]["ephemeral"] is True
    assert captured_cancel_kwargs == {}

    allowed_cancel = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    await cancel_button.callback(allowed_cancel)
    assert captured_cancel_kwargs == {"proposal_id": 44, "actor_id": 1}
    assert allowed_cancel.response.edited[0]["view"] is None


async def test_loan_decision_timeout_rejects_and_schedules_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loan request views reject stale proposals and remove buttons on timeout."""
    scheduled: list[FakeDiscordMessage] = []
    rejected: list[int] = []

    async def fake_reject_expired_loan_proposal(proposal_id: int) -> LoanProposalView:
        """Records the expired proposal rejection."""
        rejected.append(proposal_id)
        return _fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST).model_copy(
            update={"proposal_id": proposal_id, "status": LoanProposalStatus.REJECTED}
        )

    def record_scheduled(
        message: FakeDiscordMessage, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records cleanup scheduling."""
        del delay, user_name
        scheduled.append(message)

    monkeypatch.setattr(views, "reject_expired_loan_proposal", fake_reject_expired_loan_proposal)
    monkeypatch.setattr(views, "schedule_public_message_delete", record_scheduled)

    credit_message = FakeDiscordMessage()
    credit_view = CreditLoanDecisionView(proposal_id=42, lender_id=2, creator_id=1)
    credit_view.message = credit_message
    await credit_view.on_timeout()

    central_message = FakeDiscordMessage()
    central_view = CentralBankLoanDecisionView(
        bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")),
        proposal_id=43,
        creator_id=1,
    )
    central_view.message = central_message
    await central_view.on_timeout()

    assert rejected == [42, 43]
    assert scheduled == [credit_message, central_message]
    assert credit_message.edits[0]["view"] is None
    assert central_message.edits[0]["view"] is None
    assert "逾時" in credit_message.edits[0]["embed"].title
    assert "逾時" in central_message.edits[0]["embed"].title


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
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="100"
    )

    assert called is False
    assert interaction.followup.sent[0].get("ephemeral") is True
    assert "權限不足" in interaction.followup.sent[0]["embed"].title


def test_parse_admin_amount_accepts_formatted_text() -> None:
    """Verifies admin adjustment text parsing avoids Discord integer option limits."""
    assert (
        economy._parse_positive_amount(raw_amount="9,007,199,254,740,993") == 9_007_199_254_740_993
    )
    assert economy._parse_positive_amount(raw_amount=" 0001 ") == 1
    assert economy._parse_positive_amount(raw_amount=None) is None
    assert economy._parse_positive_amount(raw_amount="0") is None
    assert economy._parse_positive_amount(raw_amount="not a number") is None
    assert economy._parse_positive_amount(raw_amount="-1") is None


async def test_economy_admin_tax_accepts_string_amounts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin tax commands must parse large string amounts before database mutation."""
    captured_deltas: list[int] = []

    async def record_adjust_balance(
        user_id: int, name: str, delta: int, allow_negative: bool = False, avatar_url: str = ""
    ) -> BalanceAdjustmentResult:
        """Records parsed adjustment deltas."""
        captured_deltas.append(delta)
        return BalanceAdjustmentResult(new_balance=150 + delta, applied_delta=delta)

    monkeypatch.setattr(economy, "get_admin", fake_get_admin)
    monkeypatch.setattr(economy, "adjust_balance", record_adjust_balance)
    monkeypatch.setattr(
        interactions, "schedule_public_message_delete", ignore_scheduled_public_message
    )
    cog = EconomyCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.admin_refund_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="9,007,199,254,740,993"
    )
    await EconomyCogs.admin_collect_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="9,007,199,254,740,993"
    )

    assert captured_deltas == [9_007_199_254_740_993, -9_007_199_254_740_993]


async def test_economy_admin_tax_allows_bot_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admin tax commands may adjust the bot account."""
    captured_targets: list[tuple[int, str, int]] = []

    async def record_adjust_balance(
        user_id: int, name: str, delta: int, allow_negative: bool = False, avatar_url: str = ""
    ) -> BalanceAdjustmentResult:
        """Records target accounts and parsed adjustment deltas."""
        del allow_negative, avatar_url
        captured_targets.append((user_id, name, delta))
        return BalanceAdjustmentResult(new_balance=150 + delta, applied_delta=delta)

    monkeypatch.setattr(economy, "get_admin", fake_get_admin)
    monkeypatch.setattr(economy, "adjust_balance", record_adjust_balance)
    monkeypatch.setattr(
        interactions, "schedule_public_message_delete", ignore_scheduled_public_message
    )
    cog = EconomyCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    interaction = FakeInteraction(user=FakeUser(user_id=1))
    bot_member = FakeUser(user_id=999, name="discordbot", display_name="Dealer", bot=True)

    await EconomyCogs.admin_refund_tax.callback(cog, interaction, member=bot_member, amount="100")
    await EconomyCogs.admin_collect_tax.callback(cog, interaction, member=bot_member, amount="50")

    assert captured_targets == [(999, "discordbot", 100), (999, "discordbot", -50)]
    assert interaction.followup.sent[0].get("ephemeral") is not True
    assert interaction.followup.sent[1].get("ephemeral") is not True


async def test_economy_admin_tax_rejects_invalid_amount_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid admin tax amount text must be rejected before balance mutation."""
    called = False

    async def fake_adjust_balance_guard(
        user_id: int, name: str, delta: int, allow_negative: bool = False, avatar_url: str = ""
    ) -> BalanceAdjustmentResult:
        """Fails the test if invalid amount text reaches the mutation path."""
        nonlocal called
        called = True
        return BalanceAdjustmentResult(new_balance=0, applied_delta=0)

    monkeypatch.setattr(economy, "adjust_balance", fake_adjust_balance_guard)
    cog = EconomyCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.admin_collect_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="not a number"
    )

    assert called is False
    assert interaction.response.sent[0]["ephemeral"] is True
    assert interaction.response.sent[0]["embed"].title == "收稅失敗"
    assert "金額格式錯誤" in interaction.response.sent[0]["embed"].description
    assert interaction.followup.sent == []


async def test_give_passes_guild_avatar_urls_to_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transfer writes should cache guild avatars instead of only global avatars."""
    captured_sender_avatar_url = ""
    captured_receiver_avatar_url = ""

    async def record_transfer(  # noqa: PLR0913 -- mirrors transfer signature
        sender_id: int,
        sender_name: str,
        receiver_id: int,
        receiver_name: str,
        amount: int,
        sender_avatar_url: str = "",
        receiver_avatar_url: str = "",
    ) -> TransferResult:
        """Records transfer identity payloads."""
        nonlocal captured_sender_avatar_url, captured_receiver_avatar_url
        del sender_id, sender_name, receiver_id, receiver_name, amount
        captured_sender_avatar_url = sender_avatar_url
        captured_receiver_avatar_url = receiver_avatar_url
        return TransferResult(sender_balance=50, receiver_balance=100)

    sender = FakeUser(user_id=1, name="alice")
    receiver = FakeUser(user_id=2, name="bob")
    cached_sender = FakeUser(user_id=1, name="alice")
    cached_sender.guild_avatar = SimpleNamespace(url="https://example.test/alice-server.png")
    cached_receiver = FakeUser(user_id=2, name="bob")
    cached_receiver.guild_avatar = SimpleNamespace(url="https://example.test/bob-server.png")
    members = {cached_sender.id: cached_sender, cached_receiver.id: cached_receiver}

    async def fail_fetch_member(user_id: int) -> FakeUser:
        """Fails if the helper ignores the cached member path."""
        raise AssertionError(f"unexpected fetch_member({user_id})")

    guild = SimpleNamespace(get_member=members.get, fetch_member=fail_fetch_member)
    interaction = FakeInteraction(user=sender)
    interaction.guild = guild
    monkeypatch.setattr(economy, "transfer", record_transfer)
    monkeypatch.setattr(
        interactions, "schedule_public_message_delete", ignore_scheduled_public_message
    )
    cog = EconomyCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))

    await EconomyCogs.give.callback(cog, interaction, member=receiver, amount="100")

    assert captured_sender_avatar_url == "https://example.test/alice-server.png"
    assert captured_receiver_avatar_url == "https://example.test/bob-server.png"


async def test_give_allows_bot_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Players may transfer balance to the bot account."""
    captured_transfer: dict[str, int | str] = {}

    async def record_transfer(  # noqa: PLR0913 -- mirrors transfer signature
        sender_id: int,
        sender_name: str,
        receiver_id: int,
        receiver_name: str,
        amount: int,
        sender_avatar_url: str = "",
        receiver_avatar_url: str = "",
    ) -> TransferResult:
        """Records bot-recipient transfer identity payloads."""
        del sender_avatar_url, receiver_avatar_url
        captured_transfer.update({
            "sender_id": sender_id,
            "sender_name": sender_name,
            "receiver_id": receiver_id,
            "receiver_name": receiver_name,
            "amount": amount,
        })
        return TransferResult(sender_balance=50, receiver_balance=100)

    sender = FakeUser(user_id=1, name="alice")
    bot_receiver = FakeUser(user_id=999, name="discordbot", display_name="Dealer", bot=True)
    interaction = FakeInteraction(user=sender)
    monkeypatch.setattr(economy, "transfer", record_transfer)
    monkeypatch.setattr(
        interactions, "schedule_public_message_delete", ignore_scheduled_public_message
    )
    cog = EconomyCogs(bot=SimpleNamespace(user=bot_receiver))

    await EconomyCogs.give.callback(cog, interaction, member=bot_receiver, amount="100")

    assert captured_transfer == {
        "sender_id": 1,
        "sender_name": "alice",
        "receiver_id": 999,
        "receiver_name": "discordbot",
        "amount": 100,
    }
    assert "轉帳完成" in interaction.followup.sent[0]["embed"].title


async def test_economy_money_commands_accept_large_string_amounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loan, transfer, and collection amounts parse beyond Discord integer option limits."""
    big_amount = 9_007_199_254_740_993
    captured: dict[str, int | None] = {}

    async def record_transfer(**kwargs: Any) -> TransferResult:  # noqa: ANN401 -- command facade double
        captured["give"] = kwargs["amount"]
        return TransferResult(sender_balance=0, receiver_balance=0)

    async def record_create_personal(**kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
        captured["credit_borrow"] = kwargs["amount"]
        return _fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)

    async def record_create_central(**kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
        captured["central_bank_borrow"] = kwargs["amount"]
        return _fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)

    async def record_repay_personal(**kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
        captured["credit_repay"] = kwargs["amount"]
        return await fake_loan_payment()

    async def record_repay_central(**kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
        captured["central_bank_repay"] = kwargs["amount"]
        return await fake_loan_payment()

    async def record_call_personal(**kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
        captured["credit_call"] = kwargs["amount"]
        return await fake_loan_payment()

    async def record_call_central(**kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
        captured["central_bank_call"] = kwargs["amount"]
        return await fake_loan_payment()

    monkeypatch.setattr(economy, "transfer", record_transfer)
    monkeypatch.setattr(economy, "create_personal_loan_request", record_create_personal)
    monkeypatch.setattr(economy, "create_central_bank_loan_request", record_create_central)
    monkeypatch.setattr(economy, "repay_personal_loans", record_repay_personal)
    monkeypatch.setattr(economy, "repay_central_bank_loans", record_repay_central)
    monkeypatch.setattr(economy, "call_personal_loans", record_call_personal)
    monkeypatch.setattr(economy, "call_central_bank_loans", record_call_central)
    monkeypatch.setattr(economy, "get_central_banker", fake_get_central_banker)
    monkeypatch.setattr(
        interactions, "schedule_public_message_delete", ignore_scheduled_public_message
    )
    cog = EconomyCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    big_text = "9,007,199,254,740,993"
    member = FakeUser(user_id=2, name="bob")

    await EconomyCogs.give.callback(cog, interaction, member=member, amount=big_text)
    await EconomyCogs.credit_borrow.callback(
        cog, interaction, member=member, amount=big_text, monthly_rate_percent=3.0
    )
    await EconomyCogs.credit_repay.callback(cog, interaction, member=member, amount=big_text)
    await EconomyCogs.credit_call.callback(cog, interaction, member=member, amount=big_text)
    await EconomyCogs.central_bank_borrow.callback(
        cog, interaction, amount=big_text, monthly_rate_percent=3.0
    )
    await EconomyCogs.central_bank_repay.callback(cog, interaction, amount=big_text)
    await EconomyCogs.central_bank_call.callback(cog, interaction, member=member, amount=big_text)

    assert captured == {
        "give": big_amount,
        "credit_borrow": big_amount,
        "credit_repay": big_amount,
        "credit_call": big_amount,
        "central_bank_borrow": big_amount,
        "central_bank_repay": big_amount,
        "central_bank_call": big_amount,
    }

    await EconomyCogs.credit_call.callback(cog, interaction, member=member, amount="0")
    await EconomyCogs.central_bank_call.callback(cog, interaction, member=member, amount="")
    assert captured["credit_call"] is None
    assert captured["central_bank_call"] is None


async def test_economy_money_commands_reject_invalid_amount_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed amount text is rejected before any balance, loan, or collection mutation."""
    mutated: list[str] = []

    async def guard_transfer(**kwargs: Any) -> TransferResult:  # noqa: ANN401 -- command facade double
        del kwargs
        mutated.append("transfer")
        return TransferResult(sender_balance=0, receiver_balance=0)

    async def guard_create_personal(**kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
        del kwargs
        mutated.append("create_personal")
        return _fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)

    async def guard_create_central(**kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
        del kwargs
        mutated.append("create_central")
        return _fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)

    async def guard_payment(**kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
        del kwargs
        mutated.append("payment")
        return await fake_loan_payment()

    monkeypatch.setattr(economy, "transfer", guard_transfer)
    monkeypatch.setattr(economy, "create_personal_loan_request", guard_create_personal)
    monkeypatch.setattr(economy, "create_central_bank_loan_request", guard_create_central)
    monkeypatch.setattr(economy, "repay_personal_loans", guard_payment)
    monkeypatch.setattr(economy, "repay_central_bank_loans", guard_payment)
    monkeypatch.setattr(economy, "call_personal_loans", guard_payment)
    monkeypatch.setattr(economy, "call_central_bank_loans", guard_payment)
    monkeypatch.setattr(economy, "get_central_banker", fake_get_central_banker)
    cog = EconomyCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    member = FakeUser(user_id=2, name="bob")

    def assert_rejected(interaction: FakeInteraction, expected_title: str) -> None:
        """Asserts an ephemeral malformed-amount rejection with no mutation followup."""
        assert interaction.response.sent[0]["ephemeral"] is True
        assert interaction.response.sent[0]["embed"].title == expected_title
        assert "金額格式錯誤" in interaction.response.sent[0]["embed"].description
        assert interaction.followup.sent == []

    rejections: list[tuple[str, Callable[[FakeInteraction], Awaitable[None]]]] = [
        ("轉帳失敗", lambda i: EconomyCogs.give.callback(cog, i, member=member, amount="x")),
        (
            "借款失敗",
            lambda i: EconomyCogs.credit_borrow.callback(
                cog, i, member=member, amount="x", monthly_rate_percent=3.0
            ),
        ),
        (
            "還款失敗",
            lambda i: EconomyCogs.credit_repay.callback(cog, i, member=member, amount="x"),
        ),
        (
            "催收失敗",
            lambda i: EconomyCogs.credit_call.callback(cog, i, member=member, amount="x"),
        ),
        (
            "央行借款失敗",
            lambda i: EconomyCogs.central_bank_borrow.callback(
                cog, i, amount="x", monthly_rate_percent=3.0
            ),
        ),
        ("央行還款失敗", lambda i: EconomyCogs.central_bank_repay.callback(cog, i, amount="x")),
        (
            "央行催收失敗",
            lambda i: EconomyCogs.central_bank_call.callback(cog, i, member=member, amount="x"),
        ),
    ]
    for expected_title, invoke in rejections:
        interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
        await invoke(interaction)
        assert_rejected(interaction, expected_title)

    assert mutated == []


async def test_loss_leaderboard_uses_daily_loss_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loss leaderboard embed describes gross daily loss, not net P&L."""
    scheduled: list[FakeDiscordMessage] = []

    async def daily_losses(
        limit: int, exclude_user_ids: tuple[int, ...] = ()
    ) -> list[LossLeaderboardEntry]:
        """Returns fake daily gross loss rows."""
        return [
            LossLeaderboardEntry(user_id=1, name="alice", loss_amount=500, avatar_url=""),
            LossLeaderboardEntry(user_id=2, name="bob", loss_amount=200, avatar_url=""),
        ]

    def record_scheduled(
        message: FakeDiscordMessage, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records cleanup scheduling."""
        scheduled.append(message)

    monkeypatch.setattr(economy, "top_losers", daily_losses)
    monkeypatch.setattr(interactions, "schedule_public_message_delete", record_scheduled)
    cog = EconomyCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.loss_leaderboard.callback(cog, interaction)

    embed = interaction.followup.sent[0]["embed"]
    assert "今日輸局累計" in embed.title
    assert "累計輸" in embed.description
    assert interaction.followup.sent[0]["files"][0].filename == "economy_loss_leaderboard.png"
    assert "贏回來不抵扣" in embed.footer.text
    assert len(scheduled) == 1


async def test_loss_leaderboard_empty_state_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loss leaderboard empty state stays explicit about today's loss rows."""
    scheduled: list[FakeDiscordMessage] = []

    async def no_daily_losses(
        limit: int, exclude_user_ids: tuple[int, ...] = ()
    ) -> list[LossLeaderboardEntry]:
        """Returns an empty daily loss board."""
        return []

    def record_scheduled(
        message: FakeDiscordMessage, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records cleanup scheduling."""
        scheduled.append(message)

    monkeypatch.setattr(economy, "top_losers", no_daily_losses)
    monkeypatch.setattr(interactions, "schedule_public_message_delete", record_scheduled)
    cog = EconomyCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.loss_leaderboard.callback(cog, interaction)

    embed = interaction.followup.sent[0]["embed"]
    assert "今日輸局累計" in embed.title
    assert "今天還沒有人輸錢" in embed.description
    assert len(scheduled) == 1


async def fake_get_balance(user_id: int) -> int:
    """Returns a stable fake balance."""
    return 150


async def fake_get_portfolio(user_id: int) -> PortfolioView:
    """Returns a stable fake portfolio."""
    return PortfolioView(
        user_id=user_id,
        name="alice",
        balance=150,
        debt_principal=30,
        debt_interest=5,
        net_worth=115,
    )


async def fake_get_stock_portfolio(user_id: int) -> StockPortfolioView:
    """Returns a stable fake stock portfolio."""
    return StockPortfolioView(
        user_id=user_id,
        holdings=(
            StockPortfolioHolding(
                symbol="BCAT",
                name="破貓科技",
                price_cents=10_000,
                long_shares=2,
                long_cost_basis=200,
                long_market_value=200,
                short_shares=0,
                short_entry_value=0,
                short_collateral=0,
                short_cover_cost=0,
                equity_value=200,
                unrealized_pnl=0,
                realized_pnl=25,
            ),
        ),
        equity_value=200,
        unrealized_pnl=0,
        realized_pnl=25,
    )


async def fake_get_vip(user_id: int) -> bool:
    """Returns non-VIP status."""
    return False


async def fake_get_admin(user_id: int) -> bool:
    """Returns economy admin status."""
    return True


async def fake_top_n(limit: int, exclude_user_ids: tuple[int, ...] = ()) -> list[LeaderboardEntry]:
    """Returns one fake leaderboard row."""
    return [
        LeaderboardEntry(
            user_id=1, name="alice", balance=150, avatar_url="https://cdn.example/alice.png"
        )
    ]


async def fake_top_losers(
    limit: int, exclude_user_ids: tuple[int, ...] = ()
) -> list[LossLeaderboardEntry]:
    """Returns one fake loss leaderboard row."""
    return [
        LossLeaderboardEntry(
            user_id=1, name="alice", loss_amount=500, avatar_url="https://cdn.example/alice.png"
        )
    ]


async def fake_get_account(user_id: int) -> AccountSnapshot:
    """Returns a fake bot wallet account."""
    return AccountSnapshot(name="Bot", balance=-50, total_earned=100, total_spent=150)


async def fake_get_casino_ledger() -> CasinoLedgerSnapshot:
    """Returns a fake casino ledger snapshot."""
    return CasinoLedgerSnapshot(
        balance=-50, total_earned=100, total_spent=150, updated_at=datetime.now(tz=UTC)
    )


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


async def fake_adjust_balance(
    user_id: int, name: str, delta: int, allow_negative: bool = False, avatar_url: str = ""
) -> BalanceAdjustmentResult:
    """Returns a successful fake manual adjustment result."""
    return BalanceAdjustmentResult(new_balance=150 + delta, applied_delta=delta)


def _fake_loan_proposal(kind: LoanProposalKind) -> LoanProposalView:
    """Builds a fake loan proposal view."""
    return LoanProposalView(
        proposal_id=1,
        kind=kind,
        status=LoanProposalStatus.PENDING,
        lender_type=LoanLenderType.CENTRAL_BANK
        if kind == LoanProposalKind.CENTRAL_BANK_REQUEST
        else LoanLenderType.USER,
        borrower_id=1,
        borrower_name="alice",
        lender_id=None if kind == LoanProposalKind.CENTRAL_BANK_REQUEST else 2,
        lender_name="bob",
        amount=100,
        monthly_rate_bps=300,
        escrow_amount=0,
        created_at=datetime.now(tz=UTC),
    )


async def fake_create_loan_request(**_kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
    """Returns a fake personal request."""
    return _fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)


async def fake_create_central_bank_request(**_kwargs: Any) -> LoanProposalView:  # noqa: ANN401 -- command facade double
    """Returns a fake central-bank request."""
    return _fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)


async def fake_get_central_banker(user_id: int) -> bool:
    """Returns central banker status."""
    return True


async def fake_reject_loan_proposal(
    proposal_id: int, actor_id: int, is_central_banker: bool = False
) -> LoanProposalView:
    """Returns a rejected fake proposal."""
    proposal = _fake_loan_proposal(kind=LoanProposalKind.CENTRAL_BANK_REQUEST)
    return proposal.model_copy(update={"status": LoanProposalStatus.REJECTED})


async def fake_cancel_loan_proposal(proposal_id: int, actor_id: int) -> LoanProposalView:
    """Returns a canceled fake proposal."""
    proposal = _fake_loan_proposal(kind=LoanProposalKind.PERSONAL_REQUEST)
    return proposal.model_copy(update={"status": LoanProposalStatus.CANCELED})


async def fake_accept_loan_proposal(**_kwargs: Any) -> LoanProposalAcceptResult:  # noqa: ANN401 -- command facade double
    """Returns a fake accepted proposal result."""
    contract = LoanContractView(
        contract_id=1,
        lender_type=LoanLenderType.USER,
        lender_id=2,
        lender_name="bob",
        borrower_id=1,
        borrower_name="alice",
        principal_remaining=100,
        interest_due=0,
        monthly_rate_bps=300,
        opened_at=datetime.now(tz=UTC),
        last_interest_accrued_at=datetime.now(tz=UTC),
        status=LoanContractStatus.ACTIVE,
    )
    return LoanProposalAcceptResult(
        contract=contract,
        borrower_balance=250,
        lender_balance=100,
        central_bank_available_credit=1_000,
    )


async def fake_list_loan_contracts(user_id: int) -> list[LoanContractView]:
    """Returns one active loan contract."""
    return [
        LoanContractView(
            contract_id=1,
            lender_type=LoanLenderType.USER,
            lender_id=2,
            lender_name="bob",
            borrower_id=user_id,
            borrower_name="alice",
            principal_remaining=100,
            interest_due=3,
            monthly_rate_bps=300,
            opened_at=datetime.now(tz=UTC),
            last_interest_accrued_at=datetime.now(tz=UTC),
            status=LoanContractStatus.ACTIVE,
        )
    ]


async def fake_loan_payment(**_kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
    """Returns a fake repayment result."""
    return LoanPaymentResult(
        paid_amount=50,
        interest_paid=5,
        principal_paid=45,
        borrower_balance=100,
        lender_balance=200,
        remaining_principal=55,
        remaining_interest=0,
    )


async def fake_call_personal_loans(**_kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
    """Returns a fake personal collection result."""
    return await fake_loan_payment()


async def fake_call_central_bank_loans(**_kwargs: Any) -> LoanPaymentResult:  # noqa: ANN401 -- command facade double
    """Returns a fake central-bank collection result."""
    return await fake_loan_payment()


async def fake_get_central_bank_status(
    exclude_user_ids: tuple[int, ...] = (),
) -> CentralBankStatus:
    """Returns fake central-bank capacity."""
    return CentralBankStatus(
        total_positive_user_balance=1_000, outstanding_principal=100, available_credit=900
    )


async def fake_checkin(user_id: int, name: str, avatar_url: str) -> CheckinResult:
    """Returns a successful fake daily check-in result."""
    return CheckinResult(new_balance=600_000, amount=150_000, streak=2, is_vip=False)


async def fake_buy_vip(user_id: int, name: str, avatar_url: str) -> VipPurchaseResult:
    """Returns a successful fake VIP purchase result."""
    return VipPurchaseResult(new_balance=500_000, cost=VIP_PURCHASE_COST)


def ignore_scheduled_public_message(
    message: FakeDiscordMessage, delay: float = 180, user_name: str | None = None
) -> None:
    """Ignores cleanup scheduling in command smoke tests."""
    return


async def fake_game_balance(user_id: int) -> int:
    """Returns a small fake game balance (bot stays at 0 so it does not auto-join)."""
    if user_id == 999:
        return 0
    return 100


async def _empty_game_balance(user_id: int) -> int:
    """Returns no spendable game balance."""
    return 0


async def _wealthy_game_balance(user_id: int) -> int:
    """Returns a fake balance large enough for Dragon Gate ante (bot still at 0)."""
    if user_id == 999:
        return 0
    return 1_000_000


async def fake_dragon_gate_jackpot_snapshot() -> JackpotSnapshot:
    """Returns a stable fake Dragon Gate jackpot snapshot."""
    return JackpotSnapshot(balance=100_000)


class FakeDealer:
    """Fake casino dealer that returns deterministic banter."""

    async def taunt_bet(
        self, author_name: str, player_name: str, balance_at_start: int, bet: int, game: str
    ) -> str:
        """Returns deterministic opening banter."""
        return "taunt"

    async def settle(  # noqa: PLR0913 -- mirrors SystemNarrator.settle signature
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


async def test_bot_blackjack_participant_spreads_bet_by_true_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bot's Kelly wager rises with a favorable channel true count."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))

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
    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    interaction = FakeInteraction()
    cast("Any", interaction).user = None

    await GamesCogs.blackjack_history.callback(cog, interaction, member=None, count=10)

    assert interaction.response.deferred is False
    assert interaction.response.sent[0]["ephemeral"] is True
    content = interaction.response.sent[0]["content"]
    assert isinstance(content, str)
    assert "無法辨識使用者" in content
    assert interaction.followup.sent == []


def test_parse_wager_amount_accepts_formatted_text() -> None:
    """Verifies wager text parsing avoids Discord integer option limits."""
    assert parse_wager_amount(raw_amount="9,007,199,254,740,993") == 9_007_199_254_740_993
    assert parse_wager_amount(raw_amount=" 000 ") == 0
    assert parse_wager_amount(raw_amount=None) is None
    assert parse_wager_amount(raw_amount="not a number") is None
    assert parse_wager_amount(raw_amount="-1") is None


async def test_games_commands_run_with_patched_settlement(
    monkeypatch: pytest.MonkeyPatch, economy_isolated_db: None
) -> None:
    """Verifies game commands create lobby views with patched dependencies."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    monkeypatch.setattr(games, "schedule_public_message_delete", ignore_scheduled_public_message)
    monkeypatch.setattr(games, "get_balance", fake_game_balance)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["narrator"] = FakeDealer()

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


async def test_blackjack_lobby_start_is_owner_only(
    monkeypatch: pytest.MonkeyPatch, economy_isolated_db: None
) -> None:
    """Verifies only the Blackjack lobby owner can press Start."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    monkeypatch.setattr(games, "get_balance", fake_game_balance)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["narrator"] = FakeDealer()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="10")
    lobby_view = owner_interaction.followup.sent[0]["view"]
    assert isinstance(lobby_view, BlackjackLobbyView)

    start_button = next(
        child for child in lobby_view.children if getattr(child, "label", "") == "開始"
    )
    other_interaction = FakeInteraction(user=FakeUser(user_id=2, name="bob", display_name="Bob"))
    await start_button.callback(other_interaction)

    assert other_interaction.response.sent
    assert isinstance(other_interaction.response.sent[0]["content"], str)


async def test_blackjack_owner_overbet_sets_table_bet_to_balance(
    monkeypatch: pytest.MonkeyPatch, economy_isolated_db: None
) -> None:
    """Verifies owner over-betting clamps the shared Blackjack lobby bet."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")

    async def balance_by_user(user_id: int) -> int:
        """Returns distinct balances for owner, joining player, and the bot."""
        return {1: 300, 2: 50_000_000, 999: 0}[user_id]

    monkeypatch.setattr(games, "get_balance", balance_by_user)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["narrator"] = FakeDealer()

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
    join_interaction.message = FakeDiscordMessage()
    await join_button.callback(join_interaction)

    bob = lobby_view.participants[1]
    assert bob.display_name == "Bob"
    assert bob.bet == 300
    assert bob.balance_at_start == 50_000_000
    assert bob.is_allin is False


async def test_refresh_participants_preserves_existing_blackjack_wagers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies start-time balance refresh keeps per-seat Blackjack wagers."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")

    async def balance_by_user(user_id: int) -> int:
        """Returns enough balance for the owner and bot to keep their queued bets."""
        return {1: 500, 999: 1_000}[user_id]

    monkeypatch.setattr(games, "get_balance", balance_by_user)
    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
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
    monkeypatch: pytest.MonkeyPatch, economy_isolated_db: None
) -> None:
    """Verifies Blackjack slash bet parsing accepts values above Discord integer limits."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")

    async def balance_by_user(user_id: int) -> int:
        """Returns enough balance to cover a large formatted wager (bot stays at 0)."""
        if user_id == 999:
            return 0
        return 10_000_000_000_000_000

    monkeypatch.setattr(games, "get_balance", balance_by_user)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["narrator"] = FakeDealer()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="9,007,199,254,740,993")
    lobby_view = owner_interaction.followup.sent[0]["view"]

    assert isinstance(lobby_view, BlackjackLobbyView)
    assert lobby_view.requested_bet == 9_007_199_254_740_993
    assert lobby_view.participants[0].bet == 9_007_199_254_740_993


async def test_blackjack_string_bet_rejects_invalid_text() -> None:
    """Verifies invalid text is rejected before wager preparation."""
    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="not a number")

    assert owner_interaction.response.sent[0]["ephemeral"] is True
    assert owner_interaction.response.sent[0]["embed"].title == "下注格式錯誤"
    assert owner_interaction.response.sent[0]["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert owner_interaction.response.sent[0]["embed"].image.url == embed_spacer_url()
    assert owner_interaction.followup.sent == []
    assert owner_interaction.response.deferred is False


async def test_blackjack_owner_zero_bet_sets_table_bet_to_balance(
    monkeypatch: pytest.MonkeyPatch, economy_isolated_db: None
) -> None:
    """Verifies bet zero avoids typing a very large numeric bet."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")

    async def balance_by_user(user_id: int) -> int:
        """Returns a large owner balance that is awkward to type in Discord."""
        return {1: 300_000_000_000_000, 2: 500_000_000_000_000, 999: 0}[user_id]

    monkeypatch.setattr(games, "get_balance", balance_by_user)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["narrator"] = FakeDealer()

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="0")
    lobby_view = owner_interaction.followup.sent[0]["view"]
    assert isinstance(lobby_view, BlackjackLobbyView)
    assert lobby_view.requested_bet == 300_000_000_000_000
    assert lobby_view.participants[0].bet == 300_000_000_000_000
    assert lobby_view.participants[0].is_allin is True


async def test_blackjack_zero_bet_rejects_empty_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies zero means all in, not a zero-stake table."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    monkeypatch.setattr(games, "schedule_public_message_delete", ignore_scheduled_public_message)
    monkeypatch.setattr(games, "get_balance", _empty_game_balance)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["narrator"] = FakeDealer()

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
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    monkeypatch.setattr(games, "schedule_public_message_delete", ignore_scheduled_public_message)
    monkeypatch.setattr(games, "get_balance", _empty_game_balance)

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.dragon_gate.callback(cog, owner_interaction)

    assert owner_interaction.followup.sent[0]["wait"] is True
    assert "view" not in owner_interaction.followup.sent[0]
    assert owner_interaction.followup.sent[0]["embed"].title == "餘額不足"
    assert owner_interaction.followup.sent[0]["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert owner_interaction.followup.sent[0]["embed"].image.url == embed_spacer_url()


async def test_dragon_gate_lobby_start_is_owner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies only the Dragon Gate lobby owner can press Start."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    monkeypatch.setattr(games, "get_balance", _wealthy_game_balance)
    monkeypatch.setattr(
        games, "fetch_dragon_gate_jackpot_snapshot", fake_dragon_gate_jackpot_snapshot
    )

    cog = GamesCogs(bot=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    cog.__dict__["narrator"] = FakeDealer()

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


async def test_system_narrator_times_out_to_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies SystemNarrator returns fallback narration when the LLM call times out."""
    monkeypatch.setattr("discordbot.cogs._games.dealer.NARRATOR_AI_TIMEOUT_SECONDS", 0.01)
    narrator = SystemNarrator(
        client=cast("AsyncOpenAI", HangingClient()),
        model=ModelSettings(name=TEST_DEALER_MODEL, effort="none"),
    )

    line = await narrator.taunt_bet(
        player_name="Alice", balance_at_start=100, bet=10, game="dragon_gate"
    )

    assert line == "賭場已收到下注, 牌桌即將發牌"


async def test_games_on_ready_cleans_stale_messages_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies startup cleanup runs once per GamesCogs instance."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    bot = SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))
    calls: list[SimpleNamespace] = []

    async def record_cleanup(bot: SimpleNamespace) -> None:
        """Records the bot passed to startup cleanup."""
        calls.append(bot)

    monkeypatch.setattr(games, "delete_tracked_public_messages", record_cleanup)
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
    bot = SimpleNamespace(user=FakeUser(user_id=999, bot=True), process_commands=record_processed)
    user_message = SimpleNamespace(author=FakeUser(user_id=1, bot=False))
    await cli.DiscordBot.on_message(bot, message=user_message)
    assert processed == [user_message]
    assert rewards[0]["amount"] == cli.BASE_MESSAGE_REWARD_AMOUNT
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
