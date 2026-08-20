"""Smoke tests for cogs, setup hooks, and high-level Discord command branches."""

from __future__ import annotations

import time
from types import TracebackType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, Unpack, TypedDict, cast, get_args
import asyncio
from pathlib import Path
from datetime import UTC, datetime, timedelta
from importlib import import_module
import threading
import contextlib

import nextcord
from nextcord import Embed, Guild, Member
from nextcord.ext import commands
from logfire._internal.constants import LEVEL_NUMBERS

from discordbot import cli
from discordbot.utils import interaction_responses as interactions
from discordbot.cogs.games import cog as games
from discordbot.cogs.video import cog as video
from discordbot.cogs.economy import cog as economy
from discordbot.cogs.economy import views
from discordbot.typings.games import GameParticipant
from discordbot.utils.threads import ThreadsOutput, ThreadsConversation
from discordbot.cogs.games.cog import GamesCogs
from discordbot.cogs.video.cog import VideoCogs
from discordbot.typings.config import LoggingConfig
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
from discordbot.cogs.auto_unmute import cog as auto_unmute
from discordbot.cogs.economy.cog import EconomyCogs
from discordbot.cogs.games.wagers import parse_wager_amount
from discordbot.cogs.template.cog import TemplateCogs
from discordbot.cogs.economy.views import CreditLoanDecisionView, CentralBankLoanDecisionView
from discordbot.cogs.parse_threads import cog as parse_threads
from discordbot.cogs.auto_unmute.cog import AutoUnmuteCogs
from discordbot.cogs.games.blackjack import Card
from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME, embed_spacer_url
from discordbot.utils.media_delivery import MediaHostingService, MediaDeliveryPlanner
from discordbot.cogs.parse_threads.cog import ThreadsCogs
from discordbot.services.economy.database import (
    VIP_PURCHASE_COST,
    CreditResult,
    TransferResult,
    VipPurchaseResult,
    BalanceAdjustmentResult,
)
from discordbot.cogs.games.blackjack_views import BlackjackLobbyView
from discordbot.cogs.games.dragon_gate_views import DragonGateLobbyView

from tests.helpers.embeds import assert_embed_has_field, assert_embed_title_prefix
from tests.helpers.casting import (
    as_bot,
    as_message,
    as_discord_bot,
    as_interaction,
    as_command_context,
    make_media_hosting_config,
)
from tests.helpers.discord_mocks import (
    FakeUser,
    DiscordPayload,
    FakeInteraction,
    FakeDiscordMessage,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable, AsyncIterator

    import pytest


class SelfTimeoutCall(TypedDict):
    """Recorded auto-unmute timeout handling call."""

    member: SimpleNamespace
    until: datetime


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

    def download(
        self,
        url: str,
        quality: str,
        dry_run: bool = False,
        stop_signal: threading.Event | None = None,
    ) -> DownloadResultStub:
        """Records the download request and returns the next queued result.

        `stop_signal` is accepted and ignored: the real downloader takes it so a caller can
        abort a blocking yt-dlp run, and `/download_video` now passes one on every call.
        """
        del stop_signal
        kwargs: dict[str, str | bool] = {"url": url, "quality": quality, "dry_run": dry_run}
        self.calls.append(kwargs)
        return self.results.pop(0)


# Body of the comment every readable ParseResultStub conversation carries, so a test can assert
# the Discord expansion never renders it.
_STUB_COMMENT_TEXT = "a stranger's comment the expansion must ignore"


class ParseResultStub:
    """Context manager stub for Threads parse results."""

    def __init__(
        self,
        results: list[ThreadsOutput] | BaseException,
        exit_error: Exception | None = None,
        enter_delay_seconds: float = 0.0,
        output_folder: str | None = None,
    ) -> None:
        """Stores parsed results, the entry and exit errors, and how long the entry blocks."""
        self.results = results
        self.exit_error = exit_error
        self.enter_delay_seconds = enter_delay_seconds
        self.output_folder = output_folder
        self.exited = False
        self.wrote: Path | None = None
        self.finished = threading.Event()

    def __enter__(self) -> ThreadsConversation:
        """Returns the parsed conversation or raises the configured parsing error.

        A readable post always comes back carrying a comment, because that is what production
        yields now: the expansion is supposed to ignore them, and a stub with no comments in it
        cannot tell "ignores them" apart from "never saw any".

        `enter_delay_seconds` blocks the worker thread the way a slow-drip CDN does, so a test
        can reach the caller's give-up path with the walk still running. What happens after that
        delay is `download_media`'s shape: the media write is attempted against the folder the
        caller handed over, and never against one this rebuilds.
        """
        time.sleep(self.enter_delay_seconds)
        if self.output_folder is not None:
            # Named before the write, so an abandoned walk still says where it aimed once the
            # removal turned that write into a FileNotFoundError. Suppressed for the same
            # reason production discards it: nothing is awaiting this thread any more.
            self.wrote = Path(self.output_folder) / "clip.mp4"
            with contextlib.suppress(OSError):
                self.wrote.write_bytes(b"clip")
        self.finished.set()
        if isinstance(self.results, BaseException):
            raise self.results
        return ThreadsConversation(
            chain=self.results,
            reply_branches=(
                [[_thread_output(text=_STUB_COMMENT_TEXT, image_urls=["https://x.test/c.png"])]]
                if self.results
                else []
            ),
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Keeps fake parsed outputs available after context exit, or fails the cleanup."""
        self.exited = True
        if self.exit_error:
            raise self.exit_error


class ThreadsDownloaderStub:
    """Fake Threads downloader returning a configured parse context manager."""

    def __init__(
        self,
        results: list[ThreadsOutput] | BaseException,
        exit_error: Exception | None = None,
        enter_delay_seconds: float = 0.0,
    ) -> None:
        """Stores parsed results, both failures, and how long each parse blocks on entry."""
        self.results = results
        self.exit_error = exit_error
        self.enter_delay_seconds = enter_delay_seconds
        self.parsed: list[ParseResultStub] = []
        self.output_folders: list[str] = []

    def parse(self, url: str) -> ParseResultStub:
        """Returns a fake parse context manager, recorded so a test can inspect its exit."""
        result = ParseResultStub(
            results=self.results,
            exit_error=self.exit_error,
            enter_delay_seconds=self.enter_delay_seconds,
            output_folder=self.output_folders[-1] if self.output_folders else None,
        )
        self.parsed.append(result)
        return result


def _wire_threads(*, cog: ThreadsCogs, downloader: ThreadsDownloaderStub) -> ThreadsDownloaderStub:
    """Points the cog's per-invocation factory at one stub, recording the dir it was handed.

    The expansion builds its downloader inside a scratch directory of its own now, so the
    factory is the seam a test takes over; the same stub answers every invocation so a test can
    still read back what it was asked to do.
    """

    def factory(output_folder: str) -> ThreadsDownloaderStub:
        """Records the scratch dir this invocation was given, then serves the shared stub."""
        downloader.output_folders.append(output_folder)
        return downloader

    cog.__dict__["downloader_factory"] = factory
    return downloader


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
        """Stores generated text as a structured output message (mirrors the real Response)."""
        self.output_text = output_text
        self.output = [
            SimpleNamespace(
                type="message", content=[SimpleNamespace(type="output_text", text=output_text)]
            )
        ]


def _thread_output(  # noqa: PLR0913 -- one knob per ThreadsOutput field the embeds render
    text: str = "hello",
    image_urls: list[str] | None = None,
    video_paths: list[Path] | None = None,
    video_urls: list[str] | None = None,
    author_name: str = "alice",
    quoted: ThreadsOutput | None = None,
    quoted_unavailable: bool = False,
) -> ThreadsOutput:
    """Builds a parsed Threads output fixture."""
    return ThreadsOutput(
        text=text,
        url=f"https://www.threads.net/@{author_name}/post/abc",
        image_urls=image_urls or [],
        video_urls=video_urls or [],
        video_paths=video_paths or [],
        author_name=author_name,
        author_icon_url="https://example.test/avatar.png",
        like_count=1,
        reply_count=2,
        repost_count=3,
        quote_count=4,
        reshare_count=5,
        taken_at=datetime(2026, 1, 1, tzinfo=UTC),
        quoted=quoted,
        quoted_unavailable=quoted_unavailable,
    )


def _long_threads_chain() -> list[ThreadsOutput]:
    """Builds the measured worst-case chain that crosses the message-wide embed limit."""
    chain: list[ThreadsOutput] = []
    for index in range(10):
        prefix = f"post-{index}-"
        post = _thread_output(
            text=prefix + "x" * (500 - len(prefix)),
            author_name=(f"user-{index}-" + "a" * 30)[:30],
            video_urls=([] if index == 9 else [f"https://example.test/video-{index}.mp4"]),
        )
        post.like_count = 9_999_999
        post.reply_count = 9_999_999
        post.repost_count = 9_999_999
        post.quote_count = 9_999_999
        post.reshare_count = 9_999_999
        chain.append(post)
    return chain


async def test_template_on_message_and_ping() -> None:
    """Verifies template debug reaction and ping command response."""
    cog = TemplateCogs(bot=as_bot(fake=SimpleNamespace(latency=0.123)))
    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "debug"
    await cog.on_message(message=as_message(fake=message))
    assert message.reactions == ["🤬"]

    bot_message = FakeDiscordMessage()
    bot_message.__dict__["author"] = FakeUser(bot=True)
    bot_message.__dict__["content"] = "debug"
    await cog.on_message(message=as_message(fake=bot_message))
    assert bot_message.reactions == []

    interaction = FakeInteraction(user=FakeUser(display_name="Alice"))
    await TemplateCogs.ping.callback(cog, interaction)
    embed = interaction.followup.sent[0]["embed"]
    assert isinstance(embed, Embed)
    assert embed.title == ":ping_pong: Pong!"


async def test_video_deliver_and_download_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies video delivery, oversize URL fallback, hosting-off, and download error branches."""
    cog = VideoCogs(bot=as_bot(fake=SimpleNamespace()))
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()  # the serve dir is a pre-existing host mount; the bot never creates it
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(
            config=make_media_hosting_config(
                enabled=True, base_url="https://media.test", serve_dir=str(serve_dir)
            )
        )
    )
    small = tmp_path / "small.mp4"
    small.write_bytes(data=b"0" * 100)
    big = tmp_path / "big.mp4"
    big.write_bytes(data=b"0" * 300)

    interaction = FakeInteraction()
    await cog._deliver(
        interaction=as_interaction(fake=interaction),
        file_size_mb=1.25,
        file_path=small,
        url="https://source.test/video",
    )
    success_content = interaction.edits[-1]["content"]
    assert isinstance(success_content, str)
    assert success_content == "-# 檔案大小: 1.2MB\n-# 來源: <https://source.test/video>"
    assert interaction.edits[-1]["file"] is not None
    assert interaction.followup.sent == []

    # Too big for native upload + hosting on: post the URL, no 480p retry, no attachment.
    downloader = DownloaderStub(results=[DownloadResultStub(filename=big)])
    monkeypatch.setattr(video, "VideoDownloader", lambda output_folder: downloader)
    host_interaction = FakeInteraction(filesize_limit=200)
    await VideoCogs.download_video.callback(
        cog, host_interaction, url="https://x.test", quality="best"
    )
    assert [call["quality"] for call in downloader.calls] == ["best"]
    host_content = host_interaction.edits[-1]["content"]
    assert any(line.startswith("https://media.test/") for line in host_content.splitlines())
    # The source link is omitted so the hosted URL is the only link and Discord inline-plays it.
    assert "https://x.test" not in host_content
    assert "file" not in host_interaction.edits[-1]
    assert host_interaction.followup.sent == []

    # Too big + hosting unavailable: fall back to the "file too large" message.
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(
            config=make_media_hosting_config(enabled=True, base_url="", serve_dir="")
        )
    )
    big2 = tmp_path / "big2.mp4"
    big2.write_bytes(data=b"0" * 300)
    fail_interaction = FakeInteraction(filesize_limit=200)
    monkeypatch.setattr(
        video,
        "VideoDownloader",
        lambda output_folder: DownloaderStub(results=[DownloadResultStub(filename=big2)]),
    )
    await VideoCogs.download_video.callback(
        cog, fail_interaction, url="https://x.test", quality="best"
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
    cog = ThreadsCogs(bot=as_bot(fake=bot))
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(data=b"123")

    parent = _thread_output(text="parent", video_urls=["https://example.test/video.mp4"])
    target = _thread_output(
        image_urls=["https://example.test/1.png", "https://example.test/2.png"]
    )
    embeds = cog._build_embed_plan(results=[parent, target]).embeds
    assert len(embeds) == 3
    first_description = embeds[0].description
    assert first_description is not None
    assert "點此觀看影片" in first_description
    assert ThreadsCogs._gradient_color(index=0, total=1) == nextcord.Color.default()

    no_match = FakeDiscordMessage()
    no_match.__dict__["author"] = FakeUser(bot=False)
    no_match.__dict__["content"] = "hello"
    await cog.on_message(message=as_message(fake=no_match))
    assert no_match.reactions == []

    success_message = FakeDiscordMessage()
    success_message.__dict__["author"] = FakeUser(bot=False)
    success_message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    success_message.__dict__["guild"] = SimpleNamespace(filesize_limit=25 * 1024 * 1024)
    _wire_threads(
        cog=cog,
        downloader=ThreadsDownloaderStub(
            results=[_thread_output(video_paths=[video_file], image_urls=[])]
        ),
    )
    await cog.on_message(message=as_message(fake=success_message))
    assert success_message.suppressed
    assert success_message.replies[0]["files"]
    assert success_message.reactions[-1] == "<:greencheck:1517565102424068226>"
    # The read marker rides beside the status chain, which only ever removes its own reaction.
    assert success_message.reactions[0] == "<:threads:1535657820668559380>"
    assert all(emoji != "<:threads:1535657820668559380>" for emoji, _ in success_message.removed)
    # The parse now carries the comments too, but the expansion shows the chain only: the
    # 10-embed cap belongs to the linked post, and a comment would push its own images out.
    assert all(
        _STUB_COMMENT_TEXT not in (embed.description or "")
        for embed in success_message.replies[0]["embeds"]
    )

    warning_message = FakeDiscordMessage()
    warning_message.__dict__["author"] = FakeUser(bot=False)
    warning_message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    warning_message.__dict__["guild"] = SimpleNamespace(filesize_limit=25 * 1024 * 1024)
    _wire_threads(cog=cog, downloader=ThreadsDownloaderStub(results=[]))
    await cog.on_message(message=as_message(fake=warning_message))
    assert warning_message.reactions[-1] == "⚠️"

    error_message = FakeDiscordMessage()
    error_message.__dict__["author"] = FakeUser(bot=False)
    error_message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    error_message.__dict__["guild"] = SimpleNamespace(filesize_limit=25 * 1024 * 1024)
    _wire_threads(cog=cog, downloader=ThreadsDownloaderStub(results=RuntimeError("parse failed")))
    await cog.on_message(message=as_message(fake=error_message))
    assert error_message.reactions[-1] == "<:redcross:1517565100838355016>"


async def test_threads_cog_takes_the_scratch_dir_of_a_walk_it_gave_up_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out expansion leaves nothing behind, without waiting for the walk.

    The `requests` calls under `parse` are per-read only, so a slow-drip CDN can hold one paste
    open indefinitely; the bound is what stops it. `asyncio.to_thread` cannot cancel the walk,
    so it is the scratch directory going away that both deletes what it wrote and fails its next
    write — the same mechanism `parse_douyin` and `/download_video` get from their own `with`
    block. The exit is deliberately not called on this path: the walk is still driving that
    generator on its own thread.
    """
    monkeypatch.setattr(parse_threads, "THREADS_EXPAND_TIMEOUT_SECONDS", 0.05)
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    downloader = _wire_threads(
        cog=cog, downloader=ThreadsDownloaderStub(results=[], enter_delay_seconds=0.3)
    )

    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    message.__dict__["guild"] = SimpleNamespace(filesize_limit=25 * 1024 * 1024)
    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == "<:redcross:1517565100838355016>"
    scratch = Path(downloader.output_folders[0])
    assert not await asyncio.to_thread(scratch.exists)
    assert downloader.parsed[0].exited is False
    # The abandoned walk runs on past the give-up and writes where it was told to; what it
    # produces has to be gone with the directory rather than stranded in the system temp dir.
    assert await asyncio.to_thread(downloader.parsed[0].finished.wait, 5.0)
    wrote = downloader.parsed[0].wrote
    assert wrote is not None
    assert not await asyncio.to_thread(wrote.exists)
    assert not await asyncio.to_thread(scratch.exists)


async def test_download_video_gives_up_on_a_stalling_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """yt-dlp's own retry budget is not a ceiling, so the command carries one.

    `socket_timeout` applies per socket and each of the three retry settings multiplies it, so
    without this the user sits on "正在下載影片..." for as long as the host cares to stall. The
    stop signal is half of it: `asyncio.to_thread` cannot be cancelled, so the bound only ends
    the download because the worker is watching for it.
    """
    monkeypatch.setattr(video, "VIDEO_DOWNLOAD_TIMEOUT_SECONDS", 0.05)
    cog = VideoCogs(bot=as_bot(fake=SimpleNamespace()))

    class StallingDownloader:
        """Drips like a stalling host, and watches the stop signal like yt-dlp's progress hook."""

        def __init__(self) -> None:
            """Records which way the worker ended."""
            self.aborted = False
            self.finished = False

        def download(
            self,
            url: str,
            quality: str,
            dry_run: bool = False,
            stop_signal: threading.Event | None = None,
        ) -> DownloadResultStub:
            """Outlasts the command's bound by two orders of magnitude unless told to stop."""
            del url, quality, dry_run
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if stop_signal is not None and stop_signal.is_set():
                    self.aborted = True
                    raise RuntimeError("download stopped")
                time.sleep(0.01)
            self.finished = True
            raise AssertionError("should have been abandoned")

    downloader = StallingDownloader()
    monkeypatch.setattr(video, "VideoDownloader", lambda output_folder: downloader)
    interaction = FakeInteraction()
    await VideoCogs.download_video.callback(
        cog, as_interaction(fake=interaction), url="https://x.test", quality="best"
    )

    assert interaction.edits[-1]["content"] == "-# 檔案無法下載"
    # Any failure prints that same line, so what proves the BOUND fired is which way the worker
    # ended: aborted on the signal rather than running its stall out.
    assert downloader.aborted is True
    assert downloader.finished is False


async def test_threads_cog_trims_long_chain_to_the_message_wide_embed_limit() -> None:
    """Far ancestors are removed before a total over 6000 can make Discord reject the reply."""
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))

    plan = cog._build_embed_plan(results=_long_threads_chain())
    embeds = plan.embeds

    assert sum(parse_threads._embed_text_length(embed=embed) for embed in embeds) <= 6000
    authors = [cast("str", embed.author.name) for embed in embeds if embed.author]
    assert authors[-1].startswith("user-9-")
    assert any(author.startswith("user-8-") for author in authors)
    assert not any(author.startswith("user-0-") for author in authors)
    assert plan.omitted_posts[0].author_name.startswith("user-0-")


def test_threads_embed_plan_is_a_frozen_model() -> None:
    """The completed allocation remains an immutable model value."""
    plan = parse_threads._EmbedPlan(embeds=[], omitted_posts=[])

    assert plan.model_config["frozen"] is True
    assert plan.model_copy(update={"embeds": []}).embeds == []


async def test_threads_cog_keeps_the_target_quote_and_nearest_ancestor() -> None:
    """The quoted post remains second in priority while a farther ancestor is dropped."""
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    root = _thread_output(text="root-" + "r" * 1095, author_name="root")
    parent = _thread_output(text="parent-" + "p" * 1093, author_name="parent")
    target = _thread_output(text="target-" + "t" * 2193, author_name="target")
    target.quoted = _thread_output(
        text="quoted-" + "q" * 2193,
        author_name="quoted",
        image_urls=["https://example.test/quoted-1.png", "https://example.test/quoted-2.png"],
    )

    embeds = cog._build_embed_plan(results=[root, parent, target]).embeds

    assert sum(parse_threads._embed_text_length(embed=embed) for embed in embeds) <= 6000
    descriptions = [embed.description or "" for embed in embeds]
    assert descriptions[0].startswith("parent-")
    assert descriptions[1].startswith("target-")
    assert descriptions[2].startswith("🔗 **被引用的貼文**")
    assert all(not description.startswith("root-") for description in descriptions)
    assert sum(1 for embed in embeds if embed.image) == 2


async def test_threads_cog_drops_an_over_budget_post_with_its_gallery() -> None:
    """A quote that cannot fit does not leave its image-only embeds detached from their text."""
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    parent = _thread_output(text="nearby context", author_name="parent")
    target = _thread_output(text="t" * 3500, author_name="target")
    target.quoted = _thread_output(
        text="q" * 3000,
        author_name="quoted",
        image_urls=[f"https://example.test/quoted-{index}.png" for index in range(4)],
    )

    plan = cog._build_embed_plan(results=[parent, target])
    embeds = plan.embeds

    assert sum(parse_threads._embed_text_length(embed=embed) for embed in embeds) <= 6000
    assert [embed.author.name for embed in embeds if embed.author] == ["parent", "target"]
    assert all(not embed.image for embed in embeds)
    assert all("被引用的貼文" not in (embed.description or "") for embed in embeds)
    assert [post.author_name for post in plan.omitted_posts] == ["quoted"]


async def test_threads_cog_counts_astral_emoji_as_utf16_units() -> None:
    """Emoji-heavy posts stay safe even if Discord interprets characters as UTF-16 units."""
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    chain = [_thread_output(text="😀" * 500, author_name=f"user-{index}") for index in range(10)]

    embeds = cog._build_embed_plan(results=chain).embeds

    assert parse_threads._utf16_length(value="😀") == 2
    assert len(embeds) < len(chain)
    assert sum(parse_threads._embed_text_length(embed=embed) for embed in embeds) <= 6000
    assert [embed.author.name for embed in embeds if embed.author] == [
        "user-5",
        "user-6",
        "user-7",
        "user-8",
        "user-9",
    ]


async def test_threads_cog_delivers_a_trimmed_chain_instead_of_failing() -> None:
    """An overflow is delivered with permalink fallbacks and a success reaction."""
    bot = SimpleNamespace(user=SimpleNamespace(id=999))
    cog = ThreadsCogs(bot=as_bot(fake=bot))
    _wire_threads(cog=cog, downloader=ThreadsDownloaderStub(results=_long_threads_chain()))
    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    message.__dict__["guild"] = SimpleNamespace(filesize_limit=25 * 1024 * 1024)

    await cog.on_message(message=as_message(fake=message))

    assert len(message.replies) == 2
    embeds = message.replies[0]["embeds"]
    assert sum(parse_threads._embed_text_length(embed=embed) for embed in embeds) <= 6000
    notice = cast("str", message.replies[1]["content"])
    assert "未展開" in notice
    omitted_urls = [
        post.url for post in cog._build_embed_plan(results=_long_threads_chain()).omitted_posts
    ]
    assert all(f"<{url}>" in notice for url in omitted_urls)
    assert message.reactions[-1] == "<:greencheck:1517565102424068226>"


def test_threads_cog_paginates_omitted_post_links() -> None:
    """Every omitted permalink survives once the notice needs a second message."""
    posts = [_thread_output(author_name=f"user-{index}-" + "a" * 30) for index in range(30)]

    pages = parse_threads._omitted_post_notice_pages(posts=posts)

    assert 1 < len(pages) <= parse_threads._MAX_OMITTED_NOTICE_PAGES
    assert all(parse_threads._utf16_length(value=page) <= 2000 for page in pages)
    combined = "\n".join(pages)
    assert all(f"<{post.url}>" in combined for post in posts)
    assert "未列出" not in combined


def test_threads_cog_caps_the_notice_and_counts_what_it_drops() -> None:
    """A chain past the page cap states its remainder instead of emitting more replies."""
    posts = [_thread_output(author_name=f"user-{index}-" + "a" * 30) for index in range(100)]

    pages = parse_threads._omitted_post_notice_pages(posts=posts)

    assert len(pages) == parse_threads._MAX_OMITTED_NOTICE_PAGES
    assert all(parse_threads._utf16_length(value=page) <= 2000 for page in pages)
    combined = "\n".join(pages)
    # The header keeps naming every omitted post, and the closing line accounts for the
    # permalinks that did not fit, so the two together still add up to the real total.
    assert f"有 {len(posts)} 篇貼文未展開" in combined
    listed = sum(f"<{post.url}>" in combined for post in posts)
    assert f"其中 {len(posts) - listed} 篇的連結因訊息長度限制未列出." in pages[-1]


def test_threads_cog_counts_a_permalink_no_page_could_hold() -> None:
    """An unrenderable line is dropped into the count rather than refusing the notice."""
    posts = [_thread_output(author_name="a" * 3000), _thread_output(author_name="bob")]

    pages = parse_threads._omitted_post_notice_pages(posts=posts)

    assert len(pages) == 1
    assert parse_threads._utf16_length(value=pages[0]) <= 2000
    assert "<https://www.threads.net/@bob/post/abc>" in pages[0]
    assert "其中 1 篇的連結因訊息長度限制未列出." in pages[0]


async def test_threads_cog_keeps_the_expansion_when_a_notice_reply_fails() -> None:
    """A follow-up failure must not relabel an expansion that is already on screen."""
    bot = SimpleNamespace(user=SimpleNamespace(id=999))
    cog = ThreadsCogs(bot=as_bot(fake=bot))
    _wire_threads(cog=cog, downloader=ThreadsDownloaderStub(results=_long_threads_chain()))
    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    message.__dict__["guild"] = SimpleNamespace(filesize_limit=25 * 1024 * 1024)
    expansion_reply = message.reply
    reactions_when_the_notice_ran: list[str] = []

    async def reply_then_fail(**kwargs: object) -> None:
        if message.replies:
            reactions_when_the_notice_ran.extend(message.reactions)
            raise RuntimeError("the source message went away")
        await expansion_reply(**cast("Any", kwargs))

    message.reply = reply_then_fail  # ty: ignore[invalid-assignment]

    await cog.on_message(message=as_message(fake=message))

    assert len(message.replies) == 1
    assert message.replies[0]["embeds"]
    assert message.reactions[-1] == "<:greencheck:1517565102424068226>"
    # The ✅ is already painted by the time a notice is attempted, so a follow-up that hangs
    # rather than failing cannot leave the expansion looking unfinished either.
    assert reactions_when_the_notice_ran[-1] == "<:greencheck:1517565102424068226>"


async def test_threads_cog_delivers_when_the_notice_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A notice-building failure costs the permalinks, never the rendered expansion."""
    bot = SimpleNamespace(user=SimpleNamespace(id=999))
    cog = ThreadsCogs(bot=as_bot(fake=bot))
    _wire_threads(cog=cog, downloader=ThreadsDownloaderStub(results=_long_threads_chain()))
    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    message.__dict__["guild"] = SimpleNamespace(filesize_limit=25 * 1024 * 1024)

    def exploding_pages(*, posts: list[ThreadsOutput]) -> list[str]:
        del posts
        raise ValueError("a permalink fallback exceeds Discord's message limit")

    monkeypatch.setattr(parse_threads, "_omitted_post_notice_pages", exploding_pages)
    await cog.on_message(message=as_message(fake=message))

    assert len(message.replies) == 1
    assert message.replies[0]["embeds"]
    assert message.reactions[-1] == "<:greencheck:1517565102424068226>"


async def test_threads_cog_keeps_the_expansion_when_the_scratch_cleanup_fails() -> None:
    """A temp file the user cannot see must never repaint a delivered expansion as failed."""
    bot = SimpleNamespace(user=SimpleNamespace(id=999))
    cog = ThreadsCogs(bot=as_bot(fake=bot))
    downloader = ThreadsDownloaderStub(
        results=[_thread_output(text="貼文內容")], exit_error=OSError("read-only file system")
    )
    _wire_threads(cog=cog, downloader=downloader)
    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    message.__dict__["guild"] = SimpleNamespace(filesize_limit=25 * 1024 * 1024)

    await cog.on_message(message=as_message(fake=message))

    # The cleanup really ran and really failed, so the ✅ below is the guard's doing.
    assert downloader.parsed[0].exited
    assert len(message.replies) == 1
    assert message.replies[0]["embeds"]
    assert message.reactions[-1] == "<:greencheck:1517565102424068226>"


async def test_threads_cog_logs_both_a_failed_step_and_the_cleanup_that_failed_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swallowing the cleanup must not swallow the failure the cleanup used to replace."""
    bot = SimpleNamespace(user=SimpleNamespace(id=999))
    cog = ThreadsCogs(bot=as_bot(fake=bot))
    downloader = ThreadsDownloaderStub(
        results=[_thread_output(text="貼文內容")], exit_error=OSError("read-only file system")
    )
    _wire_threads(cog=cog, downloader=downloader)
    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    message.__dict__["guild"] = SimpleNamespace(filesize_limit=25 * 1024 * 1024)
    logged: list[tuple[str, str]] = []

    def record(message_text: str, **kwargs: Any) -> None:  # noqa: ANN401 -- logfire accepts arbitrary fields
        """Records the message and error type of each log it is bound to."""
        logged.append((message_text, kwargs["error_type"]))

    def exploding_plan(*, results: list[ThreadsOutput]) -> parse_threads._EmbedPlan:
        del results
        raise RuntimeError("the embed plan blew up")

    monkeypatch.setattr(parse_threads.logfire, "error", record)
    # The cleanup is a warning rather than an error now: the scratch directory around it removes
    # what a failing unlink left, so it is a degraded step rather than a leak nobody clears.
    monkeypatch.setattr(parse_threads.logfire, "warn", record)
    cog._build_embed_plan = exploding_plan  # ty: ignore[invalid-assignment]
    await cog.on_message(message=as_message(fake=message))

    assert downloader.parsed[0].exited
    assert message.replies == []
    assert message.reactions[-1] == "<:redcross:1517565100838355016>"
    # The step that lost the expansion is logged with its own cause rather than with the
    # OSError the cleanup used to overwrite it with, and the cleanup gets its own line.
    assert ("Could not clean up the Threads scratch files", "OSError") in logged
    assert ("Threads expansion failed outside the parse and delivery steps", "RuntimeError") in (
        logged
    )


async def test_threads_cog_shows_the_post_a_quote_post_quotes() -> None:
    """The quoted post is the subject of a quote post, so it earns its own marked embed."""
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    quoted = _thread_output(
        text="the original argument",
        author_name="bob",
        video_urls=["https://example.test/quoted.mp4"],
    )
    target = _thread_output(text="這根本是胡說", image_urls=["https://example.test/1.png"])
    target.quoted = quoted

    embeds = cog._build_embed_plan(results=[target]).embeds

    assert len(embeds) == 2
    # The target owns the message, so it stays first and the quoted post hangs off the end. That
    # root-first ordering is load-bearing elsewhere (see the gen_reply embed-card scan).
    assert embeds[0].description == "這根本是胡說"
    quoted_embed = embeds[1]
    assert quoted_embed.author.name == "bob"
    assert quoted_embed.description is not None
    assert quoted_embed.description.startswith("🔗 **被引用的貼文**")
    assert "the original argument" in quoted_embed.description
    # Its clip is never downloaded, so it is linked instead of showing as an empty embed.
    assert "點此觀看影片" in quoted_embed.description
    # Off the greyscale chain gradient on purpose: it is not a layer of the thread.
    assert quoted_embed.colour == nextcord.Color.blurple()


async def test_threads_cog_keeps_the_commentary_beside_a_quoted_gallery() -> None:
    """The shape that motivated this: one line over someone else's ten-image carousel.

    Letting the gallery compete freely for the 10-embed cap drops the commentary that owns the
    message, which would leave the reader the same fragment showing the quoted post exists to fix.
    """
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    quoted = _thread_output(
        text="the subject",
        author_name="bob",
        image_urls=[f"https://example.test/{index}.png" for index in range(10)],
    )
    target = _thread_output(text="一句話評論")
    target.quoted = quoted

    embeds = cog._build_embed_plan(results=[target]).embeds

    assert len(embeds) == 10
    assert embeds[0].description == "一句話評論"
    assert embeds[1].description is not None
    assert embeds[1].description.startswith("🔗 **被引用的貼文**")
    # Nine of the quoted post's ten images fit; the tenth loses to the commentary, not the reverse.
    assert sum(1 for embed in embeds if embed.image) == 9


async def test_threads_cog_notes_a_quoted_post_that_is_gone() -> None:
    """A gone quoted post has nothing to show, so it rides on the target instead of an embed."""
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    target = _thread_output(text="回應一下")
    target.quoted_unavailable = True

    embeds = cog._build_embed_plan(results=[target]).embeds

    assert len(embeds) == 1
    assert embeds[0].description is not None
    assert "引用的貼文目前無法瀏覽" in embeds[0].description


async def test_threads_cog_reserves_the_quoted_posts_slot_against_an_ancestors_gallery() -> None:
    """The quoted post's reservation only bites when something else wants the last slot.

    A text-only target quoting a text-only post leaves the whole budget to an image-heavy
    ancestor, so without the reservation the ancestor's tenth image takes the slot and the quoted
    post disappears from a message that is supposed to be about it.
    """
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    ancestor = _thread_output(
        text="ancestor",
        author_name="root",
        image_urls=[f"https://example.test/{index}.png" for index in range(10)],
    )
    target = _thread_output(text="commentary")
    target.quoted = _thread_output(text="the post being argued with", author_name="bob")

    embeds = cog._build_embed_plan(results=[ancestor, target]).embeds

    assert len(embeds) == 10
    descriptions = [embed.description or "" for embed in embeds]
    assert any(text.startswith("🔗 **被引用的貼文**") for text in descriptions)
    assert "commentary" in descriptions
    # The ancestor gives up two of its ten images, not the target or the quoted post.
    assert sum(1 for embed in embeds if embed.image) == 8


async def test_threads_cog_says_nothing_about_an_ancestors_quote() -> None:
    """The expansion shows the target's quote only, so an ancestor's must not be half-announced.

    `_build_output` fills `quoted_unavailable` on every parsed post, so an ungated hint told the
    reader about an ancestor's quote in exactly the case where there was nothing to show, while an
    ancestor quoting a live post said nothing at all.
    """
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    root = _thread_output(text="root commentary", author_name="root")
    root.quoted_unavailable = True

    embeds = cog._build_embed_plan(results=[root, _thread_output(text="target")]).embeds

    assert embeds[0].description == "root commentary"
    assert all("引用的貼文目前無法瀏覽" not in (embed.description or "") for embed in embeds)


async def test_threads_cog_measures_the_rendered_description_against_the_embed_limit() -> None:
    """The marker prefix and the hints are appended after any check on the raw body.

    A body sitting just under 4096 therefore crossed it once rendered, turning the ⚠️ skip the
    guard exists for into a Discord 400 and a ❌.
    """
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    target = _thread_output(text="t")
    target.quoted = _thread_output(text="q" * 4096, author_name="bob")

    embeds = cog._build_embed_plan(results=[target]).embeds

    # The guard now reads exactly this quantity, so it sees the overflow the raw text hid.
    assert max(len(embed.description or "") for embed in embeds) > 4096


async def test_threads_cog_refuses_an_oversize_quoted_post_with_a_warning() -> None:
    """The user-visible outcome of that overflow is the ⚠️ skip, never the ❌ a 400 would give."""
    cog = ThreadsCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    target = _thread_output(text="t")
    target.quoted = _thread_output(text="q" * 4096, author_name="bob")
    _wire_threads(cog=cog, downloader=ThreadsDownloaderStub(results=[target]))

    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    message.__dict__["guild"] = SimpleNamespace(filesize_limit=25 * 1024 * 1024)
    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == "⚠️"
    assert message.replies == []


async def test_threads_cog_skips_a_message_addressed_to_the_bot() -> None:
    """A mention (or a DM) hands the link to gen_reply, so the cog must not also expand it."""
    bot = SimpleNamespace(user=SimpleNamespace(id=999))
    cog = ThreadsCogs(bot=as_bot(fake=bot))
    _wire_threads(
        cog=cog, downloader=ThreadsDownloaderStub(results=RuntimeError("must not be called"))
    )

    mentioned = FakeDiscordMessage()
    mentioned.__dict__["author"] = FakeUser(bot=False)
    mentioned.__dict__["content"] = "<@999> https://www.threads.net/@alice/post/abc"
    mentioned.__dict__["guild"] = SimpleNamespace(filesize_limit=25 * 1024 * 1024)
    await cog.on_message(message=as_message(fake=mentioned))
    assert mentioned.reactions == []
    assert mentioned.replies == []

    direct_message = FakeDiscordMessage()
    direct_message.__dict__["author"] = FakeUser(bot=False)
    direct_message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    direct_message.__dict__["guild"] = None  # a DM always reaches gen_reply, mention or not
    await cog.on_message(message=as_message(fake=direct_message))
    assert direct_message.reactions == []
    assert direct_message.replies == []


async def test_threads_cog_hosts_oversized_video(tmp_path: Path) -> None:
    """A Threads video too big to attach is hosted as a URL instead of a ⚠️ refusal."""
    bot = SimpleNamespace(user=SimpleNamespace(id=999))
    cog = ThreadsCogs(bot=as_bot(fake=bot))
    (tmp_path / "serve").mkdir()  # pre-existing host mount; the bot never creates the serve dir
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(
            config=make_media_hosting_config(
                enabled=True, base_url="https://media.test", serve_dir=str(tmp_path / "serve")
            )
        )
    )
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(data=b"123")

    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    # The 1 MiB envelope margin alone overshoots this ceiling, so no combined body ever fits
    # and even a 3-byte video is peeled out to a hosted URL.
    message.__dict__["guild"] = SimpleNamespace(filesize_limit=4)
    _wire_threads(
        cog=cog,
        downloader=ThreadsDownloaderStub(
            results=[_thread_output(video_paths=[video_file], image_urls=[])]
        ),
    )

    await cog.on_message(message=as_message(fake=message))

    # The video was hosted (its URL rides the reply content) and moved out of the temp dir.
    content = message.replies[0].get("content") or ""
    assert any(line.startswith("https://media.test/") for line in content.splitlines())
    assert not video_file.exists()
    assert message.reactions[-1] == "<:greencheck:1517565102424068226>"


async def test_threads_cog_mixes_native_and_hosted_videos(tmp_path: Path) -> None:
    """A post with one small and one oversize video attaches the small and links only the big one."""
    bot = SimpleNamespace(user=SimpleNamespace(id=999))
    cog = ThreadsCogs(bot=as_bot(fake=bot))
    (tmp_path / "serve").mkdir()  # pre-existing host mount; the bot never creates the serve dir
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(
            config=make_media_hosting_config(
                enabled=True, base_url="https://media.test", serve_dir=str(tmp_path / "serve")
            )
        )
    )
    small = tmp_path / "small.mp4"
    small.write_bytes(data=b"0" * 100)
    big = tmp_path / "big.mp4"
    big.write_bytes(data=b"0" * (2 * 1024 * 1024))

    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    # The ceiling clears the small clip plus the 1 MiB envelope margin but not the 2 MiB clip,
    # so only the big one is peeled to a hosted URL while the small one attaches natively.
    message.__dict__["guild"] = SimpleNamespace(filesize_limit=1024 * 1024 + 200)
    _wire_threads(
        cog=cog,
        downloader=ThreadsDownloaderStub(
            results=[_thread_output(video_paths=[small, big], image_urls=[])]
        ),
    )

    await cog.on_message(message=as_message(fake=message))

    content = message.replies[0].get("content") or ""
    hosted = [line for line in content.splitlines() if line.startswith("https://media.test/")]
    assert len(hosted) == 1  # only the oversize clip was linked
    assert big.exists() is False  # the big clip was moved into the serve dir
    assert small.exists() is True  # the small clip stayed on disk to attach natively
    assert message.reactions[-1] == "<:greencheck:1517565102424068226>"


async def test_threads_cog_refuses_oversized_video_when_hosting_off(tmp_path: Path) -> None:
    """With hosting off, an oversize Threads video refuses the whole post (pre-#325 ⚠️ behavior)."""
    bot = SimpleNamespace(user=SimpleNamespace(id=999))
    cog = ThreadsCogs(bot=as_bot(fake=bot))
    # Explicitly disabled planner — never the no-arg default, whose config is `available` on a dev
    # box where .env enables hosting (it would write into the live serve dir).
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(config=make_media_hosting_config(enabled=False))
    )
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(data=b"123")

    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "https://www.threads.net/@alice/post/abc"
    message.__dict__["guild"] = SimpleNamespace(filesize_limit=4)  # tiny ceiling -> video oversize
    _wire_threads(
        cog=cog,
        downloader=ThreadsDownloaderStub(
            results=[_thread_output(video_paths=[video_file], image_urls=[])]
        ),
    )

    await cog.on_message(message=as_message(fake=message))

    # No host available + oversize -> whole-post ⚠️ refusal, no reply, and the file is left in place.
    assert message.reactions[-1] == "⚠️"
    assert message.replies == []
    assert video_file.exists() is True


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
    cog = AutoUnmuteCogs(bot=as_bot(fake=bot))

    guild = cast(
        "Guild",
        SimpleNamespace(
            id=123,
            name="Guild",
            get_channel=lambda channel_id: channel,
            system_channel=None,
            audit_logs=lambda action, limit: _audit_entries(bot_user),
        ),
    )
    await cog.on_message(
        message=as_message(
            fake=SimpleNamespace(
                guild=guild, author=FakeUser(bot=False), channel=SimpleNamespace(id=456)
            )
        )
    )
    assert cog._last_active_channel == {123: 456}

    monkeypatch.setattr(auto_unmute, "Messageable", FakeSendChannel)
    assert cog._resolve_channel(guild=guild) is channel
    moderator, reason = await cog._lookup_audit(guild=guild)
    assert moderator is not None
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

    self_timeout_until = datetime.now(tz=UTC) + timedelta(minutes=5)
    member = cast(
        "Member",
        SimpleNamespace(
            id=999,
            guild=guild,
            communication_disabled_until=self_timeout_until,
            edit=lambda **kwargs: _async_none(),
        ),
    )
    await cog._handle_self_timeout(member=member, until=self_timeout_until)
    assert sent == ["not today"]

    before = cast("Member", SimpleNamespace(communication_disabled_until=None))
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
) -> FakeGeneratedResponse:
    """Returns a deterministic auto-unmute response."""
    return FakeGeneratedResponse(output_text="not today")


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
    monkeypatch.setattr(economy, "buy_vip", fake_buy_vip)
    bot = SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))
    cog = EconomyCogs(bot=as_bot(fake=bot))
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
    await EconomyCogs.vip_command.callback(cog, interaction)
    assert len(interaction.followup.sent) == 17
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
    # Assert the financial summary's structure and the facade values it surfaces, not the exact
    # localized title/labels: cash 150, debt principal 30, net worth 115.
    assert_embed_title_prefix(embed=balance_embed, prefix="💰")
    assert "115" in (balance_embed.description or "")
    cash_value = assert_embed_has_field(embed=balance_embed, name="現金").value
    assert cash_value is not None
    assert "150" in cash_value
    debt_value = assert_embed_has_field(embed=balance_embed, name="債務").value
    assert debt_value is not None
    assert "30" in debt_value
    borrow_embed = interaction.followup.sent[8]["embed"]
    # The footer explains the loan-decision timeout; assert the behavioral 180s, not the copy.
    assert "180" in (borrow_embed.footer.text or "")
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
    inspected_description = inspected_member.followup.sent[0]["embed"].description
    assert inspected_description is not None
    assert "Bob" in inspected_description

    bot_receiver = FakeInteraction(user=FakeUser(user_id=1))
    await EconomyCogs.give.callback(
        cog, bot_receiver, member=FakeUser(user_id=3, name="bot", bot=True), amount="1"
    )
    bot_receiver_title = bot_receiver.followup.sent[0]["embed"].title
    assert bot_receiver_title is not None
    assert "轉帳完成" in bot_receiver_title


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
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))),
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
    await approve_button.callback(as_interaction(fake=denied))
    assert denied.response.sent[0]["ephemeral"] is True
    assert captured_accept_kwargs == {}

    allowed = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    await approve_button.callback(as_interaction(fake=allowed))
    assert captured_accept_kwargs["proposal_id"] == 42
    assert captured_accept_kwargs["actor_id"] == 1
    assert captured_accept_kwargs["allow_central_bank_self_approval"] is True
    assert allowed.response.edited[0]["view"] is None

    cancel_view = CentralBankLoanDecisionView(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))),
        proposal_id=43,
        creator_id=1,
    )
    cancel_button = next(
        child
        for child in cancel_view.children
        if getattr(child, "custom_id", "") == "central_bank:cancel"
    )
    denied_cancel = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await cancel_button.callback(as_interaction(fake=denied_cancel))
    assert denied_cancel.response.sent[0]["ephemeral"] is True
    assert captured_cancel_kwargs == {}

    allowed_cancel = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    await cancel_button.callback(as_interaction(fake=allowed_cancel))
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
    await approve_button.callback(as_interaction(fake=denied_approve))
    assert denied_approve.response.sent[0]["ephemeral"] is True
    assert captured_accept_kwargs == {}

    allowed_approve = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await approve_button.callback(as_interaction(fake=allowed_approve))
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
    await reject_button.callback(as_interaction(fake=denied_reject))
    assert denied_reject.response.sent[0]["ephemeral"] is True
    assert captured_reject_kwargs == {}

    allowed_reject = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await reject_button.callback(as_interaction(fake=allowed_reject))
    assert captured_reject_kwargs == {"proposal_id": 43, "actor_id": 2}
    assert allowed_reject.response.edited[0]["view"] is None

    cancel_view = CreditLoanDecisionView(proposal_id=44, lender_id=2, creator_id=1)
    cancel_button = next(
        child
        for child in cancel_view.children
        if getattr(child, "custom_id", "") == "credit:cancel"
    )
    denied_cancel = FakeInteraction(user=FakeUser(user_id=2, name="bob"))
    await cancel_button.callback(as_interaction(fake=denied_cancel))
    assert denied_cancel.response.sent[0]["ephemeral"] is True
    assert captured_cancel_kwargs == {}

    allowed_cancel = FakeInteraction(user=FakeUser(user_id=1, name="alice"))
    await cancel_button.callback(as_interaction(fake=allowed_cancel))
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
    credit_view.message = as_message(fake=credit_message)
    await credit_view.on_timeout()

    central_message = FakeDiscordMessage()
    central_view = CentralBankLoanDecisionView(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer"))),
        proposal_id=43,
        creator_id=1,
    )
    central_view.message = as_message(fake=central_message)
    await central_view.on_timeout()

    # order-contract: each `on_timeout` is awaited to completion before the next view exists.
    assert rejected == [42, 43]
    # order-contract: same sequential awaits, so cleanup is scheduled in construction order.
    assert scheduled == [credit_message, central_message]
    assert credit_message.edits[0]["view"] is None
    assert central_message.edits[0]["view"] is None
    credit_timeout_title = credit_message.edits[0]["embed"].title
    assert credit_timeout_title is not None
    assert "逾時" in credit_timeout_title
    central_timeout_title = central_message.edits[0]["embed"].title
    assert central_timeout_title is not None
    assert "逾時" in central_timeout_title


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
    cog = EconomyCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.admin_refund_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="100"
    )

    assert called is False
    assert interaction.followup.sent[0].get("ephemeral") is True
    admin_rejection_title = interaction.followup.sent[0]["embed"].title
    assert admin_rejection_title is not None
    assert "權限不足" in admin_rejection_title


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
    cog = EconomyCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.admin_refund_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="9,007,199,254,740,993"
    )
    await EconomyCogs.admin_collect_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="9,007,199,254,740,993"
    )

    # order-contract: each awaited command completes its balance adjustment before returning.
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
    cog = EconomyCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )
    interaction = FakeInteraction(user=FakeUser(user_id=1))
    bot_member = FakeUser(user_id=999, name="discordbot", display_name="Dealer", bot=True)

    await EconomyCogs.admin_refund_tax.callback(cog, interaction, member=bot_member, amount="100")
    await EconomyCogs.admin_collect_tax.callback(cog, interaction, member=bot_member, amount="50")

    # order-contract: each awaited command completes its balance adjustment before returning.
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
    cog = EconomyCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.admin_collect_tax.callback(
        cog, interaction, member=FakeUser(user_id=2, name="bob"), amount="not a number"
    )

    assert called is False
    assert interaction.response.sent[0]["ephemeral"] is True
    assert interaction.response.sent[0]["embed"].title == "收稅失敗"
    collect_tax_description = interaction.response.sent[0]["embed"].description
    assert collect_tax_description is not None
    assert "金額格式錯誤" in collect_tax_description
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
        return TransferResult(
            sender_balance=50, receiver_balance=100, received_amount=100, tax_amount=0
        )

    sender = FakeUser(user_id=1, name="alice")
    receiver = FakeUser(user_id=2, name="bob")
    cached_sender = FakeUser(user_id=1, name="alice")
    cached_sender.__dict__["guild_avatar"] = SimpleNamespace(
        url="https://example.test/alice-server.png"
    )
    cached_receiver = FakeUser(user_id=2, name="bob")
    cached_receiver.__dict__["guild_avatar"] = SimpleNamespace(
        url="https://example.test/bob-server.png"
    )
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
    cog = EconomyCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )

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
        return TransferResult(
            sender_balance=50, receiver_balance=100, received_amount=100, tax_amount=0
        )

    sender = FakeUser(user_id=1, name="alice")
    bot_receiver = FakeUser(user_id=999, name="discordbot", display_name="Dealer", bot=True)
    interaction = FakeInteraction(user=sender)
    monkeypatch.setattr(economy, "transfer", record_transfer)
    monkeypatch.setattr(
        interactions, "schedule_public_message_delete", ignore_scheduled_public_message
    )
    cog = EconomyCogs(bot=as_bot(fake=SimpleNamespace(user=bot_receiver)))

    await EconomyCogs.give.callback(cog, interaction, member=bot_receiver, amount="100")

    assert captured_transfer == {
        "sender_id": 1,
        "sender_name": "alice",
        "receiver_id": 999,
        "receiver_name": "discordbot",
        "amount": 100,
    }
    give_bot_receiver_title = interaction.followup.sent[0]["embed"].title
    assert give_bot_receiver_title is not None
    assert "轉帳完成" in give_bot_receiver_title


async def test_economy_money_commands_accept_large_string_amounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loan, transfer, and collection amounts parse beyond Discord integer option limits."""
    big_amount = 9_007_199_254_740_993
    captured: dict[str, int | None] = {}

    async def record_transfer(**kwargs: Any) -> TransferResult:  # noqa: ANN401 -- command facade double
        captured["give"] = kwargs["amount"]
        return TransferResult(
            sender_balance=0, receiver_balance=0, received_amount=0, tax_amount=0
        )

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
    cog = EconomyCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )
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
        return TransferResult(
            sender_balance=0, receiver_balance=0, received_amount=0, tax_amount=0
        )

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
    cog = EconomyCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )
    member = FakeUser(user_id=2, name="bob")

    def assert_rejected(interaction: FakeInteraction, expected_title: str) -> None:
        """Asserts an ephemeral malformed-amount rejection with no mutation followup."""
        assert interaction.response.sent[0]["ephemeral"] is True
        assert interaction.response.sent[0]["embed"].title == expected_title
        rejection_description = interaction.response.sent[0]["embed"].description
        assert rejection_description is not None
        assert "金額格式錯誤" in rejection_description
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
    cog = EconomyCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.loss_leaderboard.callback(cog, interaction)

    embed = interaction.followup.sent[0]["embed"]
    assert embed.title is not None
    assert "今日輸局累計" in embed.title
    assert embed.description is not None
    assert "累計輸" in embed.description
    assert interaction.followup.sent[0]["files"][0].filename == "economy_loss_leaderboard.png"
    assert embed.footer.text is not None
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
    cog = EconomyCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )
    interaction = FakeInteraction(user=FakeUser(user_id=1))

    await EconomyCogs.loss_leaderboard.callback(cog, interaction)

    embed = interaction.followup.sent[0]["embed"]
    assert embed.title is not None
    assert "今日輸局累計" in embed.title
    assert embed.description is not None
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
    return TransferResult(
        sender_balance=50, receiver_balance=100, received_amount=100, tax_amount=0
    )


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


async def fake_buy_vip(user_id: int, name: str, avatar_url: str) -> VipPurchaseResult:
    """Returns a successful fake VIP purchase result."""
    return VipPurchaseResult(new_balance=500_000, cost=VIP_PURCHASE_COST)


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
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )

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
    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )
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

    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )

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

    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )

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
    monkeypatch: pytest.MonkeyPatch, economy_isolated_db: None
) -> None:
    """Verifies owner over-betting clamps the shared Blackjack lobby bet."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")

    async def balance_by_user(user_id: int) -> int:
        """Returns distinct balances for owner, joining player, and the bot."""
        return {1: 300, 2: 50_000_000, 999: 0}[user_id]

    monkeypatch.setattr(games, "get_balance", balance_by_user)

    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )

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
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")

    async def balance_by_user(user_id: int) -> int:
        """Returns enough balance for the owner and bot to keep their queued bets."""
        return {1: 500, 999: 1_000}[user_id]

    monkeypatch.setattr(games, "get_balance", balance_by_user)
    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )
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
    """A large formatted bet is parsed without error but caps at MAX_SINGLE_BET."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")

    async def balance_by_user(user_id: int) -> int:
        """Returns enough balance to cover a large formatted wager."""
        del user_id
        return 10_000_000_000_000_000

    monkeypatch.setattr(games, "get_balance", balance_by_user)

    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="9,007,199,254,740,993")
    lobby_view = owner_interaction.followup.sent[0]["view"]

    assert isinstance(lobby_view, BlackjackLobbyView)
    # The wager is parsed and the lobby is created, but the single-bet cap applies.
    assert lobby_view.requested_bet == 1_000_000
    assert lobby_view.participants[0].bet == 1_000_000


async def test_blackjack_string_bet_rejects_invalid_text() -> None:
    """Verifies invalid text is rejected before wager preparation."""
    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )

    owner_interaction = FakeInteraction(user=FakeUser(user_id=1))
    await GamesCogs.blackjack.callback(cog, owner_interaction, bet="not a number")

    assert owner_interaction.response.sent[0]["ephemeral"] is True
    assert owner_interaction.response.sent[0]["embed"].title == "下注格式錯誤"
    assert owner_interaction.response.sent[0]["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert owner_interaction.response.sent[0]["embed"].image.url == embed_spacer_url()
    assert owner_interaction.followup.sent == []
    assert owner_interaction.response.deferred is False


async def test_blackjack_owner_zero_bet_caps_all_in_at_max_single_bet(
    monkeypatch: pytest.MonkeyPatch, economy_isolated_db: None
) -> None:
    """Bet zero means all in, but a huge balance still caps at MAX_SINGLE_BET."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")

    async def balance_by_user(user_id: int) -> int:
        """Returns a large owner balance that exceeds the single-bet cap."""
        return {1: 300_000_000_000_000, 2: 500_000_000_000_000, 999: 0}[user_id]

    monkeypatch.setattr(games, "get_balance", balance_by_user)

    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )

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
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    monkeypatch.setattr(games, "schedule_public_message_delete", ignore_scheduled_public_message)
    monkeypatch.setattr(games, "get_balance", _empty_game_balance)

    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )

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

    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )

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

    cog = GamesCogs(
        bot=as_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, display_name="Dealer")))
    )

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
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
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


def test_setup_functions_register_cogs(monkeypatch: pytest.MonkeyPatch) -> None:
    """EVERY cog directory's sync `setup` adds its own cog, with `override=True`.

    Swept off the same directory scan `_load_cogs_sync` performs rather than a hand-written
    list, which is what left half the cogs uncovered before: `setup` is the one function in
    a cog module the loader calls by name, and an `async def setup` here returns a coroutine
    nothing awaits, breaking the first command sync with nothing raised.
    """
    added: list[tuple[commands.Cog, bool | None]] = []

    def record_cog(cog: commands.Cog, override: bool | None = None) -> None:
        """Records the cog instance and override flag passed to add_cog."""
        added.append((cog, override))

    bot = SimpleNamespace(add_cog=record_cog)
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    cogs_dir = Path(cli.__file__).resolve().parent / "cogs"
    names = sorted(entry.name for entry in cogs_dir.iterdir() if (entry / "cog.py").is_file())

    assert names  # a scan that found nothing would pass every assertion below
    for name in names:
        module = import_module(f"discordbot.cogs.{name}.cog")
        module.setup(bot=as_bot(fake=bot))
        cog, override = added[-1]
        # `override=True` is what lets a reload replace the cog instead of colliding, and
        # the module check is what stops a re-export standing in for a missing cog.
        assert override is True, name
        assert type(cog).__module__ == module.__name__, name
    assert len(added) == len(names)


def test_cli_load_cogs_sync_discovers_exactly_the_cog_directories(tmp_path: Path) -> None:
    """Verifies synchronous cog loading discovers exactly the cog directories."""
    loaded: list[tuple[list[str], bool]] = []

    def record_load_extensions(modules: list[str], stop_at_error: bool) -> None:
        """Records modules passed to load_extensions."""
        loaded.append((modules, stop_at_error))

    bot = SimpleNamespace(load_extensions=record_load_extensions)
    cli.DiscordBot._load_cogs_sync(as_discord_bot(fake=bot))
    assert loaded[0][1] is True
    # An exact set, not a membership check: a discovery rule that grew a nested helper
    # package or lost a cog would still contain any single name you happened to test for.
    cogs_dir = Path(cli.__file__).resolve().parent / "cogs"
    expected = {
        f"discordbot.cogs.{entry.name}.cog"
        for entry in cogs_dir.iterdir()
        if entry.is_dir() and (entry / "cog.py").is_file()
    }
    assert set(loaded[0][0]) == expected
    assert "discordbot.cogs.template.cog" in expected


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
        _message_reward_at={},
    )
    user_message = SimpleNamespace(author=FakeUser(user_id=1, bot=False))
    await cli.DiscordBot.on_message(
        as_discord_bot(fake=bot), message=as_message(fake=user_message)
    )
    assert processed == [user_message]
    assert rewards[0]["amount"] == cli.BASE_MESSAGE_REWARD_AMOUNT
    await cli.DiscordBot.on_message(
        as_discord_bot(fake=bot), message=as_message(fake=SimpleNamespace(author=bot.user))
    )
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
        command=SimpleNamespace(qualified_name="demo"),
    )
    await cli.DiscordBot.on_command_error(
        as_discord_bot(fake=bot), as_command_context(fake=context), commands.NotOwner()
    )
    await cli.DiscordBot.on_command_error(
        as_discord_bot(fake=bot),
        as_command_context(fake=context),
        commands.MissingPermissions(missing_permissions=["kick_members"]),
    )
    await cli.DiscordBot.on_command_error(
        as_discord_bot(fake=bot),
        as_command_context(fake=context),
        commands.BotMissingPermissions(missing_permissions=["send_messages"]),
    )
    await cli.DiscordBot.on_command_error(
        as_discord_bot(fake=bot),
        as_command_context(fake=context),
        commands.CommandNotFound("nope"),
    )
    assert len(sent) == 4

    # An error the handler has no branch for used to vanish at no level at all; it now
    # reports the type nextcord wrapped, not the wrapper.
    logged: list[dict[str, Any]] = []

    def record_error(_message: str, **kwargs: Any) -> None:  # noqa: ANN401 -- logfire accepts arbitrary fields
        """Records the unhandled-command-error log."""
        logged.append(kwargs)

    monkeypatch.setattr(cli.logfire, "error", record_error)
    await cli.DiscordBot.on_command_error(
        as_discord_bot(fake=bot),
        as_command_context(fake=context),
        commands.CommandInvokeError(ValueError("boom")),
    )
    assert len(sent) == 4
    assert logged[-1]["error_type"] == "ValueError"
    assert logged[-1]["command"] == "demo"


def test_log_level_setting_accepts_only_real_logfire_levels() -> None:
    """`LOG_LEVEL` is checked against logfire's own table, not a hand-copied list."""
    accepted = set(get_args(LoggingConfig.model_fields["log_level"].annotation))
    assert accepted == set(LEVEL_NUMBERS)


async def test_cli_message_reward_cooldown_suppresses_rapid_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second message within the cooldown earns nothing; a later one earns again."""
    rewards: list[dict[str, Any]] = []

    async def record_reward(**kwargs: Any) -> CreditResult:  # noqa: ANN401 -- command facade double
        rewards.append(kwargs)
        return CreditResult(
            new_balance=10, credited_amount=10, principal_repaid=0, remaining_debt=0
        )

    async def noop_process(message: SimpleNamespace) -> None:
        del message

    monkeypatch.setattr(target=cli, name="credit_with_repayment", value=record_reward)
    bot = SimpleNamespace(
        user=FakeUser(user_id=999, bot=True), process_commands=noop_process, _message_reward_at={}
    )
    message = SimpleNamespace(author=FakeUser(user_id=1, bot=False))

    await cli.DiscordBot.on_message(as_discord_bot(fake=bot), message=as_message(fake=message))
    await cli.DiscordBot.on_message(as_discord_bot(fake=bot), message=as_message(fake=message))
    assert len(rewards) == 1

    # Backdate the last-reward stamp so the cooldown window has elapsed.
    bot._message_reward_at[1] -= cli.MESSAGE_REWARD_COOLDOWN_SECONDS + 1
    await cli.DiscordBot.on_message(as_discord_bot(fake=bot), message=as_message(fake=message))
    assert len(rewards) == 2


async def test_cli_message_reward_cooldown_prunes_expired_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expired per-user cooldown slots are dropped lazily on later messages."""
    rewards: list[dict[str, Any]] = []

    async def record_reward(**kwargs: Any) -> CreditResult:  # noqa: ANN401 -- command facade double
        rewards.append(kwargs)
        return CreditResult(
            new_balance=10, credited_amount=10, principal_repaid=0, remaining_debt=0
        )

    async def noop_process(message: SimpleNamespace) -> None:
        del message

    monkeypatch.setattr(target=cli, name="credit_with_repayment", value=record_reward)
    monkeypatch.setattr(target=cli, name="monotonic", value=lambda: 1_000.0)
    bot = SimpleNamespace(
        user=FakeUser(user_id=999, bot=True),
        process_commands=noop_process,
        _message_reward_at={1: 900.0, 2: 975.0},
        _message_reward_pruned_at=0.0,
    )

    await cli.DiscordBot.on_message(
        as_discord_bot(fake=bot),
        message=as_message(fake=SimpleNamespace(author=FakeUser(user_id=3, bot=False))),
    )

    assert 1 not in bot._message_reward_at
    assert bot._message_reward_at[2] == 975.0
    assert bot._message_reward_at[3] == 1_000.0
    assert len(rewards) == 1


async def test_cli_message_reward_cooldown_rolls_back_on_credit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed credit must not leave the user on cooldown for the next message."""
    attempts = 0

    async def flaky_reward(**kwargs: Any) -> CreditResult:  # noqa: ANN401 -- command facade double
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient DB failure")
        return CreditResult(
            new_balance=10, credited_amount=10, principal_repaid=0, remaining_debt=0
        )

    async def noop_process(message: SimpleNamespace) -> None:
        del message

    monkeypatch.setattr(target=cli, name="credit_with_repayment", value=flaky_reward)
    bot = SimpleNamespace(
        user=FakeUser(user_id=999, bot=True), process_commands=noop_process, _message_reward_at={}
    )
    message = SimpleNamespace(author=FakeUser(user_id=1, bot=False))

    await cli.DiscordBot.on_message(as_discord_bot(fake=bot), message=as_message(fake=message))
    # The first credit failed, so the slot is rolled back and the next message retries.
    assert 1 not in bot._message_reward_at
    await cli.DiscordBot.on_message(as_discord_bot(fake=bot), message=as_message(fake=message))
    assert attempts == 2
    assert bot._message_reward_at.get(1) is not None
