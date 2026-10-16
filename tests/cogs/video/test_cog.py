"""Tests for the `/download_video` command."""

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, get_args
from pathlib import Path
import threading

import pytest
import nextcord

from discordbot.cogs.video import cog as video
from discordbot.typings.video import VideoQuality
from discordbot.cogs.video.cog import QUALITY_CHOICES, VideoCogs, douyin_failure_message
from discordbot.services.platforms.ytdlp import DownloadResult, VideoDownloader
from discordbot.services.platforms.douyin import (
    DouyinError,
    DouyinDownload,
    DouyinDownloader,
    DouyinBlockedError,
    DouyinTransferError,
    DouyinUnavailableError,
)

from tests.helpers.casting import as_bot, as_interaction
from tests.helpers.link_sources import hosting_planner
from tests.helpers.discord_mocks import FakeInteraction
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from aiohttp import ClientResponse

_DOUYIN_VIDEO_URL = "https://www.douyin.com/video/7664447317017136422"
_DOUYIN_NOTE_URL = "https://www.douyin.com/note/7159955455492541733"


def test_every_quality_preset_is_answered_everywhere() -> None:
    """A preset added to the type has to be answered by every site that maps one.

    The option's own default is read off the registered command rather than spelled out here:
    nextcord types `SlashOption(default=...)` as `Any`, so it is the one preset site `ty`
    cannot see, and it is the value every `/download_video` without an explicit quality carries.
    """
    presets = set(get_args(VideoQuality))

    assert set(VideoDownloader.quality_formats) == presets
    assert set(DouyinDownloader.quality_ratios) == presets
    assert set(QUALITY_CHOICES.values()) == presets

    cog = VideoCogs(bot=as_bot(fake=object()))
    assert cog.download_video.options["quality"].default in presets


def test_every_option_is_described_in_each_locale_the_command_is() -> None:
    """An option left in English reads wrong under a command named in the caller's language."""
    cog = VideoCogs(bot=as_bot(fake=object()))
    command_locales = set(cog.download_video.description_localizations or {})

    for name, option in cog.download_video.options.items():
        assert set(option.description_localizations or {}) == command_locales, name


def test_a_quality_label_names_what_each_downloader_asks_for() -> None:
    """A label naming a resolution names exactly the ones requested, Douyin's where it differs."""
    for label, preset in QUALITY_CHOICES.items():
        named = set(re.findall(pattern=r"(\d+)p", string=label))
        if not named:
            continue
        ytdlp = set(
            re.findall(pattern=r"height<=(\d+)", string=VideoDownloader.quality_formats[preset])
        )
        douyin = DouyinDownloader.quality_ratios[preset].removesuffix("p")
        assert named == {*ytdlp, douyin}, label
        if douyin not in ytdlp:
            assert f"{douyin}p on Douyin" in label, label


class _CannedDownloader:
    """Stands in for whichever downloader `/download_video` picks, answering one canned outcome."""

    def __init__(self, outcome: DownloadResult | DouyinDownload | Exception) -> None:
        """Holds the finished download to answer with, or the error to raise."""
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def download(self, **kwargs: object) -> DownloadResult | DouyinDownload:
        """Records what the command asked for, then answers the canned outcome."""
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _install(
    monkeypatch: pytest.MonkeyPatch, outcome: DownloadResult | DouyinDownload | Exception
) -> tuple[VideoCogs, _CannedDownloader]:
    """Builds the cog with hosting off and both of its downloaders answering `outcome`."""
    cog = VideoCogs(bot=as_bot(fake=object()))
    stub = _CannedDownloader(outcome=outcome)
    monkeypatch.setattr(target=video, name="DouyinDownloader", value=lambda output_folder: stub)
    monkeypatch.setattr(target=video, name="VideoDownloader", value=lambda output_folder: stub)
    return cog, stub


async def test_download_video_extracts_a_url_from_share_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A share blob pasted into the command reaches the downloader as its bare link.

    Share buttons wrap the URL in copy, so a command that only accepted a bare URL would fail on
    the most natural thing to paste. Both downloaders are stubbed, so a blob that went through
    whole would be caught at whichever one it reached.
    """
    cog, stub = _install(monkeypatch=monkeypatch, outcome=RuntimeError("stop here"))
    blob = (
        "8.46 Y@m.QX :9pm UYm:/ 06/01 短片《临时司机》#AI短片# 内容过于真实 "
        "https://v.douyin.com/tLgj3lCAnds 复制此链接，打开Dou音搜索，直接观看视频"
    )

    await VideoCogs.download_video.callback(
        cog, as_interaction(fake=FakeInteraction()), url=blob, quality="best"
    )

    assert [call["url"] for call in stub.calls] == ["https://v.douyin.com/tLgj3lCAnds"]


async def test_video_deliver_and_download_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies video delivery, oversize URL fallback, hosting-off, and download error branches."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    small = tmp_path / "small.mp4"
    small.write_bytes(data=b"0" * (300 * 1024))
    big = tmp_path / "big.mp4"
    big.write_bytes(data=b"0" * 300)

    # Fits: attached natively, under its size and source lines.
    cog, _ = _install(monkeypatch=monkeypatch, outcome=DownloadResult(filename=small))
    interaction = FakeInteraction()
    await VideoCogs.download_video.callback(
        cog, interaction, url="https://source.test/video", quality="best"
    )
    content = interaction.edits[-1]["content"]
    assert "0.3MB" in content
    assert "<https://source.test/video>" in content
    assert interaction.edits[-1]["file"].filename == "small.mp4"
    assert interaction.followup.sent == []

    # Too big for native upload + hosting on: post the URL, no 480p retry, no attachment.
    cog, downloader = _install(monkeypatch=monkeypatch, outcome=DownloadResult(filename=big))
    cog.media_delivery = hosting_planner(serve_dir=serve_dir)
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

    # Too big + hosting off: fall back to the "file too large" message, which names the file's
    # size and the limit it exceeds as two different numbers.
    big2 = tmp_path / "big2.mp4"
    big2.write_bytes(data=b"0" * (3 * 1024 * 1024))
    cog, _ = _install(monkeypatch=monkeypatch, outcome=DownloadResult(filename=big2))
    fail_interaction = FakeInteraction(filesize_limit=2 * 1024 * 1024)
    await VideoCogs.download_video.callback(
        cog, fail_interaction, url="https://x.test", quality="best"
    )
    fail_content = fail_interaction.edits[-1]["content"]
    assert "3.0MB" in fail_content
    assert "2MB" in fail_content

    cog, _ = _install(monkeypatch=monkeypatch, outcome=RuntimeError("download failed"))
    warnings = capture_logs(monkeypatch=monkeypatch, level="warn")
    error_interaction = FakeInteraction()
    await VideoCogs.download_video.callback(
        cog, error_interaction, url="https://x.test", quality="best"
    )
    assert error_interaction.edits[-1]["content"] == video._DOWNLOAD_FAILED
    assert [message for message, _ in warnings] == ["Video download failed"]


async def test_video_ceiling_comes_from_the_interaction_the_reply_goes_out_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interaction is what carries the invoker's Nitro allowance, so it sizes the attach."""
    seen: list[object] = []

    def record_ceiling(guild: object, interaction: object = None) -> int:
        """Answers a Nitro-sized ceiling and records which interaction asked."""
        seen.append(interaction)
        return 1024**3

    monkeypatch.setattr(video, "upload_limit_for", record_ceiling)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(data=b"0" * 300)
    cog, _ = _install(monkeypatch=monkeypatch, outcome=DownloadResult(filename=clip))
    interaction = FakeInteraction(filesize_limit=200)

    await VideoCogs.download_video.callback(
        cog, interaction, url="https://source.test/video", quality="best"
    )

    assert seen == [interaction]
    assert interaction.edits[-1]["file"].filename == "clip.mp4"


class _RefusesAttachments(FakeInteraction):
    """Answers an edit that attaches a file with the 413 Discord sends past its real limit."""

    async def edit_original_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- Discord kwargs
        """Raises on an attaching edit; records any other."""
        if "file" in kwargs:
            raise nextcord.HTTPException(
                cast("ClientResponse", SimpleNamespace(status=413, reason="Payload Too Large")),
                {"code": 40005, "message": "Request entity too large"},
            )
        await super().edit_original_message(**kwargs)


async def test_a_refused_attach_is_logged_as_a_delivery_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Discord refusing a finished file is a delivery failure, never a download one.

    The upload limit the planner trusts can exceed what Discord accepts, so a file planned as an
    attachment can still be refused after it downloaded fine. The user sees the same notice.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(data=b"0" * 1024)
    cog, _ = _install(monkeypatch=monkeypatch, outcome=DownloadResult(filename=clip))
    warnings = capture_logs(monkeypatch=monkeypatch, level="warn")
    interaction = _RefusesAttachments()

    await VideoCogs.download_video.callback(
        cog, as_interaction(fake=interaction), url="https://x.test", quality="best"
    )

    assert interaction.edits[-1]["content"] == video._DOWNLOAD_FAILED
    assert [(message, fields["url"], fields["error_type"]) for message, fields in warnings] == [
        ("Video delivery failed", "https://x.test", "HTTPException")
    ]


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
            """Records whether the worker was told to stop."""
            self.aborted = False

        def download(
            self, url: str, quality: str, stop_signal: threading.Event | None = None
        ) -> DownloadResult:
            """Outlasts the command's bound by two orders of magnitude unless told to stop."""
            del url, quality
            self.aborted = stop_signal is not None and stop_signal.wait(timeout=5.0)
            raise RuntimeError("download stopped")

    downloader = StallingDownloader()
    monkeypatch.setattr(video, "VideoDownloader", lambda output_folder: downloader)
    interaction = FakeInteraction()
    await VideoCogs.download_video.callback(
        cog, as_interaction(fake=interaction), url="https://x.test", quality="best"
    )

    assert interaction.edits[-1]["content"] == video._DOWNLOAD_FAILED
    # Any failure prints that same line, so what proves the BOUND fired is which way the worker
    # ended: aborted on the signal rather than running its stall out.
    assert downloader.aborted is True


async def test_cog_routes_douyin_away_from_ytdlp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Douyin link must never reach the yt-dlp downloader, whose extractor cannot serve it."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"0" * 128)
    cog, stub = _install(
        monkeypatch=monkeypatch, outcome=DouyinDownload(is_photo=False, filenames=[clip])
    )

    def _fail(output_folder: str) -> None:
        raise AssertionError("yt-dlp must not be used for a Douyin URL")

    monkeypatch.setattr(target=video, name="VideoDownloader", value=_fail)
    interaction = FakeInteraction()

    await VideoCogs.download_video.callback(
        cog, interaction, url="https://v.douyin.com/NdlfIZPcgz4", quality="best"
    )

    assert stub.calls[0]["url"] == "https://v.douyin.com/NdlfIZPcgz4"
    # The gallery cap is applied at download time, not after.
    assert stub.calls[0]["max_images"] == video.DISCORD_ATTACHMENT_LIMIT
    assert [file.filename for file in interaction.edits[-1]["files"]] == ["clip.mp4"]


async def test_cog_states_how_many_gallery_images_were_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A capped gallery says so; silently sending a partial set would mislead the user."""
    images = []
    for index in range(3):
        image = tmp_path / f"{index}.jpg"
        image.write_bytes(b"0" * 16)
        images.append(image)

    cog, _stub = _install(
        monkeypatch=monkeypatch,
        outcome=DouyinDownload(is_photo=True, filenames=images, total_images=48),
    )
    interaction = FakeInteraction()

    await VideoCogs.download_video.callback(cog, interaction, url=_DOUYIN_NOTE_URL, quality="best")

    content = interaction.edits[-1]["content"]
    # Bounded on both sides: the post id in the source link carries "45" as well.
    assert re.search(pattern=r"(?<!\d)45(?!\d)", string=content)
    assert len(interaction.edits[-1]["files"]) == 3


async def test_cog_keeps_every_url_when_a_whole_gallery_is_hosted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A gallery hosted in full must post every URL, not just the first one.

    The bare-URL reply exists so a lone oversize video renders an inline player; sending a gallery
    down that path would silently discard every image past the first, plus the omitted-count note.
    """
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    images = []
    for index in range(3):
        image = tmp_path / f"{index}.jpg"
        image.write_bytes(bytes([index]) * 4096)  # distinct bytes so hosting cannot dedup them
        images.append(image)

    cog, _stub = _install(
        monkeypatch=monkeypatch,
        outcome=DouyinDownload(is_photo=True, filenames=images, total_images=len(images)),
    )
    cog.media_delivery = hosting_planner(serve_dir=serve_dir)
    interaction = FakeInteraction(filesize_limit=1024)

    await VideoCogs.download_video.callback(cog, interaction, url=_DOUYIN_NOTE_URL, quality="best")

    content = interaction.edits[-1]["content"]
    hosted = [line for line in content.splitlines() if line.startswith("https://media.test/")]
    assert len(hosted) == len(images)
    assert video._DOWNLOAD_FAILED not in content


def test_each_douyin_failure_gets_its_own_wording() -> None:
    """Only a filtered post may read as deleted, and only a bot wall as Douyin refusing.

    The four failures a user should act on differently each get a notice of their own, and
    anything else falls back to the generic failure line. Which notice says "deleted" (刪除)
    and which says "blocked" (擋) is the meaning that must not swap, so only those words are read.
    """
    unavailable = douyin_failure_message(error=DouyinUnavailableError("SYSTEM_ITEM_NOT_EXIST"))
    blocked = douyin_failure_message(error=DouyinBlockedError("challenge"))
    others = [
        douyin_failure_message(error=DouyinTransferError("read timed out")),
        douyin_failure_message(error=TimeoutError()),
        video._DOWNLOAD_FAILED,
    ]
    distinct = [unavailable, blocked, *others]
    assert len(set(distinct)) == len(distinct)
    assert [message for message in distinct if "刪除" in message] == [unavailable]
    assert [message for message in distinct if "擋" in message] == [blocked]
    for error in (DouyinError("boom"), OSError(28, "No space left on device")):
        assert douyin_failure_message(error=error) == video._DOWNLOAD_FAILED


@pytest.mark.parametrize(
    argnames="error",
    argvalues=[
        OSError(28, "No space left on device"),
        DouyinUnavailableError("SYSTEM_ITEM_NOT_EXIST"),
    ],
    ids=["other", "gone"],
)
async def test_cog_answers_a_failed_douyin_download_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """A failed Douyin download answers with its own wording, never an escaping exception.

    The Douyin branch runs before the command's own try block, and the bot's application-command
    error handler cannot take "正在下載影片..." down, so an escaping exception would leave it up
    indefinitely.
    """
    cog, _stub = _install(monkeypatch=monkeypatch, outcome=error)
    interaction = FakeInteraction()

    await VideoCogs.download_video.callback(
        cog, interaction, url=_DOUYIN_VIDEO_URL, quality="best"
    )

    assert interaction.edits[-1]["content"] == douyin_failure_message(error=error)


async def test_cog_posts_the_hosted_url_when_the_clip_is_oversize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An oversize clip is delivered as a hosted URL, the fallback the README advertises.

    Uses a real hosting service because the bugs this guards against only appear once hosting
    actually moves the source file out of the download folder: the URL discarded, or the size
    read off the moved file and quoted as 0.0MB. The clip is big enough that a zero shows.
    """
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"0" * (300 * 1024))

    cog, _stub = _install(
        monkeypatch=monkeypatch, outcome=DouyinDownload(is_photo=False, filenames=[clip])
    )
    cog.media_delivery = hosting_planner(serve_dir=serve_dir)
    interaction = FakeInteraction(filesize_limit=1024)

    await VideoCogs.download_video.callback(
        cog, interaction, url=_DOUYIN_VIDEO_URL, quality="best"
    )

    content = interaction.edits[-1]["content"]
    # Asserted per line rather than as a substring: the URL has to start its own line for Discord
    # to render it, so an anywhere-in-the-body match would accept a message Discord would not link.
    assert any(line.startswith("https://media.test/") for line in content.splitlines())
    assert "0.3MB" in content.splitlines()[0]
    assert video._DOWNLOAD_FAILED not in content
    # The file really was hosted, so discarding the URL would have lost a completed upload.
    assert list(serve_dir.glob("*.mp4"))
