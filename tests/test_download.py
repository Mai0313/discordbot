"""Tests for the yt-dlp downloader facade and the `/download_video` command."""

import re
import time
from types import TracebackType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, cast, get_args
from pathlib import Path
import threading

import pytest
import nextcord
from requests.exceptions import RequestException

from discordbot.cogs.video import cog as video
from discordbot.typings.video import VideoQuality
from discordbot.cogs.video.cog import QUALITY_CHOICES, VideoCogs, douyin_failure_message
from discordbot.services.platforms import ytdlp as downloader_module
from discordbot.services.platforms.ytdlp import (
    DownloadResult,
    VideoDownloader,
    DownloadStoppedError,
)
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

# What `extract_info` answers for a finished download: the least `download` reads a result from.
_DOWNLOADED_INFO = {"id": "video_id", "ext": "mp4"}

_DOUYIN_VIDEO_URL = "https://www.douyin.com/video/7664447317017136422"
_DOUYIN_NOTE_URL = "https://www.douyin.com/note/7159955455492541733"


def _install_youtube_dl_stub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, info: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Installs a yt-dlp stub answering `extract_info` with `info`; returns params and calls."""
    captured_params: list[dict[str, Any]] = []
    captured_calls: list[dict[str, Any]] = []

    class _YoutubeDLStub:
        """Small context-manager stub for yt-dlp."""

        def __init__(self, params: dict[str, Any]) -> None:
            """Records the yt-dlp params passed by the downloader."""
            self.params = params
            captured_params.append(params)

        def __enter__(self) -> Self:
            """Returns the stub instance."""
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            """Matches yt-dlp's context-manager shape."""

        def extract_info(self, url: str, download: bool) -> dict[str, Any] | None:
            """Records the call and returns the canned info dict."""
            captured_calls.append({"url": url, "download": download})
            return info

        def prepare_filename(self, info: dict[str, str]) -> str:
            """Returns the filename yt-dlp would prepare for the result."""
            return (tmp_path / f"{info['id']}.{info['ext']}").as_posix()

    monkeypatch.setattr("discordbot.services.platforms.ytdlp.YoutubeDL", _YoutubeDLStub)
    return captured_params, captured_calls


@pytest.mark.parametrize(
    argnames=("url", "expected_url"),
    argvalues=[
        (
            "https://x.com/reissuerecords/status/1917171960255058421",
            "https://x.com/reissuerecords/status/1917171960255058421",
        ),
        (
            "https://www.facebook.com/watch?v=828357636228730",
            "https://www.facebook.com/reel/828357636228730",
        ),
    ],
)
def test_download_uses_ytdlp_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str, expected_url: str
) -> None:
    """Verifies the download setup without depending on live site APIs."""
    captured_params, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=_DOWNLOADED_INFO
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    result = downloader.download(url=url, quality="best")

    assert result.filename == tmp_path / "video_id.mp4"
    assert captured_calls == [{"url": expected_url, "download": True}]
    assert captured_params[0]["format"] == downloader.quality_formats["best"]


def test_download_resolves_facebook_share_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Facebook share URLs are resolved before the yt-dlp call."""
    _captured_params, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=_DOWNLOADED_INFO
    )

    def fake_resolve(self: VideoDownloader, url: str) -> str:
        """Returns a stable resolved watch URL for the share link."""
        assert isinstance(self, VideoDownloader)
        assert url == "https://www.facebook.com/share/r/17h4SsC2p1"
        return "https://www.facebook.com/watch?v=828357636228730"

    monkeypatch.setattr(
        target=VideoDownloader, name="_resolve_facebook_share_url", value=fake_resolve
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    downloader.download(url="https://www.facebook.com/share/r/17h4SsC2p1", quality="best")

    assert captured_calls == [
        {"url": "https://www.facebook.com/reel/828357636228730", "download": True}
    ]


@pytest.mark.parametrize(
    argnames="url",
    argvalues=[
        "https://notfacebook.com/watch?v=123",
        "https://facebook.com.evil.example/watch?v=9",
        "https://notfacebook.com/share/r/17h4SsC2p1",
    ],
)
def test_a_lookalike_facebook_host_reaches_ytdlp_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str
) -> None:
    """A host that merely contains `facebook.com` is some other site, never a Facebook reel."""
    _captured_params, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=_DOWNLOADED_INFO
    )
    resolved: list[str] = []

    def record_resolve(self: VideoDownloader, url: str) -> str:
        """Records a share-link resolution, which only Facebook's own host may reach."""
        del self
        resolved.append(url)
        return url

    monkeypatch.setattr(
        target=VideoDownloader, name="_resolve_facebook_share_url", value=record_resolve
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    downloader.download(url=url, quality="best")

    assert captured_calls == [{"url": url, "download": True}]
    assert resolved == []


def test_facebook_share_resolution_never_downloads_the_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only where the request landed is wanted, so the body is left on the wire.

    Every other test stubs the resolver out, so without this one a request that downloads a
    full Facebook page just to learn where it landed stays green forever. The HEAD is failed
    on purpose: it has no body whatever it is sent, so the GET fallback below it is the only
    attempt that could ever pull a page down, and therefore the only one worth pinning.
    """
    requests_made: list[dict[str, object]] = []

    class _Response:
        """A share link that answered from the post it points at."""

        url = "https://www.facebook.com/watch?v=828357636228730"

        def close(self) -> None:
            """Releases the connection without the body ever being read."""

    class _SessionStub:
        """Records how each attempt was made."""

        def __enter__(self) -> Self:
            """Returns the stub session."""
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            """Matches requests.Session's context-manager shape."""

        def head(self, url: str, **kwargs: object) -> _Response:
            """Records a HEAD attempt, then refuses it so the GET fallback runs."""
            requests_made.append({"method": "head", "url": url, **kwargs})
            raise RequestException("share links often refuse HEAD")

        def get(self, url: str, **kwargs: object) -> _Response:
            """Records a GET attempt."""
            requests_made.append({"method": "get", "url": url, **kwargs})
            return _Response()

    monkeypatch.setattr(target=downloader_module, name="Session", value=_SessionStub)
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    resolved = downloader._resolve_facebook_share_url(
        "https://www.facebook.com/share/r/17h4SsC2p1"
    )

    assert resolved == "https://www.facebook.com/watch?v=828357636228730"
    assert [request["method"] for request in requests_made] == ["head", "get"]
    assert all(request["stream"] for request in requests_made)
    assert all(request["allow_redirects"] for request in requests_made)


def test_parse_metadata_reads_info_without_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The metadata probe maps yt-dlp's info dict and never asks for a download."""
    captured_params, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        info={
            "id": "BV1jpK86hEc8",
            "title": "a title",
            "uploader": "an uploader",
            "description": "a description",
            "duration": 63,
            "webpage_url": "https://www.bilibili.com/video/BV1jpK86hEc8",
            "is_live": False,
        },
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    metadata = downloader.parse_metadata(url="https://www.bilibili.com/video/BV1jpK86hEc8")

    assert metadata.title == "a title"
    assert metadata.uploader == "an uploader"
    assert metadata.description == "a description"
    assert metadata.duration_seconds == 63.0
    assert metadata.webpage_url == "https://www.bilibili.com/video/BV1jpK86hEc8"
    assert metadata.is_live is False
    assert metadata.from_playlist is False
    assert captured_calls == [
        {"url": "https://www.bilibili.com/video/BV1jpK86hEc8", "download": False}
    ]
    # Silent probe params: simulate with `quiet` on and no info-dict dump, and flat playlists
    # so a channel/space page never costs one request per entry.
    assert captured_params[0]["simulate"] is True
    assert captured_params[0]["skip_download"] is True
    assert captured_params[0]["quiet"] is True
    assert captured_params[0]["extract_flat"] == "in_playlist"
    assert "dump_json" not in captured_params[0]


def test_parse_metadata_defaults_absent_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fields a site does not report fall back to typed defaults instead of raising."""
    _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info={"id": "BV1", "duration": None}
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    metadata = downloader.parse_metadata(url="https://www.bilibili.com/video/BV1")

    assert metadata.title == ""
    assert metadata.uploader == ""
    assert metadata.description == ""
    assert metadata.duration_seconds == 0.0
    assert metadata.webpage_url == ""
    assert metadata.is_live is False


def test_parse_metadata_unwraps_playlist_shaped_info(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A multi-part page reporting itself playlist-shaped yields its first real entry."""
    _install_youtube_dl_stub(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        info={
            "id": "anthology",
            "entries": [None, {"id": "BV1", "title": "part one", "duration": 10}],
        },
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    metadata = downloader.parse_metadata(url="https://www.bilibili.com/video/BV1?p=1")

    assert metadata.title == "part one"
    assert metadata.duration_seconds == 10.0


def test_parse_metadata_keeps_the_playlist_page_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A playlist-shaped page keeps its own URL, so a caller can tell it from a video.

    A b23.tv short link can resolve to a user space or collection, which yt-dlp reads
    SUCCESSFULLY as a playlist; if the first entry's URL won, the caller could no longer
    detect that the page the user linked was never a single video.
    """
    _install_youtube_dl_stub(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        info={
            "id": "672328094",
            "webpage_url": "https://space.bilibili.com/672328094",
            "entries": [
                {
                    "id": "BV1",
                    "title": "newest upload",
                    "webpage_url": "https://www.bilibili.com/video/BV1",
                }
            ],
        },
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    metadata = downloader.parse_metadata(url="https://b23.tv/abc123X")

    assert metadata.title == "newest upload"
    assert metadata.webpage_url == "https://space.bilibili.com/672328094"
    assert metadata.from_playlist is True


def test_download_stop_signal_aborts_at_the_next_progress_tick(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The caller's stop signal turns into a raising progress hook inside yt-dlp.

    The download blocks its worker thread, so asyncio cancellation cannot reach it; the
    hook is the one place yt-dlp lets the caller abort mid-download.
    """
    captured_params, _ = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=_DOWNLOADED_INFO
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())
    stop_signal = threading.Event()

    downloader.download(url="https://example.com/v", quality="best", stop_signal=stop_signal)

    (hook,) = captured_params[0]["progress_hooks"]
    hook({})  # not signaled yet: the download proceeds
    stop_signal.set()
    with pytest.raises(DownloadStoppedError):
        hook({})

    # Without a signal no hook is installed, so the plain path stays untouched.
    downloader.download(url="https://example.com/v", quality="best")
    assert "progress_hooks" not in captured_params[1]


def test_parse_metadata_raises_when_ytdlp_returns_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A None info dict is a failed probe, not an empty video."""
    _install_youtube_dl_stub(monkeypatch=monkeypatch, tmp_path=tmp_path, info=None)
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(RuntimeError, match="no metadata"):
        downloader.parse_metadata(url="https://www.bilibili.com/video/BV1")


def test_get_params_bilibili_referer_handles_scheme_less_hosts(tmp_path: Path) -> None:
    """Bilibili URLs (with or without a scheme) get the Referer; lookalike hosts do not."""
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    def referer(url: str) -> object:
        params = downloader.get_params(quality="best", url=url)
        headers = params["http_headers"]
        assert isinstance(headers, dict)
        return headers.get("Referer")

    assert referer(url="https://www.bilibili.com/video/BV1") == "https://www.bilibili.com"
    assert referer(url="www.bilibili.com/video/BV1") == "https://www.bilibili.com"  # scheme-less
    assert referer(url="evil.com/?x=bilibili.com") is None  # substring lookalike
    assert referer(url="bilibili.com.attacker.com/x") is None  # suffix lookalike


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
    assert interaction.edits[-1]["content"] == (
        "-# 檔案大小: 0.3MB\n-# 來源: <https://source.test/video>"
    )
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
    assert fail_interaction.edits[-1]["content"] == (
        "-# 下載失敗\n檔案大小 3.0MB，超過上傳上限 2MB"
    )

    cog, _ = _install(monkeypatch=monkeypatch, outcome=RuntimeError("download failed"))
    warnings = capture_logs(monkeypatch=monkeypatch, level="warn")
    error_interaction = FakeInteraction()
    await VideoCogs.download_video.callback(
        cog, error_interaction, url="https://x.test", quality="best"
    )
    assert "檔案無法下載" in error_interaction.edits[-1]["content"]
    assert [message for message, _ in warnings] == ["Video download failed"]


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

    assert interaction.edits[-1]["content"] == "-# 檔案無法下載"
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
            """Records which way the worker ended."""
            self.aborted = False
            self.finished = False

        def download(
            self, url: str, quality: str, stop_signal: threading.Event | None = None
        ) -> DownloadResult:
            """Outlasts the command's bound by two orders of magnitude unless told to stop."""
            del url, quality
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
    assert "來源" in interaction.edits[-1]["content"]


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
    assert "已省略 45 張圖片" in content
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
    assert "檔案無法下載" not in content


@pytest.mark.parametrize(
    argnames=("error", "expected"),
    argvalues=[
        (DouyinUnavailableError("SYSTEM_ITEM_NOT_EXIST"), "-# 這則貼文已被刪除或設為私人"),
        (DouyinBlockedError("challenge"), "-# 抖音暫時擋住了請求，請稍後再試"),
        (DouyinTransferError("read timed out"), "-# 這次沒有抓到,稍後再試一次"),
        (TimeoutError(), "-# 抖音回應太慢,這次沒有抓到;稍後再試一次"),
        (DouyinError("boom"), "-# 檔案無法下載"),
        (OSError(28, "No space left on device"), "-# 檔案無法下載"),
    ],
    ids=["gone", "blocked", "transfer", "stalled", "douyin", "other"],
)
def test_each_douyin_failure_gets_its_own_wording(error: Exception, expected: str) -> None:
    """Only a filtered post may read as deleted, and only a bot wall as Douyin refusing.

    Exact strings, because the retryable three all end in a "try again later" that a substring
    check cannot tell apart.
    """
    assert douyin_failure_message(error=error) == expected


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
    assert content.splitlines()[0] == "-# 檔案大小: 0.3MB (過大，改用連結)"
    assert "檔案無法下載" not in content
    # The file really was hosted, so discarding the URL would have lost a completed upload.
    assert list(serve_dir.glob("*.mp4"))
