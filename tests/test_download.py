"""Tests for the yt-dlp downloader facade."""

from types import TracebackType
from typing import Any, Self, get_args
from pathlib import Path
import threading

import pytest

from discordbot.cogs.video import QUALITY_CHOICES, VideoCogs
from discordbot.utils.urls import extract_first_url
from discordbot.utils.douyin import DOUYIN_URL_RE, DouyinDownloader
from discordbot.typings.video import VideoQuality
from discordbot.utils.downloader import VideoDownloader, DownloadStoppedError

from tests.helpers.casting import as_bot


def _install_youtube_dl_stub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Installs a yt-dlp stub and returns captured params and calls."""
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

        def extract_info(self, url: str, download: bool) -> dict[str, str]:
            """Records the final URL and returns minimal media metadata."""
            captured_calls.append({"url": url, "download": download})
            return {"id": "video_id", "ext": "mp4", "title": "stub video"}

        def prepare_filename(self, info: dict[str, str]) -> str:
            """Returns the filename yt-dlp would prepare for the result."""
            return (tmp_path / f"{info['id']}.{info['ext']}").as_posix()

    monkeypatch.setattr("discordbot.utils.downloader.YoutubeDL", _YoutubeDLStub)
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
def test_download_dry_run_uses_ytdlp_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str, expected_url: str
) -> None:
    """Verifies dry-run download setup without depending on live site APIs."""
    captured_params, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with downloader.download(url=url, quality="best", dry_run=True) as result:
        assert result.title == "stub video"
        assert result.filename == tmp_path / "video_id.mp4"

    assert captured_calls == [{"url": expected_url, "download": True}]
    assert captured_params[0]["simulate"] is True
    assert captured_params[0]["skip_download"] is True
    assert captured_params[0]["format"] == downloader.quality_formats["best"]


def test_download_resolves_facebook_share_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Facebook share URLs are resolved before the yt-dlp call."""
    _captured_params, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path
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

    with downloader.download(
        url="https://www.facebook.com/share/r/17h4SsC2p1", quality="best", dry_run=True
    ) as result:
        assert result.title == "stub video"

    assert captured_calls == [
        {"url": "https://www.facebook.com/reel/828357636228730", "download": True}
    ]


def _install_metadata_stub(
    monkeypatch: pytest.MonkeyPatch, info: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Installs a yt-dlp stub whose extract_info returns a canned metadata dict."""
    captured_params: list[dict[str, Any]] = []
    captured_calls: list[dict[str, Any]] = []

    class _YoutubeDLStub:
        """Small context-manager stub for yt-dlp metadata probes."""

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

    monkeypatch.setattr("discordbot.utils.downloader.YoutubeDL", _YoutubeDLStub)
    return captured_params, captured_calls


def test_parse_metadata_reads_info_without_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The metadata probe maps yt-dlp's info dict and never asks for a download."""
    captured_params, captured_calls = _install_metadata_stub(
        monkeypatch=monkeypatch,
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

    assert metadata.video_id == "BV1jpK86hEc8"
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
    # Silent probe params: simulate without the dry_run branch's stdout-dumping shape, and
    # flat playlists so a channel/space page never costs one request per entry.
    assert captured_params[0]["simulate"] is True
    assert captured_params[0]["skip_download"] is True
    assert captured_params[0]["quiet"] is True
    assert captured_params[0]["extract_flat"] == "in_playlist"
    assert "dump_json" not in captured_params[0]


def test_parse_metadata_defaults_absent_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fields a site does not report fall back to typed defaults instead of raising."""
    _install_metadata_stub(monkeypatch=monkeypatch, info={"id": "BV1", "duration": None})
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    metadata = downloader.parse_metadata(url="https://www.bilibili.com/video/BV1")

    assert metadata.video_id == "BV1"
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
    _install_metadata_stub(
        monkeypatch=monkeypatch,
        info={
            "id": "anthology",
            "entries": [None, {"id": "BV1", "title": "part one", "duration": 10}],
        },
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    metadata = downloader.parse_metadata(url="https://www.bilibili.com/video/BV1?p=1")

    assert metadata.video_id == "BV1"
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
    _install_metadata_stub(
        monkeypatch=monkeypatch,
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

    assert metadata.video_id == "BV1"
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
    captured_params, _ = _install_youtube_dl_stub(monkeypatch=monkeypatch, tmp_path=tmp_path)
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())
    stop_signal = threading.Event()

    with downloader.download(
        url="https://example.com/v", quality="best", dry_run=True, stop_signal=stop_signal
    ):
        pass

    (hook,) = captured_params[0]["progress_hooks"]
    hook({})  # not signaled yet: the download proceeds
    stop_signal.set()
    with pytest.raises(DownloadStoppedError):
        hook({})

    # Without a signal no hook is installed, so the plain path stays untouched.
    with downloader.download(url="https://example.com/v", quality="best", dry_run=True):
        pass
    assert "progress_hooks" not in captured_params[1]


def test_parse_metadata_raises_when_ytdlp_returns_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A None info dict is a failed probe, not an empty video."""
    _install_metadata_stub(monkeypatch=monkeypatch, info=None)
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(RuntimeError, match="no metadata"):
        downloader.parse_metadata(url="https://www.bilibili.com/video/BV1")


def test_get_params_bilibili_referer_handles_scheme_less_hosts(tmp_path: Path) -> None:
    """Bilibili URLs (with or without a scheme) get the Referer; lookalike hosts do not."""
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    def referer(url: str) -> object:
        params = downloader.get_params(quality="best", dry_run=False, url=url)
        headers = params["http_headers"]
        assert isinstance(headers, dict)
        return headers.get("Referer")

    assert referer(url="https://www.bilibili.com/video/BV1") == "https://www.bilibili.com"
    assert referer(url="www.bilibili.com/video/BV1") == "https://www.bilibili.com"  # scheme-less
    assert referer(url="evil.com/?x=bilibili.com") is None  # substring lookalike
    assert referer(url="bilibili.com.attacker.com/x") is None  # suffix lookalike


def test_download_video_extracts_a_url_from_share_text() -> None:
    """A share blob pasted into the command still finds its link.

    Share buttons wrap the URL in copy — Douyin's runs straight into Chinese with no space —
    so a command that only accepted a bare URL would fail on the most natural thing to paste.
    """
    blob = (
        "8.46 Y@m.QX :9pm UYm:/ 06/01 短片《临时司机》#AI短片# 内容过于真实 "
        "https://v.douyin.com/tLgj3lCAnds 复制此链接，打开Dou音搜索，直接观看视频"
    )
    assert extract_first_url(text=blob, patterns=(DOUYIN_URL_RE,)) == (
        "https://v.douyin.com/tLgj3lCAnds"
    )


def test_download_video_leaves_a_bare_url_untouched() -> None:
    """The common case must be unchanged: a bare URL passes through as-is."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert extract_first_url(text=url, patterns=(DOUYIN_URL_RE,)) == url


def test_download_video_drops_sentence_punctuation_after_a_link() -> None:
    """A link written mid-sentence must not carry the full stop into the request."""
    assert extract_first_url(text="see https://example.com/a/b.", patterns=()) == (
        "https://example.com/a/b"
    )


def test_download_video_passes_unparseable_input_through() -> None:
    """Text with no URL is handed on unchanged, so it fails downstream as it always did."""
    assert extract_first_url(text="  not a url  ", patterns=()) == "not a url"


def test_every_quality_preset_is_answered_everywhere() -> None:
    """A preset added to the type has to be answered by every site that maps one.

    The option's own default is read off the registered command rather than spelled out here:
    nextcord types `SlashOption(default=...)` as `Any`, so it is the one preset site mypy
    cannot see, and it is the value every `/download_video` without an explicit quality carries.
    """
    presets = set(get_args(VideoQuality))

    assert set(VideoDownloader.quality_formats) == presets
    assert set(DouyinDownloader.quality_ratios) == presets
    assert set(QUALITY_CHOICES.values()) == presets

    cog = VideoCogs(bot=as_bot(fake=object()))
    assert cog.download_video.options["quality"].default in presets
