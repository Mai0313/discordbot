"""Tests for the yt-dlp downloader facade."""

from types import TracebackType
from typing import Any, Self
from pathlib import Path
import threading

import pytest
from requests.exceptions import RequestException

from discordbot.services.platforms import ytdlp as downloader_module
from discordbot.services.platforms.ytdlp import VideoDownloader, DownloadStoppedError

# What `extract_info` answers for a finished download: the least `download` reads a result from.
_DOWNLOADED_INFO = {"id": "video_id", "ext": "mp4"}


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


_SHARE_URL = "https://www.facebook.com/share/r/17h4SsC2p1"
_SHARE_TARGET = "https://www.facebook.com/watch?v=828357636228730"


class _LandedResponse:
    """A share link's answer from the post it points at."""

    def __init__(self, url: str) -> None:
        """Records where the redirect chain ended."""
        self.url = url

    def close(self) -> None:
        """Releases the connection without the body ever being read."""


def _install_share_session(
    monkeypatch: pytest.MonkeyPatch, refuse_head: bool = False
) -> list[dict[str, object]]:
    """Answers every share-link request as landing on `_SHARE_TARGET`; returns each one made.

    `refuse_head` fails the HEAD attempt the way many share links do, so the GET fallback runs.
    """
    requests_made: list[dict[str, object]] = []

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

        def head(self, url: str, **kwargs: object) -> _LandedResponse:
            """Records a HEAD attempt, refusing it when the test asks."""
            requests_made.append({"method": "head", "url": url, **kwargs})
            if refuse_head:
                raise RequestException("share links often refuse HEAD")
            return _LandedResponse(url=_SHARE_TARGET)

        def get(self, url: str, **kwargs: object) -> _LandedResponse:
            """Records a GET attempt."""
            requests_made.append({"method": "get", "url": url, **kwargs})
            return _LandedResponse(url=_SHARE_TARGET)

    monkeypatch.setattr(target=downloader_module, name="Session", value=_SessionStub)
    return requests_made


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
    requests_made = _install_share_session(monkeypatch=monkeypatch)
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    downloader.download(url=_SHARE_URL, quality="best")

    assert [request["url"] for request in requests_made] == [_SHARE_URL]
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
    requests_made = _install_share_session(monkeypatch=monkeypatch)
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    downloader.download(url=url, quality="best")

    assert captured_calls == [{"url": url, "download": True}]
    assert requests_made == []


def test_facebook_share_resolution_never_downloads_the_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only where the request landed is wanted, so the body is left on the wire.

    The HEAD is failed on purpose: it has no body whatever it is sent, so the GET fallback below
    it is the only attempt that could ever pull a page down, and therefore the only one worth
    pinning.
    """
    requests_made = _install_share_session(monkeypatch=monkeypatch, refuse_head=True)
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    resolved = downloader._resolve_facebook_share_url(_SHARE_URL)

    assert resolved == _SHARE_TARGET
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

    with pytest.raises(RuntimeError):
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
