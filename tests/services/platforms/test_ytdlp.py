"""Tests for the yt-dlp downloader facade."""

from types import TracebackType
from typing import Any, Self
from pathlib import Path
import threading
from collections.abc import Callable

import pytest
from requests.exceptions import RequestException

from discordbot.services.platforms import ytdlp as downloader_module
from discordbot.utils.file_downloads import DownloadTooLargeError
from discordbot.services.platforms.ytdlp import (
    VideoDownloader,
    DownloadStoppedError,
    download_with_stop_signal,
)
from discordbot.services.platforms.facebook import FacebookOutput, FacebookConversation
from discordbot.services.platforms.page_json import BROWSER_HEADERS

from tests.helpers.logfire_capture import capture_logs

# What `extract_info` answers for a finished download: the least `download` reads a result from.
_DOWNLOADED_INFO = {"id": "video_id", "ext": "mp4"}


def _materialize_stub_files(
    tmp_path: Path, info: dict[str, Any] | None, file_bytes: bytes
) -> None:
    """Writes fake media files on disk for every entry the stub claims was downloaded."""
    if info is None:
        return
    entries = info.get("entries")
    items = (
        [entry for entry in entries if isinstance(entry, dict)] if entries is not None else [info]
    )
    for entry in items:
        requested = entry.get("requested_downloads")
        if isinstance(requested, list) and requested:
            for item in requested:
                if isinstance(item, dict) and isinstance(item.get("filepath"), str):
                    Path(item["filepath"]).write_bytes(data=file_bytes)
        elif "id" in entry and "ext" in entry:
            (tmp_path / f"{entry['id']}.{entry['ext']}").write_bytes(data=file_bytes)


class _YoutubeDLStub:
    """Small context-manager stub for yt-dlp."""

    def __init__(  # noqa: PLR0913 -- test-double wiring across params, calls, and fixtures
        self,
        params: dict[str, Any],
        tmp_path: Path,
        info: dict[str, Any] | None,
        file_bytes: bytes | None,
        on_download: Callable[[dict[str, Any]], None] | None,
        captured_params: list[dict[str, Any]],
        captured_calls: list[dict[str, Any]],
    ) -> None:
        """Records the yt-dlp params passed by the downloader."""
        self.params = params
        self._tmp_path = tmp_path
        self._info = info
        self._file_bytes = file_bytes
        self._on_download = on_download
        self._captured_calls = captured_calls
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

    def _finish_download(self, target_info: dict[str, Any] | None) -> None:
        """Runs optional download callback and materializes fake output files."""
        if self._on_download is not None:
            self._on_download(self.params)
        if self._file_bytes is not None:
            _materialize_stub_files(
                tmp_path=self._tmp_path, info=target_info, file_bytes=self._file_bytes
            )

    def extract_info(self, url: str, download: bool) -> dict[str, Any] | None:
        """Records the call and returns the canned info dict."""
        self._captured_calls.append({"url": url, "download": download})
        if download:
            self._finish_download(target_info=self._info)
        return self._info

    def process_ie_result(
        self, ie_result: dict[str, Any], download: bool = True
    ) -> dict[str, Any]:
        """Records the download step and materializes canned output files."""
        self._captured_calls.append({"process_ie_result": True, "download": download})
        if download:
            self._finish_download(target_info=ie_result)
        return ie_result

    def prepare_filename(self, info: dict[str, Any]) -> str:
        """Returns the filename yt-dlp would prepare for the result."""
        return (self._tmp_path / f"{info.get('id', 'unknown')}.{info.get('ext', 'NA')}").as_posix()


def _install_youtube_dl_stub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    info: dict[str, Any] | None,
    file_bytes: bytes | None = b"mp4",
    on_download: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Installs a yt-dlp stub answering `extract_info` with `info`; returns params and calls."""
    captured_params: list[dict[str, Any]] = []
    captured_calls: list[dict[str, Any]] = []

    def _build(params: dict[str, Any]) -> _YoutubeDLStub:
        return _YoutubeDLStub(
            params=params,
            tmp_path=tmp_path,
            info=info,
            file_bytes=file_bytes,
            on_download=on_download,
            captured_params=captured_params,
            captured_calls=captured_calls,
        )

    monkeypatch.setattr("discordbot.services.platforms.ytdlp.YoutubeDL", _build)
    return captured_params, captured_calls


_SHARE_URL = "https://www.facebook.com/share/r/17h4SsC2p1"
_SHARE_TARGET = "https://www.facebook.com/watch?v=828357636228730"


class _LandedResponse:
    """A share link or HEAD probe response."""

    def __init__(
        self, url: str = "", headers: dict[str, str] | None = None, ok: bool = True
    ) -> None:
        """Records the landed URL, headers, and status."""
        self.url = url
        self.headers = headers or {}
        self.ok = ok

    def close(self) -> None:
        """Releases the connection without the body ever being read."""


def _install_share_session(
    monkeypatch: pytest.MonkeyPatch,
    refuse_head: bool = False,
    target_url: str = _SHARE_TARGET,
    head_lengths: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Answers share-link redirects and HEAD size probes; returns every request made."""
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
            if head_lengths is not None:
                length = head_lengths.get(url)
                headers = {"Content-Length": length} if length is not None else {}
                return _LandedResponse(url=url, headers=headers)
            return _LandedResponse(url=target_url)

        def get(self, url: str, **kwargs: object) -> _LandedResponse:
            """Records a GET attempt."""
            requests_made.append({"method": "get", "url": url, **kwargs})
            return _LandedResponse(url=target_url)

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

    assert result.filenames == [tmp_path / "video_id.mp4"]
    assert captured_calls == [{"url": expected_url, "download": True}]
    assert captured_params[0]["format"] == downloader.quality_formats["best"]
    assert captured_params[0]["playlist_items"] == "1-1"


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


def test_facebook_share_resolution_sends_browser_headers_and_ignores_login_walls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Share resolution sends `BROWSER_HEADERS` and falls through or reverts on login redirects."""
    requests_made: list[dict[str, object]] = []
    responses = iter([
        _LandedResponse(url="https://www.facebook.com/login/?next=foo"),
        _LandedResponse(url=_SHARE_TARGET),
    ])

    class _LoginWallSession:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def head(self, url: str, **kwargs: object) -> _LandedResponse:
            requests_made.append({"method": "head", "url": url, **kwargs})
            return next(responses)

        def get(self, url: str, **kwargs: object) -> _LandedResponse:
            requests_made.append({"method": "get", "url": url, **kwargs})
            return next(responses)

    monkeypatch.setattr("discordbot.services.platforms.ytdlp.Session", _LoginWallSession)
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    resolved = downloader._resolve_facebook_share_url(url=_SHARE_URL)

    assert resolved == _SHARE_TARGET
    assert [request["method"] for request in requests_made] == ["head", "get"]
    assert all(request["headers"] == BROWSER_HEADERS for request in requests_made)

    # When both HEAD and GET land on login/checkpoint/recover, return the original URL unchanged.
    responses = iter([
        _LandedResponse(url="https://www.facebook.com/checkpoint/123"),
        _LandedResponse(url="https://www.facebook.com/recover/initiate"),
    ])
    assert downloader._resolve_facebook_share_url(url=_SHARE_URL) == _SHARE_URL


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


def test_download_handles_multi_video_playlist_and_requested_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Multi-video playlists return every downloaded entry file instead of a nonexistent `*.NA`."""
    merged_path = tmp_path / "entry_1.mp4"
    playlist_info = {
        "_type": "playlist",
        "id": "1577719286659006464",
        "playlist_count": 4,
        "entries": [
            None,
            {
                "id": "entry_1",
                "ext": "webm",
                "requested_downloads": [{"filepath": merged_path.as_posix()}],
            },
            {"id": "entry_2", "ext": "mp4"},
        ],
    }
    captured_params, _ = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=playlist_info
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    result = downloader.download(url="https://x.com/user/status/1577719286659006464", max_items=10)

    assert captured_params[0]["playlist_items"] == "1-10"
    assert result.filenames == [merged_path, tmp_path / "entry_2.mp4"]
    assert result.total_items == 4
    assert result.omitted_items == 2


def test_download_raises_when_expected_files_do_not_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A nonexistent prepared path (such as `*.NA`) fails explicitly instead of leaking out."""
    _install_youtube_dl_stub(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        info={"id": "ghost", "ext": "NA"},
        file_bytes=None,
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(RuntimeError, match="no downloaded files"):
        downloader.download(url="https://x.com/user/status/1")


def test_download_rejects_non_positive_max_items(tmp_path: Path) -> None:
    """`max_items` below 1 is invalid and raises before invoking yt-dlp."""
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(ValueError, match="max_items"):
        downloader.download(url="https://x.com/user/status/1", max_items=0)


def test_every_quality_preset_keeps_heightless_fallback() -> None:
    """Every quality preset ends with `/best/bestvideo*` so height-less MP4s (Twitter GIFs) succeed."""
    for preset, fmt in VideoDownloader.quality_formats.items():
        assert fmt.endswith("/best/bestvideo*"), preset


def test_convert_facebook_post_url_resolves_video_permalink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Facebook `/posts/pfbid...` URLs resolve to their `/videos/<id>/` permalink before yt-dlp."""
    post_url = "https://www.facebook.com/attn/posts/pfbid0j1Czf2gGDVqeQ8KiMLFm3pWN8GxsQmeRrVhimWDzMuKQoR8r4b1knNsejELmUgyhl"
    video_permalink = "https://www.facebook.com/attn/videos/6968553779868435/"
    seen_urls: list[str] = []

    def fake_parse_metadata(self: object, url: str) -> FacebookConversation:
        del self
        seen_urls.append(url)
        return FacebookConversation(
            chain=[FacebookOutput(url=post_url, video_urls=[video_permalink])]
        )

    monkeypatch.setattr(
        target=downloader_module.FacebookDownloader,
        name="parse_metadata",
        value=fake_parse_metadata,
    )
    _, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=_DOWNLOADED_INFO
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    downloader.download(url=post_url)

    assert seen_urls == [post_url]
    assert captured_calls == [{"url": video_permalink, "download": True}]


def test_convert_facebook_share_to_post_resolves_video_permalink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `/share/p/...` link redirecting to a `/posts/pfbid...` URL still resolves its video."""
    share_url = "https://www.facebook.com/share/p/1abcXYZ/"
    post_url = "https://www.facebook.com/attn/posts/pfbid0j1abc"
    video_permalink = "https://www.facebook.com/attn/videos/6968553779868435/"

    _install_share_session(monkeypatch=monkeypatch, target_url=post_url)
    monkeypatch.setattr(
        target=downloader_module.FacebookDownloader,
        name="parse_metadata",
        value=lambda _self, url: FacebookConversation(
            chain=[FacebookOutput(url=url, video_urls=[video_permalink])]
        ),
    )
    _, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=_DOWNLOADED_INFO
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    downloader.download(url=share_url)

    assert captured_calls == [{"url": video_permalink, "download": True}]


def test_convert_facebook_direct_video_url_skips_post_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Direct `/<page>/videos/<id>/` URLs reach yt-dlp directly without hitting `FacebookDownloader`."""
    direct_video = "https://www.facebook.com/attn/videos/6968553779868435/"

    def fail_parse(self: object, url: str) -> FacebookConversation:
        del self, url
        raise AssertionError("FacebookDownloader must not be called for direct /videos/ URLs")

    monkeypatch.setattr(
        target=downloader_module.FacebookDownloader, name="parse_metadata", value=fail_parse
    )
    _, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=_DOWNLOADED_INFO
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    downloader.download(url=direct_video)

    assert captured_calls == [{"url": direct_video, "download": True}]


@pytest.mark.parametrize(
    argnames="behavior", argvalues=["no_video", "raises"], ids=["no-video", "parser-raises"]
)
def test_convert_facebook_post_falls_back_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, behavior: str
) -> None:
    """A Facebook post with no video or a failing parser falls back to the original URL."""
    post_url = "https://www.facebook.com/attn/posts/pfbid0j1abc"

    def fake_parse(self: object, url: str) -> FacebookConversation:
        del self
        if behavior == "raises":
            raise RuntimeError("page unavailable")
        return FacebookConversation(chain=[FacebookOutput(url=url, video_urls=[])])

    monkeypatch.setattr(
        target=downloader_module.FacebookDownloader, name="parse_metadata", value=fake_parse
    )
    warnings = capture_logs(monkeypatch=monkeypatch, level="warn")
    _, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=_DOWNLOADED_INFO
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    downloader.download(url=post_url)

    assert captured_calls == [{"url": post_url, "download": True}]
    if behavior == "raises":
        assert [message for message, _ in warnings] == [
            "Facebook post video resolution failed; falling back to original URL"
        ]
    else:
        assert warnings == []


def test_max_bytes_rejects_oversize_from_filesize_before_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pre-download `filesize` / `filesize_approx` rejects oversize media with zero download calls."""
    info = {"id": "big", "ext": "mp4", "filesize_approx": 500_000}
    _, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=info
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(DownloadTooLargeError):
        downloader.download(url="https://x.com/user/status/1", max_bytes=100_000)

    assert captured_calls == [{"url": "https://x.com/user/status/1", "download": False}]
    assert list(tmp_path.iterdir()) == []


def test_max_bytes_rejects_oversize_via_head_across_requested_formats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pre-download HEAD sums video + audio `Content-Length` and rejects before downloading."""
    video_url = "https://cdn.test/video.mp4"
    audio_url = "https://cdn.test/audio.m4a"
    info = {
        "id": "fb_vid",
        "ext": "mp4",
        "requested_formats": [
            {"url": video_url, "protocol": "https"},
            {"url": audio_url, "protocol": "https"},
        ],
    }
    requests_made = _install_share_session(
        monkeypatch=monkeypatch, head_lengths={video_url: "80000", audio_url: "40000"}
    )
    _, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=info
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(DownloadTooLargeError):
        downloader.download(url="https://www.facebook.com/attn/videos/1/", max_bytes=100_000)

    assert [req["url"] for req in requests_made] == [video_url, audio_url]
    assert captured_calls == [
        {"url": "https://www.facebook.com/attn/videos/1/", "download": False}
    ]


def test_max_bytes_skips_head_for_hls_and_uses_bitrate_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HLS (`m3u8_native`) formats skip HEAD (whose length is the tiny manifest) and use `duration * tbr`."""
    info = {
        "id": "spacex",
        "ext": "mp4",
        "duration": 100.0,
        "requested_formats": [
            {
                "url": "https://video.twimg.com/playlist.m3u8",
                "protocol": "m3u8_native",
                "tbr": 1400.0,
            },
            {"url": "https://video.twimg.com/audio.m3u8", "protocol": "m3u8_native", "tbr": 128.0},
        ],
    }
    # If HEAD were mistakenly sent, a tiny 500-byte playlist length would pass the 100 KB cap.
    requests_made = _install_share_session(
        monkeypatch=monkeypatch,
        head_lengths={
            "https://video.twimg.com/playlist.m3u8": "500",
            "https://video.twimg.com/audio.m3u8": "500",
        },
    )
    _, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=info
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(DownloadTooLargeError):
        downloader.download(url="https://x.com/SpaceX/status/1", max_bytes=100_000)

    assert requests_made == []
    assert captured_calls == [{"url": "https://x.com/SpaceX/status/1", "download": False}]


def test_max_bytes_falls_back_to_bitrate_when_head_fails_or_omits_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When HEAD refuses or omits `Content-Length`, `duration * tbr * 1000 / 8` still enforces the cap."""
    info = {
        "id": "vid",
        "ext": "mp4",
        "url": "https://cdn.test/v.mp4",
        "protocol": "https",
        "duration": 20.0,
        "tbr": 1000.0,  # ~2.5 MB
    }
    _install_share_session(monkeypatch=monkeypatch, refuse_head=True)
    _, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=info
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(DownloadTooLargeError):
        downloader.download(url="https://example.com/v", max_bytes=100_000)

    assert captured_calls == [{"url": "https://example.com/v", "download": False}]


def test_max_bytes_rejects_playlist_when_combined_entries_exceed_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Playlist pre-download probe sums across entries up to `max_items`."""
    info = {
        "_type": "playlist",
        "id": "pl",
        "entries": [
            {"id": "e1", "ext": "mp4", "filesize": 60_000},
            {"id": "e2", "ext": "mp4", "filesize": 60_000},
        ],
    }
    _, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=info
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(DownloadTooLargeError):
        downloader.download(url="https://x.com/user/status/1", max_items=4, max_bytes=100_000)

    assert captured_calls == [{"url": "https://x.com/user/status/1", "download": False}]


def test_max_bytes_proceeds_when_estimate_and_actual_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When pre-download estimate and final size fit `max_bytes`, `process_ie_result` downloads it."""
    info = {"id": "small", "ext": "mp4", "filesize": 1024}
    _, captured_calls = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=info, file_bytes=b"x" * 1024
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    result = downloader.download(url="https://x.com/user/status/1", max_bytes=2048)

    assert result.filenames == [tmp_path / "small.mp4"]
    assert captured_calls == [
        {"url": "https://x.com/user/status/1", "download": False},
        {"process_ie_result": True, "download": True},
    ]


def test_streaming_byte_cap_tracks_multi_stream_and_fragment_resets_and_cleans_partials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Streaming progress hook counts cumulative bytes across video+audio and cleans partial files."""
    preexisting = tmp_path / "keep.txt"
    preexisting.write_text(data="preserve me", encoding="utf-8")

    def simulate_multi_stream(params: dict[str, Any]) -> None:
        partial_video = tmp_path / "vid.f137.mp4"
        partial_audio = tmp_path / "vid.f140.m4a.part"
        partial_video.write_bytes(data=b"v" * 60_000)
        partial_audio.write_bytes(data=b"a" * 50_000)
        (hook,) = params["progress_hooks"]
        # Stream 1 finishes under cap (60 KB <= 100 KB).
        hook({"status": "downloading", "filename": str(partial_video), "downloaded_bytes": 60_000})
        hook({"status": "finished", "filename": str(partial_video), "downloaded_bytes": 60_000})
        # Stream 2 resets `downloaded_bytes` to 50 KB; cumulative 110 KB trips the 100 KB cap.
        hook({"status": "downloading", "filename": str(partial_audio), "downloaded_bytes": 50_000})

    _install_youtube_dl_stub(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        info={"id": "vid", "ext": "mp4"},
        on_download=simulate_multi_stream,
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(DownloadTooLargeError):
        downloader.download(url="https://example.com/v", max_bytes=100_000)

    assert preexisting.is_file()
    assert list(tmp_path.iterdir()) == [preexisting]


def test_streaming_byte_cap_handles_unnamed_stream_reset_and_total_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unnamed stream whose `downloaded_bytes` resets downwards still accumulates across resets."""
    captured_params, _ = _install_youtube_dl_stub(
        monkeypatch=monkeypatch, tmp_path=tmp_path, info=_DOWNLOADED_INFO, file_bytes=b"ok"
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())
    downloader.download(url="https://example.com/v", max_bytes=100_000)
    (hook,) = captured_params[0]["progress_hooks"]

    # Stream 1 reaches 60 KB, then resets to 50 KB without a `finished` tick -> cumulative 110 KB.
    hook({"status": "downloading", "downloaded_bytes": 60_000})
    with pytest.raises(DownloadTooLargeError):
        hook({"status": "downloading", "downloaded_bytes": 50_000})

    # Fresh hook: a single tick whose `total_bytes` exceeds cap aborts immediately.
    downloader.download(url="https://example.com/v", max_bytes=100_000)
    (hook2,) = captured_params[1]["progress_hooks"]
    with pytest.raises(DownloadTooLargeError):
        hook2({"status": "downloading", "downloaded_bytes": 1024, "total_bytes": 200_000})


def test_max_bytes_enforces_actual_disk_size_and_cleans_up_when_hooks_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even when metadata and hooks report nothing, an oversize finished file is rejected and deleted."""
    _install_youtube_dl_stub(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        info={"id": "sneaky", "ext": "mp4"},
        file_bytes=b"z" * 2000,
    )
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(DownloadTooLargeError):
        downloader.download(url="https://example.com/v", max_bytes=1000)

    assert list(tmp_path.iterdir()) == []


async def test_download_with_stop_signal_forwards_max_items_and_max_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`download_with_stop_signal` forwards `max_items` and `max_bytes` to `VideoDownloader.download`."""
    recorded: list[dict[str, object]] = []
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(data=b"mp4")

    def fake_download(self: VideoDownloader, **kwargs: object) -> downloader_module.DownloadResult:
        del self
        recorded.append({
            "url": kwargs["url"],
            "quality": kwargs["quality"],
            "has_stop": kwargs["stop_signal"] is not None,
            "max_items": kwargs["max_items"],
            "max_bytes": kwargs["max_bytes"],
        })
        return downloader_module.DownloadResult(filenames=[clip])

    monkeypatch.setattr(target=VideoDownloader, name="download", value=fake_download)
    downloader = VideoDownloader(output_folder=tmp_path.as_posix())

    result = await download_with_stop_signal(
        downloader=downloader,
        url="https://x.com/user/status/1",
        quality="medium",
        max_items=4,
        max_bytes=500_000,
    )

    assert result.filenames == [clip]
    assert recorded == [
        {
            "url": "https://x.com/user/status/1",
            "quality": "medium",
            "has_stop": True,
            "max_items": 4,
            "max_bytes": 500_000,
        }
    ]
