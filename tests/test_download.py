"""Tests for the yt-dlp downloader facade."""

from types import TracebackType
from typing import Any, Self, get_args
from pathlib import Path
import threading

import pytest
from requests.exceptions import RequestException

from discordbot.utils import downloader as downloader_module
from discordbot.utils.urls import normalized_host, extract_first_url, host_matches_domain
from discordbot.utils.douyin import DOUYIN_URL_RE, DouyinDownloader
from discordbot.typings.video import VideoQuality
from discordbot.utils.threads import THREADS_URL_RE
from discordbot.utils.youtube import YOUTUBE_URL_RE
from discordbot.cogs.video.cog import QUALITY_CHOICES, VideoCogs
from discordbot.utils.bilibili import BILIBILI_URL_RE
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


def test_host_matches_domain_refuses_a_lookalike_host() -> None:
    """The exact-label rule every site checks its host through, tested where it now lives."""
    assert host_matches_domain(host="douyin.com", domain="douyin.com")
    assert host_matches_domain(host="v.douyin.com", domain="douyin.com")
    assert not host_matches_domain(host="douyin.com.attacker.com", domain="douyin.com")
    assert not host_matches_domain(host="notdouyin.com", domain="douyin.com")
    assert not host_matches_domain(host="", domain="douyin.com")


def test_normalized_host_reads_a_scheme_less_paste_and_never_raises() -> None:
    """A pasted host with no scheme still parses, and malformed input answers rather than raising.

    This runs in routing checks that sit ahead of any error handling, so a `ValueError` out of
    urlparse would escape into a command handler that has nothing to say about it.
    """
    assert normalized_host(url="https://V.Douyin.com/abc") == "v.douyin.com"
    assert normalized_host(url="v.douyin.com/abc") == "v.douyin.com"
    assert normalized_host(url="https://[abc/x") == ""


def test_every_url_pattern_shares_the_generic_start_anchor() -> None:
    """A link glued to the end of an ASCII word is not a link to ANY of the scanners.

    The site patterns used to carry no start anchor at all, so `xhttps://v.douyin.com/abc` was
    refused by the generic scanner and matched by every site one. CJK in front is not an ASCII
    word character, so those still match (#492).
    """
    for pattern, url in (
        (DOUYIN_URL_RE, "https://v.douyin.com/tLgj3lCAnds"),
        (THREADS_URL_RE, "https://www.threads.com/@user/post/ABC123"),
        (BILIBILI_URL_RE, "https://www.bilibili.com/video/BV1jpK86hEc8"),
        (YOUTUBE_URL_RE, "https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
    ):
        assert pattern.search(string=url) is not None
        assert pattern.search(string=f"x{url}") is None
        assert pattern.search(string=f"看這個{url}") is not None


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


def test_download_video_finds_a_link_typed_flush_against_chinese() -> None:
    r"""A link with no space in front of it is still a link, but only past a non-ASCII word.

    The generic pattern used to head on `\b`, which counts CJK as a word character, so
    `這個https://...` found nothing at all (#492). A link glued to the end of an ASCII word is
    still refused, and falls through to the unchanged-passthrough behaviour below.
    """
    assert extract_first_url(text="幫我下載這個https://example.com/a/b", patterns=()) == (
        "https://example.com/a/b"
    )
    assert extract_first_url(text="xhttps://example.com/a/b", patterns=()) == (
        "xhttps://example.com/a/b"
    )


def test_download_video_passes_unparseable_input_through() -> None:
    """Text with no URL is handed on unchanged, so it fails downstream as it always did."""
    assert extract_first_url(text="  not a url  ", patterns=()) == "not a url"


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
