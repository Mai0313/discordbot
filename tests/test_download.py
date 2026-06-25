"""Tests for the yt-dlp downloader facade."""

from types import TracebackType
from typing import Any, Self
from pathlib import Path

import pytest

from discordbot.utils.downloader import VideoDownloader


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
