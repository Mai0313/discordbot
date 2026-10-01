"""Tests for the Douyin share-page parser and downloader.

Every HTTP call is stubbed. Besides the usual reason (tests must not depend on a live site),
Douyin bans a share path for tens of minutes once it is hit hard, so a test suite that reached
the real endpoint would take the whole deployment down with it.
"""

import json
import shutil
from typing import IO, Any, Self
from pathlib import Path
import tempfile
from collections.abc import Callable, Iterator

import pytest
import requests

from discordbot.typings.video import VideoQuality
from discordbot.utils.link_errors import LinkRetryableError
import discordbot.services.platforms.douyin as douyin_module
from discordbot.services.platforms.douyin import (
    DOUYIN_URL_RE,
    DouyinError,
    DouyinDownloader,
    DouyinBlockedError,
    DouyinTooLargeError,
    DouyinTransferError,
    DouyinUnavailableError,
    is_douyin_url,
    is_douyin_post_url,
)

# These downloaders are only ever asked to parse, never to write, so the folder is inert.
_SCRATCH_DIR = tempfile.gettempdir()

_VIDEO_ID = "7664447317017136422"
_PHOTO_ID = "7159955455492541733"

# Mirrors the live payload: a bare video id in `uri`, and a `playwm` (watermarked) URL in the
# list Douyin ships.
_VIDEO_ITEM: dict[str, Any] = {
    "aweme_id": _VIDEO_ID,
    "aweme_type": 4,
    "desc": "因为一顿饭差点大打出手",
    "author": {"nickname": "真探唐仁杰"},
    "images": None,
    "video": {
        "play_addr": {
            "uri": "v0200fg10000d9ep8svog65tt01vahsg",
            "url_list": [
                "https://aweme.snssdk.com/aweme/v1/playwm/?video_id=v0200fg&ratio=720p&line=0"
            ],
        }
    },
}

# A photo post deliberately keeps a non-empty `video.play_addr` (Douyin renders the gallery into
# a slideshow clip), which is exactly the trap a "does it have play_addr" check falls into.
_PHOTO_ITEM: dict[str, Any] = {
    "aweme_id": _PHOTO_ID,
    "aweme_type": 2,
    "desc": "一組圖",
    "author": {"nickname": "someone"},
    "images": [
        {
            "url_list": [
                f"https://cdn/{n}-a.webp",
                f"https://cdn/{n}-b.webp",
                f"https://cdn/{n}.jpeg",
            ],
            # Despite the name, this is the WATERMARKED variant and must never be picked.
            "download_url_list": [f"https://cdn/{n}-water.jpeg"],
            "width": 1080,
            "height": 1920,
        }
        for n in range(3)
    ],
    "video": {
        "play_addr": {
            "uri": "https://sf.douyinstatic.com/obj/audio-track",
            "url_list": ["https://aweme.snssdk.com/aweme/v1/playwm/?video_id=slideshow"],
        }
    },
}


def _router_html(page_key: str, video_info: dict[str, Any]) -> str:
    """Builds a share page whose loaderData key follows the fetched URL path."""
    payload = {
        "loaderData": {
            f"{page_key}_layout": {"ua": "stub"},
            f"{page_key}_(id)/page": {"videoInfoRes": video_info},
        },
        "errors": {},
    }
    return (
        f"<html><body></body><script>window._ROUTER_DATA = {json.dumps(payload)};</script></html>"
    )


def _ok_page(item: dict[str, Any], page_key: str = "note") -> str:
    """A share page carrying exactly one post."""
    return _router_html(page_key=page_key, video_info={"item_list": [item], "filter_list": []})


class _FakeResponse:
    """Minimal stand-in for a requests Response."""

    def __init__(
        self,
        text: str = "",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        stall_mid_stream: bool = False,
    ) -> None:
        """Stores the canned response payload."""
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self._stall_mid_stream = stall_mid_stream

    def raise_for_status(self) -> None:
        """Mimics requests' status check, carrying the response the way requests does."""
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Yields the canned body, optionally dying part-way through.

        Stalling mid-stream is the failure the CDN actually produces, and it is the only shape
        that leaves a partial file on disk, so it is what the cleanup assertions need.
        """
        if self._stall_mid_stream:
            yield self._body
            raise douyin_module.RequestException("read timed out")
        yield self._body

    def close(self) -> None:
        """Matches the Response API used by the redirect probe."""


def _install_session(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[str, dict[str, object]], _FakeResponse]
) -> list[dict[str, Any]]:
    """Replaces requests.Session with a stub driven by `handler`; returns the captured calls.

    A captured call is the request kwargs bag, so `Any` is what lets an assertion reach into a
    nested value such as `calls[0]["headers"]["User-Agent"]`.
    """
    calls: list[dict[str, Any]] = []

    class _FakeSession:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str, **kwargs: object) -> _FakeResponse:
            calls.append({"url": url, **kwargs})
            return handler(url, kwargs)

    monkeypatch.setattr(douyin_module.requests, "Session", _FakeSession)
    return calls


@pytest.fixture(autouse=True)
def _clear_payload_cache() -> Iterator[None]:
    """Keeps the module-level share-payload and short-link caches from leaking between tests."""
    douyin_module._PAYLOAD_CACHE.clear()
    douyin_module._LINK_ID_CACHE.clear()
    yield
    douyin_module._PAYLOAD_CACHE.clear()
    douyin_module._LINK_ID_CACHE.clear()


@pytest.mark.parametrize(
    argnames=("url", "expected"),
    argvalues=[
        (f"https://www.douyin.com/video/{_VIDEO_ID}", _VIDEO_ID),
        (f"https://www.douyin.com/note/{_PHOTO_ID}", _PHOTO_ID),
        (f"https://www.iesdouyin.com/share/video/{_VIDEO_ID}/?region=TW&mid=1", _VIDEO_ID),
        (f"https://www.iesdouyin.com/share/note/{_PHOTO_ID}", _PHOTO_ID),
        (f"https://m.douyin.com/share/slides/{_PHOTO_ID}", _PHOTO_ID),
        (f"https://www.douyin.com/discover?modal_id={_VIDEO_ID}", _VIDEO_ID),
        # Both shapes present: the profile's sec_uid sits in the path, the post id in the query.
        (f"https://www.douyin.com/user/MS4wLjABAAAAMOcq?modal_id={_VIDEO_ID}", _VIDEO_ID),
    ],
)
def test_extract_id_handles_every_url_form(url: str, expected: str) -> None:
    """Each accepted URL shape yields the post id, with modal_id winning over the path."""
    assert douyin_module._extract_post_id(url=url) == expected


@pytest.mark.parametrize(
    argnames=("url", "expected"),
    argvalues=[
        ("https://www.douyin.com/video/1", True),
        ("https://v.douyin.com/abc", True),
        ("https://www.iesdouyin.com/share/note/1", True),
        ("v.douyin.com/abc", True),  # scheme-less paste
        ("https://douyin.com.attacker.com/x", False),  # suffix lookalike
        ("https://evil.com/?x=douyin.com", False),  # substring lookalike
        ("https://www.ixigua.com/123", False),  # a real short-link redirect target
    ],
)
def test_is_douyin_url_matches_whole_host_labels(url: str, expected: bool) -> None:
    """Host detection never accepts a lookalike domain."""
    assert is_douyin_url(url=url) is expected


def test_url_regex_survives_the_share_blob() -> None:
    """Douyin's share text wraps the link in CJK noise; the match must stop at the punctuation."""
    blob = (
        "7.64 gOX:/ w@f.oD 05/14 世界这本书 https://v.douyin.com/iR2syBRn/ 复制此链接，打开Dou音"
    )
    assert DOUYIN_URL_RE.findall(blob) == ["https://v.douyin.com/iR2syBRn/"]
    assert DOUYIN_URL_RE.findall(f"看這個 https://www.douyin.com/video/{_VIDEO_ID}。好笑") == [
        f"https://www.douyin.com/video/{_VIDEO_ID}"
    ]


def test_short_link_resolves_via_location_without_fetching_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short link is resolved from the Location header alone.

    Following the redirect would fetch `share/video/`, a path this class never reads and whose
    WAF quota it must not spend.
    """
    target = f"https://www.iesdouyin.com/share/video/{_VIDEO_ID}/?region=TW"

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        assert kwargs["allow_redirects"] is False
        return _FakeResponse(status_code=302, headers={"Location": target})

    calls = _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    assert downloader._resolve_aweme_id(url="https://v.douyin.com/NdlfIZPcgz4") == _VIDEO_ID
    assert [call["url"] for call in calls] == ["https://v.douyin.com/NdlfIZPcgz4"]


def test_scheme_less_short_link_is_still_fetchable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scheme-less paste must reach the network layer with a scheme attached.

    The router accepts `v.douyin.com/xxx`, and once it does there is no yt-dlp fallback left, so
    handing that string to requests verbatim would make the link permanently unresolvable.
    """
    target = f"https://www.iesdouyin.com/share/video/{_VIDEO_ID}/"
    calls = _install_session(
        monkeypatch=monkeypatch,
        handler=lambda url, kwargs: _FakeResponse(status_code=302, headers={"Location": target}),
    )
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    assert downloader._resolve_aweme_id(url="v.douyin.com/NdlfIZPcgz4") == _VIDEO_ID
    assert calls[0]["url"] == "https://v.douyin.com/NdlfIZPcgz4"


def test_short_link_redirecting_off_douyin_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short link pointing at another ByteDance site must not be followed."""
    _install_session(
        monkeypatch=monkeypatch,
        handler=lambda url, kwargs: _FakeResponse(
            status_code=302, headers={"Location": "https://www.ixigua.com/7123"}
        ),
    )
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    with pytest.raises(DouyinError, match="Not a Douyin post link"):
        downloader._resolve_aweme_id(url="https://v.douyin.com/xyz")


@pytest.mark.parametrize(argnames="page_key", argvalues=["note", "video"])
def test_loader_data_key_is_scanned_not_hardcoded(
    monkeypatch: pytest.MonkeyPatch, page_key: str
) -> None:
    """The loaderData key follows the URL path, so both `note_` and `video_` must parse."""
    _install_session(
        monkeypatch=monkeypatch,
        handler=lambda url, kwargs: _FakeResponse(
            text=_ok_page(item=_VIDEO_ITEM, page_key=page_key)
        ),
    )
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    post = downloader.parse_metadata(url=f"https://www.douyin.com/video/{_VIDEO_ID}")
    assert post.title == "因为一顿饭差点大打出手"
    assert post.author_name == "真探唐仁杰"
    assert post.is_photo is False


def test_photo_post_is_not_misread_as_a_video(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gallery carries a non-empty play_addr, so the branch must key on aweme_type."""
    _install_session(
        monkeypatch=monkeypatch,
        handler=lambda url, kwargs: _FakeResponse(text=_ok_page(item=_PHOTO_ITEM)),
    )
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    post = downloader.parse_metadata(url=f"https://www.douyin.com/note/{_PHOTO_ID}")
    assert post.is_photo is True
    assert post.video_id == ""
    # The clean JPEG (last entry of url_list), never the watermarked download_url_list.
    assert post.image_urls == ["https://cdn/0.jpeg", "https://cdn/1.jpeg", "https://cdn/2.jpeg"]
    assert not any("water" in url for url in post.image_urls)


@pytest.mark.parametrize(
    argnames=("quality", "ratio"),
    argvalues=[("best", "1080p"), ("high", "1080p"), ("medium", "720p"), ("low", "540p")],
)
def test_play_url_drops_the_watermark_and_maps_quality(quality: VideoQuality, ratio: str) -> None:
    """The play endpoint replaces playwm, and each preset maps to a ratio."""
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)
    url = downloader._play_url(video_id="vid123", quality=quality)

    assert "/aweme/v1/play/" in url
    assert "playwm" not in url
    assert f"ratio={ratio}" in url


def test_filtered_post_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deleted or private post arrives as HTTP 200 with an empty item_list."""
    page = _router_html(
        page_key="note",
        video_info={
            "item_list": [],
            "filter_list": [
                {
                    "filter_reason": "SYSTEM_ITEM_NOT_EXIST",
                    "detail_msg": "内容不存在",
                    "notice": "",
                }
            ],
        },
    )
    _install_session(monkeypatch=monkeypatch, handler=lambda url, kwargs: _FakeResponse(text=page))
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    with pytest.raises(DouyinUnavailableError, match="内容不存在"):
        downloader.parse_metadata(url=f"https://www.douyin.com/video/{_VIDEO_ID}")


@pytest.mark.parametrize(
    argnames="marker", argvalues=["waf-jschallenge", "out-sha256.js", "byted_acrawler", "captcha"]
)
def test_bot_wall_is_retryable_and_never_reported_as_missing(
    monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
    """A challenge page must raise the retryable error, not the "post is gone" one.

    Reporting a WAF block as a missing post would tell the user their working link is dead.
    """
    _install_session(
        monkeypatch=monkeypatch,
        handler=lambda url, kwargs: _FakeResponse(text=f"<html><script src='{marker}'></script>"),
    )
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    with pytest.raises(DouyinBlockedError):
        downloader.parse_metadata(url=f"https://www.douyin.com/video/{_VIDEO_ID}")
    # The retryable error must not be mistaken for the unavailable one by an except clause.
    assert not issubclass(DouyinBlockedError, DouyinUnavailableError)


def test_unreadable_page_without_a_challenge_marker_is_a_plain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structure change is neither a block nor a missing post."""
    _install_session(
        monkeypatch=monkeypatch, handler=lambda url, kwargs: _FakeResponse(text="<html></html>")
    )
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    with pytest.raises(DouyinError) as excinfo:
        downloader.parse_metadata(url=f"https://www.douyin.com/video/{_VIDEO_ID}")
    assert not isinstance(excinfo.value, DouyinBlockedError | DouyinUnavailableError)


def test_a_moved_payload_shape_is_unreadable_never_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `videoInfoRes` that no longer validates must not read as a deleted post.

    The models answer a moved shape at the boundary rather than several frames later, and
    which outcome it lands on is the load-bearing half: `DouyinUnavailableError` is the one
    wording that sends somebody off to re-check a link that is perfectly fine.
    """
    page = _router_html(page_key="note", video_info={"item_list": {"0": _VIDEO_ITEM}})
    _install_session(monkeypatch=monkeypatch, handler=lambda url, kwargs: _FakeResponse(text=page))
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    with pytest.raises(DouyinError) as excinfo:
        downloader.parse_metadata(url=f"https://www.douyin.com/video/{_VIDEO_ID}")
    assert not isinstance(excinfo.value, DouyinBlockedError | DouyinUnavailableError)


def test_share_page_is_fetched_from_the_note_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normalisation must target share/note, which serves both post types."""
    calls = _install_session(
        monkeypatch=monkeypatch,
        handler=lambda url, kwargs: _FakeResponse(text=_ok_page(item=_VIDEO_ITEM)),
    )
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)
    downloader.parse_metadata(url=f"https://www.douyin.com/video/{_VIDEO_ID}")

    assert calls[0]["url"] == f"https://www.iesdouyin.com/share/note/{_VIDEO_ID}"
    assert "iPhone" in calls[0]["headers"]["User-Agent"]


def test_repeated_lookup_reuses_the_cached_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A link posted twice costs one fetch, which is what keeps the WAF quiet."""
    calls = _install_session(
        monkeypatch=monkeypatch,
        handler=lambda url, kwargs: _FakeResponse(text=_ok_page(item=_VIDEO_ITEM)),
    )
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)
    url = f"https://www.douyin.com/video/{_VIDEO_ID}"

    downloader.parse_metadata(url=url)
    downloader.parse_metadata(url=url)

    assert len(calls) == 1


def test_a_short_link_is_resolved_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeat paste of the same short link costs no redirect probe at all.

    Auto-expansion turns every paste into a request, and the WAF bans by volume, so the second
    lookup must reach Douyin fewer times than the first — not merely produce the same id.
    """
    short_url = "https://v.douyin.com/abc123"

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        # Matched as a prefix, not a substring: a bare `in` would also accept
        # `https://evil.test/?x=v.douyin.com`, and this stub decides which endpoint was hit.
        if url.startswith(short_url):
            return _FakeResponse(
                headers={"Location": f"https://www.iesdouyin.com/share/video/{_VIDEO_ID}/"}
            )
        return _FakeResponse(text=_ok_page(item=_VIDEO_ITEM))

    calls = _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    assert downloader._resolve_aweme_id(url=short_url) == _VIDEO_ID
    assert len(calls) == 1
    assert downloader._resolve_aweme_id(url=short_url) == _VIDEO_ID
    assert len(calls) == 1  # served from the cache; the short-link host was never touched again


def test_download_video_writes_the_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A video post produces exactly one file named after the post id."""

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        if "share/note" in url:
            return _FakeResponse(text=_ok_page(item=_VIDEO_ITEM))
        return _FakeResponse(body=b"video-bytes")

    _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix())

    result = downloader.download(url=f"https://www.douyin.com/video/{_VIDEO_ID}")

    assert result.is_photo is False
    assert result.filenames == [tmp_path / f"{_VIDEO_ID}.mp4"]
    assert result.filenames[0].read_bytes() == b"video-bytes"
    assert result.omitted_images == 0


def test_download_gallery_honours_the_cap_and_reports_the_remainder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A capped gallery still reports the images it left behind, never silently."""

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        if "share/note" in url:
            return _FakeResponse(text=_ok_page(item=_PHOTO_ITEM))
        return _FakeResponse(body=b"image-bytes")

    _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix())

    result = downloader.download(url=f"https://www.douyin.com/note/{_PHOTO_ID}", max_images=2)

    assert result.is_photo is True
    assert len(result.filenames) == 2
    assert result.total_images == 3
    assert result.omitted_images == 1


def test_download_retries_a_stalled_transfer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The media CDN stalls intermittently, so a failed attempt is retried from scratch."""
    attempts = {"count": 0}

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        if "share/note" in url:
            return _FakeResponse(text=_ok_page(item=_VIDEO_ITEM))
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise douyin_module.RequestException("read timed out")
        return _FakeResponse(body=b"video-bytes")

    _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix())

    result = downloader.download(url=f"https://www.douyin.com/video/{_VIDEO_ID}")

    assert result.filenames[0].read_bytes() == b"video-bytes"
    assert attempts["count"] == 2


def test_download_gives_up_after_max_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A permanently failing transfer raises rather than leaving a truncated file behind."""

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        if "share/note" in url:
            return _FakeResponse(text=_ok_page(item=_VIDEO_ITEM))
        # Dies part-way through the body, so a partial file really is on disk when the attempt
        # fails. A stub that raised before the first chunk would make the cleanup assertion below
        # pass whether or not the cleanup exists.
        return _FakeResponse(body=b"half-a-video", stall_mid_stream=True)

    _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix(), max_retries=2)

    with pytest.raises(DouyinError, match="Failed to download"):
        downloader.download(url=f"https://www.douyin.com/video/{_VIDEO_ID}")
    assert list(tmp_path.iterdir()) == []


def test_oversize_content_length_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A declared length over the cap aborts before the body is read, so no file is written.

    The whole point of the guard is to spend a couple of seconds instead of a whole time
    budget, so an assertion that merely checks the raise would pass even if the download ran
    to completion first.
    """
    read_bodies = {"count": 0}

    class _CountingResponse(_FakeResponse):
        def iter_content(self, chunk_size: int) -> Iterator[bytes]:
            """Records that the body was read at all."""
            read_bodies["count"] += 1
            yield from super().iter_content(chunk_size=chunk_size)

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        if "share/note" in url:
            return _FakeResponse(text=_ok_page(item=_VIDEO_ITEM))
        return _CountingResponse(body=b"x" * 100, headers={"Content-Length": "100"})

    calls = _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(DouyinTooLargeError):
        downloader.download(url=f"https://www.douyin.com/video/{_VIDEO_ID}", max_bytes=10)

    assert read_bodies["count"] == 0  # aborted on the header, never streamed
    assert list(tmp_path.iterdir()) == []
    # Deterministic failure: retrying would only re-fetch the same oversize file.
    assert len([call for call in calls if "share/note" not in str(call["url"])]) == 1


def test_oversize_stream_without_a_content_length_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing or lying Content-Length is caught mid-stream, and the partial file is removed."""

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        if "share/note" in url:
            return _FakeResponse(text=_ok_page(item=_VIDEO_ITEM))
        return _FakeResponse(body=b"x" * 100)  # no Content-Length header at all

    _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(DouyinTooLargeError):
        downloader.download(url=f"https://www.douyin.com/video/{_VIDEO_ID}", max_bytes=10)

    assert list(tmp_path.iterdir()) == []


def test_download_under_the_cap_is_unaffected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file inside the cap downloads exactly as it does with no cap at all."""

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        if "share/note" in url:
            return _FakeResponse(text=_ok_page(item=_VIDEO_ITEM))
        return _FakeResponse(body=b"video-bytes", headers={"Content-Length": "11"})

    _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix())

    result = downloader.download(url=f"https://www.douyin.com/video/{_VIDEO_ID}", max_bytes=1024)

    assert result.filenames[0].read_bytes() == b"video-bytes"


def test_download_reuses_a_caller_supplied_post(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Passing an already-parsed post skips the parse entirely.

    Asserting only on the downloaded bytes would pass even if the post were re-parsed, so this
    watches `parse_metadata` itself; the payload cache would otherwise hide the extra work.
    """
    parses = {"count": 0}
    original_parse = DouyinDownloader.parse_metadata

    def counting_parse(self: DouyinDownloader, url: str) -> object:
        """Counts every metadata parse the downloader performs."""
        parses["count"] += 1
        return original_parse(self, url=url)

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        if "share/note" in url:
            return _FakeResponse(text=_ok_page(item=_VIDEO_ITEM))
        return _FakeResponse(body=b"video-bytes")

    _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix())
    url = f"https://www.douyin.com/video/{_VIDEO_ID}"
    post = downloader.parse_metadata(url=url)
    monkeypatch.setattr(target=DouyinDownloader, name="parse_metadata", value=counting_parse)

    result = downloader.download(url=url, post=post)
    assert result.filenames[0].read_bytes() == b"video-bytes"
    assert parses["count"] == 0

    # Without a post the download resolves it itself, so the counter proves it is watching.
    result = downloader.download(url=url)
    assert result.filenames[0].read_bytes() == b"video-bytes"
    assert parses["count"] == 1


@pytest.mark.parametrize(
    argnames=("url", "expected"),
    argvalues=[
        (f"https://www.douyin.com/video/{_VIDEO_ID}", True),
        (f"https://www.douyin.com/user/MS4wLjABAAAAMOcq?modal_id={_VIDEO_ID}", True),
        ("https://v.douyin.com/abc123", True),
        ("https://jx.douyin.com/abc123", True),
        (f"https://www.iesdouyin.com/share/note/{_PHOTO_ID}", True),
        ("https://www.douyin.com/user/MS4wLjABAAAAMOcq", False),
        ("https://live.douyin.com/123456", False),
        ("https://www.douyin.com/search/whatever", False),
        # Feed pages share a short link's single-segment shape but not its host.
        ("https://www.douyin.com/jingxuan", False),
        ("https://www.douyin.com/discover", False),
        ("https://www.douyin.com/hot", False),
        ("https://douyin.com/follow", False),
    ],
)
def test_post_url_detection_separates_posts_from_profiles(url: str, expected: bool) -> None:
    """Only a post-shaped link may be claimed automatically.

    `DOUYIN_URL_RE` matches the host rather than the path, so without this a pasted profile or
    live room would earn a warning reaction, and spend a Douyin request to discover it was
    never a post.
    """
    assert is_douyin_post_url(url=url) is expected


def test_a_download_never_recreates_a_removed_output_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_download_to` must not re-create the output folder per file.

    A cancelled caller cannot stop the worker thread, so it may remove the scratch dir
    mid-download; re-creating it per file would silently strand every later image there
    forever. Failing the open instead turns the removal into the stop signal.
    """
    _install_session(
        monkeypatch=monkeypatch, handler=lambda url, kwargs: _FakeResponse(body=b"image-bytes")
    )
    scratch = tmp_path / "gone"  # the scratch dir a cancelled caller has already removed
    downloader = DouyinDownloader(output_folder=scratch.as_posix())

    with pytest.raises(FileNotFoundError):
        downloader._download_to(url="https://cdn.test/1.jpg", filename="1.jpg")
    assert not scratch.exists()  # nothing re-created it behind the caller's back


def test_a_removed_output_folder_stops_a_download_already_streaming(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A scratch dir removed mid-transfer stops the file already being written, too.

    Failing the next open stops only the next file, and a lone clip has none: an open handle
    keeps taking writes after the removal, so the abandoned worker would pull the whole clip
    into a deleted file.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pulled: list[int] = []

    class _RemovedMidStream(_FakeResponse):
        """Removes the scratch dir after the first chunk, the way a caller giving up would."""

        def iter_content(self, chunk_size: int) -> Iterator[bytes]:
            """Yields five chunks, recording each one pulled."""
            for index in range(5):
                if index == 1:
                    shutil.rmtree(path=scratch)
                pulled.append(index)
                yield b"x" * chunk_size

    calls = _install_session(
        monkeypatch=monkeypatch, handler=lambda url, kwargs: _RemovedMidStream()
    )
    downloader = DouyinDownloader(output_folder=scratch.as_posix())

    with pytest.raises(FileNotFoundError):
        downloader._download_to(url="https://cdn.test/clip.mp4", filename="clip.mp4")
    assert pulled == [0, 1]  # nothing read past the chunk that arrived after the removal
    assert len(calls) == 1  # a removal is not a stall worth retrying


def test_payload_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The share-payload cache must not grow one entry per link the bot has ever seen."""
    _install_session(
        monkeypatch=monkeypatch,
        handler=lambda url, kwargs: _FakeResponse(text=_ok_page(item=_VIDEO_ITEM)),
    )
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    for index in range(douyin_module._PAYLOAD_CACHE_MAX_ENTRIES + 25):
        downloader.parse_metadata(
            url=f"https://www.douyin.com/video/{7000000000000000000 + index}"
        )

    assert len(douyin_module._PAYLOAD_CACHE) <= douyin_module._PAYLOAD_CACHE_MAX_ENTRIES


def test_local_write_failure_leaves_no_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed local write must clean up its own partial file.

    Only network errors are retried, so a disk failure propagates immediately; the caller's
    gallery cleanup only knows about files it already accepted, so this one has to remove itself.
    """

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        if "share/note" in url:
            return _FakeResponse(text=_ok_page(item=_VIDEO_ITEM))
        return _FakeResponse(body=b"video-bytes")

    _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix())

    real_open = Path.open

    def failing_open(self: Path, mode: str = "r") -> IO[bytes]:
        """Writes a partial file and then fails, as a full disk would.

        The downloader only ever opens with a positional mode, so the stub mirrors that shape.
        """
        handle: IO[bytes] = real_open(self, mode)
        original_write = handle.write

        def write(data: bytes) -> int:
            original_write(data)
            raise OSError(28, "No space left on device")

        # Simulate a mid-write disk failure by shadowing the handle's bound write.
        monkeypatch.setattr(target=handle, name="write", value=write)
        return handle

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError, match="No space left"):
        downloader.download(url=f"https://www.douyin.com/video/{_VIDEO_ID}")

    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    argnames="failure",
    argvalues=[
        requests.ReadTimeout("stalled"),
        requests.exceptions.ChunkedEncodingError("connection dropped mid-body"),
    ],
    ids=["read-timeout", "cut-off-body"],
)
def test_a_stalled_share_page_read_is_retryable(
    monkeypatch: pytest.MonkeyPatch, failure: requests.RequestException
) -> None:
    """A read that never got an answer or got cut off is a come-back-later, never a missing post.

    Douyin raises its own classes, so its own fetch is what has to be asked: a synthetic error
    handed to the shared classifier passes whether or not this reader ever raises a retryable one.
    """

    def stall(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        """Fails the read the way the network did."""
        del url, kwargs
        raise failure

    _install_session(monkeypatch=monkeypatch, handler=stall)
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    with pytest.raises(DouyinBlockedError):
        downloader.parse_metadata(url=f"https://www.douyin.com/video/{_VIDEO_ID}")


@pytest.mark.parametrize(
    argnames=("status", "retryable"), argvalues=[(429, True), (503, True), (404, False)]
)
def test_a_refused_short_link_is_retryable_only_when_http_says_so(
    monkeypatch: pytest.MonkeyPatch, status: int, retryable: bool
) -> None:
    """A refused short-link hop carries no `Location`, which is not the same as no post."""
    _install_session(
        monkeypatch=monkeypatch, handler=lambda url, kwargs: _FakeResponse(status_code=status)
    )
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)

    with pytest.raises(DouyinError) as raised:
        downloader._resolve_aweme_id(url="https://v.douyin.com/AbCdEf12/")
    assert isinstance(raised.value, DouyinBlockedError) is retryable


@pytest.mark.parametrize(
    argnames=("status", "retryable"),
    argvalues=[(403, False), (404, False), (429, True), (503, True)],
)
def test_a_refused_media_download_is_retried_only_when_http_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int, retryable: bool
) -> None:
    """A status a retry will not change is asked once and never earns the retry-later mark."""
    calls = _install_session(
        monkeypatch=monkeypatch, handler=lambda url, kwargs: _FakeResponse(status_code=status)
    )
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix())

    with pytest.raises(DouyinError) as raised:
        downloader._download_to(url="https://cdn.test/v.mp4", filename="v.mp4")

    assert len(calls) == (downloader.max_retries if retryable else 1)
    assert isinstance(raised.value, LinkRetryableError) is retryable


@pytest.mark.parametrize(
    argnames="failure",
    argvalues=[
        requests.ReadTimeout("stalled"),
        requests.exceptions.ChunkedEncodingError("connection dropped mid-body"),
    ],
    ids=["read-timeout", "cut-off-body"],
)
def test_a_stalled_media_download_is_retryable_but_not_the_bot_wall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: requests.RequestException
) -> None:
    """A transfer that keeps stalling or gets cut off is retryable, and is not the bot wall.

    The download is the request that stalls in practice, so its retries running out is the
    ordinary Douyin failure; reported flat, it would read as a post with nothing showable in it.
    """

    def stall(**kwargs: object) -> Path:
        """Never completes, the way a stalling CDN transfer does not."""
        del kwargs
        raise failure

    monkeypatch.setattr(target=douyin_module, name="stream_to_file", value=stall)
    downloader = DouyinDownloader(output_folder=str(tmp_path))

    with pytest.raises(DouyinTransferError) as raised:
        downloader._download_to(url="https://example.test/v.mp4", filename="v.mp4")

    # Retryable to the expansion, but NOT the bot wall: `/download_video` answers in words,
    # and blaming a wall sends someone off to wait out something that was never there.
    assert isinstance(raised.value, LinkRetryableError)
    assert not isinstance(raised.value, DouyinBlockedError)
