"""Tests for the Douyin share-page parser and downloader.

Every HTTP call is stubbed. Besides the usual reason (tests must not depend on a live site),
Douyin bans a share path for tens of minutes once it is hit hard, so a test suite that reached
the real endpoint would take the whole deployment down with it.
"""

import json
from typing import IO, Any, Self
from pathlib import Path
import tempfile
from collections.abc import Callable, Iterator

import pytest

from discordbot.cogs import video
from discordbot.cogs.video import VideoCogs
import discordbot.utils.douyin as douyin_module
from discordbot.utils.douyin import (
    DOUYIN_URL_RE,
    DouyinError,
    DouyinDownload,
    DouyinDownloader,
    DouyinBlockedError,
    DouyinUnavailableError,
    is_douyin_url,
)
from discordbot.utils.media_delivery import (
    MediaHostingConfig,
    MediaHostingService,
    MediaDeliveryPlanner,
)

from tests.helpers.discord_mocks import FakeInteraction

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
        """Mimics requests' status check."""
        if self.status_code >= 400:
            raise douyin_module.RequestException(f"status {self.status_code}")

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
) -> list[dict[str, object]]:
    """Replaces requests.Session with a stub driven by `handler`; returns the captured calls."""
    calls: list[dict[str, object]] = []

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
    """Keeps the module-level share-payload cache from leaking between tests."""
    douyin_module._PAYLOAD_CACHE.clear()
    yield
    douyin_module._PAYLOAD_CACHE.clear()


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
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)
    assert downloader._extract_id(url=url) == expected


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
def test_play_url_drops_the_watermark_and_maps_quality(quality: str, ratio: str) -> None:
    """The play endpoint replaces playwm, and each preset maps to a ratio."""
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)
    url = downloader._play_url(video_id="vid123", quality=quality)

    assert "/aweme/v1/play/" in url
    assert "playwm" not in url
    assert f"ratio={ratio}" in url


def test_unknown_quality_falls_back_to_best() -> None:
    """An unrecognised preset must not produce a malformed ratio."""
    downloader = DouyinDownloader(output_folder=_SCRATCH_DIR)
    assert "ratio=1080p" in downloader._play_url(video_id="vid123", quality="bogus")


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


def test_download_video_writes_the_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A video post produces exactly one file named after the post id."""

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        if "share/note" in url:
            return _FakeResponse(text=_ok_page(item=_VIDEO_ITEM))
        return _FakeResponse(body=b"video-bytes")

    _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix())

    with downloader.download(url=f"https://www.douyin.com/video/{_VIDEO_ID}") as result:
        assert result.is_photo is False
        assert result.filenames == [tmp_path / f"{_VIDEO_ID}.mp4"]
        assert result.filenames[0].read_bytes() == b"video-bytes"
        assert result.omitted_images == 0

    assert not (tmp_path / f"{_VIDEO_ID}.mp4").exists()  # cleaned up on context exit


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

    with downloader.download(
        url=f"https://www.douyin.com/note/{_PHOTO_ID}", max_images=2
    ) as result:
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

    with downloader.download(url=f"https://www.douyin.com/video/{_VIDEO_ID}") as result:
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


def test_failed_gallery_leaves_no_images_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A gallery that dies part-way must not strand the images it already wrote.

    Nothing is returned on failure, so the caller never gets a handle to clean up with; the
    cleanup has to happen inside the downloader or the files live in the temp dir for good.
    """
    downloaded = {"count": 0}

    def handler(url: str, kwargs: dict[str, object]) -> _FakeResponse:
        if "share/note" in url:
            return _FakeResponse(text=_ok_page(item=_PHOTO_ITEM))
        downloaded["count"] += 1
        if downloaded["count"] == 3:
            raise douyin_module.RequestException("gone")
        return _FakeResponse(body=b"image-bytes")

    _install_session(monkeypatch=monkeypatch, handler=handler)
    downloader = DouyinDownloader(output_folder=tmp_path.as_posix(), max_retries=1)

    with pytest.raises(DouyinError):
        downloader.download(url=f"https://www.douyin.com/note/{_PHOTO_ID}")

    assert list(tmp_path.iterdir()) == []


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

    def failing_open(self: Path, *args: object, **kwargs: object) -> IO[bytes]:
        """Writes a partial file and then fails, as a full disk would."""
        handle = real_open(self, *args, **kwargs)  # type: ignore[arg-type]  # passthrough stub
        original_write = handle.write

        def write(data: bytes) -> int:
            original_write(data)
            raise OSError(28, "No space left on device")

        handle.write = write  # type: ignore[method-assign]  # simulating a mid-write disk failure
        return handle

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError, match="No space left"):
        downloader.download(url=f"https://www.douyin.com/video/{_VIDEO_ID}")

    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


class _StubDouyinDownloader:
    """Downloader stub returning a canned result or raising a canned error."""

    def __init__(self, outcome: DouyinDownload | BaseException) -> None:
        """Stores what the next download call should produce."""
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def download(self, url: str, quality: str, max_images: int | None = None) -> DouyinDownload:
        """Records the request and returns the canned result, or raises."""
        self.calls.append({"url": url, "quality": quality, "max_images": max_images})
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _install_cog(
    monkeypatch: pytest.MonkeyPatch, outcome: DouyinDownload | BaseException
) -> tuple[VideoCogs, _StubDouyinDownloader]:
    """Builds a VideoCogs wired to a stub Douyin downloader."""
    cog = VideoCogs(bot=object())
    stub = _StubDouyinDownloader(outcome=outcome)
    monkeypatch.setattr(video, "DouyinDownloader", lambda output_folder: stub)
    return cog, stub


async def test_cog_routes_douyin_away_from_ytdlp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Douyin link must never reach the yt-dlp downloader, whose extractor cannot serve it."""
    clip = tmp_path / f"{_VIDEO_ID}.mp4"
    clip.write_bytes(b"0" * 128)
    cog, stub = _install_cog(
        monkeypatch=monkeypatch,
        outcome=DouyinDownload(title="t", is_photo=False, filenames=[clip]),
    )

    def _fail(output_folder: str) -> None:
        raise AssertionError("yt-dlp must not be used for a Douyin URL")

    monkeypatch.setattr(video, "VideoDownloader", _fail)
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
        image = tmp_path / f"{_PHOTO_ID}_{index}.jpg"
        image.write_bytes(b"0" * 16)
        images.append(image)

    cog, _stub = _install_cog(
        monkeypatch=monkeypatch,
        outcome=DouyinDownload(title="t", is_photo=True, filenames=images, total_images=48),
    )
    interaction = FakeInteraction()

    await VideoCogs.download_video.callback(
        cog, interaction, url=f"https://www.douyin.com/note/{_PHOTO_ID}", quality="best"
    )

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
        image = tmp_path / f"{_PHOTO_ID}_{index}.jpg"
        image.write_bytes(bytes([index]) * 4096)  # distinct bytes so hosting cannot dedup them
        images.append(image)

    cog, _stub = _install_cog(
        monkeypatch=monkeypatch,
        outcome=DouyinDownload(
            title="t", is_photo=True, filenames=images, total_images=len(images)
        ),
    )
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(
            config=MediaHostingConfig(
                MEDIA_HOSTING_ENABLED=True,
                MEDIA_HOSTING_BASE_URL="https://media.test",
                MEDIA_HOSTING_SERVE_DIR=serve_dir.as_posix(),
            )
        )
    )
    interaction = FakeInteraction(filesize_limit=1024)

    await VideoCogs.download_video.callback(
        cog, interaction, url=f"https://www.douyin.com/note/{_PHOTO_ID}", quality="best"
    )

    content = interaction.edits[-1]["content"]
    hosted = [line for line in content.splitlines() if line.startswith("https://media.test/")]
    assert len(hosted) == len(images)
    assert "檔案無法下載" not in content


async def test_cog_reports_a_blocked_request_as_retryable_not_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bot wall must not be reported as a deleted post; the link is fine, the site is not."""
    cog, _stub = _install_cog(monkeypatch=monkeypatch, outcome=DouyinBlockedError("challenge"))
    interaction = FakeInteraction()

    await VideoCogs.download_video.callback(
        cog, interaction, url=f"https://www.douyin.com/video/{_VIDEO_ID}", quality="best"
    )

    content = interaction.edits[-1]["content"]
    assert "稍後再試" in content
    assert "刪除" not in content


async def test_cog_reports_an_unavailable_post(monkeypatch: pytest.MonkeyPatch) -> None:
    """A filtered post gets its own message rather than a generic download failure."""
    cog, _stub = _install_cog(
        monkeypatch=monkeypatch, outcome=DouyinUnavailableError("SYSTEM_ITEM_NOT_EXIST")
    )
    interaction = FakeInteraction()

    await VideoCogs.download_video.callback(
        cog, interaction, url=f"https://www.douyin.com/video/{_VIDEO_ID}", quality="best"
    )

    assert "已被刪除或設為私人" in interaction.edits[-1]["content"]


async def test_cog_falls_back_to_a_generic_message_on_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any other Douyin failure still leaves the user with a message, never a silent no-op."""
    cog, _stub = _install_cog(monkeypatch=monkeypatch, outcome=DouyinError("boom"))
    interaction = FakeInteraction()

    await VideoCogs.download_video.callback(
        cog, interaction, url=f"https://www.douyin.com/video/{_VIDEO_ID}", quality="best"
    )

    assert "檔案無法下載" in interaction.edits[-1]["content"]


async def test_cog_reports_a_non_douyin_error_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure that is not a DouyinError must not escape and strand the placeholder.

    The Douyin branch runs before the command's own try block, and the bot registers no
    application-command error handler, so an escaping exception would leave the user looking at
    "正在下載影片..." indefinitely.
    """
    cog, _stub = _install_cog(
        monkeypatch=monkeypatch, outcome=OSError(28, "No space left on device")
    )
    interaction = FakeInteraction()

    await VideoCogs.download_video.callback(
        cog, interaction, url=f"https://www.douyin.com/video/{_VIDEO_ID}", quality="best"
    )

    assert "檔案無法下載" in interaction.edits[-1]["content"]


async def test_cog_posts_the_hosted_url_when_the_clip_is_oversize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An oversize clip is delivered as a hosted URL, the fallback the README advertises.

    Uses a real hosting service because the bug this guards against only appears once hosting
    actually moves the source file out of the download folder.
    """
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    clip = tmp_path / f"{_VIDEO_ID}.mp4"
    clip.write_bytes(b"0" * 4096)

    cog, _stub = _install_cog(
        monkeypatch=monkeypatch,
        outcome=DouyinDownload(title="t", is_photo=False, filenames=[clip]),
    )
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(
            config=MediaHostingConfig(
                MEDIA_HOSTING_ENABLED=True,
                MEDIA_HOSTING_BASE_URL="https://media.test",
                MEDIA_HOSTING_SERVE_DIR=serve_dir.as_posix(),
            )
        )
    )
    interaction = FakeInteraction(filesize_limit=1024)

    await VideoCogs.download_video.callback(
        cog, interaction, url=f"https://www.douyin.com/video/{_VIDEO_ID}", quality="best"
    )

    content = interaction.edits[-1]["content"]
    # Asserted per line rather than as a substring: the URL has to start its own line for Discord
    # to render it, so an anywhere-in-the-body match would accept a message Discord would not link.
    assert any(line.startswith("https://media.test/") for line in content.splitlines())
    assert "檔案無法下載" not in content
    # The file really was hosted, so discarding the URL would have lost a completed upload.
    assert list(serve_dir.glob("*.mp4"))
