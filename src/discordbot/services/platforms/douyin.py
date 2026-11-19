"""Douyin URL parsing, share-page scraping, and media download helpers.

Reads the server-rendered `iesdouyin.com/share/note/<id>` page rather than yt-dlp's unsigned
`DouyinIE` API, supporting both videos (up to 1080p) and photo posts without login cookies.
"""

import re
import json
import time
from typing import Any, ClassVar
from pathlib import Path
import threading
from collections import OrderedDict
from urllib.parse import urljoin, parse_qs, urlparse

import logfire
from pydantic import Field, BaseModel, ValidationError, model_validator
import requests
from requests.exceptions import RequestException

from discordbot.utils.urls import (
    URL_START_ANCHOR,
    normalized_url,
    normalized_host,
    host_matches_domain,
)
from discordbot.typings.video import VideoQuality
from discordbot.typings.timeouts import (
    DOUYIN_DOWNLOAD_MAX_RETRIES,
    DOUYIN_DOWNLOAD_TIMEOUT_SECONDS,
    DOUYIN_METADATA_TIMEOUT_SECONDS,
)
from discordbot.utils.link_errors import (
    LinkReadError,
    LinkRetryableError,
    LinkUnavailableError,
    link_fetch_error,
    is_retryable_fetch_failure,
)
from discordbot.utils.asyncio_locks import KeyedLockManager, LoopLocalSemaphore
from discordbot.services.platforms.base import PlatformDownloader
from discordbot.services.platforms.file_downloads import DownloadTooLargeError, stream_to_file

# Whole-label host anchor; ASCII path ending on [A-Za-z0-9_/-] stops before CJK share punctuation.
DOUYIN_URL_RE = re.compile(
    pattern=rf"{URL_START_ANCHOR}https?://(?:[A-Za-z0-9-]+\.)*(?:douyin|iesdouyin)\.com"
    r"/[A-Za-z0-9_.?=&%/-]*[A-Za-z0-9_/-]"
)

# Non-greedy up to `</script>` so nested braces inside `_ROUTER_DATA` do not truncate the JSON.
_ROUTER_DATA_RE = re.compile(
    pattern=r"_ROUTER_DATA\s*=\s*(\{.*?\});?\s*</script>", flags=re.DOTALL
)

_PATH_ID_RE = re.compile(pattern=r"/(?:video|note|slides)/(\d+)")

# Rejects off-Douyin redirect targets such as `ixigua.com`.
_ALLOWED_HOSTS = frozenset({"douyin.com", "iesdouyin.com"})

# Hosts whose single-segment path is a short link, the one post link that carries no id.
_SHORT_LINK_HOSTS = frozenset({"v.douyin.com", "jx.douyin.com"})

# ByteDance WAF/JS-shell markers; retryable refusals, never missing posts.
_CHALLENGE_MARKERS = ("waf-jschallenge", "out-sha256.js", "byted_acrawler", "captcha")

# Douyin `aweme_type` values for photo posts (2 on web share, 68 in app; 4 is video).
_PHOTO_AWEME_TYPES = frozenset({2, 68})


def is_douyin_url(url: str) -> bool:
    """Returns whether `url` points at `douyin.com` or `iesdouyin.com` (or a subdomain of either).

    Matches whole host labels only and accepts scheme-less pastes (`v.douyin.com/xxx`).

    Args:
        url (str): The URL to test.
    """
    host = normalized_host(url=url)
    return any(host_matches_domain(host=host, domain=allowed) for allowed in _ALLOWED_HOSTS)


def _extract_post_id(url: str) -> str:
    """Reads the aweme id straight out of a URL, or returns an empty string."""
    parsed = urlparse(url=normalized_url(url=url))
    # `modal_id` wins over the path: `/user/<sec_uid>?modal_id=<id>` carries both.
    modal_ids = parse_qs(qs=parsed.query).get("modal_id")
    if modal_ids and modal_ids[0].isdigit():
        return modal_ids[0]

    match = _PATH_ID_RE.search(string=parsed.path)
    return match.group(1) if match else ""


def is_douyin_post_url(url: str) -> bool:
    """Returns whether a Douyin URL plausibly points at a single post.

    Requires an embedded post id or a single-segment path on a short-link host, rejecting
    profiles, live rooms, and feed pages.

    Args:
        url (str): A URL already known to be a Douyin URL.
    """
    if _extract_post_id(url=url):
        return True
    parsed = urlparse(url=normalized_url(url=url))
    if (parsed.hostname or "").lower() not in _SHORT_LINK_HOSTS:
        return False
    return len([segment for segment in parsed.path.split("/") if segment]) == 1


class DouyinError(LinkReadError):
    """Base error for classified Douyin lookup failures; inherits `LinkReadError`."""


class DouyinUnavailableError(DouyinError, LinkUnavailableError):
    """Douyin will not serve the post (deleted, private, region locked) or has no such page."""


class DouyinBlockedError(DouyinError, LinkRetryableError):
    """Douyin refused the request with a bot wall, cookie gate, or HTTP 429.

    Retryable: the post itself is fine.
    """


class DouyinTransferError(DouyinError, LinkRetryableError):
    """Retryable transport failure: stall, connection drop, 5xx, or exhausted media retries."""


class DouyinTooLargeError(DouyinError):
    """The media exceeds the caller's cap; deterministic and not retryable."""


def _douyin_fetch_error(error: RequestException, url: str) -> RuntimeError:
    """Returns the classified Douyin error for `error` on `url`."""
    verdict = link_fetch_error(error=error, url=url)
    if isinstance(verdict, LinkUnavailableError):
        return DouyinUnavailableError(str(verdict))
    if not isinstance(verdict, LinkRetryableError):
        return verdict
    if error.response is not None and error.response.status_code == 429:
        return DouyinBlockedError(str(verdict))
    return DouyinTransferError(str(verdict))


class DouyinMetadata(BaseModel):
    """Metadata for a single Douyin post, parsed without downloading media."""

    aweme_id: str = Field(..., description="Douyin's numeric post id.")
    title: str = Field(default="", description="Post caption, used as the display title.")
    author_name: str = Field(default="", description="Author's display nickname.")
    is_photo: bool = Field(default=False, description="True for a photo post, False for a video.")
    video_id: str = Field(
        default="", description="Douyin's internal video id, empty for a photo post."
    )
    image_urls: list[str] = Field(
        default_factory=list, description="Watermark-free image URLs, empty for a video."
    )


class DouyinDownload(BaseModel):
    """Files downloaded for one Douyin post."""

    is_photo: bool = Field(
        default=False, description="True when the files are images rather than a single video."
    )
    filenames: list[Path] = Field(
        default_factory=list, description="Local paths of the downloaded files."
    )
    total_images: int = Field(
        default=0, description="Image count in the source post, before any cap was applied."
    )

    @property
    def total_bytes(self) -> int:
        """Returns the combined byte size of the downloaded files still in the download folder."""
        return sum(path.stat().st_size for path in self.filenames if path.exists())

    @property
    def omitted_images(self) -> int:
        """Returns the number of images present in the post but not downloaded."""
        return max(0, self.total_images - len(self.filenames))


class _DouyinPayload(BaseModel):
    """Base model for Douyin share-page payloads; treats explicit JSON nulls as absent."""

    @model_validator(mode="before")
    @classmethod
    def _null_is_absent(cls, data: object) -> object:
        """Drops explicit nulls so each field falls back to its own default."""
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if value is not None}
        return data


class _DouyinPlayAddr(_DouyinPayload):
    """A video's play handle."""

    uri: str = Field(
        default="",
        description="Douyin's internal video id, which `_play_url` turns into a clean play URL.",
    )


class _DouyinVideo(_DouyinPayload):
    """A post's video object, which a photo post also carries for its slideshow render."""

    play_addr: _DouyinPlayAddr = Field(
        default_factory=_DouyinPlayAddr, description="The play handle naming the clip."
    )


class _DouyinAuthor(_DouyinPayload):
    """The posting account."""

    nickname: str = Field(default="", description="Author's display nickname.")


class _DouyinImage(_DouyinPayload):
    """One image of a photo post."""

    url_list: list[str] = Field(
        default_factory=list, description="Watermark-free URLs for this image, JPEG last."
    )


class _DouyinItem(_DouyinPayload):
    """One post, as `item_list[0]` of the share payload."""

    desc: str = Field(default="", description="Post caption.")
    aweme_type: int | None = Field(
        default=None, description="Douyin's post kind; see `_PHOTO_AWEME_TYPES`.", examples=[2, 4]
    )
    author: _DouyinAuthor = Field(
        default_factory=_DouyinAuthor, description="The posting account."
    )
    video: _DouyinVideo = Field(
        default_factory=_DouyinVideo, description="The post's video object."
    )
    images: list[_DouyinImage] = Field(
        default_factory=list, description="The post's images, empty for a video."
    )


class _DouyinFilterEntry(_DouyinPayload):
    """Refusal entry from `filter_list`; fields are alternative wordings read in priority order."""

    detail_msg: str = Field(default="", description="The fullest wording of the refusal.")
    notice: str = Field(default="", description="The short wording of the refusal.")
    filter_reason: str = Field(default="", description="The machine-ish reason code.")


class _DouyinVideoInfo(_DouyinPayload):
    """The `videoInfoRes` object from the share page's `_ROUTER_DATA`."""

    item_list: list[_DouyinItem] = Field(
        default_factory=list, description="The requested post, as a one-entry list."
    )
    filter_list: list[_DouyinFilterEntry] = Field(
        default_factory=list, description="Why the post was withheld, when `item_list` is empty."
    )


# Bounded LRU cache of validated payloads; TTL stays well under CDN `x-expires` lifetime.
_PAYLOAD_CACHE: OrderedDict[str, tuple[float, _DouyinVideoInfo]] = OrderedDict()
_PAYLOAD_CACHE_TTL_SECONDS = 300.0
_PAYLOAD_CACHE_MAX_ENTRIES = 128

# Permanent LRU cache mapping short-link URLs to aweme ids to skip repeat redirect probes.
_LINK_ID_CACHE: OrderedDict[str, str] = OrderedDict()
_LINK_ID_CACHE_MAX_ENTRIES = 512
# Guards both LRU dicts across worker threads; never held across network I/O.
_PAYLOAD_CACHE_LOCK = threading.Lock()


def _cached_link_id(url: str) -> str:
    """Returns the cached post id for a URL, or an empty string on a miss."""
    with _PAYLOAD_CACHE_LOCK:
        aweme_id = _LINK_ID_CACHE.get(url)
        if aweme_id is not None:
            _LINK_ID_CACHE.move_to_end(url)
            return aweme_id
    return ""


def _remember_link_id(url: str, aweme_id: str) -> None:
    """Records a URL's resolved post id, evicting the least recently used entry past the cap."""
    with _PAYLOAD_CACHE_LOCK:
        _LINK_ID_CACHE[url] = aweme_id
        _LINK_ID_CACHE.move_to_end(url)
        if len(_LINK_ID_CACHE) > _LINK_ID_CACHE_MAX_ENTRIES:
            _LINK_ID_CACHE.popitem(last=False)


# Process-wide guest `ttwid` cookie; retained across failures to avoid extra cookieless WAF hits.
_share_ttwid = ""


# Caps concurrent background Douyin fetches to avoid per-path WAF bans; commands bypass it.
DOUYIN_FETCH_CONCURRENCY = 2

douyin_fetch_semaphore = LoopLocalSemaphore(capacity_provider=lambda: DOUYIN_FETCH_CONCURRENCY)

# Collapses simultaneous requests for the same URL before they miss the payload cache.
douyin_url_locks: KeyedLockManager[str] = KeyedLockManager()


class DouyinDownloader(PlatformDownloader):
    """Downloads Douyin videos and photo posts via `iesdouyin.com/share/note/<id>`.

    Fetches with a mobile User-Agent and a guest `ttwid` cookie; `www.douyin.com` serves a
    `byted_acrawler` JS shell, and `share/note/<id>` serves both videos and photo posts. Short
    links resolve via `Location` headers only to stay off `share/video/` WAF quotas.
    """

    output_folder: str = Field(..., description="Directory where downloaded files are written.")
    timeout: int = Field(
        default=DOUYIN_METADATA_TIMEOUT_SECONDS,
        description="Timeout in seconds for a metadata request.",
        examples=[15, 30],
    )
    download_timeout: int = Field(
        default=DOUYIN_DOWNLOAD_TIMEOUT_SECONDS,
        description="Per-read timeout in seconds for a media download.",
        examples=[60],
    )
    max_retries: int = Field(
        default=DOUYIN_DOWNLOAD_MAX_RETRIES,
        description="Attempts made per media download before giving up.",
        examples=[3],
    )
    max_redirects: int = Field(
        default=5, description="Maximum redirect hops followed when resolving a short link."
    )

    mobile_user_agent: ClassVar[str] = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
    )

    # 1080p is source resolution on Douyin, so `best` and `high` both map to it.
    quality_ratios: ClassVar[dict[VideoQuality, str]] = {
        "best": "1080p",
        "high": "1080p",
        "medium": "720p",
        "low": "540p",
    }

    def _headers(self) -> dict[str, str]:
        """Returns the request headers Douyin's share page expects."""
        return {"User-Agent": self.mobile_user_agent, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}

    def _resolve_aweme_id(self, url: str) -> str:
        """Returns the numeric post id from `url`, resolving short links via `Location` headers."""
        cached = _cached_link_id(url=url)
        if cached:
            return cached

        # `is_douyin_url` accepts scheme-less pastes, which `requests` requires a scheme for.
        current = url if "://" in url else f"https://{url}"
        for _ in range(self.max_redirects + 1):
            # Re-check each hop; short links can redirect off-domain (e.g. `ixigua.com`).
            if not is_douyin_url(url=current):
                raise DouyinError(f"Not a Douyin post link: {url}")

            aweme_id = _extract_post_id(url=current)
            if aweme_id:
                _remember_link_id(url=url, aweme_id=aweme_id)
                return aweme_id

            location = self._redirect_target(url=current)
            if not location:
                break
            current = location

        raise DouyinError(f"Could not find a Douyin post id in: {url}")

    def _redirect_target(self, url: str) -> str:
        """Returns the absolute `Location` target of a single redirect hop without fetching the body."""
        try:
            with requests.Session() as session:
                response = session.get(
                    url,
                    headers=self._headers(),
                    allow_redirects=False,
                    timeout=self.timeout,
                    stream=True,
                )
                location = response.headers.get("Location", "")
                response.close()
                response.raise_for_status()
        except RequestException as e:
            raise _douyin_fetch_error(error=e, url=url) from e

        if not location:
            return ""
        return urljoin(base=url, url=location)

    def _fetch_share_payload(self, aweme_id: str) -> _DouyinVideoInfo:
        """Returns the cached or fetched `videoInfoRes` for `aweme_id`."""
        with _PAYLOAD_CACHE_LOCK:
            cached = _PAYLOAD_CACHE.get(aweme_id)
            if cached:
                if time.monotonic() - cached[0] < _PAYLOAD_CACHE_TTL_SECONDS:
                    _PAYLOAD_CACHE.move_to_end(aweme_id)
                    return cached[1]
                _PAYLOAD_CACHE.pop(aweme_id, None)

        url = f"https://www.iesdouyin.com/share/note/{aweme_id}"
        html = self._get_share_page(url=url)

        match = _ROUTER_DATA_RE.search(string=html)
        if not match:
            if any(marker in html for marker in _CHALLENGE_MARKERS):
                raise DouyinBlockedError(
                    f"Douyin served a bot challenge for {aweme_id}; try again later"
                )
            raise DouyinError(f"Douyin share page for {aweme_id} had no readable data")

        try:
            router_data = json.loads(match.group(1))
        except json.JSONDecodeError as e:
            raise DouyinError(f"Douyin returned malformed data for {aweme_id}: {e}") from e

        raw_info = self._find_video_info(router_data=router_data)
        if raw_info is None:
            raise DouyinError(f"Douyin returned an unexpected structure for {aweme_id}")
        try:
            info = _DouyinVideoInfo.model_validate(raw_info)
        except ValidationError as e:
            # Raise `DouyinError`, not `DouyinUnavailableError`, so schema drift stays an error.
            raise DouyinError(f"Douyin returned an unreadable post {aweme_id}: {e}") from e

        with _PAYLOAD_CACHE_LOCK:
            _PAYLOAD_CACHE[aweme_id] = (time.monotonic(), info)
            _PAYLOAD_CACHE.move_to_end(aweme_id)
            if len(_PAYLOAD_CACHE) > _PAYLOAD_CACHE_MAX_ENTRIES:
                _PAYLOAD_CACHE.popitem(last=False)
        return info

    def _get_share_page(self, url: str) -> str:
        """Returns the share page HTML for `url`, retrying up to three times to acquire `ttwid`."""
        global _share_ttwid  # noqa: PLW0603 -- the process-wide guest cookie

        for _ in range(3):
            sent = _share_ttwid
            try:
                with requests.Session() as session:
                    response = session.get(
                        url,
                        headers=self._headers(),
                        cookies={"ttwid": sent} if sent else None,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    html = response.text
                    issued = response.cookies.get(name="ttwid") or ""
            except RequestException as e:
                raise _douyin_fetch_error(error=e, url=url) from e

            if issued:
                _share_ttwid = issued
            is_shell = "_ROUTER_DATA" in html and "videoInfoRes" not in html
            if not is_shell or (sent and issued in ("", sent)):
                break
        # Only a shell answered to a cookie may mean the page moved; without one it is the gate.
        if is_shell and not sent:
            raise DouyinBlockedError(
                f"Douyin served its cookie-gate shell for {url}; try again later"
            )
        return html

    @staticmethod
    def _find_video_info(router_data: dict[str, Any]) -> dict[str, Any] | None:
        """Returns the `videoInfoRes` dict inside `loaderData`, or `None` when absent."""
        loader_data = router_data.get("loaderData")
        if not isinstance(loader_data, dict):
            return None

        # Route key varies by fetched path (`note_(id)/page` vs `video_(id)/page`).
        for page in loader_data.values():
            if isinstance(page, dict) and isinstance(page.get("videoInfoRes"), dict):
                return page["videoInfoRes"]
        return None

    @staticmethod
    def _first_item(info: _DouyinVideoInfo, aweme_id: str) -> _DouyinItem:
        """Returns `info.item_list[0]`, or raises `DouyinUnavailableError` when filtered."""
        if info.item_list:
            return info.item_list[0]

        # Filtered posts return HTTP 200 with empty `item_list` and a `filter_list`.
        if info.filter_list:
            entry = info.filter_list[0]
            reason = entry.detail_msg or entry.notice or entry.filter_reason
            raise DouyinUnavailableError(f"Douyin will not serve {aweme_id}: {reason}")
        raise DouyinUnavailableError(f"Douyin returned no post for {aweme_id}")

    def parse_metadata(self, url: str) -> DouyinMetadata:
        """Returns post metadata parsed from `url` without downloading any media.

        Args:
            url (str): The raw Douyin URL.

        Raises:
            DouyinError: If the URL cannot be resolved or the post cannot be read.
            RuntimeError: If a request failed in a way HTTP does not classify.
        """
        aweme_id = self._resolve_aweme_id(url=url)
        info = self._fetch_share_payload(aweme_id=aweme_id)
        item = self._first_item(info=info, aweme_id=aweme_id)

        # Photo posts also carry `video.play_addr` (slideshow), so check `aweme_type`/`images`.
        is_photo = item.aweme_type in _PHOTO_AWEME_TYPES or bool(item.images)

        return DouyinMetadata(
            aweme_id=aweme_id,
            title=item.desc.strip(),
            author_name=item.author.nickname,
            is_photo=is_photo,
            video_id="" if is_photo else self._video_id(item=item),
            image_urls=self._image_urls(images=item.images) if is_photo else [],
        )

    @staticmethod
    def _video_id(item: _DouyinItem) -> str:
        """Reads the internal video id used to build a play URL."""
        return item.video.play_addr.uri

    @staticmethod
    def _image_urls(images: list[_DouyinImage]) -> list[str]:
        """Returns one watermark-free JPEG URL per image in `images`."""
        # `url_list` is clean (`download_url_list` is watermarked) with JPEG last after WebP.
        return [image.url_list[-1] for image in images if image.url_list]

    def _play_url(self, video_id: str, quality: VideoQuality) -> str:
        """Returns the watermark-free `play` endpoint URL for `video_id` at `quality`."""
        # `play` serves the clean clip, unlike `play_addr.url_list`'s watermarked `playwm`.
        ratio = self.quality_ratios[quality]
        return f"https://aweme.snssdk.com/aweme/v1/play/?video_id={video_id}&ratio={ratio}&line=0"

    def _download_to(self, url: str, filename: str, max_bytes: int | None = None) -> Path:
        """Streams `url` to `output_folder/filename`, retrying stalls up to `max_retries`."""
        filepath = Path(self.output_folder) / filename

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return stream_to_file(
                    url=url,
                    filepath=filepath,
                    headers=self._headers(),
                    timeout=self.download_timeout,
                    max_bytes=max_bytes,
                )
            except DownloadTooLargeError as e:
                raise DouyinTooLargeError(str(e)) from e
            except RequestException as e:
                # Only fail fast on an HTTP status; mid-transfer drops have no response and retry.
                if e.response is not None and not is_retryable_fetch_failure(error=e):
                    raise RuntimeError(f"Failed to download Douyin media from {url}: {e}") from e
                last_error = e
                logfire.debug(
                    "Retrying a stalled Douyin media download",
                    url=url,
                    filename=filename,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error_type=type(e).__name__,
                    _exc_info=e,
                )

        # Keep exhausted retries retryable (`DouyinTransferError`) rather than `DouyinError`.
        message = f"Failed to download Douyin media from {url}: {last_error}"
        if isinstance(last_error, RequestException):
            raise DouyinTransferError(message) from last_error
        raise DouyinError(message) from last_error

    def download(
        self,
        url: str,
        quality: VideoQuality = "best",
        max_images: int | None = None,
        max_bytes: int | None = None,
        post: DouyinMetadata | None = None,
    ) -> DouyinDownload:
        """Downloads a Douyin post's media and returns its local file paths.

        Args:
            url (str): The raw Douyin URL.
            quality (VideoQuality): Requested quality preset; ignored for a photo post.
            max_images (int | None): Cap on images fetched from a photo post; `None` fetches all.
            max_bytes (int | None): Per-file size cap in bytes; `None` accepts any size.
            post (DouyinMetadata | None): Pre-parsed metadata for `url`, skipping a second parse.

        Returns:
            Downloaded file paths with `total_images` set to the post's full image count.

        Raises:
            DouyinError: If the post cannot be resolved, read, or downloaded.
            RuntimeError: If a request failed in a way HTTP does not classify.
        """
        resolved = post if post is not None else self.parse_metadata(url=url)
        if resolved.is_photo:
            return self._download_images(post=resolved, max_images=max_images, max_bytes=max_bytes)
        return self._download_video(post=resolved, quality=quality, max_bytes=max_bytes)

    def _download_video(
        self, post: DouyinMetadata, quality: VideoQuality, max_bytes: int | None = None
    ) -> DouyinDownload:
        """Downloads the watermark-free video for a post."""
        if not post.video_id:
            raise DouyinError(f"Douyin post {post.aweme_id} carries no playable video")

        filepath = self._download_to(
            url=self._play_url(video_id=post.video_id, quality=quality),
            filename=f"{post.aweme_id}.mp4",
            max_bytes=max_bytes,
        )
        return DouyinDownload(is_photo=False, filenames=[filepath])

    def _download_images(
        self, post: DouyinMetadata, max_images: int | None, max_bytes: int | None = None
    ) -> DouyinDownload:
        """Downloads a photo post's images, honouring the caller's cap."""
        if not post.image_urls:
            raise DouyinError(f"Douyin post {post.aweme_id} carries no images")

        wanted = post.image_urls if max_images is None else post.image_urls[:max_images]
        filenames: list[Path] = []
        for index, url in enumerate(wanted):
            filenames.append(
                self._download_to(
                    url=url, filename=f"{post.aweme_id}_{index + 1}.jpg", max_bytes=max_bytes
                )
            )

        return DouyinDownload(
            is_photo=True, filenames=filenames, total_images=len(post.image_urls)
        )
