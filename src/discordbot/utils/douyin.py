"""Douyin URL parsing, share-page scraping, and media download helpers.

Douyin is deliberately NOT handled by `utils.downloader`'s yt-dlp path. yt-dlp's `DouyinIE`
fetches `www.douyin.com/aweme/v1/web/aweme/detail/` unsigned, which Douyin answers with an
empty body, so it fails outright unless the caller supplies cookies. It also only ever yields
a video (never a photo post) and tops out at 720p on the samples tested.

This module instead reads the server-rendered share page, which needs no cookie and no request
signature, exposes the source-resolution video, and carries photo posts in the same payload.
The non-obvious constraints are documented on `DouyinDownloader`.
"""

import re
import json
import time
import types
from typing import ClassVar
from pathlib import Path
from functools import cached_property
from collections import OrderedDict
from urllib.parse import urljoin, parse_qs, urlparse

from pydantic import Field, BaseModel
import requests
from requests.exceptions import RequestException

# Single source of truth for detecting a Douyin URL, kept module level so the planned
# auto-expand cog can share it the way `THREADS_URL_RE` is shared by parse_threads and
# gen_reply. Douyin's own share button emits the link inside a blob of noise
# ("7.64 gOX:/ w@f.oD ... https://v.douyin.com/iR2syBRn/ 复制此链接，打开Dou音搜索"), so the
# match has to survive being surrounded by CJK text: the path is matched as ASCII URL
# characters only and must END on `[A-Za-z0-9_/-]`, which stops the match at a trailing
# `，`/`。`/`!` instead of swallowing it into the short code and breaking the lookup.
# `(?:[A-Za-z0-9-]+\.)*` requires the `.com` to be immediately followed by `/`, so a
# lookalike host such as `douyin.com.attacker.com/x` does not match.
DOUYIN_URL_RE = re.compile(
    r"https?://(?:[A-Za-z0-9-]+\.)*(?:douyin|iesdouyin)\.com/[A-Za-z0-9_.?=&%/-]*[A-Za-z0-9_/-]"
)

# `_ROUTER_DATA` is assigned in a plain inline <script>. The JSON is matched non-greedily up to
# the closing tag rather than to the first `}` so a nested object cannot truncate it.
_ROUTER_DATA_RE = re.compile(r"_ROUTER_DATA\s*=\s*(\{.*?\});?\s*</script>", re.DOTALL)

# The post id in a URL path. The other shape is a `modal_id` query parameter, which MUST be read
# before this one: `douyin.com/user/<sec_uid>?modal_id=<id>` carries both, and taking the path
# first would yield the profile's sec_uid instead of the post.
_PATH_ID_RE = re.compile(r"/(?:video|note|slides)/(\d+)")

# Hosts whose links can be resolved at all. Anything else (notably `ixigua.com`, which some
# Douyin short links redirect to) is rejected rather than guessed at.
_ALLOWED_HOSTS = frozenset({"douyin.com", "iesdouyin.com"})

# Markers of ByteDance's two bot walls. `waf-jschallenge` / `out-sha256.js` is the per-path WAF
# challenge served by iesdouyin; `byted_acrawler` is the JS shell every `www.douyin.com` page
# returns. Both mean "come back later", never "this post does not exist" - conflating the two
# would tell a user their perfectly good link is dead.
_CHALLENGE_MARKERS = ("waf-jschallenge", "out-sha256.js", "byted_acrawler", "captcha")

# Douyin tags a photo post as 2 and a video as 4 on this endpoint; the app-side endpoint uses 68
# for a photo, accepted here so a payload from either shape branches correctly.
_PHOTO_AWEME_TYPES = frozenset({2, 68})


def is_douyin_url(url: str) -> bool:
    """Reports whether a URL points at Douyin.

    Matches whole host labels, so a lookalike such as `douyin.com.attacker.com` is rejected.
    A scheme-less paste (`v.douyin.com/xxx`) is accepted, since urlparse would otherwise read
    the whole string as a path with no hostname.

    Args:
        url: The URL to test.

    Returns:
        True when the URL's host is douyin.com or iesdouyin.com, or a subdomain of either.
    """
    normalized = url if "://" in url else f"//{url}"
    try:
        host = (urlparse(normalized).hostname or "").lower()
    except ValueError:
        # urlparse raises on an unbalanced bracket ("https://[abc/x"), and this runs on raw user
        # input in the command's routing check, before any error handling wraps it.
        return False
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in _ALLOWED_HOSTS)


class DouyinError(RuntimeError):
    """Base error for every Douyin lookup failure."""


class DouyinUnavailableError(DouyinError):
    """The post exists as an id but Douyin will not serve it (deleted, private, region locked)."""


class DouyinBlockedError(DouyinError):
    """A bot wall answered instead of the post. Retryable: the post itself is fine."""


class DouyinPost(BaseModel):
    """Metadata for a single Douyin post, parsed without downloading anything.

    Attributes:
        aweme_id: Douyin's numeric post id.
        title: Post caption, used as the display title.
        author_name: Author's display nickname.
        is_photo: True for a photo post, False for a video.
        video_id: Douyin's internal video id, empty for a photo post.
        image_urls: Watermark-free image URLs, empty for a video.
    """

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
    """Files downloaded for one Douyin post.

    Attributes:
        title: Post caption, used as the display title.
        is_photo: True when the files are images rather than a single video.
        filenames: Local paths of the downloaded files.
        total_images: Image count in the source post, before any cap was applied.
    """

    title: str = Field(default="", description="Post caption, used as the display title.")
    is_photo: bool = Field(
        default=False, description="True when the files are images rather than a single video."
    )
    filenames: list[Path] = Field(
        default_factory=list, description="Local paths of the downloaded files."
    )
    total_images: int = Field(
        default=0, description="Image count in the source post, before any cap was applied."
    )

    @cached_property
    def total_bytes(self) -> int:
        """Combined size of the downloaded files.

        Cached on first access so a caller can still read it after delivery has moved a hosted
        file out of the download folder; stat-ing later would raise on the very oversize path
        that most needs the number.
        """
        return sum(path.stat().st_size for path in self.filenames if path.exists())

    @property
    def omitted_images(self) -> int:
        """Number of images present in the post but not downloaded."""
        return max(0, self.total_images - len(self.filenames))

    def unlink(self) -> None:
        """Deletes every downloaded file."""
        for path in self.filenames:
            path.unlink(missing_ok=True)

    def __enter__(self):
        """Enters the context manager.

        Returns:
            This download result.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ):
        """Exits the context manager and deletes the downloaded files.

        Args:
            exc_type: Exception type raised inside the context, if any.
            exc_val: Exception value raised inside the context, if any.
            exc_tb: Traceback raised inside the context, if any.
        """
        self.unlink()


# Parsed share payloads, keyed by aweme id, so a link posted several times in a row costs one
# fetch. The TTL is deliberately far shorter than the CDN signature lifetime baked into the
# image URLs (`x-expires`): serving a cached payload past that point would hand out URLs that
# 403 on download, which is worse than re-fetching. Bounded like the other long-lived caches in
# this project (see `_gen_reply/attachment/base.py`) so a long-running bot cannot accumulate one
# full payload per link it has ever seen.
_PAYLOAD_CACHE: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_PAYLOAD_CACHE_TTL_SECONDS = 300.0
_PAYLOAD_CACHE_MAX_ENTRIES = 128


class DouyinDownloader(BaseModel):
    """Downloads Douyin videos and photo posts via the server-rendered share page.

    Four constraints drive this implementation, each verified against the live site:

    1. Only `iesdouyin.com/share/...` is readable. Every `www.douyin.com` page (including
       `/video/`, `/note/` and `modal_id` links) returns a `byted_acrawler` JS shell that needs
       a JS engine before it serves anything, so normalising the URL is mandatory, not a
       convenience.
    2. `share/note/<id>` serves videos as well as photo posts, so it is the single normalisation
       target and there is no per-type branch in the fetch.
    3. A desktop User-Agent gets the bot wall; a mobile one gets the payload.
    4. The WAF bans per path for tens of minutes once a path is hit hard, which is why short
       links are resolved by reading `Location` only (never following the redirect into
       `share/video/`, which would spend quota on a path this class never reads) and why
       payloads are cached.

    Attributes:
        output_folder: Directory where downloaded files are written.
        timeout: Timeout in seconds for a metadata request.
        download_timeout: Per-read timeout in seconds for a media download.
        max_retries: Attempts made per media download before giving up.
        max_redirects: Maximum redirect hops followed when resolving a short link.
    """

    output_folder: str = Field(..., description="Directory where downloaded files are written.")
    timeout: int = Field(
        default=15, description="Timeout in seconds for a metadata request.", examples=[15, 30]
    )
    # Separate from `timeout` because this one bounds the gap between chunks of a video that can
    # run to tens of megabytes; the metadata timeout is far too tight for that and was observed
    # aborting an otherwise healthy transfer.
    download_timeout: int = Field(
        default=60, description="Per-read timeout in seconds for a media download.", examples=[60]
    )
    max_retries: int = Field(
        default=3, description="Attempts made per media download before giving up.", examples=[3]
    )
    max_redirects: int = Field(
        default=5, description="Maximum redirect hops followed when resolving a short link."
    )

    # Douyin serves the bot wall to a desktop UA and the real payload to a mobile one.
    mobile_user_agent: ClassVar[str] = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
    )

    # The command's quality presets mapped onto the `ratio` the play endpoint accepts. `best` and
    # `high` share 1080p because 1080p is the source resolution, not an upscale.
    quality_ratios: ClassVar[dict[str, str]] = {
        "best": "1080p",
        "high": "1080p",
        "medium": "720p",
        "low": "540p",
    }

    def _headers(self) -> dict[str, str]:
        """Returns the request headers Douyin's share page expects."""
        return {"User-Agent": self.mobile_user_agent, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}

    def _resolve_aweme_id(self, url: str) -> str:
        """Extracts the numeric post id from any accepted Douyin URL form.

        A short link (`v.douyin.com/...`, `jx.douyin.com/...`) carries no id, so it is resolved
        by following redirects with the body left unread. Only `share/(video|note|slides)/<id>`
        style targets are accepted: a short link can just as easily point at a profile
        (`share/user/...`), a sound (`share/music/...`) or another ByteDance site entirely, and
        blindly taking the last path segment would turn those into a bogus lookup.

        Args:
            url: The raw Douyin URL.

        Returns:
            The numeric aweme id.

        Raises:
            DouyinError: If the URL is not a Douyin post link or cannot be resolved.
        """
        # `is_douyin_url` accepts a scheme-less paste, so one can reach here; requests cannot fetch
        # it, and this module has already claimed the URL from the yt-dlp path, so there is nothing
        # to fall back to. Give it a scheme once, up front.
        current = url if "://" in url else f"https://{url}"
        for _ in range(self.max_redirects + 1):
            # Re-checked every hop, not just on the way in: a short link can redirect off Douyin
            # entirely (ixigua has been observed), and following that would fetch an unrelated host.
            if not is_douyin_url(url=current):
                raise DouyinError(f"Not a Douyin post link: {url}")

            aweme_id = self._extract_id(url=current)
            if aweme_id:
                return aweme_id

            location = self._redirect_target(url=current)
            if not location:
                break
            current = location

        raise DouyinError(f"Could not find a Douyin post id in: {url}")

    @staticmethod
    def _extract_id(url: str) -> str:
        """Reads the aweme id straight out of a URL, or returns an empty string."""
        parsed = urlparse(url if "://" in url else f"//{url}")
        # `modal_id` wins over the path: a `/user/<sec_uid>?modal_id=<id>` link carries both and
        # the path would give the profile id.
        modal_ids = parse_qs(parsed.query).get("modal_id")
        if modal_ids and modal_ids[0].isdigit():
            return modal_ids[0]

        match = _PATH_ID_RE.search(parsed.path)
        return match.group(1) if match else ""

    def _redirect_target(self, url: str) -> str:
        """Returns the Location of a single redirect hop, without fetching the body.

        Args:
            url: The URL to probe.

        Returns:
            The absolute redirect target, or an empty string when the URL does not redirect.
        """
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
        except RequestException as e:
            raise DouyinError(f"Failed to resolve Douyin link {url}: {e}") from e

        if not location:
            return ""
        # Douyin sends an absolute Location today, but a relative one is legal, so resolve it
        # against the request URL rather than trusting the header verbatim.
        return urljoin(url, location)

    def _fetch_share_payload(self, aweme_id: str) -> dict:
        """Fetches and parses the share page for a post id.

        Args:
            aweme_id: The numeric post id.

        Returns:
            The `videoInfoRes` object from the page's `_ROUTER_DATA`.

        Raises:
            DouyinBlockedError: If a bot wall answered instead of the post.
            DouyinError: If the page could not be fetched or its structure changed.
        """
        cached = _PAYLOAD_CACHE.get(aweme_id)
        if cached:
            if time.monotonic() - cached[0] < _PAYLOAD_CACHE_TTL_SECONDS:
                _PAYLOAD_CACHE.move_to_end(aweme_id)
                return cached[1]
            _PAYLOAD_CACHE.pop(aweme_id, None)

        # `share/note/` rather than `share/video/`: it serves both post types, so one path is
        # enough, and it keeps every request off a second path that could be banned separately.
        url = f"https://www.iesdouyin.com/share/note/{aweme_id}"
        try:
            with requests.Session() as session:
                response = session.get(url, headers=self._headers(), timeout=self.timeout)
                response.raise_for_status()
                html = response.text
        except RequestException as e:
            raise DouyinError(f"Failed to fetch Douyin post {aweme_id}: {e}") from e

        match = _ROUTER_DATA_RE.search(html)
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

        info = self._find_video_info(router_data=router_data)
        if info is None:
            raise DouyinError(f"Douyin returned an unexpected structure for {aweme_id}")

        _PAYLOAD_CACHE[aweme_id] = (time.monotonic(), info)
        _PAYLOAD_CACHE.move_to_end(aweme_id)
        if len(_PAYLOAD_CACHE) > _PAYLOAD_CACHE_MAX_ENTRIES:
            _PAYLOAD_CACHE.popitem(last=False)
        return info

    @staticmethod
    def _find_video_info(router_data: dict) -> dict | None:
        """Finds the `videoInfoRes` object inside `loaderData`.

        The `loaderData` key is named after the URL path that was fetched (`note_(id)/page` for a
        note URL, `video_(id)/page` for a video one), so hardcoding either name breaks the moment
        the other form is used. Scanning the values for the one carrying `videoInfoRes` is stable
        across both.

        Args:
            router_data: The parsed `_ROUTER_DATA` object.

        Returns:
            The `videoInfoRes` object, or None when the payload has no such entry.
        """
        loader_data = router_data.get("loaderData")
        if not isinstance(loader_data, dict):
            return None

        for page in loader_data.values():
            if isinstance(page, dict) and isinstance(page.get("videoInfoRes"), dict):
                return page["videoInfoRes"]
        return None

    @staticmethod
    def _first_item(info: dict, aweme_id: str) -> dict:
        """Returns the post object, translating Douyin's soft failures into errors.

        Douyin answers a deleted, private or region-locked post with HTTP 200, an empty
        `item_list` and a populated `filter_list`, so treating a 200 as success would surface an
        empty post instead of a real error.

        Args:
            info: The `videoInfoRes` object.
            aweme_id: The numeric post id, used in the error message.

        Returns:
            The first entry of `item_list`.

        Raises:
            DouyinUnavailableError: If Douyin filtered the post out.
        """
        item_list = info.get("item_list") or []
        if item_list:
            return item_list[0]

        filter_list = info.get("filter_list") or []
        if filter_list:
            entry = filter_list[0]
            reason = entry.get("detail_msg") or entry.get("notice") or entry.get("filter_reason")
            raise DouyinUnavailableError(f"Douyin will not serve {aweme_id}: {reason}")
        raise DouyinUnavailableError(f"Douyin returned no post for {aweme_id}")

    def parse_metadata(self, url: str) -> DouyinPost:
        """Parses a Douyin URL into post metadata WITHOUT downloading any media.

        The planned auto-expand path needs the caption and media URLs but no local files, so the
        parse is kept separate from the download the same way `ThreadsDownloader.parse_metadata`
        is split from `parse`.

        Args:
            url: The raw Douyin URL.

        Returns:
            The parsed post metadata.

        Raises:
            DouyinError: If the URL cannot be resolved or the post cannot be read.
        """
        aweme_id = self._resolve_aweme_id(url=url)
        info = self._fetch_share_payload(aweme_id=aweme_id)
        item = self._first_item(info=info, aweme_id=aweme_id)

        images = item.get("images") or []
        # Branch on `aweme_type`, never on the presence of `play_addr`: a photo post also carries
        # a non-empty `video.play_addr` holding a server-rendered slideshow, so a play_addr check
        # would classify every gallery as a video.
        is_photo = item.get("aweme_type") in _PHOTO_AWEME_TYPES or bool(images)

        return DouyinPost(
            aweme_id=aweme_id,
            title=(item.get("desc") or "").strip(),
            author_name=(item.get("author") or {}).get("nickname") or "",
            is_photo=is_photo,
            video_id="" if is_photo else self._video_id(item=item),
            image_urls=self._image_urls(images=images) if is_photo else [],
        )

    @staticmethod
    def _video_id(item: dict) -> str:
        """Reads the internal video id used to build a play URL."""
        return ((item.get("video") or {}).get("play_addr") or {}).get("uri") or ""

    @staticmethod
    def _image_urls(images: list) -> list[str]:
        """Picks one watermark-free URL per image.

        Despite the names, `url_list` holds the clean images and `download_url_list` holds the
        watermarked ones. Within `url_list` the last entry is the JPEG and the earlier ones are
        WebP, which Discord renders less consistently.

        Args:
            images: The post's `images` array.

        Returns:
            One URL per image, skipping malformed entries.
        """
        urls: list[str] = []
        for image in images:
            if not isinstance(image, dict):
                continue
            url_list = [url for url in (image.get("url_list") or []) if isinstance(url, str)]
            if url_list:
                urls.append(url_list[-1])
        return urls

    def _play_url(self, video_id: str, quality: str) -> str:
        """Builds the watermark-free play URL for a video.

        The URL Douyin ships in `play_addr.url_list` points at the `playwm` endpoint, whose
        output carries a corner logo and the author's handle, and runs a few seconds longer
        because of the appended outro. The `play` endpoint serves the same clip clean.

        Args:
            video_id: Douyin's internal video id.
            quality: One of the command's quality presets.

        Returns:
            The play endpoint URL.
        """
        ratio = self.quality_ratios.get(quality, "1080p")
        return f"https://aweme.snssdk.com/aweme/v1/play/?video_id={video_id}&ratio={ratio}&line=0"

    def _download_to(self, url: str, filename: str) -> Path:
        """Streams a remote file into the output folder, retrying a stalled transfer.

        The media CDN intermittently stalls mid-transfer, which surfaces as a read timeout
        rather than an error status, so a failed attempt is retried from scratch. A partial file
        is removed between attempts: leaving it would let a later `stat()` report a truncated
        download as a successful one.

        Args:
            url: The media URL.
            filename: The name to save the file as.

        Returns:
            The path of the written file.

        Raises:
            DouyinError: If every attempt fails.
        """
        output_path = Path(self.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename

        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                with requests.Session() as session:
                    response = session.get(
                        url, headers=self._headers(), timeout=self.download_timeout, stream=True
                    )
                    response.raise_for_status()
                    with filepath.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=1 << 16):
                            if chunk:
                                f.write(chunk)
                return filepath
            except RequestException as e:
                last_error = e
                filepath.unlink(missing_ok=True)
            except Exception:
                # A local write can fail too (a full disk surfaces from `write`, not from the
                # request), and that is not worth retrying. Clean up first: the caller's gallery
                # cleanup only knows about files it already accepted, so a partial file left here
                # would survive and take disk space with it.
                filepath.unlink(missing_ok=True)
                raise

        raise DouyinError(f"Failed to download Douyin media from {url}: {last_error}")

    def download(
        self, url: str, quality: str = "best", max_images: int | None = None
    ) -> DouyinDownload:
        """Downloads a Douyin post's media.

        Args:
            url: The raw Douyin URL.
            quality: The requested quality preset. Ignored for a photo post.
            max_images: Cap on images fetched from a photo post. None fetches all of them.

        Returns:
            The downloaded files, with `total_images` recording the post's real image count so
            the caller can report what a cap left out.

        Raises:
            DouyinError: If the post cannot be resolved, read, or downloaded.
        """
        post = self.parse_metadata(url=url)
        if post.is_photo:
            return self._download_images(post=post, max_images=max_images)
        return self._download_video(post=post, quality=quality)

    def _download_video(self, post: DouyinPost, quality: str) -> DouyinDownload:
        """Downloads the watermark-free video for a post."""
        if not post.video_id:
            raise DouyinError(f"Douyin post {post.aweme_id} carries no playable video")

        filepath = self._download_to(
            url=self._play_url(video_id=post.video_id, quality=quality),
            filename=f"{post.aweme_id}.mp4",
        )
        return DouyinDownload(title=post.title, is_photo=False, filenames=[filepath])

    def _download_images(self, post: DouyinPost, max_images: int | None) -> DouyinDownload:
        """Downloads a photo post's images, honouring the caller's cap."""
        if not post.image_urls:
            raise DouyinError(f"Douyin post {post.aweme_id} carries no images")

        wanted = post.image_urls if max_images is None else post.image_urls[:max_images]
        filenames: list[Path] = []
        try:
            for index, url in enumerate(wanted):
                filenames.append(
                    self._download_to(url=url, filename=f"{post.aweme_id}_{index + 1}.jpg")
                )
        except Exception:
            # Nothing is returned on failure, so the caller never gets a handle to clean up with:
            # a gallery that dies on image 3 would strand images 1 and 2 in the temp dir for good.
            # `_download_to` only removes its own partial file.
            for path in filenames:
                path.unlink(missing_ok=True)
            raise

        return DouyinDownload(
            title=post.title, is_photo=True, filenames=filenames, total_images=len(post.image_urls)
        )


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    downloader = DouyinDownloader(output_folder="./tmp")
    url = "https://v.douyin.com/NdlfIZPcgz4"
    console.print(downloader.parse_metadata(url=url))
    with downloader.download(url=url) as result:
        console.print(result)
