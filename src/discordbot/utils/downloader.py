"""The yt-dlp facade behind `/download_video` and `gen_reply`'s Bilibili link builder.

`VideoDownloader` wraps one `YoutubeDL` run in this project's defaults: a `VideoQuality` preset
becomes a yt-dlp format string (`quality_formats`), the file is named after the site's own id
inside the caller's `output_folder`, and separate video+audio streams are merged into mp4 where
the site allows it. Two entry points ride those defaults — `download` fetches the media and hands
back a `DownloadResult` that deletes the file when its context closes, and `parse_metadata` reads
what the page reports (title, uploader, duration, canonical URL, whether it is a live stream, and
whether the fields came from unwrapping a playlist-shaped page) without fetching any media.

What it deliberately does not do:

- Douyin. `cogs/video/cog.py` routes those links to `utils/douyin.py` before reaching here,
  because yt-dlp's extractor there needs cookies, never yields a photo post, and caps below the
  source resolution (that module's docstring has the detail).
- Choose where files land. `output_folder` is required with no default, so the directory is the
  caller's decision rather than a hidden default's; nothing here constrains which one. Keeping
  downloads out of `data/` is a caller convention: both production sites pass
  `tempfile.gettempdir()` or a `TemporaryDirectory`, while the `__main__` demo at the bottom of
  this module writes to `./data`.
- Cap bytes or leave the event loop. Both entry points block, so callers run them under
  `asyncio.to_thread`, and `download`'s `stop_signal` exists because that worker cannot be
  cancelled. Size is the caller's policy: the `/download_video` cog hosts an oversize file as a
  URL, and the Bilibili builder enforces the Files API ceiling on the finished file.

It sits in `utils/` because two cogs that may not import each other both download through it: the
`/download_video` command, and `gen_reply`'s Bilibili link builder, which probes the metadata
first and only then downloads at `AI_INGEST_QUALITY`.

Two sites are shaped specially here. Facebook is the one whose URL is rewritten before yt-dlp
sees it: a `facebook.com/share/...` link is resolved by following its redirects, and a
`/watch?v=<id>` page is rewritten to the `/reel/<id>` form. Bilibili is the one that gets an extra
request header, a `Referer` that `get_params` attaches on an exact host match rather than a
substring one, so a lookalike host cannot collect it.
"""

import types
from typing import Any, ClassVar
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
from pydantic import Field, BaseModel
from requests import Session
from requests.exceptions import RequestException

from discordbot.typings.video import VideoQuality

# Redirect chases for a facebook.com/share/... link. Fixed rather than configurable: it bounds
# one HEAD/GET against Facebook and nothing has ever needed a different value.
SHARE_RESOLVE_TIMEOUT_SECONDS = 10


class DownloadStoppedError(Exception):
    """Raised inside yt-dlp when the caller's stop signal is set mid-download.

    Nothing in yt-dlp's own error handling catches it, so it escapes `download` to whoever set
    the signal.
    """


class DownloadResult(BaseModel):
    """A downloaded video file, usable as a context manager that deletes it on the way out.

    Attributes:
        title: Video title reported by yt-dlp.
        filename: Local path of the downloaded file.
    """

    title: str = Field(..., description="Video title reported by yt-dlp.")
    filename: Path = Field(..., description="Local path of the downloaded file.")

    def unlink(self) -> None:
        """Deletes the downloaded file, tolerating one that is already gone.

        A caller may have moved the file out from under this result — hosting an oversize
        download does exactly that — and the closing context still has to clean up either way.
        """
        self.filename.unlink(missing_ok=True)

    def __enter__(self):
        """Enters the context that deletes the file on the way out.

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
        """Deletes the downloaded file, whether or not the body raised.

        Nothing is suppressed: an exception from the body propagates once the file is gone.

        Args:
            exc_type (type[BaseException] | None): Exception type raised in the body, if any.
            exc_val (BaseException | None): Exception value raised in the body, if any.
            exc_tb (types.TracebackType | None): Traceback raised in the body, if any.
        """
        self.unlink()


class VideoMetadata(BaseModel):
    """Metadata for a video, read by yt-dlp without downloading any media.

    Attributes:
        video_id: Site-native video id (e.g. a Bilibili BV id).
        title: Video title reported by yt-dlp.
        uploader: Uploader / channel display name.
        description: Full video description; callers trim it to their own budget.
        duration_seconds: Duration in seconds; 0.0 when the site does not report one.
        webpage_url: Canonical page URL after redirects, so a short link resolves.
        is_live: Whether the URL points at a live stream rather than a finished video.
        from_playlist: Whether the fields describe the first entry of a playlist-shaped
            page (a space, a collection, a season) rather than the page itself.
    """

    video_id: str = Field(default="", description="Site-native video id (e.g. a Bilibili BV id).")
    title: str = Field(default="", description="Video title reported by yt-dlp.")
    uploader: str = Field(default="", description="Uploader / channel display name.")
    description: str = Field(
        default="", description="Full video description; callers trim it to their own budget."
    )
    duration_seconds: float = Field(
        default=0.0, description="Duration in seconds; 0.0 when the site does not report one."
    )
    webpage_url: str = Field(
        default="", description="Canonical page URL after redirects, so a short link resolves."
    )
    is_live: bool = Field(default=False, description="Whether the URL points at a live stream.")
    from_playlist: bool = Field(
        default=False,
        description="Whether the fields describe a playlist-shaped page's first entry.",
    )


class VideoDownloader(BaseModel):
    """Downloads videos with yt-dlp and local project defaults.

    Attributes:
        output_folder: Directory where downloaded files are written.
    """

    output_folder: str = Field(..., description="Download folder")

    # Every preset asks for separate video+audio first, so a site offering no split rendition
    # still downloads something. The fallbacks are muxed, not video-only: a bare `best` selector
    # requires both codecs. high/medium spend their last rung on that same muxed selector with
    # the fps cap dropped, low never sets one so its last two rungs match, and only "best" ends
    # on a video-only rung.
    quality_formats: ClassVar[dict[VideoQuality, str]] = {
        "best": "bestvideo*+bestaudio/best/bestvideo*",
        "high": "bestvideo[height<=1080][fps<=60]+bestaudio/best[height<=1080][fps<=60]/best[height<=1080]",
        "medium": "bestvideo[height<=720][fps<=60]+bestaudio/best[height<=720][fps<=60]/best[height<=720]",
        "low": "bestvideo[height<=480]+bestaudio/best[height<=480]/best[height<=480]",
    }

    def _default_http_headers(self) -> dict[str, str]:
        """Builds the browser-shaped headers every request here starts from.

        A fresh dict per call, since `get_params` adds a site-specific `Referer` to what it gets.

        Returns:
            Headers for one yt-dlp run or one share-link resolution.
        """
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7",
        }

    def _resolve_facebook_share_url(self, url: str) -> str:
        """Follows redirects for a `facebook.com/share/...` link to obtain the real target.

        Only where the request landed is wanted, never the page itself, so the GET fallback
        streams: requests still follows the whole redirect chain and reports `response.url`,
        but the final page's body is left on the wire instead of being downloaded and thrown
        away. The HEAD attempt above it never had a body to begin with. A transport failure is
        not an error here: both attempts failing leaves the share link to be tried as-is.

        Args:
            url (str): The share link to resolve.

        Returns:
            The URL the redirects landed on, or `url` unchanged when nothing resolved it.
        """
        headers = self._default_http_headers()
        with Session() as session:
            for method_name in ("head", "get"):
                request_method = getattr(session, method_name)
                try:
                    response = request_method(
                        url,
                        allow_redirects=True,
                        headers=headers,
                        timeout=SHARE_RESOLVE_TIMEOUT_SECONDS,
                        stream=True,
                    )
                except RequestException:
                    continue

                final_url = response.url
                response.close()
                if final_url and final_url != url:
                    return final_url

        return url

    def _convert_facebook_url(self, url: str) -> str:
        """Normalizes a Facebook URL into the form yt-dlp is handed.

        A `/share/...` link is resolved first and its target re-enters here, so a share link that
        lands on a watch page still ends up as a reel URL. What reaches either rewrite is decided
        by a substring test for `facebook.com` on the parsed netloc, which is neither a host match
        nor tolerant of a missing scheme: a lookalike host (`notfacebook.com`,
        `facebook.com.attacker.com`) is rewritten like the real thing, while a scheme-less paste
        (`www.facebook.com/watch?v=1`) parses with an empty netloc and is returned untouched.
        Everything else is returned untouched too.

        Args:
            url (str): The URL to normalize, rewritten only when its parsed netloc contains
                `facebook.com`.

        Returns:
            The rewritten URL, or `url` unchanged when nothing applied.

        Example:
            https://www.facebook.com/watch?v=828357636228730
            -> https://www.facebook.com/reel/828357636228730
        """
        parsed = urlparse(url)

        if "facebook.com" not in parsed.netloc:
            return url

        if parsed.path.startswith("/share/"):
            resolved_url = self._resolve_facebook_share_url(url)
            if resolved_url != url:
                return self._convert_facebook_url(resolved_url)
            return url

        if parsed.path == "/watch":
            query_params = parse_qs(parsed.query)
            video_id = query_params.get("v", [None])[0]

            if video_id:
                return f"https://www.facebook.com/reel/{video_id}"

        return url

    def get_params(
        self, quality: VideoQuality, dry_run: bool, url: str | None = None
    ) -> dict[str, Any]:
        """Builds the yt-dlp parameters for one run, creating `output_folder` on the way.

        `dry_run` is yt-dlp's CLI-probe shape rather than a quiet simulation: it turns `quiet`
        off and dumps the whole info dict to stdout, which is why `parse_metadata` sets
        `simulate` itself instead of asking for it here.

        Args:
            quality (VideoQuality): Preset selecting the format string.
            dry_run (bool): Whether to simulate instead of writing a file.
            url (str | None): The URL about to be fetched, read only to decide site-specific
                headers.

        Returns:
            The parameter dict to hand `YoutubeDL`.
        """
        output_path = Path(self.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        http_headers = self._default_http_headers()
        # A pasted URL may be scheme-less (`www.bilibili.com/video/...`), which urlparse reads as
        # a path with no hostname; prepend `//` so the host parses either way. Matching the whole
        # host rather than a substring is what keeps `evil.com/?x=bilibili.com` and
        # `bilibili.com.attacker.com` from being handed the bilibili Referer.
        host = ""
        if url:
            normalized = url if "://" in url else f"//{url}"
            host = (urlparse(normalized).hostname or "").lower()
        if host == "bilibili.com" or host.endswith(".bilibili.com"):
            http_headers["Referer"] = "https://www.bilibili.com"

        params = {
            "format": self.quality_formats[quality],
            "outtmpl": f"{output_path.as_posix()}/%(id)s.%(ext)s",
            "quiet": True,
            "no_warnings": False,
            "continuedl": True,
            "noplaylist": True,
            "restrictfilenames": True,
            "writeinfojson": False,
            "writedescription": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ignoreerrors": False,
            "retries": 3,
            "fragment_retries": 3,
            # Ensure merged output is mp4 when possible (common for Discord uploads)
            "merge_output_format": "mp4",
            "http_headers": http_headers,
            "socket_timeout": 30,
            "extractor_retries": 3,
            "geo_bypass": True,
        }
        if dry_run:
            params.update({
                "simulate": True,
                "skip_download": True,
                "quiet": False,
                "dump_json": True,
            })
        return params

    def download(
        self,
        url: str,
        quality: VideoQuality = "best",
        dry_run: bool = False,
        stop_signal: threading.Event | None = None,
    ) -> DownloadResult:
        """Downloads one video and hands back a handle that deletes the file when closed.

        Blocks its thread, so callers run it under `asyncio.to_thread` — which is exactly why
        `stop_signal` exists, since cancellation cannot reach a worker thread. The signal is read
        at every yt-dlp progress tick and aborts the run with `DownloadStoppedError`, so an
        abandoned download stops within a tick instead of running to completion against a scratch
        dir the caller is already removing.

        Args:
            url (str): The video URL, before any Facebook normalization.
            quality (VideoQuality): Preset selecting the format string.
            dry_run (bool): Whether to run yt-dlp in its simulating CLI-probe shape.
            stop_signal (threading.Event | None): Set by the caller to abort mid-download.

        Returns:
            The title yt-dlp reported plus the path it wrote, as a context manager over the file.
            Under `dry_run` nothing is written, so the path is the name yt-dlp would have used
            and does not exist.

        Raises:
            DownloadStoppedError: When the caller's stop signal is set at a progress tick.
        """  # noqa: DOC502 -- raised by the progress hook, which yt-dlp calls inside this frame
        url = self._convert_facebook_url(url)

        params = self.get_params(quality=quality, dry_run=dry_run, url=url)
        if stop_signal is not None:

            def _abort_if_stopped(_progress: dict[str, Any]) -> None:
                if stop_signal.is_set():
                    raise DownloadStoppedError(f"download stopped for {url}")

            params["progress_hooks"] = [_abort_if_stopped]
        with YoutubeDL(params=params) as ydl:
            info = ydl.extract_info(url=url, download=True)
            title = info.get("title", "")
            filename = Path(ydl.prepare_filename(info))
            return DownloadResult(title=title, filename=filename)

    def parse_metadata(self, url: str) -> VideoMetadata:
        """Reads a video's metadata via yt-dlp without downloading any media.

        Deliberately not the `dry_run=True` preset: that branch flips `quiet` off and
        `dump_json` on, a CLI probe shape that prints the whole info dict to stdout.
        `extract_info(download=False)` under `simulate` fetches the same metadata silently.

        Args:
            url (str): The URL of the video to inspect.

        Returns:
            The parsed metadata; absent string fields fall back to empty, duration to 0.0.

        Raises:
            RuntimeError: When yt-dlp returns no metadata for the URL.
        """
        url = self._convert_facebook_url(url)
        params = self.get_params(quality="best", dry_run=False, url=url)
        # `extract_flat` keeps a playlist-shaped page (a channel, a user space, a collection)
        # to ONE request instead of resolving every entry over the network in a probe that is
        # supposed to take seconds.
        params.update({"simulate": True, "skip_download": True, "extract_flat": "in_playlist"})
        with YoutubeDL(params=params) as ydl:
            info = ydl.extract_info(url=url, download=False)
        if info is None:
            msg = f"yt-dlp returned no metadata for {url}"
            raise RuntimeError(msg)
        # The page's own URL wins over the first entry's below, so a caller can tell a
        # playlist-shaped page apart from the single video it asked about.
        page_url = str(info.get("webpage_url") or "")
        # `noplaylist` keeps a download to one item, but a multi-part page (e.g. a Bilibili
        # anthology) can still report itself playlist-shaped; the first entry is the part the
        # pasted URL shows. `from_playlist` records the unwrap, since only then can these
        # fields describe some other video than the page the caller asked about.
        from_playlist = False
        entries = info.get("entries")
        if entries:
            from_playlist = True
            info = next((entry for entry in entries if entry), info)
        return VideoMetadata(
            video_id=str(info.get("id") or ""),
            title=str(info.get("title") or ""),
            uploader=str(info.get("uploader") or ""),
            description=str(info.get("description") or ""),
            duration_seconds=float(info.get("duration") or 0.0),
            webpage_url=page_url or str(info.get("webpage_url") or ""),
            is_live=bool(info.get("is_live") or False),
            from_playlist=from_playlist,
        )


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    downloader = VideoDownloader(output_folder="./data")
    url = "https://www.bilibili.com/video/BV1jpK86hEc8"
    result = downloader.download(url=url, quality="low")
    console.print(f"Downloaded: {result.title} to {result.filename}")
