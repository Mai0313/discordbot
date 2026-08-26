"""yt-dlp wrapper utilities shared by `/download_video` and the Bilibili link builder.

`VideoDownloader` itself is synchronous. The metadata probe exists for the link builder,
which has to decide whether to fetch at all; `download_with_stop_signal` is the asyncio half,
for either caller having to abandon a download that outran its budget -- the reply's, or the
command's own deadline. It is here rather than at either call site because both need it and
neither may import from the other's directory.
"""

from typing import Any, ClassVar
import asyncio
from pathlib import Path
import threading
import contextlib
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
import logfire
from pydantic import Field, BaseModel
from requests import Session
from requests.exceptions import RequestException

from discordbot.utils.urls import normalized_host, host_matches_domain
from discordbot.typings.video import VideoQuality
from discordbot.typings.timeouts import (
    YTDLP_RETRIES,
    DOWNLOAD_STOP_JOIN_SECONDS,
    YTDLP_SOCKET_TIMEOUT_SECONDS,
    SHARE_RESOLVE_TIMEOUT_SECONDS,
)
from discordbot.utils.file_downloads import TemporaryDownload


class DownloadStoppedError(Exception):
    """Raised inside yt-dlp when the caller's stop signal is set mid-download."""


class DownloadResult(TemporaryDownload):
    """Represents a downloaded video file.

    Attributes:
        title: Video title reported by yt-dlp.
        filename: Local path of the downloaded file.
    """

    title: str = Field(..., description="Video title reported by yt-dlp.")
    filename: Path = Field(..., description="Local path of the downloaded file.")

    def unlink(self) -> None:
        """Deletes the downloaded file."""
        self.filename.unlink(missing_ok=True)


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

    # Static map of quality presets to yt-dlp format strings; prefers separate
    # video+audio with safe fallbacks to muxed or video-only streams.
    quality_formats: ClassVar[dict[VideoQuality, str]] = {
        "best": "bestvideo*+bestaudio/best/bestvideo*",
        "high": "bestvideo[height<=1080][fps<=60]+bestaudio/best[height<=1080][fps<=60]/best[height<=1080]",
        "medium": "bestvideo[height<=720][fps<=60]+bestaudio/best[height<=720][fps<=60]/best[height<=720]",
        "low": "bestvideo[height<=480]+bestaudio/best[height<=480]/best[height<=480]",
    }

    def _default_http_headers(self) -> dict[str, str]:
        """Returns default HTTP headers for requests."""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7",
        }

    def _resolve_facebook_share_url(self, url: str) -> str:
        """Follows redirects for facebook.com/share/... links to obtain the real target.

        Only where the request landed is wanted, never the page itself, so the GET fallback
        streams: requests still follows the whole redirect chain and reports `response.url`,
        but the final page's body is left on the wire instead of being downloaded and thrown
        away. The HEAD attempt above it never had a body to begin with.
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
        """Converts Facebook watch URL to reel URL format.

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

        # Check if it's a Facebook watch URL
        if parsed.path == "/watch":
            query_params = parse_qs(parsed.query)
            video_id = query_params.get("v", [None])[0]

            if video_id:
                return f"https://www.facebook.com/reel/{video_id}"

        return url

    def get_params(
        self, quality: VideoQuality, dry_run: bool, url: str | None = None
    ) -> dict[str, Any]:
        """Returns the yt-dlp configuration parameters.

        Args:
            quality: The requested quality preset.
            dry_run: If True, enables simulation mode.
            url: Optional URL to determine site-specific headers (e.g., bilibili).

        Returns:
            A dictionary of yt-dlp parameters.
        """
        output_path = Path(self.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Base headers safe for most sites; site-specific headers added conditionally below.
        # Match the real host (not a raw substring) so a URL like `evil.com/?x=bilibili.com`
        # or `bilibili.com.attacker.com` never gets the bilibili Referer.
        http_headers = self._default_http_headers()
        host = normalized_host(url=url) if url else ""
        if host_matches_domain(host=host, domain="bilibili.com"):
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
            "retries": YTDLP_RETRIES,
            "fragment_retries": YTDLP_RETRIES,
            # Ensure merged output is mp4 when possible (common for Discord uploads)
            "merge_output_format": "mp4",
            "http_headers": http_headers,
            "socket_timeout": YTDLP_SOCKET_TIMEOUT_SECONDS,
            "extractor_retries": YTDLP_RETRIES,
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
        """Downloads a video from the given URL.

        Args:
            url: The URL of the video to download.
            quality: The requested quality preset.
            dry_run: If True, simulates the download.
            stop_signal: Optional event a caller sets to abort the download. This method
                blocks its thread, so asyncio cancellation cannot stop it; the signal is
                checked at every yt-dlp progress tick and aborts with DownloadStoppedError.

        Returns:
            A DownloadResult instance containing the title and filename.

        Raises:
            RuntimeError: When yt-dlp returns no metadata for the URL.
        """
        # Convert Facebook watch URLs to reel format
        url = self._convert_facebook_url(url)

        params = self.get_params(quality=quality, dry_run=dry_run, url=url)
        if stop_signal is not None:

            def _abort_if_stopped(_progress: dict[str, Any]) -> None:
                if stop_signal.is_set():
                    raise DownloadStoppedError(f"download stopped for {url}")

            params["progress_hooks"] = [_abort_if_stopped]
        with YoutubeDL(params=params) as ydl:
            info = ydl.extract_info(url=url, download=True)
            if info is None:
                msg = f"yt-dlp returned no metadata for {url}"
                raise RuntimeError(msg)
            title = info.get("title", "")
            filename = Path(ydl.prepare_filename(info))
            return DownloadResult(title=title, filename=filename)

    def parse_metadata(self, url: str) -> VideoMetadata:
        """Reads a video's metadata via yt-dlp without downloading any media.

        Deliberately not the `dry_run=True` preset: that branch flips `quiet` off and
        `dump_json` on, a CLI probe shape that prints the whole info dict to stdout.
        `extract_info(download=False)` under `simulate` fetches the same metadata silently.

        Args:
            url: The URL of the video to inspect.

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


def _retrieve_quietly(task: "asyncio.Task[DownloadResult]") -> None:
    """Retrieves an abandoned task's outcome so asyncio never logs it as never-retrieved."""
    if not task.cancelled():
        task.exception()


async def download_with_stop_signal(
    *, downloader: VideoDownloader, url: str, quality: VideoQuality
) -> DownloadResult:
    """Runs the blocking download with a stop signal cancellation can actually deliver.

    `asyncio.to_thread` cannot cancel its worker, so an abandoned download (a post-route
    discard, a media timeout, a command's own deadline) would otherwise leave yt-dlp
    downloading for minutes — holding a shared thread-pool slot, writing a file nothing will
    clean up, and even re-creating the scratch dir after its removal (yt-dlp re-makes the
    output dir before each DASH format). On any interruption the signal makes the worker abort
    at its next progress tick, and the bounded join keeps the scratch dir alive until the
    worker has really stopped, so its removal never races a live writer.

    Args:
        downloader: The downloader to run, already pointed at its scratch directory.
        url: The URL to download.
        quality: The requested quality preset.

    Returns:
        The finished download.
    """
    stop_signal = threading.Event()
    download_task = asyncio.create_task(
        coro=asyncio.to_thread(
            downloader.download, url=url, quality=quality, stop_signal=stop_signal
        )
    )
    download_task.add_done_callback(_retrieve_quietly)
    try:
        return await asyncio.shield(download_task)
    except BaseException:
        stop_signal.set()
        done, _pending = await asyncio.wait({download_task}, timeout=DOWNLOAD_STOP_JOIN_SECONDS)
        if done:
            with contextlib.suppress(BaseException):
                download_task.result()
        else:
            logfire.warn(
                "Download worker ignored the stop signal within the join window",
                url=url,
                join_seconds=DOWNLOAD_STOP_JOIN_SECONDS,
            )
        raise
