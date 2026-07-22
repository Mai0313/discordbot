"""yt-dlp wrapper utilities used by the video download command."""

import types
from typing import Any, ClassVar
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
from pydantic import Field, BaseModel
from requests import Session
from requests.exceptions import RequestException

from discordbot.typings.video import VideoQuality

# Redirect chases for a facebook.com/share/... link. Fixed rather than configurable: it bounds
# one HEAD/GET against Facebook and nothing has ever needed a different value.
SHARE_RESOLVE_TIMEOUT_SECONDS = 10


class DownloadResult(BaseModel):
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
        """Exits the context manager and deletes the downloaded file.

        Args:
            exc_type: Exception type raised inside the context, if any.
            exc_val: Exception value raised inside the context, if any.
            exc_tb: Traceback raised inside the context, if any.
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
        """Follows redirects for facebook.com/share/... links to obtain the real target."""
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
        # A user-pasted URL may be scheme-less (e.g. `www.bilibili.com/video/...`), which urlparse
        # reads as a path with no hostname; prepend `//` so the host is parsed either way. The
        # exact-host match still rejects `evil.com/?x=bilibili.com` and `bilibili.com.attacker.com`.
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
        self, url: str, quality: VideoQuality = "best", dry_run: bool = False
    ) -> DownloadResult:
        """Downloads a video from the given URL.

        Args:
            url: The URL of the video to download.
            quality: The requested quality preset.
            dry_run: If True, simulates the download.

        Returns:
            A DownloadResult instance containing the title and filename.
        """
        # Convert Facebook watch URLs to reel format
        url = self._convert_facebook_url(url)

        params = self.get_params(quality=quality, dry_run=dry_run, url=url)
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
            url: The URL of the video to inspect.

        Returns:
            The parsed metadata; absent string fields fall back to empty, duration to 0.0.

        Raises:
            RuntimeError: When yt-dlp returns no metadata for the URL.
        """
        url = self._convert_facebook_url(url)
        params = self.get_params(quality="best", dry_run=False, url=url)
        params.update({"simulate": True, "skip_download": True})
        with YoutubeDL(params=params) as ydl:
            info = ydl.extract_info(url=url, download=False)
        if info is None:
            msg = f"yt-dlp returned no metadata for {url}"
            raise RuntimeError(msg)
        # `noplaylist` keeps a download to one item, but a multi-part page (e.g. a Bilibili
        # anthology) can still report itself playlist-shaped; the first entry is the part the
        # pasted URL shows.
        entries = info.get("entries")
        if entries:
            info = next((entry for entry in entries if entry), info)
        return VideoMetadata(
            video_id=str(info.get("id") or ""),
            title=str(info.get("title") or ""),
            uploader=str(info.get("uploader") or ""),
            description=str(info.get("description") or ""),
            duration_seconds=float(info.get("duration") or 0.0),
            webpage_url=str(info.get("webpage_url") or ""),
            is_live=bool(info.get("is_live") or False),
        )


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    downloader = VideoDownloader(output_folder="./data")
    url = "https://www.bilibili.com/video/BV1jpK86hEc8"
    result = downloader.download(url=url, quality="low")
    console.print(f"Downloaded: {result.title} to {result.filename}")
