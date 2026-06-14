"""yt-dlp wrapper utilities used by the video download command."""

import types
from typing import Any, ClassVar
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
from pydantic import Field, BaseModel
from requests import Session
from requests.exceptions import RequestException


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


class VideoDownloader(BaseModel):
    """Downloads videos with yt-dlp and local project defaults.

    Attributes:
        output_folder: Directory where downloaded files are written.
        max_retries: Configured maximum retry count.
        share_resolve_timeout: Timeout in seconds for resolving Facebook share URLs.
    """

    output_folder: str = Field(..., description="Download folder")
    max_retries: int = Field(
        default=5, description="Configured maximum retry count.", examples=[5, 3]
    )
    share_resolve_timeout: int = Field(
        default=10, description="Timeout (seconds) for resolving Facebook share URLs"
    )

    # Static map of quality presets to yt-dlp format strings; prefers separate
    # video+audio with safe fallbacks to muxed or video-only streams.
    quality_formats: ClassVar[dict[str, str]] = {
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
                        timeout=self.share_resolve_timeout,
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

    def get_params(self, quality: str, dry_run: bool, url: str | None = None) -> dict[str, Any]:
        """Returns the yt-dlp configuration parameters.

        Args:
            quality: The requested quality preset ('best', 'high', 'medium', 'low').
            dry_run: If True, enables simulation mode.
            url: Optional URL to determine site-specific headers (e.g., bilibili).

        Returns:
            A dictionary of yt-dlp parameters.
        """
        output_path = Path(self.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Base headers safe for most sites; site-specific headers added conditionally below
        http_headers = self._default_http_headers()
        if url and "bilibili.com" in url:
            http_headers["Referer"] = "https://www.bilibili.com"

        params = {
            "format": self.quality_formats.get(quality, "best"),
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

    def download(self, url: str, quality: str = "best", dry_run: bool = False) -> DownloadResult:
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
