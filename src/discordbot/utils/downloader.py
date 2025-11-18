from typing import Any
from pathlib import Path
import datetime
from functools import cached_property
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
import logfire
from pydantic import Field, BaseModel, computed_field
from requests import Session
from requests.exceptions import RequestException

logfire.configure(send_to_logfire=False, scrubbing=False)


class VideoDownloader(BaseModel):
    output_folder: str = Field(default="./data/downloads", description="Download folder")
    max_retries: int = Field(default=5)
    share_resolve_timeout: int = Field(
        default=10, description="Timeout (seconds) for resolving Facebook share URLs"
    )

    @computed_field
    @cached_property
    def quality_formats(self) -> dict[str, str]:
        quality_formats = {
            # Prefer separate video+audio with safe fallbacks to muxed or video-only streams
            "best": "bestvideo*+bestaudio/best/bestvideo*",
            "high": "bestvideo[height<=1080][fps<=60]+bestaudio/best[height<=1080][fps<=60]/best[height<=1080]",
            "medium": "bestvideo[height<=720][fps<=60]+bestaudio/best[height<=720][fps<=60]/best[height<=720]",
            "low": "bestvideo[height<=480]+bestaudio/best[height<=480]/best[height<=480]",
        }
        return quality_formats

    def _default_http_headers(self) -> dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7",
        }

    def _resolve_facebook_share_url(self, url: str) -> str:
        """Follow redirects for facebook.com/share/... links to obtain the real target."""
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
                except RequestException as exc:
                    logfire.warning(
                        "Failed to resolve Facebook share URL", error=str(exc), method=method_name
                    )
                    continue

                final_url = response.url
                response.close()
                if final_url and final_url != url:
                    logfire.info(f"Resolved Facebook share URL to: {final_url}")
                    return final_url

        return url

    def _convert_facebook_url(self, url: str) -> str:
        """Convert Facebook watch URL to reel URL format.

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
                logfire.info(f"Converting Facebook watch URL to reel format: {video_id}")
                return f"https://www.facebook.com/reel/{video_id}"

        return url

    def get_params(self, quality: str, dry_run: bool, url: str | None = None) -> dict[str, Any]:
        today = datetime.datetime.now().strftime("%Y%m%d")

        output_path = Path(self.output_folder) / today
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

    def download(self, url: str, quality: str = "best", dry_run: bool = False) -> tuple[str, Path]:
        # Convert Facebook watch URLs to reel format
        url = self._convert_facebook_url(url)

        params = self.get_params(quality=quality, dry_run=dry_run, url=url)
        with YoutubeDL(params=params) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "")
            filename = Path(ydl.prepare_filename(info))
            return title, filename


if __name__ == "__main__":
    downloader = VideoDownloader()
    # url = "https://x.com/reissuerecords/status/1917171960255058421"
    # url = "https://www.facebook.com/watch?v=828357636228730"  # Will be converted to reel format
    url = "https://www.facebook.com/share/r/17h4SsC2p1/"
    # url = "https://www.instagram.com/reels/DFUuxmMPz4n/"
    # url = "https://www.tiktok.com/@zachking/video/6768504823336815877"
    # url = "https://v.douyin.com/LuXDmRrZvWs"
    # url = "https://www.bilibili.com/video/BVs1BHtozkEvc"
    url = "https://www.facebook.com/share/r/1BcvhJkeMg/?mibextid=wwXIfr"
    result = downloader.download(url, "best", False)
