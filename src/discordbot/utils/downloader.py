import time
from typing import Any
from pathlib import Path
import datetime

from yt_dlp import YoutubeDL
import logfire
from pydantic import Field, BaseModel, computed_field

logfire.configure(send_to_logfire=False, scrubbing=False)


class VideoDownloader(BaseModel):
    output_folder: str = Field(default="./data/downloads", description="Download folder")
    max_retries: int = Field(default=5)

    @computed_field
    @property
    def quality_formats(self) -> dict[str, str]:
        quality_formats = {
            # Prefer separate video+audio with safe fallbacks to muxed or video-only streams
            "best": "bestvideo*+bestaudio/best/bestvideo*",
            "high": "bestvideo[height<=1080][fps<=60]+bestaudio/best[height<=1080][fps<=60]/best[height<=1080]",
            "medium": "bestvideo[height<=720][fps<=60]+bestaudio/best[height<=720][fps<=60]/best[height<=720]",
            "low": "bestvideo[height<=480]+bestaudio/best[height<=480]/best[height<=480]",
        }
        return quality_formats

    def get_params(self, quality: str, dry_run: bool, url: str | None = None) -> dict[str, Any]:
        today = datetime.datetime.now().strftime("%Y%m%d")

        output_path = Path(self.output_folder) / today
        output_path.mkdir(parents=True, exist_ok=True)

        # Base headers safe for most sites; site-specific headers added conditionally below
        http_headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7",
        }
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
        params = self.get_params(quality=quality, dry_run=dry_run, url=url)
        with YoutubeDL(params=params) as ydl:
            last_error: Exception | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    info = ydl.extract_info(url, download=True)
                    title = info.get("title", "")
                    filename = Path(ydl.prepare_filename(info))
                    return title, filename
                except Exception as exc:  # noqa: PERF203
                    last_error = exc
                    if attempt < self.max_retries:
                        logfire.warning(f"[Retry {attempt}/{self.max_retries}], retrying...")
                        time.sleep(1)
        logfire.error(f"[Failed after {self.max_retries} attempts]")
        if last_error is not None:
            raise last_error
        raise RuntimeError("Download failed with unknown error")


if __name__ == "__main__":
    downloader = VideoDownloader()
    # url = "https://x.com/reissuerecords/status/1917171960255058421"
    url = "https://www.facebook.com/share/r/17h4SsC2p1/"
    # url = "https://www.instagram.com/reels/DFUuxmMPz4n/"
    # url = "https://www.tiktok.com/@zachking/video/6768504823336815877"
    # url = "https://v.douyin.com/LuXDmRrZvWs"
    url = "https://www.bilibili.com/video/BV1BHtozkEvc"
    result = downloader.download(url, "best", False)
