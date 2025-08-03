import re
from pathlib import Path
import datetime

from yt_dlp import YoutubeDL
from pydantic import Field, BaseModel, computed_field

shorts_pattern = re.compile(r"/([^/]+)/?$")
tiktok_pattern = re.compile(r"https?://((?:vm|vt|www)\.)?tiktok\.com/.*")


class VideoDownloader(BaseModel):
    output_folder: str = Field(default="./data/downloads", description="Download folder")

    @computed_field
    @property
    def quality_formats(self) -> dict[str, str]:
        quality_formats = {
            "best": "best",
            "high": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "medium": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "low": "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "audio": "bestaudio/best",
        }
        return quality_formats

    def check_if_tiktok(self, url: str) -> bool:
        return bool(tiktok_pattern.match(url))

    def download(self, url: str, quality: str = "best") -> tuple[str, Path]:
        filename_stem = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        match = shorts_pattern.search(url)
        if match:
            found_name = match.group(1)
            if isinstance(found_name, str):
                filename_stem = found_name

        output_path = Path(self.output_folder)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        # 獲取所選畫質的格式設定
        format_option = self.quality_formats.get(quality, "best")
        is_audio_only = quality == "audio"
        outtmpl = f"{output_path.as_posix()}/{filename_stem}.%(ext)s"

        ydl_opts = {
            "format": format_option,
            "outtmpl": outtmpl,
            "quiet": False,
            "continuedl": True,
            "noplaylist": True,
            "restrictfilenames": True,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            },
            # "writeinfojson": True,
        }

        # 如果是音訊模式，轉換成 mp3
        if is_audio_only:
            ydl_opts.update({
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ]
            })

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "")
            filename = Path(ydl.prepare_filename(info))
            # 修正音訊模式下的副檔名
            if is_audio_only and filename.suffix != ".mp3":
                filename = filename.with_suffix(".mp3")
        return title, filename


if __name__ == "__main__":
    downloader = VideoDownloader()
    # url = "https://x.com/reissuerecords/status/1917171960255058421"
    # url = "https://www.facebook.com/share/r/17h4SsC2p1/"
    # url = "https://www.instagram.com/reels/DFUuxmMPz4n/"
    url = "https://www.tiktok.com/@zachking/video/6768504823336815877"
    result = downloader.download(url)
