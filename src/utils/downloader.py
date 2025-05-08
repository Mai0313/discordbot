import re
from pathlib import Path
import datetime

from yt_dlp import YoutubeDL
from pydantic import Field, BaseModel, computed_field

pattern = re.compile(r"/([^/]+)/?$")


class VideoDownloader(BaseModel):
    output_folder: str = Field(default="downloads", description="Download folder")

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

    def download(self, url: str, quality: str = "best") -> tuple[str, Path]:
        filename_stem = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        match = pattern.search(url)
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
            "continuedl": True,
            "restrictfilenames": True,
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
    url = "https://x.com/reissuerecords/status/1917171960255058421"
    url = "https://www.facebook.com/share/r/17h4SsC2p1/"
    url = "https://www.instagram.com/reels/DFUuxmMPz4n/"
    downloader.download(url)
