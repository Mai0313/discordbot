from pathlib import Path

from yt_dlp import YoutubeDL
from pydantic import Field, BaseModel


class VideoDownloader(BaseModel):
    output_folder: str = Field(default="downloads", description="Download folder")

    def download(self, url: str) -> str:
        output_path = Path(self.output_folder)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            "format": "best",
            "outtmpl": str(output_path / "%(title).40s-%(id)s.%(ext)s"),
            "continuedl": True,
            "restrictfilenames": True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            title = info_dict.get("title", "unknown_title")
            # ydl.download([url])
        return title


if __name__ == "__main__":
    downloader = VideoDownloader()
    url = "https://x.com/reissuerecords/status/1917171960255058421"  # 替换为实际的URL
    downloader.download(url)
