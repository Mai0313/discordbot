from pathlib import Path

from yt_dlp import YoutubeDL
from pydantic import Field, BaseModel


class VideoDownloader(BaseModel):
    output_folder: str = Field(default="downloads", description="Download folder")

    def download(self, url: str) -> str:
        output_path = Path(self.output_folder)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        outtmpl = str(output_path / "%(title)s.%(ext)s")

        ydl_opts = {
            "format": "best",
            "outtmpl": outtmpl,
            "continuedl": True,
            "restrictfilenames": True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return outtmpl


if __name__ == "__main__":
    downloader = VideoDownloader()
    url = "https://x.com/reissuerecords/status/1917171960255058421"  # 替换为实际的URL
    downloader.download(url)
