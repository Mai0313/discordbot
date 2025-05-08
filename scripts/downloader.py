import re
from pathlib import Path
import datetime

from yt_dlp import YoutubeDL
from pydantic import Field, BaseModel

pattern = re.compile(r"/([^/]+)/?$")


class VideoDownloader(BaseModel):
    output_folder: str = Field(default="downloads", description="Download folder")

    def download(self, url: str) -> tuple[str, str]:
        filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        match = pattern.search(url)
        if match:
            filename_stem = match.group(1)
            if isinstance(filename_stem, str):
                filename = filename_stem

        output_path = Path(self.output_folder)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        outtmpl = f"{output_path.as_posix()}/{filename}.%(ext)s"
        ydl_opts = {
            "format": "best",
            "outtmpl": outtmpl,
            "continuedl": True,
            "restrictfilenames": True,
            "writeinfojson": True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "unknown_title")
        return title, outtmpl


if __name__ == "__main__":
    downloader = VideoDownloader()
    url = "https://x.com/reissuerecords/status/1917171960255058421"
    url = "https://www.facebook.com/share/r/17h4SsC2p1/"
    url = "https://www.instagram.com/reels/DFUuxmMPz4n/"
    downloader.download(url)
