from pathlib import Path

import pytest

from discordbot.utils.downloader import VideoDownloader


@pytest.mark.parametrize(
    argnames="url",
    argvalues=[
        "https://x.com/reissuerecords/status/1917171960255058421",
        "https://www.facebook.com/share/r/17h4SsC2p1",
        # "https://www.instagram.com/reels/DFUuxmMPz4n",
        "https://www.tiktok.com/@zachking/video/6768504823336815877",
    ],
)
def test_download(url: str) -> None:
    downloader = VideoDownloader(output_folder="./data/downloads")
    with downloader.download(url=url, quality="best", dry_run=True) as result:
        assert isinstance(result.title, str)
        assert isinstance(result.filename, Path)
