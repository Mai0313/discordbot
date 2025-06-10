import pytest
from src.utils.downloader import VideoDownloader


@pytest.fixture
def downloader() -> VideoDownloader:
    return VideoDownloader()


def test_check_if_tiktok(downloader: VideoDownloader) -> None:
    tiktok_url = "https://www.tiktok.com/@user/video/123"
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert downloader.check_if_tiktok(tiktok_url) is True
    assert downloader.check_if_tiktok(youtube_url) is False


def test_quality_formats_contains_keys(downloader: VideoDownloader) -> None:
    formats = downloader.quality_formats
    expected = {"best", "high", "medium", "low", "audio"}
    assert expected.issubset(set(formats))
