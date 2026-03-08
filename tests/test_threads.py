import pytest

from discordbot.utils.threads import ThreadsDownloader


@pytest.mark.parametrize(
    argnames="url",
    argvalues=[
        "https://www.threads.com/@myun.60761/post/DVnP0ATET7d?xmt=AQF0GAejzXClnOrILy2_aqEN7a0IhvY6Nq4iAsUbI0K_Yw",
        "https://www.threads.com/@cyj308/post/DVn6dqzjzQf?hl=zh-tw",
    ],
)
def test_parse(url: str) -> None:
    downloader = ThreadsDownloader(output_folder="./data/threads")
    output = downloader.parse(url=url)
    assert isinstance(output.text, str)
    assert isinstance(output.media_urls, list)
    assert isinstance(output.media_paths, list)
