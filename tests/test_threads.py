import pytest

from discordbot.utils.threads import ThreadsDownloader


@pytest.fixture
def downloader() -> ThreadsDownloader:
    return ThreadsDownloader(output_folder="./data/threads")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.threads.com/@tpp_taiwan/post/DWWIhcQktP_",
        "https://www.threads.com/@wilson.sup/post/DWWMzCNkUUM",
        "https://www.threads.com/@yu0030722025/post/DWZmc8OkSx8",
        "https://www.threads.com/@show4653/post/DWYp35uGh4l",
        "https://www.threads.com/@cyj308/post/DVn6dqzjzQf",
        "https://www.threads.com/@tpp_taiwan/post/DWWIhcQktP_?hl=zh-tw",
        "https://www.threads.com/@tpp_taiwan/post/DWWIhcQktP_?xmt=AQF0p6UfiuvtlPVEKZ36kqN7JVKUzuMJUhGDOwfkJK6Rsw",
    ],
)
def test_parse(downloader: ThreadsDownloader, url: str) -> None:
    with downloader.parse(url=url) as output:
        assert output.text or output.image_urls, "post should have text or images"
        assert output.author_name, "author_name should not be empty"
        assert output.taken_at is not None, "taken_at should not be None"
