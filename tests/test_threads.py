import pytest

from discordbot.utils.threads import Post, ThreadData, ThreadItem, ThreadsDownloader


@pytest.fixture
def downloader() -> ThreadsDownloader:
    return ThreadsDownloader(output_folder="./data/threads")


def test_find_post_with_parents_returns_chain_in_order() -> None:
    """A `thread_items` list from Threads stores the full reply chain oldest-first."""
    chain = ThreadData(
        thread_items=[
            ThreadItem(post=Post(code="ROOT")),
            ThreadItem(post=Post(code="LVL1")),
            ThreadItem(post=Post(code="LVL2")),
            ThreadItem(post=Post(code="TARGET")),
        ]
    )
    post, parents = chain.find_post_with_parents(post_code="TARGET")
    assert post is not None
    assert post.code == "TARGET"
    assert [p.code for p in parents] == ["ROOT", "LVL1", "LVL2"]


def test_find_post_with_parents_no_match() -> None:
    chain = ThreadData(thread_items=[ThreadItem(post=Post(code="A"))])
    post, parents = chain.find_post_with_parents(post_code="MISSING")
    assert post is None
    assert parents == []


def test_find_post_with_parents_root_has_no_parents() -> None:
    chain = ThreadData(thread_items=[ThreadItem(post=Post(code="ROOT"))])
    post, parents = chain.find_post_with_parents(post_code="ROOT")
    assert post is not None
    assert parents == []


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
        "https://www.threads.com/@lift4life_nickson/post/DXy_VeVmSGK",
        "https://www.threads.com/@babe.0530/post/DXyk3qXGT6o",
    ],
)
def test_parse(downloader: ThreadsDownloader, url: str) -> None:
    with downloader.parse(url=url) as output:
        assert output.text or output.image_urls or output.video_urls, "post should have content"
        assert output.author_name, "author_name should not be empty"
        assert output.taken_at is not None, "taken_at should not be None"
