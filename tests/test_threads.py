"""Tests for Threads URL parsing and media extraction."""

import json
from pathlib import Path

import pytest

from discordbot.utils.threads import (
    THREADS_URL_RE,
    Post,
    ThreadData,
    ThreadItem,
    ThreadsURL,
    ThreadsOutput,
    ThreadsDownloader,
    ThreadsExpansionRelay,
)


@pytest.fixture
def downloader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ThreadsDownloader:
    """Provides a ThreadsDownloader that fakes media downloads."""

    def fake_download_media(self: ThreadsDownloader, url: str, filename: str) -> Path:
        """Writes fake media bytes and returns the expected output path."""
        assert url, "download url should not be empty"
        filepath = Path(self.output_folder) / filename
        filepath.write_bytes(data=b"fake media")
        return filepath

    monkeypatch.setattr(target=ThreadsDownloader, name="download_media", value=fake_download_media)
    return ThreadsDownloader(output_folder=str(tmp_path))


def test_find_post_with_parents_returns_chain_in_order() -> None:
    """Verifies that finding a post returns the correct chain of parent posts."""
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
    """Verifies that finding a non-existent post returns None and empty parents."""
    chain = ThreadData(thread_items=[ThreadItem(post=Post(code="A"))])
    post, parents = chain.find_post_with_parents(post_code="MISSING")
    assert post is None
    assert parents == []


def test_find_post_with_parents_root_has_no_parents() -> None:
    """Verifies that the root post of a thread has no parents."""
    chain = ThreadData(thread_items=[ThreadItem(post=Post(code="ROOT"))])
    post, parents = chain.find_post_with_parents(post_code="ROOT")
    assert post is not None
    assert parents == []


def test_threads_output_mutable_defaults_are_isolated(tmp_path: Path) -> None:
    """Threads output image and local video path defaults are isolated."""
    first = ThreadsOutput()
    second = ThreadsOutput()

    first.image_urls.append("https://cdn.example/image.jpg")
    first.video_paths.append(tmp_path / "clip.mp4")

    assert second.image_urls == []
    assert second.video_paths == []


def _thread_post_payload(
    code: str, username: str, text: str, reply_to_username: str = ""
) -> dict[str, object]:
    """Returns a minimal Threads post payload with parser-relevant fields."""
    return {
        "post": {
            "code": code,
            "caption": {"text": text},
            "user": {
                "username": username,
                "profile_pic_url": f"https://cdn.example/{username}.jpg",
            },
            "image_versions2": {"candidates": [{"url": f"https://cdn.example/{code}.jpg"}]},
            "text_post_app_info": {
                "direct_reply_count": 1,
                "repost_count": 2,
                "quote_count": 3,
                "reshare_count": 4,
                "is_reply": bool(reply_to_username),
                "reply_to_author": (
                    {"username": reply_to_username, "profile_pic_url": ""}
                    if reply_to_username
                    else None
                ),
            },
            "like_count": 5,
            "taken_at": 1_735_689_600,
        }
    }


def _thread_html(post_code: str) -> str:
    """Builds deterministic Threads SJS HTML for parser tests."""
    payload = {
        "require": [
            {
                "__bbox": {
                    "result": {
                        "data": {
                            "thread_items": [
                                _thread_post_payload(
                                    code="ROOT", username="root_author", text="Root post"
                                ),
                                _thread_post_payload(
                                    code=post_code,
                                    username="target_author",
                                    text=f"Target post {post_code}",
                                    reply_to_username="root_author",
                                ),
                            ]
                        }
                    }
                }
            }
        ]
    }
    return (
        f'<html><script type="application/json" data-sjs>{json.dumps(obj=payload)}</script></html>'
    )


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
        "https://www.threads.com/@babe.0530/post/DXyk3qXGT6o",
    ],
)
def test_parse(downloader: ThreadsDownloader, url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that parsing valid Threads URLs returns expected post data."""
    threads_url = ThreadsURL(raw_url=url)
    fetched_urls: list[str] = []

    def fake_fetch_html(self: ThreadsDownloader, url: str) -> str:
        """Returns deterministic HTML for the requested Threads URL."""
        fetched_urls.append(url)
        return _thread_html(post_code=threads_url.post_code)

    monkeypatch.setattr(target=ThreadsDownloader, name="_fetch_html", value=fake_fetch_html)

    with downloader.parse(url=url) as results:
        assert results, "should yield at least one post"
        target = results[-1]
        assert target.text or target.image_urls or target.video_urls, "post should have content"
        assert target.author_name, "author_name should not be empty"
        assert target.taken_at is not None, "taken_at should not be None"
    assert fetched_urls == [threads_url.clean_url]


def test_post_tolerates_null_string_fields() -> None:
    """A post whose link preview serialises image_url as null still parses."""
    post = Post.model_validate(
        obj={
            "code": "NULLPREV",
            "caption": {"text": "shared a link"},
            "user": {"username": "author", "profile_pic_url": ""},
            "text_post_app_info": {
                "link_preview_attachment": {
                    "title": "instagram.com",
                    "image_url": None,
                    "url": "https://www.instagram.com/reel/abc/",
                }
            },
        }
    )
    assert post.code == "NULLPREV"
    assert post.text_post_app_info is not None
    assert post.text_post_app_info.link_preview_attachment is not None
    assert post.text_post_app_info.link_preview_attachment.image_url == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://www.threads.com/@user/post/ABC123", True),
        ("https://threads.net/@u/post/XyZ", True),
        ("look @bot https://www.threads.com/@a.b/post/DZZ?x=1 ok", True),
        ("https://www.instagram.com/p/abc/", False),
        ("https://www.threads.com/@user", False),
        ("no url here at all", False),
    ],
)
def test_threads_url_re_matches_post_links(text: str, expected: bool) -> None:
    """The shared regex matches Threads post URLs and nothing else."""
    assert (THREADS_URL_RE.search(string=text) is not None) is expected


async def test_expansion_relay_get_or_create_is_idempotent() -> None:
    """Both cogs reach the same future for one message, and resolve sets its result once."""
    relay = ThreadsExpansionRelay()
    first = relay.get_or_create(message_id=1)
    assert relay.get_or_create(message_id=1) is first

    relay.resolve(message_id=1, message=None)
    assert first.done()
    assert await first is None


async def test_expansion_relay_resolve_unknown_is_noop() -> None:
    """Resolving a message nobody registered does not raise and creates nothing lingering."""
    relay = ThreadsExpansionRelay()
    relay.resolve(message_id=999, message=None)

    pending = relay.get_or_create(message_id=999)
    assert not pending.done()
    pending.cancel()


async def test_expansion_relay_evicts_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pending map is bounded so unawaited expansions cannot grow it without limit."""
    monkeypatch.setattr("discordbot.utils.threads._MAX_PENDING_EXPANSIONS", 2)
    relay = ThreadsExpansionRelay()
    relay.get_or_create(message_id=1)
    relay.get_or_create(message_id=2)
    relay.get_or_create(message_id=3)

    # id 1 was evicted: resolving it is a no-op and a fresh lookup yields a new pending future.
    relay.resolve(message_id=1, message=None)
    refreshed = relay.get_or_create(message_id=1)
    assert not refreshed.done()
    refreshed.cancel()
