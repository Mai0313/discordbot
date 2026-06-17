"""Tests for the Threads-context builder that feeds linked posts to the answer model."""

import pytest

from discordbot.utils.threads import ThreadsOutput, ThreadsDownloader
from discordbot.cogs._parse_threads.builder import (
    MAX_THREADS_POSTS,
    MAX_THREADS_MEDIA_PARTS,
    THREADS_CONTEXT_SEPARATOR,
    THREADS_UNAVAILABLE_NOTICE,
    build_threads_context_messages,
)

_URL = "https://www.threads.com/@alice/post/ABC123"


def _post(
    text: str = "post body",
    images: list[str] | None = None,
    videos: list[str] | None = None,
    author: str = "alice",
) -> ThreadsOutput:
    """Builds a ThreadsOutput with the engagement fields the builder renders."""
    return ThreadsOutput(
        text=text,
        url=_URL,
        image_urls=images or [],
        video_urls=videos or [],
        author_name=author,
        like_count=1,
        reply_count=2,
        repost_count=3,
        quote_count=4,
        reshare_count=5,
    )


def _stub_parse(monkeypatch: pytest.MonkeyPatch, results: list[ThreadsOutput]) -> None:
    """Replaces ThreadsDownloader.parse_metadata with a canned chain (no network)."""

    def fake_parse_metadata(self: ThreadsDownloader, *, url: str) -> list[ThreadsOutput]:
        """Returns the canned chain regardless of url."""
        del url
        return results

    monkeypatch.setattr(target=ThreadsDownloader, name="parse_metadata", value=fake_parse_metadata)


async def test_build_gemini_renders_separator_text_and_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readable post yields the separator plus a user block of text + image/video URL parts."""
    _stub_parse(
        monkeypatch, [_post(images=["https://cdn.test/a.jpg"], videos=["https://cdn.test/v.mp4"])]
    )

    blocks = await build_threads_context_messages(url=_URL, answer_model_is_gemini=True)

    assert len(blocks) == 2
    assert blocks[0]["role"] == "system"
    assert blocks[0]["content"][0]["text"] == THREADS_CONTEXT_SEPARATOR

    user = blocks[1]
    assert user["role"] == "user"
    parts = user["content"]
    assert parts[0]["type"] == "input_text"
    assert "@alice" in parts[0]["text"]
    assert "TARGET" in parts[0]["text"]

    image_part = next(part for part in parts if part["type"] == "input_image")
    file_part = next(part for part in parts if part["type"] == "input_file")
    assert image_part["image_url"] == "https://cdn.test/a.jpg"
    assert file_part["file_url"] == "https://cdn.test/v.mp4"


async def test_build_caps_media_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A large carousel is capped at MAX_THREADS_MEDIA_PARTS media parts."""
    images = [f"https://cdn.test/{index}.jpg" for index in range(MAX_THREADS_MEDIA_PARTS + 5)]
    _stub_parse(monkeypatch, [_post(images=images)])

    blocks = await build_threads_context_messages(url=_URL, answer_model_is_gemini=True)

    media = [
        part for part in blocks[1]["content"] if part["type"] in ("input_image", "input_file")
    ]
    assert len(media) == MAX_THREADS_MEDIA_PARTS


async def test_build_collects_target_media_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The linked post's media leads, so an ancestor never crowds it out of the cap."""
    ancestor = _post(text="ancestor", images=["https://cdn.test/ancestor.jpg"])
    target = _post(text="target", images=["https://cdn.test/target.jpg"])
    _stub_parse(monkeypatch, [ancestor, target])  # chain is [root, ..., target]

    blocks = await build_threads_context_messages(url=_URL, answer_model_is_gemini=True)

    images = [part["image_url"] for part in blocks[1]["content"] if part["type"] == "input_image"]
    assert images[0] == "https://cdn.test/target.jpg"


async def test_build_media_priority_is_target_then_nearest_ancestor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Media is ordered target, then direct parent, then root, so the cap drops the root first."""
    root = _post(text="root", images=["https://cdn.test/root.jpg"])
    parent = _post(text="parent", images=["https://cdn.test/parent.jpg"])
    target = _post(text="target", images=["https://cdn.test/target.jpg"])
    _stub_parse(monkeypatch, [root, parent, target])  # oldest-first

    blocks = await build_threads_context_messages(url=_URL, answer_model_is_gemini=True)

    images = [part["image_url"] for part in blocks[1]["content"] if part["type"] == "input_image"]
    assert images == [
        "https://cdn.test/target.jpg",
        "https://cdn.test/parent.jpg",
        "https://cdn.test/root.jpg",
    ]


async def test_build_caps_chain_posts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long reply chain is trimmed to the target plus its nearest ancestors."""
    chain = [
        _post(text=f"post {index}", author=f"user{index}")
        for index in range(MAX_THREADS_POSTS + 4)
    ]  # oldest-first; the last is the target
    _stub_parse(monkeypatch, chain)

    blocks = await build_threads_context_messages(url=_URL, answer_model_is_gemini=True)

    text = blocks[1]["content"][0]["text"]
    # The target and the nearest ancestors are kept; the oldest posts are dropped.
    assert "TARGET" in text
    assert f"post {MAX_THREADS_POSTS + 3}" in text  # the target (last) survives
    assert "post 0" not in text  # the oldest ancestor is trimmed
    rendered_posts = text.count("ANCESTOR") + text.count("TARGET")
    assert rendered_posts == MAX_THREADS_POSTS


async def test_build_non_gemini_rides_urls_as_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-Gemini answer model gets the URLs as text, never image/file URL parts."""
    _stub_parse(
        monkeypatch, [_post(images=["https://cdn.test/a.jpg"], videos=["https://cdn.test/v.mp4"])]
    )

    blocks = await build_threads_context_messages(url=_URL, answer_model_is_gemini=False)

    parts = blocks[1]["content"]
    assert [part["type"] for part in parts] == ["input_text"]
    text = parts[0]["text"]
    assert "https://cdn.test/a.jpg" in text
    assert "https://cdn.test/v.mp4" in text


async def test_build_empty_post_returns_unavailable_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A private/deleted post (empty parse) yields a single unavailable-notice block."""
    _stub_parse(monkeypatch, [])

    blocks = await build_threads_context_messages(url=_URL, answer_model_is_gemini=True)

    assert len(blocks) == 1
    assert blocks[0]["role"] == "system"
    assert blocks[0]["content"][0]["text"] == THREADS_UNAVAILABLE_NOTICE


async def test_build_parse_error_degrades_to_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parse error degrades to the unavailable notice instead of raising into the pipeline."""

    def boom(self: ThreadsDownloader, *, url: str) -> list[ThreadsOutput]:
        """Simulates an HTTP/parse failure."""
        raise RuntimeError("fetch failed")

    monkeypatch.setattr(target=ThreadsDownloader, name="parse_metadata", value=boom)

    blocks = await build_threads_context_messages(url=_URL, answer_model_is_gemini=True)

    assert len(blocks) == 1
    assert blocks[0]["content"][0]["text"] == THREADS_UNAVAILABLE_NOTICE
