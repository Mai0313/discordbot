"""Tests for the Threads-context builder that feeds linked posts to the answer model."""

from types import SimpleNamespace
from pathlib import Path

import pytest

from discordbot.utils.threads import ThreadsOutput, ThreadsDownloader
from discordbot.cogs._parse_threads import builder as threads_builder
from discordbot.cogs._parse_threads.builder import (
    MAX_THREADS_POSTS,
    MAX_THREADS_MEDIA_PARTS,
    THREADS_CONTEXT_SEPARATOR,
    THREADS_UNAVAILABLE_NOTICE,
    THREADS_TEXT_ONLY_SEPARATOR,
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


class _Uploads:
    """Records every media upload the builder performs and hands back canned uris."""

    def __init__(self, fail: bool = False) -> None:
        """Initializes the upload record and whether every upload should fail."""
        self.calls: list[tuple[object, str, str]] = []
        self.fail = fail

    async def __call__(
        self,
        *,
        client: object,
        source: object,
        mime_type: str,
        filename: str,
        timeout_seconds: float,
    ) -> dict[str, str] | None:
        """Stands in for `upload_as_input_file`, returning a Files-API-shaped part."""
        del client, timeout_seconds
        self.calls.append((source, mime_type, filename))
        if self.fail:
            return None
        return {
            "type": "input_file",
            "file_id": f"https://files.test/{filename}",
            "filename": filename,
        }


def _stub_media(
    monkeypatch: pytest.MonkeyPatch, *, uploads: _Uploads, image_fetch_fails: bool = False
) -> None:
    """Stubs the image fetch and the Files API upload so no network or SDK is touched."""

    async def fake_load_image_bytes(source: str) -> tuple[bytes, str]:
        """Returns canned downscaled image bytes for a URL source."""
        if image_fetch_fails:
            raise RuntimeError(f"cdn url expired: {source}")
        return b"image-bytes", "image/jpeg"

    def fake_download_media(self: ThreadsDownloader, url: str, filename: str) -> Path:
        """Writes a stand-in clip into the builder's scratch directory."""
        del url
        path = Path(self.output_folder) / filename
        path.write_bytes(b"clip-bytes")
        return path

    monkeypatch.setattr(threads_builder, "load_image_bytes", fake_load_image_bytes)
    monkeypatch.setattr(threads_builder, "upload_as_input_file", uploads)
    monkeypatch.setattr(target=ThreadsDownloader, name="download_media", value=fake_download_media)


def _client() -> SimpleNamespace:
    """A stand-in Gemini client; the stubbed upload never reaches through it."""
    return SimpleNamespace()


async def test_media_is_uploaded_and_referenced_by_files_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Media rides as input_file parts holding a Files API uri, never a remote URL.

    Handing the model an http(s) url instead makes the proxy base64-inline the media and
    leaves the native Interactions path with a uri Gemini cannot resolve, so the absence of
    `file_url` / http `image_url` is the property worth pinning.
    """
    _stub_parse(
        monkeypatch, [_post(images=["https://cdn.test/a.jpg"], videos=["https://cdn.test/v.mp4"])]
    )
    uploads = _Uploads()
    _stub_media(monkeypatch, uploads=uploads)

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    assert len(blocks) == 2
    assert blocks[0]["content"][0]["text"] == THREADS_CONTEXT_SEPARATOR

    parts = blocks[1]["content"]
    assert parts[0]["type"] == "input_text"
    assert "@alice" in parts[0]["text"]
    assert "TARGET" in parts[0]["text"]

    media = [part for part in parts if part["type"] == "input_file"]
    assert [part["file_id"] for part in media] == [
        "https://files.test/threads_image_0.jpg",
        "https://files.test/threads_video_0.mp4",
    ]
    assert all("file_url" not in part for part in media)
    assert not any(part["type"] == "input_image" for part in parts)
    # The filename keeps a real extension: the native Interactions bridge classifies by it.
    assert [part["filename"] for part in media] == ["threads_image_0.jpg", "threads_video_0.mp4"]


async def test_images_are_downscaled_before_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Images go through load_image_bytes, which downscales; raw URLs bypassed that entirely."""
    _stub_parse(monkeypatch, [_post(images=["https://cdn.test/a.jpg"])])
    uploads = _Uploads()
    _stub_media(monkeypatch, uploads=uploads)

    await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    source, mime_type, _ = uploads.calls[0]
    assert source == b"image-bytes"  # the downscaled bytes, not the URL
    assert mime_type == "image/jpeg"


async def test_video_is_uploaded_from_disk_and_cleaned_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip is downloaded to a scratch dir, uploaded by path, then removed."""
    _stub_parse(monkeypatch, [_post(videos=["https://cdn.test/v.mp4"])])
    uploads = _Uploads()
    _stub_media(monkeypatch, uploads=uploads)

    await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    source, mime_type, _ = uploads.calls[0]
    assert isinstance(source, Path)  # streamed from disk, never read into memory
    assert mime_type == "video/mp4"
    assert not source.exists()  # deleted after the upload, and its temp dir is gone too


async def test_only_the_target_posts_media_is_ingested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ancestors contribute text only; each media part now costs a fetch plus an upload."""
    ancestor = _post(text="ancestor", images=["https://cdn.test/ancestor.jpg"])
    target = _post(text="target", images=["https://cdn.test/target.jpg"])
    _stub_parse(monkeypatch, [ancestor, target])  # chain is [root, ..., target]
    uploads = _Uploads()
    _stub_media(monkeypatch, uploads=uploads)

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    media = [part for part in blocks[1]["content"] if part["type"] == "input_file"]
    assert len(media) == 1
    assert len(uploads.calls) == 1
    text = blocks[1]["content"][0]["text"]
    assert "ancestor" in text  # the ancestor still supplies context, just no media


async def test_build_caps_media_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A large carousel is capped at MAX_THREADS_MEDIA_PARTS media parts."""
    images = [f"https://cdn.test/{index}.jpg" for index in range(MAX_THREADS_MEDIA_PARTS + 5)]
    _stub_parse(monkeypatch, [_post(images=images)])
    _stub_media(monkeypatch, uploads=_Uploads())

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    media = [part for part in blocks[1]["content"] if part["type"] == "input_file"]
    assert len(media) == MAX_THREADS_MEDIA_PARTS


async def test_videos_share_the_media_budget_with_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """Images claim the budget first and videos take what is left, never exceeding the cap.

    A cap test fed images only leaves the video slice at zero, so it would pass even if the
    video half ignored the budget entirely.
    """
    images = [f"https://cdn.test/{index}.jpg" for index in range(MAX_THREADS_MEDIA_PARTS - 1)]
    videos = [f"https://cdn.test/{index}.mp4" for index in range(4)]
    _stub_parse(monkeypatch, [_post(images=images, videos=videos)])
    uploads = _Uploads()
    _stub_media(monkeypatch, uploads=uploads)

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    media = [part for part in blocks[1]["content"] if part["type"] == "input_file"]
    assert len(media) == MAX_THREADS_MEDIA_PARTS
    names = [part["filename"] for part in media]
    assert sum(name.endswith(".mp4") for name in names) == 1  # only the leftover slot
    assert sum(name.endswith(".jpg") for name in names) == MAX_THREADS_MEDIA_PARTS - 1


async def test_a_full_image_budget_leaves_no_room_for_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When images fill the cap the videos are dropped rather than pushing it over."""
    images = [f"https://cdn.test/{index}.jpg" for index in range(MAX_THREADS_MEDIA_PARTS)]
    _stub_parse(monkeypatch, [_post(images=images, videos=["https://cdn.test/v.mp4"])])
    uploads = _Uploads()
    _stub_media(monkeypatch, uploads=uploads)

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    media = [part for part in blocks[1]["content"] if part["type"] == "input_file"]
    assert len(media) == MAX_THREADS_MEDIA_PARTS
    assert all(part["filename"].endswith(".jpg") for part in media)


async def test_build_caps_chain_posts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long reply chain is trimmed to the target plus its nearest ancestors."""
    chain = [
        _post(text=f"post {index}", author=f"user{index}")
        for index in range(MAX_THREADS_POSTS + 4)
    ]  # oldest-first; the last is the target
    _stub_parse(monkeypatch, chain)
    _stub_media(monkeypatch, uploads=_Uploads())

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    text = blocks[1]["content"][0]["text"]
    # The target and the nearest ancestors are kept; the oldest posts are dropped.
    assert "TARGET" in text
    assert f"post {MAX_THREADS_POSTS + 3}" in text  # the target (last) survives
    assert "post 0" not in text  # the oldest ancestor is trimmed
    rendered_posts = text.count("ANCESTOR") + text.count("TARGET")
    assert rendered_posts == MAX_THREADS_POSTS


async def test_build_non_gemini_rides_urls_as_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-Gemini answer model gets the URLs as text and triggers no upload at all."""
    _stub_parse(
        monkeypatch, [_post(images=["https://cdn.test/a.jpg"], videos=["https://cdn.test/v.mp4"])]
    )
    uploads = _Uploads()
    _stub_media(monkeypatch, uploads=uploads)

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=False, gemini_client=_client()
    )

    # The separator must not claim the media was fetched, since only its URLs are supplied.
    assert blocks[0]["content"][0]["text"] == THREADS_TEXT_ONLY_SEPARATOR
    parts = blocks[1]["content"]
    assert [part["type"] for part in parts] == ["input_text"]
    text = parts[0]["text"]
    assert "https://cdn.test/a.jpg" in text
    assert "https://cdn.test/v.mp4" in text
    assert uploads.calls == []  # a Files uri is Gemini-only, so nothing is uploaded


async def test_failed_media_degrades_to_an_honest_text_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every fetch fails the model is told the media is NOT attached, never that it is."""
    _stub_parse(monkeypatch, [_post(images=["https://cdn.test/a.jpg"])])
    _stub_media(monkeypatch, uploads=_Uploads(), image_fetch_fails=True)

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    assert blocks[0]["content"][0]["text"] == THREADS_TEXT_ONLY_SEPARATOR
    parts = blocks[1]["content"]
    assert [part["type"] for part in parts] == ["input_text"]
    assert "https://cdn.test/a.jpg" in parts[0]["text"]


async def test_failed_upload_degrades_to_an_honest_text_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetch that works but an upload that fails must not claim the media was seen."""
    _stub_parse(monkeypatch, [_post(images=["https://cdn.test/a.jpg"])])
    _stub_media(monkeypatch, uploads=_Uploads(fail=True))

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    assert blocks[0]["content"][0]["text"] == THREADS_TEXT_ONLY_SEPARATOR
    assert [part["type"] for part in blocks[1]["content"]] == ["input_text"]


async def test_one_failed_item_does_not_sink_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    """Media items are independent, so an expired image url still leaves the video attached."""
    _stub_parse(
        monkeypatch, [_post(images=["https://cdn.test/a.jpg"], videos=["https://cdn.test/v.mp4"])]
    )
    uploads = _Uploads()
    _stub_media(monkeypatch, uploads=uploads, image_fetch_fails=True)

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    assert blocks[0]["content"][0]["text"] == THREADS_CONTEXT_SEPARATOR
    media = [part for part in blocks[1]["content"] if part["type"] == "input_file"]
    assert [part["filename"] for part in media] == ["threads_video_0.mp4"]


async def test_text_only_post_keeps_the_context_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    """A post with no media at all is fully represented, so nothing is withheld from the model."""
    _stub_parse(monkeypatch, [_post()])
    _stub_media(monkeypatch, uploads=_Uploads())

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    assert blocks[0]["content"][0]["text"] == THREADS_CONTEXT_SEPARATOR


async def test_build_empty_post_returns_unavailable_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A private/deleted post (empty parse) yields a single unavailable-notice block."""
    _stub_parse(monkeypatch, [])

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    assert len(blocks) == 1
    assert blocks[0]["role"] == "system"
    assert blocks[0]["content"][0]["text"] == THREADS_UNAVAILABLE_NOTICE


async def test_build_parse_error_degrades_to_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parse error degrades to the unavailable notice instead of raising into the pipeline."""

    def boom(self: ThreadsDownloader, *, url: str) -> list[ThreadsOutput]:
        """Simulates an HTTP/parse failure."""
        raise RuntimeError("fetch failed")

    monkeypatch.setattr(target=ThreadsDownloader, name="parse_metadata", value=boom)

    blocks = await build_threads_context_messages(
        url=_URL, answer_model_is_gemini=True, gemini_client=_client()
    )

    assert len(blocks) == 1
    assert blocks[0]["content"][0]["text"] == THREADS_UNAVAILABLE_NOTICE
