"""Tests for the Douyin-context builder that feeds linked posts to the answer model."""

from types import SimpleNamespace
import asyncio
from pathlib import Path

import pytest

from discordbot.utils.douyin import (
    DouyinPost,
    DouyinError,
    DouyinDownload,
    DouyinDownloader,
    DouyinBlockedError,
    DouyinTooLargeError,
    DouyinUnavailableError,
)
from discordbot.cogs._parse_douyin import fetch as douyin_fetch
from discordbot.cogs._parse_douyin import builder as douyin_builder
from discordbot.cogs._parse_douyin.builder import (
    DOUYIN_BLOCKED_NOTICE,
    DOUYIN_CONTEXT_SEPARATOR,
    DOUYIN_UNREADABLE_NOTICE,
    MAX_DOUYIN_INGEST_IMAGES,
    DOUYIN_UNAVAILABLE_NOTICE,
    DOUYIN_TEXT_ONLY_SEPARATOR,
    build_douyin_context_messages,
)

_URL = "https://v.douyin.com/abc123"


def _post(is_photo: bool = False, images: int = 0) -> DouyinPost:
    """Builds the parsed metadata the builder renders into its text block."""
    return DouyinPost(
        aweme_id="777",
        title="一段影片的說明",
        author_name="某個作者",
        is_photo=is_photo,
        video_id="" if is_photo else "vid",
        image_urls=[f"https://cdn.test/{index}.jpg" for index in range(images)],
    )


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


def _stub_douyin(  # noqa: PLR0913 -- one canned outcome per stage the builder can hit
    monkeypatch: pytest.MonkeyPatch,
    *,
    post: DouyinPost | None = None,
    files: list[str] | None = None,
    parse_error: Exception | None = None,
    download_error: Exception | None = None,
    uploads: _Uploads | None = None,
) -> tuple[_Uploads, dict[str, object]]:
    """Stubs the downloader and the Files API upload so no network or SDK is touched."""
    resolved_post = post or _post()
    recorded: dict[str, object] = {}

    def fake_parse_metadata(self: DouyinDownloader, url: str) -> DouyinPost:
        """Returns the canned post, or raises the canned parse failure."""
        del url
        if parse_error is not None:
            raise parse_error
        return resolved_post

    def fake_download(  # noqa: PLR0913 -- mirrors DouyinDownloader.download exactly
        self: DouyinDownloader,
        url: str,
        quality: str = "best",
        max_images: int | None = None,
        max_bytes: int | None = None,
        post: DouyinPost | None = None,
    ) -> DouyinDownload:
        """Writes canned files into the builder's scratch dir, or raises."""
        del url, quality
        recorded["max_images"] = max_images
        recorded["max_bytes"] = max_bytes
        recorded["post"] = post
        if download_error is not None:
            raise download_error
        names = files if files is not None else [f"{resolved_post.aweme_id}.mp4"]
        written: list[Path] = []
        for name in names[: max_images or len(names)]:
            path = Path(self.output_folder) / name
            path.write_bytes(b"media-bytes")
            written.append(path)
        return DouyinDownload(
            title=resolved_post.title, is_photo=resolved_post.is_photo, filenames=written
        )

    monkeypatch.setattr(target=DouyinDownloader, name="parse_metadata", value=fake_parse_metadata)
    monkeypatch.setattr(target=DouyinDownloader, name="download", value=fake_download)
    resolved_uploads = uploads or _Uploads()
    monkeypatch.setattr(douyin_builder, "upload_as_input_file", resolved_uploads)
    return resolved_uploads, recorded


async def _build(gemini: bool = True, ingest: bool = True) -> list[dict[str, object]]:
    """Runs the builder with the flags most tests share."""
    return await build_douyin_context_messages(
        url=_URL,
        answer_model_is_gemini=gemini,
        gemini_client=SimpleNamespace(),
        allow_media_ingest=ingest,
    )


async def test_the_clip_is_uploaded_and_referenced_by_files_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The video rides as an input_file holding a Files API uri, never a Douyin URL.

    A Douyin CDN url is unusable to both backends anyway (the play endpoint needs a mobile
    User-Agent), so the upload is the only shape that works, not merely the tidier one.
    """
    uploads, recorded = _stub_douyin(monkeypatch)

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == DOUYIN_CONTEXT_SEPARATOR
    parts = blocks[1]["content"]
    assert parts[0]["type"] == "input_text"
    assert "一段影片的說明" in parts[0]["text"]
    assert "某個作者" in parts[0]["text"]
    assert _URL in parts[0]["text"]

    media = [part for part in parts if part["type"] == "input_file"]
    assert [part["file_id"] for part in media] == ["https://files.test/777.mp4"]
    assert all("file_url" not in part for part in media)
    # The clip is streamed from disk and its mime is real; the extension is load-bearing on the
    # native Interactions path, which classifies a part by it.
    source, mime_type, filename = uploads.calls[0]
    assert isinstance(source, Path)
    assert mime_type == "video/mp4"
    assert filename.endswith(".mp4")
    # Only the provider's own ceiling is applied; a full-resolution clip is the point.
    assert recorded["max_bytes"] == douyin_builder.FILES_API_MAX_BYTES


async def test_the_parsed_post_is_handed_to_the_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caption is parsed once and reused, so the post is never resolved twice."""
    _, recorded = _stub_douyin(monkeypatch)

    await _build()

    assert isinstance(recorded["post"], DouyinPost)


async def test_a_gallery_is_capped_and_uploaded_as_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """A photo post rides as image parts, capped so a huge gallery cannot blow the budget."""
    uploads, recorded = _stub_douyin(
        monkeypatch,
        post=_post(is_photo=True, images=20),
        files=[f"777_{index}.jpg" for index in range(20)],
    )

    blocks = await _build()

    assert recorded["max_images"] == MAX_DOUYIN_INGEST_IMAGES
    media = [part for part in blocks[1]["content"] if part["type"] == "input_file"]
    assert len(media) == MAX_DOUYIN_INGEST_IMAGES
    assert all(mime == "image/jpeg" for _source, mime, _name in uploads.calls)
    assert "photo gallery" in blocks[1]["content"][0]["text"]


async def test_a_blocked_read_is_never_reported_as_a_missing_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A WAF block is retryable and the link is fine, so it gets its own notice wording.

    Reporting it as a deleted post is the worst failure this feature can produce.
    """
    _stub_douyin(monkeypatch, parse_error=DouyinBlockedError("bot wall"))

    blocks = await _build()

    assert len(blocks) == 1
    assert blocks[0]["content"][0]["text"] == DOUYIN_BLOCKED_NOTICE
    assert blocks[0]["content"][0]["text"] != DOUYIN_UNAVAILABLE_NOTICE


async def test_a_deleted_post_gets_the_unavailable_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    """A post Douyin refuses to serve is reported as deleted or private."""
    _stub_douyin(monkeypatch, parse_error=DouyinUnavailableError("filtered"))

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == DOUYIN_UNAVAILABLE_NOTICE


async def test_any_other_failure_never_claims_the_post_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure that says nothing about the post must not be reported as a deleted one.

    An unresolvable link, a transport error or a changed payload shape all surface as a bare
    `DouyinError`; asserting the post is gone would send the user off to re-check a link that
    is very likely fine. Only Douyin explicitly filtering the post out earns that wording.
    """
    _stub_douyin(monkeypatch, parse_error=DouyinError("could not find a post id"))

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == DOUYIN_UNREADABLE_NOTICE
    assert blocks[0]["content"][0]["text"] != DOUYIN_UNAVAILABLE_NOTICE
    assert "deleted" not in DOUYIN_UNREADABLE_NOTICE.split("does NOT")[0]


async def test_a_failed_download_still_supplies_the_caption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caption is injected unconditionally, so the model never claims it cannot open a link.

    The separator must not claim the clip was watched, or the model will describe footage it
    never received.
    """
    _stub_douyin(monkeypatch, download_error=DouyinError("cdn down"))

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == DOUYIN_TEXT_ONLY_SEPARATOR
    parts = blocks[1]["content"]
    assert [part["type"] for part in parts] == ["input_text"]
    assert "一段影片的說明" in parts[0]["text"]


async def test_an_oversize_clip_degrades_to_the_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip past the Files API ceiling is refused fast and answered from the caption."""
    _stub_douyin(monkeypatch, download_error=DouyinTooLargeError("over 2GB"))

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == DOUYIN_TEXT_ONLY_SEPARATOR


async def test_a_failed_upload_degrades_to_the_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    """A download that works but an upload that fails must not claim the clip was watched."""
    _stub_douyin(monkeypatch, uploads=_Uploads(fail=True))

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == DOUYIN_TEXT_ONLY_SEPARATOR


async def test_the_kill_switch_skips_the_media_but_keeps_the_caption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ingestion off the caption still rides, and the model is told it has not watched it."""
    uploads, _ = _stub_douyin(monkeypatch)

    blocks = await _build(ingest=False)

    assert blocks[0]["content"][0]["text"] == DOUYIN_TEXT_ONLY_SEPARATOR
    assert uploads.calls == []


async def test_a_non_gemini_answer_model_skips_the_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Files uri is Gemini-only, so another model gets the caption and no wasted upload."""
    uploads, _ = _stub_douyin(monkeypatch)

    blocks = await _build(gemini=False)

    assert blocks[0]["content"][0]["text"] == DOUYIN_TEXT_ONLY_SEPARATOR
    assert uploads.calls == []


async def test_the_scratch_directory_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The downloaded clip lives in a per-build temp dir that never outlives the build."""
    uploads, _ = _stub_douyin(monkeypatch)

    await _build()

    source, _mime, _name = uploads.calls[0]
    assert isinstance(source, Path)
    assert not source.exists()
    assert not source.parent.exists()


async def test_the_douyin_bound_is_never_held_twice_on_one_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One build must never acquire the shared Douyin bound twice; that self-deadlocks.

    `asyncio.Semaphore` is not reentrant, so wrapping the whole build in the bound while the
    download takes it again would hang the moment the bound is saturated. Driven at capacity 1
    so the failure is deterministic rather than a timing race, and bounded by wait_for so it
    surfaces as a red test instead of a hung suite.
    """
    monkeypatch.setattr(douyin_fetch, "DOUYIN_FETCH_CONCURRENCY", 1)
    _stub_douyin(monkeypatch)

    blocks = await asyncio.wait_for(_build(), timeout=5.0)

    assert blocks[0]["content"][0]["text"] == DOUYIN_CONTEXT_SEPARATOR


async def test_the_fetch_bound_is_released_before_the_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow upload must not keep another link waiting on the Douyin bound.

    The bound exists for Douyin's WAF; the upload talks to Google, so holding it across the
    upload would throttle unrelated links for no protective reason. Capacity 1 makes "still
    held" mean "the other link cannot start", which is exactly the property under test.
    """
    monkeypatch.setattr(douyin_fetch, "DOUYIN_FETCH_CONCURRENCY", 1)
    _stub_douyin(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    class _SlowUploads(_Uploads):
        async def __call__(self, **kwargs: object) -> dict[str, str] | None:
            """Blocks inside the upload until the test lets it finish."""
            started.set()
            await release.wait()
            return await super().__call__(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(douyin_builder, "upload_as_input_file", _SlowUploads())
    slow = asyncio.create_task(_build())
    try:
        await asyncio.wait_for(started.wait(), timeout=5.0)

        # A different link must get through while the first build sits in its upload.
        other = await asyncio.wait_for(
            build_douyin_context_messages(
                url="https://v.douyin.com/other",
                answer_model_is_gemini=True,
                gemini_client=SimpleNamespace(),
                allow_media_ingest=False,
            ),
            timeout=5.0,
        )
        assert other[0]["content"][0]["text"] == DOUYIN_TEXT_ONLY_SEPARATOR
    finally:
        release.set()
        await asyncio.wait_for(slow, timeout=5.0)
