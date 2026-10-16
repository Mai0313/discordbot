"""Tests for the Douyin-context builder that feeds linked posts to the answer model."""

from typing import Unpack
import asyncio
from pathlib import Path
import threading

import pytest
from openai.types.responses import EasyInputMessageParam

from discordbot.services.platforms import douyin as douyin_fetch
from discordbot.typings.context_budgets import MAX_DOUYIN_INGEST_IMAGES
from discordbot.services.platforms.douyin import (
    DouyinError,
    DouyinDownload,
    DouyinMetadata,
    DouyinBlockedError,
    DouyinTooLargeError,
    DouyinTransferError,
    DouyinUnavailableError,
)
from discordbot.cogs.gen_reply.speculation import run_until_deadline
from discordbot.cogs.gen_reply.link_sources import douyin as douyin_builder
from discordbot.cogs.gen_reply.link_sources.douyin import (
    DOUYIN_BLOCKED_NOTICE,
    DOUYIN_TRANSFER_NOTICE,
    DOUYIN_CONTEXT_SEPARATOR,
    DOUYIN_UNREADABLE_NOTICE,
    DOUYIN_UNAVAILABLE_NOTICE,
    DOUYIN_TEXT_ONLY_SEPARATOR,
    build_douyin_context_messages,
)

from tests.helpers.casting import make_stub_gemini_client
from tests.helpers.link_sources import (
    FakeUploads,
    StubDouyinOptions,
    StubDouyinDownloader,
    block_body,
    block_parts,
    block_separator,
    link_build_deadline,
    stub_douyin_downloads,
    race_every_scratch_teardown,
)

_URL = "https://v.douyin.com/abc123"


def _post(is_photo: bool = False, images: int = 0) -> DouyinMetadata:
    """Builds the parsed metadata the builder renders into its text block."""
    return DouyinMetadata(
        aweme_id="777",
        title="一段影片的說明",
        author_name="某個作者",
        is_photo=is_photo,
        video_id="" if is_photo else "vid",
        image_urls=[f"https://cdn.test/{index}.jpg" for index in range(images)],
    )


def _stub_douyin(
    monkeypatch: pytest.MonkeyPatch,
    uploads: FakeUploads | None = None,
    **canned: Unpack[StubDouyinOptions],
) -> tuple[FakeUploads, list[StubDouyinDownloader]]:
    """Stubs the downloader and the Files API upload so no network or SDK is touched.

    The post defaults to `_post()`, whose clip is `777.mp4`.
    """
    canned.setdefault("post", _post())
    canned.setdefault("files", [("777.mp4", b"media-bytes")])
    made: list[StubDouyinDownloader] = []
    monkeypatch.setattr(
        target=douyin_builder,
        name="DouyinDownloader",
        value=stub_douyin_downloads(made=made, **canned),
    )
    resolved_uploads = uploads or FakeUploads()
    monkeypatch.setattr(douyin_builder, "upload_as_input_file", resolved_uploads)
    return resolved_uploads, made


async def _build(gemini: bool = True, ingest: bool = True) -> list[EasyInputMessageParam]:
    """Runs the builder with the flags most tests share."""
    return await build_douyin_context_messages(
        url=_URL,
        answer_model_is_gemini=gemini,
        gemini_client=make_stub_gemini_client(),
        allow_media_ingest=ingest,
        deadline=link_build_deadline(),
    )


async def test_the_clip_is_uploaded_and_referenced_by_files_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The video rides as an input_file holding a Files API uri, never a Douyin URL.

    A Douyin CDN url is unusable to both backends anyway (the play endpoint needs a mobile
    User-Agent), so the upload is the only shape that works, not merely the tidier one.
    """
    uploads, made = _stub_douyin(monkeypatch)

    blocks = await _build()

    assert block_separator(blocks=blocks) == DOUYIN_CONTEXT_SEPARATOR
    parts = block_parts(blocks=blocks)
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
    # A fail-fast Content-Length guard, not a quality lever: the resolution is chosen separately
    # by `quality=AI_INGEST_QUALITY`, deliberately below what the human-facing expansion posts.
    (call,) = made[-1].download_calls
    assert call["max_bytes"] == douyin_builder.FILES_API_MAX_BYTES
    assert call["quality"] == douyin_builder.AI_INGEST_QUALITY


async def test_the_parsed_post_is_handed_to_the_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caption is parsed once and reused, so the post is never resolved twice."""
    post = _post()
    _, made = _stub_douyin(monkeypatch, post=post)

    await _build()

    (call,) = made[-1].download_calls
    assert call["post"] is post


async def test_a_gallery_is_capped_and_uploaded_as_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """A photo post rides as image parts, capped so a huge gallery cannot blow the budget."""
    uploads, made = _stub_douyin(
        monkeypatch,
        post=_post(is_photo=True, images=20),
        files=[(f"777_{index}.jpg", b"media-bytes") for index in range(20)],
    )

    blocks = await _build()

    (call,) = made[-1].download_calls
    assert call["max_images"] == MAX_DOUYIN_INGEST_IMAGES
    media = [part for part in block_parts(blocks=blocks) if part["type"] == "input_file"]
    assert len(media) == MAX_DOUYIN_INGEST_IMAGES
    assert all(mime == "image/jpeg" for _source, mime, _name in uploads.calls)
    assert "photo gallery" in block_body(blocks=blocks)


async def test_a_blocked_read_is_never_reported_as_a_missing_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A WAF block is retryable and the link is fine, so it gets its own notice wording.

    Reporting it as a deleted post is the worst failure this feature can produce.
    """
    _stub_douyin(monkeypatch, parse_error=DouyinBlockedError("bot wall"))

    blocks = await _build()

    assert len(blocks) == 1
    assert block_separator(blocks=blocks) == DOUYIN_BLOCKED_NOTICE
    assert block_separator(blocks=blocks) != DOUYIN_UNAVAILABLE_NOTICE


async def test_a_read_that_did_not_go_through_is_never_called_a_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled or dropped read is as retryable as a block, but nothing refused it.

    Telling the user Douyin blocked them sends them off to wait out a wall that was never
    there, and the unreadable notice would drop the advice to try again.
    """
    _stub_douyin(monkeypatch, parse_error=DouyinTransferError("read timed out"))

    blocks = await _build()

    assert len(blocks) == 1
    assert block_separator(blocks=blocks) == DOUYIN_TRANSFER_NOTICE
    assert "block" not in DOUYIN_TRANSFER_NOTICE
    assert "deleted" not in DOUYIN_TRANSFER_NOTICE.split("NOT")[0]


async def test_a_deleted_post_gets_the_unavailable_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    """A post Douyin refuses to serve is reported as deleted or private."""
    _stub_douyin(monkeypatch, parse_error=DouyinUnavailableError("filtered"))

    blocks = await _build()

    assert block_separator(blocks=blocks) == DOUYIN_UNAVAILABLE_NOTICE


async def test_any_other_failure_never_claims_the_post_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure that says nothing about the post must not be reported as a deleted one.

    An unresolvable link, a 403 or a 404, or a changed payload shape all surface as a bare
    `DouyinError`; asserting the post is gone would send the user off to re-check a link that
    is very likely fine. Only Douyin explicitly filtering the post out earns that wording.
    """
    _stub_douyin(monkeypatch, parse_error=DouyinError("could not find a post id"))

    blocks = await _build()

    assert block_separator(blocks=blocks) == DOUYIN_UNREADABLE_NOTICE
    assert block_separator(blocks=blocks) != DOUYIN_UNAVAILABLE_NOTICE
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

    assert block_separator(blocks=blocks) == DOUYIN_TEXT_ONLY_SEPARATOR
    parts = block_parts(blocks=blocks)
    assert [part["type"] for part in parts] == ["input_text"]
    assert "一段影片的說明" in parts[0]["text"]


async def test_an_oversize_clip_degrades_to_the_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip past the Files API ceiling is refused fast and answered from the caption."""
    _stub_douyin(monkeypatch, download_error=DouyinTooLargeError("over 2GB"))

    blocks = await _build()

    assert block_separator(blocks=blocks) == DOUYIN_TEXT_ONLY_SEPARATOR


async def test_a_failed_upload_degrades_to_the_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    """A download that works but an upload that fails must not claim the clip was watched."""
    _stub_douyin(monkeypatch, uploads=FakeUploads(fail=True))

    blocks = await _build()

    assert block_separator(blocks=blocks) == DOUYIN_TEXT_ONLY_SEPARATOR


async def test_the_kill_switch_skips_the_media_but_keeps_the_caption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ingestion off the caption still rides, and the model is told it has not watched it."""
    uploads, _ = _stub_douyin(monkeypatch)

    blocks = await _build(ingest=False)

    assert block_separator(blocks=blocks) == DOUYIN_TEXT_ONLY_SEPARATOR
    assert uploads.calls == []


async def test_a_missing_key_reads_the_caption_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key means no client to upload with, which is a text-only read, not a failure."""
    uploads, _ = _stub_douyin(monkeypatch)

    blocks = await build_douyin_context_messages(
        url=_URL,
        answer_model_is_gemini=True,
        gemini_client=None,
        allow_media_ingest=True,
        deadline=link_build_deadline(),
    )

    assert block_separator(blocks=blocks) == DOUYIN_TEXT_ONLY_SEPARATOR
    assert uploads.calls == []


async def test_a_non_gemini_answer_model_skips_the_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Files uri is Gemini-only, so another model gets the caption and no wasted upload."""
    uploads, _ = _stub_douyin(monkeypatch)

    blocks = await _build(gemini=False)

    assert block_separator(blocks=blocks) == DOUYIN_TEXT_ONLY_SEPARATOR
    assert uploads.calls == []


async def test_the_scratch_directory_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The downloaded clip lives in a per-build temp dir that never outlives the build."""
    uploads, _ = _stub_douyin(monkeypatch)

    await _build()

    source, _mime, _name = uploads.calls[0]
    assert isinstance(source, Path)
    assert not source.exists()
    assert not source.parent.exists()


async def test_a_raced_scratch_teardown_keeps_the_clip_the_build_already_uploaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing removal must not throw away media the model was about to watch.

    The builder returns its parts from inside the scratch directory's own `with`, so a raised
    cleanup would discard a finished result: the clip downloaded and uploaded, and the block
    still the text-only one, telling the model it had not watched the post it was holding.
    """
    _stub_douyin(monkeypatch)
    removed = race_every_scratch_teardown(monkeypatch)

    blocks = await _build()

    assert removed  # the teardown really ran and really failed
    assert block_separator(blocks=blocks) == DOUYIN_CONTEXT_SEPARATOR
    media = [part for part in block_parts(blocks=blocks) if part["type"] == "input_file"]
    assert [part["file_id"] for part in media] == ["https://files.test/777.mp4"]


async def test_a_raced_scratch_teardown_still_lets_the_post_route_deadline_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing removal must not swallow the grace the pipeline is timing the build with.

    `speculation.py::run_until_deadline` is `asyncio.wait_for`, which raises `TimeoutError`
    only when the builder lets the `CancelledError` out; a builder that returns a value
    instead has THAT value returned (measured on 3.12.13). A raised cleanup replaces the
    cancellation with an `OSError` the builder's own broad handler then absorbs, so the
    expired build used to come back as ordinary caption-only blocks and `pipeline.py` never
    reached the branch injecting `DOUYIN_TIMEOUT_NOTICE`: a link that never answered was
    reported to the model as one that answered without its media.
    """
    _stub_douyin(monkeypatch)
    removed = race_every_scratch_teardown(monkeypatch)
    release = threading.Event()

    class _StallingDownloader(StubDouyinDownloader):
        """Reads the post, then stalls its download the way a stalling CDN read does."""

        def download(self, *args: object, **kwargs: object) -> DouyinDownload:
            """Blocks the worker thread until the test releases it.

            Released by the test rather than slept out: `asyncio.to_thread` cannot cancel this,
            so a fixed sleep would be charged to the event loop's own shutdown join at teardown.
            """
            del args, kwargs
            release.wait(timeout=5.0)  # a backstop, so a bug here cannot hang the suite
            raise AssertionError("should have been abandoned")

    monkeypatch.setattr(target=douyin_builder, name="DouyinDownloader", value=_StallingDownloader)

    try:
        with pytest.raises(TimeoutError):
            await run_until_deadline(
                awaitable=build_douyin_context_messages(
                    url=_URL,
                    answer_model_is_gemini=True,
                    gemini_client=make_stub_gemini_client(),
                    allow_media_ingest=True,
                    # Later than the deadline enforced below, so the builder's own media bound
                    # cannot fire first: the backstop is what this test is about.
                    deadline=link_build_deadline(),
                ),
                # Far above the microseconds the metadata probe and the scratch dir cost, so a
                # loaded runner still expires with the worker inside the directory, and far
                # below the download it is abandoning.
                deadline=asyncio.get_running_loop().time() + 0.5,
            )
    finally:
        release.set()

    assert removed  # the teardown really ran and really failed


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

    assert block_separator(blocks=blocks) == DOUYIN_CONTEXT_SEPARATOR


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

    class _SlowUploads(FakeUploads):
        async def __call__(
            self,
            client: object,
            source: object,
            mime_type: str,
            filename: str,
            timeout_seconds: float,
        ) -> dict[str, str] | None:
            """Blocks inside the upload until the test lets it finish."""
            started.set()
            await release.wait()
            return await super().__call__(
                client=client,
                source=source,
                mime_type=mime_type,
                filename=filename,
                timeout_seconds=timeout_seconds,
            )

    monkeypatch.setattr(douyin_builder, "upload_as_input_file", _SlowUploads())
    slow = asyncio.create_task(_build())
    try:
        await asyncio.wait_for(started.wait(), timeout=5.0)

        # A different link must get through while the first build sits in its upload.
        other = await asyncio.wait_for(
            build_douyin_context_messages(
                url="https://v.douyin.com/other",
                answer_model_is_gemini=True,
                gemini_client=make_stub_gemini_client(),
                allow_media_ingest=False,
                deadline=link_build_deadline(),
            ),
            timeout=5.0,
        )
        assert block_separator(blocks=other) == DOUYIN_TEXT_ONLY_SEPARATOR
    finally:
        release.set()
        await asyncio.wait_for(slow, timeout=5.0)
