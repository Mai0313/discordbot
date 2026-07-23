"""Tests for the Bilibili-context builder that feeds linked videos to the answer model."""

import time
from types import SimpleNamespace
from typing import Any, cast
import asyncio
from pathlib import Path
import threading

from google import genai
import pytest

from discordbot.utils.downloader import VideoMetadata, DownloadResult, VideoDownloader
from discordbot.cogs._gen_reply.link_sources import bilibili as bilibili_builder
from discordbot.cogs._gen_reply.link_sources.bilibili import (
    BILIBILI_CONTEXT_SEPARATOR,
    BILIBILI_UNREADABLE_NOTICE,
    BILIBILI_TOO_LONG_SEPARATOR,
    BILIBILI_TEXT_ONLY_SEPARATOR,
    MAX_BILIBILI_DESCRIPTION_CHARS,
    MAX_BILIBILI_INGEST_DURATION_SECONDS,
    build_bilibili_context_messages,
)

from tests.helpers.casting import step_dicts

_URL = "https://www.bilibili.com/video/BV1jpK86hEc8"


def _fake_client() -> genai.Client:
    """Stands in for a Gemini client the builder never actually calls through."""
    return cast("genai.Client", SimpleNamespace())


def _metadata(
    duration: float = 63.0,
    is_live: bool = False,
    description: str = "影片簡介",
    webpage_url: str = "",
    from_playlist: bool = False,
) -> VideoMetadata:
    """Builds the parsed metadata the builder renders into its text block."""
    return VideoMetadata(
        video_id="BV1jpK86hEc8",
        title="一支B站影片",
        uploader="某個UP主",
        description=description,
        duration_seconds=duration,
        webpage_url=webpage_url,
        is_live=is_live,
        from_playlist=from_playlist,
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


def _stub_bilibili(  # noqa: PLR0913 -- one canned outcome per stage the builder can hit
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata: VideoMetadata | None = None,
    parse_error: Exception | None = None,
    download_error: Exception | None = None,
    file_size: int = 11,
    uploads: _Uploads | None = None,
) -> tuple[_Uploads, dict[str, list[str]]]:
    """Stubs the downloader and the Files API upload so no network or SDK is touched."""
    resolved = metadata or _metadata()
    recorded: dict[str, list[str]] = {"downloads": []}

    def fake_parse_metadata(self: VideoDownloader, url: str) -> VideoMetadata:
        """Returns the canned metadata, or raises the canned parse failure."""
        del url
        if parse_error is not None:
            raise parse_error
        return resolved

    def fake_download(
        self: VideoDownloader,
        url: str,
        quality: str = "best",
        dry_run: bool = False,
        stop_signal: object = None,
    ) -> DownloadResult:
        """Writes a canned clip into the builder's scratch dir, or raises."""
        del url, dry_run, stop_signal
        recorded["downloads"].append(quality)
        if download_error is not None:
            raise download_error
        path = Path(self.output_folder) / f"{resolved.video_id or 'clip'}.mp4"
        path.write_bytes(b"x" * file_size)
        return DownloadResult(title=resolved.title, filename=path)

    monkeypatch.setattr(target=VideoDownloader, name="parse_metadata", value=fake_parse_metadata)
    monkeypatch.setattr(target=VideoDownloader, name="download", value=fake_download)
    resolved_uploads = uploads or _Uploads()
    monkeypatch.setattr(bilibili_builder, "upload_as_input_file", resolved_uploads)
    return resolved_uploads, recorded


async def _build(gemini: bool = True, ingest: bool = True) -> list[dict[str, Any]]:
    """Runs the builder with the flags most tests share.

    The blocks are `EasyInputMessageParam`s, whose `content` is a union the assertions below
    index into part by part; `step_dicts` is what lets them read as plain JSON.
    """
    blocks = await build_bilibili_context_messages(
        url=_URL,
        answer_model_is_gemini=gemini,
        gemini_client=_fake_client(),
        allow_media_ingest=ingest,
    )
    return step_dicts(steps=blocks)


async def test_the_clip_is_uploaded_and_referenced_by_files_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The video rides as an input_file holding a Files API uri, never a Bilibili URL."""
    uploads, recorded = _stub_bilibili(monkeypatch)

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == BILIBILI_CONTEXT_SEPARATOR
    parts = blocks[1]["content"]
    assert parts[0]["type"] == "input_text"
    assert "一支B站影片" in parts[0]["text"]
    assert "某個UP主" in parts[0]["text"]
    assert "1:03" in parts[0]["text"]
    assert _URL in parts[0]["text"]

    media = [part for part in parts if part["type"] == "input_file"]
    assert [part["file_id"] for part in media] == ["https://files.test/BV1jpK86hEc8.mp4"]
    assert all("file_url" not in part for part in media)
    # The clip is streamed from disk and its mime is real; the extension is load-bearing on the
    # native Interactions path, which classifies a part by it.
    source, mime_type, filename = uploads.calls[0]
    assert isinstance(source, Path)
    assert mime_type == "video/mp4"
    assert filename.endswith(".mp4")
    # The ingest preset, not the expansion-grade one: the model samples frames anyway.
    assert recorded["downloads"] == [bilibili_builder.AI_INGEST_QUALITY]


async def test_metadata_failure_never_claims_the_video_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """yt-dlp failures are opaque extractor wording, so no failure may be reported as deleted.

    Region locks, member-only videos, live rooms behind a short link and transport errors all
    surface the same way; asserting a working link is dead would send the user off to re-check
    a link that is very likely fine.
    """
    _stub_bilibili(monkeypatch, parse_error=RuntimeError("ERROR: [BiliBili] something opaque"))

    blocks = await _build()

    assert len(blocks) == 1
    assert blocks[0]["content"][0]["text"] == BILIBILI_UNREADABLE_NOTICE
    assert "does NOT mean the video is deleted" in BILIBILI_UNREADABLE_NOTICE


async def test_a_too_long_video_skips_the_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """A video over the duration cap is never downloaded and gets the deterministic wording."""
    uploads, recorded = _stub_bilibili(
        monkeypatch, metadata=_metadata(duration=MAX_BILIBILI_INGEST_DURATION_SECONDS + 1)
    )

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == BILIBILI_TOO_LONG_SEPARATOR
    assert "一支B站影片" in blocks[1]["content"][0]["text"]
    assert recorded["downloads"] == []
    assert uploads.calls == []


async def test_a_live_stream_skips_the_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live stream has no finished file to watch, so it takes the too-long path."""
    _, recorded = _stub_bilibili(monkeypatch, metadata=_metadata(is_live=True))

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == BILIBILI_TOO_LONG_SEPARATOR
    assert recorded["downloads"] == []


async def test_a_failed_download_still_supplies_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """The metadata is injected unconditionally, so the model never claims it cannot open a link.

    The separator must not claim the clip was watched, or the model will describe footage it
    never received.
    """
    _stub_bilibili(monkeypatch, download_error=RuntimeError("network down"))

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == BILIBILI_TEXT_ONLY_SEPARATOR
    parts = blocks[1]["content"]
    assert [part["type"] for part in parts] == ["input_text"]
    assert "一支B站影片" in parts[0]["text"]


async def test_an_oversize_file_is_not_uploaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finished file past the Files API ceiling degrades to text with no upload attempt.

    yt-dlp offers no byte cap, so unlike Douyin the ceiling is enforced on the finished file.
    """
    monkeypatch.setattr(bilibili_builder, "FILES_API_MAX_BYTES", 4)
    uploads, _ = _stub_bilibili(monkeypatch, file_size=11)

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == BILIBILI_TEXT_ONLY_SEPARATOR
    assert uploads.calls == []


async def test_a_failed_upload_degrades_to_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A download that works but an upload that fails must not claim the clip was watched."""
    _stub_bilibili(monkeypatch, uploads=_Uploads(fail=True))

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == BILIBILI_TEXT_ONLY_SEPARATOR


async def test_the_kill_switch_skips_the_media_but_keeps_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ingestion off the metadata still rides, and the model is told it has not watched."""
    uploads, recorded = _stub_bilibili(monkeypatch)

    blocks = await _build(ingest=False)

    assert blocks[0]["content"][0]["text"] == BILIBILI_TEXT_ONLY_SEPARATOR
    assert recorded["downloads"] == []
    assert uploads.calls == []


async def test_a_missing_key_reads_the_text_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key means no client to upload with, which is a text-only read, not a failure."""
    uploads, _ = _stub_bilibili(monkeypatch)

    blocks = step_dicts(
        steps=await build_bilibili_context_messages(
            url=_URL, answer_model_is_gemini=True, gemini_client=None, allow_media_ingest=True
        )
    )

    assert blocks[0]["content"][0]["text"] == BILIBILI_TEXT_ONLY_SEPARATOR
    assert uploads.calls == []


async def test_a_non_gemini_answer_model_skips_the_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Files uri is Gemini-only, so another model gets the text and no wasted upload."""
    uploads, _ = _stub_bilibili(monkeypatch)

    blocks = await _build(gemini=False)

    assert blocks[0]["content"][0]["text"] == BILIBILI_TEXT_ONLY_SEPARATOR
    assert uploads.calls == []


async def test_the_scratch_directory_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The downloaded clip lives in a per-build temp dir that never outlives the build."""
    uploads, _ = _stub_bilibili(monkeypatch)

    await _build()

    source, _mime, _name = uploads.calls[0]
    assert isinstance(source, Path)
    assert not source.exists()
    assert not source.parent.exists()


async def test_the_description_is_trimmed_to_its_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tag-stuffed multi-thousand-character description keeps only its head."""
    _stub_bilibili(
        monkeypatch, metadata=_metadata(description="宣" * (MAX_BILIBILI_DESCRIPTION_CHARS + 500))
    )

    blocks = await _build()

    text = blocks[1]["content"][0]["text"]
    assert "宣" * MAX_BILIBILI_DESCRIPTION_CHARS in text
    assert "宣" * (MAX_BILIBILI_DESCRIPTION_CHARS + 1) not in text


async def test_a_resolved_short_link_lists_both_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A b23.tv paste keeps the pasted URL and adds the canonical page it resolved to."""
    _stub_bilibili(monkeypatch, metadata=_metadata(webpage_url=_URL))
    short_url = "https://b23.tv/abc123X"

    blocks = step_dicts(
        steps=await build_bilibili_context_messages(
            url=short_url,
            answer_model_is_gemini=True,
            gemini_client=_fake_client(),
            allow_media_ingest=True,
        )
    )

    text = blocks[1]["content"][0]["text"]
    assert short_url in text
    assert _URL in text


async def test_a_link_resolving_to_a_non_video_page_gets_the_neutral_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A b23.tv short link resolving to a space or collection must not impersonate a video.

    yt-dlp reads those pages SUCCESSFULLY as playlists, so without the canonical-URL check
    the builder would present the space's first video as "the linked video" — confident
    misinformation, the exact over-claim the notices exist to prevent.
    """
    uploads, recorded = _stub_bilibili(
        monkeypatch,
        metadata=_metadata(webpage_url="https://space.bilibili.com/672328094", from_playlist=True),
    )

    blocks = await _build()

    assert len(blocks) == 1
    assert blocks[0]["content"][0]["text"] == BILIBILI_UNREADABLE_NOTICE
    assert recorded["downloads"] == []
    assert uploads.calls == []


async def test_a_bangumi_redirected_single_video_is_still_ingested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A /video/ link Bilibili redirects to a bangumi page is still the linked video.

    yt-dlp resolves some `/video/BV…` links to a `/bangumi/play/ep…` canonical URL
    server-side; the result is a single video, not a playlist, so the non-video-page guard
    must not reject it — that would tell the user a perfectly watchable video is unreadable.
    """
    uploads, recorded = _stub_bilibili(
        monkeypatch,
        metadata=_metadata(webpage_url="https://www.bilibili.com/bangumi/play/ep288525"),
    )

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == BILIBILI_CONTEXT_SEPARATOR
    assert recorded["downloads"] == [bilibili_builder.AI_INGEST_QUALITY]
    assert len(uploads.calls) == 1


async def test_the_media_step_timeout_degrades_to_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A download slower than the internal media bound still yields the honest text block.

    This is the branch upholding the contract that the media step is bounded inside the
    builder, so a slow fetch degrades to text instead of being cancelled with nothing.
    """
    monkeypatch.setattr(bilibili_builder, "LINK_MEDIA_TIMEOUT_SECONDS", 0.05)
    uploads, _ = _stub_bilibili(monkeypatch)

    def slow_download(
        self: VideoDownloader,
        url: str,
        quality: str = "best",
        dry_run: bool = False,
        stop_signal: object = None,
    ) -> DownloadResult:
        """Outlasts the media bound so the internal timeout fires."""
        del url, quality, dry_run, stop_signal
        time.sleep(0.3)
        path = Path(self.output_folder) / "late.mp4"
        path.write_bytes(b"late")
        return DownloadResult(title="late", filename=path)

    monkeypatch.setattr(target=VideoDownloader, name="download", value=slow_download)

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == BILIBILI_TEXT_ONLY_SEPARATOR
    assert uploads.calls == []


async def test_a_raising_upload_degrades_to_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """An upload that raises (the realistic SDK failure) must not claim the clip was watched."""

    class _RaisingUploads(_Uploads):
        async def __call__(self, **kwargs: object) -> dict[str, str] | None:
            """Raises the way a genai SDK failure would."""
            del kwargs
            raise RuntimeError("upload exploded")

    _stub_bilibili(monkeypatch, uploads=_RaisingUploads())

    blocks = await _build()

    assert blocks[0]["content"][0]["text"] == BILIBILI_TEXT_ONLY_SEPARATOR


async def test_a_cancelled_build_signals_the_download_to_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the build makes the blocked yt-dlp worker abort instead of running on.

    `asyncio.to_thread` cannot cancel its worker, so without the stop signal a discarded
    build would leave a zombie download holding a thread-pool slot and writing into (or
    re-creating) a scratch dir that was already removed.
    """
    started = threading.Event()
    observed: list[bool] = []
    _stub_bilibili(monkeypatch)

    def blocking_download(
        self: VideoDownloader,
        url: str,
        quality: str = "best",
        dry_run: bool = False,
        stop_signal: threading.Event | None = None,
    ) -> DownloadResult:
        """Runs until the stop signal arrives, recording whether it ever did."""
        del url, quality, dry_run
        started.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if stop_signal is not None and stop_signal.is_set():
                observed.append(True)
                raise RuntimeError("stopped by signal")
            time.sleep(0.01)
        observed.append(False)
        raise RuntimeError("never signaled")

    monkeypatch.setattr(target=VideoDownloader, name="download", value=blocking_download)

    build_task = asyncio.create_task(_build())
    await asyncio.to_thread(started.wait, 5.0)
    build_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await build_task

    assert observed == [True]


async def test_the_bilibili_bound_is_never_held_twice_on_one_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One build must never acquire the shared Bilibili bound twice; that self-deadlocks.

    `asyncio.Semaphore` is not reentrant: the metadata probe deliberately stays off the
    bound, and only the download takes it, so re-wrapping the whole build in it would hang
    the moment the bound is saturated. Driven at capacity 1 so the failure is deterministic
    rather than a timing race, and bounded by wait_for so it surfaces as a red test instead
    of a hung suite.
    """
    monkeypatch.setattr(bilibili_builder, "BILIBILI_FETCH_CONCURRENCY", 1)
    _stub_bilibili(monkeypatch)

    blocks = await asyncio.wait_for(_build(), timeout=5.0)

    assert blocks[0]["content"][0]["text"] == BILIBILI_CONTEXT_SEPARATOR


async def test_the_fetch_bound_is_released_before_the_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow upload must not keep another link waiting on the Bilibili bound.

    The bound exists for the host's disk and bandwidth; the upload talks to Google, so holding
    it across the upload would throttle unrelated links for no protective reason. Capacity 1
    makes "still held" mean "the other link cannot start", which is exactly the property under
    test.
    """
    monkeypatch.setattr(bilibili_builder, "BILIBILI_FETCH_CONCURRENCY", 1)
    _stub_bilibili(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    class _SlowUploads(_Uploads):
        async def __call__(
            self,
            *,
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

    monkeypatch.setattr(bilibili_builder, "upload_as_input_file", _SlowUploads())
    slow = asyncio.create_task(_build())
    try:
        await asyncio.wait_for(started.wait(), timeout=5.0)

        # A different link must get through while the first build sits in its upload.
        other = step_dicts(
            steps=await asyncio.wait_for(
                build_bilibili_context_messages(
                    url="https://www.bilibili.com/video/av170001",
                    answer_model_is_gemini=True,
                    gemini_client=_fake_client(),
                    allow_media_ingest=False,
                ),
                timeout=5.0,
            )
        )
        assert other[0]["content"][0]["text"] == BILIBILI_TEXT_ONLY_SEPARATOR
    finally:
        release.set()
        await asyncio.wait_for(slow, timeout=5.0)
