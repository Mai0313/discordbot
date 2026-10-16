"""Tests for uploading attachments through the Gemini Files API."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from datetime import UTC, datetime

from google.genai.types import FileState

from discordbot.typings.media import LoadedMedia, UploadedFile
from discordbot.cogs.gen_reply.attachment.gemini_file_api import PendingUpload, GeminiFileUploader

from tests.helpers.gen_reply import FakeGeminiFiles, skip_files_api_poll_waits
from tests.helpers.reply_fakes import _stalled, _fake_uploader

if TYPE_CHECKING:
    from io import BytesIO

    import pytest


def _jump_files_api_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes every clock read of the Files API upload jump well past its activation bound.

    Auto-advancing rather than a hand-counted list, so the deadline trips however many
    `monotonic()` calls the upload path makes, latency logging among them.
    """
    clock = {"now": 0.0}

    def monotonic() -> float:
        """Advances fifty seconds per read."""
        clock["now"] += 50.0
        return clock["now"]

    monkeypatch.setattr("discordbot.cogs.gen_reply.files_api.time.monotonic", monotonic)


async def test_upload_file_polls_active_and_drops_unready_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies the upload polls to ACTIVE and drops files that never become usable."""
    skip_files_api_poll_waits(monkeypatch=monkeypatch)

    def _uploader(files: FakeGeminiFiles) -> GeminiFileUploader:
        return _fake_uploader(files=files)

    # PROCESSING for two polls, then ACTIVE: the file URI and its expiry are returned.
    active = _uploader(FakeGeminiFiles(processing_rounds=2))
    uploaded = await active._upload_or_pend(
        filename="doc.pdf", data=b"x", content_type="application/pdf"
    )
    assert uploaded == UploadedFile(
        uri="https://files.test/doc.pdf", expires_at=datetime(2099, 1, 1, tzinfo=UTC)
    )

    # Terminal non-active state: the file is dropped.
    failed = _uploader(FakeGeminiFiles(final_state=FileState.FAILED))
    assert (
        await failed._upload_or_pend(filename="bad.pdf", data=b"x", content_type="application/pdf")
        is None
    )

    # Never leaves PROCESSING within the bound: the timeout drops the file.
    _jump_files_api_clock(monkeypatch=monkeypatch)
    stuck = _uploader(FakeGeminiFiles(processing_rounds=99))
    pending = await stuck._upload_or_pend(filename="slow.mp4", data=b"x", content_type="video/mp4")
    assert isinstance(pending, PendingUpload)
    assert pending.name == "slow.mp4"
    assert pending.uri == "https://files.test/slow.mp4"

    # Upload raises: the file is dropped instead of aborting the reply.
    async def _raise(file: BytesIO, config: dict[str, str]) -> SimpleNamespace:
        del file, config
        raise RuntimeError("upload failed")

    boom_files = FakeGeminiFiles()
    boom = _uploader(boom_files)
    monkeypatch.setattr(boom_files, "upload", _raise)
    assert (
        await boom._upload_or_pend(filename="x.txt", data=b"x", content_type="text/plain") is None
    )


async def test_resolve_file_upload_recovers_pending_on_next_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out upload is cached as pending and re-polled, not re-uploaded, next time."""
    skip_files_api_poll_waits(monkeypatch=monkeypatch)
    # The first reference times out to PENDING.
    _jump_files_api_clock(monkeypatch=monkeypatch)

    files = FakeGeminiFiles(processing_rounds=99)
    uploader = _fake_uploader(files=files)

    load_calls = 0

    async def _load() -> LoadedMedia:
        nonlocal load_calls
        load_calls += 1
        return LoadedMedia(data=b"x", mime_type="video/mp4")

    # First reference times out while still PROCESSING: dropped for now, cached as pending.
    first = await uploader._resolve_file_upload(
        cache_key="vid", filename="v.mp4", load_data=_load, kind="file"
    )
    assert first is None
    assert "vid" in uploader._pending_uploads
    assert files.upload_calls == [("v.mp4", "video/mp4")]
    assert load_calls == 1  # downloaded once for the fresh upload

    # The file finished processing in the background; the next reference re-polls the same
    # file once and adopts it, without re-downloading or re-uploading the bytes.
    async def _active_get(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            uri=f"https://files.test/{name}",
            state=FileState.ACTIVE,
            error=None,
            expiration_time=datetime(2099, 1, 1, tzinfo=UTC),
        )

    monkeypatch.setattr(files, "get", _active_get)
    second = await uploader._resolve_file_upload(
        cache_key="vid", filename="v.mp4", load_data=_load, kind="file"
    )
    assert second == UploadedFile(
        uri="https://files.test/v.mp4", expires_at=datetime(2099, 1, 1, tzinfo=UTC)
    )
    assert "vid" not in uploader._pending_uploads
    assert files.upload_calls == [("v.mp4", "video/mp4")]  # no second upload
    assert load_calls == 1  # adopt path did not re-download the source


async def test_a_stalled_attachment_upload_drops_only_that_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An upload that never returns is given up on like a failed one, not waited on forever."""
    files = FakeGeminiFiles()
    monkeypatch.setattr(files, "upload", _stalled(call=files.upload))
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.gemini_file_api.ATTACHMENT_UPLOAD_TIMEOUT_SECONDS",
        0.01,
    )

    uploaded = await _fake_uploader(files=files)._upload_or_pend(
        filename="clip.mp4", data=b"x", content_type="video/mp4"
    )

    assert uploaded is None


async def test_a_stalled_activation_read_drops_only_that_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A poll read that never returns fails the poll, however far off the poll's own bound is."""
    skip_files_api_poll_waits(monkeypatch=monkeypatch)
    files = FakeGeminiFiles(processing_rounds=1)
    monkeypatch.setattr(files, "get", _stalled(call=files.get))
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.gemini_file_api.FILES_API_READ_TIMEOUT_SECONDS", 0.01
    )

    uploaded = await _fake_uploader(files=files)._upload_or_pend(
        filename="clip.mp4", data=b"x", content_type="video/mp4"
    )

    assert uploaded is None


async def test_a_stalled_pending_repoll_falls_back_to_a_fresh_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-poll read that never returns costs a re-upload, as a failed re-poll does."""
    files = FakeGeminiFiles()
    uploader = _fake_uploader(files=files)
    uploader._pending_uploads["vid"] = PendingUpload(
        name="files/vid",
        uri="https://files.test/files/vid",
        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
    )
    monkeypatch.setattr(files, "get", _stalled(call=files.get))
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.gemini_file_api.FILES_API_READ_TIMEOUT_SECONDS", 0.01
    )

    async def _load() -> LoadedMedia:
        return LoadedMedia(data=b"x", mime_type="video/mp4")

    uploaded = await uploader._resolve_file_upload(
        cache_key="vid", filename="v.mp4", load_data=_load, kind="file"
    )

    assert uploaded == UploadedFile(
        uri="https://files.test/v.mp4", expires_at=datetime(2099, 1, 1, tzinfo=UTC)
    )
    assert files.upload_calls == [("v.mp4", "video/mp4")]
    assert "vid" not in uploader._pending_uploads
