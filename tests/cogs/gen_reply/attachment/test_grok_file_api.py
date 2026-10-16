"""Tests for the xAI file uploader."""

from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio
from datetime import UTC, datetime, timedelta

from xai_sdk.proto import files_pb2

from discordbot.cogs.gen_reply.attachment.grok_file_api import (
    GROK_FILE_EXPIRY_SECONDS,
    GrokFileUploader,
)

from tests.helpers.reply_fakes import _att, _png_bytes
from tests.helpers.logfire_capture import capture_levels

if TYPE_CHECKING:
    import pytest

# The expiry the fake xAI upload reports back. Far future so a rendered part's cache TTL is
# unambiguously the provider's answer rather than the local fallback.
XAI_FAKE_EXPIRY = datetime(2099, 1, 1, tzinfo=UTC)


class FakeXAIFiles:
    """Fake xAI Files API resource that records uploads."""

    def __init__(
        self, file_id: str = "file-xai", expires_at: datetime | None = XAI_FAKE_EXPIRY
    ) -> None:
        """Initializes fake upload output fields."""
        self.file_id = file_id
        self.expires_at = expires_at
        self.upload_calls: list[tuple[str, bytes, int | None]] = []

    async def upload(
        self, file: bytes, filename: str, expires_after: int | None = None
    ) -> files_pb2.File:
        """Records an upload and returns a real `File` proto.

        The real proto rather than a stand-in, so the uploader's `HasField` / `ToDatetime`
        read of a protobuf Timestamp is exercised instead of mocked past.
        """
        self.upload_calls.append((filename, file, expires_after))
        uploaded = files_pb2.File(id=self.file_id, filename=filename, size=len(file))
        if self.expires_at is not None:
            uploaded.expires_at.FromDatetime(self.expires_at)
        return uploaded


class FakeXAIClient:
    """Fake xAI client exposing the async Files API used by GrokFileUploader."""

    def __init__(self, files: FakeXAIFiles | None = None) -> None:
        """Initializes the file resource."""
        self.files = files or FakeXAIFiles()


def _fake_grok_uploader(files: FakeXAIFiles | None = None) -> GrokFileUploader:
    """A GrokFileUploader with its lazy xAI client pre-seeded to a fake."""
    uploader = GrokFileUploader()
    uploader.__dict__["xai_client"] = FakeXAIClient(files=files)
    return uploader


async def test_grok_file_uploader_uploads_files_and_inlines_images() -> None:
    """The xAI uploader references files by id and keeps images inline."""
    files = FakeXAIFiles()
    renderer = _fake_grok_uploader(files=files)

    file_rendered = await renderer.render_file(
        attachment=_att(filename="notes.txt", content_type="text/plain", payload=b"hello world"),
        cache_key="notes.txt",
    )
    assert file_rendered is not None
    file_part = file_rendered.part
    file_expiry = file_rendered.expires_at
    assert file_part["type"] == "input_file"
    assert file_part["file_id"] == "file-xai"
    assert file_part["filename"] == "notes.txt"
    assert file_expiry == XAI_FAKE_EXPIRY

    # xAI resolves no file id for image input, so an image is inlined instead of uploaded.
    image_rendered = await renderer.render_image(
        source=_att(filename="pic.png", content_type="image/png", payload=_png_bytes()),
        cache_key="pic.png",
    )
    assert image_rendered is not None
    image_part = image_rendered.part
    assert image_part["type"] == "input_image"
    image_url = image_part["image_url"]
    assert image_url is not None
    assert image_url.startswith("data:image/")

    # The whole upload call: a filename, the bytes and a bare TTL in seconds. No `purpose` (xAI
    # never interprets it) and no `{anchor, seconds}` object, both of which were the OpenAI
    # client's shapes rather than xAI's.
    assert files.upload_calls == [("notes.txt", b"hello world", GROK_FILE_EXPIRY_SECONDS)]


async def test_grok_file_uploader_drops_failed_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    """XAI upload errors and id-less responses degrade to a dropped attachment."""
    idless = _fake_grok_uploader(files=FakeXAIFiles(file_id=""))
    assert (
        await idless._upload_file(
            cache_key="k", filename="bad.txt", data=b"x", content_type="text/plain", kind="file"
        )
        is None
    )

    boom = _fake_grok_uploader()

    async def _raise(
        file: bytes, filename: str, expires_after: int | None = None
    ) -> files_pb2.File:
        del file, filename, expires_after
        raise RuntimeError("upload failed")

    monkeypatch.setattr(boom.xai_client.files, "upload", _raise)
    assert (
        await boom._upload_file(
            cache_key="k", filename="x.txt", data=b"x", content_type="text/plain", kind="file"
        )
        is None
    )


async def test_grok_file_uploader_drops_an_upload_that_outruns_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled upload drops the attachment instead of holding the reply open.

    The bound is this call's only one: `xai-sdk` leaves client-streaming RPCs, which is what an
    upload is, uncovered by the timeout interceptors it installs for every other shape.
    """
    stalled = _fake_grok_uploader()

    async def _hang(
        file: bytes, filename: str, expires_after: int | None = None
    ) -> files_pb2.File:
        del file, filename, expires_after
        await asyncio.sleep(60)
        raise AssertionError("the deadline should have fired first")

    monkeypatch.setattr(stalled.xai_client.files, "upload", _hang)
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.grok_file_api.GROK_FILE_UPLOAD_TIMEOUT_SECONDS", 0.01
    )
    assert (
        await stalled._upload_file(
            cache_key="k", filename="x.txt", data=b"x", content_type="text/plain", kind="file"
        )
        is None
    )


async def test_grok_file_uploader_without_a_key_reports_a_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured xAI key is reported as a missing key, not as an upload failure."""
    monkeypatch.setenv(name="XAI_API_KEY", value="")
    records = capture_levels(monkeypatch=monkeypatch, levels=("warn", "error"))
    renderer = GrokFileUploader()
    assert (
        await renderer._upload_file(
            cache_key="k", filename="x.txt", data=b"x", content_type="text/plain", kind="file"
        )
        is None
    )
    assert [(level, fields["filename"]) for level, _, fields in records] == [("error", "x.txt")]


async def test_grok_file_uploader_falls_back_to_a_local_expiry() -> None:
    """A response without an expiry still bounds the render cache by the requested TTL."""
    renderer = _fake_grok_uploader(files=FakeXAIFiles(expires_at=None))
    uploaded = await renderer._upload_file(
        cache_key="k", filename="notes.txt", data=b"hello", content_type="text/plain", kind="file"
    )
    assert uploaded is not None
    assert uploaded.expires_at > datetime.now(tz=UTC) + timedelta(days=29)
