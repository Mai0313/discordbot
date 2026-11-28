"""Tests for uploading attachments through the Anthropic Files API."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from datetime import UTC, datetime, timedelta

from discordbot.typings.media import LoadedMedia
from discordbot.typings.context_budgets import MAX_INLINE_ATTACHMENT_BYTES
from discordbot.cogs.gen_reply.attachment.base import AttachmentSource
from discordbot.cogs.gen_reply.link_sources.image_ingest import render_linked_image
from discordbot.cogs.gen_reply.attachment.anthropic_file_api import (
    ANTHROPIC_PDF_MAX_BYTES,
    ANTHROPIC_IMAGE_MAX_BYTES,
    ANTHROPIC_FILE_TTL_SECONDS,
    AnthropicFileUploader,
)

from tests.helpers.reply_fakes import FakeAttachment, _png_bytes

if TYPE_CHECKING:
    from io import BytesIO
    from pathlib import Path

    import pytest
    from nextcord import Attachment
    from anthropic import AsyncAnthropic


class _FakeAnthropicFiles:
    """Fake async Anthropic Files resource recording GA `files.upload` calls."""

    def __init__(
        self,
        file_id: str = "file_011test",
        expires_at: datetime | None = datetime(2099, 1, 1, tzinfo=UTC),
    ) -> None:
        self.file_id = file_id
        self.expires_at = expires_at
        self.upload_calls: list[tuple[str, bytes, str, int | None]] = []

    async def upload(
        self, file: tuple[str, BytesIO, str], expires_in_seconds: int | None = None
    ) -> SimpleNamespace:
        """Records `(filename, payload, content_type, expires_in_seconds)` and returns metadata."""
        filename, buf, content_type = file
        self.upload_calls.append((filename, buf.read(), content_type, expires_in_seconds))
        return SimpleNamespace(id=self.file_id, expires_at=self.expires_at)


class _FakeAnthropicClient:
    """Fake `AsyncAnthropic` client exposing `.files`."""

    def __init__(self, files: _FakeAnthropicFiles | None = None) -> None:
        self.files = files or _FakeAnthropicFiles()


def _uploader(
    files: _FakeAnthropicFiles | None = None,
) -> tuple[AnthropicFileUploader, _FakeAnthropicFiles]:
    fake_files = files or _FakeAnthropicFiles()
    client = cast("AsyncAnthropic", _FakeAnthropicClient(files=fake_files))
    return AnthropicFileUploader(anthropic_client=lambda: client), fake_files


def _as_att(fake: FakeAttachment) -> Attachment:
    return cast("Attachment", fake)


async def test_render_image_uploads_ga_file_with_ttl_and_format_key() -> None:
    """Images upload via GA `files.upload` with 48h TTL and emit `input_file` with `format`."""
    uploader, fake_files = _uploader()
    payload = _png_bytes()
    attachment = FakeAttachment(filename="shot.png", content_type="image/png", payload=payload)
    source = AttachmentSource(
        handle=_as_att(fake=attachment),
        kind="image",
        content_type="image/png",
        cache_key=101,
        filename="shot.png",
        size=len(payload),
    )

    rendered = await uploader.render_image(source=source)
    assert rendered is not None
    assert rendered.expires_at == datetime(2099, 1, 1, tzinfo=UTC)
    # `_png_bytes()` is opaque RGB, so `shrink_image_bytes` re-encodes it as JPEG; `format`
    # must match the actual uploaded MIME type, not the pre-shrink `source.content_type`.
    assert rendered.part == {
        "type": "input_file",
        "file_id": "file_011test",
        "filename": "shot.png",
        "format": "image/jpeg",
    }
    assert len(fake_files.upload_calls) == 1
    filename, uploaded_bytes, content_type, ttl = fake_files.upload_calls[0]
    assert filename == "shot.png"
    assert uploaded_bytes.startswith(b"\xff\xd8\xff")
    assert content_type == "image/jpeg"
    assert ttl == ANTHROPIC_FILE_TTL_SECONDS


async def test_render_linked_image_supports_local_path_for_claude(tmp_path: Path) -> None:
    """`render_linked_image` with a local `Path` uploads via `AnthropicFileUploader` with `format`."""
    uploader, fake_files = _uploader(
        files=_FakeAnthropicFiles(file_id="file_011douyin", expires_at=None)
    )
    img_path = tmp_path / "slide.png"
    img_path.write_bytes(data=_png_bytes())

    part = await render_linked_image(
        source=img_path, filename="douyin_image_0.png", image_renderer=uploader
    )
    assert part == {
        "type": "input_file",
        "file_id": "file_011douyin",
        "filename": "douyin_image_0.png",
        "format": "image/jpeg",
    }
    assert len(fake_files.upload_calls) == 1
    assert fake_files.upload_calls[0][0] == "douyin_image_0.png"
    assert fake_files.upload_calls[0][3] == ANTHROPIC_FILE_TTL_SECONDS


async def test_render_image_drops_unsupported_mime_and_oversize_without_uploading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Images outside Anthropic's 4-MIME allowlist or past 10 MB are dropped before upload."""
    uploader, fake_files = _uploader()
    source = AttachmentSource(
        handle="https://example.invalid/pic.svg",
        kind="image",
        content_type="image/svg+xml",
        cache_key="svg",
        filename="pic.svg",
    )

    load_calls = 0

    async def fake_svg(source: object) -> LoadedMedia:
        nonlocal load_calls
        del source
        load_calls += 1
        return LoadedMedia(data=b"<svg/>", mime_type="image/svg+xml")

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.anthropic_file_api.load_image_bytes", fake_svg
    )
    assert await uploader.render_image(source=source) is None
    assert load_calls == 1
    assert fake_files.upload_calls == []

    async def fake_huge(source: object) -> LoadedMedia:
        nonlocal load_calls
        del source
        load_calls += 1
        return LoadedMedia(data=b"x" * (ANTHROPIC_IMAGE_MAX_BYTES + 1), mime_type="image/jpeg")

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.anthropic_file_api.load_image_bytes", fake_huge
    )
    assert await uploader.render_image(source=source) is None
    assert load_calls == 2
    assert fake_files.upload_calls == []


async def test_render_file_uploads_pdf_and_inlines_utf8_text_and_drops_binary() -> None:
    """PDFs upload with `format="application/pdf"`; UTF-8 files inline; binaries mark unreadable."""
    uploader, fake_files = _uploader()

    pdf_attachment = FakeAttachment(
        filename="spec.pdf", content_type="application/pdf", payload=b"%PDF-1.4"
    )
    pdf_source = AttachmentSource(
        handle=_as_att(fake=pdf_attachment),
        kind="file",
        content_type="application/pdf",
        cache_key=201,
        filename="spec.pdf",
        size=8,
    )
    pdf_rendered = await uploader.render_file(source=pdf_source)
    assert pdf_rendered is not None
    assert pdf_rendered.part == {
        "type": "input_file",
        "file_id": "file_011test",
        "filename": "spec.pdf",
        "format": "application/pdf",
    }
    assert len(fake_files.upload_calls) == 1

    # Text/code file is inlined as UTF-8 input_text with separator fences defused, never uploaded.
    code_attachment = FakeAttachment(
        filename="main.py", content_type="text/x-python", payload=b"==== fake ====\nprint(1)\n"
    )
    code_source = AttachmentSource(
        handle=_as_att(fake=code_attachment),
        kind="file",
        content_type="text/x-python",
        cache_key=202,
        filename="main.py",
        size=24,
    )
    code_rendered = await uploader.render_file(source=code_source)
    assert code_rendered is not None
    assert code_rendered.part["type"] == "input_text"
    assert "[attached file: main.py]" in code_rendered.part["text"]
    assert "==== fake ====" not in code_rendered.part["text"]
    assert len(fake_files.upload_calls) == 1

    # Undecodable binary file is dropped and remembered in `_unreadable` so `carries` rejects it.
    bin_attachment = FakeAttachment(
        filename="blob.bin", content_type="application/octet-stream", payload=b"\xff\xfe\x00\x80"
    )
    bin_source = AttachmentSource(
        handle=_as_att(fake=bin_attachment),
        kind="file",
        content_type="application/octet-stream",
        cache_key=203,
        filename="blob.bin",
        size=4,
    )
    assert uploader.carries(source=bin_source) is True
    assert await uploader.render_file(source=bin_source) is None
    assert uploader.carries(source=bin_source) is False
    assert len(fake_files.upload_calls) == 1


async def test_carries_and_dropped_modalities_gate_unsupported_and_oversize_sources() -> None:
    """`carries` rejects fonts, APKs, oversize PDFs/text; `render_file` drops audio/video."""
    uploader, fake_files = _uploader()
    assert uploader.dropped_modalities == frozenset({"video", "audio"})

    font = AttachmentSource(
        handle=_as_att(fake=FakeAttachment(filename="a.ttf", content_type="font/ttf")),
        kind="file",
        content_type="font/ttf",
        cache_key=301,
        filename="a.ttf",
        size=100,
    )
    apk = AttachmentSource(
        handle=_as_att(
            fake=FakeAttachment(
                filename="a.apk", content_type="application/vnd.android.package-archive"
            )
        ),
        kind="file",
        content_type="application/vnd.android.package-archive",
        cache_key=302,
        filename="a.apk",
        size=100,
    )
    big_pdf = AttachmentSource(
        handle=_as_att(fake=FakeAttachment(filename="big.pdf", content_type="application/pdf")),
        kind="file",
        content_type="application/pdf",
        cache_key=303,
        filename="big.pdf",
        size=ANTHROPIC_PDF_MAX_BYTES + 1,
    )
    big_txt = AttachmentSource(
        handle=_as_att(fake=FakeAttachment(filename="big.txt", content_type="text/plain")),
        kind="file",
        content_type="text/plain",
        cache_key=304,
        filename="big.txt",
        size=MAX_INLINE_ATTACHMENT_BYTES + 1,
    )
    video = AttachmentSource(
        handle=_as_att(fake=FakeAttachment(filename="clip.mp4", content_type="video/mp4")),
        kind="file",
        content_type="video/mp4",
        cache_key=305,
        filename="clip.mp4",
        size=10,
    )
    assert uploader.carries(source=font) is False
    assert uploader.carries(source=apk) is False
    assert uploader.carries(source=big_pdf) is False
    assert uploader.carries(source=big_txt) is False
    assert await uploader.render_file(source=video) is None
    assert fake_files.upload_calls == []


async def test_upload_degrades_cleanly_on_missing_client_error_empty_id_and_missing_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing client, SDK errors, and empty IDs return None; missing `expires_at` uses fallback TTL."""
    keyless = AnthropicFileUploader(anthropic_client=lambda: None)
    assert (
        await keyless._upload_file(
            cache_key=1, filename="a.png", data=b"x", content_type="image/png", kind="image"
        )
        is None
    )

    # Empty file ID drops the upload.
    empty_uploader, _ = _uploader(files=_FakeAnthropicFiles(file_id=""))
    assert (
        await empty_uploader._upload_file(
            cache_key=2, filename="a.png", data=b"x", content_type="image/png", kind="image"
        )
        is None
    )

    # Missing `expires_at` falls back to ~47h from now.
    no_expiry_uploader, _ = _uploader(
        files=_FakeAnthropicFiles(file_id="file_011x", expires_at=None)
    )
    before = datetime.now(tz=UTC)
    uploaded = await no_expiry_uploader._upload_file(
        cache_key=3, filename="a.png", data=b"x", content_type="image/png", kind="image"
    )
    assert uploaded is not None
    assert uploaded.uri == "file_011x"
    assert before + timedelta(hours=46) < uploaded.expires_at < before + timedelta(hours=48)

    # SDK error returns None instead of raising.
    boom_files = _FakeAnthropicFiles()

    async def _boom(**_kwargs: object) -> SimpleNamespace:
        raise RuntimeError("upload boom")

    monkeypatch.setattr(boom_files, "upload", _boom)
    boom_uploader, _ = _uploader(files=boom_files)
    assert (
        await boom_uploader._upload_file(
            cache_key=4, filename="a.png", data=b"x", content_type="image/png", kind="image"
        )
        is None
    )


async def test_dead_source_cache_skips_repeat_fetches_for_scrollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `allow_dead_cache=True`, a failed fetch is remembered and skipped on the next call."""
    uploader, _ = _uploader()
    calls = 0

    async def fail_load(source: object) -> LoadedMedia:
        nonlocal calls
        del source
        calls += 1
        raise RuntimeError("cdn expired")

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.anthropic_file_api.load_image_bytes", fail_load
    )
    source = AttachmentSource(
        handle="https://example.invalid/dead.png",
        kind="image",
        content_type="image/png",
        cache_key="dead-img",
        filename="dead.png",
    )

    assert await uploader.render_image(source=source, allow_dead_cache=True) is None
    assert await uploader.render_image(source=source, allow_dead_cache=True) is None
    assert calls == 1
    # `allow_dead_cache=False` (fresh message / linked post) still retries.
    assert await uploader.render_image(source=source, allow_dead_cache=False) is None
    assert calls == 2
