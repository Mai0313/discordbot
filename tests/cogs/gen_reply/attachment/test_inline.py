"""Tests for rendering attachments inline."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from discordbot.cogs.gen_reply.attachment.inline import InlineRenderer

from tests.helpers.reply_fakes import FakeAttachment, _att, _png_bytes

if TYPE_CHECKING:
    import pytest
    from nextcord import Attachment

    from discordbot.typings.media import LoadedMedia


async def test_inline_renderer_drops_a_clip_without_downloading_it() -> None:
    """A clip the renderer cannot carry is dropped on its MIME type, before the download.

    The modality gate keeps such a clip from reaching the renderer through the builder, so this
    pins the renderer's own drop: a direct caller never downloads a whole clip to throw it away.
    """
    clip = FakeAttachment(filename="clip.mp4", content_type="video/mp4", payload=b"0" * 32)

    rendered = await InlineRenderer().render_file(
        attachment=cast("Attachment", clip), cache_key="clip.mp4"
    )

    assert rendered is None
    assert clip.read_count == 0


async def test_inline_renderer_drops_a_source_that_fails_to_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed fetch drops only that part, and the renderer remembers nothing of it."""

    async def fail(attachment: object) -> LoadedMedia:
        """Fails the download the way an expired CDN url does."""
        del attachment
        raise RuntimeError("cdn expired")

    monkeypatch.setattr("discordbot.cogs.gen_reply.attachment.inline.load_attachment_bytes", fail)
    renderer = InlineRenderer()

    rendered = await renderer.render_file(
        attachment=cast("Attachment", FakeAttachment(filename="notes.txt")),
        cache_key="notes.txt",
        allow_dead_cache=True,
    )

    assert rendered is None
    assert not renderer._dead_sources
    assert renderer.carries(content_type="text/plain", cache_key="notes.txt", size=None)


async def test_non_gemini_answer_model_inlines_attachments() -> None:
    """A non-Gemini answer model inlines attachments instead of using the Gemini Files API."""
    renderer = InlineRenderer()

    # Image -> base64 input_image (no Files API upload).
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
    assert ";base64," in image_url

    # Text/code file -> inlined as input_text with a filename header.
    text_rendered = await renderer.render_file(
        attachment=_att(filename="notes.txt", content_type="text/plain", payload=b"hello world"),
        cache_key="notes.txt",
    )
    assert text_rendered is not None
    text_part = text_rendered.part
    assert text_part["type"] == "input_text"
    assert "hello world" in text_part["text"]
    assert "notes.txt" in text_part["text"]

    # PDF -> inlined as base64 input_file file_data (not a Files-API file_id).
    pdf_rendered = await renderer.render_file(
        attachment=_att(
            filename="doc.pdf", content_type="application/pdf", payload=b"%PDF-1.4 fake"
        ),
        cache_key="doc.pdf",
    )
    assert pdf_rendered is not None
    pdf_part = pdf_rendered.part
    assert pdf_part["type"] == "input_file"
    assert pdf_part["file_data"].startswith("data:application/pdf;base64,")
    assert "file_id" not in pdf_part

    # Non-text, non-PDF binary -> dropped.
    binary_rendered = await renderer.render_file(
        attachment=_att(
            filename="blob.bin", content_type="application/octet-stream", payload=b"\x00\x01\xff"
        ),
        cache_key="blob.bin",
    )
    assert binary_rendered is None
