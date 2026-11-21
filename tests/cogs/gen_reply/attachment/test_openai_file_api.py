"""Tests for the OpenAI file uploader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.helpers.reply_fakes import FakeOpenAIFiles, _att, _png_bytes, _fake_openai_uploader

if TYPE_CHECKING:
    from io import BytesIO
    from types import SimpleNamespace

    import pytest


async def test_openai_file_uploader_drops_failed_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI upload errors degrade to a dropped attachment."""
    errored = _fake_openai_uploader(files=FakeOpenAIFiles(status="error"))
    assert (
        await errored._upload_file(
            cache_key="k", filename="bad.txt", data=b"x", content_type="text/plain", kind="file"
        )
        is None
    )

    boom = _fake_openai_uploader(files=FakeOpenAIFiles())

    async def _raise(
        file: tuple[str, BytesIO, str],
        purpose: str,
        expires_after: dict[str, object],
        extra_body: dict[str, object] | None = None,
    ) -> SimpleNamespace:
        del file, purpose, expires_after, extra_body
        raise RuntimeError("upload failed")

    monkeypatch.setattr(boom.client.files, "create", _raise)
    assert (
        await boom._upload_file(
            cache_key="k", filename="x.txt", data=b"x", content_type="text/plain", kind="file"
        )
        is None
    )


async def test_openai_file_uploader_inherits_file_upload_render_image() -> None:
    """`OpenAIFileUploader` inherits `FileUploadRenderer.render_image` returning `input_image`."""
    uploader = _fake_openai_uploader(files=FakeOpenAIFiles())
    rendered = await uploader.render_image(
        source=_att(filename="pic.png", content_type="image/png", payload=_png_bytes()),
        cache_key="pic.png",
    )

    assert rendered is not None
    assert rendered.part["type"] == "input_image"
    assert rendered.part["file_id"] == "file-test"
