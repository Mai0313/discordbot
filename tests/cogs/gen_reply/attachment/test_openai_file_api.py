"""Tests for the OpenAI file uploader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.helpers.reply_fakes import FakeOpenAIFiles, _fake_openai_uploader

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
