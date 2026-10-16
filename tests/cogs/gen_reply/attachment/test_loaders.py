"""Tests for loading an attachment's bytes."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from datetime import UTC, datetime, timedelta

from PIL import Image, UnidentifiedImageError
import pytest
import nextcord

from discordbot.typings.media import LoadedMedia
from discordbot.cogs.gen_reply.attachment.base import DEAD_SOURCE_TTL
from discordbot.cogs.gen_reply.attachment.loaders import load_image_bytes

from tests.helpers.reply_fakes import (
    TEST_LLM_MODEL,
    FakeOpenAIFiles,
    _att,
    _png_bytes,
    _fake_uploader,
    _fake_openai_uploader,
)


async def test_dead_source_skipped_within_ttl_then_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing source is skipped (no re-fetch) for the TTL, then retried once after it."""
    calls = {"n": 0}

    def _raise_get_image_data(image_file: str) -> LoadedMedia:
        del image_file
        calls["n"] += 1
        raise RuntimeError("CDN url expired")

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.loaders.get_image_data", _raise_get_image_data
    )
    uploader = _fake_uploader()
    url = "https://example.test/dead.png"

    assert await uploader.render_image(source=url, cache_key=url, allow_dead_cache=True) is None
    assert calls["n"] == 1
    # Within the TTL the source is skipped without another fetch.
    assert await uploader.render_image(source=url, cache_key=url, allow_dead_cache=True) is None
    assert calls["n"] == 1
    # Backdating the marker past the TTL retries the fetch exactly once (self-heal).
    uploader._dead_sources[url] = datetime.now(tz=UTC) - DEAD_SOURCE_TTL - timedelta(seconds=1)
    assert await uploader.render_image(source=url, cache_key=url, allow_dead_cache=True) is None
    assert calls["n"] == 2


async def test_non_history_render_does_not_dead_cache_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current/reference renders (allow_dead_cache off) retry a transient failure, not poison it."""
    calls = {"n": 0}

    def _raise_get_image_data(image_file: str) -> LoadedMedia:
        del image_file
        calls["n"] += 1
        raise RuntimeError("transient blip")

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.loaders.get_image_data", _raise_get_image_data
    )
    uploader = _fake_uploader()
    url = "https://example.test/fresh.png"

    # Default path (current/reference): each call re-attempts the fetch and never marks dead.
    assert await uploader.render_image(source=url, cache_key=url) is None
    assert await uploader.render_image(source=url, cache_key=url) is None
    assert calls["n"] == 2
    assert url not in uploader._dead_sources


class _ImageAttachment(nextcord.Attachment):
    """A real `Attachment` subclass, so the image loader takes its attachment branch."""

    def __init__(self, content_type: str, payload: bytes) -> None:
        """Sets only what the loader reads; the slots nextcord fills from a payload stay unset."""
        self.filename = "pic.gif"
        self.content_type = content_type
        self._payload = payload

    async def read(self, use_cached: bool = False) -> bytes:
        """Returns the configured bytes instead of fetching from the CDN."""
        del use_cached
        return self._payload


async def test_an_image_attachment_mime_is_normalized_before_the_downscale() -> None:
    """Discord's reported MIME loses its parameters and case, so a GIF still passes through."""
    loaded = await load_image_bytes(
        source=_ImageAttachment(content_type="Image/GIF; charset=binary", payload=b"GIF89a")
    )
    assert loaded == LoadedMedia(data=b"GIF89a", mime_type="image/gif")


def _transparent_png() -> bytes:
    """A black square on a transparent 64x64 background."""
    image = Image.new(mode="RGBA", size=(64, 64), color=(0, 0, 0, 0))
    image.paste(im=(0, 0, 0, 255), box=(16, 16, 48, 48))
    buffered = BytesIO()
    image.save(fp=buffered, format="PNG")
    return buffered.getvalue()


def _animated_gif() -> bytes:
    """A red square on a transparent first frame, then a solid blue second frame."""
    palette = [0, 0, 0, 255, 0, 0, 0, 0, 255]
    first = Image.new(mode="P", size=(32, 32), color=0)
    first.putpalette(data=palette)
    first.paste(im=1, box=(8, 8, 24, 24))
    second = Image.new(mode="P", size=(32, 32), color=2)
    second.putpalette(data=palette)
    buffered = BytesIO()
    first.save(fp=buffered, format="GIF", save_all=True, append_images=[second], transparency=0)
    return buffered.getvalue()


@pytest.mark.parametrize(
    ("payload", "center"),
    [(_transparent_png(), (0, 0, 0, 255)), (_animated_gif(), (255, 0, 0, 255))],
    ids=["transparent-png", "animated-gif"],
)
async def test_a_linked_image_reaches_the_model_as_one_still_keeping_its_alpha(
    monkeypatch: pytest.MonkeyPatch, payload: bytes, center: tuple[int, int, int, int]
) -> None:
    """A linked image keeps its transparency, and an animated one arrives as its first frame."""
    monkeypatch.setattr(
        "discordbot.utils.images.requests.get",
        lambda url, timeout: SimpleNamespace(content=payload),
    )
    loaded = await load_image_bytes(source="https://cdn.test/linked")
    still = Image.open(fp=BytesIO(initial_bytes=loaded.data))
    assert loaded.mime_type == "image/png"
    assert getattr(still, "n_frames", 1) == 1
    rgba = still.convert("RGBA")
    assert rgba.getchannel(channel="A").getpixel(xy=(0, 0)) == 0
    assert rgba.getpixel(xy=(rgba.width // 2, rgba.height // 2)) == center


async def test_a_linked_page_that_is_not_an_image_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead CDN's HTML error page is dropped, never handed on as an image."""
    monkeypatch.setattr(
        "discordbot.utils.images.requests.get",
        lambda url, timeout: SimpleNamespace(content=b"<html>404</html>"),
    )
    with pytest.raises(UnidentifiedImageError):
        await load_image_bytes(source="https://cdn.test/gone.png")


async def test_openai_file_uploader_renders_image_and_file_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI uploads return file-id content parts for images and files."""
    files = FakeOpenAIFiles()
    renderer = _fake_openai_uploader(files=files)

    image_rendered = await renderer.render_image(
        source=_att(filename="pic.png", content_type="image/png", payload=_png_bytes()),
        cache_key="pic.png",
    )
    assert image_rendered is not None
    image_part = image_rendered.part
    image_expiry = image_rendered.expires_at
    assert image_part["type"] == "input_image"
    assert image_part["file_id"] == "file-test"
    assert image_part["detail"] == "auto"
    assert image_expiry == datetime(2099, 1, 1, tzinfo=UTC)

    url = "https://example.test/image.png"
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.attachment.loaders.get_image_data",
        lambda image_file: LoadedMedia(data=b"jpeg", mime_type="image/jpeg"),
    )
    url_image_rendered = await renderer.render_image(source=url, cache_key=url)
    assert url_image_rendered is not None
    url_image_part = url_image_rendered.part
    assert url_image_part["type"] == "input_image"
    assert url_image_part["file_id"] == "file-test"

    file_rendered = await renderer.render_file(
        attachment=_att(filename="notes.txt", content_type="text/plain", payload=b"hello world"),
        cache_key="notes.txt",
    )
    assert file_rendered is not None
    file_part = file_rendered.part
    file_expiry = file_rendered.expires_at
    assert file_part["type"] == "input_file"
    assert file_part["file_id"] == "file-test"
    assert file_part["filename"] == "notes.txt"
    assert file_expiry == datetime(2099, 1, 1, tzinfo=UTC)

    assert files.create_calls[0][0] == "pic.png"
    assert files.create_calls[0][2] == "image/jpeg"
    assert files.create_calls[0][3] == "vision"
    assert files.create_calls[0][4] == {"anchor": "created_at", "seconds": 2_592_000}
    assert files.create_calls[0][5] == {"model": TEST_LLM_MODEL}
    assert files.create_calls[1] == (
        "image.jpg",
        b"jpeg",
        "image/jpeg",
        "vision",
        {"anchor": "created_at", "seconds": 2_592_000},
        {"model": TEST_LLM_MODEL},
    )
    assert files.create_calls[2] == (
        "notes.txt",
        b"hello world",
        "text/plain",
        "user_data",
        {"anchor": "created_at", "seconds": 2_592_000},
        {"model": TEST_LLM_MODEL},
    )
