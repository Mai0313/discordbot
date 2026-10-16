"""Tests for the upload machinery every attachment handler shares."""

from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio

from discordbot.typings.media import LoadedMedia
from discordbot.cogs.gen_reply.attachment.base import loggable_cache_key

from tests.helpers.reply_fakes import _fake_uploader

if TYPE_CHECKING:
    import pytest


async def test_media_semaphore_bounds_media_io_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared semaphore caps the whole download+upload sequence, not just the upload.

    Counting concurrency in the byte loader proves non-image downloads (which run before the
    Gemini upload) are bounded too, so concurrent pipelines cannot buffer every file at once.
    """
    # The cap is module-level, and the loop-local holder reads it fresh on this test's own loop.
    monkeypatch.setattr("discordbot.cogs.gen_reply.attachment.base.MEDIA_CONCURRENCY", 2)
    uploader = _fake_uploader()
    state = {"active": 0, "peak": 0}

    async def _slow_load() -> LoadedMedia:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return LoadedMedia(data=b"x", mime_type="image/png")

    results = await asyncio.gather(*[
        uploader._resolve_file_upload(
            cache_key=f"k{index}", filename=f"f{index}", load_data=_slow_load, kind="file"
        )
        for index in range(6)
    ])

    assert all(result is not None for result in results)
    assert state["peak"] == 2


def test_loggable_cache_key_strips_url_query_token() -> None:
    """An int key logs unchanged; a URL key drops its (possibly signed) query string."""
    assert loggable_cache_key(cache_key=12345) == 12345
    assert (
        loggable_cache_key(cache_key="https://media.discordapp.net/x/y.png?ex=1&hm=secrettoken")
        == "https://media.discordapp.net/x/y.png"
    )
    assert loggable_cache_key(cache_key="https://cdn.example/a.png") == "https://cdn.example/a.png"
