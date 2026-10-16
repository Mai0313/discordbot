"""Tests for the media-cleanup cog's on_ready start gate.

Each cog's hosting service is pointed at the test's own temp serve dir before `on_ready`, so the
sweep a started loop runs touches nothing outside it.
"""

from types import SimpleNamespace
import asyncio
from pathlib import Path
import contextlib

import pytest

from discordbot.cogs.media_cleanup.cog import MediaCleanupCogs

from tests.helpers.casting import as_bot
from tests.helpers.link_sources import hosting_service


def _record_starts(monkeypatch: pytest.MonkeyPatch, cog: MediaCleanupCogs) -> list[bool]:
    """Records every start of the cog's cleanup loop, letting each one through."""
    starts: list[bool] = []
    real_start = cog.cleanup_loop.start

    def recording_start() -> asyncio.Task[None]:
        starts.append(True)
        return real_start()

    monkeypatch.setattr(cog.cleanup_loop, "start", recording_start)
    return starts


async def test_on_ready_is_inert_when_cleanup_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both caps off -> cleanup disabled -> the loop never starts and no sweep runs."""
    cog = MediaCleanupCogs(bot=as_bot(fake=SimpleNamespace()))
    cog.media_hosting = hosting_service(serve_dir=tmp_path, max_bytes=0, retention_hours=0)
    starts = _record_starts(monkeypatch=monkeypatch, cog=cog)

    await cog.on_ready()

    assert starts == []
    assert not cog.cleanup_loop.is_running()


async def test_on_ready_starts_once_across_reconnects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """on_ready fires on every reconnect, but the _started gate starts the loop only once."""
    cog = MediaCleanupCogs(bot=as_bot(fake=SimpleNamespace()))
    cog.media_hosting = hosting_service(serve_dir=tmp_path)
    starts = _record_starts(monkeypatch=monkeypatch, cog=cog)

    try:
        await cog.on_ready()
        await cog.on_ready()  # a reconnect

        assert starts == [True]
        assert cog.cleanup_loop.is_running()
    finally:
        task = cog.cleanup_loop.get_task()
        cog.cleanup_loop.cancel()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
