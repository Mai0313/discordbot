"""Tests for the media-cleanup cog's on_ready start gate.

These never let the real sweep run — it would delete against the env-resolved live serve dir — so
the startup sweep is stubbed and only the gating decision (start vs no-op) is asserted.
"""

from types import SimpleNamespace
import asyncio
from pathlib import Path

import pytest

from discordbot.utils.media_delivery import MediaHostingService
from discordbot.cogs.media_cleanup.cog import MediaCleanupCogs

from tests.helpers.casting import as_bot, make_media_hosting_config


def _service(
    *, serve_dir: Path, max_bytes: int = 8 * 1024**3, retention_hours: float = 168.0
) -> MediaHostingService:
    """A hosting service over an explicit temp serve dir (never the live env-resolved dir)."""
    return MediaHostingService(
        config=make_media_hosting_config(
            enabled=True,
            base_url="https://media.test",
            serve_dir=str(serve_dir),
            max_bytes=max_bytes,
            retention_hours=retention_hours,
        )
    )


async def test_on_ready_starts_loop_and_sweeps_once_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With hosting + a cap configured, on_ready starts the loop and exactly one sweep runs."""
    cog = MediaCleanupCogs(bot=as_bot(fake=SimpleNamespace()))
    cog.media_hosting = _service(serve_dir=tmp_path)
    swept: list[bool] = []
    first_sweep = asyncio.Event()

    async def _fake_sweep() -> None:
        swept.append(True)
        first_sweep.set()

    monkeypatch.setattr(cog, "_sweep", _fake_sweep)

    await cog.on_ready()
    await asyncio.wait_for(fut=first_sweep.wait(), timeout=5)
    await asyncio.sleep(delay=0.2)  # room for a second startup sweep, were one scheduled

    assert cog.cleanup_loop.is_running()
    assert swept == [True]
    cog.cleanup_loop.cancel()


async def test_on_ready_is_inert_when_cleanup_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both caps off -> cleanup disabled -> the loop never starts and no sweep runs."""
    cog = MediaCleanupCogs(bot=as_bot(fake=SimpleNamespace()))
    cog.media_hosting = _service(serve_dir=tmp_path, max_bytes=0, retention_hours=0)
    swept: list[bool] = []

    async def _fake_sweep() -> None:
        swept.append(True)

    monkeypatch.setattr(cog, "_sweep", _fake_sweep)

    await cog.on_ready()

    assert not cog.cleanup_loop.is_running()
    assert swept == []


async def test_on_ready_starts_once_across_reconnects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """on_ready fires on every reconnect, but the _started gate starts the loop only once."""
    cog = MediaCleanupCogs(bot=as_bot(fake=SimpleNamespace()))
    cog.media_hosting = _service(serve_dir=tmp_path)
    sweeps: list[bool] = []
    first_sweep = asyncio.Event()

    async def _fake_sweep() -> None:
        sweeps.append(True)
        first_sweep.set()

    monkeypatch.setattr(cog, "_sweep", _fake_sweep)

    await cog.on_ready()
    await cog.on_ready()  # a reconnect
    await asyncio.wait_for(fut=first_sweep.wait(), timeout=5)
    await asyncio.sleep(delay=0.2)

    assert sweeps == [True]
    cog.cleanup_loop.cancel()
