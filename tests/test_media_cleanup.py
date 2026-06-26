"""Tests for the media-cleanup cog: setup registration and the on_ready start gate.

These never let the real sweep run — it would delete against the env-resolved live serve dir — so
the startup sweep is stubbed and only the gating decision (start vs no-op) is asserted.
"""

from types import SimpleNamespace
from pathlib import Path

import pytest

from discordbot.cogs import media_cleanup
from discordbot.cogs.media_cleanup import MediaCleanupCogs
from discordbot.utils.media_delivery import MediaHostingConfig, MediaHostingService


class _FakeBot:
    """A bot stub whose wait_until_ready resolves immediately (for the loop's before_loop)."""

    async def wait_until_ready(self) -> None:
        """Returns immediately so the loop's before_loop never blocks the test."""
        return


def _service(
    *, serve_dir: Path, max_bytes: int = 8 * 1024**3, retention_hours: float = 168.0
) -> MediaHostingService:
    """A hosting service over an explicit temp serve dir (never the live env-resolved dir)."""
    return MediaHostingService(
        config=MediaHostingConfig(
            MEDIA_HOSTING_ENABLED=True,
            MEDIA_HOSTING_BASE_URL="https://media.test",
            MEDIA_HOSTING_SERVE_DIR=str(serve_dir),
            MEDIA_HOSTING_MAX_BYTES=max_bytes,
            MEDIA_HOSTING_RETENTION_HOURS=retention_hours,
        )
    )


def test_setup_registers_media_cleanup_cog() -> None:
    """The module setup registers exactly one MediaCleanupCogs with override=True."""
    added: list[tuple[object, object]] = []
    bot = SimpleNamespace(add_cog=lambda cog, override=None: added.append((cog, override)))

    media_cleanup.setup(bot=bot)

    assert len(added) == 1
    assert isinstance(added[0][0], MediaCleanupCogs)
    assert added[0][1] is True


async def test_on_ready_starts_loop_and_sweeps_once_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With hosting + a cap configured, on_ready spawns one startup sweep and starts the loop."""
    cog = MediaCleanupCogs(bot=_FakeBot())
    cog.media_hosting = _service(serve_dir=tmp_path)
    swept: list[bool] = []

    async def _fake_sweep() -> None:
        swept.append(True)

    monkeypatch.setattr(cog, "_sweep", _fake_sweep)

    await cog.on_ready()

    assert cog.cleanup_loop.is_running()
    assert cog._startup_task is not None
    await cog._startup_task
    assert swept == [True]  # exactly one immediate startup sweep
    cog.cleanup_loop.cancel()


async def test_on_ready_is_inert_when_cleanup_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both caps off -> cleanup disabled -> the loop never starts and no sweep runs."""
    cog = MediaCleanupCogs(bot=_FakeBot())
    cog.media_hosting = _service(serve_dir=tmp_path, max_bytes=0, retention_hours=0)
    swept: list[bool] = []

    async def _fake_sweep() -> None:
        swept.append(True)

    monkeypatch.setattr(cog, "_sweep", _fake_sweep)

    await cog.on_ready()

    assert not cog.cleanup_loop.is_running()
    assert cog._startup_task is None
    assert swept == []


async def test_on_ready_starts_once_across_reconnects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """on_ready fires on every reconnect, but the _started gate starts the loop only once."""
    cog = MediaCleanupCogs(bot=_FakeBot())
    cog.media_hosting = _service(serve_dir=tmp_path)
    sweeps: list[bool] = []

    async def _fake_sweep() -> None:
        sweeps.append(True)

    monkeypatch.setattr(cog, "_sweep", _fake_sweep)

    await cog.on_ready()
    first_task = cog._startup_task
    await cog.on_ready()  # a reconnect

    assert cog._startup_task is first_task  # not re-spawned
    if first_task is not None:
        await first_task
    assert sweeps == [True]
    cog.cleanup_loop.cancel()
