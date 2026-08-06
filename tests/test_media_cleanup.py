"""Tests for the media-cleanup cog's `setup` registration and its `on_ready` start gate.

Three decisions are pinned, all of them about whether the sweep is scheduled at all. Hosting
configured with at least one cap starts the `tasks.loop` AND spawns exactly one immediate startup
sweep, because the loop itself first fires a whole interval after `start()` and a restart would
otherwise leave an over-cap serve dir alone for hours. Both caps off means `cleanup_enabled` is
false, so nothing starts and nothing in the serve dir is touched. And since `on_ready` fires again
on every gateway reconnect, the cog's `_started` gate has to make the second one a no-op rather
than a second loop plus a second sweep.

The sweep itself never runs here. `MediaCleanupCogs` builds its own `MediaHostingService` from
`MediaHostingConfig()`, which reads the process environment, so a real `run_maintenance` would
delete against whatever serve dir the deployment resolves. Every test therefore monkeypatches
`_sweep` and replaces `cog.media_hosting` with `_service`, which is pointed at a `tmp_path`.
"""

from types import SimpleNamespace
from pathlib import Path

import pytest

from discordbot.cogs.media_cleanup import cog as media_cleanup
from discordbot.utils.media_delivery import MediaHostingService
from discordbot.cogs.media_cleanup.cog import MediaCleanupCogs

from tests.helpers.casting import as_bot, make_media_hosting_config


class _FakeBot:
    """A bot stub carrying only what the cog reaches for: `wait_until_ready`."""

    async def wait_until_ready(self) -> None:
        """Resolves immediately so the loop's `before_loop` never blocks the test."""
        return


def _service(
    *, serve_dir: Path, max_bytes: int = 8 * 1024**3, retention_hours: float = 168.0
) -> MediaHostingService:
    """Builds a hosting service over an explicit serve dir, never the env-resolved live one.

    The cap defaults mirror `MediaHostingConfig`'s own, so a caller overrides only the cap it is
    testing and an enabled service is otherwise shaped like the deployed one.

    Returns:
        A `MediaHostingService` built with no `.env` or process environment mixed in.
    """
    return MediaHostingService(
        config=make_media_hosting_config(
            enabled=True,
            base_url="https://media.test",
            serve_dir=str(serve_dir),
            max_bytes=max_bytes,
            retention_hours=retention_hours,
        )
    )


def test_setup_registers_media_cleanup_cog() -> None:
    """The module setup registers exactly one MediaCleanupCogs with override=True."""
    added: list[tuple[object, object]] = []
    bot = SimpleNamespace(add_cog=lambda cog, override=None: added.append((cog, override)))

    media_cleanup.setup(bot=as_bot(fake=bot))

    assert len(added) == 1
    assert isinstance(added[0][0], MediaCleanupCogs)
    assert added[0][1] is True


async def test_on_ready_starts_loop_and_sweeps_once_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With hosting + a cap configured, on_ready spawns one startup sweep and starts the loop."""
    cog = MediaCleanupCogs(bot=as_bot(fake=_FakeBot()))
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
    cog = MediaCleanupCogs(bot=as_bot(fake=_FakeBot()))
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
    cog = MediaCleanupCogs(bot=as_bot(fake=_FakeBot()))
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
