"""Tests for per-server auto-expansion settings stored in `settings.db`."""

from types import SimpleNamespace
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from discordbot.utils import server_settings
from discordbot.typings.config import clear_bot_config_cache
from discordbot.utils.server_settings import (
    GuildSettingsRow,
    get_guild_auto_expand,
    is_auto_expand_enabled,
    set_auto_expand_override,
    reset_auto_expand_overrides,
)

from tests.helpers.logfire_capture import capture_logs


def _write_local_config(path: Path, body: str) -> None:
    """Writes a throwaway local `config.toml` and clears the process config cache."""
    path.write_text(data=body, encoding="utf-8")
    clear_bot_config_cache()


async def _row_count() -> int:
    """Returns the number of persisted rows in `guild_settings`."""
    async with server_settings._database.open_session(engine=server_settings._engine) as session:
        count = await session.scalar(statement=select(func.count()).select_from(GuildSettingsRow))
        return int(count or 0)


async def test_defaults_inherit_config_toml_including_local_overrides(
    bot_config_isolated: Path,
) -> None:
    """Guilds without rows inherit `[server.auto_expand]` from `config.toml`, including None guild."""
    assert await is_auto_expand_enabled(guild_id=None, platform="twitter") is True
    assert await is_auto_expand_enabled(guild_id=101, platform="twitter") is True

    _write_local_config(
        path=bot_config_isolated, body="[server.auto_expand]\ntwitter = false\ndouyin = false\n"
    )
    server_settings._cache.clear()

    state = await get_guild_auto_expand(guild_id=101)
    assert state.effective(platform="twitter") is False
    assert state.effective(platform="douyin") is False
    assert state.effective(platform="threads") is True
    assert state.override_count() == 0
    assert await is_auto_expand_enabled(guild_id=None, platform="twitter") is False
    assert await _row_count() == 0


async def test_sparse_override_writes_and_deletes_row_when_restored_to_default() -> None:
    """Only non-default values persist; returning the last override to default deletes the row."""
    state = await set_auto_expand_override(guild_id=200, platform="twitter", enabled=False)
    assert state.effective(platform="twitter") is False
    assert state.is_overridden(platform="twitter") is True
    assert state.is_overridden(platform="threads") is False
    assert state.override_count() == 1
    assert await _row_count() == 1

    # Clearing the cache forces a DB read and proves the override persisted sparsely.
    server_settings._cache.clear()
    reloaded = await get_guild_auto_expand(guild_id=200)
    assert reloaded.effective(platform="twitter") is False
    assert reloaded.is_overridden(platform="twitter") is True
    assert reloaded.override_count() == 1

    # Toggling back to match the default clears the column to NULL and removes the empty row.
    restored = await set_auto_expand_override(guild_id=200, platform="twitter", enabled=True)
    assert restored.effective(platform="twitter") is True
    assert restored.is_overridden(platform="twitter") is False
    assert restored.override_count() == 0
    assert await _row_count() == 0


async def test_overrides_are_isolated_per_guild_and_reset_clears_all() -> None:
    """Per-guild overrides do not cross guilds, and reset clears every column for that guild."""
    await set_auto_expand_override(guild_id=1, platform="twitter", enabled=False)
    await set_auto_expand_override(guild_id=1, platform="facebook", enabled=False)
    await set_auto_expand_override(guild_id=2, platform="instagram", enabled=False)

    assert await is_auto_expand_enabled(guild_id=1, platform="twitter") is False
    assert await is_auto_expand_enabled(guild_id=1, platform="facebook") is False
    assert await is_auto_expand_enabled(guild_id=1, platform="instagram") is True
    assert await is_auto_expand_enabled(guild_id=2, platform="twitter") is True
    assert await is_auto_expand_enabled(guild_id=2, platform="instagram") is False
    assert await _row_count() == 2

    reset_state = await reset_auto_expand_overrides(guild_id=1)
    assert reset_state.override_count() == 0
    assert await is_auto_expand_enabled(guild_id=1, platform="twitter") is True
    assert await is_auto_expand_enabled(guild_id=1, platform="facebook") is True
    assert await is_auto_expand_enabled(guild_id=2, platform="instagram") is False
    assert await _row_count() == 1


async def test_cached_reads_skip_database_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the first lookup populates `_cache`, repeated reads do not hit SQLite."""
    assert await is_auto_expand_enabled(guild_id=300, platform="threads") is True
    assert 300 in server_settings._cache

    def fail_open_session(**_kwargs: object) -> object:
        raise AssertionError("database should not be queried on a cache hit")

    monkeypatch.setattr(
        target=server_settings,
        name="_database",
        value=SimpleNamespace(open_session=fail_open_session),
    )
    assert await is_auto_expand_enabled(guild_id=300, platform="threads") is True


async def test_is_auto_expand_enabled_falls_back_to_default_on_database_error(
    bot_config_isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient SQLite error logs a warning, returns the default, and does not poison cache."""
    _write_local_config(path=bot_config_isolated, body="[server.auto_expand]\ntwitter = false\n")
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")

    def broken_open_session(**_kwargs: object) -> object:
        raise OperationalError(
            statement="SELECT", params=None, orig=Exception("database is locked")
        )

    monkeypatch.setattr(
        target=server_settings,
        name="_database",
        value=SimpleNamespace(open_session=broken_open_session),
    )

    assert await is_auto_expand_enabled(guild_id=400, platform="twitter") is False
    assert await is_auto_expand_enabled(guild_id=400, platform="threads") is True
    assert 400 not in server_settings._cache
    assert len(warns) == 2
    assert warns[0][1]["guild_id"] == 400
    assert warns[0][1]["platform"] == "twitter"
