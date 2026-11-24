"""Per-server feature overrides stored sparsely in `data/database/settings.db`.

Only columns that diverge from `[server.*]` defaults in `config.toml` are stored; toggling a
setting back to its default clears the column to `NULL`, and a row with every column `NULL` is
deleted so a server falls back cleanly to future default changes.
"""

from typing import Final
from datetime import datetime

import logfire
from pydantic import Field, BaseModel
from sqlalchemy import Boolean, Integer, DateTime, delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from discordbot.utils.lru import BoundedLRU
from discordbot.typings.config import AutoExpandConfig, AutoExpandPlatform, load_bot_config
from discordbot.utils.timezone import database_now
from discordbot.utils.asyncio_locks import KeyedLockManager
from discordbot.utils.sqlite_config import SqliteBootstrap, sqlite_url

AUTO_EXPAND_PLATFORMS: Final[tuple[AutoExpandPlatform, ...]] = (
    "twitter",
    "threads",
    "facebook",
    "instagram",
    "douyin",
)

_COLUMN_NAMES: Final[dict[AutoExpandPlatform, str]] = {
    "twitter": "auto_expand_twitter",
    "threads": "auto_expand_threads",
    "facebook": "auto_expand_facebook",
    "instagram": "auto_expand_instagram",
    "douyin": "auto_expand_douyin",
}

_CACHE_MAX_ENTRIES: Final[int] = 1024

_engine: AsyncEngine = create_async_engine(url=sqlite_url(filename="settings.db"))
_database = SqliteBootstrap()
_guild_locks: KeyedLockManager[int] = KeyedLockManager()


class Base(DeclarativeBase):
    """Declarative base for per-server settings stored in `settings.db`."""


class GuildSettingsRow(Base):
    """Sparse per-guild feature overrides; NULL columns inherit `config.toml` defaults.

    Attributes:
        guild_id (Mapped[int]): Discord guild id (primary key).
        auto_expand_twitter (Mapped[bool | None]): Override for Twitter/X auto-expansion.
        auto_expand_threads (Mapped[bool | None]): Override for Threads auto-expansion.
        auto_expand_facebook (Mapped[bool | None]): Override for Facebook auto-expansion.
        auto_expand_instagram (Mapped[bool | None]): Override for Instagram auto-expansion.
        auto_expand_douyin (Mapped[bool | None]): Override for Douyin auto-expansion.
        updated_at (Mapped[datetime]): Last write timestamp in Asia/Taipei.
    """

    __tablename__ = "guild_settings"

    guild_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    auto_expand_twitter: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=None)
    auto_expand_threads: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=None)
    auto_expand_facebook: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=None)
    auto_expand_instagram: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, default=None
    )
    auto_expand_douyin: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=None)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=database_now, onupdate=database_now
    )


class AutoExpandOverrides(BaseModel):
    """Cached sparse overrides for one guild's link auto-expansion switches."""

    twitter: bool | None = Field(
        default=None, description="Per-guild override for Twitter/X, or None to inherit default."
    )
    threads: bool | None = Field(
        default=None, description="Per-guild override for Threads, or None to inherit default."
    )
    facebook: bool | None = Field(
        default=None, description="Per-guild override for Facebook, or None to inherit default."
    )
    instagram: bool | None = Field(
        default=None, description="Per-guild override for Instagram, or None to inherit default."
    )
    douyin: bool | None = Field(
        default=None, description="Per-guild override for Douyin, or None to inherit default."
    )

    def for_platform(self, platform: AutoExpandPlatform) -> bool | None:
        """Returns the stored override for `platform`, or None when inheriting the default.

        Args:
            platform (AutoExpandPlatform): Auto-expansion platform key.
        """
        return getattr(self, platform)

    def is_empty(self) -> bool:
        """Returns whether every platform inherits the default with no stored override."""
        return all(getattr(self, platform) is None for platform in AUTO_EXPAND_PLATFORMS)

    def override_count(self) -> int:
        """Returns how many platforms carry an explicit per-guild override."""
        return sum(getattr(self, platform) is not None for platform in AUTO_EXPAND_PLATFORMS)

    def with_platform(
        self, platform: AutoExpandPlatform, value: bool | None
    ) -> "AutoExpandOverrides":
        """Returns a copy with `platform`'s override replaced by `value`.

        Args:
            platform (AutoExpandPlatform): Auto-expansion platform key.
            value (bool | None): Explicit boolean override, or None to inherit the default.
        """
        return self.model_copy(update={platform: value})


class GuildAutoExpandState(BaseModel):
    """Combined defaults and sparse overrides for one guild's auto-expansion switches."""

    defaults: AutoExpandConfig = Field(
        ..., description="Deployment defaults from `[server.auto_expand]` in `config.toml`."
    )
    overrides: AutoExpandOverrides = Field(
        ..., description="Sparse per-guild overrides stored in `settings.db`."
    )

    def default_for(self, platform: AutoExpandPlatform) -> bool:
        """Returns the `config.toml` default for `platform`.

        Args:
            platform (AutoExpandPlatform): Auto-expansion platform key.
        """
        return bool(getattr(self.defaults, platform))

    def effective(self, platform: AutoExpandPlatform) -> bool:
        """Returns the effective enabled state for `platform` in this guild.

        Args:
            platform (AutoExpandPlatform): Auto-expansion platform key.
        """
        override = self.overrides.for_platform(platform=platform)
        return self.default_for(platform=platform) if override is None else override

    def is_overridden(self, platform: AutoExpandPlatform) -> bool:
        """Returns whether this guild stores an explicit override for `platform`.

        Args:
            platform (AutoExpandPlatform): Auto-expansion platform key.
        """
        return self.overrides.for_platform(platform=platform) is not None

    def override_count(self) -> int:
        """Returns how many platforms carry an explicit override in this guild."""
        return self.overrides.override_count()


_cache: BoundedLRU[int, AutoExpandOverrides] = BoundedLRU(max_entries=_CACHE_MAX_ENTRIES)


def _overrides_from_row(row: GuildSettingsRow | None) -> AutoExpandOverrides:
    """Builds `AutoExpandOverrides` from a persisted row, or empty overrides when absent."""
    if row is None:
        return AutoExpandOverrides()
    return AutoExpandOverrides(
        twitter=row.auto_expand_twitter,
        threads=row.auto_expand_threads,
        facebook=row.auto_expand_facebook,
        instagram=row.auto_expand_instagram,
        douyin=row.auto_expand_douyin,
    )


async def _load_overrides(guild_id: int) -> AutoExpandOverrides:
    """Returns cached overrides for `guild_id`, reading `settings.db` on a cache miss."""
    cached = _cache.get(key=guild_id)
    if cached is not None:
        return cached
    # Serialize cache fills with writes for the same guild so a slow SELECT cannot overwrite a
    # concurrent toggle in `_cache`.
    async with _guild_locks.hold(key=guild_id):
        cached = _cache.get(key=guild_id)
        if cached is not None:
            return cached
        async with _database.open_session(engine=_engine) as session:
            row = await session.get(entity=GuildSettingsRow, ident=guild_id)
        overrides = _overrides_from_row(row=row)
        _cache.set(key=guild_id, value=overrides)
        return overrides


async def get_guild_auto_expand(guild_id: int) -> GuildAutoExpandState:
    """Returns the effective auto-expand defaults and overrides for `guild_id`.

    Args:
        guild_id (int): Discord guild id.
    """
    defaults = load_bot_config().server.auto_expand
    overrides = await _load_overrides(guild_id=guild_id)
    return GuildAutoExpandState(defaults=defaults, overrides=overrides)


async def is_auto_expand_enabled(guild_id: int | None, platform: AutoExpandPlatform) -> bool:
    """Returns whether `platform` link auto-expansion is enabled in `guild_id`.

    Falls back to the `config.toml` default when `guild_id` is None or when reading `settings.db`
    fails with a database or OS error, so a transient SQLite error never silences expansions.

    Args:
        guild_id (int | None): Discord guild id, or None outside a guild.
        platform (AutoExpandPlatform): Auto-expansion platform key.
    """
    defaults = load_bot_config().server.auto_expand
    default_value = bool(getattr(defaults, platform))
    if guild_id is None:
        return default_value
    try:
        overrides = await _load_overrides(guild_id=guild_id)
    except (SQLAlchemyError, OSError) as exc:
        logfire.warn(
            "Failed to read server auto-expand settings; falling back to default",
            guild_id=guild_id,
            platform=platform,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
        return default_value
    override = overrides.for_platform(platform=platform)
    return default_value if override is None else override


async def set_auto_expand_override(
    guild_id: int, platform: AutoExpandPlatform, enabled: bool
) -> GuildAutoExpandState:
    """Sets `platform` to `enabled` for `guild_id`, storing `NULL` when matching the default.

    Deletes the guild row once every column returns to `NULL`.

    Args:
        guild_id (int): Discord guild id.
        platform (AutoExpandPlatform): Auto-expansion platform key to update.
        enabled (bool): Desired effective state for `platform`.
    """
    defaults = load_bot_config().server.auto_expand
    default_value = bool(getattr(defaults, platform))
    stored_value = None if enabled == default_value else enabled
    async with _guild_locks.hold(key=guild_id):
        async with _database.open_session(engine=_engine) as session, session.begin():
            row = await session.get(entity=GuildSettingsRow, ident=guild_id)
            current = _overrides_from_row(row=row)
            updated = current.with_platform(platform=platform, value=stored_value)
            if updated.is_empty():
                if row is not None:
                    await session.delete(instance=row)
            else:
                if row is None:
                    row = GuildSettingsRow(guild_id=guild_id)
                    session.add(instance=row)
                setattr(row, _COLUMN_NAMES[platform], stored_value)
                row.updated_at = database_now()
        _cache.set(key=guild_id, value=updated)
        return GuildAutoExpandState(defaults=defaults, overrides=updated)


async def reset_auto_expand_overrides(guild_id: int) -> GuildAutoExpandState:
    """Removes every auto-expand override for `guild_id` so it inherits `config.toml` defaults.

    Args:
        guild_id (int): Discord guild id.
    """
    defaults = load_bot_config().server.auto_expand
    async with _guild_locks.hold(key=guild_id):
        async with _database.open_session(engine=_engine) as session, session.begin():
            await session.execute(
                statement=delete(GuildSettingsRow).where(GuildSettingsRow.guild_id == guild_id)
            )
        empty = AutoExpandOverrides()
        _cache.set(key=guild_id, value=empty)
        return GuildAutoExpandState(defaults=defaults, overrides=empty)
