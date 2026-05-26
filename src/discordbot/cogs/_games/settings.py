"""DB-backed game balance settings with a small process cache.

Default seed values live in `_economy.database._GAME_SETTING_SEEDS` and are
inserted on first schema bootstrap. Admin edits via
`scripts/manage_game_setting.py` persist across restarts; this module reads
them once per process and caches the result. Rules modules keep
their module-level `_DEFAULT_*` constants as fallback when the DB row is
missing or unreadable, so tests and offline tooling stay deterministic.
"""

from __future__ import annotations

from typing import Final

import logfire
from sqlalchemy import select

from discordbot.cogs._games.dragon_gate import ANTE as DRAGON_GATE_DEFAULT_ANTE
from discordbot.cogs._games.dragon_gate import GAME_ID as DRAGON_GATE_GAME_ID
from discordbot.cogs._games.dragon_gate import MIN_BET as DRAGON_GATE_DEFAULT_MIN_BET

_DRAGON_GATE_ANTE_KEY: Final[str] = "ante"
_DRAGON_GATE_MIN_BET_KEY: Final[str] = "min_bet"

_setting_cache: dict[tuple[str, str], int] = {}


def invalidate_game_setting_cache() -> None:
    """Clears the process-local game setting cache."""
    _setting_cache.clear()


async def get_game_setting(game_id: str, setting_key: str, default: int) -> int:
    """Returns the persisted value for one setting, with a process cache."""
    cache_key = (game_id, setting_key)
    cached = _setting_cache.get(cache_key)
    if cached is not None:
        return cached
    value = await _read_game_setting(
        game_id=game_id, setting_key=setting_key, default=default
    )
    _setting_cache[cache_key] = value
    return value


async def set_game_setting(game_id: str, setting_key: str, value: int) -> None:
    """Persists a setting value and refreshes the process cache."""
    from discordbot.cogs._economy.database import (
        GameSetting,
        _ensure_global_state_schema,
        open_global_state_session,
    )

    await _ensure_global_state_schema()
    async with open_global_state_session() as session:
        existing = await session.get(
            entity=GameSetting, ident={"game_id": game_id, "setting_key": setting_key}
        )
        if existing is None:
            session.add(
                instance=GameSetting(
                    game_id=game_id, setting_key=setting_key, setting_value=value
                )
            )
        else:
            existing.setting_value = value
        await session.commit()
    _setting_cache[(game_id, setting_key)] = value


async def list_game_settings(game_id: str | None = None) -> tuple[tuple[str, str, int], ...]:
    """Lists persisted settings, optionally filtered by game id."""
    from discordbot.cogs._economy.database import (
        GameSetting,
        _ensure_global_state_schema,
        open_global_state_session,
    )

    await _ensure_global_state_schema()
    async with open_global_state_session() as session:
        statement = select(
            GameSetting.game_id, GameSetting.setting_key, GameSetting.setting_value
        ).order_by(GameSetting.game_id.asc(), GameSetting.setting_key.asc())
        if game_id is not None:
            statement = statement.where(GameSetting.game_id == game_id)
        result = await session.execute(statement=statement)
        return tuple((row[0], row[1], row[2]) for row in result.all())


async def get_dragon_gate_ante() -> int:
    """Returns the Dragon Gate ante in points."""
    return await get_game_setting(
        game_id=DRAGON_GATE_GAME_ID,
        setting_key=_DRAGON_GATE_ANTE_KEY,
        default=DRAGON_GATE_DEFAULT_ANTE,
    )


async def get_dragon_gate_min_bet() -> int:
    """Returns the Dragon Gate minimum bet in points."""
    return await get_game_setting(
        game_id=DRAGON_GATE_GAME_ID,
        setting_key=_DRAGON_GATE_MIN_BET_KEY,
        default=DRAGON_GATE_DEFAULT_MIN_BET,
    )


async def _read_game_setting(game_id: str, setting_key: str, default: int) -> int:
    """Reads one setting row, returning `default` on DB error or missing row."""
    from discordbot.cogs._economy.database import (
        GameSetting,
        _ensure_global_state_schema,
        open_global_state_session,
    )

    try:
        await _ensure_global_state_schema()
        async with open_global_state_session() as session:
            row = await session.get(
                entity=GameSetting, ident={"game_id": game_id, "setting_key": setting_key}
            )
            if row is None:
                return default
            return row.setting_value
    except Exception:
        logfire.warn(
            "Failed to read game setting; falling back to default",
            game_id=game_id,
            setting_key=setting_key,
            default=default,
            _exc_info=True,
        )
        return default
