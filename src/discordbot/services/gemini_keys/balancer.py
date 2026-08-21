"""Picks which configured Gemini key a piece of work runs on.

One reply is one pick, and every call that reply makes then runs on that key: a Gemini Files
API file is readable only by the project that uploaded it, and one request naming files from
two keys fails outright rather than partially. That is also why LiteLLM's own pooling cannot
do this job, and why the pick has to happen once, high up, instead of per call.

Selection is lowest-count-first for the current local day, ties going to the lowest key
number. Not round-robin, because the counts survive a restart and a new key starts behind:
lowest-first lets it catch up, where round-robin would just resume alternating and leave the
gap in place forever.

The in-memory counts are the authoritative ones and the database rows are their snapshot, so
a database that is unreachable costs the day's history and nothing else — the process still
spreads its own traffic evenly. Every database call here is therefore best-effort.
"""

import logfire

from discordbot.typings.llm import LLMConfig, GeminiKeySlot
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.timezone import database_now
from discordbot.utils.asyncio_locks import LoopLocalLock
from discordbot.services.gemini_keys.database import record_pick, read_day_counts

# Guards the day window and the counts together, so two replies arriving at once cannot read
# the same lowest key and both take it. Loop-local because a module-level `asyncio.Lock`
# binds to the first loop that waits on it and every test runs on a fresh one.
_state_lock = LoopLocalLock()
# The day the counts below belong to; a different local date rebuilds them from the database.
_counted_day: str | None = None
_counts: dict[int, int] = {}


def _today() -> str:
    """Returns the local date the counts are windowed on, as `YYYY-MM-DD`."""
    return database_now().strftime("%Y-%m-%d")


async def _load_counts(day: str) -> dict[int, int]:
    """Reads `day`'s counts, or starts the day at zero when the database cannot be read."""
    try:
        return await read_day_counts(day=day)
    except Exception as error:
        # Losing the day's history only costs continuity across a restart; the process still
        # balances its own traffic, so this must never reach the reply that triggered it.
        logfire.warn(
            "gemini key counts unavailable; starting the day at zero",
            day=day,
            error_type=type(error).__name__,
            _exc_info=error,
        )
        return {}


async def _persist(day: str, key_index: int, count: int) -> None:
    """Writes one count back, swallowing any failure."""
    try:
        await record_pick(day=day, key_index=key_index, count=count)
    except Exception as error:
        logfire.warn(
            "gemini key count not persisted",
            day=day,
            key_index=key_index,
            error_type=type(error).__name__,
            _exc_info=error,
        )


async def pick_gemini_key(config: LLMConfig) -> GeminiKeySlot | None:
    """Hands out the least-used configured key for today and counts the hand-out.

    Args:
        config: Runtime LLM config supplying the configured keys.

    Returns:
        The chosen key, or None when the deployment has no Gemini key at all. None is a
        supported state rather than an error: the caller then runs unpinned, which is what
        the bot did before any of this existed, and the Gemini-only features gate themselves
        off as they already do.
    """
    keys = config.gemini_keys
    if not keys:
        return None
    day = _today()
    async with _state_lock.get():
        global _counted_day  # noqa: PLW0603 -- module-level day window for the counts below
        if _counted_day != day:
            _counts.clear()
            _counts.update(await _load_counts(day=day))
            _counted_day = day
        chosen = min(keys, key=lambda slot: (_counts.get(slot.index, 0), slot.index))
        count = _counts.get(chosen.index, 0) + 1
        _counts[chosen.index] = count
    # Outside the lock: the reply path must not wait on a database write, and the write
    # carries an absolute count rather than an increment, so two picks landing out of order
    # cost at most a momentarily stale row that the next pick corrects.
    await _persist(day=day, key_index=chosen.index, count=count)
    return chosen


async def lease_model_catalog(config: LLMConfig) -> RuntimeModelCatalog:
    """Leases a key and returns a model catalog with every tier pinned to it.

    For the callers that need nothing else from the key: no Files API upload, no direct
    Gemini client, just their own model tier on a balanced deployment. `gen_reply` does not
    use this, because a reply needs the clients and caches too and gets a whole toolkit.

    Args:
        config: Runtime LLM config supplying the configured keys.

    Returns:
        A catalog pinned to the leased key, or an unpinned one when none is configured.
    """
    slot = await pick_gemini_key(config=config)
    return RuntimeModelCatalog(key_index=slot.index if slot is not None else None)


def reset_balancer_state() -> None:
    """Drops the in-memory day window and counts. For tests only."""
    global _counted_day  # noqa: PLW0603 -- test-only reset of the module-level window
    _counted_day = None
    _counts.clear()


__all__ = ["lease_model_catalog", "pick_gemini_key", "reset_balancer_state"]
