"""Per-day Gemini key usage counts (`data/database/llm_keys.db`).

One row per (day, key), holding how many times that key was handed out. The balancer above
this keeps the authoritative counts in memory and treats these rows as a snapshot: they exist
so a restart resumes the day's distribution instead of starting every key back at zero, and
so an operator can see how the traffic actually split.

The day is the window on purpose. A lifetime counter would make adding a fourth key unusable,
because the new key would be the lowest for as long as it took to catch up with months of
history, and would absorb every reply in the meantime. Rolling the window daily bounds that
catch-up to one day's traffic.

Its own file rather than a table in `reply.db`, because the keys are not a reply concept:
every cog that calls a Gemini model draws from the same pool. Engine, PRAGMA hooks and the
lazy schema bootstrap follow the other five databases through `SqliteBootstrap`, including
the module-level `AsyncEngine` singleton (a per-instance `cached_property` engine would leak
the pool and dialect cache).
"""

from sqlalchemy import String, Integer, select
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.dialects.sqlite import insert

from discordbot.utils.sqlite_config import SqliteBootstrap

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/llm_keys.db")


class Base(DeclarativeBase):
    """Declarative base for the key-usage table."""


class GeminiKeyUsageRow(Base):
    """How many replies one Gemini key served on one day.

    Attributes:
        day: Local (Asia/Taipei) date as `YYYY-MM-DD`, the window counts reset on.
        key_index: The key's number, the same one the `-key<n>` deployments carry.
        count: Times this key was handed out that day.
    """

    __tablename__ = "gemini_key_usage"

    day: Mapped[str] = mapped_column(String(length=10), primary_key=True)
    key_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


_database = SqliteBootstrap(metadata=Base.metadata)
_database.install_hooks(engine=_engine)


async def _ensure_schema() -> None:
    """Bootstraps this module's table once per engine (loop-local-locked)."""
    await _database.ensure_schema(engine=_engine)


def open_session() -> AsyncSession:
    """Creates an async session bound to the current llm_keys.db engine."""
    return _database.open_session(engine=_engine)


async def read_day_counts(day: str) -> dict[int, int]:
    """Returns each key's count for `day`, omitting keys that served nothing.

    Args:
        day: Local date as `YYYY-MM-DD`.

    Returns:
        Key number to count. Empty on a fresh day.
    """
    await _ensure_schema()
    async with open_session() as session:
        rows = await session.execute(select(GeminiKeyUsageRow).where(GeminiKeyUsageRow.day == day))
        return {row.key_index: row.count for row in rows.scalars()}


async def record_pick(day: str, key_index: int, count: int) -> None:
    """Writes back `key_index`'s running count for `day`.

    The caller's in-memory count is stored rather than incremented database-side, because
    that count is the authoritative one: a write that fails leaves the balancer correct for
    the rest of the process and self-heals on the next pick.

    Args:
        day: Local date as `YYYY-MM-DD`.
        key_index: The key that was handed out.
        count: The balancer's count for that key after the pick.
    """
    await _ensure_schema()
    async with open_session() as session:
        statement = insert(GeminiKeyUsageRow).values(day=day, key_index=key_index, count=count)
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=[GeminiKeyUsageRow.day, GeminiKeyUsageRow.key_index],
                set_={"count": count},
            )
        )
        await session.commit()
