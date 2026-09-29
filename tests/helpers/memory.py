"""Builders for the memory store's stored facts and consolidation deltas, and a row reader.

Every default is an ordinary per-user preference, so a test names only the fields it is about.
A fact carries no default owner: the owner is what a scope's stored identity is read back from,
so each test module binds its own.
"""

from datetime import UTC, datetime

from sqlalchemy import select

from discordbot.typings.memory import (
    MemoryFact,
    MemoryOwner,
    MemorySection,
    MemoryDurability,
    MemoryDeltaAction,
)
from discordbot.services.memory.facts import node_type_for
from discordbot.services.memory.store import GLOBAL_COMPARTMENT
from discordbot.services.memory.writer import MemoryFactDelta
from discordbot.services.memory.database import (
    MemoryJob,
    MemoryJobRow,
    open_session,
    _row_to_model,
)

# The moment a built fact was written and last confirmed, unless a test ages it on purpose.
STAMPED_AT = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)


def make_fact(  # noqa: PLR0913 -- mirrors the stored fact's own fields
    owner: MemoryOwner,
    fact_id: str = "0123456789abcdef",
    summary: str = "回覆長度偏好",
    section: MemorySection = "preference",
    durability: MemoryDurability = "stable",
    text: str = "喜歡簡短回覆",
    compartment: str = GLOBAL_COMPARTMENT,
    last_confirmed: datetime = STAMPED_AT,
    subject_id: int | None = None,
    keys: tuple[str, ...] = (),
) -> MemoryFact:
    """Builds one already-consolidated fact, with the code-stamped fields filled in."""
    return MemoryFact(
        fact_id=fact_id,
        summary=summary,
        section=section,
        durability=durability,
        text=text,
        compartment=compartment,
        owner_id=owner.owner_id,
        owner_name=owner.owner_name,
        subject_id=subject_id,
        node_type=node_type_for(section=section),
        created=STAMPED_AT,
        last_confirmed=last_confirmed,
        keys=keys,
    )


def make_delta(  # noqa: PLR0913 -- mirrors the delta schema
    action: MemoryDeltaAction = "create",
    fact_id: str = "",
    section: MemorySection = "preference",
    durability: MemoryDurability = "stable",
    summary: str = "回覆長度偏好",
    text: str = "喜歡簡短回覆",
    from_keys: tuple[str, ...] = (),
    subject_id: str = "",
    display_name: str = "",
    aliases: tuple[str, ...] = (),
) -> MemoryFactDelta:
    """Builds one consolidation delta, the shape the model answers a consolidation call with."""
    return MemoryFactDelta(
        action=action,
        fact_id=fact_id,
        section=section,
        durability=durability,
        summary=summary,
        text=text,
        from_keys=from_keys,
        subject_id=subject_id,
        display_name=display_name,
        aliases=aliases,
    )


async def get_job(scope: str) -> MemoryJob | None:
    """Reads one scope's `memory_job` row, or None when it is not tracked.

    Unwrapped on purpose: the restart sweep's bulk read degrades a failure to "nothing to
    resume", so a test asserting through it would pass without having looked.
    """
    async with open_session() as session:
        result = await session.execute(
            statement=select(MemoryJobRow).where(MemoryJobRow.scope == scope)
        )
        row = result.scalars().one_or_none()
        return _row_to_model(row=row) if row is not None else None
