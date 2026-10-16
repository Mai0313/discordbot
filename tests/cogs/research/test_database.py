"""Tests for the deep-research session store in reply.db."""

from typing import TYPE_CHECKING

from discordbot.cogs.research import database as rdb
from discordbot.utils.timezone import database_now

if TYPE_CHECKING:
    from discordbot.cogs.research.database import ResearchPhase


async def _only_resumable(thread_id: int) -> rdb.PersistentResearchSession | None:
    """The one resumable row for a thread; the store has no single-row reader left to use."""
    return next(
        (
            row
            for row in await rdb.list_resumable(created_before=database_now())
            if row.thread_id == thread_id
        ),
        None,
    )


async def test_session_round_trip() -> None:
    await rdb.insert_session(
        thread_id=1,
        owner_id=99,
        channel_id=7,
        guild_id=5,
        source_message_id=3,
        agent="antigravity-preview-09-2026",
        brief="研究 X",
    )
    session = await _only_resumable(thread_id=1)
    assert session is not None
    assert session.owner_id == 99
    assert session.interaction_id is None
    assert await _only_resumable(thread_id=999) is None


async def test_set_interaction_and_phase() -> None:
    await rdb.insert_session(
        thread_id=2,
        owner_id=1,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="antigravity-preview-09-2026",
        brief="b",
    )
    await rdb.set_interaction(thread_id=2, interaction_id="int_abc")
    session = await _only_resumable(thread_id=2)
    assert session is not None
    assert session.interaction_id == "int_abc"
    assert session.agent == "antigravity-preview-09-2026"
    await rdb.set_phase(thread_id=2, phase="done")
    assert await _only_resumable(thread_id=2) is None
    assert await rdb.active_thread_for_owner(owner_id=1) is None


async def test_active_thread_for_owner_excludes_terminal() -> None:
    await rdb.insert_session(
        thread_id=10,
        owner_id=500,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent="antigravity-preview-09-2026",
        brief="b",
    )
    assert await rdb.active_thread_for_owner(owner_id=500) == 10
    await rdb.set_phase(thread_id=10, phase="done")
    assert await rdb.active_thread_for_owner(owner_id=500) is None
    assert await rdb.active_thread_for_owner(owner_id=12345) is None


async def test_list_resumable_only_returns_researching() -> None:
    # A researching session beside two terminal ones: only the first may come back resumable.
    seeded: tuple[tuple[int, ResearchPhase], ...] = (
        (20, "researching"),
        (21, "cancelled"),
        (22, "done"),
    )
    for thread_id, phase in seeded:
        await rdb.insert_session(
            thread_id=thread_id,
            owner_id=thread_id,
            channel_id=1,
            guild_id=1,
            source_message_id=1,
            agent="antigravity-preview-09-2026",
            brief="b",
        )
        await rdb.set_interaction(thread_id=thread_id, interaction_id="int_x")
        await rdb.set_phase(thread_id=thread_id, phase=phase)
    resumable = await rdb.list_resumable(created_before=database_now())
    assert {session.thread_id for session in resumable} == {20}


async def test_a_legacy_planning_row_no_longer_blocks_its_owner() -> None:
    # Written the way the removed escalation wrote it: the phase literal is gone from the model, so
    # seed it through the ORM. A stuck row must not hold the one-per-owner slot forever.
    async with rdb.open_session() as session:
        session.add(
            rdb.ResearchSessionRow(
                thread_id=60,
                owner_id=61,
                channel_id=1,
                guild_id=1,
                source_message_id=1,
                agent="deep-research-preview-04-2026",
                interaction_id="plan_1",
                brief="b",
                phase="planning",
            )
        )
        await session.commit()
    assert await rdb.active_thread_for_owner(owner_id=61) is None
    assert await rdb.list_resumable(created_before=database_now()) == []
