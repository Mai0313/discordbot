"""Builders for the memory store's stored facts and consolidation deltas, a row reader, the
`/memory` cog, a fake model client, and waits for the pipeline's background work.

Every default is an ordinary per-user preference, so a test names only the fields it is about.
A fact carries no default owner: the owner is what a scope's stored identity is read back from,
so each test module binds its own.
"""

from types import SimpleNamespace
from typing import Protocol
import asyncio
from datetime import UTC, datetime

from pydantic import BaseModel
from sqlalchemy import select

from discordbot.typings.memory import (
    MemoryFact,
    MemoryOwner,
    MemorySection,
    MemoryDurability,
    MemoryDeltaAction,
)
from discordbot.cogs.memory.cog import MemoryCogs
from discordbot.services.memory import inflight
from discordbot.services.memory.facts import node_type_for
from discordbot.services.memory.store import GLOBAL_COMPARTMENT
from discordbot.services.memory.writer import MemoryFactDelta
from discordbot.services.memory.database import (
    MemoryJob,
    MemoryJobRow,
    open_session,
    _row_to_model,
)

from tests.helpers.casting import as_bot

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


async def assert_cleared_row(scope: str) -> MemoryJob:
    """Asserts a scope's `memory_job` row is a clear tombstone keeping nothing of the turn.

    Returns:
        The tombstone row.
    """
    job = await get_job(scope=scope)
    assert job is not None
    assert job.status == "cleared"
    assert job.transcript is None
    assert job.subject == ""
    assert job.identity == ""
    assert job.last_error is None
    return job


async def drain_memory_turns(scopes: tuple[str, ...]) -> None:
    """Awaits every memory turn queued for `scopes`, the deferred replays included.

    A failed turn is not re-raised. Each replay is started by the previous task's
    done-callback, so the next task only exists once the loop has run that callback.
    """
    for scope in scopes:
        while (task := inflight._inflight_tasks.get(key=scope)) is not None:
            await asyncio.gather(task, return_exceptions=True)


async def wait_for_persisted_writes() -> None:
    """Drains the pipeline's detached reply.db writes, for a DEFERRED turn's row.

    An ordinary turn transitions its row from the in-flight review task itself, so awaiting
    that task is enough. A deferred one stages its row, and a cleared one retires it, from a
    fire-and-forget task instead, so there a finished turn says nothing about the scope's
    `memory_job` row: reading it too early sees a state the writer is about to move on its own
    `cleared_since` check.
    """
    while inflight._db_tasks:
        await asyncio.gather(*list(inflight._db_tasks))


class MemoryAnswer(Protocol):
    """The model a test stages: one parsed output per request body and requested schema."""

    async def __call__(self, body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Returns the call's parsed output; None is a call that produced nothing usable."""
        ...


class FakeMemoryResponses:
    """Fake Responses API resource recording parse calls for memory tests.

    Every call answers with `output_parsed` unless `answer` is set, in which case `answer`
    decides from the request's user text and the schema it asked for.
    """

    def __init__(self) -> None:
        """Initializes recorded calls and the configured parsed output."""
        self.parse_models: list[str] = []
        self.parse_instructions: list[str] = []
        self.parse_bodies: list[str] = []
        self.parse_extra_kwargs: list[dict[str, object]] = []
        self.output_parsed: BaseModel | None = None
        self.answer: MemoryAnswer | None = None
        self.status: str = "completed"
        self.raises: Exception | None = None

    async def parse(  # noqa: PLR0913 -- mirrors Responses API parse signature
        self,
        model: str,
        instructions: str,
        input: list[dict[str, str]],  # noqa: A002 -- SDK parameter
        text_format: type[BaseModel],
        reasoning: dict[str, str],
        service_tier: str,
        extra_headers: dict[str, str],
        **unexpected: object,
    ) -> SimpleNamespace:
        """Records the call and returns or raises the configured result.

        `**unexpected` captures any kwarg the memory calls are not expected to
        pass (e.g. a reintroduced `max_output_tokens`) so a test can assert the
        memory path leaves the output budget to the backend.
        """
        del reasoning, service_tier, extra_headers
        body = input[0]["content"]
        self.parse_models.append(model)
        self.parse_instructions.append(instructions)
        self.parse_bodies.append(body)
        self.parse_extra_kwargs.append(unexpected)
        if self.raises is not None:
            raise self.raises
        output = (
            self.output_parsed
            if self.answer is None
            else await self.answer(body=body, text_format=text_format)
        )
        return SimpleNamespace(output_parsed=output, status=self.status, incomplete_details=None)


class FakeMemoryClient:
    """Fake OpenAI client exposing only the responses resource."""

    def __init__(self) -> None:
        """Initializes the fake responses resource."""
        self.responses = FakeMemoryResponses()


def make_memory_cog() -> MemoryCogs:
    """Builds a MemoryCogs instance around a stub bot.

    `get_guild` answers None so a guild compartment's heading falls back to its id, the
    same way it would for a server the bot has since left.
    """
    return MemoryCogs(bot=as_bot(fake=SimpleNamespace(get_guild=lambda _guild_id: None)))
