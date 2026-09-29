"""The context one consolidation or rebuild runs in, built once and handed to every step."""

from datetime import UTC, datetime

from pydantic import Field, BaseModel, ConfigDict, SkipValidation

from discordbot.typings.memory import MemoryOwner, MemoryFlavor
from discordbot.services.memory.facts import parse_identity
from discordbot.services.memory.store import flavor_of, scope_owner_id
from discordbot.services.memory.writer import MemoryWriterAI


class ConsolidationRun(BaseModel):
    """Whose memory one consolidation or rebuild writes, through which writer, and since when.

    Every compartment call, forget pass and tone call of the run reads it from here, so they all
    stamp the same owner, date their requests the same day, and abort on the same clear: one
    stamped at or after `started_at`.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    scope: str = Field(..., description="The memory scope the run writes into.")
    writer: SkipValidation[MemoryWriterAI] = Field(
        ..., description="The memory writing service running the run's LLM calls."
    )
    owner: MemoryOwner = Field(
        ..., description="The owner stamped onto every fact the run writes."
    )
    started_at: float = Field(
        ..., description="`time.monotonic()` the run's `cleared_since` checks compare against."
    )
    today: str = Field(
        ..., description="ISO date the run's requests are dated with.", examples=["2026-09-30"]
    )

    @property
    def flavor(self) -> MemoryFlavor:
        """The scope's flavor, derived so it cannot disagree with `scope`."""
        return flavor_of(scope=self.scope)


def start_run(
    scope: str, writer: MemoryWriterAI, identity: str, started_at: float
) -> ConsolidationRun:
    """Builds a run's context from the identity line the pipeline threads around.

    A line that does not parse keeps the id the scope key carries (`parse_identity`).
    """
    return ConsolidationRun(
        scope=scope,
        writer=writer,
        owner=parse_identity(identity=identity, fallback_owner_id=scope_owner_id(scope=scope)),
        started_at=started_at,
        today=datetime.now(UTC).date().isoformat(),
    )
