"""Pure types for long-term memory: the observation vocabulary and one stored fact.

The write side speaks in *observations* (one conversational signal, phase-1) and the
read side speaks in *facts* (one distilled memory, one file). ``MemorySection`` is the
shared vocabulary between them: it is an ASCII key, never the rendered heading, so the
structured LLM schema stays English while the injected document stays Traditional
Chinese (the heading tables live in ``services/memory/facts.py``).

Nothing here carries behavior: every alias is a field on a structured LLM schema or on the
stored fact, and the rule keyed off it lives a layer up. ``services/memory/extraction.py``
declares the phase-1 ``MemoryObservation`` and the phase-2 ``MemoryFactDelta``, and runs the
deterministic gates over observations (over a delta it only scrubs secret-shaped strings);
``services/memory/deltas.py`` owns every delta gate, in ``_resolve_delta`` and
``apply_deltas``, routes a raw batch into compartments off the code-stamped ``- sharing:``
and ``- source:`` fields, and ages stored facts off ``section`` plus ``durability``;
``services/memory/facts.py`` defines the one-fact-per-file format and renders a compartment
set back into the document the reply prompts are given, while ``services/memory/store.py``
is what touches the filesystem.

A fact's fields split into two ownership zones. The model authors ``summary``,
``section``, ``durability`` and the body; everything else is stamped by code and is
never shown to it, which is the whole point of the redesign — provenance the model
cannot copy wrong. ``compartment`` is the exception that proves it: it is stored only
so a hand-edited or half-migrated tree can be *detected*, and the containing directory
always wins (see ``store.read_facts``).

``MemoryConfig`` rides along because it is deployment state about how the store is kept on
disk rather than about a model call, which is why it is not a field on ``LLMConfig``.
"""

from typing import Literal
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

# Which kind of memory an observation is aiming at, and the `### <category>` header its
# raw entry is filed under. `recent_context` is the odd one: code demotes it to
# unpromotable, `source_only` and TTL-bound whatever the model said. `interaction_style`
# and `stable_preference` are additionally what the tone note is fed from.
type MemoryCategory = Literal[
    "stable_preference", "stable_fact", "interaction_style", "recurring_pattern", "recent_context"
]

# The shape of the evidence behind an observation, and the phase-1 gate's main lever: the
# last five are rejected outright, `ongoing_situation` is accepted but locked to its source
# conversation, and the rest are what a stable fact is allowed to rest on.
type MemoryEvidenceKind = Literal[
    "explicit_preference",
    "repeated_behavior",
    "correction",
    "stable_fact",
    "recurring_pattern",
    "ongoing_situation",
    "tool_usage",
    "casual_mention",
    "hypothetical",
    "bot_suggestion",
    "other_user_context",
    "unknown",
]

# How sure phase 1 is. A stable observation needs `high` and a recent-context one at least
# `medium`, so `low` never reaches the store.
type MemoryConfidence = Literal["low", "medium", "high"]

# How long a memory is meant to survive, shortest first. `volatile` and `session` exist so
# the model can say "not worth keeping" and the gate can drop it deterministically; nothing
# below `recent` is ever staged. On a stored fact the freshness sweep reads this for two of
# its rules: `stable` falls out of a per-compartment window and `permanent` never ages. The
# TTL is keyed on the `recent` SECTION instead, not on this durability, and the `permanent`
# section and a `member_alias` node are exempt as well (`deltas.sweep_stale_facts`).
type MemoryDurability = Literal["volatile", "session", "recent", "stable", "permanent"]

# Whether an observation may leave the conversation it was learned in. It is half of what
# `partition_raw_entries` buckets a user batch on: this field picks global-vs-confined and
# the code-stamped `- source:` field then picks `dm` or `g/<guild_id>`, so between them they
# decide which compartment the fact distilled from it can be written into (a server batch
# carries neither and lands in that scope's single compartment). Code may only tighten this
# to `source_only`, never loosen a `source_only` call back to `global`.
type MemorySharing = Literal["global", "source_only"]

# Which section of the rendered document a fact belongs to. One vocabulary for both
# flavors: `profile`, `fact` and `recent` are shared, the rest are flavor-specific and
# are rejected by `sections_for_flavor` when they arrive on the wrong flavor.
type MemorySection = Literal[
    "profile",
    "permanent",
    "preference",
    "fact",
    "interaction",
    "culture",
    "topic",
    "member_alias",
    "recent",
]

# What a stored file holds. Derived from `section` by code so it can never disagree
# with it; kept as its own field so a reader can select alias rows without knowing the
# section vocabulary, which is how the freshness sweep exempts them from aging.
type MemoryNodeType = Literal["memory", "member_alias"]

# What one consolidation delta asks for. `create` mints a fresh id, `update` and
# `delete` name an existing one.
type MemoryDeltaAction = Literal["create", "update", "delete"]


class MemoryOwner(BaseModel):
    """Who a scope's memory belongs to, stamped into every fact for human inspection.

    Nothing is authorized off this pair: the compartment directory is the privacy boundary,
    and the scope key already says whose memory is being read. It travels through the
    `memory_job` round-trip as one identity line, so `facts.parse_identity` and
    `facts.render_owner_identity` are the two ends of the same format.

    Attributes:
        owner_id: Discord id of the user or server.
        owner_name: Last-seen label, sanitized to one line by the renderer that built it.
    """

    model_config = ConfigDict(frozen=True)

    owner_id: int = Field(..., description="Discord id of the user or server.")
    owner_name: str = Field(..., description="Last-seen single-line label.", examples=["Alice"])


class MemoryFact(BaseModel):
    """One distilled memory, stored as a single file inside one compartment.

    Attributes carry their zone in the description: the model authors `summary`, `section`,
    `durability` and `text`, and code stamps the rest. `fact_id` is the filename stem, which
    is why it is minted from a hash of the compartment and the summary rather than chosen by
    the model; conversation content that reached it would put path traversal one injection
    away.

    Instances are frozen, so a consolidation delta replaces the whole file rather than
    editing one in place (`facts.render_fact_file` renders the text and `store.write_fact`
    writes it, tmp file then `os.replace`; `facts.parse_fact_file` reads it back and
    `store.read_facts` a compartment at a time). The freshness sweep writes nothing: it
    deletes the file outright through `store.delete_fact`.
    """

    model_config = ConfigDict(frozen=True)

    fact_id: str = Field(
        ...,
        description="Code-minted 16-hex identifier; also the filename stem.",
        examples=["9f2c41a7be03d5e8"],
    )
    summary: str = Field(
        ..., description="One-line gist, used for ordering and for the model's own index."
    )
    section: MemorySection = Field(
        ..., description="Which document section this fact renders under.", examples=["preference"]
    )
    durability: MemoryDurability = Field(
        ..., description="How the freshness sweep treats it.", examples=["stable", "permanent"]
    )
    text: str = Field(..., description="The fact body, as rendered into the reply prompt.")
    compartment: str = Field(
        ...,
        description="Compartment recorded at write time; the containing directory still wins.",
        examples=["global", "g/123456789012345678", "dm"],
    )
    owner_id: int = Field(
        ..., description="Discord id of the user or server this memory is about."
    )
    owner_name: str = Field(
        ..., description="Last-seen owner label, for human inspection only.", examples=["Alice"]
    )
    subject_id: int | None = Field(
        default=None,
        description="Member a nickname row refers to; None for an ordinary fact.",
        examples=[987654321098765432],
    )
    node_type: MemoryNodeType = Field(
        default="memory", description="Derived from `section`; selects alias rows."
    )
    created: datetime = Field(..., description="When the fact was first written.")
    last_confirmed: datetime = Field(
        ..., description="When evidence last confirmed it; drives the freshness sweep."
    )
    keys: tuple[str, ...] = Field(
        default=(),
        description="normalized_keys this fact distils from; the retry-idempotency handle.",
        examples=[("preference.reply_language.zh_tw",)],
    )


class MemoryConfig(BaseSettings):
    """Deployment switches for long-term memory, read from environment variables.

    Kept apart from `LLMConfig` because memory itself has no kill-switch — it is always
    on — and this is about how the store is kept on disk, not about a model call. Read
    once at import, where `services/memory/git_history.py` builds the single service the
    process commits through, so a change to it takes effect on the next restart.

    Attributes:
        git_history_enabled: Whether a successful consolidation commits the scope it
            wrote to the store's own git repository. Best-effort either way: the bot
            never creates the repository, and a missing one disables this silently.
    """

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    git_history_enabled: bool = Field(
        default=True,
        description="Whether memory changes are committed to the store's git repository.",
        validation_alias=AliasChoices("MEMORY_GIT_ENABLED"),
    )
