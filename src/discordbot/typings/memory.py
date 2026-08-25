"""Pure types for long-term memory: the observation vocabulary and one stored fact.

The write side speaks in *observations* (one conversational signal, phase-1) and the
read side speaks in *facts* (one distilled memory, one file). ``MemorySection`` is the
shared vocabulary between them: it is an ASCII key, never the rendered heading, so the
structured LLM schema stays English while the injected document stays Traditional
Chinese (the heading tables live in ``services/memory/facts.py``).

A fact's fields split into two ownership zones. The model authors ``summary``,
``section``, ``durability`` and the body, and names ``subject_id`` plus the keys a fact
distils from; everything else is stamped by code, which is the whole point of the
redesign — provenance the model cannot copy wrong. Stamped is not the same as hidden:
``MemoryFact`` below has what an update or delete is handed back and why. ``compartment``
is the exception that proves the ownership rule: it is stored only so a hand-edited or
half-migrated tree can be *detected*, and the containing directory always wins (see
``store.read_facts``).
"""

from typing import Literal
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

type MemoryCategory = Literal[
    "stable_preference", "stable_fact", "interaction_style", "recurring_pattern", "recent_context"
]
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
type MemoryConfidence = Literal["low", "medium", "high"]
type MemoryDurability = Literal["volatile", "session", "recent", "stable", "permanent"]
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
# section vocabulary. `allowlist_ids_from_server_memory` is NOT that reader and was not
# replaced by this: it still parses `## 成員稱呼` out of the rendered document. Today the
# freshness sweep is the only reader, exempting alias rows from aging.
type MemoryNodeType = Literal["memory", "member_alias"]

# What one consolidation delta asks for. `create` mints a fresh id, `update` and
# `delete` name an existing one.
type MemoryDeltaAction = Literal["create", "update", "delete"]


class MemoryOwner(BaseModel):
    """Who a scope's memory belongs to, stamped into every fact for human inspection.

    Attributes:
        owner_id: Discord id of the user or server.
        owner_name: Last-seen label, sanitized to one line by the renderer that built it.
    """

    model_config = ConfigDict(frozen=True)

    owner_id: int = Field(..., description="Discord id of the user or server.")
    owner_name: str = Field(..., description="Last-seen single-line label.", examples=["Alice"])


class MemoryFact(BaseModel):
    """One distilled memory, stored as a single file inside one compartment.

    The model authors `summary`, `section`, `durability` and `text`, except that a
    `member_alias` body is code-built and its `text` unread. Code owns provenance —
    `fact_id`, `compartment`, the `owner_*` and timestamp fields — and mints `fact_id`
    itself so conversation content can never influence it. Provenance is not hidden
    from the model: an update or delete has to name the id it is editing, so
    `render_existing_facts` shows the id, `keys` and `subject_id` of every stored fact
    and the delta schema takes them back.
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


class MemoryWriteSummary(BaseModel):
    """What one turn's memory work recorded, for the note shown under the reply.

    Structured rather than pre-rendered because the wording is a Discord concern and the
    pipeline is not: `services/` never composes what a user reads.

    `private` is a COUNT, not a list. A `source_only` observation is one the pipeline judged
    unsafe to repeat outside the conversation it came from, and while the reply is in that
    conversation, the note under it stays in the channel long after the exchange scrolls past.
    Saying how many were taken keeps the report honest without publishing them.

    Attributes:
        remembered: Summaries of the observations safe to name.
        private: How many were recorded but not named.
        forgotten: What the user asked to have dropped, in their own framing.
    """

    model_config = ConfigDict(frozen=True)

    remembered: tuple[str, ...] = Field(
        default=(), description="Summaries of the observations safe to name."
    )
    private: int = Field(default=0, description="Observations recorded but not named.")
    forgotten: tuple[str, ...] = Field(
        default=(), description="What the user asked to have dropped."
    )

    @property
    def is_empty(self) -> bool:
        """Whether the turn recorded nothing worth reporting."""
        return not self.remembered and not self.forgotten and self.private == 0


class MemoryCredits(BaseModel):
    """Who the usage footer credits for the memory one reply read.

    Two fields rather than one list because the footer treats them differently: a named user
    is worth printing, while an unnamed one is worth counting and nothing more. The unnamed
    are the table-only members of a server memory's `## 成員稱呼`, who carry no Discord label
    anywhere in the conversation and whose name cannot be fetched from anywhere trustworthy
    (`gen_reply/recall.py::RecallCandidate` carries why), so the alternative to counting
    them is publishing a raw snowflake nobody in the channel can resolve.

    Attributes:
        named: Short footer credits, in lookup order, for the users this reply can name.
        unnamed: How many further users it read and cannot name.
    """

    model_config = ConfigDict(frozen=True)

    named: tuple[str, ...] = Field(
        default=(), description="Short footer credits, in lookup order, for nameable users."
    )
    unnamed: int = Field(default=0, description="How many further users were read but unnamed.")

    @property
    def total(self) -> int:
        """How many users' memory this reply actually read."""
        return len(self.named) + self.unnamed


class MemoryConfig(BaseSettings):
    """Deployment switches for long-term memory, read from environment variables.

    Kept apart from `LLMConfig` because memory itself has no kill-switch — it is always
    on — and this is about how the store is kept on disk, not about a model call.

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
