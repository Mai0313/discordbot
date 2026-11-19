"""Pure types for long-term memory: observation vocabularies, facts, and config.

The write side records observations and the read side stores distilled facts (one file per
fact). `MemorySection` uses ASCII keys for structured LLM schemas that map to Traditional
Chinese headings at render time.
"""

from typing import Literal
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict, AliasChoices
from pydantic_settings import BaseSettings

# Whose memory a scope holds: one user's, or the bot's own memory of one server.
type MemoryFlavor = Literal["user", "server"]

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

# Fact document sections across both flavors: `profile`, `fact`, and `recent` are shared;
# all others are flavor-specific and rejected on the wrong flavor.
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

# Derived from `section` by code to let readers filter alias rows directly.
type MemoryNodeType = Literal["memory", "member_alias"]

type MemoryDeltaAction = Literal["create", "update", "delete"]

# `raw.md` section header for forget requests; kept out of `MemoryCategory` so observation
# readers never treat a forget instruction as a stored observation.
FORGET_REQUEST_CATEGORY = "forget_request"

# Required leading header for `tone.md`; rewrites lacking it are rejected.
TONE_HEADER = "## 語氣偏好"


class MemoryOwner(BaseModel):
    """Who a scope's memory belongs to, stamped into every fact for human inspection."""

    model_config = ConfigDict(frozen=True)

    owner_id: int = Field(..., description="Discord id of the user or server.")
    owner_name: str = Field(
        ...,
        description="Last-seen label, collapsed to one line before it arrives here.",
        examples=["Alice"],
    )


class MemoryFact(BaseModel):
    """One distilled memory, stored as a single file inside one compartment.

    The model authors `summary`, `section`, `durability`, and `text` (except `member_alias`
    rows, whose rendered body is code-built). Code stamps `fact_id`, `compartment`,
    `owner_id`, `owner_name`, `node_type`, `created`, and `last_confirmed`; updates and
    deletes still receive `fact_id`, `keys`, and `subject_id` to target existing rows.
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
        description=(
            "Compartment recorded at write time; a fact whose directory disagrees is skipped."
        ),
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
    """Structured summary of one turn's memory writes for the reply footer note.

    `private` records only a count rather than text, keeping `source_only` observations out
    of channel history.
    """

    model_config = ConfigDict(frozen=True)

    remembered: tuple[str, ...] = Field(
        default=(), description="Summaries of the observations safe to name."
    )
    private: int = Field(default=0, description="Observations recorded but not named.")
    forgotten: tuple[str, ...] = Field(
        default=(), description="What the user asked to have dropped, in their own framing."
    )


class MemoryCredits(BaseModel):
    """Users credited in the usage footer for memory read during a reply.

    Splits nameable users (`named`) from table-only `## 成員稱呼` members with no Discord
    label in the conversation (`unnamed`, counted only).
    """

    model_config = ConfigDict(frozen=True)

    named: tuple[str, ...] = Field(
        default=(), description="Short footer credits, in lookup order, for nameable users."
    )
    unnamed: int = Field(default=0, description="How many further users were read but unnamed.")

    @property
    def total(self) -> int:
        """Returns how many users' memory this reply read."""
        return len(self.named) + self.unnamed


class MemoryConfig(BaseSettings):
    """Deployment settings for long-term memory storage, read from environment variables."""

    git_history_enabled: bool = Field(
        default=True,
        description=(
            "Whether memory changes are committed to the store's git repository. Best-effort "
            "either way: the bot never creates that repository, and a missing one disables "
            "this silently."
        ),
        validation_alias=AliasChoices("MEMORY_GIT_ENABLED"),
    )
