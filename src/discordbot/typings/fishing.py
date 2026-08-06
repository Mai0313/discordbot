"""Shared vocabulary for the fishing mini-game: rarity grades, catalog rows, and cast outcomes.

Pure pydantic models, enums, and tuning constants with no cog or util dependency, which is what
lets the store (`cogs/games/fishing/database.py`), the RNG-injected roll engine (`catch.py`), the
panel views, and the offline seeding script all key off one set of names.

Four families live here:

- The enums are the persisted vocabulary. `FishGrade` is the rarity ladder as stored; a grade's
  emoji, color, label and rank live in its catalog row instead, so retuning the ladder is a data
  edit rather than a code change. `GearType` splits the single gear table into rods and bait.
  `CastStatus` names every way one cast can end.
- The `*View` models are frozen read models the store builds out of its own ORM rows and hands
  upward. They carry no field bounds, because the row they mirror is already the source of
  truth, and they are what lets the views and the pure engine work without importing ORM models.
  `FishingCatalog` bundles a whole set of them as one value, which is how `defaults.py` hands
  the default catalog to a caller that wants the rows without a database.
- The outcome models are what the engine and the store hand back. `CatchRoll` is one pure roll
  before any persistence; `PurchaseResult` and `CastResult` are settled outcomes carrying the
  wallet, durability and inventory state the caller has to show; `FishingPanelData` is the one
  aggregate, bundling the per-user views a panel render reads.
- The `*Upsert` models are the write half, reached only offline (`defaults.py` ->
  `scripts/seed_fishing.py` -> the store's `upsert_*` calls). Every field bound and both
  cross-field validators sit there because a hand-written catalog is the only way a row gets
  into the tables; runtime never seeds.

The constants are shared tuning, pinned here because a constant more than one module keys off
belongs under `typings/`. The engine reads the luck clamp and the basis-point denominator; the
store reads the per-catch cap it hands `roll_catch` and the per-purchase bait cap it enforces;
the Discord surface reads the panel idle timeout, the same denominator, and the bait cap for an
up-front check that never replaces the store's.
"""

from enum import StrEnum
from typing import Self, Final
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict, model_validator

# Idle timeout before the public fishing panel deletes itself, matching the
# stock panel and game-table convention.
FISHING_ACTION_TIMEOUT_SECONDS: Final[int] = 180
# Shared basis-point denominator for luck and value math.
FISHING_BPS_DENOMINATOR: Final[int] = 10_000
# Hard ceiling on a single catch payout, the fishing sibling of MAX_SINGLE_BET.
# Fishing is high-frequency and can be left running, so this is set well below
# the casino bet cap to bound how much one lucky cast can mint.
FISHING_MAX_SINGLE_CATCH: Final[int] = 100_000
# Luck clamp, applied per rarity step rather than to the finished factor: with
# any gear equipped, one step up the rarity ladder is worth at least 0.5x and at
# most 3.0x the step below. A grade's own factor is this raised to how many steps
# it sits above the commonest grade, so the clamp bounds a mis-typed gear shift
# without flattening the ladder the compounding exists to create. What bounds a
# mis-typed `order_index` is separate and lives in `compose_grade_weights`.
LUCK_STEP_MIN_BPS: Final[int] = 5_000
LUCK_STEP_MAX_BPS: Final[int] = 30_000
# Most bait a single purchase can stack, so one buy cannot lock a huge spend.
MAX_BAIT_PER_PURCHASE: Final[int] = 100


class FishGrade(StrEnum):
    """Rarity grades for fish, from common to mythic."""

    N = "N"
    R = "R"
    SR = "SR"
    SSR = "SSR"
    UR = "UR"


class GearType(StrEnum):
    """Kinds of purchasable fishing gear."""

    ROD = "rod"
    BAIT = "bait"


class CastStatus(StrEnum):
    """Outcome status of one cast attempt.

    `PAYOUT_DEFERRED` is the cross-database seam rather than a rules outcome: the catch is
    already committed to games.db and only the economy credit failed, so the cast counts and
    the payout is owed. Nothing retries it, which is why `settle_cast` reports it instead of
    rolling the catch back.
    """

    SUCCESS = "success"
    NO_ROD = "no_rod"
    BROKEN_ROD = "broken_rod"
    NO_BAIT = "no_bait"
    PAYOUT_DEFERRED = "payout_deferred"


class FishGradeConfigView(BaseModel):
    """Read-only grade roll weight and display metadata.

    `order_index` is read two different ways, both of them inside the engine: `roll_catch` sorts
    the grade choices by the raw value and `_fallback_species` compares it as a rarity rank,
    while `compose_grade_weights` compounds luck over a grade's POSITION in `order_index` order,
    never over the raw value. That is what lets an operator leave gaps in it, and a grade whose
    weight is zero is disabled but keeps its position, so removing one never re-spaces the rest.
    The store lists grades in `order_index` order too, but that ordering reaches nothing:
    `get_grade_config_map` collapses it into a dict before any embed builder sees it.
    """

    model_config = ConfigDict(frozen=True)

    grade: FishGrade = Field(..., description="Rarity grade this config applies to.")
    weight: int = Field(..., description="Base roll weight relative to other grades.")
    color: int = Field(..., description="Embed color for this grade as an 0xRRGGBB integer.")
    emoji: str = Field(..., description="Leading emoji shown for this grade.")
    label: str = Field(..., description="Localized display label for this grade.")
    order_index: int = Field(
        ...,
        description="Rarity rank; higher means rarer and drives luck weighting and display order.",
    )


class FishSpeciesView(BaseModel):
    """Read-only fish species catalog row."""

    model_config = ConfigDict(frozen=True)

    species_id: str = Field(..., description="Stable identifier for the fish species.")
    name: str = Field(..., description="Display name of the fish species.")
    grade: FishGrade = Field(..., description="Rarity grade of the species.")
    emoji: str = Field(..., description="Emoji shown for the species.")
    intra_grade_weight: int = Field(
        ..., description="Roll weight of this species relative to others in the same grade."
    )
    base_value: int = Field(..., description="Base sell value before the size multiplier.")
    size_min_bps: int = Field(
        ..., description="Minimum size multiplier in basis points, e.g. 5000 for 0.5x."
    )
    size_max_bps: int = Field(
        ..., description="Maximum size multiplier in basis points, e.g. 20000 for 2.0x."
    )
    image_key: str = Field(
        default="",
        description="Optional key for a future rendered image; emoji is used when empty.",
    )


class GearView(BaseModel):
    """Read-only fishing gear catalog row for a rod or bait."""

    model_config = ConfigDict(frozen=True)

    gear_id: str = Field(..., description="Stable identifier for the gear item.")
    gear_type: GearType = Field(..., description="Whether this gear is a rod or a bait.")
    name: str = Field(..., description="Display name of the gear item.")
    emoji: str = Field(..., description="Emoji shown for the gear item.")
    tier: int = Field(..., description="Relative power tier of the gear item.")
    price: int = Field(..., description="Purchase price in currency, burned on purchase.")
    rarity_shift_bps: int = Field(
        ...,
        description="Luck shift in basis points applied to rarer grades when this gear is used.",
    )
    durability: int = Field(
        default=0, description="Number of casts a rod lasts; always zero for bait."
    )
    value_bonus_bps: int = Field(
        default=0, description="Catch value bonus in basis points; used by bait, zero for rods."
    )


class BaitStackView(BaseModel):
    """Read-only owned bait stack for one user."""

    model_config = ConfigDict(frozen=True)

    bait_id: str = Field(..., description="Identifier of the owned bait.")
    name: str = Field(..., description="Display name of the bait.")
    emoji: str = Field(..., description="Emoji shown for the bait.")
    quantity: int = Field(..., description="Number of this bait the user currently owns.")


class AnglerStateView(BaseModel):
    """Read-only per-user fishing state."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(..., description="Discord user ID of the angler.")
    user_name: str = Field(default="", description="Last-seen display name of the angler.")
    rod: GearView | None = Field(
        default=None, description="Currently equipped rod, or None when the angler has no rod."
    )
    durability_remaining: int = Field(
        default=0, description="Remaining casts on the equipped rod before it breaks."
    )
    total_casts: int = Field(default=0, description="Lifetime number of casts made.")
    total_catch_value: int = Field(default=0, description="Lifetime gross value of all catches.")
    total_spent_on_gear: int = Field(
        default=0, description="Lifetime currency spent buying rods and bait."
    )
    best_catch_value: int = Field(default=0, description="Highest single-catch value achieved.")


class CatchRoll(BaseModel):
    """Pure result of one fish roll before any persistence.

    `size_bps` and `size_rank_bps` answer different questions and are both kept. Size bands are
    per grade, so the raw multiplier does not say whether a catch was a big one for its kind;
    the rank inside the species' own band does, and it is what the 大物 marker reads.
    """

    model_config = ConfigDict(frozen=True)

    species_id: str = Field(..., description="Identifier of the rolled species.")
    species_name: str = Field(..., description="Display name of the rolled species.")
    grade: FishGrade = Field(..., description="Rarity grade of the rolled species.")
    emoji: str = Field(..., description="Emoji of the rolled species.")
    size_bps: int = Field(..., description="Rolled size multiplier in basis points.")
    size_rank_bps: int = Field(
        ..., description="Where the rolled size sits inside the species' own band, 0 to 10000."
    )
    base_value: int = Field(..., description="Species base value before the size multiplier.")
    value: int = Field(
        ..., description="Final payout after size, bait bonus, and the single-catch cap."
    )
    capped: bool = Field(
        ..., description="Whether the single-catch cap reduced the otherwise-higher value."
    )


class CatchLogView(BaseModel):
    """Read-only catch record for leaderboard and history."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(..., description="Discord user ID of the angler who made the catch.")
    user_name: str = Field(..., description="Stored display name of the angler.")
    species_id: str = Field(..., description="Identifier of the caught species.")
    species_name: str = Field(..., description="Stored display name of the caught species.")
    grade: FishGrade = Field(..., description="Rarity grade of the catch.")
    emoji: str = Field(..., description="Stored emoji of the catch.")
    size_bps: int = Field(..., description="Size multiplier of the catch in basis points.")
    value: int = Field(..., description="Final catch value paid to the angler.")
    created_at: datetime = Field(..., description="Timestamp the catch was recorded.")


class PurchaseResult(BaseModel):
    """Outcome of buying a rod or bait."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(..., description="Whether the purchase completed and gear was granted.")
    gear_id: str = Field(..., description="Identifier of the gear that was bought.")
    gear_type: GearType | None = Field(
        default=None, description="Type of gear bought, or None when the purchase failed early."
    )
    quantity: int = Field(default=0, description="Quantity granted by the purchase.")
    total_cost: int = Field(default=0, description="Total currency burned by the purchase.")
    new_balance: int = Field(default=0, description="Wallet balance after the purchase.")
    reason: str = Field(
        default="", description="Failure reason when the purchase did not complete."
    )


class CastResult(BaseModel):
    """Outcome of one cast attempt."""

    model_config = ConfigDict(frozen=True)

    status: CastStatus = Field(..., description="Status of the cast attempt.")
    roll: CatchRoll | None = Field(
        default=None, description="The rolled catch when the cast succeeded, else None."
    )
    payout: int = Field(default=0, description="Currency credited for the catch.")
    new_balance: int = Field(default=0, description="Wallet balance after the catch payout.")
    rod_broke: bool = Field(default=False, description="Whether the rod broke on this cast.")
    durability_remaining: int = Field(
        default=0, description="Remaining rod durability after the cast."
    )
    bait_id: str = Field(default="", description="Identifier of the bait consumed by the cast.")
    bait_remaining: int = Field(
        default=0, description="Remaining quantity of the consumed bait after the cast."
    )


class FishingPanelData(BaseModel):
    """Aggregated state for rendering the main fishing panel."""

    model_config = ConfigDict(frozen=True)

    balance: int = Field(..., description="Angler's current wallet balance.")
    angler: AnglerStateView = Field(..., description="Angler's rod and lifetime fishing state.")
    baits: tuple[BaitStackView, ...] = Field(
        ..., description="Owned bait stacks with positive quantity."
    )
    last_catch: CatchLogView | None = Field(
        default=None, description="Most recent catch for the angler, if any."
    )


class FishGradeConfigUpsert(BaseModel):
    """DB-owned grade config payload for maintenance scripts."""

    model_config = ConfigDict(frozen=True)

    grade: FishGrade = Field(
        ..., description="Rarity grade this config applies to.", examples=["SR"]
    )
    weight: int = Field(
        ge=0, description="Base roll weight relative to other grades.", examples=[800]
    )
    color: int = Field(
        ge=0, description="Embed color as an 0xRRGGBB integer.", examples=[0x9B59B6]
    )
    emoji: str = Field(
        min_length=1, max_length=32, description="Leading emoji for this grade.", examples=["🟣"]
    )
    label: str = Field(
        min_length=1, max_length=32, description="Display label for this grade.", examples=["史詩"]
    )
    order_index: int = Field(ge=0, description="Rarity rank; higher means rarer.", examples=[2])


class FishSpeciesUpsert(BaseModel):
    """DB-owned fish species payload for maintenance scripts."""

    model_config = ConfigDict(frozen=True)

    species_id: str = Field(
        min_length=1, max_length=32, description="Stable species identifier.", examples=["carp"]
    )
    name: str = Field(min_length=1, max_length=64, description="Display name.", examples=["鯉魚"])
    grade: FishGrade = Field(..., description="Rarity grade of the species.", examples=["R"])
    emoji: str = Field(min_length=1, max_length=32, description="Species emoji.", examples=["🐠"])
    intra_grade_weight: int = Field(
        ge=1, description="Roll weight within the grade.", examples=[70]
    )
    base_value: int = Field(ge=0, description="Base value before size multiplier.", examples=[10])
    size_min_bps: int = Field(
        ge=1, description="Minimum size multiplier in basis points.", examples=[5000]
    )
    size_max_bps: int = Field(
        ge=1, description="Maximum size multiplier in basis points.", examples=[20000]
    )
    image_key: str = Field(default="", max_length=64, description="Optional future image key.")

    @model_validator(mode="after")
    def validate_size_range(self) -> Self:
        """Ensures the size multiplier range is well-ordered.

        The roll engine draws a size uniformly across this band, so an inverted one would raise
        from `random.randint` mid-cast; rejecting it here keeps it out of the table instead.

        Returns:
            The validated payload.

        Raises:
            ValueError: `size_min_bps` is greater than `size_max_bps`.
        """
        if self.size_min_bps > self.size_max_bps:
            msg = "size_min_bps cannot exceed size_max_bps"
            raise ValueError(msg)
        return self


class GearUpsert(BaseModel):
    """DB-owned fishing gear payload for maintenance scripts."""

    model_config = ConfigDict(frozen=True)

    gear_id: str = Field(
        min_length=1, max_length=32, description="Stable gear identifier.", examples=["rod_bamboo"]
    )
    gear_type: GearType = Field(
        ..., description="Whether the gear is a rod or a bait.", examples=["rod"]
    )
    name: str = Field(min_length=1, max_length=64, description="Display name.", examples=["竹竿"])
    emoji: str = Field(min_length=1, max_length=32, description="Gear emoji.", examples=["🎋"])
    tier: int = Field(ge=0, description="Relative power tier.", examples=[0])
    price: int = Field(ge=1, description="Purchase price in currency.", examples=[300])
    rarity_shift_bps: int = Field(ge=0, description="Luck shift in basis points.", examples=[0])
    durability: int = Field(ge=0, description="Casts a rod lasts; zero for bait.", examples=[30])
    value_bonus_bps: int = Field(
        ge=0, description="Catch value bonus in basis points for bait.", examples=[0]
    )

    @model_validator(mode="after")
    def validate_type_fields(self) -> Self:
        """Ensures durability is rod-only and bait carries no durability.

        A rod is granted with its catalog durability as its remaining casts, so a zero would
        sell a rod that answers `BROKEN_ROD` on its very first cast. Bait is consumed one per
        cast and its durability is never read, so a non-zero one is rejected rather than kept
        as dead data that reads like a rod's.

        Returns:
            The validated payload.

        Raises:
            ValueError: A rod carries durability below 1, or a bait carries non-zero durability.
        """
        if self.gear_type == GearType.ROD and self.durability < 1:
            msg = "rod gear must have durability of at least 1"
            raise ValueError(msg)
        if self.gear_type == GearType.BAIT and self.durability != 0:
            msg = "bait gear must have durability of 0"
            raise ValueError(msg)
        return self


class FishingCatalog(BaseModel):
    """Default catalog of grades, species, and gear bundled as one value.

    `build_default_catalog` is its only producer and tests are its only consumers; nothing under
    `src/` outside `defaults.py` touches it. The seed script takes the same rows through the
    `*Upsert` payloads instead, since those are what carry the field bounds.
    """

    model_config = ConfigDict(frozen=True)

    grades: tuple[FishGradeConfigView, ...] = Field(..., description="Default grade configs.")
    species: tuple[FishSpeciesView, ...] = Field(..., description="Default fish species rows.")
    gear: tuple[GearView, ...] = Field(..., description="Default rod and bait rows.")


__all__ = [
    "FISHING_ACTION_TIMEOUT_SECONDS",
    "FISHING_BPS_DENOMINATOR",
    "FISHING_MAX_SINGLE_CATCH",
    "LUCK_STEP_MAX_BPS",
    "LUCK_STEP_MIN_BPS",
    "MAX_BAIT_PER_PURCHASE",
    "AnglerStateView",
    "BaitStackView",
    "CastResult",
    "CastStatus",
    "CatchLogView",
    "CatchRoll",
    "FishGrade",
    "FishGradeConfigUpsert",
    "FishGradeConfigView",
    "FishSpeciesUpsert",
    "FishSpeciesView",
    "FishingCatalog",
    "FishingPanelData",
    "GearType",
    "GearUpsert",
    "GearView",
    "PurchaseResult",
]
