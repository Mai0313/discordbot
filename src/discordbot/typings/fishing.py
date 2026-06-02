"""Pure domain types, catalog, and tuning constants for the fishing mini-game.

This module holds everything that does not depend on cogs, utils, nextcord, or
the database: the rarity tiers, the fish / rod / bait catalogs, the base rarity
distribution, and the read-only result/view payloads returned by the fishing
database layer. The tuning numbers here are deliberately set so that every
`(rod, bait)` loadout has a negative per-cast expected value, which is the
anti-inflation invariant the fishing game exists to enforce. Re-balance with
`scripts/simulate_fishing.py` and keep the `tests/test_fishing.py` EV tests
green.
"""

from typing import Final, Literal

from pydantic import Field, BaseModel, ConfigDict

Rarity = Literal["N", "R", "SR", "SSR", "UR"]
CastStatus = Literal["ok", "no_rod", "no_bait"]
BuyStatus = Literal["ok", "insufficient", "unknown_item", "invalid_quantity"]
SellStatus = Literal["ok", "nothing"]

# Rarity tiers ordered from most common to rarest.
RARITY_ORDER: Final[tuple[Rarity, ...]] = ("N", "R", "SR", "SSR", "UR")

# Embed accent colour per rarity (matches the in-progress / settlement palette).
RARITY_COLOR: Final[dict[Rarity, int]] = {
    "N": 0x95A5A6,
    "R": 0x3498DB,
    "SR": 0x9B59B6,
    "SSR": 0xF1C40F,
    "UR": 0xE74C3C,
}

# Decorative badge per rarity used in text-only embeds.
RARITY_EMOJI: Final[dict[Rarity, str]] = {
    "N": "⚪",
    "R": "🔵",
    "SR": "🟣",
    "SSR": "🟡",
    "UR": "🔴",
}

# Base rarity distribution in basis points, applied after the miss/空竿 roll.
# Sums to 10,000; multiplied by rod and bait weight multipliers then renormalised.
BASE_RARITY_BPS: Final[dict[Rarity, int]] = {"N": 6000, "R": 3000, "SR": 800, "SSR": 180, "UR": 20}

# Denominator for every basis-point figure in this module.
BPS_DENOMINATOR: Final[int] = 10_000

# A rarity weight multiplier of 100 means 1.0x; 300 means 3.0x. Stored as int so
# the catalog stays free of floats and the roll can renormalise with pure ints.
WEIGHT_UNIT: Final[int] = 100

# Upper bound on a single bait purchase to keep the shop modal sane.
MAX_BAIT_PURCHASE_QUANTITY: Final[int] = 99


class FishSpecies(BaseModel):
    """One catchable fish species (or junk item) in the catalog."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(description="Stable identifier persisted on catch rows.")
    name: str = Field(description="Display name shown in embeds and the dex.")
    rarity: Rarity = Field(description="Rarity tier this species belongs to.")
    emoji: str = Field(description="Decorative emoji shown beside the species name.")
    min_mm: int = Field(description="Smallest possible size in millimetres.")
    max_mm: int = Field(description="Largest possible size in millimetres.")
    base_value: int = Field(description="Sell value in 虛擬歡樂豆 at the minimum size.")
    value_per_mm: int = Field(
        description="Extra sell value in 虛擬歡樂豆 for each millimetre above the minimum size."
    )


class RodTier(BaseModel):
    """One purchasable fishing rod tier with durability and odds modifiers."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(description="Stable identifier persisted on the angler row.")
    name: str = Field(description="Display name shown in embeds and the shop.")
    emoji: str = Field(description="Decorative emoji shown beside the rod name.")
    cost: int = Field(description="One-time purchase cost in 虛擬歡樂豆.")
    durability: int = Field(description="Number of casts before the rod breaks.")
    miss_bps: int = Field(description="Base 空竿 (no catch) chance in basis points for this rod.")
    rarity_weight: dict[Rarity, int] = Field(
        description="Per-rarity weight multiplier where 100 means 1.0x; renormalised at roll time."
    )


class BaitType(BaseModel):
    """One purchasable bait type, consumed one per cast, nudging the odds."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(description="Stable identifier persisted on the bait inventory row.")
    name: str = Field(description="Display name shown in embeds and the shop.")
    emoji: str = Field(description="Decorative emoji shown beside the bait name.")
    cost: int = Field(description="Purchase cost in 虛擬歡樂豆 per single bait.")
    miss_bps_delta: int = Field(
        description="Adjustment to the rod's 空竿 chance in basis points (negative reduces misses)."
    )
    rarity_weight: dict[Rarity, int] = Field(
        description="Per-rarity weight multiplier where 100 means 1.0x; renormalised at roll time."
    )


# --- Catalog -----------------------------------------------------------------

FISH_CATALOG: Final[tuple[FishSpecies, ...]] = (
    FishSpecies(
        key="boot",
        name="破雨鞋",
        rarity="N",
        emoji="👢",
        min_mm=50,
        max_mm=200,
        base_value=5,
        value_per_mm=0,
    ),
    FishSpecies(
        key="seaweed",
        name="海帶結",
        rarity="N",
        emoji="🥬",
        min_mm=50,
        max_mm=200,
        base_value=5,
        value_per_mm=0,
    ),
    FishSpecies(
        key="minnow",
        name="小鯽魚",
        rarity="N",
        emoji="🐟",
        min_mm=50,
        max_mm=200,
        base_value=5,
        value_per_mm=0,
    ),
    FishSpecies(
        key="crucian",
        name="肥鯽魚",
        rarity="R",
        emoji="🐠",
        min_mm=150,
        max_mm=400,
        base_value=30,
        value_per_mm=1,
    ),
    FishSpecies(
        key="squid",
        name="害羞魷魚",
        rarity="R",
        emoji="🦑",
        min_mm=150,
        max_mm=400,
        base_value=30,
        value_per_mm=1,
    ),
    FishSpecies(
        key="crab",
        name="橫行蟹",
        rarity="R",
        emoji="🦀",
        min_mm=150,
        max_mm=400,
        base_value=30,
        value_per_mm=1,
    ),
    FishSpecies(
        key="puffer",
        name="氣噗噗河豚",
        rarity="SR",
        emoji="🐡",
        min_mm=250,
        max_mm=600,
        base_value=120,
        value_per_mm=2,
    ),
    FishSpecies(
        key="tropical",
        name="花襯衫魚",
        rarity="SR",
        emoji="🐠",
        min_mm=250,
        max_mm=600,
        base_value=120,
        value_per_mm=2,
    ),
    FishSpecies(
        key="octopus",
        name="八爪外送員",
        rarity="SR",
        emoji="🐙",
        min_mm=250,
        max_mm=600,
        base_value=120,
        value_per_mm=2,
    ),
    FishSpecies(
        key="goldfish",
        name="黃金錦鯉",
        rarity="SSR",
        emoji="🎏",
        min_mm=400,
        max_mm=900,
        base_value=600,
        value_per_mm=4,
    ),
    FishSpecies(
        key="shark",
        name="迷你鯊老闆",
        rarity="SSR",
        emoji="🦈",
        min_mm=400,
        max_mm=900,
        base_value=600,
        value_per_mm=4,
    ),
    FishSpecies(
        key="lobster",
        name="龍蝦董事長",
        rarity="SSR",
        emoji="🦞",
        min_mm=400,
        max_mm=900,
        base_value=600,
        value_per_mm=4,
    ),
    FishSpecies(
        key="dragon",
        name="錦鯉之神",
        rarity="UR",
        emoji="🐉",
        min_mm=700,
        max_mm=1500,
        base_value=4000,
        value_per_mm=10,
    ),
    FishSpecies(
        key="whale",
        name="迷航藍鯨",
        rarity="UR",
        emoji="🐳",
        min_mm=700,
        max_mm=1500,
        base_value=4000,
        value_per_mm=10,
    ),
    FishSpecies(
        key="kraken",
        name="傳說海妖",
        rarity="UR",
        emoji="🌊",
        min_mm=700,
        max_mm=1500,
        base_value=4000,
        value_per_mm=10,
    ),
)

ROD_TIERS: Final[tuple[RodTier, ...]] = (
    RodTier(
        key="bamboo",
        name="竹竿",
        emoji="🎍",
        cost=800,
        durability=10,
        miss_bps=3500,
        rarity_weight={"N": 100, "R": 100, "SR": 100, "SSR": 100, "UR": 100},
    ),
    RodTier(
        key="carbon",
        name="碳纖竿",
        emoji="🎣",
        cost=6000,
        durability=25,
        miss_bps=2500,
        rarity_weight={"N": 100, "R": 130, "SR": 200, "SSR": 300, "UR": 400},
    ),
    RodTier(
        key="legend",
        name="傳說之竿",
        emoji="🔱",
        cost=40000,
        durability=60,
        miss_bps=1500,
        rarity_weight={"N": 100, "R": 150, "SR": 300, "SSR": 600, "UR": 1200},
    ),
)

BAIT_TYPES: Final[tuple[BaitType, ...]] = (
    BaitType(
        key="worm",
        name="蚯蚓",
        emoji="🪱",
        cost=20,
        miss_bps_delta=0,
        rarity_weight={"N": 100, "R": 100, "SR": 100, "SSR": 100, "UR": 100},
    ),
    BaitType(
        key="shrimp",
        name="蝦肉",
        emoji="🦐",
        cost=120,
        miss_bps_delta=-500,
        rarity_weight={"N": 100, "R": 120, "SR": 150, "SSR": 200, "UR": 200},
    ),
    BaitType(
        key="lure",
        name="擬餌糖",
        emoji="🍬",
        cost=600,
        miss_bps_delta=-1000,
        rarity_weight={"N": 100, "R": 100, "SR": 200, "SSR": 400, "UR": 800},
    ),
)

SPECIES_BY_KEY: Final[dict[str, FishSpecies]] = {species.key: species for species in FISH_CATALOG}
ROD_BY_KEY: Final[dict[str, RodTier]] = {rod.key: rod for rod in ROD_TIERS}
BAIT_BY_KEY: Final[dict[str, BaitType]] = {bait.key: bait for bait in BAIT_TYPES}
SPECIES_BY_RARITY: Final[dict[Rarity, tuple[FishSpecies, ...]]] = {
    rarity: tuple(species for species in FISH_CATALOG if species.rarity == rarity)
    for rarity in RARITY_ORDER
}


# --- Result / view payloads --------------------------------------------------


class CastOutcome(BaseModel):
    """Pure result of a single cast roll, before any persistence."""

    model_config = ConfigDict(frozen=True)

    miss: bool = Field(description="True when the cast caught nothing (空竿).")
    species: FishSpecies | None = Field(
        default=None, description="Caught species, or None on a miss."
    )
    size_mm: int = Field(default=0, description="Rolled size in millimetres, or 0 on a miss.")
    sell_value: int = Field(
        default=0, description="Sell value in 虛擬歡樂豆 for the caught fish, or 0 on a miss."
    )

    @property
    def rarity(self) -> Rarity | None:
        """Rarity of the caught species, or None on a miss."""
        return self.species.rarity if self.species is not None else None


class CastResult(BaseModel):
    """Database-backed result of an `execute_cast` call."""

    model_config = ConfigDict(frozen=True)

    status: CastStatus = Field(description="Why the cast was rejected, or 'ok' when it ran.")
    outcome: CastOutcome | None = Field(
        default=None, description="The cast roll result, or None when the cast was rejected."
    )
    rod_key: str = Field(default="", description="Rod key after the cast ('' once it broke).")
    rod_durability_after: int = Field(
        default=0, description="Remaining rod durability after this cast."
    )
    rod_broke: bool = Field(default=False, description="True when this cast broke the rod.")
    bait_key: str = Field(default="", description="Bait type consumed by this cast.")
    bait_remaining: int = Field(
        default=0, description="Remaining count of the consumed bait type."
    )
    catch_id: int | None = Field(
        default=None, description="Row id of the inserted catch, or None on a miss/rejection."
    )


class BuyResult(BaseModel):
    """Database-backed result of a rod or bait purchase."""

    model_config = ConfigDict(frozen=True)

    status: BuyStatus = Field(description="Why the purchase failed, or 'ok' when it succeeded.")
    cost: int = Field(default=0, description="Total amount debited from the wallet.")
    new_balance: int = Field(default=0, description="Wallet balance after the purchase.")


class SellResult(BaseModel):
    """Database-backed result of selling caught fish."""

    model_config = ConfigDict(frozen=True)

    status: SellStatus = Field(description="'nothing' when there was nothing to sell, else 'ok'.")
    sold_count: int = Field(default=0, description="Number of fish marked sold.")
    earned: int = Field(default=0, description="Total 虛擬歡樂豆 credited for the sale.")
    new_balance: int = Field(default=0, description="Wallet balance after the sale.")


class LoadoutView(BaseModel):
    """Read-only snapshot of an angler's wallet, equipped rod, and bait."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(description="Discord user id of the angler.")
    balance: int = Field(description="Current spendable wallet balance in 虛擬歡樂豆.")
    rod_key: str = Field(description="Equipped rod key, or '' when no rod is owned.")
    rod_durability: int = Field(description="Remaining durability on the equipped rod.")
    baits: dict[str, int] = Field(
        default_factory=dict, description="Owned bait counts keyed by bait key."
    )
    total_casts: int = Field(default=0, description="Lifetime number of casts by this angler.")


class InventoryEntry(BaseModel):
    """One unsold caught fish in the angler's 魚簍."""

    model_config = ConfigDict(frozen=True)

    catch_id: int = Field(description="Row id of the catch.")
    species_key: str = Field(description="Caught species key.")
    rarity: Rarity = Field(description="Rarity tier of the catch.")
    size_mm: int = Field(description="Caught size in millimetres.")
    sell_value: int = Field(description="Sell value in 虛擬歡樂豆 snapshotted at catch time.")


class DexEntry(BaseModel):
    """One species row in the angler's 圖鑑."""

    model_config = ConfigDict(frozen=True)

    species_key: str = Field(description="Species key for this dex slot.")
    caught: bool = Field(description="True when the angler has ever caught this species.")
    count: int = Field(default=0, description="Total times this species was caught.")
    biggest_mm: int = Field(default=0, description="Largest size of this species ever caught.")


class FishingLeaderboardRow(BaseModel):
    """One ranked row for a scalar fishing leaderboard."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(description="Discord user id of the ranked angler.")
    user_name: str = Field(description="Last-seen Discord name of the angler.")
    value: int = Field(description="Ranking value (e.g. total sale earnings).")


class BiggestCatchRow(BaseModel):
    """One ranked row for the biggest-catch leaderboard."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(description="Discord user id of the ranked angler.")
    user_name: str = Field(description="Last-seen Discord name of the angler.")
    species_key: str = Field(description="Species key of the record catch.")
    rarity: Rarity = Field(description="Rarity tier of the record catch.")
    size_mm: int = Field(description="Size of the record catch in millimetres.")


__all__ = [
    "BAIT_BY_KEY",
    "BAIT_TYPES",
    "BASE_RARITY_BPS",
    "BPS_DENOMINATOR",
    "FISH_CATALOG",
    "MAX_BAIT_PURCHASE_QUANTITY",
    "RARITY_COLOR",
    "RARITY_EMOJI",
    "RARITY_ORDER",
    "ROD_BY_KEY",
    "ROD_TIERS",
    "SPECIES_BY_KEY",
    "SPECIES_BY_RARITY",
    "WEIGHT_UNIT",
    "BaitType",
    "BiggestCatchRow",
    "BuyResult",
    "BuyStatus",
    "CastOutcome",
    "CastResult",
    "CastStatus",
    "DexEntry",
    "FishSpecies",
    "FishingLeaderboardRow",
    "InventoryEntry",
    "LoadoutView",
    "Rarity",
    "RodTier",
    "SellResult",
    "SellStatus",
]
