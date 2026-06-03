"""Default fishing catalog: grades, species, and gear.

This is the single source of truth for the seed data. It is consumed only by
`scripts/manage_fishing.py seed-defaults`, `scripts/simulate_fishing.py`, and
tests. Runtime never seeds the database from here; catalog rows are written
offline through the maintenance script.

Values are tuned so the cheapest rod+bait combo has an expected catch value
below its per-cast cost, keeping the game net-deflationary. Re-verify with
`scripts/simulate_fishing.py` after any change.
"""

from discordbot.typings.fishing import (
    GearType,
    GearView,
    FishGrade,
    GearUpsert,
    FishingCatalog,
    FishSpeciesView,
    FishSpeciesUpsert,
    FishGradeConfigView,
    FishGradeConfigUpsert,
)

# Uniform size multiplier range for every species: 0.5x to 2.0x (mean 1.25x).
_SIZE_MIN_BPS = 5_000
_SIZE_MAX_BPS = 20_000

_GRADES: tuple[FishGradeConfigView, ...] = (
    FishGradeConfigView(
        grade=FishGrade.N, weight=6_000, color=0x95A5A6, emoji="⚪", label="普通", order_index=0
    ),
    FishGradeConfigView(
        grade=FishGrade.R, weight=3_000, color=0x3498DB, emoji="🔵", label="稀有", order_index=1
    ),
    FishGradeConfigView(
        grade=FishGrade.SR, weight=800, color=0x9B59B6, emoji="🟣", label="史詩", order_index=2
    ),
    FishGradeConfigView(
        grade=FishGrade.SSR, weight=180, color=0xF1C40F, emoji="🟡", label="傳說", order_index=3
    ),
    FishGradeConfigView(
        grade=FishGrade.UR, weight=20, color=0xE74C3C, emoji="🔴", label="神話", order_index=4
    ),
)


def _species(  # noqa: PLR0913 -- a species row needs id, name, grade, emoji, weight, and value
    species_id: str, name: str, grade: FishGrade, emoji: str, weight: int, base_value: int
) -> FishSpeciesView:
    """Builds one default species row with the shared size range."""
    return FishSpeciesView(
        species_id=species_id,
        name=name,
        grade=grade,
        emoji=emoji,
        intra_grade_weight=weight,
        base_value=base_value,
        size_min_bps=_SIZE_MIN_BPS,
        size_max_bps=_SIZE_MAX_BPS,
    )


_SPECIES: tuple[FishSpeciesView, ...] = (
    _species(
        species_id="minnow", name="小雜魚", grade=FishGrade.N, emoji="🐟", weight=60, base_value=1
    ),
    _species(
        species_id="sardine", name="沙丁魚", grade=FishGrade.N, emoji="🐟", weight=40, base_value=3
    ),
    _species(
        species_id="carp", name="鯉魚", grade=FishGrade.R, emoji="🐠", weight=70, base_value=10
    ),
    _species(
        species_id="bass", name="鱸魚", grade=FishGrade.R, emoji="🐠", weight=30, base_value=20
    ),
    _species(
        species_id="pufferfish",
        name="河豚",
        grade=FishGrade.SR,
        emoji="🐡",
        weight=60,
        base_value=60,
    ),
    _species(
        species_id="octopus",
        name="章魚",
        grade=FishGrade.SR,
        emoji="🐙",
        weight=40,
        base_value=100,
    ),
    _species(
        species_id="swordfish",
        name="旗魚",
        grade=FishGrade.SSR,
        emoji="🗡️",
        weight=70,
        base_value=375,
    ),
    _species(
        species_id="shark", name="鯊魚", grade=FishGrade.SSR, emoji="🦈", weight=30, base_value=750
    ),
    _species(
        species_id="whale",
        name="鯨魚",
        grade=FishGrade.UR,
        emoji="🐋",
        weight=70,
        base_value=2_000,
    ),
    _species(
        species_id="dragon", name="龍", grade=FishGrade.UR, emoji="🐉", weight=30, base_value=5_000
    ),
)


_GEAR: tuple[GearView, ...] = (
    GearView(
        gear_id="rod_bamboo",
        gear_type=GearType.ROD,
        name="竹竿",
        emoji="🎋",
        tier=0,
        price=300,
        rarity_shift_bps=0,
        durability=30,
    ),
    GearView(
        gear_id="rod_carbon",
        gear_type=GearType.ROD,
        name="碳纖維竿",
        emoji="⭐",
        tier=1,
        price=2_000,
        rarity_shift_bps=150,
        durability=80,
    ),
    GearView(
        gear_id="rod_legend",
        gear_type=GearType.ROD,
        name="傳說竿",
        emoji="🌟",
        tier=2,
        price=12_000,
        rarity_shift_bps=400,
        durability=200,
    ),
    GearView(
        gear_id="bait_worm",
        gear_type=GearType.BAIT,
        name="蟲餌",
        emoji="🪱",
        tier=0,
        price=30,
        rarity_shift_bps=0,
        value_bonus_bps=0,
    ),
    GearView(
        gear_id="bait_shrimp",
        gear_type=GearType.BAIT,
        name="蝦餌",
        emoji="🦐",
        tier=1,
        price=80,
        rarity_shift_bps=100,
        value_bonus_bps=500,
    ),
    GearView(
        gear_id="bait_lure",
        gear_type=GearType.BAIT,
        name="路亞",
        emoji="✨",
        tier=2,
        price=200,
        rarity_shift_bps=250,
        value_bonus_bps=1_500,
    ),
)


def build_default_catalog() -> FishingCatalog:
    """Returns the default grades, species, and gear as typed views."""
    return FishingCatalog(grades=_GRADES, species=_SPECIES, gear=_GEAR)


def default_grade_upserts() -> tuple[FishGradeConfigUpsert, ...]:
    """Returns the default grade configs as validated upsert payloads."""
    return tuple(FishGradeConfigUpsert(**grade.model_dump()) for grade in _GRADES)


def default_species_upserts() -> tuple[FishSpeciesUpsert, ...]:
    """Returns the default species as validated upsert payloads."""
    return tuple(FishSpeciesUpsert(**species.model_dump()) for species in _SPECIES)


def default_gear_upserts() -> tuple[GearUpsert, ...]:
    """Returns the default gear as validated upsert payloads."""
    return tuple(GearUpsert(**gear.model_dump()) for gear in _GEAR)


__all__ = [
    "build_default_catalog",
    "default_gear_upserts",
    "default_grade_upserts",
    "default_species_upserts",
]
