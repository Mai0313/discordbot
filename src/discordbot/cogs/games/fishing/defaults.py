"""Default fishing catalog: the grade ladder, the species table, and the rod and bait rows.

This is the single source of truth for the seed data, consumed only by `scripts/seed_fishing.py`
and tests. Runtime never seeds the database from here; catalog rows are written offline, so
retuning a value below changes nothing in a deployed bot until that script applies it. The same
rows are published in two shapes: `build_default_catalog` bundles them as the bound-free `*View`
read models, for a caller that wants a catalog without a database, and the three
`default_*_upserts` functions re-validate them through the `*Upsert` payloads the seed script
writes, which are where the field bounds and the cross-field rules live.

Fishing is deliberately no longer a currency sink (#351). Values are tuned so every rod+bait
combo returns at least its per-cast cost and no more than 1.7x it, so that upgrading either the
rod or the bait raises both the rare-catch rate and the return, and so that on any given bait
each grade's payout band sits entirely above the grade below it. `tests/test_fishing_catch.py`
computes those figures exactly rather than by sampling and fails on any retuning that breaks
them, so it also covers gear added here.

Adding a rod needs the most care: the rod axis at the cheapest bait clears the
never-pays-backwards rule by under half a percentage point (107.18 / 107.64 / 109.56), and a
mid-tier rod interpolated between the two below breaks it. A total rarity shift at or above
20000 bps also buys nothing further, since `LUCK_STEP_MAX_BPS` saturates there.
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

# Size multiplier range per grade, so the number on the reveal reads as a reward
# rather than a punishment: a 傳說 never surfaces below 2.00x and a 史詩 never
# below 1.50x. The bands also keep grade payouts from overlapping, together with
# the base values below: the worst catch of a grade outpays the best of the one
# under it.
_SIZE_BPS_BY_GRADE: dict[FishGrade, tuple[int, int]] = {
    FishGrade.N: (5_000, 12_000),
    FishGrade.R: (10_000, 16_000),
    FishGrade.SR: (15_000, 22_000),
    FishGrade.SSR: (20_000, 30_000),
    FishGrade.UR: (30_000, 50_000),
}

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
    """Builds one default species row, taking its size band from `_SIZE_BPS_BY_GRADE`.

    The band belongs to the grade rather than to the fish, so every species in a grade shares
    one and a row added here cannot be given a multiplier range that contradicts what its grade
    promises on the reveal.

    Args:
        species_id (str): Stable identifier for the species row.
        name (str): Display name of the species.
        grade (FishGrade): Rarity grade whose size band the row inherits.
        emoji (str): Emoji shown for the species.
        weight (int): Roll weight against the other species of the same grade.
        base_value (int): Base sell value, before the size multiplier and the bait bonus.

    Returns:
        The species row, with its grade's size band filled in.
    """
    size_min_bps, size_max_bps = _SIZE_BPS_BY_GRADE[grade]
    return FishSpeciesView(
        species_id=species_id,
        name=name,
        grade=grade,
        emoji=emoji,
        intra_grade_weight=weight,
        base_value=base_value,
        size_min_bps=size_min_bps,
        size_max_bps=size_max_bps,
    )


_SPECIES: tuple[FishSpeciesView, ...] = (
    _species(
        species_id="minnow", name="小雜魚", grade=FishGrade.N, emoji="🐟", weight=60, base_value=12
    ),
    _species(
        species_id="sardine",
        name="沙丁魚",
        grade=FishGrade.N,
        emoji="🐟",
        weight=40,
        base_value=20,
    ),
    _species(
        species_id="carp", name="鯉魚", grade=FishGrade.R, emoji="🐠", weight=70, base_value=26
    ),
    _species(
        species_id="bass", name="鱸魚", grade=FishGrade.R, emoji="🐠", weight=30, base_value=44
    ),
    _species(
        species_id="pufferfish",
        name="河豚",
        grade=FishGrade.SR,
        emoji="🐡",
        weight=60,
        base_value=48,
    ),
    _species(
        species_id="octopus", name="章魚", grade=FishGrade.SR, emoji="🐙", weight=40, base_value=86
    ),
    _species(
        species_id="swordfish",
        name="旗魚",
        grade=FishGrade.SSR,
        emoji="🗡️",
        weight=70,
        base_value=160,
    ),
    _species(
        species_id="shark", name="鯊魚", grade=FishGrade.SSR, emoji="🦈", weight=30, base_value=280
    ),
    _species(
        species_id="whale", name="鯨魚", grade=FishGrade.UR, emoji="🐋", weight=70, base_value=520
    ),
    _species(
        species_id="dragon", name="龍", grade=FishGrade.UR, emoji="🐉", weight=30, base_value=950
    ),
)


# Prices carry the fix for the return ratio running backwards. Per-cast cost is
# additive (rod price over durability, plus bait price) while the value the gear
# buys is multiplicative, so a rod or bait priced by feel rather than by what it
# returns is what made the top combo pay back a fifth of what the starter did.
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
        price=1_440,
        rarity_shift_bps=1_500,
        durability=80,
    ),
    GearView(
        gear_id="rod_legend",
        gear_type=GearType.ROD,
        name="傳說竿",
        emoji="🌟",
        tier=2,
        price=6_800,
        rarity_shift_bps=4_000,
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
        price=55,
        rarity_shift_bps=1_000,
        value_bonus_bps=5_000,
    ),
    GearView(
        gear_id="bait_lure",
        gear_type=GearType.BAIT,
        name="路亞",
        emoji="✨",
        tier=2,
        price=110,
        rarity_shift_bps=2_500,
        value_bonus_bps=15_000,
    ),
)


def build_default_catalog() -> FishingCatalog:
    """Bundles the default grades, species, and gear as one frozen read model.

    The `*View` shape carries no field bounds, so this is what a caller that wants the catalog
    without a database gets; `tests/test_fishing_catch.py` reads the whole tuning off it. The
    seed script takes the same rows through the upsert payloads below instead.

    Returns:
        The whole default catalog as one value.
    """
    return FishingCatalog(grades=_GRADES, species=_SPECIES, gear=_GEAR)


def default_grade_upserts() -> tuple[FishGradeConfigUpsert, ...]:
    """Re-validates the default grade configs as the write payloads the seed script takes.

    Round-tripping each view through `FishGradeConfigUpsert` is what applies the field bounds
    the read model deliberately does not carry, so a retuning that leaves a negative weight or a
    blank label above fails here rather than reaching the table.

    Returns:
        One upsert payload per default grade, in catalog order.
    """
    return tuple(FishGradeConfigUpsert(**grade.model_dump()) for grade in _GRADES)


def default_species_upserts() -> tuple[FishSpeciesUpsert, ...]:
    """Re-validates the default species as the write payloads the seed script takes.

    `FishSpeciesUpsert` is where the field bounds and the well-ordered size-band check live, so
    an inverted `_SIZE_BPS_BY_GRADE` entry is caught here instead of raising out of
    `random.randint` mid-cast once the row is in the table.

    Returns:
        One upsert payload per default species, in catalog order.
    """
    return tuple(FishSpeciesUpsert(**species.model_dump()) for species in _SPECIES)


def default_gear_upserts() -> tuple[GearUpsert, ...]:
    """Re-validates the default gear as the write payloads the seed script takes.

    `GearUpsert` is where the rod-versus-bait rule lives, so a rod added above without
    durability, which would sell a rod that answers `BROKEN_ROD` on its first cast, is rejected
    here instead of being written to the table.

    Returns:
        One upsert payload per default rod and bait, in catalog order.
    """
    return tuple(GearUpsert(**gear.model_dump()) for gear in _GEAR)


__all__ = [
    "build_default_catalog",
    "default_gear_upserts",
    "default_grade_upserts",
    "default_species_upserts",
]
