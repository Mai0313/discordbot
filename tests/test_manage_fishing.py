"""Tests for the fishing catalog management script."""

import pytest
from scripts import manage_fishing
from pydantic import ValidationError

from discordbot.typings.fishing import GearType, FishGrade, GearUpsert, FishSpeciesUpsert
from discordbot.cogs._fishing.database import list_gear, list_fish_species, list_grade_configs


def test_parse_args_accepts_seed_defaults() -> None:
    """The CLI parses the seed-defaults command."""
    args = manage_fishing._parse_args(argv=["seed-defaults"])
    assert args.command == "seed-defaults"


def test_gear_upsert_rejects_zero_durability_rod() -> None:
    """A rod payload with zero durability is rejected."""
    with pytest.raises(ValidationError):
        GearUpsert(
            gear_id="rod_x",
            gear_type=GearType.ROD,
            name="x",
            emoji="🎣",
            tier=0,
            price=1,
            rarity_shift_bps=0,
            durability=0,
            value_bonus_bps=0,
        )


def test_gear_upsert_rejects_durable_bait() -> None:
    """A bait payload carrying durability is rejected."""
    with pytest.raises(ValidationError):
        GearUpsert(
            gear_id="bait_x",
            gear_type=GearType.BAIT,
            name="x",
            emoji="🪱",
            tier=0,
            price=1,
            rarity_shift_bps=0,
            durability=5,
            value_bonus_bps=0,
        )


def test_species_upsert_rejects_inverted_size_range() -> None:
    """A species payload with min size above max size is rejected."""
    with pytest.raises(ValidationError):
        FishSpeciesUpsert(
            species_id="x",
            name="x",
            grade=FishGrade.N,
            emoji="🐟",
            intra_grade_weight=1,
            base_value=1,
            size_min_bps=20_000,
            size_max_bps=5_000,
        )


@pytest.mark.usefixtures("fishing_isolated_db")
async def test_seed_defaults_is_idempotent() -> None:
    """Seeding the defaults twice yields the same catalog row counts."""
    await manage_fishing._seed_defaults()
    first = (
        len(await list_grade_configs()),
        len(await list_fish_species()),
        len(await list_gear()),
    )
    await manage_fishing._seed_defaults()
    second = (
        len(await list_grade_configs()),
        len(await list_fish_species()),
        len(await list_gear()),
    )
    assert first == second == (5, 10, 6)


@pytest.mark.usefixtures("fishing_isolated_db")
async def test_upsert_grade_via_cli() -> None:
    """The upsert-grade command writes a grade config row."""
    await manage_fishing._async_main(
        argv=[
            "upsert-grade",
            "UR",
            "--weight",
            "10",
            "--color",
            "0xE74C3C",
            "--emoji",
            "🔴",
            "--label",
            "神話",
            "--order-index",
            "4",
        ]
    )
    grades = await list_grade_configs()
    assert any(grade.grade == FishGrade.UR and grade.weight == 10 for grade in grades)


@pytest.mark.usefixtures("fishing_isolated_db")
async def test_list_commands_run() -> None:
    """The list commands run end to end after seeding."""
    await manage_fishing._seed_defaults()
    await manage_fishing._async_main(argv=["list-grades"])
    await manage_fishing._async_main(argv=["list-fish"])
    await manage_fishing._async_main(argv=["list-gear"])
