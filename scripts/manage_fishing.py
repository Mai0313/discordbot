"""Manage fishing catalog rows in data/database/games.db.

The bot never seeds fishing catalog rows at runtime; this script is the only way
to create or tune grades, species, and gear. Run it while the bot is stopped.

Usage::

    uv run python scripts/manage_fishing.py seed-defaults
    uv run python scripts/manage_fishing.py list-fish
    uv run python scripts/manage_fishing.py upsert-gear --help
"""

import asyncio
from pathlib import Path
import argparse
from collections.abc import Sequence

from rich.table import Table
from rich.console import Console

from discordbot.typings.fishing import (
    GearType,
    FishGrade,
    GearUpsert,
    FishSpeciesUpsert,
    FishGradeConfigUpsert,
)
from discordbot.cogs._fishing.database import (
    list_gear,
    upsert_gear,
    list_fish_species,
    list_grade_configs,
    upsert_fish_species,
    upsert_grade_config,
)
from discordbot.cogs._fishing.defaults import (
    default_gear_upserts,
    default_grade_upserts,
    default_species_upserts,
)

console = Console()


def _hex_or_int(value: str) -> int:
    """Parses a color argument given as decimal or 0x-prefixed hex."""
    return int(value, 0)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Create, update, and list fishing catalog rows in data/database/games.db."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(name="seed-defaults", help="Seed or update the default catalog.")
    subparsers.add_parser(name="list-grades", help="List grade config rows.")
    subparsers.add_parser(name="list-fish", help="List fish species rows.")
    subparsers.add_parser(name="list-gear", help="List gear rows.")

    grade = subparsers.add_parser(name="upsert-grade", help="Create or update one grade config.")
    grade.add_argument("grade", choices=[member.value for member in FishGrade])
    grade.add_argument("--weight", required=True, type=int)
    grade.add_argument("--color", required=True, type=_hex_or_int)
    grade.add_argument("--emoji", required=True)
    grade.add_argument("--label", required=True)
    grade.add_argument("--order-index", required=True, type=int)

    species = subparsers.add_parser(name="upsert-fish", help="Create or update one fish species.")
    species.add_argument("species_id")
    species.add_argument("--name", required=True)
    species.add_argument("--grade", required=True, choices=[member.value for member in FishGrade])
    species.add_argument("--emoji", required=True)
    species.add_argument("--weight", required=True, type=int)
    species.add_argument("--base-value", required=True, type=int)
    species.add_argument("--size-min-bps", required=True, type=int)
    species.add_argument("--size-max-bps", required=True, type=int)
    species.add_argument("--image-key", default="")

    gear = subparsers.add_parser(name="upsert-gear", help="Create or update one gear item.")
    gear.add_argument("gear_id")
    gear.add_argument("--gear-type", required=True, choices=[member.value for member in GearType])
    gear.add_argument("--name", required=True)
    gear.add_argument("--emoji", required=True)
    gear.add_argument("--tier", required=True, type=int)
    gear.add_argument("--price", required=True, type=int)
    gear.add_argument("--rarity-shift-bps", default=0, type=int)
    gear.add_argument("--durability", default=0, type=int)
    gear.add_argument("--value-bonus-bps", default=0, type=int)
    return parser.parse_args(args=argv)


async def _seed_defaults() -> None:
    """Idempotently seeds the default catalog."""
    grades = default_grade_upserts()
    species = default_species_upserts()
    gear = default_gear_upserts()
    for config in grades:
        await upsert_grade_config(config=config)
    for fish in species:
        await upsert_fish_species(species=fish)
    for item in gear:
        await upsert_gear(gear=item)
    console.print(f"Seeded {len(grades)} grades, {len(species)} species, {len(gear)} gear rows.")


async def _list_grades() -> None:
    """Prints grade config rows."""
    table = Table(title="Fish grades")
    for column in ("Grade", "Weight", "Order", "Color", "Emoji", "Label"):
        table.add_column(column)
    for config in await list_grade_configs():
        table.add_row(
            config.grade.value,
            f"{config.weight:,}",
            str(config.order_index),
            f"0x{config.color:06X}",
            config.emoji,
            config.label,
        )
    console.print(table)


async def _list_fish() -> None:
    """Prints fish species rows."""
    table = Table(title="Fish species")
    for column in ("ID", "Name", "Grade", "Emoji", "Weight", "Base Value", "Size Min", "Size Max"):
        table.add_column(column)
    for fish in await list_fish_species():
        table.add_row(
            fish.species_id,
            fish.name,
            fish.grade.value,
            fish.emoji,
            str(fish.intra_grade_weight),
            f"{fish.base_value:,}",
            str(fish.size_min_bps),
            str(fish.size_max_bps),
        )
    console.print(table)


async def _list_gear() -> None:
    """Prints gear rows."""
    table = Table(title="Fishing gear")
    for column in (
        "ID",
        "Type",
        "Name",
        "Emoji",
        "Tier",
        "Price",
        "Rarity+",
        "Durability",
        "Value+",
    ):
        table.add_column(column)
    for item in await list_gear():
        table.add_row(
            item.gear_id,
            item.gear_type.value,
            item.name,
            item.emoji,
            str(item.tier),
            f"{item.price:,}",
            str(item.rarity_shift_bps),
            str(item.durability),
            str(item.value_bonus_bps),
        )
    console.print(table)


async def _async_main(argv: Sequence[str] | None = None) -> None:
    """Runs the CLI."""
    args = _parse_args(argv=argv)
    if args.command == "seed-defaults":
        await _seed_defaults()
        return
    if args.command == "list-grades":
        await _list_grades()
        return
    if args.command == "list-fish":
        await _list_fish()
        return
    if args.command == "list-gear":
        await _list_gear()
        return
    if args.command == "upsert-grade":
        config = await upsert_grade_config(
            config=FishGradeConfigUpsert(
                grade=FishGrade(args.grade),
                weight=args.weight,
                color=args.color,
                emoji=args.emoji,
                label=args.label,
                order_index=args.order_index,
            )
        )
        console.print(f"Upserted grade: [bold]{config.grade.value}[/bold] {config.label}")
        return
    if args.command == "upsert-fish":
        fish = await upsert_fish_species(
            species=FishSpeciesUpsert(
                species_id=args.species_id,
                name=args.name,
                grade=FishGrade(args.grade),
                emoji=args.emoji,
                intra_grade_weight=args.weight,
                base_value=args.base_value,
                size_min_bps=args.size_min_bps,
                size_max_bps=args.size_max_bps,
                image_key=args.image_key,
            )
        )
        console.print(f"Upserted species: [bold]{fish.species_id}[/bold] {fish.name}")
        return
    item = await upsert_gear(
        gear=GearUpsert(
            gear_id=args.gear_id,
            gear_type=GearType(args.gear_type),
            name=args.name,
            emoji=args.emoji,
            tier=args.tier,
            price=args.price,
            rarity_shift_bps=args.rarity_shift_bps,
            durability=args.durability,
            value_bonus_bps=args.value_bonus_bps,
        )
    )
    console.print(f"Upserted gear: [bold]{item.gear_id}[/bold] {item.name}")


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the fishing catalog maintenance CLI."""
    # data/ is gitignored and may not exist on a fresh checkout seeded before the
    # bot's first run, so create it here like cli.py does before any DB write.
    Path("./data/database").mkdir(parents=True, exist_ok=True)
    asyncio.run(main=_async_main(argv=argv))


if __name__ == "__main__":
    main()
