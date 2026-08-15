"""One-shot admin helper for writing the fishing catalog into `games.db`.

Grades, species, and gear are the tunable source of truth and are seeded
offline; runtime never writes them. `defaults.py` is the one catalog definition,
so retuning it has no effect on a deployed bot until this script applies it.

Usage::

    uv run python scripts/seed_fishing.py --dry-run
    uv run python scripts/seed_fishing.py

Per-user state (rods, bait, catch history) is never touched.
"""

from typing import Literal
import asyncio
from pathlib import Path
import argparse
from collections.abc import Sequence

from pydantic import Field, BaseModel, ConfigDict
from rich.console import Console

from discordbot.cogs.games.fishing.database import (
    list_gear,
    upsert_gear,
    list_fish_species,
    list_grade_configs,
    upsert_fish_species,
    upsert_grade_config,
)
from discordbot.cogs.games.fishing.defaults import (
    default_gear_upserts,
    default_grade_upserts,
    default_species_upserts,
)

console = Console()

CatalogRowKind = Literal["grade", "species", "gear"]


class CatalogRowChange(BaseModel):
    """What one seeded catalog row would change, or did change."""

    model_config = ConfigDict(frozen=True)

    kind: CatalogRowKind = Field(..., description="Which catalog table the row belongs to.")
    row_id: str = Field(..., description="Primary key of the row, for example rod_legend.")
    created: bool = Field(..., description="Whether the row did not exist before this run.")
    changes: tuple[str, ...] = Field(
        ..., description="One `field old -> new` line per differing field on an existing row."
    )


class CatalogSeedSummary(BaseModel):
    """Summary of one seeding run over the whole catalog."""

    model_config = ConfigDict(frozen=True)

    changes: tuple[CatalogRowChange, ...] = Field(
        ..., description="Rows that were created or updated, in catalog order."
    )
    unchanged: int = Field(..., description="Rows already matching the default catalog.")
    dry_run: bool = Field(..., description="Whether the run only reported without writing.")


def _diff(payload: BaseModel, existing: BaseModel | None) -> tuple[str, ...]:
    """Returns one line per field where the stored row differs from the default.

    Compared field by field off the payload rather than by dumping both models,
    so a view carrying extra columns (timestamps, say) never reads as a change.
    """
    if existing is None:
        return ()
    return tuple(
        f"{name} {getattr(existing, name)} -> {value}"
        for name, value in payload.model_dump().items()
        if getattr(existing, name) != value
    )


async def seed_fishing_catalog(dry_run: bool = False) -> CatalogSeedSummary:
    """Writes the default grades, species, and gear into the fishing database.

    Every row is upserted through the same maintenance API a manual edit would
    use, so a partial run is safe to repeat. Rows an operator added beyond the
    defaults are left alone; this seeds, it does not reconcile.

    Args:
        dry_run (bool): Whether to report the pending changes without writing them.

    Returns:
        CatalogSeedSummary: What each row would change, or did change.
    """
    grades = {row.grade: row for row in await list_grade_configs()}
    species = {row.species_id: row for row in await list_fish_species()}
    gear = {row.gear_id: row for row in await list_gear()}

    changes: list[CatalogRowChange] = []
    unchanged = 0
    for grade_payload in default_grade_upserts():
        existing = grades.get(grade_payload.grade)
        diff = _diff(payload=grade_payload, existing=existing)
        if existing is not None and not diff:
            unchanged += 1
            continue
        changes.append(
            CatalogRowChange(
                kind="grade",
                row_id=grade_payload.grade.value,
                created=existing is None,
                changes=diff,
            )
        )
        if not dry_run:
            await upsert_grade_config(config=grade_payload)

    for species_payload in default_species_upserts():
        existing = species.get(species_payload.species_id)
        diff = _diff(payload=species_payload, existing=existing)
        if existing is not None and not diff:
            unchanged += 1
            continue
        changes.append(
            CatalogRowChange(
                kind="species",
                row_id=species_payload.species_id,
                created=existing is None,
                changes=diff,
            )
        )
        if not dry_run:
            await upsert_fish_species(species=species_payload)

    for gear_payload in default_gear_upserts():
        existing = gear.get(gear_payload.gear_id)
        diff = _diff(payload=gear_payload, existing=existing)
        if existing is not None and not diff:
            unchanged += 1
            continue
        changes.append(
            CatalogRowChange(
                kind="gear", row_id=gear_payload.gear_id, created=existing is None, changes=diff
            )
        )
        if not dry_run:
            await upsert_gear(gear=gear_payload)

    return CatalogSeedSummary(changes=tuple(changes), unchanged=unchanged, dry_run=dry_run)


def _print_summary(summary: CatalogSeedSummary) -> None:
    """Prints a human-readable seeding summary."""
    title = "Dry run" if summary.dry_run else "Fishing catalog seeded"
    console.print(f"[bold]{title}[/bold]")
    console.print(f"changed: {len(summary.changes)}")
    console.print(f"unchanged: {summary.unchanged}")
    for change in summary.changes:
        verb = "create" if change.created else "update"
        console.print(f"{verb} {change.kind} {change.row_id}")
        for line in change.changes:
            console.print(f"  {line}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Write the default fishing grades, species, and gear into games.db."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the pending catalog changes without writing to the database.",
    )
    return parser.parse_args(args=argv)


async def _async_main(argv: Sequence[str] | None = None) -> None:
    """Runs the CLI."""
    args = _parse_args(argv=argv)
    summary = await seed_fishing_catalog(dry_run=args.dry_run)
    _print_summary(summary=summary)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the fishing catalog seeding CLI.

    Args:
        argv (Sequence[str] | None): Optional argument sequence to parse instead of `sys.argv`.
    """
    # data/ is gitignored and may not exist on a fresh checkout seeded before the
    # bot's first run, so create it here like cli.py does before any DB write.
    Path("./data/database").mkdir(parents=True, exist_ok=True)
    asyncio.run(main=_async_main(argv=argv))


if __name__ == "__main__":
    main()
