"""Monte Carlo the fishing roll engine to confirm it stays net-deflationary.

Runs ``roll_catch`` over the default catalog for every rod and bait combination
and reports the expected catch value per cast versus the per-cast cost (bait
price plus the amortized rod price), the net bean flow per cast (which must be
negative), grade hit rates, and how often the per-catch cap binds. Re-run after
any catalog change, the same way the stock simulator re-measures volatility.

Usage::

    uv run python scripts/simulate_fishing.py
    uv run python scripts/simulate_fishing.py --casts 2000000 --seed 1
"""

from random import Random
import argparse
from collections.abc import Sequence

from rich.table import Table
from rich.console import Console

from discordbot.typings.fishing import (
    FISHING_MAX_SINGLE_CATCH,
    GearType,
    GearView,
    FishGrade,
    FishingCatalog,
)
from discordbot.cogs._fishing.catch import roll_catch
from discordbot.cogs._fishing.defaults import build_default_catalog

console = Console()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description="Monte Carlo the fishing roll engine.")
    parser.add_argument("--casts", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(args=argv)


def _cost_per_cast(rod: GearView, bait: GearView) -> float:
    """Returns the amortized per-cast cost of a rod and bait combo."""
    rod_amortized = rod.price / rod.durability if rod.durability > 0 else float(rod.price)
    return bait.price + rod_amortized


def _simulate_combo(
    catalog: FishingCatalog, rod: GearView, bait: GearView, casts: int, seed: int
) -> tuple[float, dict[FishGrade, int], int]:
    """Returns (ev_per_cast, grade_counts, capped_count) for one combo."""
    rng = Random(seed)  # noqa: S311 -- deterministic simulation, not cryptography
    total_value = 0
    grade_counts: dict[FishGrade, int] = {grade.grade: 0 for grade in catalog.grades}
    capped = 0
    for _ in range(casts):
        roll = roll_catch(
            rng=rng,
            grade_configs=catalog.grades,
            species=catalog.species,
            rod=rod,
            bait=bait,
            max_value=FISHING_MAX_SINGLE_CATCH,
        )
        total_value += roll.value
        grade_counts[roll.grade] += 1
        capped += int(roll.capped)
    return total_value / casts, grade_counts, capped


def _report(args: argparse.Namespace) -> None:
    """Runs all combos and prints the net-deflation and grade-distribution tables."""
    catalog = build_default_catalog()
    rods = [item for item in catalog.gear if item.gear_type == GearType.ROD]
    baits = [item for item in catalog.gear if item.gear_type == GearType.BAIT]

    combo_table = Table(title=f"Fishing net flow per cast ({args.casts:,} casts/combo)")
    for column in ("Rod", "Bait", "EV/cast", "Cost/cast", "Net/cast", "Cap %", "Verdict"):
        combo_table.add_column(column, justify="right")

    all_sink = True
    cheapest = (rods[0], baits[0])
    cheapest_counts: dict[FishGrade, int] = {}
    for rod_index, rod in enumerate(rods):
        for bait_index, bait in enumerate(baits):
            seed = args.seed + rod_index * len(baits) + bait_index
            ev, grade_counts, capped = _simulate_combo(
                catalog=catalog, rod=rod, bait=bait, casts=args.casts, seed=seed
            )
            cost = _cost_per_cast(rod=rod, bait=bait)
            net = ev - cost
            is_sink = net < 0
            all_sink = all_sink and is_sink
            if rod is cheapest[0] and bait is cheapest[1]:
                cheapest_counts = grade_counts
            verdict = "[green]sink[/green]" if is_sink else "[red]FAUCET[/red]"
            combo_table.add_row(
                rod.name,
                bait.name,
                f"{ev:.2f}",
                f"{cost:.2f}",
                f"{net:+.2f}",
                f"{capped / args.casts * 100:.3f}",
                verdict,
            )
    console.print(combo_table)

    grade_table = Table(title="Grade hit rate (cheapest combo, shift=0)")
    for column in ("Grade", "Theoretical %", "Observed %"):
        grade_table.add_column(column, justify="right")
    total_weight = sum(grade.weight for grade in catalog.grades)
    for grade in sorted(catalog.grades, key=lambda config: config.order_index):
        theoretical = grade.weight / total_weight * 100 if total_weight else 0.0
        observed = cheapest_counts.get(grade.grade, 0) / args.casts * 100
        grade_table.add_row(grade.grade.value, f"{theoretical:.3f}", f"{observed:.3f}")
    console.print(grade_table)

    if all_sink:
        console.print("[green]All combos are net-deflationary.[/green]")
    else:
        console.print("[red]WARNING: at least one combo is a faucet; retune the catalog.[/red]")


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the fishing simulation CLI."""
    _report(args=_parse_args(argv=argv))


if __name__ == "__main__":
    main()
