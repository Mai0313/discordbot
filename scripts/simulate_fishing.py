"""Monte Carlo measurement of the fishing game's per-cast 虛擬歡樂豆 sink.

Every `(rod, bait)` loadout must be a net sink: the amortised cast cost
(bait price plus rod price spread over its durability) must exceed the mean
sell value of what it catches. This script draws many seeded casts per loadout
and reports the empirical mean net result next to the analytic `per_cast_ev`,
so a balance change that accidentally turns a loadout 虛擬歡樂豆-positive is
caught before it ships. Run:

    uv run python scripts/simulate_fishing.py --casts 1000000 --seed 12345

A non-zero exit code means at least one loadout is not a sink.
"""

from random import Random
import argparse

from rich.table import Table
from rich.console import Console

from discordbot.typings.fishing import ROD_TIERS, BAIT_TYPES, RARITY_ORDER, RodTier, BaitType
from discordbot.cogs._games.fishing import cast_fish, per_cast_ev, loadout_cost

console = Console()


def _simulate_loadout(
    *, rng: Random, rod: RodTier, bait: BaitType, casts: int
) -> dict[str, float]:
    """Runs `casts` seeded casts for one loadout and returns aggregate tallies."""
    total_value = 0
    misses = 0
    rarity_counts = dict.fromkeys(RARITY_ORDER, 0)
    for _ in range(casts):
        outcome = cast_fish(rng=rng, rod=rod, bait=bait)
        if outcome.miss or outcome.species is None:
            misses += 1
            continue
        total_value += outcome.sell_value
        rarity_counts[outcome.species.rarity] += 1
    cost = loadout_cost(rod=rod, bait=bait)
    mean_net = total_value / casts - cost
    return {
        "mean_sell": total_value / casts,
        "cost": cost,
        "mean_net": mean_net,
        "analytic_ev": per_cast_ev(rod=rod, bait=bait),
        "miss_rate": misses / casts,
        **{f"rate_{rarity}": rarity_counts[rarity] / casts for rarity in RARITY_ORDER},
    }


def _render_report(*, rows: list[tuple[str, dict[str, float]]], casts: int, seed: int) -> Table:
    """Builds the rich summary table across every loadout."""
    table = Table(title=f"Fishing per-cast sink over {casts:,} casts/loadout (seed {seed})")
    table.add_column("loadout")
    table.add_column("miss%", justify="right")
    for rarity in RARITY_ORDER:
        table.add_column(f"{rarity}%", justify="right")
    table.add_column("mean_sell", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("mean_net", justify="right")
    table.add_column("analytic_ev", justify="right")
    table.add_column("sink", justify="center")
    for label, stats in rows:
        table.add_row(
            label,
            f"{stats['miss_rate'] * 100:.1f}",
            *(f"{stats[f'rate_{rarity}'] * 100:.2f}" for rarity in RARITY_ORDER),
            f"{stats['mean_sell']:.1f}",
            f"{stats['cost']:.1f}",
            f"{stats['mean_net']:+.1f}",
            f"{stats['analytic_ev']:+.1f}",
            "✅" if stats["mean_net"] < 0 else "❌",
        )
    return table


def main() -> None:
    """Runs the simulation across every loadout and exits non-zero if any is not a sink."""
    parser = argparse.ArgumentParser(description="Measure the fishing per-cast 虛擬歡樂豆 sink.")
    parser.add_argument("--casts", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=12_345)
    args = parser.parse_args()
    rng = Random(args.seed)  # noqa: S311 -- reproducible measurement, not security-sensitive.
    rows: list[tuple[str, dict[str, float]]] = []
    for rod in ROD_TIERS:
        for bait in BAIT_TYPES:
            stats = _simulate_loadout(rng=rng, rod=rod, bait=bait, casts=args.casts)
            rows.append((f"{rod.name} + {bait.name}", stats))
    console.print(_render_report(rows=rows, casts=args.casts, seed=args.seed))
    not_sink = [label for label, stats in rows if stats["mean_net"] >= 0]
    if not_sink:
        console.print(f"[red]NOT A SINK:[/red] {', '.join(not_sink)}")
        raise SystemExit(1)
    console.print("[green]All loadouts are net 虛擬歡樂豆 sinks.[/green]")


if __name__ == "__main__":
    main()
