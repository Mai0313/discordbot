"""Manage fishing state in data/fishing.db (offline maintenance).

Usage::

    uv run python scripts/manage_fishing.py catalog
    uv run python scripts/manage_fishing.py inspect 123456789
    uv run python scripts/manage_fishing.py grant-rod 123456789 carbon
    uv run python scripts/manage_fishing.py grant-bait 123456789 worm 20
    uv run python scripts/manage_fishing.py reset 123456789
"""

import asyncio
import argparse
from collections.abc import Sequence

from rich.table import Table
from rich.console import Console

from discordbot.typings.fishing import (
    ROD_TIERS,
    BAIT_TYPES,
    ROD_BY_KEY,
    FISH_CATALOG,
    SPECIES_BY_KEY,
)
from discordbot.cogs._games.fishing import per_cast_ev, loadout_cost
from discordbot.cogs._games.fishing_database import (
    get_dex,
    grant_rod,
    grant_bait,
    reset_user,
    get_loadout,
    list_inventory,
)

console = Console()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description="Inspect and maintain data/fishing.db rows.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(name="catalog", help="List fish, rods, baits, and per-loadout EV.")

    inspect = subparsers.add_parser(name="inspect", help="Show one user's fishing state.")
    inspect.add_argument("user_id", type=int)

    grant_rod_parser = subparsers.add_parser(
        name="grant-rod", help="Grant a rod without charging."
    )
    grant_rod_parser.add_argument("user_id", type=int)
    grant_rod_parser.add_argument("rod_key")
    grant_rod_parser.add_argument("--name", default="manual")

    grant_bait_parser = subparsers.add_parser(
        name="grant-bait", help="Grant bait without charging."
    )
    grant_bait_parser.add_argument("user_id", type=int)
    grant_bait_parser.add_argument("bait_key")
    grant_bait_parser.add_argument("quantity", type=int)
    grant_bait_parser.add_argument("--name", default="manual")

    reset = subparsers.add_parser(name="reset", help="Delete all fishing state for a user.")
    reset.add_argument("user_id", type=int)
    return parser.parse_args(args=argv)


async def _show_catalog() -> None:
    """Prints the fish, rod, and bait catalogs plus per-loadout EV."""
    fish_table = Table(title="Fish catalog")
    fish_table.add_column("rarity")
    fish_table.add_column("key")
    fish_table.add_column("name")
    fish_table.add_column("size_mm", justify="right")
    fish_table.add_column("base_value", justify="right")
    for species in FISH_CATALOG:
        fish_table.add_row(
            species.rarity,
            species.key,
            f"{species.emoji} {species.name}",
            f"{species.min_mm}-{species.max_mm}",
            f"{species.base_value:,}",
        )
    console.print(fish_table)

    ev_table = Table(title="Per-cast expected value (must be negative)")
    ev_table.add_column("rod")
    ev_table.add_column("bait")
    ev_table.add_column("cost", justify="right")
    ev_table.add_column("ev", justify="right")
    for rod in ROD_TIERS:
        for bait in BAIT_TYPES:
            ev = per_cast_ev(rod=rod, bait=bait)
            ev_table.add_row(
                rod.name,
                bait.name,
                f"{loadout_cost(rod=rod, bait=bait):.1f}",
                f"[red]{ev:+.1f}[/red]" if ev >= 0 else f"{ev:+.1f}",
            )
    console.print(ev_table)


async def _inspect(user_id: int) -> None:
    """Prints one user's loadout, dex completion, and recent catches."""
    loadout = await get_loadout(user_id=user_id)
    rod = ROD_BY_KEY.get(loadout.rod_key)
    rod_text = (
        f"{rod.name} {loadout.rod_durability}/{rod.durability}" if rod is not None else "(none)"
    )
    bait_text = ", ".join(f"{key}x{qty}" for key, qty in loadout.baits.items()) or "(none)"
    console.print(
        f"user_id={user_id} balance={loadout.balance:,} rod={rod_text} "
        f"bait={bait_text} total_casts={loadout.total_casts}"
    )

    dex = await get_dex(user_id=user_id)
    caught = sum(1 for entry in dex if entry.caught)
    console.print(f"dex completion: {caught}/{len(dex)}")

    inventory = await list_inventory(user_id=user_id)
    table = Table(title="Unsold catches (latest)")
    table.add_column("catch_id", justify="right")
    table.add_column("species")
    table.add_column("rarity")
    table.add_column("size_mm", justify="right")
    table.add_column("sell_value", justify="right")
    for entry in inventory:
        species = SPECIES_BY_KEY.get(entry.species_key)
        table.add_row(
            str(entry.catch_id),
            species.name if species is not None else entry.species_key,
            entry.rarity,
            str(entry.size_mm),
            f"{entry.sell_value:,}",
        )
    console.print(table)


async def _run(args: argparse.Namespace) -> None:
    """Dispatches the parsed subcommand."""
    if args.command == "catalog":
        await _show_catalog()
    elif args.command == "inspect":
        await _inspect(user_id=args.user_id)
    elif args.command == "grant-rod":
        ok = await grant_rod(user_id=args.user_id, user_name=args.name, rod_key=args.rod_key)
        console.print("granted" if ok else f"unknown rod: {args.rod_key}")
    elif args.command == "grant-bait":
        ok = await grant_bait(
            user_id=args.user_id,
            user_name=args.name,
            bait_key=args.bait_key,
            quantity=args.quantity,
        )
        console.print(
            "granted" if ok else f"invalid bait/quantity: {args.bait_key} {args.quantity}"
        )
    elif args.command == "reset":
        await reset_user(user_id=args.user_id)
        console.print(f"reset user {args.user_id}")


def main() -> None:
    """CLI entry point."""
    asyncio.run(main=_run(args=_parse_args()))


if __name__ == "__main__":
    main()
