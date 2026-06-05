"""Manage simulated stock company rows in data/database/stock.db.

Usage::

    uv run python scripts/manage_stock_company.py list
    uv run python scripts/manage_stock_company.py upsert --help
"""

import asyncio
from pathlib import Path
import argparse
from collections.abc import Sequence

from rich.table import Table
from rich.console import Console

from discordbot.typings.stock import StockProfileUpsert
from discordbot.cogs._stock.database import (
    list_stock_profiles,
    upsert_stock_profile,
    list_stock_supply_audit,
)

console = Console()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Create, update, and list stock_profile rows in data/database/stock.db."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(name="list", help="List stock company profile rows.")
    subparsers.add_parser(name="audit", help="List stock supply and aggregate exposure.")

    upsert = subparsers.add_parser(
        name="upsert", help="Create or update one stock company profile row."
    )
    upsert.add_argument("symbol")
    upsert.add_argument("--name", required=True)
    upsert.add_argument("--category", required=True)
    upsert.add_argument("--price-cents", required=True, type=int)
    upsert.add_argument("--total-shares", required=True, type=int)
    upsert.add_argument("--float-shares", required=True, type=int)
    upsert.add_argument("--base-volatility-bps", required=True, type=int)
    upsert.add_argument("--volatility-amplifier-bps", required=True, type=int)
    upsert.add_argument("--liquidity-shares", required=True, type=int)
    upsert.add_argument("--fair-value-cents", required=True, type=int)
    upsert.add_argument("--mean-reversion-bps", required=True, type=int)
    upsert.add_argument("--max-tick-change-bps", required=True, type=int)
    upsert.add_argument("--news-cadence-hours", required=True, type=int)
    return parser.parse_args(args=argv)


def _profile_from_args(args: argparse.Namespace) -> StockProfileUpsert:
    """Builds a typed stock profile payload from CLI arguments."""
    return StockProfileUpsert(
        symbol=args.symbol,
        name=args.name,
        category=args.category,
        price_cents=args.price_cents,
        total_shares=args.total_shares,
        float_shares=args.float_shares,
        base_volatility_bps=args.base_volatility_bps,
        volatility_amplifier_bps=args.volatility_amplifier_bps,
        liquidity_shares=args.liquidity_shares,
        fair_value_cents=args.fair_value_cents,
        mean_reversion_bps=args.mean_reversion_bps,
        max_tick_change_bps=args.max_tick_change_bps,
        news_cadence_hours=args.news_cadence_hours,
    )


async def _list_profiles() -> None:
    """Prints stock profiles."""
    table = Table(title="Stock companies")
    for column in ("Symbol", "Name", "Category", "Price", "Fair Value", "Liquidity", "News Hours"):
        table.add_column(column)
    for profile in await list_stock_profiles():
        table.add_row(
            profile.symbol,
            profile.name,
            profile.category,
            str(profile.price_cents),
            str(profile.fair_value_cents),
            f"{profile.liquidity_shares:,}",
            str(profile.news_cadence_hours),
        )
    console.print(table)


async def _audit_supply() -> None:
    """Prints stock supply and aggregate exposure."""
    table = Table(title="Stock supply audit")
    for column in (
        "Symbol",
        "Name",
        "Float",
        "Long",
        "Short",
        "Long Available",
        "Short Available",
        "Liquidity",
        "Non-final Ops",
    ):
        table.add_column(column)
    for audit in await list_stock_supply_audit():
        table.add_row(
            audit.symbol,
            audit.name,
            f"{audit.float_shares:,}",
            f"{audit.long_shares:,}",
            f"{audit.short_shares:,}",
            f"{audit.available_long_shares:,}",
            f"{audit.available_short_shares:,}",
            f"{audit.liquidity_shares:,}",
            str(audit.non_final_operations),
        )
    console.print(table)


async def _async_main(argv: Sequence[str] | None = None) -> None:
    """Runs the CLI."""
    args = _parse_args(argv=argv)
    if args.command == "list":
        await _list_profiles()
        return
    if args.command == "audit":
        await _audit_supply()
        return
    profile = await upsert_stock_profile(profile=_profile_from_args(args=args))
    console.print(f"Upserted stock company: [bold]{profile.symbol}[/bold] {profile.name}")


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the stock company maintenance CLI."""
    # data/ is gitignored and may not exist on a fresh checkout seeded before the
    # bot's first run, so create it here like cli.py does before any DB write.
    Path("./data/database").mkdir(parents=True, exist_ok=True)
    asyncio.run(_async_main(argv=argv))


if __name__ == "__main__":
    main()
