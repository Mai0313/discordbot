"""List stock operations that need manual reconciliation.

Usage::

    uv run python scripts/manage_stock_reconciliation.py list
"""

import asyncio
from pathlib import Path
import argparse
from collections.abc import Sequence

from rich.console import Console

from discordbot.cogs._stock.database import list_reconciliation_operations
from discordbot.cogs._economy.presentation import amount_code

console = Console()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect non-final stock operations stored in data/database/stock.db."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(name="list", help="List stock operations requiring manual review.")
    return parser.parse_args(args=argv)


async def _async_main(argv: Sequence[str] | None = None) -> None:
    """Runs the CLI."""
    _parse_args(argv=argv)
    operations = await list_reconciliation_operations()
    console.print(f"[bold]Stock reconciliation operations[/bold]: {len(operations)}")
    for operation in operations:
        console.print(
            f"{operation.operation_id} | {operation.status.value} | "
            f"{operation.symbol} | {operation.user_name} ({operation.user_id})"
        )
        if operation.failure_reason:
            console.print(f"  reason: {operation.failure_reason}")
        for leg in operation.legs:
            console.print(
                f"  #{leg.leg_order} {leg.leg_type.value} {leg.shares:,} shares | "
                f"wallet {amount_code(amount=leg.wallet_delta, signed=True)}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the stock reconciliation CLI."""
    # data/ is gitignored and may not exist on a fresh checkout seeded before the
    # bot's first run, so create it here like cli.py does before any DB write.
    Path("./data/database").mkdir(parents=True, exist_ok=True)
    asyncio.run(_async_main(argv=argv))


if __name__ == "__main__":
    main()
