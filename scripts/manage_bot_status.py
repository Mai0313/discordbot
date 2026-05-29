"""Manage the bot presence rotation stored in data/global_state.db.

The status task picks a random enabled line every minute; an empty table falls
back to the built-in default. Run while the bot is stopped.

Usage::

    uv run python scripts/manage_bot_status.py list
    uv run python scripts/manage_bot_status.py add "playing with fire" --order 1
    uv run python scripts/manage_bot_status.py remove 3
"""

import asyncio
import argparse
from collections.abc import Sequence

from rich.console import Console

from discordbot.typings.economy import BotStatusEntry
from discordbot.cogs._economy.database import (
    add_bot_status,
    remove_bot_status,
    list_bot_status_rows,
)

console = Console()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Add, remove, or list bot presence lines stored in data/global_state.db."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser(name="add", help="Add a presence line.")
    add_parser.add_argument("status_text", help="The presence text to display.")
    add_parser.add_argument(
        "--order", type=int, default=0, help="Ascending rotation/display order hint."
    )
    add_parser.add_argument(
        "--disabled", action="store_true", help="Add the line but leave it disabled."
    )

    remove_parser = subparsers.add_parser(name="remove", help="Remove a presence line by id.")
    remove_parser.add_argument("status_id", type=int, help="The status_id to remove.")

    subparsers.add_parser(name="list", help="List current presence lines.")
    return parser.parse_args(args=argv)


def _print_rows(rows: list[BotStatusEntry]) -> None:
    """Prints the presence rotation rows."""
    console.print(f"[bold]Bot statuses[/bold]: {len(rows)}")
    for row in rows:
        state = "enabled" if row.enabled else "disabled"
        console.print(f"#{row.status_id} [{state}] order={row.order_index}: {row.status_text}")


async def _async_main(argv: Sequence[str] | None = None) -> None:
    """Runs the CLI."""
    args = _parse_args(argv=argv)
    if args.command == "add":
        status_id = await add_bot_status(
            status_text=args.status_text, order_index=args.order, enabled=not args.disabled
        )
        console.print(f"[bold]Added[/bold] status #{status_id}: {args.status_text}")
    elif args.command == "remove":
        removed = await remove_bot_status(status_id=args.status_id)
        console.print(
            f"[bold]Removed[/bold] status #{args.status_id}"
            if removed
            else f"[bold]No status #{args.status_id}[/bold]"
        )
    else:
        _print_rows(rows=await list_bot_status_rows())


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the bot status management CLI.

    Args:
        argv (Sequence[str] | None): Optional argument sequence to parse instead of `sys.argv`.
    """
    asyncio.run(main=_async_main(argv=argv))


if __name__ == "__main__":
    main()
