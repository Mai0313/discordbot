"""Manage tunable game balance settings stored in `data/global_state.db`.

Usage::

    uv run python scripts/manage_game_setting.py list
    uv run python scripts/manage_game_setting.py list dragon_gate
    uv run python scripts/manage_game_setting.py get dragon_gate ante
    uv run python scripts/manage_game_setting.py set dragon_gate ante 6000
"""

import asyncio
import argparse
from collections.abc import Sequence

from rich.console import Console

from discordbot.cogs._games.settings import (
    get_game_setting,
    set_game_setting,
    list_game_settings,
)

console = Console()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Read or write rows in the `game_setting` table of data/global_state.db."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(name="list", help="List persisted game settings.")
    list_parser.add_argument(
        "game_id", nargs="?", default=None, help="Optional game id filter (e.g. dragon_gate)."
    )

    get_parser = subparsers.add_parser(name="get", help="Print one persisted setting.")
    get_parser.add_argument("game_id", help="Game id (e.g. dragon_gate).")
    get_parser.add_argument("setting_key", help="Setting key (e.g. ante).")

    set_parser = subparsers.add_parser(name="set", help="Insert or update one setting.")
    set_parser.add_argument("game_id", help="Game id (e.g. dragon_gate).")
    set_parser.add_argument("setting_key", help="Setting key (e.g. ante).")
    set_parser.add_argument("value", type=int, help="Integer value to persist.")
    return parser.parse_args(args=argv)


def _print_rows(rows: tuple[tuple[str, str, int], ...]) -> None:
    """Prints persisted game settings to the console."""
    console.print(f"[bold]Game settings[/bold]: {len(rows)}")
    for game_id, setting_key, value in rows:
        console.print(f"{game_id}.{setting_key} = {value:,}")


async def _async_main(argv: Sequence[str] | None = None) -> None:
    """Runs the CLI."""
    args = _parse_args(argv=argv)
    if args.command == "list":
        rows = await list_game_settings(game_id=args.game_id)
        _print_rows(rows=rows)
    elif args.command == "get":
        value = await get_game_setting(
            game_id=args.game_id, setting_key=args.setting_key, default=0
        )
        console.print(f"{args.game_id}.{args.setting_key} = {value:,}")
    else:
        await set_game_setting(
            game_id=args.game_id, setting_key=args.setting_key, value=args.value
        )
        console.print(
            f"[bold]Updated[/bold] {args.game_id}.{args.setting_key} = {args.value:,}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the game setting management CLI.

    Args:
        argv (Sequence[str] | None): Optional argument sequence to parse instead of `sys.argv`.
    """
    asyncio.run(main=_async_main(argv=argv))


if __name__ == "__main__":
    main()
