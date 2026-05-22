"""Manage central banker access for central-bank lending commands.

Usage::

    uv run python scripts/manage_central_banker.py grant 1010833712956592200 --name mai9999
    uv run python scripts/manage_central_banker.py revoke 1010833712956592200
    uv run python scripts/manage_central_banker.py list
"""

from typing import Literal
import asyncio
import argparse
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from rich.console import Console

from discordbot.typings.economy import CentralBankerAccount
from discordbot.cogs._economy.database import (
    get_central_banker,
    set_central_banker,
    list_central_bankers,
)

console = Console()

CentralBankerAction = Literal["grant", "revoke"]


class CentralBankerChange(BaseModel):
    """Summary of a central banker flag change."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    name: str
    action: CentralBankerAction
    applied: bool
    is_central_banker: bool


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Grant, revoke, or list central bankers stored in data/economy.db."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    grant_parser = subparsers.add_parser(name="grant", help="Grant central banker access.")
    grant_parser.add_argument("user_id", type=int, help="Discord user ID to grant.")
    grant_parser.add_argument(
        "--name", default="", help="Display name to store when creating or updating the account."
    )
    grant_parser.add_argument(
        "--avatar-url",
        default="",
        help="Avatar URL to store when creating or updating the account.",
    )

    revoke_parser = subparsers.add_parser(name="revoke", help="Revoke central banker access.")
    revoke_parser.add_argument("user_id", type=int, help="Discord user ID to revoke.")
    revoke_parser.add_argument(
        "--name", default="", help="Display name to refresh if the account exists."
    )

    subparsers.add_parser(name="list", help="List current central bankers.")
    return parser.parse_args(args=argv)


async def grant_central_banker(
    user_id: int, name: str = "", avatar_url: str = ""
) -> CentralBankerChange:
    """Grants central banker access to a Discord user.

    Args:
        user_id (int): Discord user ID to grant.
        name (str): Display name to store when creating or updating the account.
        avatar_url (str): Avatar URL to store when creating or updating the account.

    Returns:
        CentralBankerChange: Applied flag state after the update.
    """
    applied = await set_central_banker(
        user_id=user_id, name=name, avatar_url=avatar_url, is_central_banker=True
    )
    return CentralBankerChange(
        user_id=user_id,
        name=name or str(user_id),
        action="grant",
        applied=applied,
        is_central_banker=await get_central_banker(user_id=user_id),
    )


async def revoke_central_banker(user_id: int, name: str = "") -> CentralBankerChange:
    """Revokes central banker access from an existing Discord user.

    Args:
        user_id (int): Discord user ID to revoke.
        name (str): Display name to refresh if the account exists.

    Returns:
        CentralBankerChange: Applied flag state after the update.
    """
    applied = await set_central_banker(user_id=user_id, name=name, is_central_banker=False)
    return CentralBankerChange(
        user_id=user_id,
        name=name or str(user_id),
        action="revoke",
        applied=applied,
        is_central_banker=await get_central_banker(user_id=user_id),
    )


async def list_central_banker_accounts() -> list[CentralBankerAccount]:
    """Lists all central bankers.

    Returns:
        list[CentralBankerAccount]: Current central banker accounts.
    """
    return await list_central_bankers()


def _print_change(change: CentralBankerChange) -> None:
    """Prints one central banker flag change."""
    console.print(
        "[bold]Central banker updated[/bold]" if change.applied else "[bold]No matching row[/bold]"
    )
    console.print(f"user_id: {change.user_id}")
    console.print(f"name: {change.name}")
    console.print(f"action: {change.action}")
    console.print(f"is_central_banker: {change.is_central_banker}")


def _print_central_bankers(central_bankers: list[CentralBankerAccount]) -> None:
    """Prints the central banker account list."""
    console.print(f"[bold]Central bankers[/bold]: {len(central_bankers)}")
    for central_banker in central_bankers:
        console.print(f"{central_banker.user_id}: {central_banker.name}")


async def _async_main(argv: Sequence[str] | None = None) -> None:
    """Runs the CLI."""
    args = _parse_args(argv=argv)
    if args.command == "grant":
        change = await grant_central_banker(
            user_id=args.user_id, name=args.name, avatar_url=args.avatar_url
        )
        _print_change(change=change)
    elif args.command == "revoke":
        change = await revoke_central_banker(user_id=args.user_id, name=args.name)
        _print_change(change=change)
    else:
        _print_central_bankers(central_bankers=await list_central_banker_accounts())


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the central banker management CLI.

    Args:
        argv (Sequence[str] | None): Optional argument sequence to parse instead of `sys.argv`.
    """
    asyncio.run(main=_async_main(argv=argv))


if __name__ == "__main__":
    main()
