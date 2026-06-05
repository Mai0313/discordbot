"""Manage economy admins for Discord-side maintenance commands.

Usage::

    uv run python scripts/manage_admin.py grant 1010833712956592200 --name mai9999
    uv run python scripts/manage_admin.py revoke 1010833712956592200
    uv run python scripts/manage_admin.py list
"""

from typing import Literal
import asyncio
from pathlib import Path
import argparse
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from rich.console import Console

from discordbot.typings.economy import AdminAccount
from discordbot.cogs._economy.database import get_admin, set_admin, list_admins

console = Console()

AdminAction = Literal["grant", "revoke"]


class AdminChange(BaseModel):
    """Summary of an admin flag change."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    name: str
    action: AdminAction
    applied: bool
    is_admin: bool


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Grant, revoke, or list economy admins stored in data/database/economy.db."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    grant_parser = subparsers.add_parser(name="grant", help="Grant economy admin access.")
    grant_parser.add_argument("user_id", type=int, help="Discord user ID to grant.")
    grant_parser.add_argument(
        "--name", default="", help="Display name to store when creating or updating the account."
    )
    grant_parser.add_argument(
        "--avatar-url",
        default="",
        help="Avatar URL to store when creating or updating the account.",
    )

    revoke_parser = subparsers.add_parser(name="revoke", help="Revoke economy admin access.")
    revoke_parser.add_argument("user_id", type=int, help="Discord user ID to revoke.")
    revoke_parser.add_argument(
        "--name", default="", help="Display name to refresh if the account exists."
    )

    subparsers.add_parser(name="list", help="List current economy admins.")
    return parser.parse_args(args=argv)


async def grant_admin(user_id: int, name: str = "", avatar_url: str = "") -> AdminChange:
    """Grants economy admin access to a Discord user.

    Args:
        user_id (int): Discord user ID to grant.
        name (str): Display name to store when creating or updating the account.
        avatar_url (str): Avatar URL to store when creating or updating the account.

    Returns:
        AdminChange: Applied flag state after the update.
    """
    applied = await set_admin(user_id=user_id, name=name, avatar_url=avatar_url, is_admin=True)
    return AdminChange(
        user_id=user_id,
        name=name or str(user_id),
        action="grant",
        applied=applied,
        is_admin=await get_admin(user_id=user_id),
    )


async def revoke_admin(user_id: int, name: str = "") -> AdminChange:
    """Revokes economy admin access from an existing Discord user.

    Args:
        user_id (int): Discord user ID to revoke.
        name (str): Display name to refresh if the account exists.

    Returns:
        AdminChange: Applied flag state after the update.
    """
    applied = await set_admin(user_id=user_id, name=name, is_admin=False)
    return AdminChange(
        user_id=user_id,
        name=name or str(user_id),
        action="revoke",
        applied=applied,
        is_admin=await get_admin(user_id=user_id),
    )


async def list_admin_accounts() -> list[AdminAccount]:
    """Lists all economy admins.

    Returns:
        list[AdminAccount]: Current economy admin accounts.
    """
    return await list_admins()


def _print_change(change: AdminChange) -> None:
    """Prints one admin flag change."""
    console.print(
        "[bold]Admin updated[/bold]" if change.applied else "[bold]No matching row[/bold]"
    )
    console.print(f"user_id: {change.user_id}")
    console.print(f"name: {change.name}")
    console.print(f"action: {change.action}")
    console.print(f"is_admin: {change.is_admin}")


def _print_admins(admins: list[AdminAccount]) -> None:
    """Prints the admin account list."""
    console.print(f"[bold]Economy admins[/bold]: {len(admins)}")
    for admin in admins:
        console.print(f"{admin.user_id}: {admin.name}")


async def _async_main(argv: Sequence[str] | None = None) -> None:
    """Runs the CLI."""
    args = _parse_args(argv=argv)
    if args.command == "grant":
        change = await grant_admin(
            user_id=args.user_id, name=args.name, avatar_url=args.avatar_url
        )
        _print_change(change=change)
    elif args.command == "revoke":
        change = await revoke_admin(user_id=args.user_id, name=args.name)
        _print_change(change=change)
    else:
        _print_admins(admins=await list_admin_accounts())


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the admin management CLI.

    Args:
        argv (Sequence[str] | None): Optional argument sequence to parse instead of `sys.argv`.
    """
    # data/ is gitignored and may not exist on a fresh checkout seeded before the
    # bot's first run, so create it here like cli.py does before any DB write.
    Path("./data/database").mkdir(parents=True, exist_ok=True)
    asyncio.run(main=_async_main(argv=argv))


if __name__ == "__main__":
    main()
