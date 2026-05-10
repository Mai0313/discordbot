"""One-shot admin helper for adjusting a user's point balance.

Usage::

    uv run python scripts/modify_balance.py 123456789012345678 500 --name alice
    uv run python scripts/modify_balance.py 123456789012345678 -200
    uv run python scripts/modify_balance.py 123456789012345678 -999 --allow-negative
"""

import asyncio
import argparse
from datetime import UTC, datetime
from dataclasses import dataclass
from collections.abc import Sequence

from rich.console import Console

from discordbot.cogs._economy import database
from discordbot.cogs._economy.database import UserAccount

console = Console()


@dataclass(frozen=True)
class BalanceChange:
    """Summary of a manual balance adjustment."""

    user_id: int
    name: str
    before: int
    requested_delta: int
    applied_delta: int
    after: int
    created: bool
    dry_run: bool


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Adjust a Discord user's economy balance by a signed delta."
    )
    parser.add_argument("user_id", type=int, help="Discord user ID to modify.")
    parser.add_argument(
        "delta", type=int, help="Signed amount to add or subtract, for example 500 or -200."
    )
    parser.add_argument(
        "--name",
        default="",
        help="Display name to store on the account. Existing name is kept when omitted.",
    )
    parser.add_argument(
        "--allow-negative",
        action="store_true",
        help="Allow the resulting balance to go below zero. By default it clamps at zero.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the resulting change without writing to the database.",
    )
    return parser.parse_args(args=argv)


async def modify_balance(
    *, user_id: int, name: str, delta: int, allow_negative: bool = False, dry_run: bool = False
) -> BalanceChange:
    """Applies a signed balance adjustment and returns a summary."""
    async with database.open_session() as session:
        account = await session.get(entity=UserAccount, ident=user_id)
        created = account is None

        if account is None:
            if delta <= 0 and not allow_negative:
                return BalanceChange(
                    user_id=user_id,
                    name=name or str(user_id),
                    before=0,
                    requested_delta=delta,
                    applied_delta=0,
                    after=0,
                    created=False,
                    dry_run=dry_run,
                )
            account = UserAccount(user_id=user_id, name=name or str(user_id))
            session.add(instance=account)
            await session.flush()

        before = account.balance
        after = before + delta if allow_negative else max(before + delta, 0)
        applied_delta = after - before

        if dry_run:
            await session.rollback()
            return BalanceChange(
                user_id=user_id,
                name=name or account.name,
                before=before,
                requested_delta=delta,
                applied_delta=applied_delta,
                after=after,
                created=created,
                dry_run=True,
            )

        account.balance = after
        if name:
            account.name = name
        if applied_delta > 0:
            account.total_earned += applied_delta
        elif applied_delta < 0:
            account.total_spent += -applied_delta
        account.updated_at = datetime.now(tz=UTC)

        await session.commit()
        return BalanceChange(
            user_id=user_id,
            name=account.name,
            before=before,
            requested_delta=delta,
            applied_delta=applied_delta,
            after=after,
            created=created,
            dry_run=False,
        )


def _print_change(change: BalanceChange) -> None:
    """Prints a human-readable change summary."""
    title = "Dry run" if change.dry_run else "Balance modified"
    console.print(f"[bold]{title}[/bold]")
    console.print(f"user_id: {change.user_id}")
    console.print(f"name: {change.name}")
    console.print(f"created: {change.created}")
    console.print(f"before: {change.before:,}")
    console.print(f"requested_delta: {change.requested_delta:+,}")
    console.print(f"applied_delta: {change.applied_delta:+,}")
    console.print(f"after: {change.after:,}")


async def _async_main(argv: Sequence[str] | None = None) -> None:
    """Runs the CLI."""
    args = _parse_args(argv=argv)
    change = await modify_balance(
        user_id=args.user_id,
        name=args.name,
        delta=args.delta,
        allow_negative=args.allow_negative,
        dry_run=args.dry_run,
    )
    _print_change(change=change)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    asyncio.run(main=_async_main(argv=argv))


if __name__ == "__main__":
    main()
