"""One-shot admin helper for adjusting a user's point balance.

Usage::

    uv run python scripts/modify_balance.py 1010833712956592200 500 --name mai9999
    uv run python scripts/modify_balance.py 1010833712956592200 -200
    uv run python scripts/modify_balance.py 1010833712956592200 -999 --allow-negative
    uv run python scripts/modify_balance.py all 50000
"""

from typing import Literal
import asyncio
import argparse
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from rich.console import Console

from discordbot.cogs._economy import database
from discordbot.cogs._economy.presentation import CURRENCY_NAME, currency_text

console = Console()

BalanceTarget = int | Literal["all"]


class BalanceChange(BaseModel):
    """Summary of a manual balance adjustment."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    name: str
    before: int
    requested_delta: int
    applied_delta: int
    after: int
    created: bool
    dry_run: bool


class BulkBalanceChange(BaseModel):
    """Summary of a manual balance adjustment applied to existing accounts."""

    model_config = ConfigDict(frozen=True)

    changes: tuple[BalanceChange, ...]
    requested_delta: int
    applied_delta: int
    dry_run: bool


def _parse_target(value: str) -> BalanceTarget:
    """Parses a single-user ID or the special all-users target."""
    if value.lower() == "all":
        return "all"
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("target must be a Discord user ID or 'all'.") from error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            f"Adjust one Discord user's, or every existing user's, {CURRENCY_NAME} "
            "balance by a signed delta."
        )
    )
    parser.add_argument(
        "target",
        type=_parse_target,
        help="Discord user ID to modify, or 'all' for every existing economy account.",
    )
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
    """Applies a manual economy balance adjustment via the database API.

    Routes the change through the same helpers the bot itself uses: positive
    deltas go through ``add_balance`` and negative deltas go through
    ``settle_game`` (which clamps at zero) or ``house_settle`` (which allows
    negative balances). Dry runs read the current state and compute the
    expected applied delta without writing.

    Args:
        user_id: Discord user ID whose account should be adjusted.
        name: Display name to store when creating or updating the account.
        delta: Signed amount to add to the current balance.
        allow_negative: Whether the resulting balance may go below zero.
        dry_run: Whether to compute and return the change without committing it.

    Returns:
        A `BalanceChange` summary describing the requested and applied change.
    """
    account = await database.get_account(user_id=user_id)
    created = account is None
    existing_name = account[0] if account is not None else ""
    before = account[1] if account is not None else 0
    effective_name = name or existing_name or str(user_id)

    after = before + delta if allow_negative else max(before + delta, 0)
    applied_delta = after - before

    if dry_run or applied_delta == 0:
        return BalanceChange(
            user_id=user_id,
            name=effective_name,
            before=before,
            requested_delta=delta,
            applied_delta=applied_delta,
            after=after,
            created=created and not dry_run and applied_delta != 0,
            dry_run=dry_run,
        )

    if applied_delta > 0:
        new_balance = await database.add_balance(
            user_id=user_id, name=effective_name, amount=applied_delta
        )
    elif allow_negative:
        # house_settle is the only public helper that allows the resulting
        # balance to go below zero; the admin CLI is its only non-dealer caller.
        new_balance = await database.house_settle(
            user_id=user_id, name=effective_name, delta=applied_delta
        )
    else:
        new_balance = await database.settle_game(
            user_id=user_id, name=effective_name, delta=applied_delta
        )

    return BalanceChange(
        user_id=user_id,
        name=effective_name,
        before=before,
        requested_delta=delta,
        applied_delta=applied_delta,
        after=new_balance,
        created=created,
        dry_run=False,
    )


async def modify_all_balances(
    *, delta: int, allow_negative: bool = False, dry_run: bool = False
) -> BulkBalanceChange:
    """Applies a manual balance adjustment to every existing economy account.

    The account list is read from the current database before writes begin.
    Missing Discord users are intentionally ignored because there is no row to
    enumerate or update.

    Args:
        delta: Signed amount to add to each existing balance.
        allow_negative: Whether resulting balances may go below zero.
        dry_run: Whether to compute the changes without committing them.

    Returns:
        A `BulkBalanceChange` summary for the requested operation.
    """
    accounts = await database.top_n(limit=2_147_483_647)
    changes: list[BalanceChange] = []
    for account in accounts:
        changes.append(
            await modify_balance(
                user_id=account[0],
                name=account[1],
                delta=delta,
                allow_negative=allow_negative,
                dry_run=dry_run,
            )
        )
    return BulkBalanceChange(
        changes=tuple(changes),
        requested_delta=delta,
        applied_delta=sum(change.applied_delta for change in changes),
        dry_run=dry_run,
    )


def _print_change(change: BalanceChange) -> None:
    """Prints a human-readable change summary."""
    title = "Dry run" if change.dry_run else "Balance modified"
    console.print(f"[bold]{title}[/bold]")
    console.print(f"user_id: {change.user_id}")
    console.print(f"name: {change.name}")
    console.print(f"created: {change.created}")
    console.print(f"before: {currency_text(amount=change.before)}")
    console.print(f"requested_delta: {currency_text(amount=change.requested_delta, signed=True)}")
    console.print(f"applied_delta: {currency_text(amount=change.applied_delta, signed=True)}")
    console.print(f"after: {currency_text(amount=change.after)}")


def _print_bulk_change(change: BulkBalanceChange) -> None:
    """Prints a human-readable bulk change summary."""
    title = "Dry run" if change.dry_run else "Balances modified"
    console.print(f"[bold]{title}[/bold]")
    console.print(f"accounts: {len(change.changes)}")
    console.print(
        f"requested_delta_each: {currency_text(amount=change.requested_delta, signed=True)}"
    )
    console.print(
        f"applied_delta_total: {currency_text(amount=change.applied_delta, signed=True)}"
    )
    for item in change.changes:
        console.print(
            f"{item.user_id} ({item.name}): "
            f"{currency_text(amount=item.before)} -> {currency_text(amount=item.after)} "
            f"({currency_text(amount=item.applied_delta, signed=True)})"
        )


async def _async_main(argv: Sequence[str] | None = None) -> None:
    """Runs the CLI."""
    args = _parse_args(argv=argv)
    if args.target == "all":
        bulk_change = await modify_all_balances(
            delta=args.delta, allow_negative=args.allow_negative, dry_run=args.dry_run
        )
        _print_bulk_change(change=bulk_change)
    else:
        change = await modify_balance(
            user_id=args.target,
            name=args.name,
            delta=args.delta,
            allow_negative=args.allow_negative,
            dry_run=args.dry_run,
        )
        _print_change(change=change)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the manual balance adjustment CLI.

    Parses command-line arguments, applies the requested balance change, and
    prints a human-readable summary to the console.

    Args:
        argv: Optional argument sequence to parse instead of `sys.argv`.
    """
    asyncio.run(main=_async_main(argv=argv))


if __name__ == "__main__":
    main()
