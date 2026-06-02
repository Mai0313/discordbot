"""One-shot offline reset for the economy after hyperinflation.

The bot MUST be stopped and the database files backed up before running this
for real. A dry run reads current state and prints the projected change without
writing anything. Always dry-run first, ideally against a copy of the databases.

Usage::

    # Preview the wallet transform only (no writes):
    uv run python scripts/reset_economy.py wallets --mode log-compress --dry-run

    # Reset only wallets with logarithmic compression:
    uv run python scripts/reset_economy.py wallets --mode log-compress --floor 1000 --scale 1000

    # Full reset: wallets + casino counters + ledger + jackpot + loans + stocks:
    uv run python scripts/reset_economy.py all --reset-stocks --dry-run
    uv run python scripts/reset_economy.py all --reset-stocks
"""

import asyncio
import argparse
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from rich.console import Console

from discordbot.typings.economy import WalletResetMode, WalletResetSummary
from discordbot.cogs._stock.database import reset_all_positions
from discordbot.cogs._economy.database import (
    top_n,
    reset_all_wallets,
    reset_casino_ledger,
    reset_jackpot_pools,
    compute_reset_balance,
    expire_loan_proposals,
    forgive_loan_contracts,
    reset_casino_daily_counters,
    count_wallet_invariant_violations,
)
from discordbot.cogs._fishing.database import reset_all_fishing
from discordbot.cogs._economy.presentation import CURRENCY_NAME, currency_text

console = Console()

_MODE_BY_FLAG: dict[str, WalletResetMode] = {
    "log-compress": WalletResetMode.LOG_COMPRESS,
    "fixed": WalletResetMode.FIXED,
    "wipe": WalletResetMode.WIPE,
}


class EconomyResetSummary(BaseModel):
    """Aggregated outcome of a full economy reset."""

    model_config = ConfigDict(frozen=True)

    wallets: WalletResetSummary
    casino_counters_cleared: int
    loans_forgiven: int
    proposals_canceled: int
    stock_positions_reset: int | None
    fishing_anglers_reset: int | None
    casino_ledger_reset: bool
    jackpot_pools_reset: bool
    invariant_violations: int
    dry_run: bool


async def reset_everything(  # noqa: PLR0913 -- exposes every wallet-transform knob plus toggles
    mode: WalletResetMode,
    floor: int,
    scale: int,
    amount: int,
    reset_stocks: bool,
    reset_fishing: bool,
    dry_run: bool,
) -> EconomyResetSummary:
    """Runs the wallet reset plus every companion cleanup.

    On a dry run only the wallet transform is computed; the companion mutators
    are left untouched and reported as zero, because they are simple state
    resets with no preview value.

    Returns:
        A summary of the full reset.
    """
    wallets = await reset_all_wallets(
        mode=mode, floor=floor, scale=scale, fixed_amount=amount, dry_run=dry_run
    )
    if dry_run:
        return EconomyResetSummary(
            wallets=wallets,
            casino_counters_cleared=0,
            loans_forgiven=0,
            proposals_canceled=0,
            stock_positions_reset=0 if reset_stocks else None,
            fishing_anglers_reset=0 if reset_fishing else None,
            casino_ledger_reset=False,
            jackpot_pools_reset=False,
            invariant_violations=0,
            dry_run=True,
        )

    casino_counters_cleared = await reset_casino_daily_counters()
    loans_forgiven = await forgive_loan_contracts()
    proposals_canceled = await expire_loan_proposals()
    await reset_casino_ledger()
    await reset_jackpot_pools()
    stock_positions_reset = await reset_all_positions() if reset_stocks else None
    fishing_anglers_reset = await reset_all_fishing() if reset_fishing else None
    invariant_violations = await count_wallet_invariant_violations()
    return EconomyResetSummary(
        wallets=wallets,
        casino_counters_cleared=casino_counters_cleared,
        loans_forgiven=loans_forgiven,
        proposals_canceled=proposals_canceled,
        stock_positions_reset=stock_positions_reset,
        fishing_anglers_reset=fishing_anglers_reset,
        casino_ledger_reset=True,
        jackpot_pools_reset=True,
        invariant_violations=invariant_violations,
        dry_run=False,
    )


async def _print_wallet_sample(mode: WalletResetMode, floor: int, scale: int, amount: int) -> None:
    """Prints the projected before/after for the current top 10 balances."""
    top = await top_n(limit=10, include_hidden=True)
    if not top:
        return
    console.print("[bold]Top 10 balance transform[/bold]")
    for entry in top:
        after = compute_reset_balance(
            entry.balance, mode, floor=floor, scale=scale, fixed_amount=amount
        )
        console.print(
            f"{entry.user_id} ({entry.name}): "
            f"{currency_text(amount=entry.balance, compact=True)} -> "
            f"{currency_text(amount=after, compact=True)}"
        )


def _print_wallet_summary(summary: WalletResetSummary) -> None:
    """Prints a human-readable wallet reset summary."""
    title = "Dry run" if summary.dry_run else "Wallets reset"
    console.print(f"[bold]{title}[/bold]")
    console.print(f"mode: {summary.mode.value}")
    console.print(f"accounts: {summary.accounts}")
    console.print(f"total before: {currency_text(amount=summary.total_before)}")
    console.print(f"total after: {currency_text(amount=summary.total_after)}")
    console.print(f"max before: {currency_text(amount=summary.max_before)}")
    console.print(f"max after: {currency_text(amount=summary.max_after)}")


def _print_full_summary(summary: EconomyResetSummary) -> None:
    """Prints a human-readable full reset summary."""
    _print_wallet_summary(summary=summary.wallets)
    if summary.dry_run:
        console.print("[yellow]Dry run: companion resets were not executed.[/yellow]")
        return
    console.print(f"casino counters cleared: {summary.casino_counters_cleared}")
    console.print(f"loans forgiven: {summary.loans_forgiven}")
    console.print(f"proposals canceled: {summary.proposals_canceled}")
    console.print(f"casino ledger reset: {summary.casino_ledger_reset}")
    console.print(f"jackpot pools reset: {summary.jackpot_pools_reset}")
    if summary.stock_positions_reset is not None:
        console.print(f"stock positions reset: {summary.stock_positions_reset}")
    if summary.fishing_anglers_reset is not None:
        console.print(f"fishing anglers reset: {summary.fishing_anglers_reset}")
    color = "green" if summary.invariant_violations == 0 else "red"
    console.print(
        f"[{color}]wallet invariant violations after reset: "
        f"{summary.invariant_violations}[/{color}]"
    )


def _add_wallet_args(parser: argparse.ArgumentParser) -> None:
    """Adds the shared wallet-transform options to a subparser."""
    parser.add_argument(
        "--mode",
        choices=list(_MODE_BY_FLAG),
        default="log-compress",
        help="Wallet transform: log-compress (default), fixed, or wipe.",
    )
    parser.add_argument(
        "--floor",
        type=int,
        default=1_000,
        help="Log-compress baseline added to positive balances.",
    )
    parser.add_argument(
        "--scale", type=int, default=1_000, help="Log-compress multiplier on the log10 term."
    )
    parser.add_argument(
        "--amount", type=int, default=10_000, help="Fixed-mode starting balance for every account."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the change without writing to the database.",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description=f"Reset the {CURRENCY_NAME} economy after hyperinflation (bot must be offline)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    wallets_parser = subparsers.add_parser("wallets", help="Reset only user wallets.")
    _add_wallet_args(parser=wallets_parser)

    all_parser = subparsers.add_parser(
        "all", help="Reset wallets, casino counters, ledger, jackpot, and loans."
    )
    _add_wallet_args(parser=all_parser)
    all_parser.add_argument(
        "--reset-stocks",
        action="store_true",
        help="Also flatten every stock position and zero realized P&L.",
    )
    all_parser.add_argument(
        "--reset-fishing",
        action="store_true",
        help="Also clear every angler's rod, bait, and catch history.",
    )

    return parser.parse_args(args=argv)


async def _async_main(argv: Sequence[str] | None = None) -> None:
    """Runs the CLI."""
    args = _parse_args(argv=argv)
    mode = _MODE_BY_FLAG[args.mode]

    if args.dry_run:
        await _print_wallet_sample(
            mode=mode, floor=args.floor, scale=args.scale, amount=args.amount
        )

    if args.command == "wallets":
        summary = await reset_all_wallets(
            mode=mode,
            floor=args.floor,
            scale=args.scale,
            fixed_amount=args.amount,
            dry_run=args.dry_run,
        )
        _print_wallet_summary(summary=summary)
        return

    full_summary = await reset_everything(
        mode=mode,
        floor=args.floor,
        scale=args.scale,
        amount=args.amount,
        reset_stocks=args.reset_stocks,
        reset_fishing=args.reset_fishing,
        dry_run=args.dry_run,
    )
    _print_full_summary(summary=full_summary)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the offline economy reset CLI.

    Parses command-line arguments, applies the requested reset, and prints a
    human-readable summary to the console.

    Args:
        argv (Sequence[str] | None): Optional argument sequence to parse instead of `sys.argv`.
    """
    asyncio.run(main=_async_main(argv=argv))


if __name__ == "__main__":
    main()
