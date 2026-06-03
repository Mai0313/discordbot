"""Tests for the offline economy reset helpers and script."""

from datetime import datetime

import pytest
from scripts import reset_economy as reset_script
from sqlalchemy import select, update

from discordbot.typings.economy import AccountSnapshot, WalletResetMode, WalletResetSummary
from discordbot.cogs._economy.database import (
    UserWallet,
    JackpotPool,
    get_account,
    open_session,
    adjust_balance,
    set_wallet_exact,
    get_casino_ledger,
    reset_all_wallets,
    list_loan_contracts,
    reset_casino_ledger,
    reset_jackpot_pools,
    accept_loan_proposal,
    get_jackpot_snapshot,
    compute_reset_balance,
    expire_loan_proposals,
    apply_round_settlement,
    forgive_loan_contracts,
    get_casino_daily_stats,
    open_global_state_session,
    reset_casino_daily_counters,
    create_personal_loan_request,
    count_wallet_invariant_violations,
)

pytestmark = pytest.mark.usefixtures("economy_isolated_db")

_WHALE_BALANCE = 35749999999999999999999999675438673965869784416508048648395374


async def _add_balance(user_id: int, name: str, amount: int) -> None:
    """Seeds a balance through the manual adjustment API."""
    await adjust_balance(user_id=user_id, name=name, delta=amount)


# compute_reset_balance (pure) ----------------------------------------------


def test_compute_reset_balance_wipe_is_zero() -> None:
    """WIPE always returns zero."""
    assert compute_reset_balance(_WHALE_BALANCE, WalletResetMode.WIPE) == 0


def test_compute_reset_balance_fixed_is_flat() -> None:
    """FIXED returns the configured starting amount for any balance."""
    assert compute_reset_balance(1, WalletResetMode.FIXED, fixed_amount=10_000) == 10_000
    assert (
        compute_reset_balance(_WHALE_BALANCE, WalletResetMode.FIXED, fixed_amount=10_000) == 10_000
    )


def test_compute_reset_balance_log_compress_is_monotonic_and_bounded() -> None:
    """LOG_COMPRESS preserves ordering while collapsing magnitude."""
    empty = compute_reset_balance(0, WalletResetMode.LOG_COMPRESS)
    small = compute_reset_balance(50_000, WalletResetMode.LOG_COMPRESS)
    normal = compute_reset_balance(7_000_000, WalletResetMode.LOG_COMPRESS)
    whale = compute_reset_balance(_WHALE_BALANCE, WalletResetMode.LOG_COMPRESS)

    assert empty == 0
    assert empty < small < normal < whale
    # 58 orders of magnitude collapse into a tight band.
    assert whale < 100_000


def test_compute_reset_balance_log_compress_clamps_negative_inputs() -> None:
    """A mistyped negative floor/scale cannot produce a negative balance."""
    assert (
        compute_reset_balance(1_000_000, WalletResetMode.LOG_COMPRESS, floor=0, scale=-1000) == 0
    )


# reset_all_wallets ----------------------------------------------------------


async def _seed_mixed_wallets() -> None:
    """Seeds a broken-invariant row, a normal row, and a whale row."""
    # Deliberately broken invariant: balance != total_earned - total_spent.
    await set_wallet_exact(user_id=1, balance=100, total_earned=999, total_spent=0, name="broken")
    await set_wallet_exact(
        user_id=2, balance=7_000_000, total_earned=7_000_000, total_spent=0, name="normal"
    )
    await set_wallet_exact(
        user_id=3, balance=_WHALE_BALANCE, total_earned=_WHALE_BALANCE, total_spent=0, name="whale"
    )


async def test_reset_all_wallets_log_compress_repairs_invariant_and_preserves_rank() -> None:
    """LOG_COMPRESS rewrites every triple, fixing the invariant and keeping order."""
    await _seed_mixed_wallets()
    assert await count_wallet_invariant_violations() == 1

    summary = await reset_all_wallets(mode=WalletResetMode.LOG_COMPRESS)

    assert isinstance(summary, WalletResetSummary)
    assert summary.accounts == 3
    assert summary.mode is WalletResetMode.LOG_COMPRESS
    assert summary.max_before == _WHALE_BALANCE
    assert summary.max_after < 100_000
    assert not summary.dry_run

    broken = await get_account(user_id=1)
    normal = await get_account(user_id=2)
    whale = await get_account(user_id=3)
    assert broken is not None
    assert normal is not None
    assert whale is not None
    # Invariant holds by construction for every row.
    assert await count_wallet_invariant_violations() == 0
    for account in (broken, normal, whale):
        assert account.total_spent == 0
        assert account.total_earned == account.balance
    # Rank ordering preserved.
    assert broken.balance < normal.balance < whale.balance


async def test_reset_all_wallets_fixed_sets_flat_balance() -> None:
    """FIXED assigns the same starting balance to every account."""
    await _seed_mixed_wallets()

    await reset_all_wallets(mode=WalletResetMode.FIXED, fixed_amount=10_000)

    for user_id in (1, 2, 3):
        account = await get_account(user_id=user_id)
        assert account is not None
        assert (account.balance, account.total_earned, account.total_spent) == (10_000, 10_000, 0)


async def test_reset_all_wallets_wipe_zeroes_everything() -> None:
    """WIPE zeroes every wallet triple."""
    await _seed_mixed_wallets()

    await reset_all_wallets(mode=WalletResetMode.WIPE)

    for user_id in (1, 2, 3):
        account = await get_account(user_id=user_id)
        assert account is not None
        assert (account.balance, account.total_earned, account.total_spent) == (0, 0, 0)


async def test_reset_all_wallets_updates_wallet_timestamps() -> None:
    """Wallet rows rewritten by an offline reset get a fresh update timestamp."""
    await _seed_mixed_wallets()
    old_timestamp = datetime(2026, 1, 1)
    async with open_session() as session:
        await session.execute(statement=update(UserWallet).values(updated_at=old_timestamp))
        await session.commit()

    await reset_all_wallets(mode=WalletResetMode.LOG_COMPRESS)

    async with open_session() as session:
        updated_at_values = (
            (await session.execute(statement=select(UserWallet.updated_at))).scalars().all()
        )

    assert updated_at_values
    assert all(updated_at > old_timestamp for updated_at in updated_at_values)


async def test_reset_all_wallets_dry_run_does_not_write() -> None:
    """A dry run computes the summary without mutating any wallet."""
    await _seed_mixed_wallets()

    summary = await reset_all_wallets(mode=WalletResetMode.WIPE, dry_run=True)

    assert summary.dry_run
    assert summary.total_after == 0
    whale = await get_account(user_id=3)
    assert whale is not None
    assert whale.balance == _WHALE_BALANCE


# set_wallet_exact -----------------------------------------------------------


async def test_set_wallet_exact_creates_and_overwrites() -> None:
    """set_wallet_exact writes the full triple, creating then overwriting the row."""
    await set_wallet_exact(user_id=7, balance=500, total_earned=500, total_spent=0, name="alice")
    assert await get_account(user_id=7) == AccountSnapshot(
        name="alice", balance=500, total_earned=500, total_spent=0
    )

    await set_wallet_exact(user_id=7, balance=10, total_earned=30, total_spent=20, name="alice")
    assert await get_account(user_id=7) == AccountSnapshot(
        name="alice", balance=10, total_earned=30, total_spent=20
    )


# Companion resets -----------------------------------------------------------


async def test_reset_casino_ledger_zeroes_the_ledger() -> None:
    """Resetting the casino ledger clears its cumulative P&L."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-40, casino_delta=40
    )
    assert (await get_casino_ledger()).balance == 40

    await reset_casino_ledger()

    ledger = await get_casino_ledger()
    assert (ledger.balance, ledger.total_earned, ledger.total_spent) == (0, 0, 0)


async def test_reset_casino_daily_counters_removes_rows() -> None:
    """Resetting daily counters deletes per-user casino rows."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-40, casino_delta=40
    )
    assert (await get_casino_daily_stats(user_id=1)).daily_loss == 40

    removed = await reset_casino_daily_counters()

    assert removed >= 1
    assert (await get_casino_daily_stats(user_id=1)).daily_loss == 0


async def test_reset_jackpot_pools_restores_seed() -> None:
    """Resetting jackpot pools restores the seed balance and generation 0."""
    async with open_global_state_session() as session:
        session.add(
            JackpotPool(
                game_id="dragon_gate",
                pool_balance=999_999,
                total_contributed=500,
                total_claimed=300,
                seeded_amount=50,
                generation=5,
            )
        )
        await session.commit()

    await reset_jackpot_pools()

    snapshot = await get_jackpot_snapshot(game_id="dragon_gate")
    assert snapshot.balance == 1_000
    assert snapshot.generation == 0


async def test_forgive_loan_contracts_closes_active_contracts() -> None:
    """Forgiving loans closes active contracts without moving balances."""
    await _add_balance(user_id=2, name="bob", amount=1_000)
    proposal = await create_personal_loan_request(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=500
    )
    assert proposal is not None
    accepted = await accept_loan_proposal(
        proposal_id=proposal.proposal_id, actor_id=2, actor_name="bob"
    )
    assert accepted is not None
    assert len(await list_loan_contracts(user_id=1)) == 1

    forgiven = await forgive_loan_contracts()

    assert forgiven == 1
    assert await list_loan_contracts(user_id=1) == []


async def test_expire_loan_proposals_cancels_pending() -> None:
    """Expiring proposals cancels every pending request."""
    await _add_balance(user_id=2, name="bob", amount=1_000)
    proposal = await create_personal_loan_request(
        borrower_id=1, borrower_name="alice", lender_id=2, lender_name="bob", amount=500
    )
    assert proposal is not None

    canceled = await expire_loan_proposals()

    assert canceled == 1


# Script ---------------------------------------------------------------------


def test_script_parse_args_defaults_to_log_compress() -> None:
    """The wallets subcommand defaults to log-compress mode."""
    args = reset_script._parse_args(argv=["wallets"])
    assert args.command == "wallets"
    assert args.mode == "log-compress"
    assert not args.dry_run


def test_script_parse_args_all_accepts_reset_stocks() -> None:
    """The all subcommand accepts the stock and fishing reset opt-ins."""
    args = reset_script._parse_args(argv=["all", "--reset-stocks", "--reset-fishing", "--dry-run"])
    assert args.command == "all"
    assert args.reset_stocks
    assert args.reset_fishing
    assert args.dry_run


async def test_script_reset_everything_dry_run_skips_companions() -> None:
    """A dry run previews wallets and leaves companion state untouched."""
    await _seed_mixed_wallets()

    summary = await reset_script.reset_everything(
        mode=WalletResetMode.LOG_COMPRESS,
        floor=1_000,
        scale=1_000,
        amount=10_000,
        reset_stocks=True,
        reset_fishing=True,
        dry_run=True,
    )

    assert summary.dry_run
    assert summary.wallets.accounts == 3
    assert not summary.casino_ledger_reset
    whale = await get_account(user_id=3)
    assert whale is not None
    assert whale.balance == _WHALE_BALANCE
