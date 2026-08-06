"""The ledger every money-moving feature writes through, over `data/database/economy.db`.

It is the only writer of that file, and it owns seven tables: identity, VIP, admin and check-in
state on `user_account`; spendable money and lifetime gross totals on `user_wallet`; per-day
casino counters on `casino_account`; long-term lending on `loan_proposal` / `loan_contract`; and
the two house-side rows, `jackpot_pool` and `casino_ledger`. No ORM row escapes the module: every
structured value handed back is one of the frozen models in `typings/economy.py`, so a caller sees
plain ints and never a mapped instance it could mutate behind the engine's back.

Callers come from both layers above. `cli.py`'s per-message reward and `cogs/economy/` (the
`/pocat`, `/balance`, `/give`, `/checkin`, `/vip`, `/leaderboard`, `/loss_leaderboard`, `/casino`
and lending surface) sit directly on top; `cogs/games/` settles Blackjack, Dragon Gate and fishing
here; and two sibling engines, `services/stock/database.py` and `cogs/games/fishing/database.py`,
spend and collect wallet cash through this module rather than keeping money of their own. Nothing
here imports a cog or touches a Discord object. `scripts/modify_balance.py` is the offline path,
and it goes through `adjust_balance` like everything else.

The engine is a module-level `AsyncEngine` singleton. Putting `create_async_engine()` on a
per-instance `cached_property` would leak the connection pool, dialect cache, and inspector cache
for every Discord interaction (the same lesson `cogs/log_msg/cog.py` captures for the sync engine
it still uses for pandas `to_sql`).

Every balance-mutating write path is atomic at the SQLite transaction level. Most paths are a
single UPSERT (`INSERT ... ON CONFLICT DO UPDATE`) or a conditional
`UPDATE ... WHERE ... RETURNING`; multi-row finance paths still roll back as one unit when a
conditional write loses a race. The previous implementation read the row in Python, mutated
`account.balance`, and committed; two coroutines racing on the same user would lose updates, and
two coroutines racing on a brand-new user would both `INSERT` and one would raise `IntegrityError`.
Where a value cannot be pinned in the predicate the loop reads, computes, and writes conditionally
against the observed value, retrying a small fixed number of times; those budgets exist to turn a
degenerate hot-row livelock into a raised `RuntimeError` rather than an endless spin.

The accounting invariant every write path preserves is `total_earned - total_spent == balance` per
wallet: an applied positive delta bumps `total_earned`, an applied negative one bumps
`total_spent`, and the APPLIED amount is what is recorded rather than the requested one, since a
clamped debit may collect less than it asked for. There is no transaction table, so those two
lifetime totals are the only history a wallet keeps. Every money column is a `StoredInteger`
(canonical decimal text on disk, unbounded `int` in Python) to escape SQLite's 64-bit ceiling,
which is why sums and comparisons go through the UDFs `utils/stored_integer.py` registers, and why
`top_n` has to build an explicit numeric ORDER BY instead of sorting the column directly.

PRAGMA setup at connect-time enables WAL (so reads don't block on writes), sets a tolerant
`busy_timeout`, and picks `synchronous=NORMAL` (the right durability trade-off in WAL: every commit
fsyncs the WAL frame, and the main file is fsynced on checkpoint).

We use `aiosqlite` so every DB call stays on the event loop: no `asyncio.to_thread` shim, no
separate `_*_sync` helpers. Each operation opens an `AsyncSession` bound to the current `_engine`,
so tests can monkeypatch `_engine` per-test and every subsequent call sees the swap.

VIP, admin status, and leaderboard visibility are boolean columns on `user_account`. VIP bumps
daily check-in rewards and the player's winning payout from games. The flag is permanent once set.
Admin and central-banker status gate maintenance-only economy commands and are managed out-of-band
by scripts. Daily casino counters live on `casino_account` so `/loss_leaderboard` can read
current-day gross losses without scanning an audit log.

Long-term lending lives in `loan_proposal` and `loan_contract`. Personal loan requests debit the
lender on acceptance, and central-bank loans mint borrower balance on approval.

Shared jackpot pools and the casino ledger live in the same `economy.db` file as the per-user rows,
so runtime casino and jackpot settlement applies the player delta and the house-side mirror in one
atomic SQLite transaction.

Two shapes recur through the file and are worth reading once. A `*_in_session` helper takes the
caller's `AsyncSession`, writes without committing, and leaves the commit / rollback decision to
the public entry point that opened it, which is what lets a settlement touch a wallet, a daily
counter and a pool as one unit. And every public call that touches the database opens with
`await _ensure_schema()`, directly or through the one it delegates to, since the schema is created
lazily on first use rather than by a migration step.
"""

from time import monotonic
from typing import TYPE_CHECKING, Any, Final, cast
import asyncio
from datetime import datetime, timedelta
from collections.abc import Sequence

import logfire
from sqlalchemy import (
    Index,
    String,
    Boolean,
    Integer,
    DateTime,
    case,
    desc,
    func,
    text,
    event,
    select,
    update,
)
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.sql.dml import ReturningInsert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.dialects.sqlite import insert

from discordbot.utils.timezone import as_taipei as _as_taipei
from discordbot.utils.timezone import database_now as _database_now
from discordbot.typings.economy import (
    TRANSFER_TAX_BPS,
    MIN_INTEREST_DAYS,
    VIP_PURCHASE_COST,
    CHECKIN_STREAK_CYCLE,
    MAX_LOAN_MONTHLY_RATE_BPS,
    MIN_LOAN_MONTHLY_RATE_BPS,
    BASE_CHECKIN_REWARD_AMOUNT,
    DEFAULT_LOAN_MONTHLY_RATE_BPS,
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
    AdminAccount,
    CreditResult,
    CheckinResult,
    PortfolioView,
    LoanLenderType,
    TransferResult,
    WalletDeltaLeg,
    AccountSnapshot,
    JackpotSnapshot,
    CasinoDailyStats,
    LeaderboardEntry,
    LoanContractView,
    LoanProposalKind,
    LoanProposalView,
    CentralBankStatus,
    LoanPaymentResult,
    VipPurchaseResult,
    LoanContractStatus,
    LoanProposalStatus,
    CasinoLedgerSnapshot,
    LossLeaderboardEntry,
    RoundSettlementResult,
    BalanceAdjustmentResult,
    JackpotSettlementResult,
    JackpotSettlementRequest,
    LoanProposalAcceptResult,
    OrderedWalletDeltaResult,
    JackpotSettlementBatchResult,
)
from discordbot.utils.asyncio_locks import LoopLocalLock
from discordbot.utils.sqlite_config import ensure_sqlite_hooks, configure_sqlite_connection
from discordbot.utils.stored_integer import StoredInteger
from discordbot.utils.stored_integer import stored_int_to_int as _stored_int_to_int
from discordbot.utils.stored_integer import stored_int_to_text as _stored_int_to_text

if TYPE_CHECKING:
    from sqlalchemy.engine import CursorResult
    from sqlalchemy.sql.elements import ColumnElement

# SELECT-then-conditional-UPDATE loops keep a small retry budget. With WAL +
# busy_timeout, contention is rare and resolves on the first or second retry;
# the bound prevents a degenerate hot-row livelock.
_CHECKIN_MAX_RETRIES: Final[int] = 8
_VIP_PURCHASE_MAX_RETRIES: Final[int] = 8
_CLAMPED_DELTA_MAX_RETRIES: Final[int] = 8
_JACKPOT_CLAIM_MAX_RETRIES: Final[int] = 8
_ECONOMY_LEADERBOARD_CACHE_TTL_SECONDS: Final[float] = 5.0
# Blackjack VIP perk: 1.2x payout on winning rounds, applied as floor(delta * 6 / 5).
_VIP_WIN_MULTIPLIER_NUM: Final[int] = 6
_VIP_WIN_MULTIPLIER_DEN: Final[int] = 5

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/economy.db")


def _taipei_midnight(now: datetime) -> datetime:
    """Returns the most recent Asia/Taipei 00:00 boundary at or before `now`.

    Every daily reset in the economy is keyed on this one boundary, so a check-in streak and a
    casino counter always roll over together regardless of the caller's own timezone.

    Args:
        now (datetime): The instant to snap back to midnight.

    Returns:
        Taipei-local midnight of the day containing `now`.
    """
    local = _as_taipei(dt=now)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def _configure_sqlite_connection(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Configures a newly opened economy SQLite connection.

    Foreign keys are enabled defensively for any future FK constraint; no table here declares one
    today. This is the economy's only deviation from the shared PRAGMA setup.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
    """
    configure_sqlite_connection(dbapi_connection=dbapi_connection, enable_foreign_keys=True)


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """SQLAlchemy `connect` listener: configures each connection the pool opens.

    Registered at import time against the engine that existed then, which is why
    `ensure_sqlite_hooks` re-installs it on whatever `_engine` currently is.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
        _connection_record (Any): SQLAlchemy's pool record, unused.
    """
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _configure_sqlite_on_checkout(
    dbapi_connection: object, _connection_record: object, _connection_proxy: object
) -> None:
    """SQLAlchemy `checkout` listener: configures a pooled connection on its way out.

    The `connect` hook alone misses connections a test-swapped engine had already pooled before
    the listener was installed; `ensure_sqlite_hooks`'s docstring has the full reasoning.

    Args:
        dbapi_connection (object): The pooled DBAPI connection being handed out.
        _connection_record (object): SQLAlchemy's pool record, unused.
        _connection_proxy (object): SQLAlchemy's connection proxy, unused.
    """
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


class Base(DeclarativeBase):
    """Base class for economy ORM models."""

    pass


class UserAccount(Base):
    """Persistent identity, VIP, admin, and check-in state for a Discord user.

    Spendable balance and lifetime gross totals live in `user_wallet`. Debt
    state lives in `loan_contract` and daily casino counters live in
    `casino_account`. `last_checkin_at` is nullable for users who have never
    checked in.

    Attributes:
        user_id: Discord user ID; primary key.
        name: Last-seen Discord username (refreshed on every write).
        avatar_url: Last-seen Discord avatar URL (refreshed on writes that carry it).
        updated_at: Taiwan-local timestamp of the last write.
        is_vip: Permanent VIP flag toggled by a successful `/vip` purchase.
        last_checkin_at: Timestamp of the latest `/checkin` payout; `None`
            for users who have never checked in.
        checkin_streak: Consecutive-day streak (1..`CHECKIN_STREAK_CYCLE`),
            persisted after the latest `/checkin`. 0 means never checked in.
        is_admin: Whether the user can run Discord-side economy admin commands.
        is_central_banker: Whether the user may approve central-bank loans. Set
            offline by direct DB write, and separate from Discord admin.
        hide_from_leaderboard: Whether the account is omitted from public balance
            and daily casino loss leaderboards.
    """

    __tablename__ = "user_account"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(length=128), default="")
    avatar_url: Mapped[str] = mapped_column(String(length=2048), default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )
    is_vip: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_checkin_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    checkin_streak: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_central_banker: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False
    )
    hide_from_leaderboard: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False
    )


class UserWallet(Base):
    """Spendable balance and lifetime gross totals for a Discord user.

    Split from `user_account` so a money write never has to touch identity or flags. All three
    amounts are `StoredInteger` and `balance == total_earned - total_spent` holds per row. `name`
    is a denormalized last-seen username every write path refreshes; nothing reads it today, since
    the leaderboards join `user_account` for the display name.
    """

    __tablename__ = "user_wallet"
    __table_args__ = (
        # StoredInteger persists decimal text, so /leaderboard uses an integer-aware
        # ORDER BY expression. This index still helps point lookups and future
        # schema migration paths, but it cannot satisfy that computed sort by itself.
        Index("ix_user_wallet_balance", "balance"),
    )

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    balance: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_earned: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_spent: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


class CasinoAccount(Base):
    """Daily per-user casino counters for loss leaderboard queries.

    One row per user, rewritten rather than appended: `day_started_at` is the day the counters
    belong to, and a settlement arriving after Taipei midnight resets them instead of adding.
    So this table is today's tally, never a history, which is why `/loss_leaderboard` needs no
    audit log to scan.

    Attributes:
        user_id: Discord user ID; primary key.
        name: Last-seen Discord username for quick inspection.
        day_started_at: Asia/Taipei midnight for the stored counters.
        daily_loss: Current-day gross loss from player-side casino settlements,
            stored as a decimal string.
        daily_win: Current-day gross win from player-side casino settlements,
            stored as a decimal string.
        daily_net: Current-day signed net casino result, stored as a decimal string.
        updated_at: Taiwan-local timestamp of the last casino counter write.
    """

    __tablename__ = "casino_account"
    __table_args__ = (
        # The day prefix is what /loss_leaderboard's equality filter rides on. The suffix cannot
        # supply the ordering: decimal text sorts by length first, so the query orders on
        # length(daily_loss) and SQLite has to sort the day's rows itself.
        Index("ix_casino_account_day_loss", "day_started_at", "daily_loss"),
    )

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    day_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    daily_loss: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    daily_win: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    daily_net: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


class LoanProposal(Base):
    """Pending long-term lending proposal.

    Personal loan requests wait for the target lender to accept. Central-bank
    requests wait for a central banker approval and do not escrow a user
    balance.

    A proposal is terminal once `status` leaves `pending`; there is no un-reject. Both creators
    write `escrow_amount=0` today, because a personal loan debits the lender at acceptance rather
    than at proposal time, so the refund a reject / cancel / expire runs against it is a no-op in
    practice. The row is kept after the decision as the audit trail for the contract it opened.
    """

    __tablename__ = "loan_proposal"
    __table_args__ = (
        Index("ix_loan_proposal_status_kind", "status", "kind"),
        Index("ix_loan_proposal_borrower_status", "borrower_id", "status"),
        Index("ix_loan_proposal_lender_status", "lender_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(length=32), nullable=False)
    status: Mapped[str] = mapped_column(String(length=16), default="pending", nullable=False)
    lender_type: Mapped[str] = mapped_column(String(length=16), nullable=False)
    borrower_id: Mapped[int] = mapped_column(Integer, nullable=False)
    borrower_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    borrower_avatar_url: Mapped[str] = mapped_column(
        String(length=2048), default="", nullable=False
    )
    lender_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lender_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    lender_avatar_url: Mapped[str] = mapped_column(String(length=2048), default="", nullable=False)
    creator_id: Mapped[int] = mapped_column(Integer, nullable=False)
    amount: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    monthly_rate_bps: Mapped[int] = mapped_column(
        Integer, default=DEFAULT_LOAN_MONTHLY_RATE_BPS, nullable=False
    )
    escrow_amount: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


class LoanContract(Base):
    """Active or closed long-term loan contract.

    Interest is simple and accrued lazily: `last_interest_accrued_at` marks the point interest has
    already been charged up to, and every read or payment path advances it a whole day at a time
    (`_loan_interest_delta`). Acceptance prepays `MIN_INTEREST_DAYS` and parks that timestamp in
    the FUTURE, which is what makes borrow-then-instantly-repay still cost the borrower; do not
    read it as "last touched". A contract closes only when principal and interest both reach zero.
    """

    __tablename__ = "loan_contract"
    __table_args__ = (
        Index("ix_loan_contract_borrower_status", "borrower_id", "status"),
        Index("ix_loan_contract_lender_status", "lender_id", "status"),
        Index("ix_loan_contract_lender_type_status", "lender_type", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    proposal_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lender_type: Mapped[str] = mapped_column(String(length=16), nullable=False)
    lender_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lender_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    lender_avatar_url: Mapped[str] = mapped_column(String(length=2048), default="", nullable=False)
    borrower_id: Mapped[int] = mapped_column(Integer, nullable=False)
    borrower_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    borrower_avatar_url: Mapped[str] = mapped_column(
        String(length=2048), default="", nullable=False
    )
    original_principal: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    principal_remaining: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    interest_due: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_interest_paid: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_principal_paid: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    monthly_rate_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(length=16), default="active", nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)
    last_interest_accrued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


class JackpotPool(Base):
    """Per-game cumulative jackpot shared across every table of that game.

    One row per game (keyed by `game_id`). Wager flows update
    `pool_balance` atomically while `total_contributed` /
    `total_claimed` accumulate gross in/out flows so the seeded
    on-the-house amount stays distinguishable from organic player
    contributions.

    Attributes:
        game_id: Stable game identifier (e.g. `"dragon_gate"`); primary key.
        pool_balance: Current spendable jackpot for the game.
        total_contributed: Lifetime gross amount that flowed into the pool
            (positive deltas from player losses + ante).
        total_claimed: Lifetime gross amount paid out from the pool
            (absolute value of negative deltas from player wins).
        seeded_amount: Lifetime on-the-house seed total; bookkeeping only,
            never decremented.
        generation: Incremented every time a seeded pool is depleted and
            replenished, so stale table snapshots cannot claim the next seed.
        updated_at: Taiwan-local timestamp of the last write.
    """

    __tablename__ = "jackpot_pool"

    game_id: Mapped[str] = mapped_column(String(length=32), primary_key=True)
    pool_balance: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_contributed: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_claimed: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    seeded_amount: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    generation: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


class CasinoLedger(Base):
    """Cumulative profit and loss for the casino system (cross-server).

    The casino is the dealer in Blackjack. Player wins flow out of this row,
    player losses flow in. There is no bot-account coupling: the bot is now a
    regular player at the table, and its `user_wallet` no longer doubles as
    the house ledger. `balance` may go negative when payouts exceed take-in;
    `total_earned` and `total_spent` accumulate gross flows so `/casino` can
    show direction of volume, not just net.

    Attributes:
        ledger_id: Stable identifier (e.g. `"casino"`); primary key.
        balance: Signed cumulative P&L.
        total_earned: Lifetime gross inflows (from player losses).
        total_spent: Lifetime gross outflows (to player wins).
        updated_at: Taiwan-local timestamp of the last write.
    """

    __tablename__ = "casino_ledger"

    ledger_id: Mapped[str] = mapped_column(String(length=32), primary_key=True)
    balance: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_earned: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_spent: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


CASINO_LEDGER_ID: Final[str] = "casino"


# On-the-house seed amount for each registered jackpot pool. The seed is bookkeeping only:
# nothing is decremented to fund it, neither a wallet nor the casino ledger, so /casino P&L stays
# unaffected by the donation. Seeded pools are also topped back up to this amount whenever they
# are drained.
_JACKPOT_SEEDS: Final[tuple[tuple[str, int], ...]] = (("dragon_gate", 1_000),)


def _jackpot_seed_amount(game_id: str) -> int:
    """Returns the configured seed amount for a jackpot game.

    A zero answer is also the test for "this pool is not seeded", which is what keeps the
    replenish path from inventing money for a game that never asked for a seed.

    Args:
        game_id (str): Game identifier (jackpot row primary key).

    Returns:
        The configured seed, or 0 for a game with no seed entry.
    """
    for seed_game_id, seed_amount in _JACKPOT_SEEDS:
        if seed_game_id == game_id:
            return seed_amount
    return 0


# Track which engine the schema has already been bootstrapped on. Storing
# the engine identity (not just a bool) means swapping `_engine` (e.g. tests
# pointing it at a temp file) automatically forces another schema check.
# SQLAlchemy's SQLite `create_all(checkfirst=True)` still has a check-then-create
# race under concurrent first use, so schema creation is serialized with
# loop-local locks.
_schema_ready_for: AsyncEngine | None = None
_schema_lock = LoopLocalLock()
_loan_accept_lock = LoopLocalLock()
type _TopNCacheKey = tuple[int, int | None, tuple[int, ...], bool]
type _TopLosersCacheKey = tuple[int, int, tuple[int, ...], bool, datetime]
_top_n_cache: dict[_TopNCacheKey, tuple[float, tuple[LeaderboardEntry, ...]]] = {}
_top_losers_cache: dict[_TopLosersCacheKey, tuple[float, tuple[LossLeaderboardEntry, ...]]] = {}


def invalidate_economy_leaderboard_cache() -> None:
    """Clears process-local leaderboard row caches.

    These are keyed on the query, so a write leaves them holding the wrong answer
    until they are cleared. The rendered board images are keyed on the rows
    themselves and expire on their own, which is why nothing here reaches into a
    renderer.
    """
    _top_n_cache.clear()
    _top_losers_cache.clear()


def _cached_top_n_rows(cache_key: _TopNCacheKey) -> list[LeaderboardEntry] | None:
    """Returns cached balance leaderboard rows when the short TTL is still valid.

    Evicts the entry on the way out when it has aged past the TTL, so a key nobody asks for again
    does not sit in the dict forever.

    Args:
        cache_key (_TopNCacheKey): Engine identity plus the query's own arguments.

    Returns:
        A fresh copy of the cached rows, or None when there is no live entry.
    """
    cached = _top_n_cache.get(cache_key)
    if cached is None:
        return None
    cached_at, rows = cached
    if monotonic() - cached_at > _ECONOMY_LEADERBOARD_CACHE_TTL_SECONDS:
        _top_n_cache.pop(cache_key, None)
        return None
    return list(rows)


def _cached_top_loser_rows(cache_key: _TopLosersCacheKey) -> list[LossLeaderboardEntry] | None:
    """Returns cached loss leaderboard rows when the short TTL is still valid.

    The Taipei day is part of the key, so a rollover cannot serve yesterday's losers even inside
    the TTL window.

    Args:
        cache_key (_TopLosersCacheKey): Engine identity, the query's arguments, and the Taipei day.

    Returns:
        A fresh copy of the cached rows, or None when there is no live entry.
    """
    cached = _top_losers_cache.get(cache_key)
    if cached is None:
        return None
    cached_at, rows = cached
    if monotonic() - cached_at > _ECONOMY_LEADERBOARD_CACHE_TTL_SECONDS:
        _top_losers_cache.pop(cache_key, None)
        return None
    return list(rows)


def _stored_integer_desc_order(column: Any) -> tuple[Any, ...]:  # noqa: ANN401 -- SQLAlchemy columns are generic expressions
    """Returns ORDER BY terms for descending numeric order over decimal text.

    A `StoredInteger` column sorts lexicographically in SQL, where "9" beats "10", and the
    registered UDF only yields a sign for a pair of values. So the numeric order is rebuilt out of
    five terms: sign first, then for positives longer text (more digits) before shorter and
    lexicographic within a length, and for negatives the mirror image. Doing it in SQL rather than
    in Python is what lets the caller apply `LIMIT` before any row is materialized.

    Args:
        column (Any): The `StoredInteger` column to order by.

    Returns:
        ORDER BY terms to splat into `order_by`, highest value first.
    """
    sign = func.discordbot_int_compare_text(column, _stored_int_to_text(value=0))
    positive_length = case((sign > 0, func.length(column)), else_=0)
    negative_length = case((sign < 0, func.length(column)), else_=0)
    positive_text = case((sign > 0, column), else_="")
    negative_text = case((sign < 0, column), else_="")
    return (
        desc(sign),
        desc(positive_length),
        desc(positive_text),
        negative_length.asc(),
        negative_text.asc(),
    )


def _current_schema_lock() -> asyncio.Lock:
    """Returns the schema bootstrap lock bound to the current event loop.

    Returns:
        The `asyncio.Lock` guarding `create_all` for the loop now running.
    """
    return _schema_lock.get()


def _current_loan_accept_lock() -> asyncio.Lock:
    """Returns the loan-approval lock bound to the current event loop.

    Approval reads central-bank capacity and then spends it, so two approvals must not interleave
    or both would see the same free credit and mint past it.

    Returns:
        The `asyncio.Lock` guarding loan acceptance for the loop now running.
    """
    return _loan_accept_lock.get()


async def _ensure_schema() -> None:
    """Bootstraps the economy schema, jackpot seeds, and casino ledger once per engine.

    Every public database call awaits this first, so there is no separate migration step and a
    fresh `data/database/economy.db` is created on first use. The connection listeners are
    re-installed on each call, since a test that swapped `_engine` produced one the import-time
    listener never reached. The seed inserts are `ON CONFLICT DO NOTHING`, which is what makes the
    whole body idempotent against an existing file.
    """
    global _schema_ready_for  # noqa: PLW0603 -- module-level cache by engine identity
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    if _schema_ready_for is _engine:
        return
    async with _current_schema_lock():
        if _schema_ready_for is _engine:
            return
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            for seed_game_id, seed_amount in _JACKPOT_SEEDS:
                await conn.execute(
                    statement=insert(JackpotPool)
                    .values(
                        game_id=seed_game_id,
                        pool_balance=_stored_int_to_text(value=seed_amount),
                        total_contributed="0",
                        total_claimed="0",
                        seeded_amount=_stored_int_to_text(value=seed_amount),
                        generation=0,
                        updated_at=_database_now(),
                    )
                    .on_conflict_do_nothing(index_elements=["game_id"])
                )
            await conn.execute(
                statement=insert(CasinoLedger)
                .values(
                    ledger_id=CASINO_LEDGER_ID,
                    balance="0",
                    total_earned="0",
                    total_spent="0",
                    updated_at=_database_now(),
                )
                .on_conflict_do_nothing(index_elements=["ledger_id"])
            )
        _schema_ready_for = _engine


def open_session() -> AsyncSession:
    """Creates an async session bound to the current economy database engine.

    Reads `_engine` at call time rather than through a cached factory, so a test that monkeypatches
    it onto a `tmp_path` is picked up by the very next call. `expire_on_commit=False` keeps ORM
    attributes readable after the commit, which the loan paths rely on when they project a row into
    a view. The schema is NOT ensured here; the public entry points do that.

    Returns:
        An `AsyncSession` using the current module-level `_engine`.
    """
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    return AsyncSession(bind=_engine, expire_on_commit=False)


def checkin_reward(streak: int, is_vip: bool) -> int:
    """Returns the gross check-in payout for a streak day.

    The reward formula is `BASE * (1 + (streak - 1) * 0.5)` where `streak`
    is the 1..`CHECKIN_STREAK_CYCLE` day in the cycle. VIP doubles the base
    before the streak bonus.

    Pure arithmetic with no database access, so `cogs/economy/embeds.py` calls it to show what a
    streak day or the VIP perk is worth before anyone commits to either.

    Args:
        streak (int): Streak counter for this check-in (1..`CHECKIN_STREAK_CYCLE`).
        is_vip (bool): VIP status of the account at check-in time.

    Returns:
        Integer reward amount.
    """
    base = BASE_CHECKIN_REWARD_AMOUNT * (2 if is_vip else 1)
    multiplier = 1.0 + (streak - 1) * 0.5
    return int(base * multiplier)


def monthly_rate_percent_to_bps(monthly_rate_percent: float) -> int:
    """Converts a user-facing monthly percent into basis points.

    Clamps into the allowed band rather than rejecting, so an out-of-range slash option becomes the
    nearest legal rate instead of a failed command.

    Args:
        monthly_rate_percent (float): Rate as the user typed it, in percent.

    Returns:
        The rate in basis points, within `MIN_LOAN_MONTHLY_RATE_BPS`..`MAX_LOAN_MONTHLY_RATE_BPS`.
    """
    return max(
        MIN_LOAN_MONTHLY_RATE_BPS,
        min(MAX_LOAN_MONTHLY_RATE_BPS, round(monthly_rate_percent * 100)),
    )


def monthly_rate_bps_to_percent(monthly_rate_bps: int) -> float:
    """Converts stored monthly basis points into a display percent.

    Args:
        monthly_rate_bps (int): Rate in basis points as persisted on the proposal or contract.

    Returns:
        The same rate in percent, for display only.
    """
    return monthly_rate_bps / 100


def apply_vip_blackjack_bonus(delta: int, is_vip: bool) -> int:
    """Applies the VIP 1.2x payout multiplier on a winning player delta.

    The bonus only fires on positive deltas (wins). Pushes and losses pass
    through unchanged so VIP never softens a loss.

    Integer arithmetic, floored, so the perk never mints a fraction. It lives here rather than in
    the games cog because `cogs/economy/embeds.py` also uses it to show a prospective VIP what the
    perk is worth; `cogs/games/settlement.py` is what decides how much of the boost the casino
    ledger absorbs.

    Args:
        delta (int): Pre-bonus player delta for the round.
        is_vip (bool): VIP status of the account at settlement time.

    Returns:
        Post-bonus player delta.
    """
    if not is_vip or delta <= 0:
        return delta
    return delta * _VIP_WIN_MULTIPLIER_NUM // _VIP_WIN_MULTIPLIER_DEN


async def _upsert_user_metadata_in_session(
    session: AsyncSession, user_id: int, name: str, avatar_url: str, now: datetime
) -> None:
    """Creates or refreshes the user identity row without touching wallet state.

    An empty `name` or `avatar_url` is treated as "unknown", not as a value: the UPSERT's update
    branch omits the column entirely so a caller with no Discord objects in hand cannot wipe a
    known name or avatar. On insert the name falls back to the numeric id so the row is never
    blank.

    Args:
        session (AsyncSession): Active session; the write is left uncommitted.
        user_id (int): Discord user ID to create or refresh.
        name (str): Last-seen Discord username, or empty to leave the stored one.
        avatar_url (str): Last-seen Discord avatar URL, or empty to leave the stored one.
        now (datetime): `_database_now()` value pinned for this transaction.
    """
    effective_name = name or str(user_id)
    stmt = insert(UserAccount).values(
        user_id=user_id,
        name=effective_name,
        avatar_url=avatar_url,
        updated_at=now,
        is_vip=False,
        last_checkin_at=None,
        checkin_streak=0,
        is_admin=False,
        is_central_banker=False,
        hide_from_leaderboard=False,
    )
    set_: dict[str, Any] = {"updated_at": now}
    if name:
        set_["name"] = effective_name
    if avatar_url:
        set_["avatar_url"] = avatar_url
    await session.execute(
        statement=stmt.on_conflict_do_update(index_elements=["user_id"], set_=set_)
    )


def _build_credit_upsert(
    user_id: int, name: str, amount: int, now: datetime
) -> ReturningInsert[tuple[int]]:
    """Builds the UPSERT that credits `amount` points into `user_wallet`.

    Builds only; the caller executes it inside its own transaction. Because the increment is
    expressed in SQL (`balance + amount`) rather than read into Python first, two concurrent
    credits cannot lose each other's update and no retry loop is needed. Caller guarantees
    `amount > 0` and refreshes `user_account` metadata separately.

    Args:
        user_id (int): Discord user ID to credit.
        name (str): Last-seen Discord username, or empty to leave the stored one.
        amount (int): Points to add; also added to `total_earned`.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        A SQLAlchemy `Insert` with `on_conflict_do_update` and `returning(balance)`.
    """
    effective_name = name or str(user_id)
    stmt = insert(UserWallet).values(
        user_id=user_id,
        name=effective_name,
        balance=amount,
        total_earned=amount,
        total_spent=0,
        updated_at=now,
    )
    set_: dict[str, Any] = {
        "balance": UserWallet.balance + amount,
        "total_earned": UserWallet.total_earned + amount,
        "updated_at": now,
    }
    if name:
        set_["name"] = effective_name
    return stmt.on_conflict_do_update(index_elements=["user_id"], set_=set_).returning(
        UserWallet.balance
    )


def _build_signed_delta_upsert(
    user_id: int, name: str, delta: int, now: datetime
) -> ReturningInsert[tuple[int]]:
    """Builds the UPSERT applying a signed `delta` with NO clamp on wallet balance.

    The one wallet write allowed to leave a balance below zero, reached only through
    `adjust_balance(allow_negative=True)`; every gameplay debit goes through the clamped path
    instead. `total_earned` / `total_spent` still take the gross halves of the delta, so the
    `balance == total_earned - total_spent` invariant survives a negative balance.

    Args:
        user_id (int): Discord user ID whose wallet the delta applies to.
        name (str): Last-seen Discord username, or empty to leave the stored one.
        delta (int): Signed amount to apply, with no floor at zero.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        A SQLAlchemy `Insert` with `on_conflict_do_update` and `returning(balance)`.
    """
    effective_name = name or str(user_id)
    initial_earned = max(delta, 0)
    initial_spent = max(-delta, 0)
    stmt = insert(UserWallet).values(
        user_id=user_id,
        name=effective_name,
        balance=delta,
        total_earned=initial_earned,
        total_spent=initial_spent,
        updated_at=now,
    )
    set_: dict[str, Any] = {
        "balance": UserWallet.balance + delta,
        "total_earned": UserWallet.total_earned + initial_earned,
        "total_spent": UserWallet.total_spent + initial_spent,
        "updated_at": now,
    }
    if name:
        set_["name"] = effective_name
    return stmt.on_conflict_do_update(index_elements=["user_id"], set_=set_).returning(
        UserWallet.balance
    )


async def _apply_daily_casino_delta_in_session(
    session: AsyncSession, user_id: int, name: str, delta: int, now: datetime
) -> None:
    """Accumulates current-day gross casino counters in `casino_account`.

    Loss and win are tracked separately and both as POSITIVE magnitudes, so `/loss_leaderboard`
    reads gross losses that a later win never offsets. The day rollover is lazy and happens inside
    the same UPSERT: a stored `day_started_at` that is not today makes each counter take the new
    delta outright instead of adding to it, so a stale row resets itself at the first settlement
    after Taipei midnight and no scheduled job is needed. A zero delta is skipped entirely, which
    is what keeps a push off the loss board.

    Args:
        session (AsyncSession): Active session; the write is left uncommitted.
        user_id (int): Discord user ID the counters belong to.
        name (str): Last-seen Discord username, falling back to the numeric id.
        delta (int): Signed player-side casino delta actually applied.
        now (datetime): `_database_now()` value pinned for this transaction.
    """
    if delta == 0:
        return
    today_midnight = _taipei_midnight(now=now)
    loss_delta = max(-delta, 0)
    win_delta = max(delta, 0)
    loss_delta_text = str(loss_delta)
    win_delta_text = str(win_delta)
    delta_text = str(delta)
    same_day = CasinoAccount.day_started_at == today_midnight
    await session.execute(
        statement=insert(CasinoAccount)
        .values(
            user_id=user_id,
            name=name or str(user_id),
            day_started_at=today_midnight,
            daily_loss=loss_delta_text,
            daily_win=win_delta_text,
            daily_net=delta_text,
            updated_at=now,
        )
        .on_conflict_do_update(
            index_elements=["user_id"],
            set_={
                "name": name or str(user_id),
                "day_started_at": today_midnight,
                "daily_loss": case(
                    (
                        same_day,
                        func.discordbot_int_add_text(CasinoAccount.daily_loss, loss_delta_text),
                    ),
                    else_=loss_delta_text,
                ),
                "daily_win": case(
                    (
                        same_day,
                        func.discordbot_int_add_text(CasinoAccount.daily_win, win_delta_text),
                    ),
                    else_=win_delta_text,
                ),
                "daily_net": case(
                    (same_day, func.discordbot_int_add_text(CasinoAccount.daily_net, delta_text)),
                    else_=delta_text,
                ),
                "updated_at": now,
            },
        )
    )
    invalidate_economy_leaderboard_cache()


async def _credit_with_repayment_in_session(  # noqa: PLR0913 -- session helper keeps income writes atomic
    session: AsyncSession, user_id: int, name: str, avatar_url: str, amount: int, now: datetime
) -> CreditResult:
    """Credits income inside the caller's transaction.

    Long-term loans are explicit repayment actions now, so passive income does
    not auto-repay debt. The public function name is preserved because message
    and chat reward callers are intentionally routed through one income facade.
    Caller must guarantee `amount > 0`.

    Args:
        session (AsyncSession): Active session; the write is left uncommitted.
        user_id (int): Discord user ID receiving the credit.
        name (str): Last-seen Discord username, or empty to leave the stored one.
        avatar_url (str): Last-seen Discord avatar URL, or empty to leave the stored one.
        amount (int): Gross income amount; must be positive.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        The post-credit balance, with both repayment fields zero.
    """
    await _upsert_user_metadata_in_session(
        session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
    )
    result = await session.execute(
        statement=_build_credit_upsert(user_id=user_id, name=name, amount=amount, now=now)
    )
    new_balance = result.scalar_one()
    invalidate_economy_leaderboard_cache()
    return CreditResult(
        new_balance=new_balance, credited_amount=amount, principal_repaid=0, remaining_debt=0
    )


async def _apply_clamped_delta_in_session(  # noqa: PLR0913 -- session helper needs identity and delta state
    session: AsyncSession, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
) -> tuple[int, int]:
    """Applies a clamped signed delta and returns the balance plus applied delta.

    The clamp has to be computed in Python (the new balance depends on the old one), so this is a
    read-then-conditional-write loop: the observed balance is pinned in the UPDATE predicate, and a
    writer that loses the race matches zero rows and retries against the fresh value. Without that
    pin two concurrent clamped debits would both compute their applied delta from the same stale
    balance and over-collect. A negative delta against a missing row is a no-op so manual clamp
    operations do not create zero-balance accounts.

    Args:
        session (AsyncSession): Active session; the writes are left uncommitted.
        user_id (int): Discord user ID whose wallet the delta applies to.
        name (str): Last-seen Discord username, or empty to leave the stored one.
        avatar_url (str): Last-seen Discord avatar URL, or empty to leave the stored one.
        delta (int): Signed amount to apply; a debit stops at a zero balance.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        `(new_balance, applied_delta)`, where the applied delta is smaller in magnitude than
        `delta` whenever a debit hit the zero floor.

    Raises:
        RuntimeError: `_CLAMPED_DELTA_MAX_RETRIES` conditional writes all lost their race.
    """
    if delta == 0:
        read_result = await session.execute(
            statement=select(UserWallet.balance).where(UserWallet.user_id == user_id)
        )
        return read_result.scalar_one_or_none() or 0, 0

    for _ in range(_CLAMPED_DELTA_MAX_RETRIES):
        read_result = await session.execute(
            statement=select(UserWallet.balance).where(UserWallet.user_id == user_id)
        )
        current_balance = read_result.scalar_one_or_none()

        if current_balance is None:
            if delta < 0:
                return 0, 0
            insert_result = await _try_insert_clamped_positive_delta_in_session(
                session=session, user_id=user_id, name=name, delta=delta, now=now
            )
            if insert_result is not None:
                await _upsert_user_metadata_in_session(
                    session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
                )
                return insert_result
            continue

        update_result = await _try_update_clamped_delta_in_session(
            session=session,
            user_id=user_id,
            name=name,
            current_balance=current_balance,
            delta=delta,
            now=now,
        )
        if update_result is not None:
            await _upsert_user_metadata_in_session(
                session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
            )
            return update_result

    raise RuntimeError(f"apply_clamped_delta retry budget exhausted for user_id={user_id}")


async def _try_insert_clamped_positive_delta_in_session(
    session: AsyncSession, user_id: int, name: str, delta: int, now: datetime
) -> tuple[int, int] | None:
    """Attempts to create a missing wallet row for a positive clamped delta.

    `ON CONFLICT DO NOTHING` is what makes this safe to lose: another coroutine that inserted the
    row first leaves this one with no returned balance, and the caller retries against the row now
    visible instead of raising `IntegrityError`.

    Args:
        session (AsyncSession): Active session; the write is left uncommitted.
        user_id (int): Discord user ID to create the wallet for.
        name (str): Last-seen Discord username, falling back to the numeric id.
        delta (int): Positive amount to seed the wallet with.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        `(new_balance, delta)` when this call created the row, or None when it lost the race.
    """
    insert_stmt = (
        insert(UserWallet)
        .values(
            user_id=user_id,
            name=name or str(user_id),
            balance=delta,
            total_earned=delta,
            total_spent=0,
            updated_at=now,
        )
        .on_conflict_do_nothing(index_elements=["user_id"])
        .returning(UserWallet.balance)
    )
    insert_result = await session.execute(statement=insert_stmt)
    inserted_balance = insert_result.scalar_one_or_none()
    if inserted_balance is None:
        return None
    invalidate_economy_leaderboard_cache()
    return inserted_balance, delta


async def _try_update_clamped_delta_in_session(  # noqa: PLR0913 -- conditional write needs observed row state
    session: AsyncSession, user_id: int, name: str, current_balance: int, delta: int, now: datetime
) -> tuple[int, int] | None:
    """Attempts one conditional clamped update against an existing wallet row.

    `current_balance` is both the input to the clamp and the WHERE predicate, so the write lands
    only if nothing moved the balance in between. An already-negative balance (an admin adjustment
    left it there) absorbs no further debit rather than being driven deeper. Only the ACTUALLY
    applied amount reaches `total_earned` / `total_spent`, which is what keeps the accounting
    invariant true through a clamp.

    Args:
        session (AsyncSession): Active session; the write is left uncommitted.
        user_id (int): Discord user ID whose wallet the delta applies to.
        name (str): Last-seen Discord username, or empty to leave the stored one.
        current_balance (int): Balance observed by the caller's SELECT, pinned in the predicate.
        delta (int): Signed amount to apply; a debit stops at a zero balance.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        `(new_balance, applied_delta)`, or None when the predicate matched no row and the caller
        should re-read and retry.
    """
    if delta < 0 and current_balance <= 0:
        new_balance = current_balance
    elif delta < 0:
        new_balance = max(current_balance + delta, 0)
    else:
        new_balance = current_balance + delta
    applied = new_balance - current_balance
    update_values: dict[str, Any] = {"balance": new_balance, "updated_at": now}
    if name:
        update_values["name"] = name
    if applied > 0:
        update_values["total_earned"] = UserWallet.total_earned + applied
    elif applied < 0:
        update_values["total_spent"] = UserWallet.total_spent - applied

    update_result = await session.execute(
        statement=update(UserWallet)
        .where(UserWallet.user_id == user_id, UserWallet.balance == current_balance)
        .values(**update_values)
        .returning(UserWallet.balance)
    )
    if update_result.scalar_one_or_none() is None:
        return None
    if applied != 0:
        invalidate_economy_leaderboard_cache()
    return new_balance, applied


async def _apply_signed_delta_in_session(  # noqa: PLR0913 -- session helper needs identity and signed delta
    session: AsyncSession, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
) -> int:
    """Applies a signed delta to a wallet without clamping.

    Reached only from `adjust_balance(allow_negative=True)`, the admin escape hatch that is allowed
    to leave a balance below zero. Player-side losses use the clamped path instead. Needs no retry
    loop: the whole delta is expressed in SQL, so there is no observed value to lose a race on.

    Args:
        session (AsyncSession): Active session; the writes are left uncommitted.
        user_id (int): Discord user ID whose wallet the delta applies to.
        name (str): Last-seen Discord username, or empty to leave the stored one.
        avatar_url (str): Last-seen Discord avatar URL, or empty to leave the stored one.
        delta (int): Signed amount to apply, with no floor at zero.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        Wallet balance after the write.
    """
    await _upsert_user_metadata_in_session(
        session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
    )
    stmt = _build_signed_delta_upsert(user_id=user_id, name=name, delta=delta, now=now)
    result = await session.execute(statement=stmt)
    new_balance = result.scalar_one()
    if delta != 0:
        invalidate_economy_leaderboard_cache()
    return new_balance


async def _apply_casino_ledger_delta_in_session(
    session: AsyncSession, delta: int, now: datetime
) -> int:
    """Applies a signed delta to the global casino ledger row (no clamp).

    The casino is allowed to run cumulative negative P&L when payouts exceed
    take-in. `total_earned` / `total_spent` accumulate gross flows.

    This is the house side of a settlement and it is NOT the bot's wallet: the bot plays as an
    ordinary user, so `/pocat` and `/casino` read different rows. The write is a pure SQL
    increment, so it rides the caller's player transaction with no retry loop of its own.

    Args:
        session (AsyncSession): Active session; the write is left uncommitted.
        delta (int): Signed change to apply to the casino balance.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        Casino ledger balance after the write.
    """
    initial_earned = max(delta, 0)
    initial_spent = max(-delta, 0)
    stmt = (
        insert(CasinoLedger)
        .values(
            ledger_id=CASINO_LEDGER_ID,
            balance=delta,
            total_earned=initial_earned,
            total_spent=initial_spent,
            updated_at=now,
        )
        .on_conflict_do_update(
            index_elements=["ledger_id"],
            set_={
                "balance": CasinoLedger.balance + delta,
                "total_earned": CasinoLedger.total_earned + initial_earned,
                "total_spent": CasinoLedger.total_spent + initial_spent,
                "updated_at": now,
            },
        )
        .returning(CasinoLedger.balance)
    )
    result = await session.execute(statement=stmt)
    return result.scalar_one()


async def _read_casino_ledger_balance_in_session(session: AsyncSession) -> int:
    """Reads the current casino ledger balance, returning 0 when missing.

    `_ensure_schema` seeds the row, so a miss only happens against a database created outside this
    module; 0 is returned rather than raising so a settlement that moved no house money still
    answers with a balance.

    Args:
        session (AsyncSession): Active session to read through.

    Returns:
        The casino ledger balance, or 0 when the row does not exist.
    """
    result = await session.execute(
        statement=select(CasinoLedger.balance).where(CasinoLedger.ledger_id == CASINO_LEDGER_ID)
    )
    return result.scalar_one_or_none() or 0


async def _rollback_sessions(*sessions: AsyncSession) -> None:
    """Rolls back sessions without masking the original settlement exception.

    Called from an `except` block that is about to re-raise, so a rollback that itself fails is
    logged and swallowed: losing the real settlement error to a cleanup error would leave nothing
    to debug from.

    Args:
        *sessions (AsyncSession): Sessions to roll back, each independently.
    """
    for session in sessions:
        try:
            await session.rollback()
        except Exception:
            logfire.warn("Failed to roll back settlement session", _exc_info=True)


async def get_casino_ledger() -> CasinoLedgerSnapshot:
    """Returns the cumulative casino system ledger snapshot.

    What `/casino` shows. A missing row answers as all-zero stamped with the current time rather
    than None, so the command has nothing to special-case before the first settlement.

    Returns:
        The house-side P&L: signed balance plus the two lifetime gross flows.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(
                CasinoLedger.balance,
                CasinoLedger.total_earned,
                CasinoLedger.total_spent,
                CasinoLedger.updated_at,
            ).where(CasinoLedger.ledger_id == CASINO_LEDGER_ID)
        )
        row = result.one_or_none()
    if row is None:
        return CasinoLedgerSnapshot(
            balance=0, total_earned=0, total_spent=0, updated_at=_database_now()
        )
    balance, total_earned, total_spent, updated_at = row
    return CasinoLedgerSnapshot(
        balance=balance, total_earned=total_earned, total_spent=total_spent, updated_at=updated_at
    )


async def get_casino_daily_stats(user_id: int) -> CasinoDailyStats:
    """Returns the current-day casino loss/win/net for one user.

    Returns all-zero when no row exists or when the stored counters are from a
    previous Taipei day (the next casino settlement will reset them anyway).

    Read-only: it does not perform the day rollover it detects, so a stale row stays on disk until
    the user's next settlement rewrites it.

    Args:
        user_id (int): Discord user ID to look up.

    Returns:
        Today's gross loss, gross win and signed net for the user.
    """
    await _ensure_schema()
    today_midnight = _taipei_midnight(now=_database_now())
    async with open_session() as session:
        result = await session.execute(
            statement=select(
                CasinoAccount.daily_loss,
                CasinoAccount.daily_win,
                CasinoAccount.daily_net,
                CasinoAccount.day_started_at,
            ).where(CasinoAccount.user_id == user_id)
        )
        row = result.one_or_none()
    if row is None:
        return CasinoDailyStats(daily_loss=0, daily_win=0, daily_net=0)
    daily_loss, daily_win, daily_net, day_started_at = row
    if day_started_at is None or _as_taipei(dt=day_started_at) != today_midnight:
        return CasinoDailyStats(daily_loss=0, daily_win=0, daily_net=0)
    return CasinoDailyStats(daily_loss=daily_loss, daily_win=daily_win, daily_net=daily_net)


async def _apply_player_delta_in_session(  # noqa: PLR0913 -- player settlement needs identity and audit metadata
    session: AsyncSession, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
) -> tuple[int, int]:
    """Applies a casino player delta and returns the balance plus actual delta.

    A win goes through the shared income facade; a loss goes through the clamped path so it can
    never drive a player negative. Either way the daily `casino_account` counters take the amount
    that was ACTUALLY applied, so a partially collected loss is reported at its collected size. A
    zero delta (a push) only reads the balance and is deliberately kept off the loss board.

    Args:
        session (AsyncSession): Active session; the writes are left uncommitted.
        user_id (int): Discord user ID for the player.
        name (str): Last-seen Discord username, or empty to leave the stored one.
        avatar_url (str): Last-seen Discord avatar URL, or empty to leave the stored one.
        delta (int): Signed player-side change for the round.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        `(new_balance, applied_delta)` after the wallet and counter writes.
    """
    if delta > 0:
        credit_result = await _credit_with_repayment_in_session(
            session=session,
            user_id=user_id,
            name=name,
            avatar_url=avatar_url,
            amount=delta,
            now=now,
        )
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=user_id, name=name, delta=delta, now=now
        )
        return credit_result.new_balance, delta
    if delta < 0:
        new_balance, applied_delta = await _apply_clamped_delta_in_session(
            session=session,
            user_id=user_id,
            name=name,
            avatar_url=avatar_url,
            delta=delta,
            now=now,
        )
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=user_id, name=name, delta=applied_delta, now=now
        )
        return new_balance, applied_delta
    read_result = await session.execute(
        statement=select(UserWallet.balance).where(UserWallet.user_id == user_id)
    )
    return read_result.scalar_one_or_none() or 0, 0


async def _apply_jackpot_player_delta_in_session(  # noqa: PLR0913 -- jackpot settlement needs identity and audit metadata
    session: AsyncSession, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
) -> tuple[int, int]:
    """Applies a jackpot player delta and returns the balance plus applied delta.

    Positive deltas keep the existing casino payout path and count as fully
    applied. Negative deltas clamp at zero so Dragon Gate losses cannot drive
    the player account negative; the returned delta is the actual debit.

    The body is identical to `_apply_player_delta_in_session` today; the difference is in who
    calls it, since the jackpot batch is the caller that compares the applied delta against what it
    asked for and rejects the batch when a required full debit fell short.

    Args:
        session (AsyncSession): Active session; the writes are left uncommitted.
        user_id (int): Discord user ID for the player.
        name (str): Last-seen Discord username, or empty to leave the stored one.
        avatar_url (str): Last-seen Discord avatar URL, or empty to leave the stored one.
        delta (int): Signed player-side change, already capped to the pool by the caller.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        `(new_balance, applied_delta)` after the wallet and counter writes.
    """
    if delta > 0:
        credit_result = await _credit_with_repayment_in_session(
            session=session,
            user_id=user_id,
            name=name,
            avatar_url=avatar_url,
            amount=delta,
            now=now,
        )
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=user_id, name=name, delta=delta, now=now
        )
        return credit_result.new_balance, delta
    if delta < 0:
        new_balance, applied_delta = await _apply_clamped_delta_in_session(
            session=session,
            user_id=user_id,
            name=name,
            avatar_url=avatar_url,
            delta=delta,
            now=now,
        )
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=user_id, name=name, delta=applied_delta, now=now
        )
        return new_balance, applied_delta
    read_result = await session.execute(
        statement=select(UserWallet.balance).where(UserWallet.user_id == user_id)
    )
    return read_result.scalar_one_or_none() or 0, 0


async def credit_with_repayment(
    user_id: int, name: str, amount: int, avatar_url: str = ""
) -> CreditResult:
    """Credits `amount` to the user through the shared income path.

    Long-term loans must be repaid with explicit repayment or collection
    commands. Message, chat, and casino payout income therefore lands fully in
    balance and only increases `total_earned`.

    This is the single income facade: the per-message reward, `/checkin`, casino payouts and
    fishing payouts all land here, which is what makes "how does money enter a wallet" one
    question with one answer. A non-positive amount is a no-op that still reports the balance.

    Args:
        user_id (int): Discord user ID receiving the credit.
        name (str): Last-seen Discord username to store on the account.
        amount (int): Gross income amount; must be positive for the repayment
            path to run.
        avatar_url (str): Last-seen Discord avatar URL to store when available.

    Returns:
        Outcome capturing post-credit balance. Repayment fields are zero
        because passive income no longer auto-repays long-term loans.
    """
    await _ensure_schema()
    if amount <= 0:
        return CreditResult(
            new_balance=await get_balance(user_id=user_id),
            credited_amount=0,
            principal_repaid=0,
            remaining_debt=0,
        )
    now = _database_now()
    async with open_session() as session:
        result = await _credit_with_repayment_in_session(
            session=session,
            user_id=user_id,
            name=name,
            avatar_url=avatar_url,
            amount=amount,
            now=now,
        )
        await session.commit()
        invalidate_economy_leaderboard_cache()
        return result


async def adjust_balance(
    user_id: int, name: str, delta: int, allow_negative: bool = False, avatar_url: str = ""
) -> BalanceAdjustmentResult:
    """Applies an explicit manual balance adjustment.

    This is the public maintenance API for scripts and admin tooling. It does
    not touch loan contracts or daily casino counters, so leaderboards and
    house P&L remain clean.

    It is the only route to an unclamped wallet write, and a settlement helper must never be
    borrowed for an admin tweak: those move the casino ledger and the daily counters too.

    Args:
        user_id (int): Discord user ID whose balance should be adjusted.
        name (str): Last-seen Discord username to store on the account.
        delta (int): Signed amount to apply.
        allow_negative (bool): Whether the resulting balance may go below zero.
        avatar_url (str): Last-seen Discord avatar URL to store when available.

    Returns:
        The post-adjustment balance and the applied delta after any clamp.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        if delta == 0:
            result = await session.execute(
                statement=select(UserWallet.balance).where(UserWallet.user_id == user_id)
            )
            new_balance = result.scalar_one_or_none() or 0
            return BalanceAdjustmentResult(new_balance=new_balance, applied_delta=0)
        if allow_negative:
            new_balance = await _apply_signed_delta_in_session(
                session=session,
                user_id=user_id,
                name=name,
                avatar_url=avatar_url,
                delta=delta,
                now=now,
            )
            applied_delta = delta
        else:
            new_balance, applied_delta = await _apply_clamped_delta_in_session(
                session=session,
                user_id=user_id,
                name=name,
                avatar_url=avatar_url,
                delta=delta,
                now=now,
            )
        await session.commit()
        invalidate_economy_leaderboard_cache()
        return BalanceAdjustmentResult(new_balance=new_balance, applied_delta=applied_delta)


async def apply_ordered_wallet_deltas(
    user_id: int, name: str, deltas: Sequence[WalletDeltaLeg], avatar_url: str = ""
) -> OrderedWalletDeltaResult | None:
    """Applies ordered full-debit wallet deltas without netting.

    This helper is for non-casino domains that need gross wallet accounting but
    must reject insufficient funds instead of clamping a debit. Positive legs
    increment `total_earned` and negative legs increment `total_spent` in
    the order supplied by the caller. The transaction rolls back if any debit
    cannot be applied in full.

    Order matters and the legs are never netted: a debit-then-credit pair moves both lifetime
    totals by their gross amounts, which is what lets stock and fishing show a real spend and a
    real payout rather than one difference. Because the legs run in sequence, an ordering that
    debits before it credits can fail on funds the same call was about to add.

    Args:
        user_id (int): Discord user ID whose wallet should be updated.
        name (str): Last-seen Discord username to store on the wallet row.
        deltas (Sequence[WalletDeltaLeg]): Ordered signed wallet legs.
        avatar_url (str): Last-seen Discord avatar URL to store when available.

    Returns:
        The post-leg balance and applied deltas, or `None` when a full debit
        cannot be covered.
    """
    await _ensure_schema()
    now = _database_now()
    applied: list[int] = []
    async with open_session() as session:
        await _upsert_user_metadata_in_session(
            session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
        )
        balance = await _apply_ordered_wallet_deltas_in_session(
            session=session, user_id=user_id, name=name, deltas=deltas, now=now, applied=applied
        )
        if balance is None:
            await session.rollback()
            return None
        await session.commit()
        invalidate_economy_leaderboard_cache()
        return OrderedWalletDeltaResult(new_balance=balance, applied_deltas=tuple(applied))


async def _apply_ordered_wallet_deltas_in_session(  # noqa: PLR0913 -- session helper carries identity and output accumulator
    session: AsyncSession,
    user_id: int,
    name: str,
    deltas: Sequence[WalletDeltaLeg],
    now: datetime,
    applied: list[int],
) -> int | None:
    """Applies ordered wallet legs inside the caller's economy transaction.

    Each debit is one conditional `UPDATE ... WHERE balance >= debit`, so an insufficient balance
    is detected by the write itself rather than by a prior read that could go stale. Returning None
    abandons the run partway through; the caller's rollback is what makes the batch all-or-nothing.
    `applied` is an output parameter, filled in leg order including explicit zeros, so a caller
    can line the result up with the legs it passed.

    Args:
        session (AsyncSession): Active session; the writes are left uncommitted.
        user_id (int): Discord user ID whose wallet the legs apply to.
        name (str): Last-seen Discord username, falling back to the numeric id.
        deltas (Sequence[WalletDeltaLeg]): Ordered signed wallet legs.
        now (datetime): `_database_now()` value pinned for this transaction.
        applied (list[int]): Accumulator the applied delta of each leg is appended to.

    Returns:
        The balance after the last leg, or None when a debit could not be covered in full.
    """
    balance_result = await session.execute(
        statement=select(UserWallet.balance).where(UserWallet.user_id == user_id)
    )
    balance = balance_result.scalar_one_or_none() or 0
    effective_name = name or str(user_id)
    for leg in deltas:
        delta = leg.delta
        if delta == 0:
            applied.append(0)
            continue
        if delta > 0:
            credit_result = await session.execute(
                statement=_build_credit_upsert(
                    user_id=user_id, name=effective_name, amount=delta, now=now
                )
            )
            balance = credit_result.scalar_one()
            applied.append(delta)
            continue
        debit = -delta
        debit_result = await session.execute(
            statement=update(UserWallet)
            .where(UserWallet.user_id == user_id, UserWallet.balance >= debit)
            .values(
                balance=UserWallet.balance - debit,
                total_spent=UserWallet.total_spent + debit,
                name=effective_name,
                updated_at=now,
            )
            .returning(UserWallet.balance)
        )
        new_balance = debit_result.scalar_one_or_none()
        if new_balance is None:
            return None
        balance = new_balance
        applied.append(delta)
    if any(delta != 0 for delta in applied):
        invalidate_economy_leaderboard_cache()
    return balance


async def apply_round_settlement(
    player_id: int,
    player_account_name: str,
    player_delta: int,
    casino_delta: int,
    player_avatar_url: str = "",
) -> RoundSettlementResult:
    """Applies a finished round's net delta and mirrors casino P&L.

    Positive player deltas go through the shared income path. Negative player
    deltas clamp at zero; when a loss cannot be fully collected, the casino
    ledger only records the actual collected debit. The player write and the
    casino mirror live in the same `data/database/economy.db` file and commit
    as one atomic transaction.

    A loss the player could not fully cover shrinks the house side to match: the caller asks for
    the full mirror, and this narrows `casino_delta` to what was actually collected so the ledger
    never books income nobody paid. The two halves are one transaction, so a failure anywhere
    leaves neither written.

    Args:
        player_id (int): Discord user ID for the player account.
        player_account_name (str): Account name to store for the player.
        player_delta (int): Signed net change for the player. Losses are clamped at
            zero and may apply less than the requested debit.
        casino_delta (int): Signed change to apply to the casino ledger balance.
        player_avatar_url (str): Last-seen Discord avatar URL for the player.

    Returns:
        A `RoundSettlementResult` with the post-write player and casino balances.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        try:
            player_balance, applied_player_delta = await _apply_player_delta_in_session(
                session=session,
                user_id=player_id,
                name=player_account_name,
                avatar_url=player_avatar_url,
                delta=player_delta,
                now=now,
            )

            casino_delta_to_apply = casino_delta
            if player_delta < 0 and casino_delta > 0:
                casino_delta_to_apply = min(casino_delta, max(-applied_player_delta, 0))

            if casino_delta_to_apply == 0:
                casino_balance = await _read_casino_ledger_balance_in_session(session=session)
            else:
                casino_balance = await _apply_casino_ledger_delta_in_session(
                    session=session, delta=casino_delta_to_apply, now=now
                )
            await session.commit()
        except Exception:
            await _rollback_sessions(session)
            raise
    invalidate_economy_leaderboard_cache()
    return RoundSettlementResult(player_balance=player_balance, casino_balance=casino_balance)


async def apply_blackjack_settlement(
    player_id: int,
    player_account_name: str,
    player_delta: int,
    casino_delta: int,
    player_avatar_url: str = "",
) -> RoundSettlementResult:
    """Applies Blackjack player payout and casino ledger deltas.

    Blackjack can include system-funded bonuses (e.g. five-card 21) that credit
    the player and count as casino payout but must not move the `/casino`
    ledger. The caller passes `casino_delta` explicitly so the bonus stays
    excluded.

    A named alias over `apply_round_settlement` with no behavior of its own; it exists so the
    Blackjack call site reads as what it is and so the reason the two deltas disagree has somewhere
    to live.

    Args:
        player_id (int): Discord user ID for the player account.
        player_account_name (str): Account name to store for the player.
        player_delta (int): Signed net change for the player, bonuses included.
        casino_delta (int): Signed change to apply to the casino ledger, bonuses excluded.
        player_avatar_url (str): Last-seen Discord avatar URL for the player.

    Returns:
        A `RoundSettlementResult` with the post-write player and casino balances.
    """
    return await apply_round_settlement(
        player_id=player_id,
        player_account_name=player_account_name,
        player_avatar_url=player_avatar_url,
        player_delta=player_delta,
        casino_delta=casino_delta,
    )


async def get_jackpot_pool(game_id: str) -> int:
    """Returns the current `pool_balance` for a game's shared jackpot.

    Reading the seeded row is the canonical way to surface the current
    pool to a view (lobby start, every active-table refresh). Seeded pools
    are replenished before returning if an older process left them drained.
    Returns `0` when the row hasn't been seeded yet so a freshly-introduced
    game can short-circuit cleanly.

    Args:
        game_id (str): Game identifier (e.g. `"dragon_gate"`).

    Returns:
        The current pool balance in points.
    """
    snapshot = await get_jackpot_snapshot(game_id=game_id)
    return snapshot.balance


async def get_jackpot_snapshot(game_id: str) -> JackpotSnapshot:
    """Returns the current jackpot balance and generation for a shared pool.

    A read that can WRITE: a seeded pool sitting at or below zero is replenished here and the
    commit is this call's, which is why a view refresh is enough to bring a drained pool back.
    The generation is the guard a view carries into its next settlement, so a stale button press
    cannot spend a pool that has since been reseeded.

    Args:
        game_id (str): Game identifier (e.g. `"dragon_gate"`).

    Returns:
        The pool balance and generation after any replenishment, all-zero for an unseeded game.
    """
    await _ensure_schema()
    async with open_session() as session:
        snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
            session=session, game_id=game_id, now=_database_now()
        )
        await session.commit()
        return snapshot


async def _replenish_jackpot_if_depleted_in_session(
    session: AsyncSession, game_id: str, balance: int, generation: int, now: datetime
) -> JackpotSnapshot:
    """Tops a seeded jackpot back up when the stored balance is drained.

    Only a game with a configured seed is ever topped up, so an unseeded pool at zero stays at
    zero. `seeded_amount` takes the whole restoration including any overdraft, since it is the
    running total of on-the-house money and must stay comparable with what players contributed.
    Bumping `generation` is what retires every snapshot a table was still holding. The UPDATE is
    guarded on `pool_balance <= 0`, so two concurrent refreshes cannot seed the pool twice: the
    loser matches no row and reports the balance it came in with.

    Args:
        session (AsyncSession): Active session; the write is left uncommitted.
        game_id (str): Game identifier (jackpot row primary key).
        balance (int): Pool balance observed by the caller.
        generation (int): Pool generation observed by the caller.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        The snapshot after any reseed, or the caller's observed values unchanged.
    """
    seed_amount = _jackpot_seed_amount(game_id=game_id)
    if seed_amount <= 0 or balance > 0:
        return JackpotSnapshot(balance=balance, generation=generation)
    replenishment = seed_amount - min(balance, 0)
    stmt = (
        update(JackpotPool)
        .where(JackpotPool.game_id == game_id)
        .where(JackpotPool.pool_balance <= 0)
        .values(
            pool_balance=seed_amount,
            seeded_amount=JackpotPool.seeded_amount + replenishment,
            generation=JackpotPool.generation + 1,
            updated_at=now,
        )
        .returning(JackpotPool.pool_balance, JackpotPool.generation)
    )
    result = await session.execute(statement=stmt)
    row = result.one_or_none()
    if row is None:
        return JackpotSnapshot(balance=balance, generation=generation)
    return JackpotSnapshot(balance=row[0], generation=row[1])


async def _apply_jackpot_delta_in_session(
    session: AsyncSession, game_id: str, delta: int, now: datetime
) -> tuple[JackpotSnapshot, bool]:
    """Applies a signed delta to a game's jackpot pool inside the caller's session.

    Positive deltas accumulate `total_contributed` (player losses /
    antes flowing into the pool); negative deltas accumulate
    `total_claimed` with the absolute value (winning payouts flowing
    out). Seeded pools are topped back up automatically after a drain, so
    the returned balance is always ready for the next table.

    The depletion flag is read BEFORE the replenishment, so the caller can tell a player their
    win emptied the pool even though the returned balance already shows it refilled. The negative
    branch is unused today: the one caller only feeds contributions in, because a payout has to go
    through `_claim_jackpot_payout_in_session`, which caps the claim at what the pool holds.

    Args:
        session (AsyncSession): Active session; the writes are left uncommitted.
        game_id (str): Game identifier (jackpot row primary key).
        delta (int): Signed point adjustment to apply to `pool_balance`.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        A tuple containing the pool balance after the write and any automatic
        replenishment, plus whether the pool was depleted by this write.
    """
    contributed_add = max(delta, 0)
    claimed_add = max(-delta, 0)
    stmt = (
        insert(JackpotPool)
        .values(
            game_id=game_id,
            pool_balance=delta,
            total_contributed=contributed_add,
            total_claimed=claimed_add,
            seeded_amount=0,
            generation=0,
            updated_at=now,
        )
        .on_conflict_do_update(
            index_elements=["game_id"],
            set_={
                "pool_balance": JackpotPool.pool_balance + delta,
                "total_contributed": JackpotPool.total_contributed + contributed_add,
                "total_claimed": JackpotPool.total_claimed + claimed_add,
                "updated_at": now,
            },
        )
        .returning(JackpotPool.pool_balance, JackpotPool.generation)
    )
    result = await session.execute(statement=stmt)
    pool_balance, generation = result.one()
    jackpot_depleted = pool_balance <= 0 and _jackpot_seed_amount(game_id=game_id) > 0
    snapshot = await _replenish_jackpot_if_depleted_in_session(
        session=session, game_id=game_id, balance=pool_balance, generation=generation, now=now
    )
    return snapshot, jackpot_depleted


async def _read_jackpot_snapshot_or_replenish_in_session(
    session: AsyncSession, game_id: str, now: datetime
) -> JackpotSnapshot:
    """Reads the jackpot balance, replenishing the seed if depleted.

    Returns a zero snapshot if no pool row exists for the game.

    The read every jackpot path funnels through, so nobody ever observes a drained seeded pool.

    Args:
        session (AsyncSession): Active session; any reseed is left uncommitted.
        game_id (str): Game identifier (jackpot row primary key).
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        The pool balance and generation after any reseed.
    """
    result = await session.execute(
        statement=select(JackpotPool.pool_balance, JackpotPool.generation).where(
            JackpotPool.game_id == game_id
        )
    )
    row = result.one_or_none()
    if row is None:
        return JackpotSnapshot(balance=0, generation=0)
    pool_balance, generation = row
    return await _replenish_jackpot_if_depleted_in_session(
        session=session, game_id=game_id, balance=pool_balance, generation=generation, now=now
    )


async def _claim_jackpot_payout_in_session(
    session: AsyncSession,
    game_id: str,
    amount: int,
    expected_generation: int | None,
    now: datetime,
) -> tuple[int, JackpotSnapshot, bool]:
    """Atomically claims up to `amount` from the requested jackpot generation.

    A payout is capped at what the pool holds, so a win larger than the pool pays out the pool and
    the caller settles the player at that reduced figure. The conditional UPDATE pins both the
    observed balance and the observed generation, which is what stops two winners claiming the same
    points; a lost race re-reads and retries. `expected_generation` is the caller's own staleness
    guard: a view that saw the pool before a reseed claims nothing at all rather than spending the
    fresh seed on an action taken against the old one.

    Args:
        session (AsyncSession): Active session; the writes are left uncommitted.
        game_id (str): Game identifier (jackpot row primary key).
        amount (int): Requested payout; a non-positive amount claims nothing.
        expected_generation (int | None): Pool generation the caller observed, or None to accept
            whatever generation is current.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        `(claimed_amount, snapshot_after_any_reseed, depleted_by_this_claim)`.

    Raises:
        RuntimeError: `_JACKPOT_CLAIM_MAX_RETRIES` conditional writes all lost their race.
    """
    if amount <= 0:
        snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
            session=session, game_id=game_id, now=now
        )
        return 0, snapshot, False

    for _ in range(_JACKPOT_CLAIM_MAX_RETRIES):
        snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
            session=session, game_id=game_id, now=now
        )
        if expected_generation is not None and snapshot.generation != expected_generation:
            return 0, snapshot, False
        claim = min(amount, snapshot.balance)
        if claim <= 0:
            return 0, snapshot, False

        new_balance = snapshot.balance - claim
        stmt = (
            update(JackpotPool)
            .where(JackpotPool.game_id == game_id)
            .where(JackpotPool.pool_balance == snapshot.balance)
            .where(JackpotPool.generation == snapshot.generation)
            .values(
                pool_balance=new_balance,
                total_claimed=JackpotPool.total_claimed + claim,
                updated_at=now,
            )
            .returning(JackpotPool.pool_balance, JackpotPool.generation)
        )
        result = await session.execute(statement=stmt)
        row = result.one_or_none()
        if row is None:
            continue

        pool_balance, generation = row
        jackpot_depleted = pool_balance <= 0 and _jackpot_seed_amount(game_id=game_id) > 0
        final_snapshot = await _replenish_jackpot_if_depleted_in_session(
            session=session, game_id=game_id, balance=pool_balance, generation=generation, now=now
        )
        return claim, final_snapshot, jackpot_depleted

    raise RuntimeError(f"claim_jackpot_payout retry budget exhausted for game_id={game_id}")


async def apply_jackpot_settlement(  # noqa: PLR0913 -- public jackpot facade mirrors player identity + snapshot guard
    player_id: int,
    player_account_name: str,
    player_delta: int,
    game_id: str,
    player_avatar_url: str = "",
    expected_jackpot_generation: int | None = None,
) -> JackpotSettlementResult:
    """Atomic player-and-jackpot settlement for a single wager event.

    This is a convenience wrapper around `apply_jackpot_settlement_batch`.

    Args:
        player_id (int): Discord user ID for the player.
        player_account_name (str): Account name to store on the player row.
        player_delta (int): Signed net change for the player. Losses are written
            as a negative delta and the absolute value flows into the pool.
        game_id (str): Jackpot game identifier (e.g. `"dragon_gate"`).
        player_avatar_url (str): Last-seen Discord avatar URL for the player.
        expected_jackpot_generation (int | None): Optional pool generation observed by the
            caller. Positive payouts only claim from this generation.

    Returns:
        The single-player jackpot settlement outcome.
    """
    result = await apply_jackpot_settlement_batch(
        game_id=game_id,
        settlements=(
            JackpotSettlementRequest(
                player_id=player_id,
                player_account_name=player_account_name,
                player_avatar_url=player_avatar_url,
                player_delta=player_delta,
                expected_jackpot_generation=expected_jackpot_generation,
            ),
        ),
    )
    return JackpotSettlementResult(
        player_balance=result.player_balances.get(player_id, 0),
        jackpot_balance=result.jackpot_balance,
        jackpot_generation=result.jackpot_generation,
        applied_player_delta=result.applied_player_deltas.get(player_id, 0),
        jackpot_depleted=result.jackpot_depleted,
        rejected=player_id in result.rejected_player_ids,
    )


async def _full_debit_rejections_in_session(
    session: AsyncSession, settlements: Sequence[JackpotSettlementRequest]
) -> tuple[int, ...]:
    """Returns required-full-debit player IDs that cannot cover their debits.

    A pre-flight check so a table ante can be refused before a single write happens, rather than
    written and rolled back. Debits for one player are summed across the batch, since two antes
    from the same seat have to be affordable together. It is advisory only: the batch still
    verifies each applied debit as it goes, which is what covers a balance that moves in between.

    Args:
        session (AsyncSession): Active session to read balances through.
        settlements (Sequence[JackpotSettlementRequest]): The batch about to be applied.

    Returns:
        Player IDs whose required debits exceed their balance, empty when the batch can proceed.
    """
    required_debits: dict[int, int] = {}
    for settlement in settlements:
        if settlement.require_full_debit and settlement.player_delta < 0:
            required_debits[settlement.player_id] = (
                required_debits.get(settlement.player_id, 0) - settlement.player_delta
            )
    if not required_debits:
        return ()

    result = await session.execute(
        statement=select(UserWallet.user_id, UserWallet.balance).where(
            UserWallet.user_id.in_(other=tuple(required_debits))
        )
    )
    balances = {row[0]: row[1] for row in result.all()}
    return tuple(
        user_id
        for user_id, required in required_debits.items()
        if balances.get(user_id, 0) < required
    )


async def apply_jackpot_settlement_batch(
    game_id: str, settlements: Sequence[JackpotSettlementRequest]
) -> JackpotSettlementBatchResult:
    """Coordinates one or more player settlements against a jackpot pool.

    Positive player deltas (wins) are capped to the live pool balance inside
    this transaction, then credited through the shared income path. Negative
    deltas normally clamp at zero and feed the pool with the actual debit.
    Required-full-debit settlements reject the whole batch instead. If a seeded
    pool is drained, the same transaction restores its on-the-house seed.
    Player and jackpot rows live in the same `data/database/economy.db` file,
    so the whole batch commits as one atomic transaction.

    Rejection is all-or-nothing and reports itself through `rejected_player_ids` with empty balance
    maps rather than by raising: a table ante one seat cannot afford leaves the pool and every
    other seat untouched. The rollback mid-loop is safe because nothing has been committed yet, so
    one call discards every player and pool write the batch had made.

    Args:
        game_id (str): Jackpot game identifier (e.g. `"dragon_gate"`).
        settlements (Sequence[JackpotSettlementRequest]): Player-side settlements, applied in
            order.

    Returns:
        The latest balance for each touched player, the actual applied deltas,
        and the final jackpot balance after the final settlement and any reseed.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        player_balances: dict[int, int] = {}
        applied_player_deltas: dict[int, int] = {}
        jackpot_snapshot: JackpotSnapshot | None = None
        jackpot_depleted = False

        try:
            rejected_player_ids = await _full_debit_rejections_in_session(
                session=session, settlements=settlements
            )
            if rejected_player_ids:
                jackpot_snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
                    session=session, game_id=game_id, now=now
                )
                await session.commit()
                return JackpotSettlementBatchResult(
                    player_balances={},
                    applied_player_deltas={},
                    jackpot_balance=jackpot_snapshot.balance,
                    jackpot_generation=jackpot_snapshot.generation,
                    rejected_player_ids=rejected_player_ids,
                )

            for settlement in settlements:
                effective_player_delta = settlement.player_delta
                if effective_player_delta > 0:
                    claim, jackpot_snapshot, depleted = await _claim_jackpot_payout_in_session(
                        session=session,
                        game_id=game_id,
                        amount=effective_player_delta,
                        expected_generation=settlement.expected_jackpot_generation,
                        now=now,
                    )
                    effective_player_delta = claim
                    jackpot_depleted = jackpot_depleted or depleted

                (
                    player_balance,
                    applied_player_delta,
                ) = await _apply_jackpot_player_delta_in_session(
                    session=session,
                    user_id=settlement.player_id,
                    name=settlement.player_account_name,
                    avatar_url=settlement.player_avatar_url,
                    delta=effective_player_delta,
                    now=now,
                )
                if (
                    settlement.require_full_debit
                    and applied_player_delta != effective_player_delta
                ):
                    # Nothing in the batch is committed yet, so one rollback
                    # discards every player and jackpot write so far.
                    await session.rollback()
                    jackpot_snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
                        session=session, game_id=game_id, now=now
                    )
                    await session.commit()
                    return JackpotSettlementBatchResult(
                        player_balances={},
                        applied_player_deltas={},
                        jackpot_balance=jackpot_snapshot.balance,
                        jackpot_generation=jackpot_snapshot.generation,
                        rejected_player_ids=(settlement.player_id,),
                    )
                player_balances[settlement.player_id] = player_balance
                applied_player_deltas[settlement.player_id] = applied_player_delta

                if applied_player_delta == 0:
                    jackpot_snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
                        session=session, game_id=game_id, now=now
                    )
                    continue

                if applied_player_delta < 0:
                    jackpot_snapshot, depleted = await _apply_jackpot_delta_in_session(
                        session=session, game_id=game_id, delta=-applied_player_delta, now=now
                    )
                    jackpot_depleted = jackpot_depleted or depleted

            if jackpot_snapshot is None:
                jackpot_snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
                    session=session, game_id=game_id, now=now
                )

            await session.commit()
            if any(delta != 0 for delta in applied_player_deltas.values()):
                invalidate_economy_leaderboard_cache()
            return JackpotSettlementBatchResult(
                player_balances=player_balances,
                applied_player_deltas=applied_player_deltas,
                jackpot_balance=jackpot_snapshot.balance,
                jackpot_generation=jackpot_snapshot.generation,
                jackpot_depleted=jackpot_depleted,
            )
        except Exception:
            await session.rollback()
            raise


def _next_checkin_streak(
    last_checkin_at: datetime | None,
    current_streak: int,
    today_midnight: datetime,
    yesterday_midnight: datetime,
    tomorrow_midnight: datetime,
) -> int | None:
    """Returns the streak counter for the next check-in.

    Returns `None` when the user has already checked in today.

    A streak advances only from yesterday: any longer gap, and a full cycle already completed,
    both restart at 1. Pure and database-free, so the three day boundaries are passed in rather
    than derived here and the whole rule can be exercised without a clock.

    Args:
        last_checkin_at (datetime | None): Stored `last_checkin_at` (Taipei-naive) or `None`.
        current_streak (int): Currently-persisted streak counter.
        today_midnight (datetime): 00:00 Asia/Taipei for the request day.
        yesterday_midnight (datetime): 00:00 Asia/Taipei for the prior day.
        tomorrow_midnight (datetime): 00:00 Asia/Taipei for the next day.

    Returns:
        The streak number to persist, or `None` if today is already done.
    """
    if last_checkin_at is None:
        return 1
    last_local = _as_taipei(dt=last_checkin_at)
    if today_midnight <= last_local < tomorrow_midnight:
        return None
    if (
        yesterday_midnight <= last_local < today_midnight
        and 0 < current_streak < CHECKIN_STREAK_CYCLE
    ):
        return current_streak + 1
    return 1


async def _insert_first_checkin_in_session(
    session: AsyncSession, user_id: int, name: str, avatar_url: str, now: datetime
) -> tuple[int, int, int, bool] | None:
    """Inserts a fresh user row crediting the day-1 check-in reward.

    Returns `None` when another coroutine already inserted the row so
    the caller retries on the next loop iteration.

    A brand-new account cannot be VIP, so the reward is computed at day 1 with `is_vip=False`
    without reading anything back.

    Args:
        session (AsyncSession): Active session; the writes are left uncommitted.
        user_id (int): Discord user ID checking in.
        name (str): Last-seen Discord username to store on the account.
        avatar_url (str): Last-seen Discord avatar URL to store when available.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        `(reward, balance_after, streak_after, vip_after)` on success or
        `None` when `ON CONFLICT DO NOTHING` rejected the insert.
    """
    new_streak = 1
    reward = checkin_reward(streak=new_streak, is_vip=False)
    insert_stmt = (
        insert(UserAccount)
        .values(
            user_id=user_id,
            name=name or str(user_id),
            avatar_url=avatar_url,
            is_vip=False,
            last_checkin_at=now,
            checkin_streak=new_streak,
            is_admin=False,
            is_central_banker=False,
            hide_from_leaderboard=False,
            updated_at=now,
        )
        .on_conflict_do_nothing(index_elements=["user_id"])
    )
    insert_result = cast("CursorResult[Any]", await session.execute(statement=insert_stmt))
    if (insert_result.rowcount or 0) == 0:
        return None
    credit_result = await session.execute(
        statement=_build_credit_upsert(user_id=user_id, name=name, amount=reward, now=now)
    )
    balance_after = credit_result.scalar_one()
    invalidate_economy_leaderboard_cache()
    return reward, balance_after, new_streak, False


async def _update_checkin_row_in_session(  # noqa: PLR0913 -- session helper carries account identity + observed row
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    now: datetime,
    new_streak: int,
    row: tuple[datetime | None, int, bool, str],
) -> tuple[int, int, int, bool] | None:
    """Performs the conditional UPDATE for an existing account.

    The WHERE clause pins `last_checkin_at` to the observed value so
    concurrent check-ins cannot double-credit.

    The VIP flag comes from the row this call already read, so a VIP bought moments earlier is
    honored on this very check-in rather than the next one. The credit only runs after the
    conditional UPDATE has claimed the day, which is the ordering that makes the pin worth having.

    Args:
        session (AsyncSession): Active session; the writes are left uncommitted.
        user_id (int): Discord user ID checking in.
        name (str): Last-seen Discord username to refresh on the account.
        avatar_url (str): Last-seen Discord avatar URL to refresh when set.
        now (datetime): `_database_now()` value pinned for this transaction.
        new_streak (int): Streak counter chosen by `_next_checkin_streak`.
        row (tuple[datetime | None, int, bool, str]): Tuple returned by the prior SELECT.

    Returns:
        `(reward, balance_after, streak_after, vip_after)` on success or
        `None` when the conditional UPDATE matched zero rows.
    """
    last_checkin_at, _current_streak, is_vip, existing_name = row
    reward = checkin_reward(streak=new_streak, is_vip=is_vip)

    update_values: dict[str, Any] = {
        "last_checkin_at": now,
        "checkin_streak": new_streak,
        "updated_at": now,
    }
    if name and name != existing_name:
        update_values["name"] = name
    if avatar_url:
        update_values["avatar_url"] = avatar_url

    last_checkin_gate: ColumnElement[bool]
    if last_checkin_at is None:
        last_checkin_gate = UserAccount.last_checkin_at.is_(None)
    else:
        last_checkin_gate = UserAccount.last_checkin_at == last_checkin_at

    stmt = (
        update(UserAccount)
        .where(UserAccount.user_id == user_id, last_checkin_gate)
        .values(**update_values)
        .returning(UserAccount.checkin_streak, UserAccount.is_vip)
    )
    update_result = await session.execute(statement=stmt)
    updated_row = update_result.one_or_none()
    if updated_row is None:
        return None
    streak_after, vip_after = updated_row
    credit_result = await session.execute(
        statement=_build_credit_upsert(user_id=user_id, name=name, amount=reward, now=now)
    )
    balance_after = credit_result.scalar_one()
    invalidate_economy_leaderboard_cache()
    return reward, balance_after, streak_after, bool(vip_after)


async def checkin(user_id: int, name: str, avatar_url: str = "") -> CheckinResult | None:
    """Records a daily check-in and credits the streak-adjusted reward.

    Returns `None` when the user has already checked in today (Taipei
    local date). On first check-in or after a missed day the streak resets
    to 1; otherwise the streak advances by 1 and cycles back to 1 after
    reaching `CHECKIN_STREAK_CYCLE`. The reward is computed with
    `checkin_reward` and persisted alongside the streak counter in the
    same write. VIP perks (2x base) read the persisted flag inside the
    same transaction so a freshly-bought VIP immediately applies on the
    next check-in.

    The SELECT-then-conditional-UPDATE pattern (gated on the
    observed `last_checkin_at` value) prevents two parallel coroutines
    from double-crediting. First-sight INSERTs use ``ON CONFLICT DO
    NOTHING`` to defer to whichever writer landed first; the loser falls
    through to the next retry with the freshly-visible row.

    Args:
        user_id (int): Discord user ID checking in.
        name (str): Last-seen Discord username to store on the account.
        avatar_url (str): Last-seen Discord avatar URL to store when available.

    Returns:
        `CheckinResult` describing the credit, or `None` when the user
        already checked in today. `None` also covers the exhausted retry budget,
        so the caller cannot tell the two apart.
    """
    await _ensure_schema()
    now = _database_now()
    today_midnight = _taipei_midnight(now=now)
    yesterday_midnight = today_midnight - timedelta(days=1)
    tomorrow_midnight = today_midnight + timedelta(days=1)

    async with open_session() as session:
        for _ in range(_CHECKIN_MAX_RETRIES):
            read_result = await session.execute(
                statement=select(
                    UserAccount.last_checkin_at,
                    UserAccount.checkin_streak,
                    UserAccount.is_vip,
                    UserAccount.name,
                ).where(UserAccount.user_id == user_id)
            )
            row = read_result.one_or_none()

            if row is None:
                outcome = await _insert_first_checkin_in_session(
                    session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
                )
            else:
                new_streak = _next_checkin_streak(
                    last_checkin_at=row[0],
                    current_streak=row[1],
                    today_midnight=today_midnight,
                    yesterday_midnight=yesterday_midnight,
                    tomorrow_midnight=tomorrow_midnight,
                )
                if new_streak is None:
                    return None
                outcome = await _update_checkin_row_in_session(
                    session=session,
                    user_id=user_id,
                    name=name,
                    avatar_url=avatar_url,
                    now=now,
                    new_streak=new_streak,
                    row=cast("tuple[datetime | None, int, bool, str]", row),
                )

            if outcome is None:
                await session.rollback()
                continue

            reward, balance_after, streak_after, vip_after = outcome
            await session.commit()
            invalidate_economy_leaderboard_cache()
            return CheckinResult(
                new_balance=balance_after, amount=reward, streak=streak_after, is_vip=vip_after
            )

        return None


async def buy_vip(user_id: int, name: str, avatar_url: str = "") -> VipPurchaseResult | None:
    """Promotes the user to VIP after debiting `VIP_PURCHASE_COST` points.

    Returns `None` when the user is already VIP, has insufficient balance,
    or the retry budget for the conditional UPDATE was exhausted.

    Also `None` for a user with no wallet row: the join finds nothing, and someone who has never
    earned a point cannot afford `VIP_PURCHASE_COST` anyway. The debit and the flag are two
    conditional writes in one transaction, each pinned on what was observed (balance unchanged,
    `is_vip` still false), so a race can charge nobody twice and cannot grant the flag for free.
    VIP is permanent, so there is no revoke path here.

    Args:
        user_id (int): Discord user ID purchasing VIP.
        name (str): Last-seen Discord username to store on the account.
        avatar_url (str): Last-seen Discord avatar URL to store when available.

    Returns:
        `VipPurchaseResult` describing the post-purchase balance, or
        `None` when the purchase was rejected.
    """
    await _ensure_schema()
    now = _database_now()
    cost = VIP_PURCHASE_COST

    async with open_session() as session:
        for _ in range(_VIP_PURCHASE_MAX_RETRIES):
            read_result = await session.execute(
                statement=select(UserWallet.balance, UserAccount.is_vip, UserAccount.name)
                .select_from(UserAccount)
                .join(UserWallet, UserWallet.user_id == UserAccount.user_id)
                .where(UserAccount.user_id == user_id)
            )
            row = read_result.one_or_none()
            if row is None:
                return None
            balance, is_vip, existing_name = row
            if is_vip:
                return None
            if balance < cost:
                return None

            new_balance = balance - cost
            wallet_values: dict[str, Any] = {
                "balance": new_balance,
                "total_spent": UserWallet.total_spent + cost,
                "updated_at": now,
            }
            if name:
                wallet_values["name"] = name
            wallet_result = await session.execute(
                statement=update(UserWallet)
                .where(UserWallet.user_id == user_id, UserWallet.balance == balance)
                .values(**wallet_values)
                .returning(UserWallet.balance)
            )
            wallet_row = wallet_result.one_or_none()
            if wallet_row is None:
                await session.rollback()
                continue

            update_values: dict[str, Any] = {"is_vip": True, "updated_at": now}
            if name and name != existing_name:
                update_values["name"] = name
            if avatar_url:
                update_values["avatar_url"] = avatar_url

            stmt = (
                update(UserAccount)
                .where(UserAccount.user_id == user_id, UserAccount.is_vip.is_(False))
                .values(**update_values)
                .returning(UserAccount.user_id)
            )
            update_result = await session.execute(statement=stmt)
            updated_row = update_result.one_or_none()
            if updated_row is None:
                await session.rollback()
                continue

            await session.commit()
            invalidate_economy_leaderboard_cache()
            return VipPurchaseResult(new_balance=wallet_row[0], cost=cost)

        return None


async def get_balance(user_id: int) -> int:
    """Returns the current balance for a user.

    Args:
        user_id (int): Discord user ID to look up.

    Returns:
        The current balance, or 0 if the user has never been seen.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserWallet.balance).where(UserWallet.user_id == user_id)
        )
        return result.scalar_one_or_none() or 0


async def get_vip(user_id: int) -> bool:
    """Returns whether the user owns the VIP perk.

    The flag is permanent once set, so a caller may read it outside the settlement transaction it
    affects: the worst a race costs is the bonus on one in-flight round.

    Args:
        user_id (int): Discord user ID to look up.

    Returns:
        `True` when the account has `is_vip` set, else `False`.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.is_vip).where(UserAccount.user_id == user_id)
        )
        return bool(result.scalar_one_or_none())


async def get_admin(user_id: int) -> bool:
    """Returns whether the user can run economy admin commands.

    Independent of Discord's own permissions: an economy admin is a flag on this table, and a
    guild administrator without it has no economy powers.

    Args:
        user_id (int): Discord user ID to look up.

    Returns:
        `True` when the account has `is_admin` set, else `False`.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.is_admin).where(UserAccount.user_id == user_id)
        )
        return bool(result.scalar_one_or_none())


async def set_admin(user_id: int, name: str, is_admin: bool, avatar_url: str = "") -> bool:
    """Sets the economy admin flag for a Discord user.

    Granting admin creates a zero-balance account row if the user has never
    touched the economy system. Revoking admin updates an existing row only;
    missing users are left untouched so revoke operations do not create empty
    account rows.

    Args:
        user_id (int): Discord user ID to modify.
        name (str): Last-seen Discord username to store when available.
        is_admin (bool): Desired admin flag value.
        avatar_url (str): Last-seen Discord avatar URL to store when available.

    Returns:
        `True` when a row was created or updated; `False` when revoking a
        missing user.
    """
    await _ensure_schema()
    now = _database_now()
    effective_name = name or str(user_id)
    async with open_session() as session:
        if is_admin:
            stmt = insert(UserAccount).values(
                user_id=user_id,
                name=effective_name,
                avatar_url=avatar_url,
                updated_at=now,
                is_vip=False,
                last_checkin_at=None,
                checkin_streak=0,
                is_admin=True,
            )
            set_: dict[str, Any] = {"is_admin": True, "updated_at": now}
            if name:
                set_["name"] = effective_name
            if avatar_url:
                set_["avatar_url"] = avatar_url
            result = await session.execute(
                statement=stmt.on_conflict_do_update(
                    index_elements=["user_id"], set_=set_
                ).returning(UserAccount.user_id)
            )
            await session.commit()
            return result.scalar_one_or_none() is not None

        values: dict[str, Any] = {"is_admin": False, "updated_at": now}
        if name:
            values["name"] = effective_name
        if avatar_url:
            values["avatar_url"] = avatar_url
        result = await session.execute(
            statement=update(UserAccount)
            .where(UserAccount.user_id == user_id)
            .values(**values)
            .returning(UserAccount.user_id)
        )
        await session.commit()
        return result.scalar_one_or_none() is not None


async def list_admins() -> list[AdminAccount]:
    """Returns all economy admins ordered by user ID.

    Unfiltered and unpaged; the flag is set by hand, so the set stays small enough for one query.

    Returns:
        One entry per account with `is_admin` set, ordered by user ID.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.user_id, UserAccount.name)
            .where(UserAccount.is_admin.is_(True))
            .order_by(UserAccount.user_id)
        )
        return [AdminAccount(user_id=row[0], name=row[1]) for row in result.all()]


async def get_central_banker(user_id: int) -> bool:
    """Returns whether the user can operate central-bank lending commands.

    A third flag, separate from both Discord admin and `is_admin`, set offline by direct DB write.
    Approving a central-bank loan mints money, so nothing in the Discord surface grants it.

    Args:
        user_id (int): Discord user ID to look up.

    Returns:
        `True` when the account has `is_central_banker` set, else `False`.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.is_central_banker).where(UserAccount.user_id == user_id)
        )
        return bool(result.scalar_one_or_none())


async def set_central_banker(
    user_id: int, name: str, is_central_banker: bool, avatar_url: str = ""
) -> bool:
    """Sets the central banker flag for a Discord user.

    Mirrors `set_admin`: granting creates the account row when it is missing, revoking updates an
    existing row only, so a revoke never leaves an empty account behind.

    Args:
        user_id (int): Discord user ID to modify.
        name (str): Last-seen Discord username to store when available.
        is_central_banker (bool): Desired central-banker flag value.
        avatar_url (str): Last-seen Discord avatar URL to store when available.

    Returns:
        `True` when a row was created or updated; `False` when revoking a missing user.
    """
    await _ensure_schema()
    now = _database_now()
    effective_name = name or str(user_id)
    async with open_session() as session:
        if is_central_banker:
            stmt = insert(UserAccount).values(
                user_id=user_id,
                name=effective_name,
                avatar_url=avatar_url,
                updated_at=now,
                is_vip=False,
                last_checkin_at=None,
                checkin_streak=0,
                is_admin=False,
                is_central_banker=True,
            )
            set_: dict[str, Any] = {"is_central_banker": True, "updated_at": now}
            if name:
                set_["name"] = effective_name
            if avatar_url:
                set_["avatar_url"] = avatar_url
            result = await session.execute(
                statement=stmt.on_conflict_do_update(
                    index_elements=["user_id"], set_=set_
                ).returning(UserAccount.user_id)
            )
            await session.commit()
            return result.scalar_one_or_none() is not None

        values: dict[str, Any] = {"is_central_banker": False, "updated_at": now}
        if name:
            values["name"] = effective_name
        if avatar_url:
            values["avatar_url"] = avatar_url
        result = await session.execute(
            statement=update(UserAccount)
            .where(UserAccount.user_id == user_id)
            .values(**values)
            .returning(UserAccount.user_id)
        )
        await session.commit()
        return result.scalar_one_or_none() is not None


async def get_account(user_id: int) -> AccountSnapshot | None:
    """Returns the stored account snapshot for a user.

    Keyed on `user_account`, with the wallet outer-joined: someone who has an identity row but has
    never held money reads back as all-zero rather than as missing.

    Args:
        user_id (int): Discord user ID to look up.

    Returns:
        An account snapshot, or `None` if the user has never been seen.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(
                UserAccount.name,
                UserWallet.balance,
                UserWallet.total_earned,
                UserWallet.total_spent,
            )
            .select_from(UserAccount)
            .outerjoin(UserWallet, UserWallet.user_id == UserAccount.user_id)
            .where(UserAccount.user_id == user_id)
        )
        row = result.one_or_none()
        if row is None:
            return None
        return AccountSnapshot(
            name=row[0], balance=row[1] or 0, total_earned=row[2] or 0, total_spent=row[3] or 0
        )


async def transfer(  # noqa: PLR0913 -- transfer needs sender and receiver identity snapshots
    sender_id: int,
    sender_name: str,
    receiver_id: int,
    receiver_name: str,
    amount: int,
    sender_avatar_url: str = "",
    receiver_avatar_url: str = "",
) -> TransferResult | None:
    """Atomically moves points from sender to receiver, burning a transfer tax.

    The debit is a single conditional `UPDATE` gated on `balance >= amount`;
    if that returns no row the transfer is rejected without ever touching
    the receiver. The credit is a UPSERT in the same transaction, so the
    receiver row is created on first contact and the whole transfer is one
    all-or-nothing operation. Both balances are returned from the same SQL
    writes, so callers do not need extra reads after a successful transfer.

    The sender is debited the full `amount`, but the receiver only receives
    `amount - tax` where `tax = amount * TRANSFER_TAX_BPS // 10_000`. The
    burned difference is removed from circulation entirely, acting as a
    permanent money sink. Per-side the `balance == total_earned - total_spent`
    invariant is preserved (sender `total_spent += amount`, receiver
    `total_earned += net`).

    Args:
        sender_id (int): Discord user ID to debit.
        sender_name (str): Last-seen Discord username to store on the sender account.
        receiver_id (int): Discord user ID to credit.
        receiver_name (str): Last-seen Discord username to store on the receiver account.
        amount (int): Number of points to transfer.
        sender_avatar_url (str): Last-seen Discord avatar URL for the sender.
        receiver_avatar_url (str): Last-seen Discord avatar URL for the receiver.

    Returns:
        The post-transfer balances when the transfer committed, or `None`
        when validation failed or the sender had insufficient funds.
    """
    await _ensure_schema()
    if amount <= 0 or sender_id == receiver_id:
        return None

    now = _database_now()
    async with open_session() as session:
        debit_values: dict[str, Any] = {
            "balance": UserWallet.balance - amount,
            "total_spent": UserWallet.total_spent + amount,
            "updated_at": now,
        }
        if sender_name:
            debit_values["name"] = sender_name

        debit_stmt = (
            update(UserWallet)
            .where(UserWallet.user_id == sender_id, UserWallet.balance >= amount)
            .values(**debit_values)
            .returning(UserWallet.balance)
        )
        debit_result = await session.execute(statement=debit_stmt)
        debit_row = debit_result.one_or_none()
        if debit_row is None:
            await session.rollback()
            return None
        sender_balance = debit_row[0]
        await _upsert_user_metadata_in_session(
            session=session,
            user_id=sender_id,
            name=sender_name,
            avatar_url=sender_avatar_url,
            now=now,
        )

        tax = amount * TRANSFER_TAX_BPS // 10_000
        net = amount - tax
        credit_stmt = _build_credit_upsert(
            user_id=receiver_id, name=receiver_name, amount=net, now=now
        )
        await _upsert_user_metadata_in_session(
            session=session,
            user_id=receiver_id,
            name=receiver_name,
            avatar_url=receiver_avatar_url,
            now=now,
        )
        credit_result = await session.execute(statement=credit_stmt)
        receiver_balance = credit_result.scalar_one()

        await session.commit()
        invalidate_economy_leaderboard_cache()
        return TransferResult(
            sender_balance=sender_balance,
            receiver_balance=receiver_balance,
            received_amount=net,
            tax_amount=tax,
        )


async def top_n(
    limit: int | None = 10, exclude_user_ids: tuple[int, ...] = (), include_hidden: bool = False
) -> list[LeaderboardEntry]:
    """Returns accounts ordered by balance descending.

    `exclude_user_ids` filters out specific accounts (notably the bot's
    own house ledger row) before applying the limit, so the leaderboard
    always shows real players. Stored integer values are sorted in SQL with
    explicit decimal-text aware order terms so the query can still apply
    `LIMIT` before rows reach Python.

    Rows are cached process-locally for `_ECONOMY_LEADERBOARD_CACHE_TTL_SECONDS` on the query's own
    arguments, so a burst of `/leaderboard` calls costs one query. Every write path calls
    `invalidate_economy_leaderboard_cache`, so the TTL is a backstop rather than the correctness
    story; the engine's identity is part of the key so a test that swaps `_engine` cannot read
    another database's rows.

    Args:
        limit (int | None): Maximum number of accounts to return, or `None` to return all
            matching accounts.
        exclude_user_ids (tuple[int, ...]): User IDs to filter out before applying the limit.
        include_hidden (bool): Whether to include accounts marked as hidden from
            public leaderboards.

    Returns:
        Leaderboard entries ordered by balance descending. `avatar_url` is
        empty when the user has never been seen by an avatar-aware write path.
    """
    await _ensure_schema()
    if limit is not None and limit <= 0:
        return []
    exclude_key = tuple(sorted(exclude_user_ids))
    cache_key: _TopNCacheKey = (id(_engine), limit, exclude_key, include_hidden)
    cached_rows = _cached_top_n_rows(cache_key=cache_key)
    if cached_rows is not None:
        return cached_rows
    async with open_session() as session:
        stmt = select(
            UserWallet.user_id, UserAccount.name, UserWallet.balance, UserAccount.avatar_url
        ).join(UserAccount, UserAccount.user_id == UserWallet.user_id)
        if not include_hidden:
            stmt = stmt.where(UserAccount.hide_from_leaderboard.is_(False))
        if exclude_user_ids:
            stmt = stmt.where(UserWallet.user_id.notin_(other=exclude_user_ids))
        stmt = stmt.order_by(*_stored_integer_desc_order(column=UserWallet.balance))
        if limit is not None:
            stmt = stmt.limit(limit=limit)
        result = await session.execute(statement=stmt)
        rows = tuple(
            LeaderboardEntry(user_id=row[0], name=row[1], balance=row[2], avatar_url=row[3] or "")
            for row in result.all()
        )
        _top_n_cache[cache_key] = (monotonic(), rows)
        return list(rows)


async def top_losers(
    limit: int = 10, exclude_user_ids: tuple[int, ...] = (), include_hidden: bool = False
) -> list[LossLeaderboardEntry]:
    """Returns the biggest gross casino losers for the current Taipei day.

    The leaderboard reads persisted `casino_account` daily counters. Writes lazily reset stale
    counters at the first casino settlement after Taipei midnight, while this
    query filters by today's `day_started_at` so yesterday's counters
    never leak into a new day.

    Losses are gross, so a later win does not offset one and the board answers "who lost the most
    today", not "who is down the most". Ordering is by text length then text because every stored
    loss is non-negative, which makes the `_stored_integer_desc_order` machinery unnecessary here.

    Args:
        limit (int): Maximum number of accounts to return.
        exclude_user_ids (tuple[int, ...]): User IDs to filter out before applying the limit.
        include_hidden (bool): Whether to include accounts marked as hidden from
            public leaderboards.

    Returns:
        Loss leaderboard entries ordered by loss descending. `loss_amount`
        is always positive.
    """
    await _ensure_schema()
    if limit <= 0:
        return []
    now = _database_now()
    today_midnight = _taipei_midnight(now=now)
    exclude_key = tuple(sorted(exclude_user_ids))
    cache_key: _TopLosersCacheKey = (
        id(_engine),
        limit,
        exclude_key,
        include_hidden,
        today_midnight,
    )
    cached_rows = _cached_top_loser_rows(cache_key=cache_key)
    if cached_rows is not None:
        return cached_rows

    async with open_session() as session:
        stmt = (
            select(
                CasinoAccount.user_id,
                CasinoAccount.name,
                UserAccount.avatar_url,
                CasinoAccount.daily_loss,
            )
            .select_from(CasinoAccount)
            .join(UserAccount, UserAccount.user_id == CasinoAccount.user_id)
            .where(CasinoAccount.day_started_at == today_midnight, CasinoAccount.daily_loss != "0")
            .order_by(desc(func.length(CasinoAccount.daily_loss)), desc(CasinoAccount.daily_loss))
            .limit(limit=limit)
        )
        if not include_hidden:
            stmt = stmt.where(UserAccount.hide_from_leaderboard.is_(False))
        if exclude_user_ids:
            stmt = stmt.where(UserAccount.user_id.notin_(other=exclude_user_ids))
        result = await session.execute(statement=stmt)
        rows: list[LossLeaderboardEntry] = []
        for row in result.all():
            loss_amount = _stored_int_to_int(value=row[3])
            if loss_amount <= 0:
                continue
            rows.append(
                LossLeaderboardEntry(
                    user_id=row[0],
                    name=row[1] or str(row[0]),
                    loss_amount=loss_amount,
                    avatar_url=row[2] or "",
                )
            )
        _top_losers_cache[cache_key] = (monotonic(), tuple(rows))
        return rows


def _loan_proposal_view(proposal: LoanProposal) -> LoanProposalView:
    """Projects an ORM loan proposal into an immutable API view.

    The boundary that keeps a mapped row from escaping the module, so a caller cannot mutate one
    and cannot hold a handle that expires with the session. The string columns are rebuilt into
    their enums here, which is where an unrecognised persisted value surfaces as a `ValueError`.

    Args:
        proposal (LoanProposal): The mapped row to project.

    Returns:
        A frozen view of the proposal.
    """
    return LoanProposalView(
        proposal_id=proposal.id,
        kind=LoanProposalKind(proposal.kind),
        status=LoanProposalStatus(proposal.status),
        lender_type=LoanLenderType(proposal.lender_type),
        borrower_id=proposal.borrower_id,
        borrower_name=proposal.borrower_name,
        lender_id=proposal.lender_id,
        lender_name=proposal.lender_name,
        amount=proposal.amount,
        monthly_rate_bps=proposal.monthly_rate_bps,
        escrow_amount=proposal.escrow_amount,
        created_at=proposal.created_at,
    )


def _loan_contract_view(contract: LoanContract) -> LoanContractView:
    """Projects an ORM loan contract into an immutable API view.

    Carries only what a caller displays or decides on; the lifetime paid totals and the avatar URLs
    stay on the row. Interest is whatever was last accrued onto it, so the caller has to have run
    the accrual first for the figure to be current.

    Args:
        contract (LoanContract): The mapped row to project.

    Returns:
        A frozen view of the contract.
    """
    return LoanContractView(
        contract_id=contract.id,
        lender_type=LoanLenderType(contract.lender_type),
        lender_id=contract.lender_id,
        lender_name=contract.lender_name,
        borrower_id=contract.borrower_id,
        borrower_name=contract.borrower_name,
        principal_remaining=contract.principal_remaining,
        interest_due=contract.interest_due,
        monthly_rate_bps=contract.monthly_rate_bps,
        opened_at=contract.opened_at,
        last_interest_accrued_at=contract.last_interest_accrued_at,
        status=LoanContractStatus(contract.status),
    )


def _loan_proposal_is_expired(proposal: LoanProposal, now: datetime) -> bool:
    """Returns whether a pending loan proposal has passed its decision window.

    Expiry is evaluated lazily whenever a proposal is touched, so there is no sweeper task and a
    proposal nobody looks at again simply stays `pending` on disk. Only a `pending` proposal can
    expire; anything already decided answers False.

    Args:
        proposal (LoanProposal): The mapped proposal row.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        True when the proposal is pending and older than `LOAN_PROPOSAL_TIMEOUT_SECONDS`.
    """
    if proposal.status != LoanProposalStatus.PENDING:
        return False
    elapsed_seconds = (_as_taipei(dt=now) - _as_taipei(dt=proposal.created_at)).total_seconds()
    return elapsed_seconds >= LOAN_PROPOSAL_TIMEOUT_SECONDS


async def _reject_expired_loan_proposal_in_session(
    session: AsyncSession, proposal: LoanProposal, now: datetime
) -> LoanProposalView | None:
    """Marks an expired pending proposal as rejected inside the caller's session.

    Every decision path (accept, reject, cancel, the explicit expiry call) runs this first, so a
    decision arriving after the window closes rejects the proposal instead of acting on it. The
    UPDATE is pinned on `status = pending`, and the in-memory row is patched to match so the
    caller can project it without re-reading. A non-expired or already-decided proposal returns
    None and is left untouched.

    Args:
        session (AsyncSession): Active session; the writes are left uncommitted.
        proposal (LoanProposal): The mapped proposal row.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        The rejected proposal's view when this call expired it, else None.
    """
    if not _loan_proposal_is_expired(proposal=proposal, now=now):
        return None
    status_result = await session.execute(
        statement=update(LoanProposal)
        .where(LoanProposal.id == proposal.id, LoanProposal.status == LoanProposalStatus.PENDING)
        .values(status=LoanProposalStatus.REJECTED, updated_at=now)
        .returning(LoanProposal.id)
    )
    if status_result.scalar_one_or_none() is None:
        return None
    await _refund_proposal_escrow_in_session(session=session, proposal=proposal, now=now)
    proposal.status = LoanProposalStatus.REJECTED
    proposal.updated_at = now
    return _loan_proposal_view(proposal=proposal)


def _loan_interest_delta(
    principal_remaining: int, monthly_rate_bps: int, last_accrued_at: datetime, now: datetime
) -> tuple[int, datetime]:
    """Returns simple-interest delta and the timestamp covered by accrual.

    Accrual is per WHOLE elapsed day on a 30-day month, and the returned timestamp advances only
    by the days actually charged, so the leftover hours are carried forward rather than lost.
    Interest is simple, computed on `principal_remaining` alone: it never compounds onto interest
    already owed. Pure and database-free.

    Args:
        principal_remaining (int): Outstanding principal to charge interest on.
        monthly_rate_bps (int): Monthly simple-interest rate in basis points.
        last_accrued_at (datetime): The point interest has already been charged up to.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        `(interest, accrued_until)`, degrading to `(0, last_accrued_at)` when less than a whole
        day has passed or the contract owes no principal.
    """
    if principal_remaining <= 0 or monthly_rate_bps <= 0:
        return 0, last_accrued_at
    elapsed_seconds = (_as_taipei(dt=now) - _as_taipei(dt=last_accrued_at)).total_seconds()
    elapsed_days = int(elapsed_seconds // 86_400)
    if elapsed_days <= 0:
        return 0, last_accrued_at
    interest = principal_remaining * monthly_rate_bps * elapsed_days // (10_000 * 30)
    return interest, _as_taipei(dt=last_accrued_at) + timedelta(days=elapsed_days)


async def _accrue_contract_interest_in_session(
    session: AsyncSession, contract: LoanContract, now: datetime
) -> None:
    """Persists lazy simple-interest accrual for one active contract.

    Mutates the mapped row and flushes, leaving the commit to the caller. Because interest is only
    charged when someone looks, every read path that shows a debt figure (`get_portfolio`,
    `list_loan_contracts`) is a WRITE path too. Idempotent within a day: a second call in the same
    24 hours accrues nothing, so running it on every read costs no extra interest.

    Args:
        session (AsyncSession): Active session; the flush is left uncommitted.
        contract (LoanContract): The mapped contract row, updated in place.
        now (datetime): `_database_now()` value pinned for this transaction.
    """
    if contract.status != LoanContractStatus.ACTIVE:
        return
    interest, accrued_until = _loan_interest_delta(
        principal_remaining=contract.principal_remaining,
        monthly_rate_bps=contract.monthly_rate_bps,
        last_accrued_at=contract.last_interest_accrued_at,
        now=now,
    )
    if interest <= 0:
        return
    contract.interest_due += interest
    contract.last_interest_accrued_at = accrued_until
    contract.updated_at = now
    await session.flush()


async def _central_bank_status_in_session(
    session: AsyncSession, exclude_user_ids: tuple[int, ...] = ()
) -> CentralBankStatus:
    """Computes central-bank lending capacity from positive user balances.

    Capacity is anchored to money that actually exists: the sum of positive wallet balances, with
    negative ones ignored so an overdrawn admin adjustment cannot shrink everyone's credit. A
    central-bank loan MINTS, so its principal is subtracted twice (see the inline comment) and the
    result is floored at zero. `exclude_user_ids` is how the caller keeps the bot's own wallet out
    of the pool it lends against.

    Args:
        session (AsyncSession): Active session to read through.
        exclude_user_ids (tuple[int, ...]): Wallets to leave out of the lending pool.

    Returns:
        The total pool, the outstanding minted principal, and what is still lendable.
    """
    balance_stmt = select(UserWallet.balance)
    if exclude_user_ids:
        balance_stmt = balance_stmt.where(UserWallet.user_id.notin_(other=exclude_user_ids))
    total_result = await session.execute(statement=balance_stmt)
    total_positive_user_balance = sum(
        balance for balance in total_result.scalars().all() if balance > 0
    )

    debt_result = await session.execute(
        statement=select(LoanContract.principal_remaining).where(
            LoanContract.lender_type == LoanLenderType.CENTRAL_BANK,
            LoanContract.status == LoanContractStatus.ACTIVE,
        )
    )
    outstanding_principal = sum(debt_result.scalars().all())
    # Central-bank loans mint into user balances, so subtract outstanding
    # principal once to estimate the pre-loan pool and once for already-used
    # capacity.
    base_lending_pool = max(total_positive_user_balance - outstanding_principal, 0)
    return CentralBankStatus(
        total_positive_user_balance=total_positive_user_balance,
        outstanding_principal=outstanding_principal,
        available_credit=max(base_lending_pool - outstanding_principal, 0),
    )


async def get_central_bank_status(exclude_user_ids: tuple[int, ...] = ()) -> CentralBankStatus:
    """Returns current central-bank lending capacity.

    A snapshot with no lock behind it, so it is an estimate for display; the acceptance path
    re-reads capacity under `_current_loan_accept_lock` before it mints anything.

    Args:
        exclude_user_ids (tuple[int, ...]): Wallets to leave out of the lending pool.

    Returns:
        The total pool, the outstanding minted principal, and what is still lendable.
    """
    await _ensure_schema()
    async with open_session() as session:
        return await _central_bank_status_in_session(
            session=session, exclude_user_ids=exclude_user_ids
        )


async def create_personal_loan_request(  # noqa: PLR0913 -- proposal needs both identities
    borrower_id: int,
    borrower_name: str,
    lender_id: int,
    lender_name: str,
    amount: int,
    monthly_rate_bps: int = DEFAULT_LOAN_MONTHLY_RATE_BPS,
    borrower_avatar_url: str = "",
    lender_avatar_url: str = "",
) -> LoanProposalView | None:
    """Creates a borrower-initiated personal loan request.

    Nothing moves yet: no escrow is taken and no balance changes, because a personal loan debits
    the lender at acceptance. The rate is clamped into the allowed band rather than rejected. The
    proposal expires on its own `LOAN_PROPOSAL_TIMEOUT_SECONDS` window, evaluated lazily.

    Args:
        borrower_id (int): Discord user ID asking to borrow.
        borrower_name (str): Last-seen borrower username, falling back to the numeric id.
        lender_id (int): Discord user ID being asked to lend.
        lender_name (str): Last-seen lender username, falling back to the numeric id.
        amount (int): Principal being requested.
        monthly_rate_bps (int): Monthly simple-interest rate in basis points.
        borrower_avatar_url (str): Last-seen Discord avatar URL for the borrower.
        lender_avatar_url (str): Last-seen Discord avatar URL for the lender.

    Returns:
        The pending proposal, or None for a non-positive amount or a self-directed request.
    """
    await _ensure_schema()
    if amount <= 0 or borrower_id == lender_id:
        return None
    now = _database_now()
    async with open_session() as session:
        proposal = LoanProposal(
            kind=LoanProposalKind.PERSONAL_REQUEST,
            status=LoanProposalStatus.PENDING,
            lender_type=LoanLenderType.USER,
            borrower_id=borrower_id,
            borrower_name=borrower_name or str(borrower_id),
            borrower_avatar_url=borrower_avatar_url,
            lender_id=lender_id,
            lender_name=lender_name or str(lender_id),
            lender_avatar_url=lender_avatar_url,
            creator_id=borrower_id,
            amount=amount,
            monthly_rate_bps=max(
                MIN_LOAN_MONTHLY_RATE_BPS, min(MAX_LOAN_MONTHLY_RATE_BPS, monthly_rate_bps)
            ),
            escrow_amount=0,
            created_at=now,
            updated_at=now,
        )
        session.add(proposal)
        await session.commit()
        return _loan_proposal_view(proposal=proposal)


async def create_central_bank_loan_request(
    borrower_id: int,
    borrower_name: str,
    amount: int,
    monthly_rate_bps: int = DEFAULT_LOAN_MONTHLY_RATE_BPS,
    borrower_avatar_url: str = "",
) -> LoanProposalView | None:
    """Creates a borrower-initiated central-bank loan request.

    There is no lender user, so `lender_id` is NULL and the display name is a fixed literal. The
    requested amount is not checked against lending capacity here; that happens under the
    acceptance lock, where the capacity read and the mint are one unit.

    Args:
        borrower_id (int): Discord user ID asking to borrow.
        borrower_name (str): Last-seen borrower username, falling back to the numeric id.
        amount (int): Principal being requested.
        monthly_rate_bps (int): Monthly simple-interest rate in basis points.
        borrower_avatar_url (str): Last-seen Discord avatar URL for the borrower.

    Returns:
        The pending proposal, or None for a non-positive amount.
    """
    await _ensure_schema()
    if amount <= 0:
        return None
    now = _database_now()
    async with open_session() as session:
        proposal = LoanProposal(
            kind=LoanProposalKind.CENTRAL_BANK_REQUEST,
            status=LoanProposalStatus.PENDING,
            lender_type=LoanLenderType.CENTRAL_BANK,
            borrower_id=borrower_id,
            borrower_name=borrower_name or str(borrower_id),
            borrower_avatar_url=borrower_avatar_url,
            lender_id=None,
            lender_name="Central Bank",
            lender_avatar_url="",
            creator_id=borrower_id,
            amount=amount,
            monthly_rate_bps=max(
                MIN_LOAN_MONTHLY_RATE_BPS, min(MAX_LOAN_MONTHLY_RATE_BPS, monthly_rate_bps)
            ),
            escrow_amount=0,
            created_at=now,
            updated_at=now,
        )
        session.add(proposal)
        await session.commit()
        return _loan_proposal_view(proposal=proposal)


async def _refund_proposal_escrow_in_session(
    session: AsyncSession, proposal: LoanProposal, now: datetime
) -> int | None:
    """Refunds escrowed proposal funds and returns the lender's balance.

    A no-op in practice: both creators write `escrow_amount=0`, since a personal loan debits the
    lender at acceptance rather than at proposal time. It stays on every reject / cancel / expire
    path so re-introducing escrow does not have to re-find them all.

    Args:
        session (AsyncSession): Active session; the write is left uncommitted.
        proposal (LoanProposal): The proposal whose escrow is being released.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        The lender's post-refund balance, or None when there was nothing to refund.
    """
    if proposal.escrow_amount <= 0 or proposal.lender_id is None:
        return None
    await _upsert_user_metadata_in_session(
        session=session,
        user_id=proposal.lender_id,
        name=proposal.lender_name,
        avatar_url=proposal.lender_avatar_url,
        now=now,
    )
    credit_result = await session.execute(
        statement=_build_credit_upsert(
            user_id=proposal.lender_id,
            name=proposal.lender_name,
            amount=proposal.escrow_amount,
            now=now,
        )
    )
    return credit_result.scalar_one()


async def reject_expired_loan_proposal(proposal_id: int) -> LoanProposalView | None:
    """Rejects a pending loan proposal if its decision window has expired.

    The explicit sweep a view calls when its timeout fires, so the stored proposal and the message
    the user is looking at agree. Idempotent: a proposal already decided, or not yet expired,
    leaves the transaction rolled back and answers None.

    Args:
        proposal_id (int): Row ID of the proposal to expire.

    Returns:
        The rejected proposal's view when this call expired it, else None.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        result = await session.execute(
            statement=select(LoanProposal).where(
                LoanProposal.id == proposal_id, LoanProposal.status == LoanProposalStatus.PENDING
            )
        )
        proposal = result.scalar_one_or_none()
        if proposal is None:
            return None
        expired = await _reject_expired_loan_proposal_in_session(
            session=session, proposal=proposal, now=now
        )
        if expired is None:
            await session.rollback()
            return None
        await session.commit()
        return expired


async def cancel_loan_proposal(proposal_id: int, actor_id: int) -> LoanProposalView | None:
    """Cancels a pending proposal created by `actor_id`.

    Only the creator may cancel, enforced in the SELECT's own predicate, so a proposal belonging to
    someone else is indistinguishable from a missing one. A proposal that turns out to have expired
    is committed as REJECTED and reported as None, so the caller never sees a cancel succeed on
    something the timeout already took.

    Args:
        proposal_id (int): Row ID of the proposal to cancel.
        actor_id (int): Discord user ID attempting the cancel; must be the creator.

    Returns:
        The canceled proposal's view, or None when it was missing, not the actor's, already
        decided, or expired instead.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        result = await session.execute(
            statement=select(LoanProposal).where(
                LoanProposal.id == proposal_id,
                LoanProposal.status == LoanProposalStatus.PENDING,
                LoanProposal.creator_id == actor_id,
            )
        )
        proposal = result.scalar_one_or_none()
        if proposal is None:
            return None
        expired = await _reject_expired_loan_proposal_in_session(
            session=session, proposal=proposal, now=now
        )
        if expired is not None:
            await session.commit()
            return None
        await _refund_proposal_escrow_in_session(session=session, proposal=proposal, now=now)
        status_result = await session.execute(
            statement=update(LoanProposal)
            .where(
                LoanProposal.id == proposal_id, LoanProposal.status == LoanProposalStatus.PENDING
            )
            .values(status=LoanProposalStatus.CANCELED, updated_at=now)
            .returning(LoanProposal.id)
        )
        if status_result.scalar_one_or_none() is None:
            await session.rollback()
            return None
        proposal.status = LoanProposalStatus.CANCELED
        await session.commit()
        return _loan_proposal_view(proposal=proposal)


async def reject_loan_proposal(
    proposal_id: int, actor_id: int, is_central_banker: bool = False
) -> LoanProposalView | None:
    """Rejects a pending proposal when `actor_id` is allowed to decide it.

    Who may decide depends on the kind: the named lender for a personal request, any central banker
    for a central-bank one. The caller supplies `is_central_banker` because the flag is read from
    the same table this transaction writes; a wrong actor is answered None with nothing written.

    Args:
        proposal_id (int): Row ID of the proposal to reject.
        actor_id (int): Discord user ID attempting the rejection.
        is_central_banker (bool): Whether the actor holds the central-banker flag.

    Returns:
        The rejected proposal's view, or None when it was missing, not the actor's to decide,
        already decided, or expired instead.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        result = await session.execute(
            statement=select(LoanProposal).where(
                LoanProposal.id == proposal_id, LoanProposal.status == LoanProposalStatus.PENDING
            )
        )
        proposal = result.scalar_one_or_none()
        if proposal is None:
            return None
        expired = await _reject_expired_loan_proposal_in_session(
            session=session, proposal=proposal, now=now
        )
        if expired is not None:
            await session.commit()
            return None
        allowed = False
        if proposal.kind == LoanProposalKind.PERSONAL_REQUEST:
            allowed = proposal.lender_id == actor_id
        elif proposal.kind == LoanProposalKind.CENTRAL_BANK_REQUEST:
            allowed = is_central_banker
        if not allowed:
            return None
        await _refund_proposal_escrow_in_session(session=session, proposal=proposal, now=now)
        status_result = await session.execute(
            statement=update(LoanProposal)
            .where(
                LoanProposal.id == proposal_id, LoanProposal.status == LoanProposalStatus.PENDING
            )
            .values(status=LoanProposalStatus.REJECTED, updated_at=now)
            .returning(LoanProposal.id)
        )
        if status_result.scalar_one_or_none() is None:
            await session.rollback()
            return None
        proposal.status = LoanProposalStatus.REJECTED
        await session.commit()
        return _loan_proposal_view(proposal=proposal)


async def accept_loan_proposal(  # noqa: PLR0913 -- approval needs proposal, actor, and central-bank policy
    proposal_id: int,
    actor_id: int,
    actor_name: str,
    actor_avatar_url: str = "",
    is_central_banker: bool = False,
    central_bank_exclude_user_ids: tuple[int, ...] = (),
    allow_central_bank_self_approval: bool = False,
) -> LoanProposalAcceptResult | None:
    """Accepts a pending loan proposal and opens the loan contract.

    The lock is the whole reason this wrapper exists: central-bank capacity is read and then spent,
    so two approvals running concurrently would both see the same free credit and mint past it.
    It is process-wide and loop-local, which bounds only this process; a second bot against the
    same file is not covered.

    Args:
        proposal_id (int): Row ID of the proposal to accept.
        actor_id (int): Discord user ID approving it.
        actor_name (str): Last-seen username to store for the approving lender.
        actor_avatar_url (str): Last-seen Discord avatar URL for the approver.
        is_central_banker (bool): Whether the actor holds the central-banker flag.
        central_bank_exclude_user_ids (tuple[int, ...]): Wallets to leave out of the lending pool.
        allow_central_bank_self_approval (bool): Whether a central banker may approve their own
            borrow request; keep this false in production.

    Returns:
        The opened contract with the post-write balances, or None when the proposal was missing,
        expired, not the actor's to decide, unaffordable, or beat by a concurrent decision.
    """
    await _ensure_schema()
    async with _current_loan_accept_lock():
        return await _accept_loan_proposal_locked(
            proposal_id=proposal_id,
            actor_id=actor_id,
            actor_name=actor_name,
            actor_avatar_url=actor_avatar_url,
            is_central_banker=is_central_banker,
            central_bank_exclude_user_ids=central_bank_exclude_user_ids,
            allow_central_bank_self_approval=allow_central_bank_self_approval,
        )


async def _accept_loan_proposal_locked(  # noqa: C901, PLR0911, PLR0913 -- proposal-kind branches must stay in one transaction
    proposal_id: int,
    actor_id: int,
    actor_name: str,
    actor_avatar_url: str = "",
    is_central_banker: bool = False,
    central_bank_exclude_user_ids: tuple[int, ...] = (),
    allow_central_bank_self_approval: bool = False,
) -> LoanProposalAcceptResult | None:
    """Accepts a loan proposal while the caller holds the acceptance lock.

    Opens with `BEGIN IMMEDIATE` so SQLite's write lock is held from the first read: the capacity
    and proposal state this decides on must not be re-readable by another writer before the mint
    lands. The two kinds settle differently and deliberately: a personal loan DEBITS the lender
    (a failed debit aborts the whole acceptance), while a central-bank loan MINTS with no lender
    side at all, which is why its capacity check is the only thing standing between it and
    inflation. Acceptance then prepays `MIN_INTEREST_DAYS` of interest and parks
    `last_interest_accrued_at` past that window, so an instant repayment still costs the borrower.
    Every rejection path returns None rather than raising, so the caller shows one refusal.

    Args:
        proposal_id (int): Row ID of the proposal to accept.
        actor_id (int): Discord user ID approving it.
        actor_name (str): Last-seen username to store for the approving lender.
        actor_avatar_url (str): Last-seen Discord avatar URL for the approver.
        is_central_banker (bool): Whether the actor holds the central-banker flag.
        central_bank_exclude_user_ids (tuple[int, ...]): Wallets to leave out of the lending pool.
        allow_central_bank_self_approval (bool): Whether a central banker may approve their own
            borrow request; keep this false in production.

    Returns:
        The opened contract with the post-write balances, or None on any refusal.
    """
    now = _database_now()
    async with open_session() as session:
        # Acquire SQLite's write lock before reading capacity or proposal state.
        await session.execute(statement=text("BEGIN IMMEDIATE"))
        result = await session.execute(
            statement=select(LoanProposal).where(
                LoanProposal.id == proposal_id, LoanProposal.status == LoanProposalStatus.PENDING
            )
        )
        proposal = result.scalar_one_or_none()
        if proposal is None:
            return None
        expired = await _reject_expired_loan_proposal_in_session(
            session=session, proposal=proposal, now=now
        )
        if expired is not None:
            await session.commit()
            return None

        lender_balance: int | None = None
        central_status: CentralBankStatus | None = None
        if proposal.kind == LoanProposalKind.PERSONAL_REQUEST:
            if proposal.lender_id != actor_id:
                return None
            await _upsert_user_metadata_in_session(
                session=session,
                user_id=actor_id,
                name=actor_name,
                avatar_url=actor_avatar_url,
                now=now,
            )
            debit_values: dict[str, Any] = {
                "name": actor_name or proposal.lender_name or str(actor_id),
                "balance": UserWallet.balance - proposal.amount,
                "total_spent": UserWallet.total_spent + proposal.amount,
                "updated_at": now,
            }
            debit_result = await session.execute(
                statement=update(UserWallet)
                .where(UserWallet.user_id == actor_id, UserWallet.balance >= proposal.amount)
                .values(**debit_values)
                .returning(UserWallet.balance)
            )
            lender_balance = debit_result.scalar_one_or_none()
            if lender_balance is None:
                await session.rollback()
                return None
            proposal.lender_name = actor_name or proposal.lender_name
            proposal.lender_avatar_url = actor_avatar_url or proposal.lender_avatar_url
        elif proposal.kind == LoanProposalKind.CENTRAL_BANK_REQUEST:
            if not is_central_banker:
                return None
            if proposal.borrower_id == actor_id and not allow_central_bank_self_approval:
                return None
            central_status = await _central_bank_status_in_session(
                session=session, exclude_user_ids=central_bank_exclude_user_ids
            )
            if central_status.available_credit < proposal.amount:
                return None
        else:
            return None

        status_result = await session.execute(
            statement=update(LoanProposal)
            .where(
                LoanProposal.id == proposal_id, LoanProposal.status == LoanProposalStatus.PENDING
            )
            .values(status=LoanProposalStatus.ACCEPTED, updated_at=now)
            .returning(LoanProposal.id)
        )
        if status_result.scalar_one_or_none() is None:
            await session.rollback()
            return None

        await _upsert_user_metadata_in_session(
            session=session,
            user_id=proposal.borrower_id,
            name=proposal.borrower_name,
            avatar_url=proposal.borrower_avatar_url,
            now=now,
        )
        credit_result = await session.execute(
            statement=_build_credit_upsert(
                user_id=proposal.borrower_id,
                name=proposal.borrower_name,
                amount=proposal.amount,
                now=now,
            )
        )
        borrower_balance = credit_result.scalar_one()
        invalidate_economy_leaderboard_cache()
        # Prepay MIN_INTEREST_DAYS of interest so borrowers cannot dodge interest
        # by repaying immediately. last_interest_accrued_at points past the
        # prepaid window, so _loan_interest_delta returns 0 until real time
        # catches up and then accrues normally.
        prepaid_interest = (
            proposal.amount * proposal.monthly_rate_bps * MIN_INTEREST_DAYS // (10_000 * 30)
        )
        prepaid_end = now + timedelta(days=MIN_INTEREST_DAYS)
        contract = LoanContract(
            proposal_id=proposal.id,
            lender_type=proposal.lender_type,
            lender_id=proposal.lender_id,
            lender_name=proposal.lender_name,
            lender_avatar_url=proposal.lender_avatar_url,
            borrower_id=proposal.borrower_id,
            borrower_name=proposal.borrower_name,
            borrower_avatar_url=proposal.borrower_avatar_url,
            original_principal=proposal.amount,
            principal_remaining=proposal.amount,
            interest_due=prepaid_interest,
            total_interest_paid=0,
            total_principal_paid=0,
            monthly_rate_bps=proposal.monthly_rate_bps,
            status=LoanContractStatus.ACTIVE,
            opened_at=now,
            last_interest_accrued_at=prepaid_end,
            updated_at=now,
        )
        session.add(contract)
        await session.commit()
        invalidate_economy_leaderboard_cache()
        if proposal.kind == LoanProposalKind.CENTRAL_BANK_REQUEST:
            central_status = await get_central_bank_status(
                exclude_user_ids=central_bank_exclude_user_ids
            )
        return LoanProposalAcceptResult(
            contract=_loan_contract_view(contract=contract),
            borrower_balance=borrower_balance,
            lender_balance=lender_balance,
            central_bank_available_credit=(
                central_status.available_credit if central_status is not None else None
            ),
        )


async def _loan_contracts_for_payment_in_session(
    session: AsyncSession,
    borrower_id: int,
    lender_type: LoanLenderType,
    lender_id: int | None = None,
) -> list[LoanContract]:
    """Returns active contracts in repayment priority order.

    Oldest first, with the row ID as the tiebreak, so a partial payment always retires the longest
    outstanding debt first and the order is stable across calls. `lender_id` narrows the set only
    for a personal loan; a central-bank query covers every such contract the borrower holds.

    Args:
        session (AsyncSession): Active session to read through.
        borrower_id (int): Discord user ID whose debts are being collected.
        lender_type (LoanLenderType): Which side of the ledger to repay.
        lender_id (int | None): The personal lender to narrow to; ignored for central bank.

    Returns:
        Active contracts, oldest first, empty when the borrower owes this lender nothing.
    """
    stmt = (
        select(LoanContract)
        .where(
            LoanContract.borrower_id == borrower_id,
            LoanContract.lender_type == lender_type,
            LoanContract.status == LoanContractStatus.ACTIVE,
        )
        .order_by(LoanContract.opened_at, LoanContract.id)
    )
    if lender_type == LoanLenderType.USER:
        stmt = stmt.where(LoanContract.lender_id == lender_id)
    result = await session.execute(statement=stmt)
    return list(result.scalars().all())


async def _apply_loan_payment_in_session(  # noqa: PLR0913 -- payment needs actor identity and contract set
    session: AsyncSession,
    contracts: Sequence[LoanContract],
    borrower_id: int,
    borrower_name: str,
    borrower_avatar_url: str,
    amount: int,
    now: datetime,
) -> LoanPaymentResult | None:
    """Applies a repayment or forced collection across ordered contracts.

    Interest is accrued per contract first, then paid before principal, so a payment that only
    covers the interest leaves the principal whole. Each contract's share is debited through the
    clamped path, so a borrower who cannot cover the full amount pays what they have and the loop
    stops there rather than failing, which is what makes forced collection partial rather than
    all-or-nothing. A personal lender is credited the collected amount in the same transaction; the
    central bank is not, since its principal was minted and repaying it burns.

    Returning None means nothing was collected at all, and the caller rolls back.

    Args:
        session (AsyncSession): Active session; the writes are left uncommitted.
        contracts (Sequence[LoanContract]): Active contracts in repayment priority order.
        borrower_id (int): Discord user ID being debited.
        borrower_name (str): Username to store, falling back to the contract's own copy.
        borrower_avatar_url (str): Avatar URL to store, falling back to the contract's own copy.
        amount (int): Total the borrower is willing (or forced) to pay across all contracts.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        What was paid, split into interest and principal, with the contracts this closed; None
        when no money moved.
    """
    if amount <= 0 or not contracts:
        return None

    amount_remaining = amount
    total_paid = 0
    total_interest_paid = 0
    total_principal_paid = 0
    borrower_balance = 0
    lender_balance: int | None = None
    closed_contract_ids: list[int] = []

    for contract in contracts:
        if amount_remaining <= 0:
            break
        await _accrue_contract_interest_in_session(session=session, contract=contract, now=now)
        owed = contract.interest_due + contract.principal_remaining
        if owed <= 0:
            continue
        requested = min(amount_remaining, owed)
        borrower_balance, applied_delta = await _apply_clamped_delta_in_session(
            session=session,
            user_id=borrower_id,
            name=borrower_name or contract.borrower_name,
            avatar_url=borrower_avatar_url or contract.borrower_avatar_url,
            delta=-requested,
            now=now,
        )
        paid = -applied_delta
        if paid <= 0:
            break

        interest_paid = min(paid, contract.interest_due)
        principal_paid = min(paid - interest_paid, contract.principal_remaining)
        contract.interest_due -= interest_paid
        contract.principal_remaining -= principal_paid
        contract.total_interest_paid += interest_paid
        contract.total_principal_paid += principal_paid
        contract.updated_at = now
        if contract.interest_due == 0 and contract.principal_remaining == 0:
            contract.status = LoanContractStatus.CLOSED
            contract.closed_at = now
            closed_contract_ids.append(contract.id)

        if contract.lender_type == LoanLenderType.USER and contract.lender_id is not None:
            await _upsert_user_metadata_in_session(
                session=session,
                user_id=contract.lender_id,
                name=contract.lender_name,
                avatar_url=contract.lender_avatar_url,
                now=now,
            )
            credit_result = await session.execute(
                statement=_build_credit_upsert(
                    user_id=contract.lender_id, name=contract.lender_name, amount=paid, now=now
                )
            )
            lender_balance = credit_result.scalar_one()
            # The borrower debit above already cleared the leaderboard cache for
            # this transaction, so the lender credit needs no extra invalidation.

        total_paid += paid
        total_interest_paid += interest_paid
        total_principal_paid += principal_paid
        amount_remaining -= paid
        if paid < requested:
            break

    if total_paid == 0:
        return None
    remaining_principal = sum(contract.principal_remaining for contract in contracts)
    remaining_interest = sum(contract.interest_due for contract in contracts)
    return LoanPaymentResult(
        paid_amount=total_paid,
        interest_paid=total_interest_paid,
        principal_paid=total_principal_paid,
        borrower_balance=borrower_balance,
        lender_balance=lender_balance,
        remaining_principal=remaining_principal,
        remaining_interest=remaining_interest,
        closed_contract_ids=tuple(closed_contract_ids),
    )


async def repay_personal_loans(
    borrower_id: int,
    borrower_name: str,
    lender_id: int,
    amount: int,
    borrower_avatar_url: str = "",
) -> LoanPaymentResult | None:
    """Repays active personal loans from `borrower_id` to `lender_id`.

    The borrower-initiated path: they name the amount, and it is spread over their contracts with
    that one lender, oldest first. The lender is credited what was actually collected.

    Args:
        borrower_id (int): Discord user ID repaying.
        borrower_name (str): Last-seen borrower username to store.
        lender_id (int): Discord user ID being repaid.
        amount (int): Total the borrower is paying across their contracts with this lender.
        borrower_avatar_url (str): Last-seen Discord avatar URL for the borrower.

    Returns:
        What was paid and what remains, or None when there was nothing to repay or the borrower
        could not cover any of it.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        contracts = await _loan_contracts_for_payment_in_session(
            session=session,
            borrower_id=borrower_id,
            lender_type=LoanLenderType.USER,
            lender_id=lender_id,
        )
        result = await _apply_loan_payment_in_session(
            session=session,
            contracts=contracts,
            borrower_id=borrower_id,
            borrower_name=borrower_name,
            borrower_avatar_url=borrower_avatar_url,
            amount=amount,
            now=now,
        )
        if result is None:
            await session.rollback()
            return None
        await session.commit()
        invalidate_economy_leaderboard_cache()
        return result


async def call_personal_loans(
    lender_id: int,
    borrower_id: int,
    borrower_name: str,
    amount: int | None = None,
    borrower_avatar_url: str = "",
) -> LoanPaymentResult | None:
    """Forcibly collects active personal loans owed to `lender_id`.

    The lender-initiated mirror of `repay_personal_loans`. Interest is accrued across every
    contract BEFORE the total owed is worked out, so `amount=None` collects a figure that includes
    interest earned right up to now. Collection is partial by design: a borrower who cannot cover
    it pays what they have and keeps the remainder as debt.

    Args:
        lender_id (int): Discord user ID collecting.
        borrower_id (int): Discord user ID being collected from.
        borrower_name (str): Last-seen borrower username to store.
        amount (int | None): Amount to collect, or None for everything owed.
        borrower_avatar_url (str): Last-seen Discord avatar URL for the borrower.

    Returns:
        What was collected and what remains, or None when there was nothing owed or nothing
        could be collected.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        contracts = await _loan_contracts_for_payment_in_session(
            session=session,
            borrower_id=borrower_id,
            lender_type=LoanLenderType.USER,
            lender_id=lender_id,
        )
        for contract in contracts:
            await _accrue_contract_interest_in_session(session=session, contract=contract, now=now)
        total_owed = sum(
            contract.principal_remaining + contract.interest_due for contract in contracts
        )
        payment_amount = amount if amount is not None else max(total_owed, 1)
        result = await _apply_loan_payment_in_session(
            session=session,
            contracts=contracts,
            borrower_id=borrower_id,
            borrower_name=borrower_name,
            borrower_avatar_url=borrower_avatar_url,
            amount=payment_amount,
            now=now,
        )
        if result is None:
            await session.rollback()
            return None
        await session.commit()
        invalidate_economy_leaderboard_cache()
        return result


async def repay_central_bank_loans(
    borrower_id: int, borrower_name: str, amount: int, borrower_avatar_url: str = ""
) -> LoanPaymentResult | None:
    """Repays active central-bank loans for a borrower.

    The repaid points are burned: there is no lender wallet to credit, which is the other half of
    the mint at approval and what keeps central-bank lending inflation-neutral over a loan's life.

    Args:
        borrower_id (int): Discord user ID repaying.
        borrower_name (str): Last-seen borrower username to store.
        amount (int): Total the borrower is paying across their central-bank contracts.
        borrower_avatar_url (str): Last-seen Discord avatar URL for the borrower.

    Returns:
        What was paid and what remains, or None when there was nothing to repay or the borrower
        could not cover any of it.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        contracts = await _loan_contracts_for_payment_in_session(
            session=session, borrower_id=borrower_id, lender_type=LoanLenderType.CENTRAL_BANK
        )
        result = await _apply_loan_payment_in_session(
            session=session,
            contracts=contracts,
            borrower_id=borrower_id,
            borrower_name=borrower_name,
            borrower_avatar_url=borrower_avatar_url,
            amount=amount,
            now=now,
        )
        if result is None:
            await session.rollback()
            return None
        await session.commit()
        invalidate_economy_leaderboard_cache()
        return result


async def call_central_bank_loans(
    borrower_id: int, borrower_name: str, amount: int | None = None, borrower_avatar_url: str = ""
) -> LoanPaymentResult | None:
    """Forcibly collects active central-bank loans from a borrower.

    The central banker's collection command. Like the personal mirror, interest is accrued across
    every contract before the owed total is computed, so `amount=None` collects everything owed as
    of now; the collected points are burned rather than credited to anyone.

    Args:
        borrower_id (int): Discord user ID being collected from.
        borrower_name (str): Last-seen borrower username to store.
        amount (int | None): Amount to collect, or None for everything owed.
        borrower_avatar_url (str): Last-seen Discord avatar URL for the borrower.

    Returns:
        What was collected and what remains, or None when there was nothing owed or nothing
        could be collected.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        contracts = await _loan_contracts_for_payment_in_session(
            session=session, borrower_id=borrower_id, lender_type=LoanLenderType.CENTRAL_BANK
        )
        for contract in contracts:
            await _accrue_contract_interest_in_session(session=session, contract=contract, now=now)
        total_owed = sum(
            contract.principal_remaining + contract.interest_due for contract in contracts
        )
        payment_amount = amount if amount is not None else max(total_owed, 1)
        result = await _apply_loan_payment_in_session(
            session=session,
            contracts=contracts,
            borrower_id=borrower_id,
            borrower_name=borrower_name,
            borrower_avatar_url=borrower_avatar_url,
            amount=payment_amount,
            now=now,
        )
        if result is None:
            await session.rollback()
            return None
        await session.commit()
        invalidate_economy_leaderboard_cache()
        return result


async def list_loan_contracts(
    user_id: int, include_closed: bool = False
) -> list[LoanContractView]:
    """Lists loan contracts where the user is borrower or personal lender.

    Accrues and persists interest-due on active contracts first (a write),
    matching `get_portfolio`'s lazy-accrual behavior, so the returned views
    reflect interest owed up to now.

    Both sides of a personal loan are matched, so one call answers "what do I owe" and "what am I
    owed" together; a central-bank contract only ever appears for its borrower.

    Args:
        user_id (int): Discord user ID to list contracts for.
        include_closed (bool): Whether contracts already repaid in full are included.

    Returns:
        Contract views, oldest first, with interest accrued up to now.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        stmt = select(LoanContract).where(
            (LoanContract.borrower_id == user_id) | (LoanContract.lender_id == user_id)
        )
        if not include_closed:
            stmt = stmt.where(LoanContract.status == LoanContractStatus.ACTIVE)
        stmt = stmt.order_by(LoanContract.opened_at, LoanContract.id)
        result = await session.execute(statement=stmt)
        contracts = list(result.scalars().all())
        for contract in contracts:
            await _accrue_contract_interest_in_session(session=session, contract=contract, now=now)
        await session.commit()
        return [_loan_contract_view(contract=contract) for contract in contracts]


async def _portfolio_in_session(
    session: AsyncSession, user_id: int, now: datetime
) -> PortfolioView:
    """Builds a portfolio view, accruing active debt interest first.

    Net worth is wallet minus principal minus accrued interest, so it can go negative and the
    caller must be ready for that. Debt is counted from the borrower side only; money lent out is
    not an asset here. A user with no account row reads back as the numeric id at zero.

    Args:
        session (AsyncSession): Active session; the accrual is left uncommitted.
        user_id (int): Discord user ID to build the portfolio for.
        now (datetime): `_database_now()` value pinned for this transaction.

    Returns:
        Wallet balance, outstanding principal and interest, and the resulting net worth.
    """
    account_result = await session.execute(
        statement=select(UserAccount.name, UserWallet.balance)
        .select_from(UserAccount)
        .outerjoin(UserWallet, UserWallet.user_id == UserAccount.user_id)
        .where(UserAccount.user_id == user_id)
    )
    account_row = account_result.one_or_none()
    name = str(user_id)
    balance = 0
    if account_row is not None:
        name = account_row[0]
        balance = account_row[1] or 0

    debt_result = await session.execute(
        statement=select(LoanContract).where(
            LoanContract.borrower_id == user_id, LoanContract.status == LoanContractStatus.ACTIVE
        )
    )
    debt_contracts = list(debt_result.scalars().all())
    for contract in debt_contracts:
        await _accrue_contract_interest_in_session(session=session, contract=contract, now=now)
    debt_principal = sum(contract.principal_remaining for contract in debt_contracts)
    debt_interest = sum(contract.interest_due for contract in debt_contracts)

    return PortfolioView(
        user_id=user_id,
        name=name,
        balance=balance,
        debt_principal=debt_principal,
        debt_interest=debt_interest,
        net_worth=balance - debt_principal - debt_interest,
    )


async def get_portfolio(user_id: int) -> PortfolioView:
    """Returns a user's current portfolio and estimated net worth.

    What `/balance` shows, and a WRITE despite the name: the interest accrual it runs is committed
    here, so reading a portfolio is what keeps a debt figure honest.

    Args:
        user_id (int): Discord user ID to build the portfolio for.

    Returns:
        Wallet balance, outstanding principal and interest, and the resulting net worth.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        portfolio = await _portfolio_in_session(session=session, user_id=user_id, now=now)
        await session.commit()
        return portfolio
