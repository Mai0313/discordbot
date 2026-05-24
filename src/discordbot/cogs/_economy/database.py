"""Persistent point-balance store for the economy cog.

The engine is a module-level `AsyncEngine` singleton. Putting
`create_async_engine()` on a per-instance `cached_property` would leak the
connection pool, dialect cache, and inspector cache for every Discord
interaction (the same lesson `cogs/log_msg.py` captures for the sync engine
it still uses for pandas `to_sql`).

Every balance-mutating write path is atomic at the SQLite transaction level.
Most paths are a single UPSERT (`INSERT ... ON CONFLICT DO UPDATE`) or a
conditional `UPDATE ... WHERE ... RETURNING`; multi-row finance paths still roll
back as one unit when a conditional write loses a race. The previous
implementation read the row in Python, mutated `account.balance`, and
committed; two coroutines racing on the same user would lose updates, and two
coroutines racing on a brand-new user would both `INSERT` and one would raise
`IntegrityError`.

PRAGMA setup at connect-time enables WAL (so reads don't block on writes),
sets a tolerant `busy_timeout`, and picks `synchronous=NORMAL` (the right
durability trade-off in WAL: every commit fsyncs the WAL frame, and the
main file is fsynced on checkpoint).

We use `aiosqlite` so every DB call stays on the event loop: no
`asyncio.to_thread` shim, no separate `_*_sync` helpers. Each operation
opens an `AsyncSession` bound to the current `_engine`, so tests can
monkeypatch `_engine` per-test and every subsequent call sees the swap.

VIP, admin status, and leaderboard visibility are boolean columns on
`user_account`. VIP bumps daily check-in rewards and the player's winning
payout from games. The flag is permanent once set. Admin and central-banker
status gate maintenance-only economy commands and are managed out-of-band by
scripts. Daily casino counters live on `casino_account` so
`/loss_leaderboard` can read current-day gross losses without scanning an audit
log.

Long-term lending lives in `loan_proposal` and `loan_contract`. Personal
loan requests debit the lender on acceptance, and central-bank loans mint
borrower balance on approval.

Shared jackpot pools live in `data/global_state.db` because they are bot-wide
state, not per-user economy rows. Runtime jackpot settlement coordinates writes
across the economy and global-state DB sessions and rolls both back on ordinary
errors before either commit; SQLite still cannot make a hard crash between two
database-file commits fully atomic.
"""

from typing import Any, Final, cast
import asyncio
from datetime import datetime, timezone, timedelta
from collections.abc import Sequence

from sqlalchemy import (
    Text,
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
from sqlalchemy.types import TypeDecorator
from sqlalchemy.sql.dml import ReturningInsert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.dialects.sqlite import insert

from discordbot.typings.economy import (
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
    LeaderboardEntry,
    LoanContractView,
    LoanProposalKind,
    LoanProposalView,
    CentralBankStatus,
    LoanPaymentResult,
    VipPurchaseResult,
    LoanContractStatus,
    LoanProposalStatus,
    CentralBankerAccount,
    LossLeaderboardEntry,
    BalanceAdjustmentResult,
    JackpotSettlementResult,
    JackpotSettlementRequest,
    LoanProposalAcceptResult,
    OrderedWalletDeltaResult,
    JackpotSettlementBatchResult,
)

# SELECT-then-conditional-UPDATE loops keep a small retry budget. With WAL +
# busy_timeout, contention is rare and resolves on the first or second retry;
# the bound prevents a degenerate hot-row livelock.
_CHECKIN_MAX_RETRIES: Final[int] = 8
_VIP_PURCHASE_MAX_RETRIES: Final[int] = 8
_CLAMPED_DELTA_MAX_RETRIES: Final[int] = 8
_JACKPOT_CLAIM_MAX_RETRIES: Final[int] = 8
# Blackjack VIP perk: 1.5x payout on winning rounds, applied as floor(delta * 3 / 2).
_VIP_WIN_MULTIPLIER_NUM: Final[int] = 3
_VIP_WIN_MULTIPLIER_DEN: Final[int] = 2
TAIWAN_TIMEZONE: Final[timezone] = timezone(offset=timedelta(hours=8), name="Asia/Taipei")

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/economy.db")
_global_state_engine: AsyncEngine = create_async_engine(
    url="sqlite+aiosqlite:///data/global_state.db"
)


def _database_now() -> datetime:
    """Returns the wall-clock timestamp used for persisted economy rows."""
    return datetime.now(tz=TAIWAN_TIMEZONE)


def _as_taipei(dt: datetime) -> datetime:
    """Returns `dt` re-interpreted in Asia/Taipei (treating naive as Taipei)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TAIWAN_TIMEZONE)
    return dt.astimezone(tz=TAIWAN_TIMEZONE)


def _taipei_midnight(now: datetime) -> datetime:
    """Returns the most recent Asia/Taipei 00:00 boundary at or before `now`."""
    local = _as_taipei(dt=now)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def _stored_int_to_int(value: object) -> int:
    """Parses a persisted decimal-string integer into a Python int."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        return _stored_int_to_int(value=value.decode())
    if isinstance(value, str):
        normalized = value.strip()
        return int(normalized or "0")
    msg = f"Unsupported stored integer type: {type(value)!r}"
    raise TypeError(msg)


def _stored_int_to_text(value: int) -> str:
    """Returns canonical decimal text for a persisted integer."""
    return str(value)


def _sqlite_int_add_text(left: Any, right: Any) -> str:  # noqa: ANN401 -- SQLite UDF inputs can be any scalar type
    """Adds two persisted integers and returns canonical decimal text."""
    return _stored_int_to_text(
        value=_stored_int_to_int(value=left) + _stored_int_to_int(value=right)
    )


def _sqlite_int_compare_text(left: Any, right: Any) -> int:  # noqa: ANN401 -- SQLite UDF inputs can be any scalar type
    """Compares two persisted integers for SQLite predicates."""
    left_int = _stored_int_to_int(value=left)
    right_int = _stored_int_to_int(value=right)
    return (left_int > right_int) - (left_int < right_int)


def _int_add_text(column: ColumnElement[Any], delta: int) -> ColumnElement[Any]:
    """Builds a SQLite expression that adds `delta` to a decimal-text column."""
    return cast(
        "ColumnElement[Any]",
        func.discordbot_int_add_text(column, _stored_int_to_text(value=delta)),
    )


def _int_compare_text(column: ColumnElement[Any], value: int) -> ColumnElement[int]:
    """Builds a SQLite expression that compares a decimal-text column."""
    return cast(
        "ColumnElement[int]",
        func.discordbot_int_compare_text(column, _stored_int_to_text(value=value)),
    )


class StoredIntegerComparator(TypeDecorator.Comparator[int]):
    """Routes SQL arithmetic and comparisons through integer-aware UDFs."""

    def __add__(self, other: object) -> ColumnElement[Any]:
        return _int_add_text(
            column=cast("ColumnElement[Any]", self.expr), delta=_stored_int_to_int(value=other)
        )

    def __sub__(self, other: object) -> ColumnElement[Any]:
        return _int_add_text(
            column=cast("ColumnElement[Any]", self.expr), delta=-_stored_int_to_int(value=other)
        )

    def __gt__(self, other: object) -> ColumnElement[bool]:
        return cast(
            "ColumnElement[bool]",
            _int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=_stored_int_to_int(value=other)
            )
            > 0,
        )

    def __ge__(self, other: object) -> ColumnElement[bool]:
        return cast(
            "ColumnElement[bool]",
            _int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=_stored_int_to_int(value=other)
            )
            >= 0,
        )

    def __lt__(self, other: object) -> ColumnElement[bool]:
        return cast(
            "ColumnElement[bool]",
            _int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=_stored_int_to_int(value=other)
            )
            < 0,
        )

    def __le__(self, other: object) -> ColumnElement[bool]:
        return cast(
            "ColumnElement[bool]",
            _int_compare_text(
                column=cast("ColumnElement[Any]", self.expr), value=_stored_int_to_int(value=other)
            )
            <= 0,
        )


class StoredInteger(TypeDecorator[int]):
    """Persists Python integers as decimal text in SQLite."""

    impl = Text
    cache_ok = True
    comparator_factory = StoredIntegerComparator

    def process_bind_param(self, value: object | None, dialect: Any) -> str:  # noqa: ANN401 -- SQLAlchemy hook signature
        """Converts a Python integer into canonical decimal text."""
        return _stored_int_to_text(value=_stored_int_to_int(value=value))

    def process_result_value(self, value: object | None, dialect: Any) -> int:  # noqa: ANN401 -- SQLAlchemy hook signature
        """Converts persisted decimal text into a Python integer."""
        return _stored_int_to_int(value=value)


def _configure_sqlite_connection(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Sets WAL mode + a tolerant busy_timeout on every new connection.

    WAL flips the read/write lock so readers never block on writes;
    `synchronous=NORMAL` is the right durability trade-off in WAL (every
    commit fsyncs the WAL frame, the main file is fsynced on checkpoint);
    `busy_timeout` gives the writer 5 s to wait under contention; foreign
    keys are enabled defensively for any future FK constraint.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    dbapi_connection.create_function("discordbot_int_add_text", 2, _sqlite_int_add_text)
    dbapi_connection.create_function("discordbot_int_compare_text", 2, _sqlite_int_compare_text)
    cursor.close()


@event.listens_for(_engine.sync_engine, "connect")
@event.listens_for(_global_state_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Configures a newly opened SQLite connection."""
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _configure_sqlite_on_checkout(
    dbapi_connection: object, _connection_record: object, _connection_proxy: object
) -> None:
    """Configures pooled connections from test-swapped engines."""
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _ensure_sqlite_hooks(engine: AsyncEngine) -> None:
    """Installs SQLite connection hooks on the active engine."""
    if not event.contains(target=engine.sync_engine, identifier="connect", fn=_configure_sqlite):
        event.listen(target=engine.sync_engine, identifier="connect", fn=_configure_sqlite)
    if not event.contains(
        target=engine.sync_engine, identifier="checkout", fn=_configure_sqlite_on_checkout
    ):
        event.listen(
            target=engine.sync_engine, identifier="checkout", fn=_configure_sqlite_on_checkout
        )


class Base(DeclarativeBase):
    """Base class for economy ORM models."""

    pass


class GlobalStateBase(DeclarativeBase):
    """Base class for bot-wide global state ORM models."""

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
    """Spendable balance and lifetime gross totals for a Discord user."""

    __tablename__ = "user_wallet"
    __table_args__ = (
        # /leaderboard does ORDER BY balance DESC LIMIT 10; the index turns a
        # full scan into a bounded walk. SQLite can read an ASC index backwards
        # to satisfy ORDER BY DESC.
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

    Attributes:
        user_id: Discord user ID; primary key.
        name: Last-seen Discord username for quick inspection.
        day_started_at: Asia/Taipei midnight for the stored counters.
        daily_loss: Current-day gross loss from player-side casino settlements, stored as a decimal string.
        daily_win: Current-day gross win from player-side casino settlements, stored as a decimal string.
        daily_net: Current-day signed net casino result, stored as a decimal string.
        updated_at: Taiwan-local timestamp of the last casino counter write.
    """

    __tablename__ = "casino_account"
    __table_args__ = (
        # /loss_leaderboard filters to one Taipei day and orders by gross loss.
        # SQLite can read the daily loss suffix backwards for DESC ordering.
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
    """Active or closed long-term loan contract."""

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


class JackpotPool(GlobalStateBase):
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


# On-the-house seed amount for each registered jackpot pool. The seed is
# bookkeeping only — the bot's user_account row is never decremented to fund
# it, so /house P&L stays unaffected by the donation. Seeded pools are also
# topped back up to this amount whenever they are drained.
_JACKPOT_SEEDS: Final[tuple[tuple[str, int], ...]] = (("dragon_gate", 100_000),)


def _jackpot_seed_amount(game_id: str) -> int:
    """Returns the configured seed amount for a jackpot game."""
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
_global_state_schema_ready_for: AsyncEngine | None = None
_schema_lock: asyncio.Lock | None = None
_schema_lock_loop: asyncio.AbstractEventLoop | None = None
_global_state_schema_lock: asyncio.Lock | None = None
_global_state_schema_lock_loop: asyncio.AbstractEventLoop | None = None
_loan_accept_lock: asyncio.Lock | None = None
_loan_accept_lock_loop: asyncio.AbstractEventLoop | None = None


def _current_schema_lock() -> asyncio.Lock:
    """Returns a schema bootstrap lock bound to the current event loop."""
    global _schema_lock, _schema_lock_loop  # noqa: PLW0603 -- module-level loop-local lock
    loop = asyncio.get_running_loop()
    if _schema_lock is None or _schema_lock_loop is not loop:
        _schema_lock = asyncio.Lock()
        _schema_lock_loop = loop
    return _schema_lock


def _current_global_state_schema_lock() -> asyncio.Lock:
    """Returns a global-state schema bootstrap lock bound to the current event loop."""
    global _global_state_schema_lock, _global_state_schema_lock_loop  # noqa: PLW0603 -- module-level loop-local lock
    loop = asyncio.get_running_loop()
    if _global_state_schema_lock is None or _global_state_schema_lock_loop is not loop:
        _global_state_schema_lock = asyncio.Lock()
        _global_state_schema_lock_loop = loop
    return _global_state_schema_lock


def _current_loan_accept_lock() -> asyncio.Lock:
    """Serializes loan approval so central-bank capacity is consumed once."""
    global _loan_accept_lock, _loan_accept_lock_loop  # noqa: PLW0603 -- module-level loop-local lock
    loop = asyncio.get_running_loop()
    if _loan_accept_lock is None or _loan_accept_lock_loop is not loop:
        _loan_accept_lock = asyncio.Lock()
        _loan_accept_lock_loop = loop
    return _loan_accept_lock


async def _ensure_global_state_schema() -> None:
    """Bootstraps bot-wide state in `data/global_state.db`."""
    global _global_state_schema_ready_for  # noqa: PLW0603 -- module-level cache by engine identity
    _ensure_sqlite_hooks(engine=_global_state_engine)
    if _global_state_schema_ready_for is _global_state_engine:
        return
    async with _current_global_state_schema_lock():
        if _global_state_schema_ready_for is _global_state_engine:
            return
        async with _global_state_engine.begin() as conn:
            await conn.run_sync(GlobalStateBase.metadata.create_all)
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
        _global_state_schema_ready_for = _global_state_engine


async def _ensure_schema() -> None:
    """Bootstraps current economy and bot-wide state schemas once per engine."""
    global _schema_ready_for  # noqa: PLW0603 -- module-level cache by engine identity
    _ensure_sqlite_hooks(engine=_engine)
    if _schema_ready_for is _engine:
        return
    async with _current_schema_lock():
        if _schema_ready_for is _engine:
            return
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await _ensure_global_state_schema()
        _schema_ready_for = _engine


def open_session() -> AsyncSession:
    """Creates an async session bound to the current economy database engine.

    Returns:
        An `AsyncSession` using the current module-level `_engine`.
    """
    _ensure_sqlite_hooks(engine=_engine)
    return AsyncSession(bind=_engine, expire_on_commit=False)


def open_global_state_session() -> AsyncSession:
    """Creates an async session bound to the bot-wide global state DB."""
    _ensure_sqlite_hooks(engine=_global_state_engine)
    return AsyncSession(bind=_global_state_engine, expire_on_commit=False)


def checkin_reward(streak: int, is_vip: bool) -> int:
    """Returns the gross check-in payout for a streak day.

    The reward formula is `BASE * (1 + (streak - 1) * 0.5)` where `streak`
    is the 1..`CHECKIN_STREAK_CYCLE` day in the cycle. VIP doubles the base
    before the streak bonus.

    Args:
        streak: Streak counter for this check-in (1..`CHECKIN_STREAK_CYCLE`).
        is_vip: VIP status of the account at check-in time.

    Returns:
        Integer reward amount.
    """
    base = BASE_CHECKIN_REWARD_AMOUNT * (2 if is_vip else 1)
    multiplier = 1.0 + (streak - 1) * 0.5
    return int(base * multiplier)


def monthly_rate_percent_to_bps(monthly_rate_percent: float) -> int:
    """Converts a user-facing monthly percent into basis points."""
    return max(
        MIN_LOAN_MONTHLY_RATE_BPS,
        min(MAX_LOAN_MONTHLY_RATE_BPS, round(monthly_rate_percent * 100)),
    )


def monthly_rate_bps_to_percent(monthly_rate_bps: int) -> float:
    """Converts stored monthly basis points into a display percent."""
    return monthly_rate_bps / 100


def apply_vip_blackjack_bonus(delta: int, is_vip: bool) -> int:
    """Applies the VIP 1.5x payout multiplier on a winning player delta.

    The bonus only fires on positive deltas (wins). Pushes and losses pass
    through unchanged so VIP never softens a loss.

    Args:
        delta: Pre-bonus player delta for the round.
        is_vip: VIP status of the account at settlement time.

    Returns:
        Post-bonus player delta.
    """
    if not is_vip or delta <= 0:
        return delta
    return delta * _VIP_WIN_MULTIPLIER_NUM // _VIP_WIN_MULTIPLIER_DEN


async def _upsert_user_metadata_in_session(
    session: AsyncSession, user_id: int, name: str, avatar_url: str, now: datetime
) -> None:
    """Creates or refreshes the user identity row without touching wallet state."""
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
    """UPSERT that credits `amount` points into `user_wallet`.

    Caller guarantees `amount > 0` and refreshes `user_account` metadata
    separately.

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
    """UPSERT applying a signed `delta` with NO clamp on wallet balance.

    Used for the dealer's house-ledger row, which is allowed to go negative
    when the casino has paid out more than it took in. `total_earned` /
    `total_spent` still accumulate gross flows so `/house` can show the
    direction of the volume, not just the net.

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
    """Accumulates current-day gross casino counters in `casino_account`."""
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


async def _credit_with_repayment_in_session(  # noqa: PLR0913 -- session helper keeps income writes atomic
    session: AsyncSession, user_id: int, name: str, avatar_url: str, amount: int, now: datetime
) -> CreditResult:
    """Credits income inside the caller's transaction.

    Long-term loans are explicit repayment actions now, so passive income does
    not auto-repay debt. The public function name is preserved because message
    and chat reward callers are intentionally routed through one income facade.
    Caller must guarantee `amount > 0`.
    """
    await _upsert_user_metadata_in_session(
        session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
    )
    result = await session.execute(
        statement=_build_credit_upsert(user_id=user_id, name=name, amount=amount, now=now)
    )
    return CreditResult(
        new_balance=result.scalar_one(),
        credited_amount=amount,
        principal_repaid=0,
        remaining_debt=0,
    )


async def _apply_clamped_delta_in_session(  # noqa: PLR0913 -- session helper needs identity and delta state
    session: AsyncSession, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
) -> tuple[int, int]:
    """Applies a clamped signed delta and returns the balance plus applied delta.

    The observed balance is pinned in the UPDATE predicate, so concurrent
    clamped debits cannot both compute their applied delta from the same stale
    balance. A negative delta against a missing row is a no-op so manual clamp
    operations do not create zero-balance accounts.
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
    """Attempts to create a missing account for a positive clamped delta."""
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
    return inserted_balance, delta


async def _try_update_clamped_delta_in_session(  # noqa: PLR0913 -- conditional write needs observed row state
    session: AsyncSession, user_id: int, name: str, current_balance: int, delta: int, now: datetime
) -> tuple[int, int] | None:
    """Attempts one conditional clamped update against an existing account."""
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
    return new_balance, applied


async def _apply_signed_delta_in_session(  # noqa: PLR0913 -- session helper needs identity and signed delta
    session: AsyncSession, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
) -> int:
    """Applies a signed delta without clamping.

    Used for dealer-side mirrors (`HOUSE_SETTLE`), which may run cumulative
    negative P&L. Player-side losses use the clamped path instead.
    """
    await _upsert_user_metadata_in_session(
        session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
    )
    stmt = _build_signed_delta_upsert(user_id=user_id, name=name, delta=delta, now=now)
    result = await session.execute(statement=stmt)
    return result.scalar_one()


async def _apply_player_delta_in_session(  # noqa: PLR0913 -- player settlement needs identity and audit metadata
    session: AsyncSession, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
) -> tuple[int, int]:
    """Applies a casino player delta and returns the balance plus actual delta."""
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

    Args:
        user_id: Discord user ID receiving the credit.
        name: Last-seen Discord username to store on the account.
        amount: Gross income amount; must be positive for the repayment
            path to run.
        avatar_url: Last-seen Discord avatar URL to store when available.

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
        return result


async def adjust_balance(
    user_id: int, name: str, delta: int, allow_negative: bool = False, avatar_url: str = ""
) -> BalanceAdjustmentResult:
    """Applies an explicit manual balance adjustment.

    This is the public maintenance API for scripts and admin tooling. It does
    does not touch loan contracts or daily casino counters, so leaderboards and
    house P&L remain clean.

    Args:
        user_id: Discord user ID whose balance should be adjusted.
        name: Last-seen Discord username to store on the account.
        delta: Signed amount to apply.
        allow_negative: Whether the resulting balance may go below zero.
        avatar_url: Last-seen Discord avatar URL to store when available.

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

    Args:
        user_id: Discord user ID whose wallet should be updated.
        name: Last-seen Discord username to store on the wallet row.
        deltas: Ordered signed wallet legs.
        avatar_url: Last-seen Discord avatar URL to store when available.

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
        return OrderedWalletDeltaResult(new_balance=balance, applied_deltas=tuple(applied))


async def _apply_ordered_wallet_deltas_in_session(  # noqa: PLR0913 -- session helper carries identity and output accumulator
    session: AsyncSession,
    user_id: int,
    name: str,
    deltas: Sequence[WalletDeltaLeg],
    now: datetime,
    applied: list[int],
) -> int | None:
    """Applies ordered wallet legs inside the caller's economy transaction."""
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
    return balance


async def apply_round_settlement(  # noqa: PLR0913 -- atomic settlement needs both ledger keys
    player_id: int,
    player_account_name: str,
    player_delta: int,
    dealer_id: int,
    dealer_name: str,
    dealer_delta: int,
    player_avatar_url: str = "",
    dealer_avatar_url: str = "",
) -> tuple[int, int]:
    """Applies a finished round's net delta and mirrors house P&L atomically.

    Sharing a session (and therefore a single SQLite transaction) means a
    crash between the player and dealer writes cannot leave the dealer
    ledger drifting from the player result. Positive player deltas go through
    the shared income path. Negative player deltas clamp at zero; when a loss
    cannot be fully collected, the dealer ledger only records the actual
    collected debit.

    Args:
        player_id: Discord user ID for the player account.
        player_account_name: Account name to store for the player.
        player_avatar_url: Last-seen Discord avatar URL for the player.
        player_delta: Signed net change for the player. Losses are clamped at
            zero and may apply less than the requested debit.
        dealer_id: Discord user ID for the dealer ledger row.
        dealer_name: Account name to store for the dealer ledger row.
        dealer_avatar_url: Last-seen Discord avatar URL for the dealer.
        dealer_delta: Signed change to apply to the dealer ledger balance.

    Returns:
        A `(player_balance_after, dealer_balance_after)` tuple.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        player_balance, applied_player_delta = await _apply_player_delta_in_session(
            session=session,
            user_id=player_id,
            name=player_account_name,
            avatar_url=player_avatar_url,
            delta=player_delta,
            now=now,
        )

        dealer_delta_to_apply = dealer_delta
        if player_delta < 0 and dealer_delta > 0:
            dealer_delta_to_apply = min(dealer_delta, max(-applied_player_delta, 0))

        if dealer_delta_to_apply == 0:
            dealer_result = await session.execute(
                statement=select(UserWallet.balance).where(UserWallet.user_id == dealer_id)
            )
            dealer_balance = dealer_result.scalar_one_or_none() or 0
        else:
            dealer_balance = await _apply_signed_delta_in_session(
                session=session,
                user_id=dealer_id,
                name=dealer_name,
                avatar_url=dealer_avatar_url,
                delta=dealer_delta_to_apply,
                now=now,
            )
        await session.commit()
        return player_balance, dealer_balance


async def apply_blackjack_settlement(  # noqa: PLR0913 -- atomic settlement needs both ledger keys
    player_id: int,
    player_account_name: str,
    player_delta: int,
    dealer_id: int,
    dealer_name: str,
    dealer_delta: int,
    player_avatar_url: str = "",
    dealer_avatar_url: str = "",
) -> tuple[int, int]:
    """Applies Blackjack player payout and dealer ledger deltas atomically.

    Blackjack can include system-funded bonuses that should credit the
    player and count as casino payout, but must not move the `/house`
    ledger. This wrapper keeps the one-transaction write path while making
    the independent dealer-side delta explicit at the call site.
    """
    return await apply_round_settlement(
        player_id=player_id,
        player_account_name=player_account_name,
        player_avatar_url=player_avatar_url,
        player_delta=player_delta,
        dealer_id=dealer_id,
        dealer_name=dealer_name,
        dealer_avatar_url=dealer_avatar_url,
        dealer_delta=dealer_delta,
    )


async def get_jackpot_pool(game_id: str) -> int:
    """Returns the current `pool_balance` for a game's shared jackpot.

    Reading the seeded row is the canonical way to surface the current
    pool to a view (lobby start, every active-table refresh). Seeded pools
    are replenished before returning if an older process left them drained.
    Returns `0` when the row hasn't been seeded yet so a freshly-introduced
    game can short-circuit cleanly.

    Args:
        game_id: Game identifier (e.g. `"dragon_gate"`).

    Returns:
        The current pool balance in points.
    """
    snapshot = await get_jackpot_snapshot(game_id=game_id)
    return snapshot.balance


async def get_jackpot_snapshot(game_id: str) -> JackpotSnapshot:
    """Returns the current jackpot balance and generation for a shared pool."""
    await _ensure_global_state_schema()
    async with open_global_state_session() as session:
        snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
            session=session, game_id=game_id, now=_database_now()
        )
        await session.commit()
        return snapshot


async def _replenish_jackpot_if_depleted_in_session(
    session: AsyncSession, game_id: str, balance: int, generation: int, now: datetime
) -> JackpotSnapshot:
    """Tops a seeded jackpot back up when the stored balance is drained."""
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

    Args:
        session: Active SQLAlchemy session bound to `_global_state_engine`.
        game_id: Game identifier (jackpot row primary key).
        delta: Signed point adjustment to apply to `pool_balance`.
        now: `_database_now()` value pinned for this transaction.

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
    """Atomically claims up to `amount` from the requested jackpot generation."""
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
        player_id: Discord user ID for the player.
        player_account_name: Account name to store on the player row.
        player_delta: Signed net change for the player. Losses are written
            as a negative delta and the absolute value flows into the pool.
        game_id: Jackpot game identifier (e.g. `"dragon_gate"`).
        player_avatar_url: Last-seen Discord avatar URL for the player.
        expected_jackpot_generation: Optional pool generation observed by the
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
    """Returns required-full-debit player IDs that cannot cover their debits."""
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
    pool is drained, the same global-state transaction restores its
    on-the-house seed. Economy and jackpot writes now live in separate SQLite
    files, so ordinary exceptions roll both sessions back before either commit,
    but a hard crash between the two final commits is not cross-file atomic.

    Args:
        game_id: Jackpot game identifier (e.g. `"dragon_gate"`).
        settlements: Player-side settlements to apply in order.

    Returns:
        The latest balance for each touched player, the actual applied deltas,
        and the final jackpot balance after the final settlement and any reseed.
    """
    await _ensure_schema()
    await _ensure_global_state_schema()
    now = _database_now()
    async with open_session() as economy_session, open_global_state_session() as global_session:
        player_balances: dict[int, int] = {}
        applied_player_deltas: dict[int, int] = {}
        jackpot_snapshot: JackpotSnapshot | None = None
        jackpot_depleted = False

        try:
            rejected_player_ids = await _full_debit_rejections_in_session(
                session=economy_session, settlements=settlements
            )
            if rejected_player_ids:
                jackpot_snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
                    session=global_session, game_id=game_id, now=now
                )
                await global_session.commit()
                await economy_session.commit()
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
                        session=global_session,
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
                    session=economy_session,
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
                    await economy_session.rollback()
                    await global_session.rollback()
                    jackpot_snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
                        session=global_session, game_id=game_id, now=now
                    )
                    await global_session.commit()
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
                        session=global_session, game_id=game_id, now=now
                    )
                    continue

                if applied_player_delta < 0:
                    jackpot_snapshot, depleted = await _apply_jackpot_delta_in_session(
                        session=global_session,
                        game_id=game_id,
                        delta=-applied_player_delta,
                        now=now,
                    )
                    jackpot_depleted = jackpot_depleted or depleted

            if jackpot_snapshot is None:
                jackpot_snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
                    session=global_session, game_id=game_id, now=now
                )

            await global_session.commit()
            await economy_session.commit()
            return JackpotSettlementBatchResult(
                player_balances=player_balances,
                applied_player_deltas=applied_player_deltas,
                jackpot_balance=jackpot_snapshot.balance,
                jackpot_generation=jackpot_snapshot.generation,
                jackpot_depleted=jackpot_depleted,
            )
        except Exception:
            await economy_session.rollback()
            await global_session.rollback()
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

    Args:
        last_checkin_at: Stored `last_checkin_at` (Taipei-naive) or `None`.
        current_streak: Currently-persisted streak counter.
        today_midnight: 00:00 Asia/Taipei for the request day.
        yesterday_midnight: 00:00 Asia/Taipei for the prior day.
        tomorrow_midnight: 00:00 Asia/Taipei for the next day.

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

    Args:
        session: Active SQLAlchemy session.
        user_id: Discord user ID checking in.
        name: Last-seen Discord username to store on the account.
        avatar_url: Last-seen Discord avatar URL to store when available.
        now: `_database_now()` value pinned for this transaction.

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
    insert_result = await session.execute(statement=insert_stmt)
    if (insert_result.rowcount or 0) == 0:
        return None
    credit_result = await session.execute(
        statement=_build_credit_upsert(user_id=user_id, name=name, amount=reward, now=now)
    )
    balance_after = credit_result.scalar_one()
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

    Args:
        session: Active SQLAlchemy session.
        user_id: Discord user ID checking in.
        name: Last-seen Discord username to refresh on the account.
        avatar_url: Last-seen Discord avatar URL to refresh when set.
        now: `_database_now()` value pinned for this transaction.
        new_streak: Streak counter chosen by `_next_checkin_streak`.
        row: Tuple returned by the prior SELECT.

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
        user_id: Discord user ID checking in.
        name: Last-seen Discord username to store on the account.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        `CheckinResult` describing the credit, or `None` when the user
        already checked in today.
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
            return CheckinResult(
                new_balance=balance_after, amount=reward, streak=streak_after, is_vip=vip_after
            )

        return None


async def buy_vip(user_id: int, name: str, avatar_url: str = "") -> VipPurchaseResult | None:
    """Promotes the user to VIP after debiting `VIP_PURCHASE_COST` points.

    Returns `None` when the user is already VIP, has insufficient balance,
    or the retry budget for the conditional UPDATE was exhausted.

    Args:
        user_id: Discord user ID purchasing VIP.
        name: Last-seen Discord username to store on the account.
        avatar_url: Last-seen Discord avatar URL to store when available.

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
            return VipPurchaseResult(new_balance=wallet_row[0], cost=cost)

        return None


async def get_balance(user_id: int) -> int:
    """Returns the current balance for a user.

    Args:
        user_id: Discord user ID to look up.

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

    Args:
        user_id: Discord user ID to look up.

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

    Args:
        user_id: Discord user ID to look up.

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
        user_id: Discord user ID to modify.
        name: Last-seen Discord username to store when available.
        is_admin: Desired admin flag value.
        avatar_url: Last-seen Discord avatar URL to store when available.

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
    """Returns all economy admins ordered by user ID."""
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.user_id, UserAccount.name)
            .where(UserAccount.is_admin.is_(True))
            .order_by(UserAccount.user_id)
        )
        return [AdminAccount(user_id=row[0], name=row[1]) for row in result.all()]


async def get_central_banker(user_id: int) -> bool:
    """Returns whether the user can operate central-bank lending commands."""
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.is_central_banker).where(UserAccount.user_id == user_id)
        )
        return bool(result.scalar_one_or_none())


async def set_central_banker(
    user_id: int, name: str, is_central_banker: bool, avatar_url: str = ""
) -> bool:
    """Sets the central banker flag for a Discord user."""
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


async def list_central_bankers() -> list[CentralBankerAccount]:
    """Returns all central bankers ordered by user ID."""
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.user_id, UserAccount.name)
            .where(UserAccount.is_central_banker.is_(True))
            .order_by(UserAccount.user_id)
        )
        return [CentralBankerAccount(user_id=row[0], name=row[1]) for row in result.all()]


async def get_account(user_id: int) -> AccountSnapshot | None:
    """Returns the stored account snapshot for a user.

    Args:
        user_id: Discord user ID to look up.

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
    """Atomically moves points from sender to receiver.

    The debit is a single conditional `UPDATE` gated on `balance >= amount`;
    if that returns no row the transfer is rejected without ever touching
    the receiver. The credit is a UPSERT in the same transaction, so the
    receiver row is created on first contact and the whole transfer is one
    all-or-nothing operation. Both balances are returned from the same SQL
    writes, so callers do not need extra reads after a successful transfer.

    Args:
        sender_id: Discord user ID to debit.
        sender_name: Last-seen Discord username to store on the sender account.
        receiver_id: Discord user ID to credit.
        receiver_name: Last-seen Discord username to store on the receiver account.
        amount: Number of points to transfer.
        sender_avatar_url: Last-seen Discord avatar URL for the sender.
        receiver_avatar_url: Last-seen Discord avatar URL for the receiver.

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

        credit_stmt = _build_credit_upsert(
            user_id=receiver_id, name=receiver_name, amount=amount, now=now
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
        return TransferResult(sender_balance=sender_balance, receiver_balance=receiver_balance)


async def top_n(
    limit: int | None = 10, exclude_user_ids: tuple[int, ...] = (), include_hidden: bool = False
) -> list[LeaderboardEntry]:
    """Returns accounts ordered by balance descending.

    `exclude_user_ids` filters out specific accounts (notably the bot's
    own house ledger row) before applying the limit, so the leaderboard
    always shows real players. The `ix_user_wallet_balance` index keeps
    this query cheap even as the user table grows.

    Args:
        limit: Maximum number of accounts to return, or `None` to return all
            matching accounts.
        exclude_user_ids: User IDs to filter out before applying the limit.
        include_hidden: Whether to include accounts marked as hidden from
            public leaderboards.

    Returns:
        Leaderboard entries ordered by balance descending. `avatar_url` is
        empty when the user has never been seen by an avatar-aware write path.
    """
    await _ensure_schema()
    async with open_session() as session:
        stmt = select(
            UserWallet.user_id, UserAccount.name, UserWallet.balance, UserAccount.avatar_url
        ).join(UserAccount, UserAccount.user_id == UserWallet.user_id)
        if not include_hidden:
            stmt = stmt.where(UserAccount.hide_from_leaderboard.is_(False))
        if exclude_user_ids:
            stmt = stmt.where(UserWallet.user_id.notin_(other=exclude_user_ids))
        result = await session.execute(statement=stmt)
        rows = [
            LeaderboardEntry(user_id=row[0], name=row[1], balance=row[2], avatar_url=row[3] or "")
            for row in result.all()
        ]
        rows.sort(key=lambda entry: entry.balance, reverse=True)
        return rows if limit is None else rows[:limit]


async def top_losers(
    limit: int = 10, exclude_user_ids: tuple[int, ...] = (), include_hidden: bool = False
) -> list[LossLeaderboardEntry]:
    """Returns the biggest gross casino losers for the current Taipei day.

    The leaderboard reads persisted `casino_account` daily counters. Writes lazily reset stale
    counters at the first casino settlement after Taipei midnight, while this
    query filters by today's `day_started_at` so yesterday's counters
    never leak into a new day.

    Args:
        limit: Maximum number of accounts to return.
        exclude_user_ids: User IDs to filter out before applying the limit.
        include_hidden: Whether to include accounts marked as hidden from
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
        return rows


def _loan_proposal_view(proposal: LoanProposal) -> LoanProposalView:
    """Projects an ORM loan proposal into an immutable API view."""
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
    """Projects an ORM loan contract into an immutable API view."""
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
    """Returns whether a pending loan proposal has passed its decision window."""
    if proposal.status != LoanProposalStatus.PENDING:
        return False
    elapsed_seconds = (_as_taipei(dt=now) - _as_taipei(dt=proposal.created_at)).total_seconds()
    return elapsed_seconds >= LOAN_PROPOSAL_TIMEOUT_SECONDS


async def _reject_expired_loan_proposal_in_session(
    session: AsyncSession, proposal: LoanProposal, now: datetime
) -> LoanProposalView | None:
    """Marks an expired pending proposal as rejected inside the caller's session."""
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
    """Returns simple-interest delta and the timestamp covered by accrual."""
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
    """Persists lazy simple-interest accrual for one active contract."""
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
    """Computes central-bank lending capacity from positive user balances."""
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
    """Returns current central-bank lending capacity."""
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
    """Creates a borrower-initiated personal loan request."""
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
    """Creates a borrower-initiated central-bank loan request."""
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
    """Refunds escrowed proposal funds and returns lender balance."""
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
    """Rejects a pending loan proposal if its decision window has expired."""
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
    """Cancels a pending proposal created by `actor_id`."""
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
    """Rejects a pending proposal when `actor_id` is allowed to decide it."""
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
    """Accepts a pending loan proposal and opens the loan contract."""
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
    """Accepts a loan proposal while the caller holds the acceptance lock."""
    now = _database_now()
    async with open_session() as session:
        # Acquire SQLite's write lock before reading capacity or proposal state.
        await session.execute(statement=text(text="BEGIN IMMEDIATE"))
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
            interest_due=0,
            total_interest_paid=0,
            total_principal_paid=0,
            monthly_rate_bps=proposal.monthly_rate_bps,
            status=LoanContractStatus.ACTIVE,
            opened_at=now,
            last_interest_accrued_at=now,
            updated_at=now,
        )
        session.add(contract)
        await session.commit()
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
    """Returns active contracts in repayment priority order."""
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
    """Applies a repayment or forced collection across ordered contracts."""
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
    """Repays active personal loans from `borrower_id` to `lender_id`."""
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
        return result


async def call_personal_loans(
    lender_id: int,
    borrower_id: int,
    borrower_name: str,
    amount: int | None = None,
    borrower_avatar_url: str = "",
) -> LoanPaymentResult | None:
    """Forcibly collects active personal loans owed to `lender_id`."""
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
        return result


async def repay_central_bank_loans(
    borrower_id: int, borrower_name: str, amount: int, borrower_avatar_url: str = ""
) -> LoanPaymentResult | None:
    """Repays active central-bank loans for a borrower."""
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
        return result


async def call_central_bank_loans(
    borrower_id: int, borrower_name: str, amount: int | None = None, borrower_avatar_url: str = ""
) -> LoanPaymentResult | None:
    """Forcibly collects active central-bank loans from a borrower."""
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
        return result


async def list_loan_contracts(
    user_id: int, include_closed: bool = False
) -> list[LoanContractView]:
    """Lists loan contracts where the user is borrower or personal lender."""
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
    """Builds a portfolio view, accruing active debt interest first."""
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
    """Returns a user's current portfolio and estimated net worth."""
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        portfolio = await _portfolio_in_session(session=session, user_id=user_id, now=now)
        await session.commit()
        return portfolio
