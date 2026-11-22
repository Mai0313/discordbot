"""Persistent point-balance store for the economy cog.

Every balance-mutating write path is atomic at the SQLite transaction level via UPSERT or
conditional `UPDATE ... WHERE ... RETURNING`; multi-row writes roll back as one unit on a lost
race.

VIP bumps winning Blackjack payouts and is permanent once set. Admin status gates economy-admin
commands and is set out-of-band via `set_admin`. Daily casino counters live on `casino_account`
(one row per user). Personal loan approval debits the lender; central-bank approval mints
principal bounded by the per-guild lending pool (`guild_participant`) and the per-borrower
ceiling. Casino and jackpot settlements commit player and house writes in one transaction.
"""

from time import monotonic
from typing import Any, Final
from datetime import datetime, timedelta
from contextlib import AbstractAsyncContextManager
from collections.abc import Mapping, Sequence

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
    LEADERBOARD_SIZE,
    TRANSFER_TAX_BPS,
    MIN_INTEREST_DAYS,
    VIP_PURCHASE_COST,
    CENTRAL_BANK_BASE_CAPACITY,
    DEFAULT_LOAN_MONTHLY_RATE_BPS,
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
    CreditResult,
    PortfolioView,
    LoanLenderType,
    TransferResult,
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
    CasinoLedgerSnapshot,
    LossLeaderboardEntry,
    RoundSettlementResult,
    BalanceAdjustmentResult,
    JackpotSettlementResult,
    JackpotSettlementRequest,
    LoanProposalAcceptResult,
    LoanProposalExpiredError,
    JackpotSettlementBatchResult,
    clamped_balance,
    simple_interest,
    clamp_loan_rate_bps,
    central_bank_credit_ceiling,
)
from discordbot.utils.asyncio_locks import LoopLocalLock
from discordbot.utils.sqlite_config import SqliteBootstrap
from discordbot.utils.stored_integer import StoredInteger, int_add_text, int_compare_text
from discordbot.services.economy.holds import spending, spendable, held_amount

# SELECT-then-conditional-UPDATE loops keep a small retry budget. The bound is
# there to stop a degenerate hot-row livelock, not to ride out contention.
_CONDITIONAL_WRITE_MAX_RETRIES: Final[int] = 8
_ECONOMY_LEADERBOARD_CACHE_TTL_SECONDS: Final[float] = 5.0

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/economy.db")


def _taipei_midnight(now: datetime) -> datetime:
    """Returns the most recent Asia/Taipei 00:00 boundary at or before `now`."""
    local = _as_taipei(dt=now)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


class Base(DeclarativeBase):
    """Base class for economy ORM models."""


class UserAccount(Base):
    """Persistent identity, VIP, and admin state for a Discord user.

    Spendable balance and lifetime gross totals live in `user_wallet`. Debt
    state lives in `loan_contract` and daily casino counters live in
    `casino_account`.

    Attributes:
        user_id (Mapped[int]): Discord user ID primary key.
        name (Mapped[str]): Last-seen Discord username (refreshed on every write).
        avatar_url (Mapped[str]): Last-seen Discord avatar URL (refreshed on writes that carry it).
        updated_at (Mapped[datetime]): Taiwan-local timestamp of the last write.
        is_vip (Mapped[bool]): Permanent VIP flag toggled by a successful `/vip` purchase.
        is_admin (Mapped[bool]): Whether the user can run Discord-side economy admin commands.
        hide_from_leaderboard (Mapped[bool]): Whether the account is omitted from public balance
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
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    hide_from_leaderboard: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False
    )


class UserWallet(Base):
    """Spendable balance and lifetime gross totals for a Discord user."""

    __tablename__ = "user_wallet"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    balance: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_earned: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_spent: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


class GuildParticipant(Base):
    """One user recorded as taking part in one guild's economy.

    Rows are append-only: recorded only for rewarded message authors and `/central_bank` callers,
    never for `member:` targets, and never pruned. The central bank reads this table to determine
    which wallets back a guild's lending pool.
    """

    __tablename__ = "guild_participant"

    guild_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


class CasinoAccount(Base):
    """Daily per-user casino counters for loss leaderboard queries.

    Attributes:
        user_id (Mapped[int]): Discord user ID primary key.
        name (Mapped[str]): Last-seen Discord username for quick inspection.
        day_started_at (Mapped[datetime | None]): Asia/Taipei midnight for the stored counters.
        daily_loss (Mapped[int]): Current-day gross loss from player-side casino settlements.
        daily_win (Mapped[int]): Current-day gross win from player-side casino settlements.
        daily_net (Mapped[int]): Current-day signed net casino result.
        updated_at (Mapped[datetime]): Taiwan-local timestamp of the last casino counter write.
    """

    __tablename__ = "casino_account"
    __table_args__ = (
        # The daily loss ranking filters to one Taipei day, which this index serves.
        # Its ordering half does not: the counters are decimal text, so the
        # query sorts by length(daily_loss) first and SQLite falls back to a
        # temp B-tree for the ORDER BY.
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
    requests wait for a server administrator's approval and do not escrow a user
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


class JackpotPool(Base):
    """Per-game cumulative jackpot shared across every table of that game.

    One row per game (keyed by `game_id`). Tracks `total_contributed`,
    `total_claimed`, and `seeded_amount` alongside `pool_balance`.

    Attributes:
        game_id (Mapped[str]): Game identifier primary key.
        pool_balance (Mapped[int]): Current spendable jackpot for the game.
        total_contributed (Mapped[int]): Lifetime gross amount that flowed into the pool
            (positive deltas from player losses + ante).
        total_claimed (Mapped[int]): Lifetime gross amount paid out from the pool
            (absolute value of negative deltas from player wins).
        seeded_amount (Mapped[int]): Lifetime on-the-house seed total; bookkeeping only,
            never decremented.
        generation (Mapped[int]): Incremented each time a seeded pool is depleted and
            replenished.
        updated_at (Mapped[datetime]): Taiwan-local timestamp of the last write.
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

    Player wins flow out of this row; player losses flow in. The bot's own
    `user_wallet` is a player wallet, not the house ledger. `balance` may go
    negative when payouts exceed take-in.

    Attributes:
        ledger_id (Mapped[str]): Singleton ledger identifier primary key.
        balance (Mapped[int]): Signed cumulative P&L.
        total_earned (Mapped[int]): Lifetime gross inflows (from player losses).
        total_spent (Mapped[int]): Lifetime gross outflows (to player wins).
        updated_at (Mapped[datetime]): Taiwan-local timestamp of the last write.
    """

    __tablename__ = "casino_ledger"

    ledger_id: Mapped[str] = mapped_column(String(length=32), primary_key=True)
    balance: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_earned: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_spent: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


class CentralBankLedger(Base):
    """Cumulative interest retained by the central bank for relending.

    Central-bank principal is minted on approval and burned on repayment; only collected
    interest accumulates here and is added to `CENTRAL_BANK_BASE_CAPACITY`.

    Attributes:
        ledger_id (Mapped[str]): Singleton ledger identifier primary key.
        balance (Mapped[int]): Retained interest available to lend again.
        total_earned (Mapped[int]): Lifetime gross interest collected.
        updated_at (Mapped[datetime]): Taiwan-local timestamp of the last write.
    """

    __tablename__ = "central_bank_ledger"

    ledger_id: Mapped[str] = mapped_column(String(length=32), primary_key=True)
    balance: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_earned: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


CASINO_LEDGER_ID: Final[str] = "casino"
CENTRAL_BANK_LEDGER_ID: Final[str] = "central_bank"


# On-the-house seed amount for each registered jackpot pool. The seed is
# bookkeeping only — nothing is debited to fund it, so casino P&L never moves
# for it. A seeded pool is topped back up to this amount whenever it drains.
_JACKPOT_SEEDS: Final[Mapping[str, int]] = {"dragon_gate": 1_000}

# Serializes loan approval so central-bank capacity is consumed once.
_loan_accept_lock = LoopLocalLock()
type _TopNCacheKey = tuple[int | None, bool]
type _TopLosersCacheKey = tuple[int, datetime]
_top_n_cache: dict[_TopNCacheKey, tuple[float, tuple[LeaderboardEntry, ...]]] = {}
_top_losers_cache: dict[_TopLosersCacheKey, tuple[float, tuple[LossLeaderboardEntry, ...]]] = {}


def invalidate_economy_leaderboard_cache() -> None:
    """Clears process-local leaderboard row caches.

    Leaves rendered board image caches untouched; those are keyed on row data.
    """
    _top_n_cache.clear()
    _top_losers_cache.clear()


async def _commit_balance_write(session: AsyncSession) -> None:
    """Commits a balance-mutating write, then clears the leaderboard caches."""
    await session.commit()
    # Clear after commit so a concurrent read cannot cache pre-commit rows until TTL expiry.
    invalidate_economy_leaderboard_cache()


def _cached_leaderboard_rows[K, R](
    cache: dict[K, tuple[float, tuple[R, ...]]], cache_key: K
) -> list[R] | None:
    """Returns cached leaderboard rows when the short TTL is still valid."""
    cached = cache.get(cache_key)
    if cached is None:
        return None
    cached_at, rows = cached
    if monotonic() - cached_at > _ECONOMY_LEADERBOARD_CACHE_TTL_SECONDS:
        cache.pop(cache_key, None)
        return None
    return list(rows)


def _stored_integer_desc_order(column: Any) -> tuple[Any, ...]:  # noqa: ANN401 -- SQLAlchemy columns are generic expressions
    """Returns ORDER BY terms for descending numeric order over decimal text."""
    sign = int_compare_text(column=column, value=0)
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


# Foreign keys are enabled defensively for any future FK constraint.
_database = SqliteBootstrap(enable_foreign_keys=True)


def open_session() -> AbstractAsyncContextManager[AsyncSession]:
    """Returns an async session context manager on the current economy database engine.

    The first session on an engine bootstraps the schema and its seed rows.
    """
    return _database.open_session(engine=_engine)


async def _balance_in_session(session: AsyncSession, user_id: int) -> int:
    """Reads one wallet balance, returning 0 for a user with no wallet row."""
    result = await session.execute(
        statement=select(UserWallet.balance).where(UserWallet.user_id == user_id)
    )
    return result.scalar_one_or_none() or 0


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
        is_admin=False,
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
) -> ReturningInsert[int]:
    """Returns a wallet upsert statement that credits `amount` points.

    Caller guarantees `amount > 0` and refreshes `user_account` metadata
    separately.
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
) -> ReturningInsert[int]:
    """Returns an unclamped wallet upsert statement applying a signed `delta`."""
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
                    (same_day, int_add_text(column=CasinoAccount.daily_loss, delta=loss_delta)),
                    else_=loss_delta_text,
                ),
                "daily_win": case(
                    (same_day, int_add_text(column=CasinoAccount.daily_win, delta=win_delta)),
                    else_=win_delta_text,
                ),
                "daily_net": case(
                    (same_day, int_add_text(column=CasinoAccount.daily_net, delta=delta)),
                    else_=delta_text,
                ),
                "updated_at": now,
            },
        )
    )


async def _take_back_daily_casino_loss_in_session(
    session: AsyncSession, user_id: int, amount: int, now: datetime
) -> None:
    """Subtracts a refunded debit from today's `casino_account` loss on the same Taipei day."""
    await session.execute(
        statement=update(CasinoAccount)
        .where(
            CasinoAccount.user_id == user_id,
            CasinoAccount.day_started_at == _taipei_midnight(now=now),
            CasinoAccount.daily_loss >= amount,
        )
        .values(
            daily_loss=CasinoAccount.daily_loss - amount,
            daily_net=CasinoAccount.daily_net + amount,
            updated_at=now,
        )
    )


async def _take_back_daily_casino_win_in_session(
    session: AsyncSession, user_id: int, amount: int, now: datetime
) -> None:
    """Subtracts a clawed-back win from today's `casino_account` win on the same Taipei day."""
    if amount <= 0:
        return
    await session.execute(
        statement=update(CasinoAccount)
        .where(
            CasinoAccount.user_id == user_id,
            CasinoAccount.day_started_at == _taipei_midnight(now=now),
            CasinoAccount.daily_win >= amount,
        )
        .values(
            daily_win=CasinoAccount.daily_win - amount,
            daily_net=CasinoAccount.daily_net - amount,
            updated_at=now,
        )
    )


async def _credit_with_repayment_in_session(  # noqa: PLR0913 -- session helper keeps income writes atomic
    session: AsyncSession, user_id: int, name: str, avatar_url: str, amount: int, now: datetime
) -> CreditResult:
    """Credits income inside the caller's transaction without repaying loans.

    Caller must guarantee `amount > 0`.
    """
    await _upsert_user_metadata_in_session(
        session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
    )
    result = await session.execute(
        statement=_build_credit_upsert(user_id=user_id, name=name, amount=amount, now=now)
    )
    new_balance = result.scalar_one()
    return CreditResult(new_balance=new_balance)


async def _apply_clamped_delta_in_session(  # noqa: PLR0913 -- session helper needs identity, delta, and optional hold floor
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    delta: int,
    now: datetime,
    floor: int = 0,
) -> tuple[int, int]:
    """Applies a clamped signed delta via conditional UPDATE, returning `(balance, applied)`.

    Caller must guarantee `delta != 0`. Negative deltas stop at `floor` (default 0) so a
    caller can leave held table stakes untouched; a negative delta on a missing row returns `(0, 0)`.
    """
    for _ in range(_CONDITIONAL_WRITE_MAX_RETRIES):
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
            floor=floor,
        )
        if update_result is not None:
            if update_result[1] != 0 or floor == 0:
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


async def _try_update_clamped_delta_in_session(  # noqa: PLR0913 -- conditional write needs observed row state and floor
    session: AsyncSession,
    user_id: int,
    name: str,
    current_balance: int,
    delta: int,
    now: datetime,
    floor: int = 0,
) -> tuple[int, int] | None:
    """Attempts one conditional clamped update against an existing account."""
    if delta < 0 and floor > 0:
        new_balance = min(current_balance, max(current_balance + delta, floor))
    else:
        new_balance = clamped_balance(balance=current_balance, delta=delta, allow_negative=False)
    applied = new_balance - current_balance
    if applied == 0 and floor > 0:
        return current_balance, 0
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
    """Applies an unclamped signed `delta` to a user wallet, returning the new balance."""
    await _upsert_user_metadata_in_session(
        session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
    )
    stmt = _build_signed_delta_upsert(user_id=user_id, name=name, delta=delta, now=now)
    result = await session.execute(statement=stmt)
    return result.scalar_one()


async def _apply_casino_ledger_delta_in_session(
    session: AsyncSession, delta: int, now: datetime
) -> int:
    """Applies an unclamped signed `delta` to the global casino ledger, returning its balance."""
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
    """Reads the current casino ledger balance, returning 0 when missing."""
    result = await session.execute(
        statement=select(CasinoLedger.balance).where(CasinoLedger.ledger_id == CASINO_LEDGER_ID)
    )
    return result.scalar_one_or_none() or 0


async def _central_bank_ledger_balance_in_session(session: AsyncSession) -> int:
    """Reads the lifetime central-bank interest balance, returning 0 when missing."""
    result = await session.execute(
        statement=select(CentralBankLedger.balance).where(
            CentralBankLedger.ledger_id == CENTRAL_BANK_LEDGER_ID
        )
    )
    return result.scalar_one_or_none() or 0


async def _credit_central_bank_ledger_in_session(
    session: AsyncSession, amount: int, now: datetime
) -> None:
    """Books interest the central bank just collected, in the payment's transaction."""
    if amount <= 0:
        return
    await session.execute(
        statement=insert(CentralBankLedger)
        .values(
            ledger_id=CENTRAL_BANK_LEDGER_ID, balance=amount, total_earned=amount, updated_at=now
        )
        .on_conflict_do_update(
            index_elements=["ledger_id"],
            set_={
                "balance": CentralBankLedger.balance + amount,
                "total_earned": CentralBankLedger.total_earned + amount,
                "updated_at": now,
            },
        )
    )


async def _rollback_session(session: AsyncSession) -> None:
    """Rolls back a session without masking the original settlement exception."""
    try:
        await session.rollback()
    except Exception:
        logfire.warn("Failed to roll back settlement session", _exc_info=True)


async def get_casino_ledger() -> CasinoLedgerSnapshot:
    """Returns the cumulative casino system ledger snapshot."""
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


async def _apply_player_delta_in_session(  # noqa: PLR0913 -- player settlement needs identity and the delta
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    delta: int,
    now: datetime,
    refund: bool = False,
) -> tuple[int, int]:
    """Applies a casino or jackpot player delta, returning `(new_balance, applied_delta)`.

    Negative deltas clamp at zero; `refund=True` reverses today's opposing daily counter
    (loss on positive refund, win on negative clawback) instead of booking a fresh win or loss.
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
        if refund:
            await _take_back_daily_casino_loss_in_session(
                session=session, user_id=user_id, amount=delta, now=now
            )
        else:
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
        if refund:
            await _take_back_daily_casino_win_in_session(
                session=session, user_id=user_id, amount=-applied_delta, now=now
            )
        else:
            await _apply_daily_casino_delta_in_session(
                session=session, user_id=user_id, name=name, delta=applied_delta, now=now
            )
        return new_balance, applied_delta
    return await _balance_in_session(session=session, user_id=user_id), 0


async def credit_with_repayment(
    user_id: int, name: str, amount: int, avatar_url: str = ""
) -> CreditResult:
    """Credits `amount` to the user through the shared income path.

    Does not repay loans; income lands in `balance` and increments `total_earned`.

    Args:
        user_id (int): Discord user ID receiving the credit.
        name (str): Last-seen Discord username to store on the account.
        amount (int): Gross income amount; a non-positive amount credits nothing
            and returns the current balance.
        avatar_url (str): Last-seen Discord avatar URL to store when available.

    Returns:
        Outcome capturing post-credit balance.
    """
    if amount <= 0:
        return CreditResult(new_balance=await get_balance(user_id=user_id))
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
        await _commit_balance_write(session=session)
        return result


async def adjust_balance(
    user_id: int, name: str, delta: int, allow_negative: bool = False, avatar_url: str = ""
) -> BalanceAdjustmentResult:
    """Applies an explicit manual balance adjustment.

    Touches neither loan contracts, daily casino counters, nor the casino ledger.

    Args:
        user_id (int): Discord user ID whose balance should be adjusted.
        name (str): Last-seen Discord username to store on the account.
        delta (int): Signed amount to apply.
        allow_negative (bool): Whether the resulting balance may go below zero.
        avatar_url (str): Last-seen Discord avatar URL to store when available.

    Returns:
        The post-adjustment balance and the applied delta after any clamp.
    """
    now = _database_now()
    async with open_session() as session:
        if delta == 0:
            new_balance = await _balance_in_session(session=session, user_id=user_id)
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
        await _commit_balance_write(session=session)
        return BalanceAdjustmentResult(new_balance=new_balance, applied_delta=applied_delta)


async def apply_blackjack_settlement(
    player_id: int,
    player_account_name: str,
    player_delta: int,
    casino_delta: int,
    player_avatar_url: str = "",
) -> RoundSettlementResult:
    """Applies a finished Blackjack round's player delta and mirrors casino P&L.

    Positive player deltas credit in full; negative deltas clamp at zero,
    reducing a positive `casino_delta` by any uncollected shortfall. Pass
    `casino_delta` without system-funded bonuses (such as five-card 21).
    Commits player and casino writes in one atomic transaction.

    Args:
        player_id (int): Discord user ID for the player account.
        player_account_name (str): Account name to store for the player.
        player_delta (int): Signed net change for the player. Losses are clamped at
            zero and may apply less than the requested debit.
        casino_delta (int): Signed change to apply to the casino ledger balance.
        player_avatar_url (str): Last-seen Discord avatar URL for the player.

    Returns:
        A `RoundSettlementResult` with the post-write player and casino balances
        and the player delta actually applied.
    """
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

            # What the casino books is reduced by what could not be collected, and by nothing
            # else. Capping it at the player's net movement instead would also deduct a
            # system-funded bonus, which is already added back into that net and which the house
            # never paid — on a round whose loss collected in full.
            uncollected = applied_player_delta - player_delta
            casino_delta_to_apply = casino_delta
            if casino_delta > 0 and uncollected > 0:
                casino_delta_to_apply = max(casino_delta - uncollected, 0)

            if casino_delta_to_apply == 0:
                casino_balance = await _read_casino_ledger_balance_in_session(session=session)
            else:
                casino_balance = await _apply_casino_ledger_delta_in_session(
                    session=session, delta=casino_delta_to_apply, now=now
                )
            await _commit_balance_write(session=session)
        except Exception:
            await _rollback_session(session=session)
            raise
    return RoundSettlementResult(
        player_balance=player_balance,
        casino_balance=casino_balance,
        applied_player_delta=applied_player_delta,
    )


async def get_jackpot_snapshot(game_id: str) -> JackpotSnapshot:
    """Returns the current jackpot balance and generation for a shared pool.

    Args:
        game_id (str): Jackpot game identifier.
    """
    async with open_session() as session:
        snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
            session=session, game_id=game_id, now=_database_now()
        )
        await session.commit()
        return snapshot


async def _replenish_jackpot_if_depleted_in_session(
    session: AsyncSession, game_id: str, balance: int, generation: int, now: datetime
) -> JackpotSnapshot:
    """Tops a seeded jackpot back up when the stored balance is drained."""
    seed_amount = _JACKPOT_SEEDS.get(game_id, 0)
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
    """Applies a signed `delta` to a jackpot pool, returning `(snapshot, depleted)`."""
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
    jackpot_depleted = pool_balance <= 0 and _JACKPOT_SEEDS.get(game_id, 0) > 0
    snapshot = await _replenish_jackpot_if_depleted_in_session(
        session=session, game_id=game_id, balance=pool_balance, generation=generation, now=now
    )
    return snapshot, jackpot_depleted


async def _read_jackpot_snapshot_or_replenish_in_session(
    session: AsyncSession, game_id: str, now: datetime
) -> JackpotSnapshot:
    """Reads a jackpot snapshot, replenishing if depleted or returning zero if missing."""
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
    for _ in range(_CONDITIONAL_WRITE_MAX_RETRIES):
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
        jackpot_depleted = pool_balance <= 0 and _JACKPOT_SEEDS.get(game_id, 0) > 0
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
    refund: bool = False,
) -> JackpotSettlementResult:
    """Applies a single player wager settlement against a shared jackpot pool.

    Args:
        player_id (int): Discord user ID for the player.
        player_account_name (str): Account name to store on the player row.
        player_delta (int): Signed net change for the player. Losses are written
            as a negative delta and the absolute value flows into the pool.
        game_id (str): Jackpot game identifier (e.g. `"dragon_gate"`).
        player_avatar_url (str): Last-seen Discord avatar URL for the player.
        expected_jackpot_generation (int | None): Optional pool generation observed by the
            caller. Positive payouts only claim from this generation.
        refund (bool): Whether this reverses an earlier table flow instead of booking a fresh
            win or loss in daily casino counters.

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
                refund=refund,
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


async def _rejected_jackpot_batch_result(
    session: AsyncSession, game_id: str, rejected_player_ids: tuple[int, ...], now: datetime
) -> JackpotSettlementBatchResult:
    """Commits the untouched jackpot state and reports a batch that applied nothing."""
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


async def apply_jackpot_settlement_batch(
    game_id: str, settlements: Sequence[JackpotSettlementRequest]
) -> JackpotSettlementBatchResult:
    """Coordinates one or more player settlements against a jackpot pool.

    Positive player deltas (wins) are capped to the live pool balance inside
    this transaction, then credited through the shared income path. A refund is
    not capped: the pool pays it in full, and a pool another table has drained
    in the meantime is reseeded to cover the shortfall. Negative
    deltas normally clamp at zero and feed the pool with the actual debit.
    Required-full-debit settlements reject the whole batch instead. If a seeded
    pool is drained, the same transaction restores its on-the-house seed. The
    whole batch commits as one atomic transaction.

    Args:
        game_id (str): Jackpot game identifier (e.g. `"dragon_gate"`).
        settlements (Sequence[JackpotSettlementRequest]): Player-side settlements to apply in order.

    Returns:
        The latest balance for each touched player, the actual applied deltas,
        and the final jackpot balance after the final settlement and any reseed.
    """
    now = _database_now()
    async with open_session() as session:
        player_balances: dict[int, int] = {}
        applied_player_deltas: dict[int, int] = {}
        jackpot_snapshot: JackpotSnapshot | None = None
        jackpot_depleted = False
        any_applied = False

        try:
            rejected_player_ids = await _full_debit_rejections_in_session(
                session=session, settlements=settlements
            )
            if rejected_player_ids:
                return await _rejected_jackpot_batch_result(
                    session=session,
                    game_id=game_id,
                    rejected_player_ids=rejected_player_ids,
                    now=now,
                )

            for settlement in settlements:
                effective_player_delta = settlement.player_delta
                if effective_player_delta > 0 and not settlement.refund:
                    claim, jackpot_snapshot, depleted = await _claim_jackpot_payout_in_session(
                        session=session,
                        game_id=game_id,
                        amount=effective_player_delta,
                        expected_generation=settlement.expected_jackpot_generation,
                        now=now,
                    )
                    effective_player_delta = claim
                    jackpot_depleted = jackpot_depleted or depleted

                (player_balance, applied_player_delta) = await _apply_player_delta_in_session(
                    session=session,
                    user_id=settlement.player_id,
                    name=settlement.player_account_name,
                    avatar_url=settlement.player_avatar_url,
                    delta=effective_player_delta,
                    now=now,
                    refund=settlement.refund,
                )
                if (
                    settlement.require_full_debit
                    and applied_player_delta != effective_player_delta
                ):
                    await session.rollback()
                    return await _rejected_jackpot_batch_result(
                        session=session,
                        game_id=game_id,
                        rejected_player_ids=(settlement.player_id,),
                        now=now,
                    )
                player_balances[settlement.player_id] = player_balance
                applied_player_deltas[settlement.player_id] = (
                    applied_player_deltas.get(settlement.player_id, 0) + applied_player_delta
                )
                if applied_player_delta == 0:
                    jackpot_snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
                        session=session, game_id=game_id, now=now
                    )
                    continue
                any_applied = True

                if applied_player_delta < 0 or settlement.refund:
                    jackpot_snapshot, depleted = await _apply_jackpot_delta_in_session(
                        session=session, game_id=game_id, delta=-applied_player_delta, now=now
                    )
                    jackpot_depleted = jackpot_depleted or depleted

            if jackpot_snapshot is None:
                jackpot_snapshot = await _read_jackpot_snapshot_or_replenish_in_session(
                    session=session, game_id=game_id, now=now
                )

            if any_applied:
                await _commit_balance_write(session=session)
            else:
                await session.commit()
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


async def buy_vip(user_id: int, name: str, avatar_url: str = "") -> VipPurchaseResult | None:
    """Promotes the user to VIP after debiting `VIP_PURCHASE_COST` points.

    Returns `None` when the user is missing either the account or the wallet
    row, is already VIP, has insufficient spendable balance, or the retry
    budget for the conditional UPDATE was exhausted.

    Args:
        user_id (int): Discord user ID purchasing VIP.
        name (str): Last-seen Discord username to store on the account.
        avatar_url (str): Last-seen Discord avatar URL to store when available.

    Returns:
        `VipPurchaseResult` describing the post-purchase balance, or
        `None` when the purchase was rejected.
    """
    now = _database_now()
    cost = VIP_PURCHASE_COST

    async with spending(user_id=user_id), open_session() as session:
        for _ in range(_CONDITIONAL_WRITE_MAX_RETRIES):
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
            if spendable(user_id=user_id, balance=balance) < cost:
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

            await _commit_balance_write(session=session)
            return VipPurchaseResult(new_balance=wallet_row[0], cost=cost)

        return None


async def get_balance(user_id: int) -> int:
    """Returns the current balance for a user, or 0 if never seen.

    Args:
        user_id (int): Discord user ID to look up.
    """
    async with open_session() as session:
        return await _balance_in_session(session=session, user_id=user_id)


async def get_balances(user_ids: Sequence[int]) -> dict[int, int]:
    """Returns current balances keyed by user ID, defaulting unseen users to 0.

    Args:
        user_ids (Sequence[int]): Discord user IDs to look up.
    """
    unique_ids = tuple(dict.fromkeys(user_ids))
    if not unique_ids:
        return {}
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserWallet.user_id, UserWallet.balance).where(
                UserWallet.user_id.in_(other=unique_ids)
            )
        )
        stored = {row[0]: row[1] for row in result.all()}
    return {user_id: stored.get(user_id, 0) for user_id in unique_ids}


async def get_spendable_balance(user_id: int) -> int:
    """Returns the user's balance less what a casino table holds of it (`holds.py`).

    Args:
        user_id (int): Discord user ID to look up.
    """
    return spendable(user_id=user_id, balance=await get_balance(user_id=user_id))


async def get_vip(user_id: int) -> bool:
    """Returns whether the user owns the VIP perk.

    Args:
        user_id (int): Discord user ID to look up.
    """
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.is_vip).where(UserAccount.user_id == user_id)
        )
        return bool(result.scalar_one_or_none())


async def get_vips(user_ids: Sequence[int]) -> dict[int, bool]:
    """Returns VIP flags keyed by user ID, defaulting unseen users to False.

    Args:
        user_ids (Sequence[int]): Discord user IDs to look up.
    """
    unique_ids = tuple(dict.fromkeys(user_ids))
    if not unique_ids:
        return {}
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.user_id, UserAccount.is_vip).where(
                UserAccount.user_id.in_(other=unique_ids)
            )
        )
        stored = {row[0]: bool(row[1]) for row in result.all()}
    return {user_id: stored.get(user_id, False) for user_id in unique_ids}


async def get_admin(user_id: int) -> bool:
    """Returns whether the user can run economy admin commands.

    Args:
        user_id (int): Discord user ID to look up.
    """
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.is_admin).where(UserAccount.user_id == user_id)
        )
        return bool(result.scalar_one_or_none())


async def set_admin(user_id: int, name: str, is_admin: bool, avatar_url: str = "") -> bool:
    """Sets the economy admin flag for a Discord user.

    Granting creates the `user_account` row if missing without creating a
    `user_wallet` row. Revoking updates an existing row only and ignores
    missing users.

    Args:
        user_id (int): Discord user ID to modify.
        name (str): Last-seen Discord username to store when available.
        is_admin (bool): Desired admin flag value.
        avatar_url (str): Last-seen Discord avatar URL to store when available.

    Returns:
        `True` when a row was created or updated; `False` when revoking a
        missing user.
    """
    now = _database_now()
    effective_name = name or str(user_id)
    values: dict[str, Any] = {"is_admin": is_admin, "updated_at": now}
    if name:
        values["name"] = effective_name
    if avatar_url:
        values["avatar_url"] = avatar_url
    async with open_session() as session:
        if is_admin:
            statement = (
                insert(UserAccount)
                .values(
                    user_id=user_id,
                    name=effective_name,
                    avatar_url=avatar_url,
                    updated_at=now,
                    is_vip=False,
                    is_admin=True,
                )
                .on_conflict_do_update(index_elements=["user_id"], set_=values)
                .returning(UserAccount.user_id)
            )
        else:
            statement = (
                update(UserAccount)
                .where(UserAccount.user_id == user_id)
                .values(**values)
                .returning(UserAccount.user_id)
            )
        result = await session.execute(statement=statement)
        await session.commit()
        return result.scalar_one_or_none() is not None


async def get_account(user_id: int) -> AccountSnapshot | None:
    """Returns the stored account snapshot for a user, or `None` if never seen.

    Args:
        user_id (int): Discord user ID to look up.
    """
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


async def _debit_spendable_in_session(  # noqa: PLR0913 -- debit needs actor identity and amount
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    amount: int,
    now: datetime,
    wallet_name: str = "",
) -> int | None:
    """Debits `amount` above active table holds; returns new balance or `None` if short.

    Caller must hold `spending(user_id)`.
    """
    held = held_amount(user_id=user_id)
    if held is None:
        return None
    debit_values: dict[str, Any] = {
        "balance": UserWallet.balance - amount,
        "total_spent": UserWallet.total_spent + amount,
        "updated_at": now,
    }
    effective_wallet_name = wallet_name or name
    if effective_wallet_name:
        debit_values["name"] = effective_wallet_name
    debit_result = await session.execute(
        statement=update(UserWallet)
        .where(UserWallet.user_id == user_id, UserWallet.balance >= amount + held)
        .values(**debit_values)
        .returning(UserWallet.balance)
    )
    new_balance = debit_result.scalar_one_or_none()
    if new_balance is None:
        return None
    await _upsert_user_metadata_in_session(
        session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
    )
    return new_balance


async def transfer(  # noqa: PLR0913 -- transfer needs sender and receiver identity snapshots
    sender_id: int,
    sender_name: str,
    receiver_id: int,
    receiver_name: str,
    amount: int,
    sender_avatar_url: str = "",
    receiver_avatar_url: str = "",
) -> TransferResult | None:
    """Atomically transfers points from sender to receiver, burning a transfer tax.

    Serializes on `spending(sender_id)` and rejects when any table holds the entire
    wallet or `balance < amount + sender_held`. The sender is debited `amount`, the
    receiver is credited `amount - tax` (`tax = amount * TRANSFER_TAX_BPS // 10_000`),
    and the tax is burned.

    Args:
        sender_id (int): Discord user ID to debit.
        sender_name (str): Last-seen Discord username to store on the sender account.
        receiver_id (int): Discord user ID to credit.
        receiver_name (str): Last-seen Discord username to store on the receiver account.
        amount (int): Number of points to transfer.
        sender_avatar_url (str): Last-seen Discord avatar URL for the sender.
        receiver_avatar_url (str): Last-seen Discord avatar URL for the receiver.

    Returns:
        The post-transfer balances when committed, or `None` when validation
        failed or spendable funds were insufficient.
    """
    if amount <= 0 or sender_id == receiver_id:
        return None

    now = _database_now()
    async with spending(user_id=sender_id), open_session() as session:
        sender_balance = await _debit_spendable_in_session(
            session=session,
            user_id=sender_id,
            name=sender_name,
            avatar_url=sender_avatar_url,
            amount=amount,
            now=now,
        )
        if sender_balance is None:
            await session.rollback()
            return None

        tax = amount * TRANSFER_TAX_BPS // 10_000
        net = amount - tax
        credit_result = await _credit_with_repayment_in_session(
            session=session,
            user_id=receiver_id,
            name=receiver_name,
            avatar_url=receiver_avatar_url,
            amount=net,
            now=now,
        )

        await _commit_balance_write(session=session)
        return TransferResult(
            sender_balance=sender_balance,
            receiver_balance=credit_result.new_balance,
            received_amount=net,
            tax_amount=tax,
        )


async def top_n(
    limit: int | None = LEADERBOARD_SIZE, include_hidden: bool = False
) -> list[LeaderboardEntry]:
    """Returns accounts ordered by balance descending, omitting hidden accounts unless requested.

    `avatar_url` is empty when an account has never been seen by an avatar-aware write path.

    Args:
        limit (int | None): Maximum number of accounts to return, or `None` for all matching.
        include_hidden (bool): Whether to include accounts hidden from public leaderboards.
    """
    if limit is not None and limit <= 0:
        return []
    cache_key: _TopNCacheKey = (limit, include_hidden)
    cached_rows = _cached_leaderboard_rows(cache=_top_n_cache, cache_key=cache_key)
    if cached_rows is not None:
        return cached_rows
    async with open_session() as session:
        stmt = select(
            UserWallet.user_id, UserAccount.name, UserWallet.balance, UserAccount.avatar_url
        ).join(UserAccount, UserAccount.user_id == UserWallet.user_id)
        if not include_hidden:
            stmt = stmt.where(UserAccount.hide_from_leaderboard.is_(False))
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


async def top_losers(limit: int = LEADERBOARD_SIZE) -> list[LossLeaderboardEntry]:
    """Returns the biggest gross casino losers for the current Taipei day, loss descending.

    Omits hidden accounts and zero-loss rows; `loss_amount` is always positive.

    Args:
        limit (int): Maximum number of accounts to return.
    """
    if limit <= 0:
        return []
    now = _database_now()
    today_midnight = _taipei_midnight(now=now)
    cache_key: _TopLosersCacheKey = (limit, today_midnight)
    cached_rows = _cached_leaderboard_rows(cache=_top_losers_cache, cache_key=cache_key)
    if cached_rows is not None:
        return cached_rows

    async with open_session() as session:
        result = await session.execute(
            statement=select(
                CasinoAccount.user_id,
                UserAccount.name,
                UserAccount.avatar_url,
                CasinoAccount.daily_loss,
            )
            .select_from(CasinoAccount)
            .join(UserAccount, UserAccount.user_id == CasinoAccount.user_id)
            .where(
                CasinoAccount.day_started_at == today_midnight,
                CasinoAccount.daily_loss != "0",
                UserAccount.hide_from_leaderboard.is_(False),
            )
            .order_by(desc(func.length(CasinoAccount.daily_loss)), desc(CasinoAccount.daily_loss))
            .limit(limit=limit)
        )
        rows: list[LossLeaderboardEntry] = []
        for row in result.all():
            loss_amount = row[3]
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
        created_at=proposal.created_at,
    )


def _loan_contract_view(contract: LoanContract, now: datetime | None = None) -> LoanContractView:
    """Projects an ORM loan contract into an immutable API view.

    When `now` is provided, computes accrued interest in memory without mutating `contract`.
    """
    if now is None:
        interest_due = contract.interest_due
        last_interest_accrued_at = contract.last_interest_accrued_at
    else:
        interest_due, last_interest_accrued_at = _accrued_interest_for_contract(
            contract=contract, now=now
        )
    return LoanContractView(
        contract_id=contract.id,
        lender_type=LoanLenderType(contract.lender_type),
        lender_id=contract.lender_id,
        lender_name=contract.lender_name,
        borrower_id=contract.borrower_id,
        borrower_name=contract.borrower_name,
        principal_remaining=contract.principal_remaining,
        interest_due=interest_due,
        monthly_rate_bps=contract.monthly_rate_bps,
        opened_at=contract.opened_at,
        last_interest_accrued_at=last_interest_accrued_at,
        status=LoanContractStatus(contract.status),
    )


def _loan_proposal_is_expired(proposal: LoanProposal, now: datetime) -> bool:
    """Returns whether a pending loan proposal has passed its decision window."""
    if proposal.status != LoanProposalStatus.PENDING:
        return False
    elapsed_seconds = (_as_taipei(dt=now) - _as_taipei(dt=proposal.created_at)).total_seconds()
    return elapsed_seconds >= LOAN_PROPOSAL_TIMEOUT_SECONDS


async def _decide_pending_proposal_in_session(
    session: AsyncSession, proposal_id: int, status: LoanProposalStatus, now: datetime
) -> bool:
    """Moves a pending proposal to `status`, returning whether it was still pending."""
    result = await session.execute(
        statement=update(LoanProposal)
        .where(LoanProposal.id == proposal_id, LoanProposal.status == LoanProposalStatus.PENDING)
        .values(status=status, updated_at=now)
        .returning(LoanProposal.id)
    )
    return result.scalar_one_or_none() is not None


async def _reject_expired_loan_proposal_in_session(
    session: AsyncSession, proposal: LoanProposal, now: datetime
) -> LoanProposalView | None:
    """Marks an expired pending proposal as rejected inside the caller's session."""
    if not _loan_proposal_is_expired(proposal=proposal, now=now):
        return None
    if not await _decide_pending_proposal_in_session(
        session=session, proposal_id=proposal.id, status=LoanProposalStatus.REJECTED, now=now
    ):
        return None
    proposal.status = LoanProposalStatus.REJECTED
    proposal.updated_at = now
    return _loan_proposal_view(proposal=proposal)


async def _undecided_proposal_in_session(
    session: AsyncSession, proposal_id: int, now: datetime, creator_id: int | None = None
) -> LoanProposal | None:
    """Loads an unexpired pending proposal, or `None` if missing, decided, or wrong creator.

    Commits rejection and raises `LoanProposalExpiredError` if the proposal has expired.
    """
    statement = select(LoanProposal).where(
        LoanProposal.id == proposal_id, LoanProposal.status == LoanProposalStatus.PENDING
    )
    if creator_id is not None:
        statement = statement.where(LoanProposal.creator_id == creator_id)
    result = await session.execute(statement=statement)
    proposal = result.scalar_one_or_none()
    if proposal is None:
        return None
    expired = await _reject_expired_loan_proposal_in_session(
        session=session, proposal=proposal, now=now
    )
    if expired is not None:
        await session.commit()
        raise LoanProposalExpiredError
    return proposal


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
    interest = simple_interest(
        principal=principal_remaining, monthly_rate_bps=monthly_rate_bps, days=elapsed_days
    )
    return interest, _as_taipei(dt=last_accrued_at) + timedelta(days=elapsed_days)


def _accrued_interest_for_contract(contract: LoanContract, now: datetime) -> tuple[int, datetime]:
    """Returns `(interest_due, last_interest_accrued_at)` including unpersisted accrual at `now`."""
    if contract.status != LoanContractStatus.ACTIVE:
        return contract.interest_due, contract.last_interest_accrued_at
    interest, accrued_until = _loan_interest_delta(
        principal_remaining=contract.principal_remaining,
        monthly_rate_bps=contract.monthly_rate_bps,
        last_accrued_at=contract.last_interest_accrued_at,
        now=now,
    )
    if interest <= 0:
        return contract.interest_due, contract.last_interest_accrued_at
    return contract.interest_due + interest, accrued_until


def _accrue_contract_interest(contract: LoanContract, now: datetime) -> bool:
    """Applies lazy simple-interest accrual to an active ORM contract without flushing."""
    interest_due, accrued_until = _accrued_interest_for_contract(contract=contract, now=now)
    if interest_due == contract.interest_due:
        return False
    contract.interest_due = interest_due
    contract.last_interest_accrued_at = accrued_until
    contract.updated_at = now
    return True


async def _is_guild_participant_in_session(
    session: AsyncSession, guild_id: int, user_id: int
) -> bool:
    """Reports whether one user is recorded as taking part in one guild's economy."""
    result = await session.execute(
        statement=select(GuildParticipant.user_id)
        .where(GuildParticipant.guild_id == guild_id, GuildParticipant.user_id == user_id)
        .limit(limit=1)
    )
    return result.scalar_one_or_none() is not None


async def _outstanding_central_bank_principal_in_session(session: AsyncSession) -> int:
    """Returns every unpaid unit of central-bank principal, across all guilds."""
    result = await session.execute(
        statement=select(LoanContract.principal_remaining).where(
            LoanContract.lender_type == LoanLenderType.CENTRAL_BANK,
            LoanContract.status == LoanContractStatus.ACTIVE,
        )
    )
    return sum(result.scalars().all())


async def _central_bank_status_in_session(
    session: AsyncSession, guild_id: int
) -> CentralBankStatus:
    """Computes a guild's central-bank lending capacity from guild collateral minus global debt."""
    ledger_balance = await _central_bank_ledger_balance_in_session(session=session)
    participants = select(GuildParticipant.user_id).where(GuildParticipant.guild_id == guild_id)
    count_result = await session.execute(
        statement=select(func.count()).select_from(participants.subquery())
    )
    participant_count = count_result.scalar_one()

    # Subquery avoids SQLite's 32,766 bind-parameter limit as guild participants accumulate;
    # sum in Python because StoredInteger columns are decimal text and SQLite SUM uses floats.
    total_result = await session.execute(
        statement=select(UserWallet.balance).where(
            UserWallet.user_id.in_(participants.scalar_subquery())
        )
    )
    total_positive_user_balance = sum(
        balance for balance in total_result.scalars().all() if balance > 0
    )

    outstanding_principal = await _outstanding_central_bank_principal_in_session(session=session)
    # Central-bank loans mint into user balances: subtract outstanding principal once for
    # pre-loan collateral and once below for already-used capacity.
    base_lending_pool = max(total_positive_user_balance - outstanding_principal, 0)
    return CentralBankStatus(
        participant_count=participant_count,
        total_positive_user_balance=total_positive_user_balance,
        outstanding_principal=outstanding_principal,
        # Bank capital sits inside `- outstanding_principal` so a fully leveraged guild hits
        # zero instead of an unexhaustible floor; subtracted once since never minted into wallets.
        available_credit=max(
            base_lending_pool
            + CENTRAL_BANK_BASE_CAPACITY
            + ledger_balance
            - outstanding_principal,
            0,
        ),
        ledger_balance=ledger_balance,
    )


async def _user_total_debt_in_session(session: AsyncSession, user_id: int, now: datetime) -> int:
    """Returns a borrower's total active debt as of `now`, including unpersisted interest."""
    result = await session.execute(
        statement=select(
            LoanContract.principal_remaining,
            LoanContract.interest_due,
            LoanContract.monthly_rate_bps,
            LoanContract.last_interest_accrued_at,
        ).where(
            LoanContract.borrower_id == user_id, LoanContract.status == LoanContractStatus.ACTIVE
        )
    )
    return sum(
        principal
        + interest
        + _loan_interest_delta(
            principal_remaining=principal,
            monthly_rate_bps=monthly_rate_bps,
            last_accrued_at=last_accrued_at,
            now=now,
        )[0]
        for principal, interest, monthly_rate_bps, last_accrued_at in result.all()
    )


async def _credit_ceiling_in_session(session: AsyncSession, user_id: int, now: datetime) -> int:
    """Returns how much more central-bank credit one borrower may still draw as of `now`."""
    balance = await _balance_in_session(session=session, user_id=user_id)
    total_debt = await _user_total_debt_in_session(session=session, user_id=user_id, now=now)
    return central_bank_credit_ceiling(balance=balance, total_debt=total_debt)


async def get_central_bank_status(guild_id: int) -> CentralBankStatus:
    """Returns one guild's current central-bank lending capacity.

    Args:
        guild_id (int): Discord guild ID to inspect.
    """
    async with open_session() as session:
        return await _central_bank_status_in_session(session=session, guild_id=guild_id)


async def get_credit_ceiling(user_id: int) -> int:
    """Returns how much more central-bank credit `user_id` may still draw.

    Args:
        user_id (int): Discord user ID of the borrower.
    """
    async with open_session() as session:
        return await _credit_ceiling_in_session(
            session=session, user_id=user_id, now=_database_now()
        )


async def record_guild_participant(guild_id: int, user_id: int) -> None:
    """Records that `user_id` takes part in `guild_id`'s economy.

    Pass only a command caller or rewarded message author, never a `member:` target.

    Args:
        guild_id (int): Discord guild ID where the participation occurred.
        user_id (int): Discord user ID of the participating member.
    """
    now = _database_now()
    async with open_session() as session:
        await session.execute(
            statement=insert(GuildParticipant)
            .values(guild_id=guild_id, user_id=user_id, updated_at=now)
            .on_conflict_do_update(
                index_elements=["guild_id", "user_id"], set_={"updated_at": now}
            )
        )
        await session.commit()


async def _insert_loan_proposal(  # noqa: PLR0913 -- a proposal records both parties and its terms
    kind: LoanProposalKind,
    lender_type: LoanLenderType,
    borrower_id: int,
    borrower_name: str,
    borrower_avatar_url: str,
    lender_id: int | None,
    lender_name: str,
    lender_avatar_url: str,
    amount: int,
    monthly_rate_bps: int,
) -> LoanProposalView:
    """Records a pending proposal the borrower created, its rate clamped into range."""
    now = _database_now()
    async with open_session() as session:
        proposal = LoanProposal(
            kind=kind,
            status=LoanProposalStatus.PENDING,
            lender_type=lender_type,
            borrower_id=borrower_id,
            borrower_name=borrower_name or str(borrower_id),
            borrower_avatar_url=borrower_avatar_url,
            lender_id=lender_id,
            lender_name=lender_name,
            lender_avatar_url=lender_avatar_url,
            creator_id=borrower_id,
            amount=amount,
            monthly_rate_bps=clamp_loan_rate_bps(monthly_rate_bps=monthly_rate_bps),
            created_at=now,
            updated_at=now,
        )
        session.add(instance=proposal)
        await session.commit()
        return _loan_proposal_view(proposal=proposal)


async def create_personal_loan_request(  # noqa: PLR0913 -- proposal needs both identities
    borrower_id: int,
    borrower_name: str,
    lender_id: int,
    lender_name: str,
    amount: int,
    monthly_rate_bps: int = DEFAULT_LOAN_MONTHLY_RATE_BPS,
    borrower_avatar_url: str = "",
    lender_avatar_url: str = "",
) -> LoanProposalView:
    """Creates and returns a borrower-initiated personal loan request.

    Args:
        borrower_id (int): Discord user ID of the borrower.
        borrower_name (str): Last-seen Discord username of the borrower.
        lender_id (int): Discord user ID of the requested lender.
        lender_name (str): Last-seen Discord username of the requested lender.
        amount (int): Principal amount requested.
        monthly_rate_bps (int): Monthly simple-interest rate in basis points.
        borrower_avatar_url (str): Last-seen Discord avatar URL of the borrower.
        lender_avatar_url (str): Last-seen Discord avatar URL of the requested lender.

    Raises:
        ValueError: When `amount` is not positive or the borrower names themselves as lender.
    """
    if amount <= 0 or borrower_id == lender_id:
        msg = (
            "A personal loan request needs a positive amount and a lender other than the borrower"
        )
        raise ValueError(msg)
    return await _insert_loan_proposal(
        kind=LoanProposalKind.PERSONAL_REQUEST,
        lender_type=LoanLenderType.USER,
        borrower_id=borrower_id,
        borrower_name=borrower_name,
        borrower_avatar_url=borrower_avatar_url,
        lender_id=lender_id,
        lender_name=lender_name or str(lender_id),
        lender_avatar_url=lender_avatar_url,
        amount=amount,
        monthly_rate_bps=monthly_rate_bps,
    )


async def create_central_bank_loan_request(
    borrower_id: int,
    borrower_name: str,
    amount: int,
    monthly_rate_bps: int = DEFAULT_LOAN_MONTHLY_RATE_BPS,
    borrower_avatar_url: str = "",
) -> LoanProposalView:
    """Creates and returns a borrower-initiated central-bank loan request.

    Args:
        borrower_id (int): Discord user ID of the borrower.
        borrower_name (str): Last-seen Discord username of the borrower.
        amount (int): Principal amount requested.
        monthly_rate_bps (int): Monthly simple-interest rate in basis points.
        borrower_avatar_url (str): Last-seen Discord avatar URL of the borrower.

    Raises:
        ValueError: When `amount` is not positive.
    """
    if amount <= 0:
        msg = "A central-bank loan request needs a positive amount"
        raise ValueError(msg)
    return await _insert_loan_proposal(
        kind=LoanProposalKind.CENTRAL_BANK_REQUEST,
        lender_type=LoanLenderType.CENTRAL_BANK,
        borrower_id=borrower_id,
        borrower_name=borrower_name,
        borrower_avatar_url=borrower_avatar_url,
        lender_id=None,
        lender_name="Central Bank",
        lender_avatar_url="",
        amount=amount,
        monthly_rate_bps=monthly_rate_bps,
    )


async def reject_expired_loan_proposal(proposal_id: int) -> LoanProposalView | None:
    """Rejects a pending loan proposal if its decision window has expired.

    Args:
        proposal_id (int): Database ID of the loan proposal.

    Returns:
        The rejected proposal view, or `None` when not pending or not yet expired.
    """
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


async def reject_loan_proposals_created_before(cutoff: datetime) -> None:
    """Rejects every proposal still pending from before `cutoff`.

    Unsettled decision views do not survive process restarts.

    Args:
        cutoff (datetime): Proposals created before this timestamp are rejected.
    """
    async with open_session() as session:
        result = await session.execute(
            statement=update(LoanProposal)
            .where(
                LoanProposal.status == LoanProposalStatus.PENDING,
                # SQLite keeps the Taipei wall clock without its offset.
                LoanProposal.created_at < _as_taipei(dt=cutoff),
            )
            .values(status=LoanProposalStatus.REJECTED, updated_at=_database_now())
            .returning(LoanProposal.id)
        )
        proposal_ids = list(result.scalars())
        await session.commit()
    if proposal_ids:
        logfire.info(
            "Rejected loan requests an earlier process left waiting", proposal_ids=proposal_ids
        )


async def cancel_loan_proposal(proposal_id: int, actor_id: int) -> LoanProposalView | None:
    """Cancels a pending proposal created by `actor_id`.

    Args:
        proposal_id (int): Database ID of the pending loan proposal.
        actor_id (int): Discord user ID of the caller canceling the proposal.

    Returns:
        The canceled proposal view, or `None` when missing, already decided, or
        not created by `actor_id`.

    Raises:
        LoanProposalExpiredError: When this call found the request past its decision window
            and rejected it instead.
    """
    now = _database_now()
    async with open_session() as session:
        proposal = await _undecided_proposal_in_session(
            session=session, proposal_id=proposal_id, now=now, creator_id=actor_id
        )
        if proposal is None:
            return None
        if not await _decide_pending_proposal_in_session(
            session=session, proposal_id=proposal_id, status=LoanProposalStatus.CANCELED, now=now
        ):
            await session.rollback()
            return None
        proposal.status = LoanProposalStatus.CANCELED
        await session.commit()
        return _loan_proposal_view(proposal=proposal)


async def reject_loan_proposal(
    proposal_id: int, actor_id: int, approver_is_guild_admin: bool = False
) -> LoanProposalView | None:
    """Rejects a pending proposal when `actor_id` is allowed to decide it.

    Args:
        proposal_id (int): Database ID of the pending loan proposal.
        actor_id (int): Discord user ID of the caller rejecting the proposal.
        approver_is_guild_admin (bool): Whether `actor_id` administers the server.

    Returns:
        The rejected proposal view, or `None` when missing, already decided, or
        `actor_id` is not allowed to decide it.

    Raises:
        LoanProposalExpiredError: When this call found the request past its decision window
            and rejected it instead.
    """
    now = _database_now()
    async with open_session() as session:
        proposal = await _undecided_proposal_in_session(
            session=session, proposal_id=proposal_id, now=now
        )
        if proposal is None:
            return None
        allowed = False
        if proposal.kind == LoanProposalKind.PERSONAL_REQUEST:
            allowed = proposal.lender_id == actor_id
        elif proposal.kind == LoanProposalKind.CENTRAL_BANK_REQUEST:
            allowed = approver_is_guild_admin
        if not allowed:
            return None
        if not await _decide_pending_proposal_in_session(
            session=session, proposal_id=proposal_id, status=LoanProposalStatus.REJECTED, now=now
        ):
            await session.rollback()
            return None
        proposal.status = LoanProposalStatus.REJECTED
        await session.commit()
        return _loan_proposal_view(proposal=proposal)


async def _authorize_proposal_funding_in_session(  # noqa: PLR0913 -- validates and funds personal or central-bank proposal
    session: AsyncSession,
    proposal: LoanProposal,
    actor_id: int,
    actor_name: str,
    actor_avatar_url: str,
    approver_is_guild_admin: bool,
    guild_id: int | None,
    allow_central_bank_self_approval: bool,
    now: datetime,
) -> tuple[bool, int | None]:
    """Validates permissions and funding for `proposal`, returning `(authorized, lender_balance)`."""
    if proposal.kind == LoanProposalKind.PERSONAL_REQUEST:
        if proposal.lender_id != actor_id:
            return False, None
        lender_balance = await _debit_spendable_in_session(
            session=session,
            user_id=actor_id,
            name=actor_name,
            avatar_url=actor_avatar_url,
            amount=proposal.amount,
            now=now,
            wallet_name=actor_name or proposal.lender_name or str(actor_id),
        )
        if lender_balance is None:
            return False, None
        proposal.lender_name = actor_name or proposal.lender_name
        proposal.lender_avatar_url = actor_avatar_url or proposal.lender_avatar_url
        return True, lender_balance

    if proposal.kind == LoanProposalKind.CENTRAL_BANK_REQUEST:
        if (
            not approver_is_guild_admin
            or guild_id is None
            or (proposal.borrower_id == actor_id and not allow_central_bank_self_approval)
        ):
            return False, None
        # Recompute under BEGIN IMMEDIATE so concurrent approvals cannot double-spend capacity.
        central_status = await _central_bank_status_in_session(session=session, guild_id=guild_id)
        borrower_ceiling = await _credit_ceiling_in_session(
            session=session, user_id=proposal.borrower_id, now=now
        )
        return min(central_status.available_credit, borrower_ceiling) >= proposal.amount, None

    return False, None


async def accept_loan_proposal(  # noqa: PLR0913 -- public loan approval accepts actor and guild context
    proposal_id: int,
    actor_id: int,
    actor_name: str,
    actor_avatar_url: str = "",
    approver_is_guild_admin: bool = False,
    guild_id: int | None = None,
    allow_central_bank_self_approval: bool = False,
) -> LoanProposalAcceptResult | None:
    """Accepts a pending loan proposal and opens the loan contract.

    Args:
        proposal_id (int): Database ID of the pending loan proposal.
        actor_id (int): Discord user ID of the caller approving the proposal.
        actor_name (str): Last-seen Discord username of the caller.
        actor_avatar_url (str): Last-seen Discord avatar URL of the caller.
        approver_is_guild_admin (bool): Whether `actor_id` administers the server.
        guild_id (int | None): Discord guild ID for central-bank capacity checks.
        allow_central_bank_self_approval (bool): Whether a central-bank borrower may approve
            their own request.

    Returns:
        The opened contract and post-approval balances, or `None` when rejected by
        validation, permissions, or insufficient capacity or spendable funds.

    Raises:
        LoanProposalExpiredError: When this call found the request past its decision window
            and rejected it instead.
    """
    async with spending(user_id=actor_id), _loan_accept_lock.get(), open_session() as session:
        now = _database_now()
        # Acquire SQLite's write lock before reading capacity or proposal state.
        await session.execute(statement=text("BEGIN IMMEDIATE"))
        proposal = await _undecided_proposal_in_session(
            session=session, proposal_id=proposal_id, now=now
        )
        if proposal is None:
            await session.rollback()
            return None

        authorized, lender_balance = await _authorize_proposal_funding_in_session(
            session=session,
            proposal=proposal,
            actor_id=actor_id,
            actor_name=actor_name,
            actor_avatar_url=actor_avatar_url,
            approver_is_guild_admin=approver_is_guild_admin,
            guild_id=guild_id,
            allow_central_bank_self_approval=allow_central_bank_self_approval,
            now=now,
        )
        if not authorized:
            await session.rollback()
            return None

        if not await _decide_pending_proposal_in_session(
            session=session, proposal_id=proposal_id, status=LoanProposalStatus.ACCEPTED, now=now
        ):
            await session.rollback()
            return None

        credit_result = await _credit_with_repayment_in_session(
            session=session,
            user_id=proposal.borrower_id,
            name=proposal.borrower_name,
            avatar_url=proposal.borrower_avatar_url,
            amount=proposal.amount,
            now=now,
        )
        # Front-load MIN_INTEREST_DAYS of interest and advance last_interest_accrued_at past the
        # window so immediate repayment still pays the minimum and later accrual resumes normally.
        prepaid_interest = simple_interest(
            principal=proposal.amount,
            monthly_rate_bps=proposal.monthly_rate_bps,
            days=MIN_INTEREST_DAYS,
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
        session.add(instance=contract)
        await session.flush()
        central_bank_available_credit: int | None = None
        if proposal.kind == LoanProposalKind.CENTRAL_BANK_REQUEST and guild_id is not None:
            central_status = await _central_bank_status_in_session(
                session=session, guild_id=guild_id
            )
            central_bank_available_credit = central_status.available_credit
        await _commit_balance_write(session=session)
        return LoanProposalAcceptResult(
            contract=_loan_contract_view(contract=contract),
            borrower_balance=credit_result.new_balance,
            lender_balance=lender_balance,
            central_bank_available_credit=central_bank_available_credit,
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


async def _pay_lender_side_in_session(
    session: AsyncSession, contract: LoanContract, paid: int, interest_paid: int, now: datetime
) -> int | None:
    """Credits a contract payment to the lender, returning a personal lender's balance or None."""
    if contract.lender_type == LoanLenderType.CENTRAL_BANK:
        await _credit_central_bank_ledger_in_session(
            session=session, amount=interest_paid, now=now
        )
        return None
    if contract.lender_id is None:
        return None
    # The contract's lender identity dates from acceptance, so an empty one keeps the newer
    # stored name and avatar.
    credit_result = await _credit_with_repayment_in_session(
        session=session, user_id=contract.lender_id, name="", avatar_url="", amount=paid, now=now
    )
    return credit_result.new_balance


async def _apply_loan_payment_in_session(  # noqa: PLR0913 -- payment needs actor identity, contract set, and hold floor
    session: AsyncSession,
    contracts: Sequence[LoanContract],
    borrower_id: int,
    borrower_name: str,
    borrower_avatar_url: str,
    amount: int,
    now: datetime,
    floor: int = 0,
) -> LoanPaymentResult | None:
    """Applies a repayment or forced collection across already-accrued ordered contracts."""
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
            floor=floor,
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

        credited_lender_balance = await _pay_lender_side_in_session(
            session=session, contract=contract, paid=paid, interest_paid=interest_paid, now=now
        )
        if credited_lender_balance is not None:
            lender_balance = credited_lender_balance

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


async def _collect_loan_payment(  # noqa: PLR0913 -- a payment names both sides, the amount and its scope
    borrower_id: int,
    borrower_name: str,
    borrower_avatar_url: str,
    lender_type: LoanLenderType,
    lender_id: int | None,
    amount: int | None,
    guild_id: int | None = None,
) -> LoanPaymentResult | None:
    """Applies a repayment or forced collection across active contracts, or `None` if unpaid."""
    now = _database_now()
    async with spending(user_id=borrower_id), open_session() as session:
        # Both voluntary repayment and forced collection leave held table stakes untouched.
        held = held_amount(user_id=borrower_id)
        if held is None:
            return None
        if guild_id is not None and not await _is_guild_participant_in_session(
            session=session, guild_id=guild_id, user_id=borrower_id
        ):
            return None
        contracts = await _loan_contracts_for_payment_in_session(
            session=session, borrower_id=borrower_id, lender_type=lender_type, lender_id=lender_id
        )
        if not contracts:
            return None
        accrued_any = False
        for contract in contracts:
            accrued_any = _accrue_contract_interest(contract=contract, now=now) or accrued_any
        if accrued_any:
            await session.flush()
        if amount is None:
            total_owed = sum(
                contract.principal_remaining + contract.interest_due for contract in contracts
            )
            amount = max(total_owed, 1)
        result = await _apply_loan_payment_in_session(
            session=session,
            contracts=contracts,
            borrower_id=borrower_id,
            borrower_name=borrower_name,
            borrower_avatar_url=borrower_avatar_url,
            amount=amount,
            now=now,
            floor=held,
        )
        if result is None:
            await session.rollback()
            return None
        await _commit_balance_write(session=session)
        return result


async def repay_personal_loans(
    borrower_id: int,
    borrower_name: str,
    lender_id: int,
    amount: int,
    borrower_avatar_url: str = "",
) -> LoanPaymentResult | None:
    """Repays active personal loans from `borrower_id` to `lender_id`.

    Args:
        borrower_id (int): Discord user ID of the repaying borrower.
        borrower_name (str): Last-seen Discord username of the borrower.
        lender_id (int): Discord user ID of the lender to repay.
        amount (int): Maximum points to apply against active debt.
        borrower_avatar_url (str): Last-seen Discord avatar URL of the borrower.

    Returns:
        The committed payment breakdown, or `None` when nothing was paid.
    """
    return await _collect_loan_payment(
        borrower_id=borrower_id,
        borrower_name=borrower_name,
        borrower_avatar_url=borrower_avatar_url,
        lender_type=LoanLenderType.USER,
        lender_id=lender_id,
        amount=amount,
    )


async def call_personal_loans(
    lender_id: int,
    borrower_id: int,
    borrower_name: str,
    amount: int | None = None,
    borrower_avatar_url: str = "",
) -> LoanPaymentResult | None:
    """Forcibly collects active personal loans owed to `lender_id`.

    Args:
        lender_id (int): Discord user ID of the collecting lender.
        borrower_id (int): Discord user ID of the borrower to collect from.
        borrower_name (str): Last-seen Discord username of the borrower.
        amount (int | None): Maximum points to collect, or `None` for all owed.
        borrower_avatar_url (str): Last-seen Discord avatar URL of the borrower.

    Returns:
        The committed collection breakdown, or `None` when nothing was collected.
    """
    return await _collect_loan_payment(
        borrower_id=borrower_id,
        borrower_name=borrower_name,
        borrower_avatar_url=borrower_avatar_url,
        lender_type=LoanLenderType.USER,
        lender_id=lender_id,
        amount=amount,
    )


async def repay_central_bank_loans(
    borrower_id: int, borrower_name: str, amount: int, borrower_avatar_url: str = ""
) -> LoanPaymentResult | None:
    """Repays active central-bank loans for a borrower.

    Args:
        borrower_id (int): Discord user ID of the repaying borrower.
        borrower_name (str): Last-seen Discord username of the borrower.
        amount (int): Maximum points to apply against active debt.
        borrower_avatar_url (str): Last-seen Discord avatar URL of the borrower.

    Returns:
        The committed payment breakdown, or `None` when nothing was paid.
    """
    return await _collect_loan_payment(
        borrower_id=borrower_id,
        borrower_name=borrower_name,
        borrower_avatar_url=borrower_avatar_url,
        lender_type=LoanLenderType.CENTRAL_BANK,
        lender_id=None,
        amount=amount,
    )


async def call_central_bank_loans(
    guild_id: int,
    borrower_id: int,
    borrower_name: str,
    amount: int | None = None,
    borrower_avatar_url: str = "",
) -> LoanPaymentResult | None:
    """Forcibly collects active central-bank loans from a borrower.

    Refuses borrowers not recorded in `guild_id`'s `guild_participant` table.

    Args:
        guild_id (int): Discord guild ID whose economy `borrower_id` must participate in.
        borrower_id (int): Discord user ID of the borrower to collect from.
        borrower_name (str): Last-seen Discord username of the borrower.
        amount (int | None): Maximum points to collect, or `None` for all owed.
        borrower_avatar_url (str): Last-seen Discord avatar URL of the borrower.

    Returns:
        The committed collection breakdown, or `None` when refused or nothing was collected.
    """
    return await _collect_loan_payment(
        borrower_id=borrower_id,
        borrower_name=borrower_name,
        borrower_avatar_url=borrower_avatar_url,
        lender_type=LoanLenderType.CENTRAL_BANK,
        lender_id=None,
        amount=amount,
        guild_id=guild_id,
    )


async def list_loan_contracts(user_id: int) -> list[LoanContractView]:
    """Returns active loan contracts where `user_id` is borrower or personal lender.

    Computes accrued interest in memory without writing to the database.

    Args:
        user_id (int): Discord user ID to inspect.
    """
    now = _database_now()
    async with open_session() as session:
        result = await session.execute(
            statement=select(LoanContract)
            .where(
                (LoanContract.borrower_id == user_id) | (LoanContract.lender_id == user_id),
                LoanContract.status == LoanContractStatus.ACTIVE,
            )
            .order_by(LoanContract.opened_at, LoanContract.id)
        )
        contracts = list(result.scalars().all())
        return [_loan_contract_view(contract=contract, now=now) for contract in contracts]


async def _portfolio_in_session(
    session: AsyncSession, user_id: int, now: datetime
) -> PortfolioView:
    """Builds a read-only portfolio view, including unpersisted interest accrual."""
    account_result = await session.execute(
        statement=select(UserAccount.name, UserAccount.is_vip, UserWallet.balance)
        .select_from(UserAccount)
        .outerjoin(UserWallet, UserWallet.user_id == UserAccount.user_id)
        .where(UserAccount.user_id == user_id)
    )
    account_row = account_result.one_or_none()
    name = str(user_id)
    is_vip = False
    balance = 0
    if account_row is not None:
        name = account_row[0]
        is_vip = bool(account_row[1])
        balance = account_row[2] or 0

    debt_result = await session.execute(
        statement=select(
            LoanContract.principal_remaining,
            LoanContract.interest_due,
            LoanContract.monthly_rate_bps,
            LoanContract.last_interest_accrued_at,
        ).where(
            LoanContract.borrower_id == user_id, LoanContract.status == LoanContractStatus.ACTIVE
        )
    )
    debt_principal = 0
    debt_interest = 0
    for principal, interest, monthly_rate_bps, last_accrued_at in debt_result.all():
        debt_principal += principal
        debt_interest += (
            interest
            + _loan_interest_delta(
                principal_remaining=principal,
                monthly_rate_bps=monthly_rate_bps,
                last_accrued_at=last_accrued_at,
                now=now,
            )[0]
        )

    return PortfolioView(
        user_id=user_id,
        name=name,
        balance=balance,
        debt_principal=debt_principal,
        debt_interest=debt_interest,
        net_worth=balance - debt_principal - debt_interest,
        is_vip=is_vip,
    )


async def get_portfolio(user_id: int) -> PortfolioView:
    """Returns a user's current portfolio and estimated net worth.

    Computes accrued interest in memory without writing to the database.

    Args:
        user_id (int): Discord user ID to inspect.
    """
    now = _database_now()
    async with open_session() as session:
        return await _portfolio_in_session(session=session, user_id=user_id, now=now)
