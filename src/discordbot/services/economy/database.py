"""Persistent point-balance store for the economy cog.

The engine is a module-level `AsyncEngine` singleton. Putting
`create_async_engine()` on a per-instance `cached_property` would leak the
connection pool, dialect cache, and inspector cache for every Discord
interaction.

Every balance-mutating write path is atomic at the SQLite transaction level.
Most paths are a single UPSERT (`INSERT ... ON CONFLICT DO UPDATE`) or a
conditional `UPDATE ... WHERE ... RETURNING`; multi-row finance paths still roll
back as one unit when a conditional write loses a race. Reading a row into
Python and writing the mutated value back would lose an update whenever two
coroutines race on the same user, and would raise `IntegrityError` when two of
them insert the same brand-new user.

We use `aiosqlite` so every DB call stays on the event loop. Each operation
opens an `AsyncSession` bound to the current `_engine`, so tests can
monkeypatch `_engine` per-test and every subsequent call sees the swap.

VIP bumps the player's winning payout from games and is permanent once set.
Admin status gates maintenance-only economy commands and is set out-of-band by a
direct DB write; `set_admin` exists for that path rather than for a runtime
caller. Daily casino counters live on `casino_account` so a current-day loss
ranking needs no audit-log scan.

Personal loan requests debit the lender on acceptance, and central-bank loans
mint borrower balance on approval. What bounds that minting is the per-borrower
ceiling in `typings/economy.py`, not the approver: approval belongs to a Discord
server administrator, and anyone can become one by creating a server. The
per-guild lending pool is a second, looser throttle on top, and `guild_participant`
is what says whose balance backs which guild.

Shared jackpot pools and the casino ledger live in the same `economy.db` file
as the per-user rows, so runtime casino and jackpot settlement applies the
player delta and the house-side mirror in one atomic SQLite transaction.
"""

from time import monotonic
from typing import Any, Final, Literal
import asyncio
from datetime import datetime, timedelta
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
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, AsyncConnection, create_async_engine
from sqlalchemy.dialects.sqlite import insert

from discordbot.utils.timezone import as_taipei as _as_taipei
from discordbot.utils.timezone import database_now as _database_now
from discordbot.typings.economy import (
    TRANSFER_TAX_BPS,
    MIN_INTEREST_DAYS,
    VIP_PURCHASE_COST,
    MAX_LOAN_MONTHLY_RATE_BPS,
    MIN_LOAN_MONTHLY_RATE_BPS,
    CENTRAL_BANK_BASE_CAPACITY,
    DEFAULT_LOAN_MONTHLY_RATE_BPS,
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
    CreditResult,
    PortfolioView,
    LoanLenderType,
    TransferResult,
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
    JackpotSettlementBatchResult,
    central_bank_credit_ceiling,
)
from discordbot.utils.asyncio_locks import LoopLocalLock
from discordbot.utils.sqlite_config import SqliteBootstrap
from discordbot.utils.stored_integer import StoredInteger, int_add_text, int_compare_text
from discordbot.utils.stored_integer import stored_int_to_int as _stored_int_to_int
from discordbot.utils.stored_integer import stored_int_to_text as _stored_int_to_text

# SELECT-then-conditional-UPDATE loops keep a small retry budget. The bound is
# there to stop a degenerate hot-row livelock, not to ride out contention.
_VIP_PURCHASE_MAX_RETRIES: Final[int] = 8
_CLAMPED_DELTA_MAX_RETRIES: Final[int] = 8
_JACKPOT_CLAIM_MAX_RETRIES: Final[int] = 8
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
        name: Last-seen Discord username (refreshed on every write).
        avatar_url: Last-seen Discord avatar URL (refreshed on writes that carry it).
        updated_at: Taiwan-local timestamp of the last write.
        is_vip: Permanent VIP flag toggled by a successful `/vip` purchase.
        is_admin: Whether the user can run Discord-side economy admin commands.
        is_central_banker: Dead. Central-bank approval is a Discord server
            administrator's now, read off the interaction. The column stays
            because `_ensure_schema` is one `create_all`, which never alters an
            existing table, so dropping it would break a deployed database.
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
        # No query filters on the balance alone — the two that mention it pin the primary
        # key as well — and the ranking sort is a computed integer-aware expression this
        # cannot satisfy either. It stays because the schema is never altered in place.
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


class GuildParticipant(Base):
    """One user recorded as taking part in one guild's economy.

    Identity and balance are cross-server by design, so nothing on `user_wallet`
    says which guild a balance should back. This table is the only thing that
    does, and the central bank reads it to decide whose money backs a guild's
    lending. It cannot be derived instead: the gateway runs without the members
    intent, so the bot cannot enumerate a guild's membership at all.

    A row records the CALLER of an economy command or the author of a rewarded
    message. It must never record the target of a `member:` option, which would
    let anyone import a stranger's balance into a pool they administer.

    Nothing ever removes a row. Leaving is invisible here — the gateway runs
    without the members intent — so a sweep would have to guess, and guessing
    wrong takes collateral away from a guild whose member simply went quiet. The
    cost of keeping them is bounded by what the row backs: a departed member's
    balance still counts toward that guild's collateral, which the whole bank's
    outstanding principal is subtracted from either way.
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
        name: Last-seen Discord username for quick inspection.
        day_started_at: Asia/Taipei midnight for the stored counters.
        daily_loss: Current-day gross loss from player-side casino settlements, stored as a decimal string.
        daily_win: Current-day gross win from player-side casino settlements, stored as a decimal string.
        daily_net: Current-day signed net casino result, stored as a decimal string.
        updated_at: Taiwan-local timestamp of the last casino counter write.
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
    # Always zero: nothing escrows a proposal. The column stays because
    # `_ensure_schema` is one `create_all`, which never alters an existing table,
    # so dropping it would break a deployed database.
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


class JackpotPool(Base):
    """Per-game cumulative jackpot shared across every table of that game.

    One row per game (keyed by `game_id`). Wager flows update
    `pool_balance` atomically while `total_contributed` /
    `total_claimed` accumulate gross in/out flows so the seeded
    on-the-house amount stays distinguishable from organic player
    contributions.

    Attributes:
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
    player losses flow in. The bot sits at the table as an ordinary player, so
    its `user_wallet` is not the house ledger; this row is. `balance` may go
    negative when payouts exceed take-in; `total_earned` and `total_spent`
    accumulate gross flows so a reader can see direction of volume, not just net.

    Attributes:
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


class CentralBankLedger(Base):
    """Interest the central bank has kept, which it lends out again.

    Central-bank principal is minted on approval and burned on repayment, so it
    nets to nothing. The interest on top used to be burned with it; it is kept
    here instead and added to the starting capital in `typings/economy.py`, the
    two together being the bank's own money to lend. The row holds earnings only,
    never the starting capital, so a database created before it existed starts
    at zero and needs nothing seeded into it.

    Attributes:
        balance: Interest kept and available to lend again.
        total_earned: Lifetime gross interest, so a reader sees volume, not just net.
        updated_at: Taiwan-local timestamp of the last write.
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

_loan_accept_lock = LoopLocalLock()
type _TopNCacheKey = tuple[int, int | None, bool]
type _TopLosersCacheKey = tuple[int, int, bool, datetime]
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


def _current_loan_accept_lock() -> asyncio.Lock:
    """Serializes loan approval so central-bank capacity is consumed once."""
    return _loan_accept_lock.get()


async def _seed_singleton_rows(conn: AsyncConnection) -> None:
    """Seeds the jackpot pools and the two ledgers, inside `create_all`'s transaction.

    All of them are singleton rows the rest of this module assumes exist, so they are
    written in the same transaction that creates the tables rather than in one of their
    own. Every insert ignores a conflict, which is what makes a repeat bootstrap on the
    same file a no-op instead of a reset — including on a database that predates one of
    these tables, where `create_all` adds the table and this seeds it on the same boot.

    Args:
        conn: The open connection `create_all` just ran on.
    """
    for seed_game_id, seed_amount in _JACKPOT_SEEDS.items():
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
    await conn.execute(
        statement=insert(CentralBankLedger)
        .values(
            ledger_id=CENTRAL_BANK_LEDGER_ID,
            balance="0",
            total_earned="0",
            updated_at=_database_now(),
        )
        .on_conflict_do_nothing(index_elements=["ledger_id"])
    )


# Foreign keys are enabled defensively for any future FK constraint.
_database = SqliteBootstrap(
    metadata=Base.metadata, enable_foreign_keys=True, after_create=_seed_singleton_rows
)
_database.install_hooks(engine=_engine)


async def _ensure_schema() -> None:
    """Bootstraps the economy schema, jackpot seeds, and casino ledger once per engine."""
    await _database.ensure_schema(engine=_engine)


def open_session() -> AsyncSession:
    """Creates an async session bound to the current economy database engine.

    Returns:
        An `AsyncSession` using the current module-level `_engine`.
    """
    return _database.open_session(engine=_engine)


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

    Only an explicitly unclamped adjustment may use this; a player-side loss
    clamps at zero, and the casino's own negative P&L is a `casino_ledger` row
    rather than a wallet. `total_earned` / `total_spent` still accumulate gross
    flows, so `balance == total_earned - total_spent` holds through a negative
    balance.

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
    invalidate_economy_leaderboard_cache()


async def _credit_with_repayment_in_session(  # noqa: PLR0913 -- session helper keeps income writes atomic
    session: AsyncSession, user_id: int, name: str, avatar_url: str, amount: int, now: datetime
) -> CreditResult:
    """Credits income inside the caller's transaction.

    Despite the name, nothing here repays a loan: income does not settle debt,
    which takes an explicit repayment or collection. The name stays because
    routing income through a single facade is deliberate. Caller must guarantee
    `amount > 0`.
    """
    await _upsert_user_metadata_in_session(
        session=session, user_id=user_id, name=name, avatar_url=avatar_url, now=now
    )
    result = await session.execute(
        statement=_build_credit_upsert(user_id=user_id, name=name, amount=amount, now=now)
    )
    new_balance = result.scalar_one()
    invalidate_economy_leaderboard_cache()
    return CreditResult(new_balance=new_balance, credited_amount=amount)


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
    invalidate_economy_leaderboard_cache()
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
    if applied != 0:
        invalidate_economy_leaderboard_cache()
    return new_balance, applied


async def _apply_signed_delta_in_session(  # noqa: PLR0913 -- session helper needs identity and signed delta
    session: AsyncSession, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
) -> int:
    """Applies a signed delta without clamping.

    Player-side losses use the clamped path, and the casino mirror has its own
    row writer (`_apply_casino_ledger_delta_in_session`).
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

    Args:
        session: Active SQLAlchemy session bound to `_engine`.
        delta: Signed change to apply to the casino balance.
        now: `_database_now()` value pinned for this transaction.

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
    """Books interest the central bank just collected, in the payment's transaction.

    Args:
        session: Active SQLAlchemy session bound to `_engine`.
        amount: Interest collected; non-positive amounts are ignored.
        now: `_database_now()` value pinned for this transaction.
    """
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


async def _rollback_sessions(*sessions: AsyncSession) -> None:
    """Rolls back sessions without masking the original settlement exception."""
    for session in sessions:
        try:
            await session.rollback()
        except Exception:
            logfire.warn("Failed to roll back settlement session", _exc_info=True)


async def get_casino_ledger() -> CasinoLedgerSnapshot:
    """Returns the cumulative casino system ledger snapshot."""
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
    """Applies a casino or jackpot player delta and returns the balance plus applied delta.

    Positive deltas take the shared income path and count as fully applied.
    Negative deltas clamp at zero so a casino or Dragon Gate loss cannot drive
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

    A long-term loan is settled only by an explicit repayment or collection, so
    income lands fully in balance and only increases `total_earned`.

    Args:
        user_id: Discord user ID receiving the credit.
        name: Last-seen Discord username to store on the account.
        amount: Gross income amount; a non-positive amount credits nothing
            and returns the current balance.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        Outcome capturing post-credit balance.
    """
    await _ensure_schema()
    if amount <= 0:
        return CreditResult(new_balance=await get_balance(user_id=user_id), credited_amount=0)
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
        invalidate_economy_leaderboard_cache()
        return BalanceAdjustmentResult(new_balance=new_balance, applied_delta=applied_delta)


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
    ledger records less by exactly what was left uncollected, and by nothing
    else — `player_delta` carries system-funded bonuses the house never paid.
    The player write and the casino mirror live in the same
    `data/database/economy.db` file and commit as one atomic transaction.

    Args:
        player_id: Discord user ID for the player account.
        player_account_name: Account name to store for the player.
        player_avatar_url: Last-seen Discord avatar URL for the player.
        player_delta: Signed net change for the player. Losses are clamped at
            zero and may apply less than the requested debit.
        casino_delta: Signed change to apply to the casino ledger balance.

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

            # What the casino books is reduced by what could not be collected, and by nothing
            # else. Capping it at the player's NET movement instead would also deduct a
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

    Seeded pools are replenished before returning if an older process left them
    drained. Returns `0` when the row hasn't been seeded yet so a
    freshly-introduced game can short-circuit cleanly.

    Args:
        game_id: Game identifier (e.g. `"dragon_gate"`).

    Returns:
        The current pool balance in points.
    """
    snapshot = await get_jackpot_snapshot(game_id=game_id)
    return snapshot.balance


async def get_jackpot_snapshot(game_id: str) -> JackpotSnapshot:
    """Returns the current jackpot balance and generation for a shared pool."""
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
    """Applies a signed delta to a game's jackpot pool inside the caller's session.

    Positive deltas accumulate `total_contributed` (player losses /
    antes flowing into the pool); negative deltas accumulate
    `total_claimed` with the absolute value (winning payouts flowing
    out). Seeded pools are topped back up automatically after a drain, so
    the returned balance is always ready for the next table.

    Args:
        session: Active SQLAlchemy session bound to `_engine`.
        game_id: Game identifier (jackpot row primary key).
        delta: Signed point adjustment to apply to `pool_balance`.
        now: `_database_now()` value pinned for this transaction.

    Returns:
        The pool snapshot after the write and any automatic replenishment,
        plus whether this write depleted the pool.
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
    jackpot_depleted = pool_balance <= 0 and _JACKPOT_SEEDS.get(game_id, 0) > 0
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
) -> JackpotSettlementResult:
    """Atomic player-and-jackpot settlement for a single wager event.

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
    this transaction, then credited through the shared income path. Negative
    deltas normally clamp at zero and feed the pool with the actual debit.
    Required-full-debit settlements reject the whole batch instead. If a seeded
    pool is drained, the same transaction restores its on-the-house seed.
    Player and jackpot rows live in the same `data/database/economy.db` file,
    so the whole batch commits as one atomic transaction.

    Args:
        game_id: Jackpot game identifier (e.g. `"dragon_gate"`).
        settlements: Player-side settlements to apply in order.

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
                return await _rejected_jackpot_batch_result(
                    session=session,
                    game_id=game_id,
                    rejected_player_ids=rejected_player_ids,
                    now=now,
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

                (player_balance, applied_player_delta) = await _apply_player_delta_in_session(
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
                    return await _rejected_jackpot_batch_result(
                        session=session,
                        game_id=game_id,
                        rejected_player_ids=(settlement.player_id,),
                        now=now,
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


async def buy_vip(user_id: int, name: str, avatar_url: str = "") -> VipPurchaseResult | None:
    """Promotes the user to VIP after debiting `VIP_PURCHASE_COST` points.

    Returns `None` when the user is missing either the account or the wallet
    row, is already VIP, has insufficient balance, or the retry budget for the
    conditional UPDATE was exhausted.

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
            invalidate_economy_leaderboard_cache()
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


async def _set_account_flag(
    user_id: int, name: str, flag: Literal["is_admin"], value: bool, avatar_url: str
) -> bool:
    """Grants or revokes one `user_account` permission flag.

    Granting creates the identity row if the user has never touched the economy
    system; no wallet row is created, so the balance still reads 0. Revoking
    updates an existing row only; missing users are left untouched so revoke
    operations do not create empty account rows.

    Returns:
        `True` when a row was created or updated; `False` when revoking a
        missing user.
    """
    await _ensure_schema()
    now = _database_now()
    effective_name = name or str(user_id)
    values: dict[str, Any] = {flag: value, "updated_at": now}
    if name:
        values["name"] = effective_name
    if avatar_url:
        values["avatar_url"] = avatar_url
    async with open_session() as session:
        if value:
            insert_values: dict[str, Any] = {
                "user_id": user_id,
                "name": effective_name,
                "avatar_url": avatar_url,
                "updated_at": now,
                "is_vip": False,
                "is_admin": False,
                "is_central_banker": False,
                flag: True,
            }
            statement = (
                insert(UserAccount)
                .values(**insert_values)
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


async def set_admin(user_id: int, name: str, is_admin: bool, avatar_url: str = "") -> bool:
    """Sets the economy admin flag for a Discord user.

    Args:
        user_id: Discord user ID to modify.
        name: Last-seen Discord username to store when available.
        is_admin: Desired admin flag value.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        `True` when a row was created or updated; `False` when revoking a
        missing user.
    """
    return await _set_account_flag(
        user_id=user_id, name=name, flag="is_admin", value=is_admin, avatar_url=avatar_url
    )


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


async def top_n(limit: int | None = 10, include_hidden: bool = False) -> list[LeaderboardEntry]:
    """Returns accounts ordered by balance descending.

    Hidden accounts are the only rows dropped (`include_hidden`): the bot ranks
    as an ordinary player, and the casino's own P&L is a `casino_ledger` row
    rather than a wallet. Stored integer values are sorted in SQL with explicit
    decimal-text aware order terms so the query can still apply `LIMIT` before
    rows reach Python.

    Args:
        limit: Maximum number of accounts to return, or `None` to return all
            matching accounts.
        include_hidden: Whether to include accounts marked as hidden from
            public leaderboards.

    Returns:
        Leaderboard entries ordered by balance descending. `avatar_url` is
        empty when the user has never been seen by an avatar-aware write path.
    """
    await _ensure_schema()
    if limit is not None and limit <= 0:
        return []
    cache_key: _TopNCacheKey = (id(_engine), limit, include_hidden)
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


async def top_losers(limit: int = 10, include_hidden: bool = False) -> list[LossLeaderboardEntry]:
    """Returns the biggest gross casino losers for the current Taipei day.

    The leaderboard reads persisted `casino_account` daily counters. Writes lazily reset stale
    counters at the first casino settlement after Taipei midnight, while this
    query filters by today's `day_started_at` so yesterday's counters
    never leak into a new day.

    Args:
        limit: Maximum number of accounts to return.
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
    cache_key: _TopLosersCacheKey = (id(_engine), limit, include_hidden, today_midnight)
    cached_rows = _cached_leaderboard_rows(cache=_top_losers_cache, cache_key=cache_key)
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


async def _is_guild_participant_in_session(
    session: AsyncSession, guild_id: int, user_id: int
) -> bool:
    """Reports whether one user is recorded as taking part in one guild's economy."""
    result = await session.execute(
        statement=select(GuildParticipant.user_id)
        .where(GuildParticipant.guild_id == guild_id, GuildParticipant.user_id == user_id)
        .limit(1)
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
    session: AsyncSession, guild_id: int, exclude_user_ids: tuple[int, ...] = ()
) -> CentralBankStatus:
    """Computes one guild's central-bank lending capacity.

    What is per guild is the COLLATERAL: a guild lends against the balances of the
    people who take part in it. What is not, and cannot be, is the debt. Wallets
    cross servers and debt does not follow them, so a loan minted in one guild and
    handed to somebody who takes part in another arrives there as collateral with
    nothing owed against it. Charging only the local participants' debt therefore
    stops bounding anything the moment three accounts hold accounts in three
    guilds: measured on this code, 1,000 became 686,826,650,532 in forty rounds of
    borrow-then-`/give` and was still accelerating. Subtracting the whole bank's
    outstanding principal is what closes that, at the cost of making the credit
    budget shared — a guild's own wealth decides how much of the bank it may draw
    on, not how much the bank has.
    """
    ledger_balance = await _central_bank_ledger_balance_in_session(session=session)
    participants = select(GuildParticipant.user_id).where(GuildParticipant.guild_id == guild_id)
    if exclude_user_ids:
        participants = participants.where(GuildParticipant.user_id.notin_(other=exclude_user_ids))
    count_result = await session.execute(
        statement=select(func.count()).select_from(participants.subquery())
    )
    participant_count = count_result.scalar_one()

    # Joined rather than fetching the ids and binding them into an `IN (...)`: a guild's
    # participants only ever accumulate, and one bind parameter per id stops working at
    # SQLite's 32,766-variable ceiling. Summed in Python rather than by the database,
    # because money columns are decimal text and SQLite's own SUM reads them as floats.
    total_result = await session.execute(
        statement=select(UserWallet.balance).where(
            UserWallet.user_id.in_(participants.scalar_subquery())
        )
    )
    total_positive_user_balance = sum(
        balance for balance in total_result.scalars().all() if balance > 0
    )

    outstanding_principal = await _outstanding_central_bank_principal_in_session(session=session)
    # Central-bank loans mint into user balances, so subtract outstanding
    # principal once to estimate the pre-loan pool and once for already-used
    # capacity.
    base_lending_pool = max(total_positive_user_balance - outstanding_principal, 0)
    return CentralBankStatus(
        participant_count=participant_count,
        total_positive_user_balance=total_positive_user_balance,
        outstanding_principal=outstanding_principal,
        # The bank's own capital — its starting capital plus the interest it has kept — is
        # INSIDE the outstanding subtraction, so lending depletes it and a fully leveraged
        # guild reaches zero. Adding it outside instead pins the pool at a floor it can
        # never fall through, which takes the pool out of the bounding job altogether: the
        # per-borrower ceiling cannot cover for it, because a ceiling clamped at zero stops
        # charging the debt of a borrower who has given their balance away, and a pair
        # alternating `/give` then mints without limit. Measured. Subtracted once rather
        # than twice because, unlike the participants' balances, that capital was never
        # minted into anybody's wallet. Lending the kept interest back out cannot be farmed
        # either: every unit of it was paid out of a borrower's own balance first.
        available_credit=max(
            base_lending_pool
            + CENTRAL_BANK_BASE_CAPACITY
            + ledger_balance
            - outstanding_principal,
            0,
        ),
        ledger_balance=ledger_balance,
    )


async def _user_total_debt_in_session(session: AsyncSession, user_id: int) -> int:
    """Returns everything one borrower owes across every active contract."""
    result = await session.execute(
        statement=select(LoanContract.principal_remaining, LoanContract.interest_due).where(
            LoanContract.borrower_id == user_id, LoanContract.status == LoanContractStatus.ACTIVE
        )
    )
    return sum(principal + interest for principal, interest in result.all())


async def _credit_ceiling_in_session(session: AsyncSession, user_id: int) -> int:
    """Returns how much more central-bank credit one borrower may still draw."""
    balance_result = await session.execute(
        statement=select(UserWallet.balance).where(UserWallet.user_id == user_id)
    )
    balance = balance_result.scalar_one_or_none() or 0
    total_debt = await _user_total_debt_in_session(session=session, user_id=user_id)
    return central_bank_credit_ceiling(balance=balance, total_debt=total_debt)


async def get_central_bank_status(
    guild_id: int, exclude_user_ids: tuple[int, ...] = ()
) -> CentralBankStatus:
    """Returns one guild's current central-bank lending capacity."""
    await _ensure_schema()
    async with open_session() as session:
        return await _central_bank_status_in_session(
            session=session, guild_id=guild_id, exclude_user_ids=exclude_user_ids
        )


async def get_credit_ceiling(user_id: int) -> int:
    """Returns how much more central-bank credit `user_id` may still draw."""
    await _ensure_schema()
    async with open_session() as session:
        return await _credit_ceiling_in_session(session=session, user_id=user_id)


async def record_guild_participant(guild_id: int, user_id: int) -> None:
    """Records that `user_id` takes part in `guild_id`'s economy.

    Only ever the caller of a command or the author of a rewarded message. Passing
    somebody else's id here hands their whole balance to a guild's lending pool
    without their knowledge.
    """
    await _ensure_schema()
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
    proposal_id: int, actor_id: int, approver_is_guild_admin: bool = False
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
            allowed = approver_is_guild_admin
        if not allowed:
            return None
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
    approver_is_guild_admin: bool = False,
    guild_id: int | None = None,
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
            approver_is_guild_admin=approver_is_guild_admin,
            guild_id=guild_id,
            central_bank_exclude_user_ids=central_bank_exclude_user_ids,
            allow_central_bank_self_approval=allow_central_bank_self_approval,
        )


async def _accept_loan_proposal_locked(  # noqa: C901, PLR0911, PLR0913 -- proposal-kind branches must stay in one transaction
    proposal_id: int,
    actor_id: int,
    actor_name: str,
    actor_avatar_url: str = "",
    approver_is_guild_admin: bool = False,
    guild_id: int | None = None,
    central_bank_exclude_user_ids: tuple[int, ...] = (),
    allow_central_bank_self_approval: bool = False,
) -> LoanProposalAcceptResult | None:
    """Accepts a loan proposal while the caller holds the acceptance lock."""
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
            if not approver_is_guild_admin or guild_id is None:
                return None
            if proposal.borrower_id == actor_id and not allow_central_bank_self_approval:
                return None
            # Both bounds are recomputed here rather than passed in, so the
            # BEGIN IMMEDIATE transaction this runs in is what serialises them and
            # two approvals racing cannot each see the capacity the other spends.
            central_status = await _central_bank_status_in_session(
                session=session, guild_id=guild_id, exclude_user_ids=central_bank_exclude_user_ids
            )
            borrower_ceiling = await _credit_ceiling_in_session(
                session=session, user_id=proposal.borrower_id
            )
            if min(central_status.available_credit, borrower_ceiling) < proposal.amount:
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
        if proposal.kind == LoanProposalKind.CENTRAL_BANK_REQUEST and guild_id is not None:
            central_status = await get_central_bank_status(
                guild_id=guild_id, exclude_user_ids=central_bank_exclude_user_ids
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


async def _pay_lender_side_in_session(
    session: AsyncSession, contract: LoanContract, paid: int, interest_paid: int, now: datetime
) -> int | None:
    """Passes one contract's payment on to whoever lent it.

    A personal lender is credited the whole payment. The central bank is credited only
    the interest: its principal was minted on approval and nets out by being burned here,
    while the interest is the bank's own earnings.

    Returns:
        A personal lender's balance after the credit, or None when there is no user on the
        other side. The caller keeps the last real balance rather than overwriting it, since
        one payment can cross several contracts.
    """
    if contract.lender_type == LoanLenderType.CENTRAL_BANK:
        await _credit_central_bank_ledger_in_session(
            session=session, amount=interest_paid, now=now
        )
        return None
    if contract.lender_id is None:
        return None
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
    # The borrower debit in the caller already cleared the leaderboard cache for this
    # transaction, so the lender credit needs no extra invalidation.
    return credit_result.scalar_one()


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
        invalidate_economy_leaderboard_cache()
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
        invalidate_economy_leaderboard_cache()
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
        invalidate_economy_leaderboard_cache()
        return result


async def call_central_bank_loans(
    guild_id: int,
    borrower_id: int,
    borrower_name: str,
    amount: int | None = None,
    borrower_avatar_url: str = "",
) -> LoanPaymentResult | None:
    """Forcibly collects active central-bank loans from a borrower.

    Refuses a borrower who does not take part in `guild_id`'s economy. Approval
    now rests on being an administrator of some guild, which anyone can arrange
    by creating one, so without this an administrator anywhere could sweep the
    balance of any borrower in the whole economy.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        if not await _is_guild_participant_in_session(
            session=session, guild_id=guild_id, user_id=borrower_id
        ):
            return None
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
