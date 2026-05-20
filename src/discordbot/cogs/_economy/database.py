"""Persistent point-balance store for the economy cog.

The engine is a module-level `AsyncEngine` singleton. Putting
`create_async_engine()` on a per-instance `cached_property` would leak the
connection pool, dialect cache, and inspector cache for every Discord
interaction (the same lesson `cogs/log_msg.py` captures for the sync engine
it still uses for pandas `to_sql`).

Every balance-mutating write path is an **atomic** SQL statement — either a
SQLite UPSERT (`INSERT ... ON CONFLICT DO UPDATE`) or a conditional
`UPDATE ... WHERE ... RETURNING`. The previous implementation read the row
in Python, mutated `account.balance`, and committed; two coroutines racing
on the same user would lose updates, and two coroutines racing on a
brand-new user would both `INSERT` and one would raise `IntegrityError`.
The UPSERT pattern fixes both. `borrow`, `repay` and the inner
`_credit_with_repayment_in_session` keep SELECT-then-conditional-UPDATE
retry loops because they need to inspect the current value to compute the
next state, but each retry is bounded and the WHERE clause guarantees no
double-spend.

PRAGMA setup at connect-time enables WAL (so reads don't block on writes),
sets a tolerant `busy_timeout`, and picks `synchronous=NORMAL` (the right
durability trade-off in WAL: every commit fsyncs the WAL frame, and the
main file is fsynced on checkpoint).

We use `aiosqlite` so every DB call stays on the event loop: no
`asyncio.to_thread` shim, no separate `_*_sync` helpers. Each operation
opens an `AsyncSession` bound to the current `_engine`, so tests can
monkeypatch `_engine` per-test and every subsequent call sees the swap.

Loan support is integrated into the same ``user_account`` row (no separate
``loan_account`` table) so chat reward and casino payout can pay down debt
inside the same single-row UPDATE. Loans expire every Asia/Taipei midnight:
the lazy ``_reset_expired_loan_in_session`` helper wipes ``loan_principal``
when ``loan_opened_at`` is older than today's local midnight. The audit log
lives in a separate ``point_transaction`` table that every mutating helper
writes into via ``_log_transaction_in_session``.

VIP and admin status are boolean columns on ``user_account``. VIP bumps daily
check-in rewards, the borrow cap, and the player's winning payout from games.
The flag is permanent once set. Admin status gates maintenance-only economy
commands and is managed out-of-band by scripts. Daily casino counters also live
on ``user_account`` so `/loss_leaderboard` can read current-day gross losses
without scanning the audit log.
"""

from typing import TYPE_CHECKING, Any, Final, cast
from datetime import UTC, datetime, timezone, timedelta
from collections.abc import Sequence

from sqlalchemy import Index, String, Boolean, Integer, DateTime, desc, text, event, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.sql.dml import ReturningInsert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.dialects.sqlite import insert

from discordbot.typings.economy import (
    VIP_PURCHASE_COST,
    CHECKIN_STREAK_CYCLE,
    BASE_CHECKIN_REWARD_AMOUNT,
    LoanView,
    RepayResult,
    AdminAccount,
    BorrowResult,
    CreditResult,
    CheckinResult,
    TransferResult,
    AccountSnapshot,
    JackpotSnapshot,
    TransactionKind,
    LeaderboardEntry,
    VipPurchaseResult,
    LossLeaderboardEntry,
    BalanceAdjustmentResult,
    JackpotSettlementResult,
    JackpotSettlementRequest,
    JackpotSettlementBatchResult,
)

if TYPE_CHECKING:
    from sqlalchemy.sql.elements import ColumnElement

# borrow / repay / _credit_with_repayment_in_session keep a small retry
# budget for SELECT-then-conditional-UPDATE loops. With WAL + busy_timeout,
# contention is rare and resolves on the first or second retry; the bound
# prevents a degenerate hot-row livelock.
_BORROW_MAX_RETRIES: Final[int] = 8
_CREDIT_WITH_REPAYMENT_MAX_RETRIES: Final[int] = 8
_REPAY_MAX_RETRIES: Final[int] = 8
_CHECKIN_MAX_RETRIES: Final[int] = 8
_VIP_PURCHASE_MAX_RETRIES: Final[int] = 8
_CLAMPED_DELTA_MAX_RETRIES: Final[int] = 8
_JACKPOT_CLAIM_MAX_RETRIES: Final[int] = 8
# Blackjack VIP perk: 1.5x payout on winning rounds, applied as floor(delta * 3 / 2).
_VIP_WIN_MULTIPLIER_NUM: Final[int] = 3
_VIP_WIN_MULTIPLIER_DEN: Final[int] = 2
TAIWAN_TIMEZONE: Final[timezone] = timezone(offset=timedelta(hours=8), name="Asia/Taipei")
_BorrowState = tuple[int, str, int, datetime | None]

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/economy.db")


def _database_now() -> datetime:
    """Returns the wall-clock timestamp used for persisted economy rows."""
    return datetime.now(tz=TAIWAN_TIMEZONE)


def _as_taipei(dt: datetime) -> datetime:
    """Returns ``dt`` re-interpreted in Asia/Taipei (treating naive as Taipei)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TAIWAN_TIMEZONE)
    return dt.astimezone(tz=TAIWAN_TIMEZONE)


def _taipei_midnight(now: datetime) -> datetime:
    """Returns the most recent Asia/Taipei 00:00 boundary at or before ``now``."""
    local = _as_taipei(dt=now)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
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
    cursor.close()


class Base(DeclarativeBase):
    """Base class for economy ORM models."""

    pass


class UserAccount(Base):
    """Persistent balance, loan, VIP, and check-in state for a Discord user.

    Loan and check-in columns live on the same row as the balance so single
    UPDATEs can settle multiple effects atomically. ``loan_opened_at`` is
    nullable because a user that has never borrowed has no opening date;
    ``last_checkin_at`` is nullable for users who have never checked in.

    Attributes:
        user_id: Discord user ID; primary key.
        name: Last-seen Discord username (refreshed on every write).
        avatar_url: Last-seen Discord avatar URL (refreshed on writes that carry it).
        balance: Current spendable point balance.
        total_earned: Lifetime points earned (chat rewards, game wins, transfers in).
        total_spent: Lifetime points removed (game losses, transfers out).
        updated_at: Taiwan-local timestamp of the last write.
        loan_principal: Currently outstanding loan principal (wiped daily at midnight).
        loan_total_borrowed: Lifetime gross borrowed amount.
        loan_total_repaid: Lifetime gross repaid amount.
        loan_opened_at: Timestamp the user borrowed for the current daily window;
            ``None`` while the user has never borrowed or after the nightly reset.
        is_vip: Permanent VIP flag toggled by a successful ``/vip`` purchase.
        last_checkin_at: Timestamp of the latest ``/checkin`` payout; ``None``
            for users who have never checked in.
        checkin_streak: Consecutive-day streak (1..``CHECKIN_STREAK_CYCLE``),
            persisted after the latest ``/checkin``. 0 means never checked in.
        is_admin: Whether the user can run Discord-side economy admin commands.
        casino_day_started_at: Asia/Taipei midnight for the stored daily casino counters.
        daily_casino_loss: Current-day gross loss from player-side casino settlements.
        daily_casino_win: Current-day gross win from player-side casino settlements.
        daily_casino_net: Current-day signed net casino result.
    """

    __tablename__ = "user_account"
    __table_args__ = (
        # /leaderboard does ORDER BY balance DESC LIMIT 10; the index turns a
        # full scan into a bounded walk. SQLite can use an ASC index to satisfy
        # ORDER BY DESC by reading it backwards, so no DESC index is needed.
        Index("ix_user_account_balance", "balance"),
        # /loss_leaderboard filters to one Taipei day and orders by gross loss.
        # SQLite can read the daily loss suffix backwards for DESC ordering.
        Index("ix_user_account_casino_day_loss", "casino_day_started_at", "daily_casino_loss"),
    )

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(length=128), default="")
    avatar_url: Mapped[str] = mapped_column(String(length=2048), default="", nullable=False)
    balance: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_earned: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_spent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )
    loan_principal: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    loan_total_borrowed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    loan_total_repaid: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    loan_opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_vip: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_checkin_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    checkin_streak: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    casino_day_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    daily_casino_loss: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    daily_casino_win: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    daily_casino_net: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class PointTransaction(Base):
    """Append-only audit log of every persistent balance change.

    One row per balance-mutating event. ``balance_after`` and ``debt_after``
    reflect the user's state *after* the row's write, so consecutive rows
    for the same user can be diffed to reconstruct every income / spend.
    ``occurred_at`` carries the Asia/Taipei wall clock for daily audit slices.

    Attributes:
        id: Autoincrementing primary key.
        user_id: Discord user ID this row belongs to.
        kind: ``TransactionKind`` enum value as string.
        delta: Signed change applied to balance by this transaction.
        balance_after: Balance after this transaction.
        debt_after: ``loan_principal`` after this transaction.
        note: Optional free-text annotation (e.g. counterparty for transfers).
        occurred_at: Taiwan-local timestamp of the event.
    """

    __tablename__ = "point_transaction"
    __table_args__ = (
        Index("ix_point_transaction_user_time", "user_id", "occurred_at"),
        # Audit/debug views often filter by (occurred_at, kind); the composite
        # index keeps daily-window scans cheap as the log grows.
        Index("ix_point_transaction_time_kind", "occurred_at", "kind"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(length=32), nullable=False)
    delta: Mapped[int] = mapped_column(Integer, nullable=False)
    balance_after: Mapped[int] = mapped_column(Integer, nullable=False)
    debt_after: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    note: Mapped[str | None] = mapped_column(String(length=256), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class JackpotPool(Base):
    """Per-game cumulative jackpot shared across every table of that game.

    One row per game (keyed by ``game_id``). Wager flows update
    ``pool_balance`` atomically while ``total_contributed`` /
    ``total_claimed`` accumulate gross in/out flows so the seeded
    on-the-house amount stays distinguishable from organic player
    contributions.

    Attributes:
        game_id: Stable game identifier (e.g. ``"dragon_gate"``); primary key.
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
    pool_balance: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_contributed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_claimed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    seeded_amount: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
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
# pointing it at a temp file) automatically forces another schema check;
# production never re-enters past the fast path. We intentionally do NOT
# use an asyncio.Lock here: module-level Locks bind to the first event loop
# they're awaited from and break under pytest's per-test loops. `create_all`
# is idempotent (CREATE TABLE IF NOT EXISTS), so a benign double-create on
# initial races is fine.
_schema_ready_for: AsyncEngine | None = None


async def _ensure_schema() -> None:  # noqa: C901, PLR0912 -- idempotent SQLite migrations are safest kept inline
    """Bootstraps the schema once per ``_engine`` value.

    Idempotent migrations:

    * Adds ``avatar_url`` to legacy DBs that predated the avatar cache.
    * Adds ``is_vip`` / ``last_checkin_at`` / ``checkin_streak`` /
      ``is_admin`` and daily casino counters so newer account flags keep
      working on older DBs.
    * Adds ``debt_after`` to legacy audit logs that predated loan context.
    * Drops legacy ``loan_interest`` / ``loan_last_accrual_at`` columns
      so the new model can INSERT fresh rows without violating their old
      ``NOT NULL`` constraints. Requires SQLite 3.35+ (`DROP COLUMN`),
      which is bundled with every Python ≥ 3.11.
    """
    global _schema_ready_for  # noqa: PLW0603 -- module-level cache by engine identity
    if _schema_ready_for is _engine:
        return
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        result = await conn.execute(statement=text(text="PRAGMA table_info(user_account)"))
        existing_columns = {row[1] for row in result.all()}
        if "avatar_url" not in existing_columns:
            await conn.execute(
                statement=text(
                    text="ALTER TABLE user_account ADD COLUMN avatar_url VARCHAR(2048) NOT NULL DEFAULT ''"
                )
            )
        if "is_vip" not in existing_columns:
            await conn.execute(
                statement=text(
                    text="ALTER TABLE user_account ADD COLUMN is_vip BOOLEAN NOT NULL DEFAULT 0"
                )
            )
        if "last_checkin_at" not in existing_columns:
            await conn.execute(
                statement=text(text="ALTER TABLE user_account ADD COLUMN last_checkin_at DATETIME")
            )
        if "checkin_streak" not in existing_columns:
            await conn.execute(
                statement=text(
                    text="ALTER TABLE user_account ADD COLUMN checkin_streak INTEGER NOT NULL DEFAULT 0"
                )
            )
        if "is_admin" not in existing_columns:
            await conn.execute(
                statement=text(
                    text="ALTER TABLE user_account ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0"
                )
            )
        if "casino_day_started_at" not in existing_columns:
            await conn.execute(
                statement=text(
                    text="ALTER TABLE user_account ADD COLUMN casino_day_started_at DATETIME"
                )
            )
        if "daily_casino_loss" not in existing_columns:
            await conn.execute(
                statement=text(
                    text="ALTER TABLE user_account ADD COLUMN daily_casino_loss INTEGER NOT NULL DEFAULT 0"
                )
            )
        if "daily_casino_win" not in existing_columns:
            await conn.execute(
                statement=text(
                    text="ALTER TABLE user_account ADD COLUMN daily_casino_win INTEGER NOT NULL DEFAULT 0"
                )
            )
        if "daily_casino_net" not in existing_columns:
            await conn.execute(
                statement=text(
                    text="ALTER TABLE user_account ADD COLUMN daily_casino_net INTEGER NOT NULL DEFAULT 0"
                )
            )
        if "loan_interest" in existing_columns:
            await conn.execute(
                statement=text(text="ALTER TABLE user_account DROP COLUMN loan_interest")
            )
        if "loan_last_accrual_at" in existing_columns:
            await conn.execute(
                statement=text(text="ALTER TABLE user_account DROP COLUMN loan_last_accrual_at")
            )
        await conn.execute(
            statement=text(
                text=(
                    "CREATE INDEX IF NOT EXISTS ix_user_account_casino_day_loss "
                    "ON user_account (casino_day_started_at, daily_casino_loss)"
                )
            )
        )
        result = await conn.execute(statement=text(text="PRAGMA table_info(point_transaction)"))
        transaction_columns = {row[1] for row in result.all()}
        if "debt_after" not in transaction_columns:
            await conn.execute(
                statement=text(
                    text="ALTER TABLE point_transaction ADD COLUMN debt_after INTEGER NOT NULL DEFAULT 0"
                )
            )
        result = await conn.execute(statement=text(text="PRAGMA table_info(jackpot_pool)"))
        jackpot_columns = {row[1] for row in result.all()}
        if "generation" not in jackpot_columns:
            await conn.execute(
                statement=text(
                    text="ALTER TABLE jackpot_pool ADD COLUMN generation INTEGER NOT NULL DEFAULT 0"
                )
            )
        for seed_game_id, seed_amount in _JACKPOT_SEEDS:
            await conn.execute(
                statement=insert(JackpotPool)
                .values(
                    game_id=seed_game_id,
                    pool_balance=seed_amount,
                    total_contributed=0,
                    total_claimed=0,
                    seeded_amount=seed_amount,
                    generation=0,
                    updated_at=_database_now(),
                )
                .on_conflict_do_nothing(index_elements=["game_id"])
            )
    _schema_ready_for = _engine


def open_session() -> AsyncSession:
    """Creates an async session bound to the current economy database engine.

    Returns:
        An `AsyncSession` using the current module-level `_engine`.
    """
    return AsyncSession(bind=_engine, expire_on_commit=False)


def credit_limit(user: Any, is_vip: bool = False) -> int:  # noqa: ANN401 -- accepts nextcord.User | nextcord.Member; both expose `created_at`
    """Returns the borrowing cap for a Discord account based on its age.

    Computed entirely from ``user.created_at`` (which Discord reconstructs
    from the snowflake ID), so the same cap applies in DMs, guilds, and
    across servers, and a freshly-created account cannot farm by re-joining
    different guilds. Older Discord accounts borrow more because they
    represent a more stable identity. VIP accounts double the cap.

    Args:
        user: A ``nextcord.User`` or ``nextcord.Member`` whose ``created_at``
            timestamp is inspected.
        is_vip: Whether the user owns the VIP perk; doubles the cap.

    Returns:
        Maximum total principal the account is allowed to carry within a
        single Taipei calendar day.
    """
    age_days = (datetime.now(tz=UTC) - user.created_at).days
    if age_days < 30:
        base = 1_000
    elif age_days < 180:
        base = 10_000
    elif age_days < 365:
        base = 50_000
    elif age_days < 365 * 3:
        base = 200_000
    else:
        base = 500_000
    return base * 2 if is_vip else base


def checkin_reward(streak: int, is_vip: bool) -> int:
    """Returns the gross check-in payout for a streak day.

    The reward formula is ``BASE * (1 + (streak - 1) * 0.5)`` where ``streak``
    is the 1..``CHECKIN_STREAK_CYCLE`` day in the cycle. VIP doubles the base
    before the streak bonus.

    Args:
        streak: Streak counter for this check-in (1..``CHECKIN_STREAK_CYCLE``).
        is_vip: VIP status of the account at check-in time.

    Returns:
        Integer reward amount.
    """
    base = BASE_CHECKIN_REWARD_AMOUNT * (2 if is_vip else 1)
    multiplier = 1.0 + (streak - 1) * 0.5
    return int(base * multiplier)


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


def _build_credit_upsert(
    user_id: int, name: str, amount: int, now: datetime, avatar_url: str = ""
) -> ReturningInsert[tuple[int]]:
    """UPSERT that credits ``amount`` points (caller guarantees ``amount > 0``).

    On INSERT, the new row starts at ``balance = total_earned = amount``,
    ``total_spent = 0``. On UPDATE, ``balance`` and ``total_earned`` are
    each incremented by ``amount``. ``name`` is only refreshed when the
    caller actually supplied one, mirroring the previous Python-side
    "only update if non-empty and different" rule.

    Returns:
        A SQLAlchemy `Insert` with `on_conflict_do_update` and `returning(balance)`.
    """
    insert_name = name or str(user_id)
    stmt = insert(UserAccount).values(
        user_id=user_id,
        name=insert_name,
        avatar_url=avatar_url,
        balance=amount,
        total_earned=amount,
        total_spent=0,
        updated_at=now,
    )
    set_: dict[str, Any] = {
        "balance": UserAccount.balance + amount,
        "total_earned": UserAccount.total_earned + amount,
        "updated_at": now,
    }
    if name:
        set_["name"] = insert_name
    if avatar_url:
        set_["avatar_url"] = avatar_url
    return stmt.on_conflict_do_update(index_elements=["user_id"], set_=set_).returning(
        UserAccount.balance
    )


def _build_signed_delta_upsert(
    user_id: int, name: str, delta: int, now: datetime, avatar_url: str = ""
) -> ReturningInsert[tuple[int]]:
    """UPSERT applying a signed ``delta`` with NO clamp on the resulting balance.

    Used for the dealer's house-ledger row, which is allowed to go negative
    when the casino has paid out more than it took in. `total_earned` /
    `total_spent` still accumulate gross flows so `/house` can show the
    direction of the volume, not just the net.

    Returns:
        A SQLAlchemy `Insert` with `on_conflict_do_update` and `returning(balance)`.
    """
    insert_name = name or str(user_id)
    initial_earned = max(delta, 0)
    initial_spent = max(-delta, 0)
    stmt = insert(UserAccount).values(
        user_id=user_id,
        name=insert_name,
        avatar_url=avatar_url,
        balance=delta,
        total_earned=initial_earned,
        total_spent=initial_spent,
        updated_at=now,
    )
    set_: dict[str, Any] = {
        "balance": UserAccount.balance + delta,
        "total_earned": UserAccount.total_earned + initial_earned,
        "total_spent": UserAccount.total_spent + initial_spent,
        "updated_at": now,
    }
    if name:
        set_["name"] = insert_name
    if avatar_url:
        set_["avatar_url"] = avatar_url
    return stmt.on_conflict_do_update(index_elements=["user_id"], set_=set_).returning(
        UserAccount.balance
    )


async def _log_transaction_in_session(  # noqa: PLR0913 -- audit row is wide on purpose
    session: AsyncSession,
    user_id: int,
    kind: TransactionKind,
    delta: int,
    balance_after: int,
    note: str | None,
    now: datetime,
    debt_after: int | None = None,
) -> None:
    """Appends one row to ``point_transaction`` from inside the caller's session.

    Skips the write when ``delta == 0`` so push-style settlements (where the
    house ledger doesn't actually move) don't clutter the audit log. When
    ``debt_after`` is not supplied, a single SELECT reads the user's current
    principal to keep the row self-contained; callers that have just computed
    the new debt locally should pass it in to skip the read.
    """
    if delta == 0:
        return
    if debt_after is None:
        debt_result = await session.execute(
            statement=select(UserAccount.loan_principal).where(UserAccount.user_id == user_id)
        )
        debt_after = debt_result.scalar_one_or_none() or 0
    await session.execute(
        statement=insert(PointTransaction).values(
            user_id=user_id,
            kind=kind.value,
            delta=delta,
            balance_after=balance_after,
            debt_after=debt_after,
            note=note,
            occurred_at=now,
        )
    )


async def _apply_daily_casino_delta_in_session(
    session: AsyncSession, user_id: int, delta: int, now: datetime
) -> None:
    """Accumulates current-day gross casino counters on the player account."""
    if delta == 0:
        return
    today_midnight = _taipei_midnight(now=now)
    stale_day = (UserAccount.casino_day_started_at.is_(None)) | (
        UserAccount.casino_day_started_at != today_midnight
    )
    await session.execute(
        statement=update(UserAccount)
        .where(UserAccount.user_id == user_id, stale_day)
        .values(
            casino_day_started_at=today_midnight,
            daily_casino_loss=0,
            daily_casino_win=0,
            daily_casino_net=0,
            updated_at=now,
        )
    )

    loss_delta = max(-delta, 0)
    win_delta = max(delta, 0)
    await session.execute(
        statement=update(UserAccount)
        .where(UserAccount.user_id == user_id)
        .values(
            casino_day_started_at=today_midnight,
            daily_casino_loss=UserAccount.daily_casino_loss + loss_delta,
            daily_casino_win=UserAccount.daily_casino_win + win_delta,
            daily_casino_net=UserAccount.daily_casino_net + delta,
            updated_at=now,
        )
    )


async def _reset_expired_loan_in_session(
    session: AsyncSession, user_id: int, now: datetime
) -> None:
    """Wipes ``loan_principal`` when the loan was opened before today's midnight.

    Loans live for the rest of the calendar day in Asia/Taipei. The next
    write or read after midnight zeroes the principal and clears
    ``loan_opened_at`` so subsequent ``/borrow`` calls re-arm the daily
    window. The reset is unconditional (a forgiveness, not a clawback) so
    we do not touch ``balance``: users keep whatever they did with the
    borrowed funds. We deliberately do NOT emit an audit row for the
    reset: it is a deterministic system event and clutters the per-user
    history without adding information.
    """
    today_midnight = _taipei_midnight(now=now)
    read_result = await session.execute(
        statement=select(UserAccount.loan_principal, UserAccount.loan_opened_at).where(
            UserAccount.user_id == user_id
        )
    )
    row = read_result.one_or_none()
    if row is None:
        return
    principal, opened_at = row[0], row[1]
    if principal <= 0 or opened_at is None:
        return
    opened_at = _as_taipei(dt=opened_at)
    if opened_at >= today_midnight:
        return
    await session.execute(
        statement=update(UserAccount)
        .where(UserAccount.user_id == user_id)
        .values(loan_principal=0, loan_opened_at=None, updated_at=now)
    )


async def _credit_with_repayment_in_session(  # noqa: PLR0913 -- single-row income pipeline kept linear for readability
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    amount: int,
    kind: TransactionKind,
    note: str | None,
    now: datetime,
) -> CreditResult:
    """Inner credit-with-auto-repay pipeline; caller must commit the session.

    Pipeline (all inside the caller's transaction):

    1. ``_reset_expired_loan_in_session`` clears the previous day's loan.
    2. SELECT current balance + loan state.
    3. ``to_repay = min(amount * auto_repay_ratio_percent // 100, principal)``.
       ``auto_repay_ratio_percent`` is currently ``0`` (auto-repay disabled);
       bump it back to ``50`` to restore the old "half of positive income goes
       to principal" behavior without restructuring callers.
    4. Repayment debits ``loan_principal``; remainder credits balance.
    5. Conditional UPDATE gated on the values we read; retry on conflict.
    6. ``_log_transaction_in_session`` writes one audit row with
       ``delta = credited_amount`` and the post-state debt.

    Caller must guarantee ``amount > 0``.
    """
    auto_repay_ratio_percent = 0  # disabled; was 50 — pipeline preserved for easy re-enable
    await _reset_expired_loan_in_session(session=session, user_id=user_id, now=now)

    for _ in range(_CREDIT_WITH_REPAYMENT_MAX_RETRIES):
        read_result = await session.execute(
            statement=select(
                UserAccount.balance,
                UserAccount.name,
                UserAccount.loan_principal,
                UserAccount.total_earned,
                UserAccount.loan_total_repaid,
            ).where(UserAccount.user_id == user_id)
        )
        row = read_result.one_or_none()

        if row is None:
            # No existing row → straight credit, no debt to pay down. UPSERT
            # protects against a parallel INSERT that may have just landed.
            insert_name = name or str(user_id)
            base_stmt = insert(UserAccount).values(
                user_id=user_id,
                name=insert_name,
                avatar_url=avatar_url,
                balance=amount,
                total_earned=amount,
                total_spent=0,
                loan_principal=0,
                loan_total_borrowed=0,
                loan_total_repaid=0,
                loan_opened_at=None,
                updated_at=now,
            )
            set_values: dict[str, Any] = {
                "balance": UserAccount.balance + amount,
                "total_earned": UserAccount.total_earned + amount,
                "updated_at": now,
            }
            if name:
                set_values["name"] = insert_name
            if avatar_url:
                set_values["avatar_url"] = avatar_url
            upsert_stmt = base_stmt.on_conflict_do_update(
                index_elements=["user_id"], set_=set_values
            ).returning(UserAccount.balance, UserAccount.loan_principal)
            insert_result = await session.execute(statement=upsert_stmt)
            balance_after, principal_after = insert_result.one()
            await _log_transaction_in_session(
                session=session,
                user_id=user_id,
                kind=kind,
                delta=amount,
                balance_after=balance_after,
                debt_after=principal_after,
                note=note,
                now=now,
            )
            return CreditResult(
                new_balance=balance_after,
                credited_amount=amount,
                principal_repaid=0,
                remaining_debt=principal_after,
            )

        (starting_balance, existing_name, principal, total_earned, total_repaid) = row
        to_repay = min(amount * auto_repay_ratio_percent // 100, principal)
        credited = amount - to_repay

        new_balance = starting_balance + credited
        new_principal = principal - to_repay
        new_total_earned = total_earned + credited
        new_total_repaid = total_repaid + to_repay

        update_values: dict[str, Any] = {
            "balance": new_balance,
            "total_earned": new_total_earned,
            "loan_principal": new_principal,
            "loan_total_repaid": new_total_repaid,
            "updated_at": now,
        }
        if name and name != existing_name:
            update_values["name"] = name
        if avatar_url:
            update_values["avatar_url"] = avatar_url

        stmt = (
            update(UserAccount)
            .where(
                UserAccount.user_id == user_id,
                UserAccount.balance == starting_balance,
                UserAccount.loan_principal == principal,
            )
            .values(**update_values)
            .returning(UserAccount.balance)
        )
        update_result = await session.execute(statement=stmt)
        if update_result.one_or_none() is None:
            # Conflicting write between our SELECT and UPDATE; nothing has
            # been committed in this helper yet, so re-read and try again.
            continue

        await _log_transaction_in_session(
            session=session,
            user_id=user_id,
            kind=kind,
            delta=credited,
            balance_after=new_balance,
            debt_after=new_principal,
            note=note,
            now=now,
        )
        return CreditResult(
            new_balance=new_balance,
            credited_amount=credited,
            principal_repaid=to_repay,
            remaining_debt=new_principal,
        )

    raise RuntimeError(f"credit_with_repayment retry budget exhausted for user_id={user_id}")


async def _apply_clamped_delta_in_session(  # noqa: PLR0913 -- session helper needs ledger identity + kind
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    delta: int,
    kind: TransactionKind,
    now: datetime,
    note: str | None = None,
) -> tuple[int, int]:
    """Applies a clamped signed delta and logs any resulting audit row.

    The observed balance is pinned in the UPDATE predicate, so concurrent
    clamped debits cannot both compute their applied delta from the same stale
    balance. A negative delta against a missing row is a no-op so manual clamp
    operations do not create zero-balance accounts.
    """
    if delta == 0:
        read_result = await session.execute(
            statement=select(UserAccount.balance).where(UserAccount.user_id == user_id)
        )
        return read_result.scalar_one_or_none() or 0, 0

    insert_name = name or str(user_id)
    for _ in range(_CLAMPED_DELTA_MAX_RETRIES):
        read_result = await session.execute(
            statement=select(UserAccount.balance, UserAccount.name).where(
                UserAccount.user_id == user_id
            )
        )
        row = read_result.one_or_none()

        if row is None:
            if delta < 0:
                return 0, 0
            insert_result = await _try_insert_clamped_positive_delta_in_session(
                session=session,
                user_id=user_id,
                insert_name=insert_name,
                avatar_url=avatar_url,
                delta=delta,
                kind=kind,
                now=now,
                note=note,
            )
            if insert_result is not None:
                return insert_result
            continue

        current_balance, existing_name = row
        update_result = await _try_update_clamped_delta_in_session(
            session=session,
            user_id=user_id,
            name=name,
            avatar_url=avatar_url,
            current_balance=current_balance,
            existing_name=existing_name,
            kind=kind,
            delta=delta,
            now=now,
            note=note,
        )
        if update_result is not None:
            return update_result

    raise RuntimeError(f"apply_clamped_delta retry budget exhausted for user_id={user_id}")


async def _try_insert_clamped_positive_delta_in_session(  # noqa: PLR0913 -- mirrors the caller's audit identity
    session: AsyncSession,
    user_id: int,
    insert_name: str,
    avatar_url: str,
    delta: int,
    kind: TransactionKind,
    now: datetime,
    note: str | None = None,
) -> tuple[int, int] | None:
    """Attempts to create a missing account for a positive clamped delta."""
    insert_stmt = (
        insert(UserAccount)
        .values(
            user_id=user_id,
            name=insert_name,
            avatar_url=avatar_url,
            balance=delta,
            total_earned=delta,
            total_spent=0,
            updated_at=now,
        )
        .on_conflict_do_nothing(index_elements=["user_id"])
        .returning(UserAccount.balance)
    )
    insert_result = await session.execute(statement=insert_stmt)
    inserted_balance = insert_result.scalar_one_or_none()
    if inserted_balance is None:
        return None
    await _log_transaction_in_session(
        session=session,
        user_id=user_id,
        kind=kind,
        delta=delta,
        balance_after=inserted_balance,
        note=note,
        now=now,
    )
    return inserted_balance, delta


async def _try_update_clamped_delta_in_session(  # noqa: PLR0913 -- conditional write needs observed row state
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    current_balance: int,
    existing_name: str,
    delta: int,
    kind: TransactionKind,
    now: datetime,
    note: str | None = None,
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
    if applied > 0:
        update_values["total_earned"] = UserAccount.total_earned + applied
    elif applied < 0:
        update_values["total_spent"] = UserAccount.total_spent - applied
    if name and name != existing_name:
        update_values["name"] = name
    if avatar_url:
        update_values["avatar_url"] = avatar_url

    update_result = await session.execute(
        statement=update(UserAccount)
        .where(UserAccount.user_id == user_id, UserAccount.balance == current_balance)
        .values(**update_values)
        .returning(UserAccount.balance)
    )
    if update_result.scalar_one_or_none() is None:
        return None
    await _log_transaction_in_session(
        session=session,
        user_id=user_id,
        kind=kind,
        delta=applied,
        balance_after=new_balance,
        note=note,
        now=now,
    )
    return new_balance, applied


async def _apply_signed_delta_in_session(  # noqa: PLR0913 -- session helper needs ledger identity + delta + kind
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    delta: int,
    kind: TransactionKind,
    now: datetime,
    note: str | None = None,
) -> int:
    """Applies a signed delta without clamping and logs the audit row.

    Used for dealer-side mirrors (``HOUSE_SETTLE``), which may run cumulative
    negative P&L. Player-side losses use the clamped path instead.
    """
    stmt = _build_signed_delta_upsert(
        user_id=user_id, name=name, avatar_url=avatar_url, delta=delta, now=now
    )
    result = await session.execute(statement=stmt)
    new_balance = result.scalar_one()
    await _log_transaction_in_session(
        session=session,
        user_id=user_id,
        kind=kind,
        delta=delta,
        balance_after=new_balance,
        note=note,
        now=now,
    )
    return new_balance


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
            kind=TransactionKind.CASINO_PAYOUT,
            note=None,
            now=now,
        )
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=user_id, delta=delta, now=now
        )
        return credit_result.new_balance, delta
    if delta < 0:
        new_balance, applied_delta = await _apply_clamped_delta_in_session(
            session=session,
            user_id=user_id,
            name=name,
            avatar_url=avatar_url,
            delta=delta,
            kind=TransactionKind.CASINO_BET,
            now=now,
        )
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=user_id, delta=applied_delta, now=now
        )
        return new_balance, applied_delta
    read_result = await session.execute(
        statement=select(UserAccount.balance).where(UserAccount.user_id == user_id)
    )
    return read_result.scalar_one_or_none() or 0, 0


async def _apply_jackpot_player_delta_in_session(  # noqa: PLR0913 -- jackpot settlement needs identity and audit metadata
    session: AsyncSession, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
) -> tuple[int, int]:
    """Applies a jackpot player delta and returns the balance plus applied delta.

    Positive deltas keep the existing casino payout path, including the
    (currently disabled) loan auto-repayment slice, and count as fully
    applied — debt repayment is still player value when the ratio ever
    flips back on. Negative deltas clamp at zero so Dragon Gate losses
    cannot drive the player account negative; the returned delta is the
    actual debit.
    """
    if delta > 0:
        credit_result = await _credit_with_repayment_in_session(
            session=session,
            user_id=user_id,
            name=name,
            avatar_url=avatar_url,
            amount=delta,
            kind=TransactionKind.CASINO_PAYOUT,
            note=None,
            now=now,
        )
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=user_id, delta=delta, now=now
        )
        return credit_result.new_balance, delta
    if delta < 0:
        new_balance, applied_delta = await _apply_clamped_delta_in_session(
            session=session,
            user_id=user_id,
            name=name,
            avatar_url=avatar_url,
            delta=delta,
            kind=TransactionKind.CASINO_BET,
            now=now,
        )
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=user_id, delta=applied_delta, now=now
        )
        return new_balance, applied_delta
    read_result = await session.execute(
        statement=select(UserAccount.balance).where(UserAccount.user_id == user_id)
    )
    return read_result.scalar_one_or_none() or 0, 0


async def credit_with_repayment(  # noqa: PLR0913 -- public DB facade mirrors one income event
    user_id: int,
    name: str,
    amount: int,
    kind: TransactionKind,
    note: str | None = None,
    avatar_url: str = "",
) -> CreditResult:
    """Credits ``amount`` to the user, optionally diverting a slice to debt.

    The auto-repay ratio is currently 0 (see
    ``_credit_with_repayment_in_session``) so every positive ``amount``
    lands in balance and the loan is untouched. The repayment pipeline
    itself is kept so flipping the ratio back to 50 re-enables the old
    half-of-income behavior without callers changing. Any portion not used
    for repayment goes to balance and bumps ``total_earned``. Writes one
    audit row via the helper.

    Args:
        user_id: Discord user ID receiving the credit.
        name: Last-seen Discord username to store on the account.
        amount: Gross income amount; must be positive for the repayment
            path to run.
        kind: Audit-log category for this credit (e.g. ``CHAT_REWARD``,
            ``CASINO_PAYOUT``).
        note: Optional free-text annotation for the audit row.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        Outcome capturing post-credit balance, the credited slice, and the
        repayment amount.
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
            kind=kind,
            note=note,
            now=now,
        )
        await session.commit()
        return result


async def adjust_balance(  # noqa: PLR0913 -- admin adjustment needs identity, delta, clamp policy, and audit metadata
    user_id: int,
    name: str,
    delta: int,
    allow_negative: bool = False,
    avatar_url: str = "",
    note: str | None = None,
) -> BalanceAdjustmentResult:
    """Applies an explicit manual balance adjustment.

    This is the public maintenance API for scripts and admin tooling. It does
    not trigger loan auto-repayment and it logs ``MANUAL_ADJUSTMENT`` rather
    than pretending to be casino activity, so leaderboards and house P&L remain
    clean.

    Args:
        user_id: Discord user ID whose balance should be adjusted.
        name: Last-seen Discord username to store on the account.
        delta: Signed amount to apply.
        allow_negative: Whether the resulting balance may go below zero.
        avatar_url: Last-seen Discord avatar URL to store when available.
        note: Optional audit-log annotation.

    Returns:
        The post-adjustment balance and the applied delta after any clamp.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        if delta == 0:
            result = await session.execute(
                statement=select(UserAccount.balance).where(UserAccount.user_id == user_id)
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
                kind=TransactionKind.MANUAL_ADJUSTMENT,
                now=now,
                note=note,
            )
            applied_delta = delta
        else:
            new_balance, applied_delta = await _apply_clamped_delta_in_session(
                session=session,
                user_id=user_id,
                name=name,
                avatar_url=avatar_url,
                delta=delta,
                kind=TransactionKind.MANUAL_ADJUSTMENT,
                now=now,
                note=note,
            )
        await session.commit()
        return BalanceAdjustmentResult(new_balance=new_balance, applied_delta=applied_delta)


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
    ``_credit_with_repayment_in_session`` so the auto-repay rule (currently
    disabled at 0%) applies to casino profit. Negative player deltas clamp at
    zero; when a loss cannot be fully collected, the dealer ledger only records
    the actual collected debit.

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
                statement=select(UserAccount.balance).where(UserAccount.user_id == dealer_id)
            )
            dealer_balance = dealer_result.scalar_one_or_none() or 0
        else:
            dealer_balance = await _apply_signed_delta_in_session(
                session=session,
                user_id=dealer_id,
                name=dealer_name,
                avatar_url=dealer_avatar_url,
                delta=dealer_delta_to_apply,
                kind=TransactionKind.HOUSE_SETTLE,
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
    """Returns the current ``pool_balance`` for a game's shared jackpot.

    Reading the seeded row is the canonical way to surface the current
    pool to a view (lobby start, every active-table refresh). Seeded pools
    are replenished before returning if an older process left them drained.
    Returns ``0`` when the row hasn't been seeded yet so a freshly-introduced
    game can short-circuit cleanly.

    Args:
        game_id: Game identifier (e.g. ``"dragon_gate"``).

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

    Positive deltas accumulate ``total_contributed`` (player losses /
    antes flowing into the pool); negative deltas accumulate
    ``total_claimed`` with the absolute value (winning payouts flowing
    out). Seeded pools are topped back up automatically after a drain, so
    the returned balance is always ready for the next table.

    Args:
        session: Active SQLAlchemy session bound to ``_engine``.
        game_id: Game identifier (jackpot row primary key).
        delta: Signed point adjustment to apply to ``pool_balance``.
        now: ``_database_now()`` value pinned for this transaction.

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
    """Atomically claims up to ``amount`` from the requested jackpot generation."""
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

    This is a convenience wrapper around ``apply_jackpot_settlement_batch``.

    Args:
        player_id: Discord user ID for the player.
        player_account_name: Account name to store on the player row.
        player_delta: Signed net change for the player. Losses are written
            as a negative delta and the absolute value flows into the pool.
        game_id: Jackpot game identifier (e.g. ``"dragon_gate"``).
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
        statement=select(UserAccount.user_id, UserAccount.balance).where(
            UserAccount.user_id.in_(other=tuple(required_debits))
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
    """Atomically applies one or more player settlements against a jackpot pool.

    Positive player deltas (wins) are capped to the live pool balance inside
    this transaction, then credited via the auto-repayment path (the
    diversion ratio is currently 0%, so wins land fully in the player
    balance). Negative deltas normally clamp at zero and feed the pool with
    the actual debit.
    Required-full-debit settlements reject the whole batch instead. If a seeded
    pool is drained, the same transaction restores its on-the-house seed.

    Args:
        game_id: Jackpot game identifier (e.g. ``"dragon_gate"``).
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

            player_balance, applied_player_delta = await _apply_jackpot_player_delta_in_session(
                session=session,
                user_id=settlement.player_id,
                name=settlement.player_account_name,
                avatar_url=settlement.player_avatar_url,
                delta=effective_player_delta,
                now=now,
            )
            if settlement.require_full_debit and applied_player_delta != effective_player_delta:
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
        return JackpotSettlementBatchResult(
            player_balances=player_balances,
            applied_player_deltas=applied_player_deltas,
            jackpot_balance=jackpot_snapshot.balance,
            jackpot_generation=jackpot_snapshot.generation,
            jackpot_depleted=jackpot_depleted,
        )


async def _read_borrow_state_in_session(
    session: AsyncSession, user_id: int
) -> _BorrowState | None:
    """Reads the balance and loan fields needed by the borrow retry loop."""
    result = await session.execute(
        statement=select(
            UserAccount.balance,
            UserAccount.name,
            UserAccount.loan_principal,
            UserAccount.loan_opened_at,
        ).where(UserAccount.user_id == user_id)
    )
    row = result.one_or_none()
    if row is None:
        return None
    return row[0], row[1], row[2], row[3]


async def _try_insert_borrow_in_session(  # noqa: PLR0913 -- borrow insert needs account identity and cap
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    amount: int,
    credit_limit_value: int,
    now: datetime,
) -> tuple[BorrowResult | None, bool]:
    """Attempts first-borrow INSERT; returns ``(result, retry_needed)``."""
    effective_amount = min(amount, credit_limit_value)
    if effective_amount <= 0:
        return None, False
    insert_name = name or str(user_id)
    insert_stmt = (
        insert(UserAccount)
        .values(
            user_id=user_id,
            name=insert_name,
            avatar_url=avatar_url,
            balance=effective_amount,
            total_earned=0,
            total_spent=0,
            loan_principal=effective_amount,
            loan_total_borrowed=effective_amount,
            loan_total_repaid=0,
            loan_opened_at=now,
            updated_at=now,
        )
        .on_conflict_do_nothing(index_elements=["user_id"])
        .returning(UserAccount.balance, UserAccount.loan_principal)
    )
    insert_result = await session.execute(statement=insert_stmt)
    inserted_row = insert_result.one_or_none()
    if inserted_row is None:
        return None, True
    balance_after, principal_after = inserted_row
    await _log_transaction_in_session(
        session=session,
        user_id=user_id,
        kind=TransactionKind.BORROW,
        delta=effective_amount,
        balance_after=balance_after,
        debt_after=principal_after,
        note=None,
        now=now,
    )
    return (
        BorrowResult(
            new_balance=balance_after, principal=principal_after, borrowed_amount=effective_amount
        ),
        False,
    )


async def _try_update_borrow_in_session(  # noqa: PLR0913 -- borrow update needs observed state for CAS
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    amount: int,
    credit_limit_value: int,
    now: datetime,
    state: _BorrowState,
) -> tuple[BorrowResult | None, bool]:
    """Attempts conditional borrow UPDATE; returns ``(result, retry_needed)``."""
    current_balance, existing_name, current_principal, loan_opened_at = state
    effective_amount = min(amount, credit_limit_value - current_principal)
    if effective_amount <= 0:
        return None, False

    new_balance = current_balance + effective_amount
    new_principal = current_principal + effective_amount
    update_values: dict[str, Any] = {
        "balance": new_balance,
        "loan_principal": new_principal,
        "loan_total_borrowed": UserAccount.loan_total_borrowed + effective_amount,
        "loan_opened_at": loan_opened_at or now,
        "updated_at": now,
    }
    if name and name != existing_name:
        update_values["name"] = name
    if avatar_url:
        update_values["avatar_url"] = avatar_url

    loan_opened_gate: ColumnElement[bool]
    if loan_opened_at is None:
        loan_opened_gate = UserAccount.loan_opened_at.is_(None)
    else:
        loan_opened_gate = UserAccount.loan_opened_at == loan_opened_at

    update_stmt = (
        update(UserAccount)
        .where(
            UserAccount.user_id == user_id,
            UserAccount.balance == current_balance,
            UserAccount.loan_principal == current_principal,
            loan_opened_gate,
        )
        .values(**update_values)
        .returning(UserAccount.balance, UserAccount.loan_principal)
    )
    update_result = await session.execute(statement=update_stmt)
    updated_row = update_result.one_or_none()
    if updated_row is None:
        return None, True

    balance_after, principal_after = updated_row
    await _log_transaction_in_session(
        session=session,
        user_id=user_id,
        kind=TransactionKind.BORROW,
        delta=effective_amount,
        balance_after=balance_after,
        debt_after=principal_after,
        note=None,
        now=now,
    )
    return (
        BorrowResult(
            new_balance=balance_after, principal=principal_after, borrowed_amount=effective_amount
        ),
        False,
    )


async def borrow(
    user_id: int, name: str, amount: int, credit_limit_value: int, avatar_url: str = ""
) -> BorrowResult | None:
    """Disburses up to ``amount`` points to the user as new principal.

    Rejected (``None`` returned) when ``amount`` is non-positive or when the
    user has no remaining credit. Requests above the remaining daily credit
    are clamped to that remaining amount. Loans expire at the next Asia/Taipei
    midnight, so the daily cap matches the daily reset window. Borrowed funds
    do **not** bump ``total_earned`` — debt isn't earnings.

    Args:
        user_id: Discord user ID for the borrower.
        name: Last-seen Discord username.
        amount: Requested amount to borrow (must be positive).
        credit_limit_value: Maximum allowed post-borrow principal; the
            caller is expected to compute this with ``credit_limit``.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        ``BorrowResult`` capturing the new balance and loan state, or
        ``None`` when the request was rejected.
    """
    await _ensure_schema()
    if amount <= 0:
        return None
    now = _database_now()

    async with open_session() as session:
        for _ in range(_BORROW_MAX_RETRIES):
            await _reset_expired_loan_in_session(session=session, user_id=user_id, now=now)
            state = await _read_borrow_state_in_session(session=session, user_id=user_id)
            if state is None:
                result, retry_needed = await _try_insert_borrow_in_session(
                    session=session,
                    user_id=user_id,
                    name=name,
                    avatar_url=avatar_url,
                    amount=amount,
                    credit_limit_value=credit_limit_value,
                    now=now,
                )
            else:
                result, retry_needed = await _try_update_borrow_in_session(
                    session=session,
                    user_id=user_id,
                    name=name,
                    avatar_url=avatar_url,
                    amount=amount,
                    credit_limit_value=credit_limit_value,
                    now=now,
                    state=state,
                )
            if retry_needed:
                await session.rollback()
                continue
            if result is None:
                return None
            await session.commit()
            return result

        return None


async def repay(user_id: int, name: str, amount: int, avatar_url: str = "") -> RepayResult | None:
    """Pays down principal, debited from the user's balance.

    Effective repayment is clamped to ``min(amount, balance, principal)``
    so over-requests automatically reduce to the largest legal value. The
    user's ``total_spent`` is intentionally **not** bumped because repaying
    a loan isn't spending in the gameplay sense; the ``loan_total_repaid``
    column is the right place to track the repayment.

    Args:
        user_id: Discord user ID for the borrower.
        name: Last-seen Discord username.
        amount: Maximum amount to apply against debt (must be positive).
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        ``RepayResult`` on success, or ``None`` when there's no debt, no
        positive balance, or the retry budget was exhausted.
    """
    await _ensure_schema()
    if amount <= 0:
        return None
    now = _database_now()

    async with open_session() as session:
        await _reset_expired_loan_in_session(session=session, user_id=user_id, now=now)

        for _ in range(_REPAY_MAX_RETRIES):
            read_result = await session.execute(
                statement=select(
                    UserAccount.balance,
                    UserAccount.name,
                    UserAccount.loan_principal,
                    UserAccount.loan_total_repaid,
                ).where(UserAccount.user_id == user_id)
            )
            row = read_result.one_or_none()
            if row is None:
                return None
            current_balance, existing_name, principal, total_repaid = row
            if principal == 0 or current_balance <= 0:
                return None

            effective = min(amount, current_balance, principal)
            new_balance = current_balance - effective
            new_principal = principal - effective
            new_total_repaid = total_repaid + effective

            update_values: dict[str, Any] = {
                "balance": new_balance,
                "loan_principal": new_principal,
                "loan_total_repaid": new_total_repaid,
                "updated_at": now,
            }
            if name and name != existing_name:
                update_values["name"] = name
            if avatar_url:
                update_values["avatar_url"] = avatar_url

            stmt = (
                update(UserAccount)
                .where(
                    UserAccount.user_id == user_id,
                    UserAccount.balance == current_balance,
                    UserAccount.loan_principal == principal,
                )
                .values(**update_values)
                .returning(UserAccount.balance)
            )
            update_result = await session.execute(statement=stmt)
            if update_result.one_or_none() is None:
                await session.rollback()
                continue

            await _log_transaction_in_session(
                session=session,
                user_id=user_id,
                kind=TransactionKind.REPAY,
                delta=-effective,
                balance_after=new_balance,
                debt_after=new_principal,
                note=None,
                now=now,
            )
            await session.commit()
            return RepayResult(
                new_balance=new_balance, principal_repaid=effective, remaining_debt=new_principal
            )
        return None


def _next_checkin_streak(
    last_checkin_at: datetime | None,
    current_streak: int,
    today_midnight: datetime,
    yesterday_midnight: datetime,
    tomorrow_midnight: datetime,
) -> int | None:
    """Returns the streak counter for the next check-in.

    Returns ``None`` when the user has already checked in today.

    Args:
        last_checkin_at: Stored ``last_checkin_at`` (Taipei-naive) or ``None``.
        current_streak: Currently-persisted streak counter.
        today_midnight: 00:00 Asia/Taipei for the request day.
        yesterday_midnight: 00:00 Asia/Taipei for the prior day.
        tomorrow_midnight: 00:00 Asia/Taipei for the next day.

    Returns:
        The streak number to persist, or ``None`` if today is already done.
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

    Returns ``None`` when another coroutine already inserted the row so
    the caller retries on the next loop iteration.

    Args:
        session: Active SQLAlchemy session.
        user_id: Discord user ID checking in.
        name: Last-seen Discord username to store on the account.
        avatar_url: Last-seen Discord avatar URL to store when available.
        now: ``_database_now()`` value pinned for this transaction.

    Returns:
        ``(reward, balance_after, streak_after, vip_after)`` on success or
        ``None`` when ``ON CONFLICT DO NOTHING`` rejected the insert.
    """
    new_streak = 1
    reward = checkin_reward(streak=new_streak, is_vip=False)
    insert_stmt = (
        insert(UserAccount)
        .values(
            user_id=user_id,
            name=name or str(user_id),
            avatar_url=avatar_url,
            balance=reward,
            total_earned=reward,
            total_spent=0,
            loan_principal=0,
            loan_total_borrowed=0,
            loan_total_repaid=0,
            loan_opened_at=None,
            is_vip=False,
            last_checkin_at=now,
            checkin_streak=new_streak,
            updated_at=now,
        )
        .on_conflict_do_nothing(index_elements=["user_id"])
    )
    insert_result = await session.execute(statement=insert_stmt)
    if (insert_result.rowcount or 0) == 0:
        return None
    return reward, reward, new_streak, False


async def _update_checkin_row_in_session(  # noqa: PLR0913 -- session helper carries account identity + observed row
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    now: datetime,
    new_streak: int,
    row: tuple[int, datetime | None, int, bool, str],
) -> tuple[int, int, int, bool] | None:
    """Performs the conditional UPDATE for an existing account.

    The WHERE clause pins ``balance`` and ``last_checkin_at`` to the values
    observed in the SELECT so concurrent writers can't double-credit.

    Args:
        session: Active SQLAlchemy session.
        user_id: Discord user ID checking in.
        name: Last-seen Discord username to refresh on the account.
        avatar_url: Last-seen Discord avatar URL to refresh when set.
        now: ``_database_now()`` value pinned for this transaction.
        new_streak: Streak counter chosen by ``_next_checkin_streak``.
        row: Tuple returned by the prior SELECT.

    Returns:
        ``(reward, balance_after, streak_after, vip_after)`` on success or
        ``None`` when the conditional UPDATE matched zero rows.
    """
    current_balance, last_checkin_at, _current_streak, is_vip, existing_name = row
    reward = checkin_reward(streak=new_streak, is_vip=is_vip)
    new_balance = current_balance + reward

    update_values: dict[str, Any] = {
        "balance": new_balance,
        "total_earned": UserAccount.total_earned + reward,
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
        .where(
            UserAccount.user_id == user_id,
            UserAccount.balance == current_balance,
            last_checkin_gate,
        )
        .values(**update_values)
        .returning(UserAccount.balance, UserAccount.checkin_streak, UserAccount.is_vip)
    )
    update_result = await session.execute(statement=stmt)
    updated_row = update_result.one_or_none()
    if updated_row is None:
        return None
    balance_after, streak_after, vip_after = updated_row
    return reward, balance_after, streak_after, bool(vip_after)


async def checkin(user_id: int, name: str, avatar_url: str = "") -> CheckinResult | None:
    """Records a daily check-in and credits the streak-adjusted reward.

    Returns ``None`` when the user has already checked in today (Taipei
    local date). On first check-in or after a missed day the streak resets
    to 1; otherwise the streak advances by 1 and cycles back to 1 after
    reaching ``CHECKIN_STREAK_CYCLE``. The reward is computed with
    ``checkin_reward`` and persisted alongside the streak counter in the
    same write. VIP perks (2x base) read the persisted flag inside the
    same transaction so a freshly-bought VIP immediately applies on the
    next check-in.

    The SELECT-then-conditional-UPDATE pattern (gated on the
    observed ``last_checkin_at`` value) prevents two parallel coroutines
    from double-crediting. First-sight INSERTs use ``ON CONFLICT DO
    NOTHING`` to defer to whichever writer landed first; the loser falls
    through to the next retry with the freshly-visible row.

    Args:
        user_id: Discord user ID checking in.
        name: Last-seen Discord username to store on the account.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        ``CheckinResult`` describing the credit, or ``None`` when the user
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
                    UserAccount.balance,
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
                    last_checkin_at=row[1],
                    current_streak=row[2],
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
                    row=cast("tuple[int, datetime | None, int, bool, str]", row),
                )

            if outcome is None:
                await session.rollback()
                continue

            reward, balance_after, streak_after, vip_after = outcome
            await _log_transaction_in_session(
                session=session,
                user_id=user_id,
                kind=TransactionKind.CHECKIN_REWARD,
                delta=reward,
                balance_after=balance_after,
                note=f"streak {streak_after}",
                now=now,
            )
            await session.commit()
            return CheckinResult(
                new_balance=balance_after, amount=reward, streak=streak_after, is_vip=vip_after
            )

        return None


async def buy_vip(user_id: int, name: str, avatar_url: str = "") -> VipPurchaseResult | None:
    """Promotes the user to VIP after debiting ``VIP_PURCHASE_COST`` points.

    Returns ``None`` when the user is already VIP, has insufficient balance,
    or the retry budget for the conditional UPDATE was exhausted.

    Args:
        user_id: Discord user ID purchasing VIP.
        name: Last-seen Discord username to store on the account.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        ``VipPurchaseResult`` describing the post-purchase balance, or
        ``None`` when the purchase was rejected.
    """
    await _ensure_schema()
    now = _database_now()
    cost = VIP_PURCHASE_COST

    async with open_session() as session:
        for _ in range(_VIP_PURCHASE_MAX_RETRIES):
            read_result = await session.execute(
                statement=select(UserAccount.balance, UserAccount.is_vip, UserAccount.name).where(
                    UserAccount.user_id == user_id
                )
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
            update_values: dict[str, Any] = {
                "balance": new_balance,
                "total_spent": UserAccount.total_spent + cost,
                "is_vip": True,
                "updated_at": now,
            }
            if name and name != existing_name:
                update_values["name"] = name
            if avatar_url:
                update_values["avatar_url"] = avatar_url

            stmt = (
                update(UserAccount)
                .where(
                    UserAccount.user_id == user_id,
                    UserAccount.balance == balance,
                    UserAccount.is_vip.is_(False),
                )
                .values(**update_values)
                .returning(UserAccount.balance)
            )
            update_result = await session.execute(statement=stmt)
            updated_row = update_result.one_or_none()
            if updated_row is None:
                await session.rollback()
                continue

            await _log_transaction_in_session(
                session=session,
                user_id=user_id,
                kind=TransactionKind.VIP_PURCHASE,
                delta=-cost,
                balance_after=updated_row[0],
                note=None,
                now=now,
            )
            await session.commit()
            return VipPurchaseResult(new_balance=updated_row[0], cost=cost)

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
            statement=select(UserAccount.balance).where(UserAccount.user_id == user_id)
        )
        return result.scalar_one_or_none() or 0


async def get_vip(user_id: int) -> bool:
    """Returns whether the user owns the VIP perk.

    Args:
        user_id: Discord user ID to look up.

    Returns:
        ``True`` when the account has ``is_vip`` set, else ``False``.
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
        ``True`` when the account has ``is_admin`` set, else ``False``.
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
        ``True`` when a row was created or updated; ``False`` when revoking a
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
                balance=0,
                total_earned=0,
                total_spent=0,
                updated_at=now,
                loan_principal=0,
                loan_total_borrowed=0,
                loan_total_repaid=0,
                loan_opened_at=None,
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
                UserAccount.balance,
                UserAccount.total_earned,
                UserAccount.total_spent,
            ).where(UserAccount.user_id == user_id)
        )
        row = result.one_or_none()
        if row is None:
            return None
        return AccountSnapshot(
            name=row[0], balance=row[1], total_earned=row[2], total_spent=row[3]
        )


async def get_loan_view(user_id: int) -> LoanView | None:
    """Returns a stored snapshot of the user's loan state.

    The principal is read after applying the daily reset so callers always
    see the current Taipei-day picture; a loan opened the previous day will
    surface as ``principal == 0`` and ``opened_at is None``.

    Args:
        user_id: Discord user ID to look up.

    Returns:
        ``LoanView`` for the user, or ``None`` if the user has never been
        seen by the economy DB.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        await _reset_expired_loan_in_session(session=session, user_id=user_id, now=now)
        await session.commit()
        result = await session.execute(
            statement=select(
                UserAccount.loan_principal,
                UserAccount.loan_opened_at,
                UserAccount.loan_total_borrowed,
                UserAccount.loan_total_repaid,
            ).where(UserAccount.user_id == user_id)
        )
        row = result.one_or_none()
        if row is None:
            return None
        return LoanView(
            principal=row[0], opened_at=row[1], total_borrowed=row[2], total_repaid=row[3]
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
            "balance": UserAccount.balance - amount,
            "total_spent": UserAccount.total_spent + amount,
            "updated_at": now,
        }
        if sender_name:
            debit_values["name"] = sender_name
        if sender_avatar_url:
            debit_values["avatar_url"] = sender_avatar_url

        debit_stmt = (
            update(UserAccount)
            .where(UserAccount.user_id == sender_id, UserAccount.balance >= amount)
            .values(**debit_values)
            .returning(UserAccount.balance)
        )
        debit_result = await session.execute(statement=debit_stmt)
        debit_row = debit_result.one_or_none()
        if debit_row is None:
            await session.rollback()
            return None
        sender_balance = debit_row[0]

        credit_stmt = _build_credit_upsert(
            user_id=receiver_id,
            name=receiver_name,
            avatar_url=receiver_avatar_url,
            amount=amount,
            now=now,
        )
        credit_result = await session.execute(statement=credit_stmt)
        receiver_balance = credit_result.scalar_one()

        sender_note = f"to {receiver_name or receiver_id} ({receiver_id})"
        receiver_note = f"from {sender_name or sender_id} ({sender_id})"
        await _log_transaction_in_session(
            session=session,
            user_id=sender_id,
            kind=TransactionKind.TRANSFER_OUT,
            delta=-amount,
            balance_after=sender_balance,
            note=sender_note,
            now=now,
        )
        await _log_transaction_in_session(
            session=session,
            user_id=receiver_id,
            kind=TransactionKind.TRANSFER_IN,
            delta=amount,
            balance_after=receiver_balance,
            note=receiver_note,
            now=now,
        )
        await session.commit()
        return TransferResult(sender_balance=sender_balance, receiver_balance=receiver_balance)


async def top_n(limit: int = 10, exclude_user_ids: tuple[int, ...] = ()) -> list[LeaderboardEntry]:
    """Returns accounts ordered by balance descending.

    ``exclude_user_ids`` filters out specific accounts (notably the bot's
    own house ledger row) before applying the limit, so the leaderboard
    always shows real players. The ``ix_user_account_balance`` index keeps
    this query cheap even as the user table grows.

    Args:
        limit: Maximum number of accounts to return.
        exclude_user_ids: User IDs to filter out before applying the limit.

    Returns:
        Leaderboard entries ordered by balance descending. ``avatar_url`` is
        empty when the user has never been seen by an avatar-aware write path.
    """
    await _ensure_schema()
    async with open_session() as session:
        stmt = select(
            UserAccount.user_id, UserAccount.name, UserAccount.balance, UserAccount.avatar_url
        ).order_by(desc(UserAccount.balance))
        if exclude_user_ids:
            stmt = stmt.where(UserAccount.user_id.notin_(other=exclude_user_ids))
        stmt = stmt.limit(limit=limit)
        result = await session.execute(statement=stmt)
        return [
            LeaderboardEntry(user_id=row[0], name=row[1], balance=row[2], avatar_url=row[3] or "")
            for row in result.all()
        ]


async def top_losers(
    limit: int = 10, exclude_user_ids: tuple[int, ...] = ()
) -> list[LossLeaderboardEntry]:
    """Returns the biggest gross casino losers for the current Taipei day.

    The leaderboard reads persisted ``user_account`` daily counters instead
    of aggregating ``point_transaction`` rows. Writes lazily reset stale
    counters at the first casino settlement after Taipei midnight, while this
    query filters by today's ``casino_day_started_at`` so yesterday's counters
    never leak into a new day.

    Args:
        limit: Maximum number of accounts to return.
        exclude_user_ids: User IDs to filter out before applying the limit.

    Returns:
        Loss leaderboard entries ordered by loss descending. ``loss_amount``
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
                UserAccount.user_id,
                UserAccount.name,
                UserAccount.avatar_url,
                UserAccount.daily_casino_loss,
            )
            .where(
                UserAccount.casino_day_started_at == today_midnight,
                UserAccount.daily_casino_loss > 0,
            )
            .order_by(desc(UserAccount.daily_casino_loss))
            .limit(limit=limit)
        )
        if exclude_user_ids:
            stmt = stmt.where(UserAccount.user_id.notin_(other=exclude_user_ids))
        result = await session.execute(statement=stmt)
        return [
            LossLeaderboardEntry(
                user_id=row[0],
                name=row[1] or str(row[0]),
                loss_amount=row[3],
                avatar_url=row[2] or "",
            )
            for row in result.all()
        ]
