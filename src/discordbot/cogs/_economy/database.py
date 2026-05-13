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
The UPSERT pattern fixes both. `place_bet`, `repay` and the inner
`_credit_with_repayment_in_session` keep a SELECT-then-conditional-UPDATE
retry loop because they need to inspect the current value to compute the
delta, but the retry is bounded and the WHERE clause guarantees no
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

VIP is a single boolean column on ``user_account``. It bumps daily check-in
rewards, the borrow cap, and the player's winning payout from games. The
flag is permanent once set.
"""

from typing import Any, Final
from datetime import UTC, datetime, timezone, timedelta

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

from discordbot.typings.economy import (
    VIP_PURCHASE_COST,
    CHECKIN_STREAK_CYCLE,
    CHECKIN_STREAK_BONUS_STEP,
    BASE_CHECKIN_REWARD_AMOUNT,
    LoanView,
    PlacedBet,
    RepayResult,
    BorrowResult,
    CreditResult,
    CheckinResult,
    TransferResult,
    TransactionKind,
    VipPurchaseResult,
)

# place_bet / repay / _credit_with_repayment_in_session keep a small retry
# budget for SELECT-then-conditional-UPDATE loops. With WAL + busy_timeout,
# contention is rare and resolves on the first or second retry; the bound
# prevents a degenerate hot-row livelock.
_PLACE_BET_MAX_RETRIES: Final[int] = 8
_CREDIT_WITH_REPAYMENT_MAX_RETRIES: Final[int] = 8
_REPAY_MAX_RETRIES: Final[int] = 8
_CHECKIN_MAX_RETRIES: Final[int] = 8
_VIP_PURCHASE_MAX_RETRIES: Final[int] = 8
# Blackjack VIP perk: 1.5x payout on winning rounds, applied as floor(delta * 3 / 2).
_VIP_WIN_MULTIPLIER_NUM: Final[int] = 3
_VIP_WIN_MULTIPLIER_DEN: Final[int] = 2
TAIWAN_TIMEZONE: Final[timezone] = timezone(offset=timedelta(hours=8), name="Asia/Taipei")

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
    """

    __tablename__ = "user_account"
    __table_args__ = (
        # /leaderboard does ORDER BY balance DESC LIMIT 10; the index turns a
        # full scan into a bounded walk. SQLite can use an ASC index to satisfy
        # ORDER BY DESC by reading it backwards, so no DESC index is needed.
        Index("ix_user_account_balance", "balance"),
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


class PointTransaction(Base):
    """Append-only audit log of every persistent balance change.

    One row per balance-mutating event. ``balance_after`` and ``debt_after``
    reflect the user's state *after* the row's write, so consecutive rows
    for the same user can be diffed to reconstruct every income / spend.
    ``occurred_at`` carries the Asia/Taipei wall clock so daily windows
    (e.g. the loss leaderboard) can simply filter on the column.

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
        # /loss_leaderboard filters by (occurred_at, kind); the composite
        # index keeps the daily window scan cheap as the audit log grows.
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
        seeded_amount: One-time on-the-house seed; bookkeeping only,
            never decremented.
        updated_at: Taiwan-local timestamp of the last write.
    """

    __tablename__ = "jackpot_pool"

    game_id: Mapped[str] = mapped_column(String(length=32), primary_key=True)
    pool_balance: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_contributed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_claimed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    seeded_amount: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


# Initial on-the-house seeds for each registered jackpot pool. The seed is
# bookkeeping only — the bot's user_account row is never decremented to fund
# it, so /house P&L stays unaffected by the donation.
_JACKPOT_SEEDS: Final[tuple[tuple[str, int], ...]] = (("dragon_gate", 100_000),)


# Track which engine the schema has already been bootstrapped on. Storing
# the engine identity (not just a bool) means swapping `_engine` (e.g. tests
# pointing it at a temp file) automatically forces another schema check;
# production never re-enters past the fast path. We intentionally do NOT
# use an asyncio.Lock here: module-level Locks bind to the first event loop
# they're awaited from and break under pytest's per-test loops. `create_all`
# is idempotent (CREATE TABLE IF NOT EXISTS), so a benign double-create on
# initial races is fine.
_schema_ready_for: AsyncEngine | None = None


async def _ensure_schema() -> None:
    """Bootstraps the schema once per ``_engine`` value.

    Idempotent migrations:

    * Adds ``avatar_url`` to legacy DBs that predated the avatar cache.
    * Adds ``is_vip`` / ``last_checkin_at`` / ``checkin_streak`` so VIP
      and check-in features keep working on DBs from before they shipped.
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
        if "loan_interest" in existing_columns:
            await conn.execute(
                statement=text(text="ALTER TABLE user_account DROP COLUMN loan_interest")
            )
        if "loan_last_accrual_at" in existing_columns:
            await conn.execute(
                statement=text(text="ALTER TABLE user_account DROP COLUMN loan_last_accrual_at")
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

    The reward formula is ``BASE * (1 + (streak - 1) * CHECKIN_STREAK_BONUS_STEP)``
    where ``streak`` is the 1..``CHECKIN_STREAK_CYCLE`` day in the cycle.
    VIP doubles the base before the streak bonus.

    Args:
        streak: Streak counter for this check-in (1..``CHECKIN_STREAK_CYCLE``).
        is_vip: VIP status of the account at check-in time.

    Returns:
        Integer reward amount.
    """
    base = BASE_CHECKIN_REWARD_AMOUNT * (2 if is_vip else 1)
    multiplier = 1.0 + (streak - 1) * CHECKIN_STREAK_BONUS_STEP
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


def _build_clamped_settle_upsert(
    user_id: int, name: str, delta: int, now: datetime, avatar_url: str = ""
) -> ReturningInsert[tuple[int]]:
    """UPSERT applying a signed ``delta`` with the resulting balance clamped at 0.

    Mirrors the original `settle_game` semantics: positive `delta` credits;
    negative `delta` debits but never lets the balance go negative. The
    `applied_delta` (post-clamp) is what feeds `total_earned` / `total_spent`,
    so a loss larger than the balance only counts the actual amount spent.

    Returns:
        A SQLAlchemy `Insert` with `on_conflict_do_update` and `returning(balance)`.
    """
    insert_name = name or str(user_id)
    initial_balance = max(delta, 0)
    stmt = insert(UserAccount).values(
        user_id=user_id,
        name=insert_name,
        avatar_url=avatar_url,
        balance=initial_balance,
        total_earned=initial_balance,
        total_spent=0,
        updated_at=now,
    )
    new_balance_expr = func.max(UserAccount.balance + delta, 0)
    applied_delta_expr = new_balance_expr - UserAccount.balance
    set_: dict[str, Any] = {
        "balance": new_balance_expr,
        "total_earned": UserAccount.total_earned
        + case((applied_delta_expr > 0, applied_delta_expr), else_=0),
        "total_spent": UserAccount.total_spent
        + case((applied_delta_expr < 0, -applied_delta_expr), else_=0),
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
    """Inner credit-with-50%-auto-repay pipeline; caller must commit the session.

    Pipeline (all inside the caller's transaction):

    1. ``_reset_expired_loan_in_session`` clears the previous day's loan.
    2. SELECT current balance + loan state.
    3. ``to_repay = min(amount // 2, principal)``.
    4. Repayment debits ``loan_principal``; remainder credits balance.
    5. Conditional UPDATE gated on the values we read; retry on conflict.
    6. ``_log_transaction_in_session`` writes one audit row with
       ``delta = credited_amount`` and the post-state debt.

    Caller must guarantee ``amount > 0``.
    """
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
        to_repay = min(amount // 2, principal)
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


async def _settle_game_in_session(  # noqa: PLR0913 -- session helper needs both ledger keys + kind
    session: AsyncSession,
    user_id: int,
    name: str,
    avatar_url: str,
    delta: int,
    kind: TransactionKind,
    now: datetime,
) -> int:
    """Applies a clamped settle delta and logs the resulting audit row.

    The clamp means a stale caller asking for a larger loss than the user
    can afford settles as "spent everything you had", not as a negative
    balance. To log the *applied* delta rather than the requested one,
    this helper reads the pre-update balance first and diffs against the
    post-update value returned by the UPSERT.
    """
    pre_result = await session.execute(
        statement=select(UserAccount.balance).where(UserAccount.user_id == user_id)
    )
    pre_balance = pre_result.scalar_one_or_none() or 0
    stmt = _build_clamped_settle_upsert(
        user_id=user_id, name=name, avatar_url=avatar_url, delta=delta, now=now
    )
    result = await session.execute(statement=stmt)
    new_balance = result.scalar_one()
    applied = new_balance - pre_balance
    await _log_transaction_in_session(
        session=session,
        user_id=user_id,
        kind=kind,
        delta=applied,
        balance_after=new_balance,
        note=None,
        now=now,
    )
    return new_balance


async def _casino_debit_in_session(  # noqa: PLR0913 -- session helper needs ledger identity + delta
    session: AsyncSession, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
) -> int:
    """Applies a casino loss without clamping the player's balance at zero."""
    stmt = _build_signed_delta_upsert(
        user_id=user_id, name=name, avatar_url=avatar_url, delta=delta, now=now
    )
    result = await session.execute(statement=stmt)
    new_balance = result.scalar_one()
    await _log_transaction_in_session(
        session=session,
        user_id=user_id,
        kind=TransactionKind.CASINO_BET,
        delta=delta,
        balance_after=new_balance,
        note=None,
        now=now,
    )
    return new_balance


async def _house_settle_in_session(  # noqa: PLR0913 -- session helper needs ledger identity + delta
    session: AsyncSession, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
) -> int:
    """Applies a signed delta to the dealer ledger and logs the audit row.

    Unlike ``_settle_game_in_session`` this never clamps; the dealer is
    allowed to go negative. The applied delta equals the requested delta
    so no pre-read is required.
    """
    stmt = _build_signed_delta_upsert(
        user_id=user_id, name=name, avatar_url=avatar_url, delta=delta, now=now
    )
    result = await session.execute(statement=stmt)
    new_balance = result.scalar_one()
    await _log_transaction_in_session(
        session=session,
        user_id=user_id,
        kind=TransactionKind.HOUSE_SETTLE,
        delta=delta,
        balance_after=new_balance,
        note=None,
        now=now,
    )
    return new_balance


async def add_balance(user_id: int, name: str, amount: int, avatar_url: str = "") -> int:
    """Adds points to the user balance without touching the loan state.

    Non-positive amounts no-op and return the existing balance so callers
    can pass raw token counts (which can be 0 for cached responses) without
    a guard. Intentionally **does not** trigger 50% auto-repayment; this
    is the low-level credit primitive (mainly used by tests and any future
    "no-questions-asked" credit path). Production code paths that should
    pay down debt use ``credit_with_repayment`` instead.

    Implemented as a single SQLite UPSERT, so two coroutines racing on the
    same user can neither lose updates nor crash with an `IntegrityError`
    on a brand-new account.

    Args:
        user_id: Discord user ID to credit.
        name: Last-seen Discord username to store on the account.
        amount: Number of points to add.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        The user's current balance after the operation.
    """
    await _ensure_schema()
    if amount <= 0:
        return await get_balance(user_id=user_id)
    stmt = _build_credit_upsert(
        user_id=user_id, name=name, avatar_url=avatar_url, amount=amount, now=_database_now()
    )
    async with open_session() as session:
        result = await session.execute(statement=stmt)
        await session.commit()
        return result.scalar_one()


async def credit_with_repayment(  # noqa: PLR0913 -- public DB facade mirrors one income event
    user_id: int,
    name: str,
    amount: int,
    kind: TransactionKind,
    note: str | None = None,
    avatar_url: str = "",
) -> CreditResult:
    """Credits ``amount`` to the user while diverting 50% to repay debt first.

    Repayment goes entirely against principal (the interest system has been
    removed) and is capped at the user's outstanding loan. Any portion not
    used for repayment goes to balance and bumps ``total_earned``. Writes
    one audit row via the helper.

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


async def place_bet(
    user_id: int, name: str, requested_bet: int, avatar_url: str = ""
) -> PlacedBet | None:
    """Atomically withdraws a wager.

    Bets larger than the current balance are clamped to the full available
    balance (auto all-in). The conditional update protects against stale
    balance reads from concurrent game commands, so the same points cannot be
    spent twice. The retry loop runs at most ``_PLACE_BET_MAX_RETRIES`` times;
    in WAL mode with `busy_timeout`, contention almost always resolves on the
    first or second retry.

    Args:
        user_id: Discord user ID placing the wager.
        name: Last-seen Discord username to store on the account.
        requested_bet: Requested wager amount in points.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        The withdrawn wager details, or `None` when the user has no spendable
        balance, the requested bet is not positive, or the retry budget was
        exhausted under unusual contention.
    """
    await _ensure_schema()
    if requested_bet <= 0:
        return None

    async with open_session() as session:
        for _ in range(_PLACE_BET_MAX_RETRIES):
            read_result = await session.execute(
                statement=select(UserAccount.balance, UserAccount.name).where(
                    UserAccount.user_id == user_id
                )
            )
            row = read_result.one_or_none()
            if row is None or row[0] <= 0:
                return None
            starting_balance, existing_name = row[0], row[1]
            effective_bet = min(requested_bet, starting_balance)

            now = _database_now()
            update_values: dict[str, Any] = {
                "balance": UserAccount.balance - effective_bet,
                "total_spent": UserAccount.total_spent + effective_bet,
                "updated_at": now,
            }
            if name and name != existing_name:
                update_values["name"] = name
            if avatar_url:
                update_values["avatar_url"] = avatar_url

            stmt = (
                update(UserAccount)
                .where(UserAccount.user_id == user_id, UserAccount.balance == starting_balance)
                .values(**update_values)
                .returning(UserAccount.balance)
            )
            update_result = await session.execute(statement=stmt)
            updated_row = update_result.one_or_none()
            if updated_row is None:
                # Someone committed a balance change between our SELECT and
                # UPDATE; rollback the autobegun transaction and retry with
                # a fresh read.
                await session.rollback()
                continue
            await _log_transaction_in_session(
                session=session,
                user_id=user_id,
                kind=TransactionKind.CASINO_BET,
                delta=-effective_bet,
                balance_after=updated_row[0],
                note=None,
                now=now,
            )
            await session.commit()
            return PlacedBet(
                amount=effective_bet,
                balance_after=updated_row[0],
                is_allin=effective_bet < requested_bet,
            )
        return None


async def settle_game(user_id: int, name: str, delta: int, avatar_url: str = "") -> int:
    """Applies a signed player balance adjustment.

    This is kept as a clamped low-level adjustment helper. Current casino
    commands use `apply_round_settlement()` so unfinished in-memory rounds do
    not mutate balances, and finished losses can still debit below zero.
    Implemented as a single UPSERT, so concurrent settlements on the same user
    can't lose updates. The audit log records the *applied* delta (after
    clamping).

    Args:
        user_id: Discord user ID whose balance is adjusted.
        name: Last-seen Discord username to store on the account.
        delta: Signed point adjustment to apply.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        The user's current balance after settlement.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        new_balance = await _settle_game_in_session(
            session=session,
            user_id=user_id,
            name=name,
            avatar_url=avatar_url,
            delta=delta,
            kind=TransactionKind.CASINO_PAYOUT,
            now=now,
        )
        await session.commit()
        return new_balance


async def house_settle(user_id: int, name: str, delta: int, avatar_url: str = "") -> int:
    """Records a dealer-side settlement.

    Used to track the bot's casino P&L over time. The dealer has effectively
    infinite funds (it backs every bet), so unlike `settle_game` we
    deliberately do not clamp at zero. A long-running losing streak
    should surface as a negative balance, with `total_earned` /
    `total_spent` accumulating gross flows in each direction.

    Implemented as a single UPSERT so the bot's house-ledger row — which is
    the single hottest row in the schema, since every player settlement
    mirrors into it — doesn't lose updates under concurrent games.

    Args:
        user_id: Discord user ID for the dealer ledger row.
        name: Last-seen display name to store on the ledger row.
        delta: Signed point adjustment to apply.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        The dealer ledger balance after settlement, which may be negative.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        new_balance = await _house_settle_in_session(
            session=session,
            user_id=user_id,
            name=name,
            avatar_url=avatar_url,
            delta=delta,
            now=now,
        )
        await session.commit()
        return new_balance


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
    ``_credit_with_repayment_in_session`` so the 50% auto-repay rule applies
    to casino profit. Negative deltas are debited without a zero clamp so a
    player cannot evade a bad in-memory round by moving funds before it settles.

    Args:
        player_id: Discord user ID for the player account.
        player_account_name: Account name to store for the player.
        player_avatar_url: Last-seen Discord avatar URL for the player.
        player_delta: Signed net change for the player. Losses may make the
            balance negative.
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
        if player_delta > 0:
            credit_result = await _credit_with_repayment_in_session(
                session=session,
                user_id=player_id,
                name=player_account_name,
                avatar_url=player_avatar_url,
                amount=player_delta,
                kind=TransactionKind.CASINO_PAYOUT,
                note=None,
                now=now,
            )
            player_balance = credit_result.new_balance
        elif player_delta < 0:
            player_balance = await _casino_debit_in_session(
                session=session,
                user_id=player_id,
                name=player_account_name,
                avatar_url=player_avatar_url,
                delta=player_delta,
                now=now,
            )
        else:
            read_result = await session.execute(
                statement=select(UserAccount.balance).where(UserAccount.user_id == player_id)
            )
            player_balance = read_result.scalar_one_or_none() or 0

        if dealer_delta == 0:
            dealer_result = await session.execute(
                statement=select(UserAccount.balance).where(UserAccount.user_id == dealer_id)
            )
            dealer_balance = dealer_result.scalar_one_or_none() or 0
        else:
            dealer_balance = await _house_settle_in_session(
                session=session,
                user_id=dealer_id,
                name=dealer_name,
                avatar_url=dealer_avatar_url,
                delta=dealer_delta,
                now=now,
            )
        await session.commit()
        return player_balance, dealer_balance


async def get_jackpot_pool(game_id: str) -> int:
    """Returns the current ``pool_balance`` for a game's shared jackpot.

    Reading the seeded row is the canonical way to surface the current
    pool to a view (lobby start, every active-table refresh). Returns ``0``
    when the row hasn't been seeded yet so a freshly-introduced game can
    short-circuit cleanly.

    Args:
        game_id: Game identifier (e.g. ``"dragon_gate"``).

    Returns:
        The current pool balance in points.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(JackpotPool.pool_balance).where(JackpotPool.game_id == game_id)
        )
        return result.scalar_one_or_none() or 0


async def _apply_jackpot_delta_in_session(
    session: AsyncSession, game_id: str, delta: int, now: datetime
) -> int:
    """Applies a signed delta to a game's jackpot pool inside the caller's session.

    Positive deltas accumulate ``total_contributed`` (player losses /
    antes flowing into the pool); negative deltas accumulate
    ``total_claimed`` with the absolute value (winning payouts flowing
    out). The pool is allowed to dip below zero in edge cases (e.g. a
    win larger than the remaining balance), mirroring the dealer ledger's
    no-clamp behaviour.

    Args:
        session: Active SQLAlchemy session bound to ``_engine``.
        game_id: Game identifier (jackpot row primary key).
        delta: Signed point adjustment to apply to ``pool_balance``.
        now: ``_database_now()`` value pinned for this transaction.

    Returns:
        The pool balance after the write.
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
        .returning(JackpotPool.pool_balance)
    )
    result = await session.execute(statement=stmt)
    return result.scalar_one()


async def apply_jackpot_settlement(
    player_id: int,
    player_account_name: str,
    player_delta: int,
    game_id: str,
    player_avatar_url: str = "",
) -> tuple[int, int]:
    """Atomic player-and-jackpot settlement for a single wager event.

    Mirrors ``apply_round_settlement`` but routes the counter-party flow
    into the shared jackpot pool rather than the dealer ledger. Positive
    player deltas (wins) credit the player via the 50% auto-repayment
    path and drain the pool by the same amount; negative deltas (losses)
    debit the player without a zero clamp and feed the pool. Both writes
    share one SQLite transaction so a crash between them cannot leave the
    pool drifting from the player result.

    Args:
        player_id: Discord user ID for the player.
        player_account_name: Account name to store on the player row.
        player_delta: Signed net change for the player. Losses are written
            as a negative delta and the absolute value flows into the pool.
        game_id: Jackpot game identifier (e.g. ``"dragon_gate"``).
        player_avatar_url: Last-seen Discord avatar URL for the player.

    Returns:
        A ``(player_balance_after, jackpot_balance_after)`` tuple.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        if player_delta > 0:
            credit_result = await _credit_with_repayment_in_session(
                session=session,
                user_id=player_id,
                name=player_account_name,
                avatar_url=player_avatar_url,
                amount=player_delta,
                kind=TransactionKind.CASINO_PAYOUT,
                note=None,
                now=now,
            )
            player_balance = credit_result.new_balance
        elif player_delta < 0:
            player_balance = await _casino_debit_in_session(
                session=session,
                user_id=player_id,
                name=player_account_name,
                avatar_url=player_avatar_url,
                delta=player_delta,
                now=now,
            )
        else:
            read_result = await session.execute(
                statement=select(UserAccount.balance).where(UserAccount.user_id == player_id)
            )
            player_balance = read_result.scalar_one_or_none() or 0

        if player_delta == 0:
            pool_result = await session.execute(
                statement=select(JackpotPool.pool_balance).where(JackpotPool.game_id == game_id)
            )
            jackpot_balance = pool_result.scalar_one_or_none() or 0
        else:
            jackpot_balance = await _apply_jackpot_delta_in_session(
                session=session, game_id=game_id, delta=-player_delta, now=now
            )
        await session.commit()
        return player_balance, jackpot_balance


async def borrow(
    user_id: int, name: str, amount: int, credit_limit_value: int, avatar_url: str = ""
) -> BorrowResult | None:
    """Disburses ``amount`` points to the user as new principal.

    Rejected (``None`` returned) when ``amount`` is non-positive or when
    the post-borrow principal (existing + requested amount) would exceed
    ``credit_limit_value``. Loans expire at the next Asia/Taipei midnight,
    so the daily cap matches the daily reset window. Borrowed funds do
    **not** bump ``total_earned`` — debt isn't earnings.

    Args:
        user_id: Discord user ID for the borrower.
        name: Last-seen Discord username.
        amount: Amount to borrow (must be positive).
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
        await _reset_expired_loan_in_session(session=session, user_id=user_id, now=now)

        read_result = await session.execute(
            statement=select(UserAccount.loan_principal).where(UserAccount.user_id == user_id)
        )
        current_principal = read_result.scalar_one_or_none() or 0

        if current_principal + amount > credit_limit_value:
            return None

        insert_name = name or str(user_id)
        base_stmt = insert(UserAccount).values(
            user_id=user_id,
            name=insert_name,
            avatar_url=avatar_url,
            balance=amount,
            total_earned=0,
            total_spent=0,
            loan_principal=amount,
            loan_total_borrowed=amount,
            loan_total_repaid=0,
            loan_opened_at=now,
            updated_at=now,
        )
        set_values: dict[str, Any] = {
            "balance": UserAccount.balance + amount,
            "loan_principal": UserAccount.loan_principal + amount,
            "loan_total_borrowed": UserAccount.loan_total_borrowed + amount,
            "loan_opened_at": func.coalesce(UserAccount.loan_opened_at, now),
            "updated_at": now,
        }
        if name:
            set_values["name"] = insert_name
        if avatar_url:
            set_values["avatar_url"] = avatar_url
        upsert_stmt = base_stmt.on_conflict_do_update(
            index_elements=["user_id"], set_=set_values
        ).returning(UserAccount.balance, UserAccount.loan_principal)
        result = await session.execute(statement=upsert_stmt)
        balance_after, principal_after = result.one()

        await _log_transaction_in_session(
            session=session,
            user_id=user_id,
            kind=TransactionKind.BORROW,
            delta=amount,
            balance_after=balance_after,
            debt_after=principal_after,
            note=None,
            now=now,
        )
        await session.commit()
        return BorrowResult(new_balance=balance_after, principal=principal_after)


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
    """Returns the streak counter for the next check-in, or ``None`` when
    the user has already checked in today.

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

    Concurrency: a SELECT-then-conditional-UPDATE pattern (gated on the
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
                    row=row,
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


async def get_account(user_id: int) -> tuple[str, int, int, int] | None:
    """Returns the stored account snapshot for a user.

    Args:
        user_id: Discord user ID to look up.

    Returns:
        A `(name, balance, total_earned, total_spent)` tuple, or `None` if the
        user has never been seen.
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
        return (row[0], row[1], row[2], row[3])


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
        sender_avatar_url: Last-seen Discord avatar URL for the sender.
        receiver_id: Discord user ID to credit.
        receiver_name: Last-seen Discord username to store on the receiver account.
        receiver_avatar_url: Last-seen Discord avatar URL for the receiver.
        amount: Number of points to transfer.

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


async def top_n(
    limit: int = 10, exclude_user_ids: tuple[int, ...] = ()
) -> list[tuple[int, str, int, str]]:
    """Returns accounts ordered by balance descending.

    ``exclude_user_ids`` filters out specific accounts (notably the bot's
    own house ledger row) before applying the limit, so the leaderboard
    always shows real players. The ``ix_user_account_balance`` index keeps
    this query cheap even as the user table grows.

    Args:
        limit: Maximum number of accounts to return.
        exclude_user_ids: User IDs to filter out before applying the limit.

    Returns:
        `(user_id, name, balance, avatar_url)` tuples ordered by balance
        descending. ``avatar_url`` is empty when the user has never been
        seen by an avatar-aware write path.
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
        return [(row[0], row[1], row[2], row[3]) for row in result.all()]


async def top_losers(
    limit: int = 10, exclude_user_ids: tuple[int, ...] = ()
) -> list[tuple[int, str, int, str]]:
    """Returns the biggest net casino losers since today's Taipei midnight.

    "Loss" sums every ``CASINO_BET`` and ``CASINO_PAYOUT`` delta written to
    the audit log within the current Asia/Taipei day. A user with a net-
    negative casino position is on the leaderboard; net-positive users are
    filtered out. The reported loss is the absolute value of the net so the
    `/loss_leaderboard` embed reads naturally.

    The audit log is the only source of truth: the daily window resets
    automatically when the date rolls over, no background task required.

    Args:
        limit: Maximum number of accounts to return.
        exclude_user_ids: User IDs to filter out before applying the limit.

    Returns:
        `(user_id, name, loss_amount, avatar_url)` tuples ordered by loss
        descending. ``loss_amount`` is always positive.
    """
    await _ensure_schema()
    if limit <= 0:
        return []
    now = _database_now()
    today_midnight = _taipei_midnight(now=now)
    net_delta = func.sum(PointTransaction.delta).label("net_delta")

    async with open_session() as session:
        stmt = (
            select(PointTransaction.user_id, UserAccount.name, UserAccount.avatar_url, net_delta)
            .join(UserAccount, UserAccount.user_id == PointTransaction.user_id, isouter=True)
            .where(
                PointTransaction.occurred_at >= today_midnight,
                PointTransaction.kind.in_(
                    other=(TransactionKind.CASINO_BET.value, TransactionKind.CASINO_PAYOUT.value)
                ),
            )
            .group_by(PointTransaction.user_id, UserAccount.name, UserAccount.avatar_url)
            .having(net_delta < 0)
            .order_by(net_delta)
            .limit(limit=limit)
        )
        if exclude_user_ids:
            stmt = stmt.where(PointTransaction.user_id.notin_(other=exclude_user_ids))
        result = await session.execute(statement=stmt)
        return [
            (row[0], row[1] or str(row[0]), int(-row[3]), row[2] or "") for row in result.all()
        ]
