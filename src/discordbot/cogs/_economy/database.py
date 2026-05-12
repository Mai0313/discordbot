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
inside the same single-row UPDATE. The audit log lives in a separate
``point_transaction`` table that every mutating helper writes into via
``_log_transaction_in_session``.
"""

from typing import Any, Final
from datetime import UTC, datetime

from sqlalchemy import (
    Index,
    String,
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
    LoanView,
    PlacedBet,
    RepayResult,
    BorrowResult,
    CreditResult,
    TransferResult,
    TransactionKind,
)

# place_bet / repay / _credit_with_repayment_in_session keep a small retry
# budget for SELECT-then-conditional-UPDATE loops. With WAL + busy_timeout,
# contention is rare and resolves on the first or second retry; the bound
# prevents a degenerate hot-row livelock.
_PLACE_BET_MAX_RETRIES: Final[int] = 8
_CREDIT_WITH_REPAYMENT_MAX_RETRIES: Final[int] = 8
_REPAY_MAX_RETRIES: Final[int] = 8

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/economy.db")


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
    """Persistent balance and loan state for a Discord user.

    Loan-related columns live on the same row as the balance so a single
    UPDATE can credit balance and pay down debt atomically. ``loan_opened_at``
    and ``loan_last_accrual_at`` are nullable because a user that has never
    borrowed has no opening date and no accrual baseline.

    Attributes:
        user_id: Discord user ID; primary key.
        name: Last-seen Discord username (refreshed on every write).
        avatar_url: Last-seen Discord avatar URL (refreshed on writes that carry it).
        balance: Current spendable point balance.
        total_earned: Lifetime points earned (chat rewards, game wins, transfers in).
        total_spent: Lifetime points removed (game losses, transfers out).
        updated_at: UTC timestamp of the last write.
        loan_principal: Currently outstanding loan principal.
        loan_interest: Currently accrued (and not-yet-paid) interest.
        loan_total_borrowed: Lifetime gross borrowed amount.
        loan_total_repaid: Lifetime gross repaid amount.
        loan_last_accrual_at: Timestamp the stored interest was last brought
            up to date; ``None`` while the user has never borrowed.
        loan_opened_at: Timestamp the user first borrowed; ``None`` while
            the user has never borrowed.
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
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        onupdate=lambda: datetime.now(tz=UTC),
    )
    loan_principal: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    loan_interest: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    loan_total_borrowed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    loan_total_repaid: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    loan_last_accrual_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    loan_opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PointTransaction(Base):
    """Append-only audit log of every persistent balance change.

    One row per balance-mutating event. ``balance_after`` and ``debt_after``
    reflect the user's state *after* the row's write, so consecutive rows
    for the same user can be diffed to reconstruct every income / spend.

    Attributes:
        id: Autoincrementing primary key.
        user_id: Discord user ID this row belongs to.
        kind: ``TransactionKind`` enum value as string.
        delta: Signed change applied to balance by this transaction.
        balance_after: Balance after this transaction.
        debt_after: ``loan_principal + loan_interest`` after this transaction.
        note: Optional free-text annotation (e.g. counterparty for transfers).
        occurred_at: UTC timestamp of the event.
    """

    __tablename__ = "point_transaction"
    __table_args__ = (Index("ix_point_transaction_user_time", "user_id", "occurred_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(length=32), nullable=False)
    delta: Mapped[int] = mapped_column(Integer, nullable=False)
    balance_after: Mapped[int] = mapped_column(Integer, nullable=False)
    debt_after: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    note: Mapped[str | None] = mapped_column(String(length=256), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


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
    """Bootstraps the schema once per ``_engine`` value."""
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
    _schema_ready_for = _engine


def open_session() -> AsyncSession:
    """Creates an async session bound to the current economy database engine.

    Returns:
        An `AsyncSession` using the current module-level `_engine`.
    """
    return AsyncSession(bind=_engine, expire_on_commit=False)


def credit_limit(*, user: Any) -> int:  # noqa: ANN401 -- accepts nextcord.User | nextcord.Member; both expose `created_at`
    """Returns the borrowing cap for a Discord account based on its age.

    Computed entirely from ``user.created_at`` (which Discord reconstructs
    from the snowflake ID), so the same cap applies in DMs, guilds, and
    across servers, and a freshly-created account cannot farm by re-joining
    different guilds. Older Discord accounts borrow more because they
    represent a more stable identity.

    Args:
        user: A ``nextcord.User`` or ``nextcord.Member`` whose ``created_at``
            timestamp is inspected.

    Returns:
        Maximum total debt (principal + accrued interest) the account is
        allowed to carry at any single time.
    """
    age_days = (datetime.now(tz=UTC) - user.created_at).days
    if age_days < 30:
        return 1_000
    if age_days < 180:
        return 10_000
    if age_days < 365:
        return 50_000
    if age_days < 365 * 3:
        return 200_000
    return 500_000


def accrual_delta(*, principal: int, last_accrual_at: datetime, now: datetime) -> int:
    """Returns whole-point interest accrued since ``last_accrual_at``.

    Simple interest at 1% per day on the outstanding principal. Compounding
    is intentionally absent so the user can predict the cost. Returns 0
    when the elapsed days * principal floors to zero so sub-point fractions
    accumulate across calls instead of being permanently rounded off.

    SQLite returns naive datetimes for ``DateTime(timezone=True)`` columns,
    so both arguments are coerced to UTC-aware before arithmetic — the
    persisted timestamps were always written as UTC.

    Args:
        principal: Outstanding principal at the reference time.
        last_accrual_at: Timestamp the stored interest was last brought up to date.
        now: Reference time for the accrual.

    Returns:
        Whole-point interest delta to add to ``loan_interest``; always >= 0.
    """
    if principal <= 0:
        return 0
    if last_accrual_at.tzinfo is None:
        last_accrual_at = last_accrual_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    elapsed_days = (now - last_accrual_at).total_seconds() / 86400.0
    if elapsed_days <= 0:
        return 0
    return int(principal * 0.01 * elapsed_days)


def _build_credit_upsert(
    *, user_id: int, name: str, amount: int, now: datetime, avatar_url: str = ""
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
    *, user_id: int, name: str, delta: int, now: datetime, avatar_url: str = ""
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
    *, user_id: int, name: str, delta: int, now: datetime, avatar_url: str = ""
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
    *,
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
    debt to keep the row self-contained; callers that have just computed the
    new debt locally should pass it in to skip the read.
    """
    if delta == 0:
        return
    if debt_after is None:
        debt_result = await session.execute(
            statement=select(UserAccount.loan_principal, UserAccount.loan_interest).where(
                UserAccount.user_id == user_id
            )
        )
        debt_row = debt_result.one_or_none()
        debt_after = (debt_row[0] + debt_row[1]) if debt_row is not None else 0
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


async def _accrue_interest_in_session(
    session: AsyncSession, *, user_id: int, now: datetime
) -> None:
    """Brings ``loan_interest`` up to date in the caller's session.

    Computes the accrual in Python (``accrual_delta``), then applies it with
    a conditional UPDATE gated on the principal and last-accrual values we
    just read. If a concurrent transaction accrues at the same moment the
    conditional UPDATE matches no row, this helper silently no-ops; the
    next accrual will pick up where the winning writer left off.

    Sub-point fractions (``int()`` floor of ``principal * 0.01 * elapsed``)
    are intentionally not persisted, and ``loan_last_accrual_at`` is left
    untouched when the floor is zero so those fractions accumulate across
    calls instead of evaporating.
    """
    read_result = await session.execute(
        statement=select(UserAccount.loan_principal, UserAccount.loan_last_accrual_at).where(
            UserAccount.user_id == user_id
        )
    )
    row = read_result.one_or_none()
    if row is None:
        return
    principal, last_accrual = row[0], row[1]
    if principal <= 0 or last_accrual is None:
        return
    delta = accrual_delta(principal=principal, last_accrual_at=last_accrual, now=now)
    if delta <= 0:
        return
    await session.execute(
        statement=update(UserAccount)
        .where(
            UserAccount.user_id == user_id,
            UserAccount.loan_principal == principal,
            UserAccount.loan_last_accrual_at == last_accrual,
        )
        .values(
            loan_interest=UserAccount.loan_interest + delta,
            loan_last_accrual_at=now,
            updated_at=now,
        )
    )


async def _credit_with_repayment_in_session(  # noqa: PLR0913 -- single-row income pipeline kept linear for readability
    session: AsyncSession,
    *,
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

    1. ``_accrue_interest_in_session`` brings ``loan_interest`` up to date.
    2. SELECT current balance + loan state.
    3. ``to_repay = min(amount // 2, principal + interest)``.
    4. Interest is paid first, then principal; remainder credits to balance.
    5. Conditional UPDATE gated on the values we read; retry on conflict.
    6. ``_log_transaction_in_session`` writes one audit row with
       ``delta = credited_amount`` and the freshly computed ``debt_after``.

    Caller must guarantee ``amount > 0``.
    """
    await _accrue_interest_in_session(session=session, user_id=user_id, now=now)

    for _ in range(_CREDIT_WITH_REPAYMENT_MAX_RETRIES):
        read_result = await session.execute(
            statement=select(
                UserAccount.balance,
                UserAccount.name,
                UserAccount.loan_principal,
                UserAccount.loan_interest,
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
                loan_interest=0,
                loan_total_borrowed=0,
                loan_total_repaid=0,
                loan_last_accrual_at=None,
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
            ).returning(UserAccount.balance, UserAccount.loan_principal, UserAccount.loan_interest)
            insert_result = await session.execute(statement=upsert_stmt)
            balance_after, principal_after, interest_after = insert_result.one()
            await _log_transaction_in_session(
                session=session,
                user_id=user_id,
                kind=kind,
                delta=amount,
                balance_after=balance_after,
                debt_after=principal_after + interest_after,
                note=note,
                now=now,
            )
            return CreditResult(
                new_balance=balance_after,
                credited_amount=amount,
                interest_repaid=0,
                principal_repaid=0,
                remaining_debt=principal_after + interest_after,
            )

        (starting_balance, existing_name, principal, interest, total_earned, total_repaid) = row
        debt_total = principal + interest
        to_repay = min(amount // 2, debt_total)
        interest_repaid = min(to_repay, interest)
        principal_repaid = to_repay - interest_repaid
        credited = amount - to_repay

        new_balance = starting_balance + credited
        new_interest = interest - interest_repaid
        new_principal = principal - principal_repaid
        new_total_earned = total_earned + credited
        new_total_repaid = total_repaid + to_repay

        update_values: dict[str, Any] = {
            "balance": new_balance,
            "total_earned": new_total_earned,
            "loan_interest": new_interest,
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
                UserAccount.loan_interest == interest,
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
            debt_after=new_principal + new_interest,
            note=note,
            now=now,
        )
        return CreditResult(
            new_balance=new_balance,
            credited_amount=credited,
            interest_repaid=interest_repaid,
            principal_repaid=principal_repaid,
            remaining_debt=new_principal + new_interest,
        )

    raise RuntimeError(f"credit_with_repayment retry budget exhausted for user_id={user_id}")


async def _settle_game_in_session(  # noqa: PLR0913 -- session helper needs both ledger keys + kind
    session: AsyncSession,
    *,
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


async def _house_settle_in_session(  # noqa: PLR0913 -- session helper needs ledger identity + delta
    session: AsyncSession, *, user_id: int, name: str, avatar_url: str, delta: int, now: datetime
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
        user_id=user_id, name=name, avatar_url=avatar_url, amount=amount, now=datetime.now(tz=UTC)
    )
    async with open_session() as session:
        result = await session.execute(statement=stmt)
        await session.commit()
        return result.scalar_one()


async def credit_with_repayment(  # noqa: PLR0913 -- public DB facade mirrors one income event
    *,
    user_id: int,
    name: str,
    amount: int,
    kind: TransactionKind,
    note: str | None = None,
    avatar_url: str = "",
) -> CreditResult:
    """Credits ``amount`` to the user while diverting 50% to repay debt first.

    Repayment order is interest first, then principal, capped at the user's
    outstanding debt. Any portion not used for repayment goes to balance
    and bumps ``total_earned``. Writes one audit row via the helper.

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
        repayment breakdown.
    """
    await _ensure_schema()
    if amount <= 0:
        return CreditResult(
            new_balance=await get_balance(user_id=user_id),
            credited_amount=0,
            interest_repaid=0,
            principal_repaid=0,
            remaining_debt=0,
        )
    now = datetime.now(tz=UTC)
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

            now = datetime.now(tz=UTC)
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

    Casino commands withdraw the bet up front with `place_bet()` and then pass
    the gross payout here when the round resolves. Losses are still clamped at
    zero so a stale caller never leaves a player in the red. Implemented as a
    single UPSERT, so concurrent settlements on the same user can't lose
    updates. The audit log records the *applied* delta (after clamping).

    Args:
        user_id: Discord user ID whose balance is adjusted.
        name: Last-seen Discord username to store on the account.
        delta: Signed point adjustment to apply.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        The user's current balance after settlement.
    """
    await _ensure_schema()
    now = datetime.now(tz=UTC)
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
    now = datetime.now(tz=UTC)
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
    *,
    player_id: int,
    player_account_name: str,
    player_avatar_url: str = "",
    payout: int,
    dealer_id: int,
    dealer_name: str,
    dealer_avatar_url: str = "",
    dealer_delta: int,
) -> tuple[int, int]:
    """Credits the player payout (with auto-repay) and mirrors house P&L atomically.

    Sharing a session (and therefore a single SQLite transaction) means a
    crash between the player and dealer writes cannot leave the dealer
    ledger drifting from the player payout. The player payout goes through
    ``_credit_with_repayment_in_session`` so the 50% auto-repay rule
    applies on casino wins exactly as it does on chat rewards.

    Args:
        player_id: Discord user ID for the player account.
        player_account_name: Account name to store for the player.
        player_avatar_url: Last-seen Discord avatar URL for the player.
        payout: Gross amount to credit back to the player.
        dealer_id: Discord user ID for the dealer ledger row.
        dealer_name: Account name to store for the dealer ledger row.
        dealer_avatar_url: Last-seen Discord avatar URL for the dealer.
        dealer_delta: Signed change to apply to the dealer ledger balance.

    Returns:
        A `(player_balance_after, dealer_balance_after)` tuple.
    """
    await _ensure_schema()
    now = datetime.now(tz=UTC)
    async with open_session() as session:
        if payout > 0:
            credit_result = await _credit_with_repayment_in_session(
                session=session,
                user_id=player_id,
                name=player_account_name,
                avatar_url=player_avatar_url,
                amount=payout,
                kind=TransactionKind.CASINO_PAYOUT,
                note=None,
                now=now,
            )
            player_balance = credit_result.new_balance
        else:
            read_result = await session.execute(
                statement=select(UserAccount.balance).where(UserAccount.user_id == player_id)
            )
            player_balance = read_result.scalar_one_or_none() or 0

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


async def borrow(
    *, user_id: int, name: str, amount: int, credit_limit_value: int, avatar_url: str = ""
) -> BorrowResult | None:
    """Disburses ``amount`` points to the user as new principal.

    Rejected (``None`` returned) when ``amount`` is non-positive or when
    the post-borrow total debt (existing principal + accrued interest +
    requested amount) would exceed ``credit_limit_value``. Interest is
    accrued first so the limit check uses up-to-date numbers. Borrowed
    funds do **not** bump ``total_earned`` — debt isn't earnings.

    Args:
        user_id: Discord user ID for the borrower.
        name: Last-seen Discord username.
        amount: Amount to borrow (must be positive).
        credit_limit_value: Maximum allowed post-borrow total debt; the
            caller is expected to compute this with ``credit_limit``.
        avatar_url: Last-seen Discord avatar URL to store when available.

    Returns:
        ``BorrowResult`` capturing the new balance and loan state, or
        ``None`` when the request was rejected.
    """
    await _ensure_schema()
    if amount <= 0:
        return None
    now = datetime.now(tz=UTC)

    async with open_session() as session:
        await _accrue_interest_in_session(session=session, user_id=user_id, now=now)

        read_result = await session.execute(
            statement=select(UserAccount.loan_principal, UserAccount.loan_interest).where(
                UserAccount.user_id == user_id
            )
        )
        row = read_result.one_or_none()
        current_principal = row[0] if row is not None else 0
        current_interest = row[1] if row is not None else 0

        if current_principal + current_interest + amount > credit_limit_value:
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
            loan_interest=0,
            loan_total_borrowed=amount,
            loan_total_repaid=0,
            loan_last_accrual_at=now,
            loan_opened_at=now,
            updated_at=now,
        )
        set_values: dict[str, Any] = {
            "balance": UserAccount.balance + amount,
            "loan_principal": UserAccount.loan_principal + amount,
            "loan_total_borrowed": UserAccount.loan_total_borrowed + amount,
            "loan_last_accrual_at": now,
            "loan_opened_at": func.coalesce(UserAccount.loan_opened_at, now),
            "updated_at": now,
        }
        if name:
            set_values["name"] = insert_name
        if avatar_url:
            set_values["avatar_url"] = avatar_url
        upsert_stmt = base_stmt.on_conflict_do_update(
            index_elements=["user_id"], set_=set_values
        ).returning(UserAccount.balance, UserAccount.loan_principal, UserAccount.loan_interest)
        result = await session.execute(statement=upsert_stmt)
        balance_after, principal_after, interest_after = result.one()

        await _log_transaction_in_session(
            session=session,
            user_id=user_id,
            kind=TransactionKind.BORROW,
            delta=amount,
            balance_after=balance_after,
            debt_after=principal_after + interest_after,
            note=None,
            now=now,
        )
        await session.commit()
        return BorrowResult(
            new_balance=balance_after, principal=principal_after, interest=interest_after
        )


async def repay(
    *, user_id: int, name: str, amount: int, avatar_url: str = ""
) -> RepayResult | None:
    """Pays down interest first then principal, debited from the user's balance.

    Effective repayment is clamped to ``min(amount, balance, debt_total)``
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
        balance, or the retry budget was exhausted.
    """
    await _ensure_schema()
    if amount <= 0:
        return None
    now = datetime.now(tz=UTC)

    async with open_session() as session:
        await _accrue_interest_in_session(session=session, user_id=user_id, now=now)

        for _ in range(_REPAY_MAX_RETRIES):
            read_result = await session.execute(
                statement=select(
                    UserAccount.balance,
                    UserAccount.name,
                    UserAccount.loan_principal,
                    UserAccount.loan_interest,
                    UserAccount.loan_total_repaid,
                ).where(UserAccount.user_id == user_id)
            )
            row = read_result.one_or_none()
            if row is None:
                return None
            current_balance, existing_name, principal, interest, total_repaid = row
            debt_total = principal + interest
            if debt_total == 0 or current_balance == 0:
                return None

            effective = min(amount, current_balance, debt_total)
            interest_repaid = min(effective, interest)
            principal_repaid = effective - interest_repaid

            new_balance = current_balance - effective
            new_interest = interest - interest_repaid
            new_principal = principal - principal_repaid
            new_total_repaid = total_repaid + effective

            update_values: dict[str, Any] = {
                "balance": new_balance,
                "loan_interest": new_interest,
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
                    UserAccount.loan_interest == interest,
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
                debt_after=new_principal + new_interest,
                note=None,
                now=now,
            )
            await session.commit()
            return RepayResult(
                new_balance=new_balance,
                interest_repaid=interest_repaid,
                principal_repaid=principal_repaid,
                remaining_debt=new_principal + new_interest,
            )
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


async def get_loan_view(*, user_id: int) -> LoanView | None:
    """Returns a stored snapshot of the user's loan state.

    The interest value is the persisted (``loan_interest``) column. Callers
    that want the effective interest "as of now" should pass ``principal``
    + ``last_accrual_at`` through ``accrual_delta`` and add the result to
    ``interest_stored``.

    Args:
        user_id: Discord user ID to look up.

    Returns:
        ``LoanView`` for the user, or ``None`` if the user has never been
        seen by the economy DB.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(
                UserAccount.loan_principal,
                UserAccount.loan_interest,
                UserAccount.loan_last_accrual_at,
                UserAccount.loan_opened_at,
                UserAccount.loan_total_borrowed,
                UserAccount.loan_total_repaid,
            ).where(UserAccount.user_id == user_id)
        )
        row = result.one_or_none()
        if row is None:
            return None
        return LoanView(
            principal=row[0],
            interest_stored=row[1],
            last_accrual_at=row[2],
            opened_at=row[3],
            total_borrowed=row[4],
            total_repaid=row[5],
        )


async def transfer(  # noqa: PLR0913 -- transfer needs sender and receiver identity snapshots
    *,
    sender_id: int,
    sender_name: str,
    sender_avatar_url: str = "",
    receiver_id: int,
    receiver_name: str,
    receiver_avatar_url: str = "",
    amount: int,
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

    now = datetime.now(tz=UTC)
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
    *, limit: int = 10, exclude_user_ids: tuple[int, ...] = ()
) -> list[tuple[int, str, int]]:
    """Returns accounts ordered by balance descending.

    ``exclude_user_ids`` filters out specific accounts (notably the bot's
    own house ledger row) before applying the limit, so the leaderboard
    always shows real players. The ``ix_user_account_balance`` index keeps
    this query cheap even as the user table grows.

    Args:
        limit: Maximum number of accounts to return.
        exclude_user_ids: User IDs to filter out before applying the limit.

    Returns:
        `(user_id, name, balance)` tuples ordered by balance descending.
    """
    await _ensure_schema()
    async with open_session() as session:
        stmt = select(UserAccount.user_id, UserAccount.name, UserAccount.balance).order_by(
            desc(UserAccount.balance)
        )
        if exclude_user_ids:
            stmt = stmt.where(UserAccount.user_id.notin_(other=exclude_user_ids))
        stmt = stmt.limit(limit=limit)
        result = await session.execute(statement=stmt)
        return [(row[0], row[1], row[2]) for row in result.all()]
