"""Persistent point-balance store for the economy cog.

The engine is a module-level `AsyncEngine` singleton. Putting
`create_async_engine()` on a per-instance `cached_property` would leak the
connection pool, dialect cache, and inspector cache for every Discord
interaction (the same lesson `cogs/log_msg.py` captures for the sync engine
it still uses for pandas `to_sql`).

Every write path is an **atomic** SQL statement — either a SQLite UPSERT
(`INSERT ... ON CONFLICT DO UPDATE`) or a conditional
`UPDATE ... WHERE ... RETURNING`. The previous implementation read the row
in Python, mutated `account.balance`, and committed; two coroutines racing
on the same user would lose updates, and two coroutines racing on a
brand-new user would both `INSERT` and one would raise `IntegrityError`.
The UPSERT pattern fixes both. `place_bet` keeps a SELECT-then-UPDATE
retry loop because it needs to return the actual `effective_bet`, but the
retry is bounded and the WHERE clause guarantees no double-spend.

PRAGMA setup at connect-time enables WAL (so reads don't block on writes),
sets a tolerant `busy_timeout`, and picks `synchronous=NORMAL` (the right
durability trade-off in WAL: every commit fsyncs the WAL frame, and the
main file is fsynced on checkpoint).

We use `aiosqlite` so every DB call stays on the event loop: no
`asyncio.to_thread` shim, no separate `_*_sync` helpers. Each operation
opens an `AsyncSession` bound to the current `_engine`, so tests can
monkeypatch `_engine` per-test and every subsequent call sees the swap.
"""

from typing import Any, Final
from datetime import UTC, datetime
from dataclasses import dataclass

from sqlalchemy import Index, String, Integer, DateTime, case, desc, func, event, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.sql.dml import ReturningInsert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.dialects.sqlite import insert

# place_bet keeps a small retry budget for the SELECT-then-conditional-UPDATE
# loop. With WAL + busy_timeout, contention is rare and resolves on the
# first or second retry; the bound prevents a degenerate hot-row livelock.
_PLACE_BET_MAX_RETRIES: Final[int] = 8

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


class _Base(DeclarativeBase):
    """Declarative base for economy tables."""


class UserAccount(_Base):
    """Persistent point balance for a Discord user.

    Attributes:
        user_id: Discord user ID; primary key.
        name: Last-seen Discord username (refreshed on every write).
        balance: Current spendable point balance.
        total_earned: Lifetime points earned (chat rewards, game wins, transfers in).
        total_spent: Lifetime points removed (game losses, transfers out).
        updated_at: UTC timestamp of the last write.
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
    balance: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_earned: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_spent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        onupdate=lambda: datetime.now(tz=UTC),
    )


@dataclass(frozen=True)
class PlacedBet:
    """A successfully withdrawn wager.

    Attributes:
        amount: Actual amount withdrawn. This may be lower than the requested amount for all-in.
        balance_after: Account balance after the bet was withdrawn.
        is_allin: True when the requested bet was clamped to the available balance.
    """

    amount: int
    balance_after: int
    is_allin: bool


@dataclass(frozen=True)
class TransferResult:
    """A successful point transfer.

    Attributes:
        sender_balance: Sender balance after the debit.
        receiver_balance: Receiver balance after the credit.
    """

    sender_balance: int
    receiver_balance: int


async def _create_schema(engine: AsyncEngine) -> None:
    """Creates tables and indexes on first use; safe to call repeatedly."""
    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)


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
    await _create_schema(engine=_engine)
    _schema_ready_for = _engine


def open_session() -> AsyncSession:
    """Creates an async session bound to the current economy database engine.

    Returns:
        An `AsyncSession` using the current module-level `_engine`.
    """
    return AsyncSession(bind=_engine, expire_on_commit=False)


def _build_credit_upsert(
    *, user_id: int, name: str, amount: int, now: datetime
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
    return stmt.on_conflict_do_update(index_elements=["user_id"], set_=set_).returning(
        UserAccount.balance
    )


def _build_clamped_settle_upsert(
    *, user_id: int, name: str, delta: int, now: datetime
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
    return stmt.on_conflict_do_update(index_elements=["user_id"], set_=set_).returning(
        UserAccount.balance
    )


def _build_signed_delta_upsert(
    *, user_id: int, name: str, delta: int, now: datetime
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
    return stmt.on_conflict_do_update(index_elements=["user_id"], set_=set_).returning(
        UserAccount.balance
    )


async def add_balance(user_id: int, name: str, amount: int) -> int:
    """Adds points to the user balance.

    Non-positive amounts no-op and return the existing balance so callers
    can pass raw token counts (which can be 0 for cached responses) without
    a guard.

    Implemented as a single SQLite UPSERT, so two coroutines racing on the
    same user can neither lose updates nor crash with an `IntegrityError`
    on a brand-new account.

    Args:
        user_id: Discord user ID to credit.
        name: Last-seen Discord username to store on the account.
        amount: Number of points to add.

    Returns:
        The user's current balance after the operation.
    """
    await _ensure_schema()
    if amount <= 0:
        return await get_balance(user_id=user_id)
    stmt = _build_credit_upsert(
        user_id=user_id, name=name, amount=amount, now=datetime.now(tz=UTC)
    )
    async with open_session() as session:
        result = await session.execute(statement=stmt)
        await session.commit()
        return result.scalar_one()


async def place_bet(user_id: int, name: str, requested_bet: int) -> PlacedBet | None:
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

            update_values: dict[str, Any] = {
                "balance": UserAccount.balance - effective_bet,
                "total_spent": UserAccount.total_spent + effective_bet,
                "updated_at": datetime.now(tz=UTC),
            }
            if name and name != existing_name:
                update_values["name"] = name

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
            await session.commit()
            return PlacedBet(
                amount=effective_bet,
                balance_after=updated_row[0],
                is_allin=effective_bet < requested_bet,
            )
        return None


async def settle_game(user_id: int, name: str, delta: int) -> int:
    """Applies a signed player balance adjustment.

    Casino commands withdraw the bet up front with `place_bet()` and then pass
    the gross payout here when the round resolves. Losses are still clamped at
    zero so a stale caller never leaves a player in the red. Implemented as a
    single UPSERT, so concurrent settlements on the same user can't lose
    updates.

    Args:
        user_id: Discord user ID whose balance is adjusted.
        name: Last-seen Discord username to store on the account.
        delta: Signed point adjustment to apply.

    Returns:
        The user's current balance after settlement.
    """
    await _ensure_schema()
    stmt = _build_clamped_settle_upsert(
        user_id=user_id, name=name, delta=delta, now=datetime.now(tz=UTC)
    )
    async with open_session() as session:
        result = await session.execute(statement=stmt)
        await session.commit()
        return result.scalar_one()


async def house_settle(user_id: int, name: str, delta: int) -> int:
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

    Returns:
        The dealer ledger balance after settlement, which may be negative.
    """
    await _ensure_schema()
    stmt = _build_signed_delta_upsert(
        user_id=user_id, name=name, delta=delta, now=datetime.now(tz=UTC)
    )
    async with open_session() as session:
        result = await session.execute(statement=stmt)
        await session.commit()
        return result.scalar_one()


async def apply_round_settlement(  # noqa: PLR0913 -- atomic settlement needs both ledger keys
    *,
    player_id: int,
    player_account_name: str,
    payout: int,
    dealer_id: int,
    dealer_name: str,
    dealer_delta: int,
) -> tuple[int, int]:
    """Credits the player payout and mirrors the house P&L in one transaction.

    Sharing a session (and therefore a single SQLite transaction) means a
    crash between the two writes cannot leave the dealer ledger drifting
    from the player payout. ``payout`` must be non-negative (it's the gross
    amount to credit back after the upfront bet withdrawal). ``dealer_delta``
    is the signed change to apply to the dealer ledger.

    Args:
        player_id: Discord user ID for the player account.
        player_account_name: Account name to store for the player.
        payout: Gross amount to credit back to the player.
        dealer_id: Discord user ID for the dealer ledger row.
        dealer_name: Account name to store for the dealer ledger row.
        dealer_delta: Signed change to apply to the dealer ledger balance.

    Returns:
        A `(player_balance_after, dealer_balance_after)` tuple.
    """
    await _ensure_schema()
    now = datetime.now(tz=UTC)
    async with open_session() as session:
        if payout > 0:
            player_result = await session.execute(
                statement=_build_credit_upsert(
                    user_id=player_id, name=player_account_name, amount=payout, now=now
                )
            )
            player_balance = player_result.scalar_one()
        else:
            # Loss: place_bet already debited the player's row, so just read
            # the current balance. If the row somehow doesn't exist (shouldn't
            # happen because place_bet must have created it), treat as zero.
            read_result = await session.execute(
                statement=select(UserAccount.balance).where(UserAccount.user_id == player_id)
            )
            player_balance = read_result.scalar_one_or_none() or 0

        dealer_result = await session.execute(
            statement=_build_signed_delta_upsert(
                user_id=dealer_id, name=dealer_name, delta=dealer_delta, now=now
            )
        )
        dealer_balance = dealer_result.scalar_one()
        await session.commit()
        return player_balance, dealer_balance


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


async def transfer(
    *, sender_id: int, sender_name: str, receiver_id: int, receiver_name: str, amount: int
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
            user_id=receiver_id, name=receiver_name, amount=amount, now=now
        )
        credit_result = await session.execute(statement=credit_stmt)
        receiver_balance = credit_result.scalar_one()
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
