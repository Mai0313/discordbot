"""Persistent point-balance store for the economy cog.

The engine is a module-level `AsyncEngine` singleton. Putting
`create_async_engine()` on a per-instance `cached_property` would leak the
connection pool, dialect cache, and inspector cache for every Discord
interaction (the same lesson `cogs/log_msg.py` captures for the sync engine
it still uses for pandas `to_sql`).

We use `aiosqlite` so every DB call stays on the event loop: no
`asyncio.to_thread` shim, no separate `_*_sync` helpers. Each operation
opens an `AsyncSession` bound to the current `_engine`, so tests can
monkeypatch `_engine` per-test and every subsequent call sees the swap.
"""

from typing import Final
import asyncio
from datetime import UTC, datetime
from dataclasses import dataclass

from sqlalchemy import String, Integer, DateTime, desc, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

_DB_URL: Final[str] = "sqlite+aiosqlite:///data/economy.db"
_engine: AsyncEngine = create_async_engine(url=_DB_URL)


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


async def _create_schema(engine: AsyncEngine) -> None:
    """Creates tables on first import; safe to call repeatedly."""
    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)


# Bootstrap the schema synchronously at import time so the first user-facing
# command doesn't race against table creation. asyncio.run() is safe here
# because nothing else is on the loop yet (this module is imported before
# the bot connects to Discord).
asyncio.run(main=_create_schema(engine=_engine))


def open_session() -> AsyncSession:
    """Creates an async session bound to the current economy database engine.

    Returns:
        An `AsyncSession` using the current module-level `_engine`.
    """
    return AsyncSession(bind=_engine, expire_on_commit=False)


async def _get_or_create(session: AsyncSession, user_id: int, name: str) -> UserAccount:
    """Returns the account row for ``user_id``, creating it on first sight.

    Refreshes the cached display name when Discord shows us a new value, so
    leaderboards stay readable even after username changes.
    """
    account = await session.get(entity=UserAccount, ident=user_id)
    if account is None:
        account = UserAccount(user_id=user_id, name=name or str(user_id))
        session.add(instance=account)
        await session.flush()
        return account
    if name and account.name != name:
        account.name = name
    return account


def _apply_balance_delta(account: UserAccount, delta: int, *, allow_negative: bool = False) -> int:
    """Applies a signed balance delta and updates gross lifetime totals."""
    starting_balance = account.balance
    account.balance = (
        account.balance + delta if allow_negative else max(account.balance + delta, 0)
    )
    applied_delta = account.balance - starting_balance
    if applied_delta > 0:
        account.total_earned += applied_delta
    elif applied_delta < 0:
        account.total_spent += -applied_delta
    return account.balance


async def add_balance(user_id: int, name: str, amount: int) -> int:
    """Adds points to the user balance.

    Non-positive amounts no-op and return the existing balance so callers
    can pass raw token counts (which can be 0 for cached responses) without
    a guard.

    Args:
        user_id: Discord user ID to credit.
        name: Last-seen Discord username to store on the account.
        amount: Number of points to add.

    Returns:
        The user's current balance after the operation.
    """
    async with open_session() as session:
        account = await _get_or_create(session=session, user_id=user_id, name=name)
        if amount > 0:
            _apply_balance_delta(account=account, delta=amount)
            await session.commit()
        return account.balance


async def place_bet(user_id: int, name: str, requested_bet: int) -> PlacedBet | None:
    """Atomically withdraws a wager.

    Bets larger than the current balance are clamped to the full available
    balance (auto all-in). The conditional update protects against stale
    balance reads from concurrent game commands, so the same points cannot be
    spent twice.

    Args:
        user_id: Discord user ID placing the wager.
        name: Last-seen Discord username to store on the account.
        requested_bet: Requested wager amount in points.

    Returns:
        The withdrawn wager details, or `None` when the user has no spendable
        balance or the requested bet is not positive.
    """
    if requested_bet <= 0:
        return None

    async with open_session() as session:
        while True:
            account = await session.get(entity=UserAccount, ident=user_id)
            if account is None or account.balance <= 0:
                return None

            starting_balance = account.balance
            effective_bet = min(requested_bet, starting_balance)
            stmt = (
                update(UserAccount)
                .where(UserAccount.user_id == user_id, UserAccount.balance == starting_balance)
                .values(
                    name=name or account.name,
                    balance=UserAccount.balance - effective_bet,
                    total_spent=UserAccount.total_spent + effective_bet,
                    updated_at=datetime.now(tz=UTC),
                )
                .returning(UserAccount.balance)
            )
            result = await session.execute(statement=stmt)
            row = result.one_or_none()
            if row is None:
                await session.rollback()
                continue

            await session.commit()
            return PlacedBet(
                amount=effective_bet, balance_after=row[0], is_allin=effective_bet < requested_bet
            )


async def settle_game(user_id: int, name: str, delta: int) -> int:
    """Applies a signed player balance adjustment.

    Casino commands withdraw the bet up front with `place_bet()` and then pass
    the gross payout here when the round resolves. Losses are still clamped at
    zero so a stale caller never leaves a player in the red.

    Args:
        user_id: Discord user ID whose balance is adjusted.
        name: Last-seen Discord username to store on the account.
        delta: Signed point adjustment to apply.

    Returns:
        The user's current balance after settlement.
    """
    async with open_session() as session:
        account = await _get_or_create(session=session, user_id=user_id, name=name)
        _apply_balance_delta(account=account, delta=delta)
        await session.commit()
        return account.balance


async def house_settle(user_id: int, name: str, delta: int) -> int:
    """Records a dealer-side settlement.

    Used to track the bot's casino P&L over time. The dealer has effectively
    infinite funds (it backs every bet), so unlike `settle_game` we
    deliberately do not clamp at zero. A long-running losing streak
    should surface as a negative balance, with `total_earned` /
    `total_spent` accumulating gross flows in each direction.

    Args:
        user_id: Discord user ID for the dealer ledger row.
        name: Last-seen display name to store on the ledger row.
        delta: Signed point adjustment to apply.

    Returns:
        The dealer ledger balance after settlement, which may be negative.
    """
    async with open_session() as session:
        account = await _get_or_create(session=session, user_id=user_id, name=name)
        _apply_balance_delta(account=account, delta=delta, allow_negative=True)
        await session.commit()
        return account.balance


async def get_balance(user_id: int) -> int:
    """Returns the current balance for a user.

    Args:
        user_id: Discord user ID to look up.

    Returns:
        The current balance, or 0 if the user has never been seen.
    """
    async with open_session() as session:
        account = await session.get(entity=UserAccount, ident=user_id)
        if account is None:
            return 0
        return account.balance


async def get_account(user_id: int) -> tuple[str, int, int, int] | None:
    """Returns the stored account snapshot for a user.

    Args:
        user_id: Discord user ID to look up.

    Returns:
        A `(name, balance, total_earned, total_spent)` tuple, or `None` if the
        user has never been seen.
    """
    async with open_session() as session:
        account = await session.get(entity=UserAccount, ident=user_id)
        if account is None:
            return None
        return (account.name, account.balance, account.total_earned, account.total_spent)


async def transfer(
    *, sender_id: int, sender_name: str, receiver_id: int, receiver_name: str, amount: int
) -> bool:
    """Atomically moves points from sender to receiver.

    Returns False (and rolls back) when the sender is the receiver, the
    amount is non-positive, or the sender lacks funds. Both rows are
    upserted in the same transaction so a crash mid-transfer can't
    double-credit or vanish points.

    Args:
        sender_id: Discord user ID to debit.
        sender_name: Last-seen Discord username to store on the sender account.
        receiver_id: Discord user ID to credit.
        receiver_name: Last-seen Discord username to store on the receiver account.
        amount: Number of points to transfer.

    Returns:
        True when the transfer committed, or False when validation failed or
        the sender had insufficient funds.
    """
    if amount <= 0 or sender_id == receiver_id:
        return False
    async with open_session() as session:
        while True:
            sender = await _get_or_create(session=session, user_id=sender_id, name=sender_name)
            if sender.balance < amount:
                return False

            starting_balance = sender.balance
            debit_stmt = (
                update(UserAccount)
                .where(UserAccount.user_id == sender_id, UserAccount.balance == starting_balance)
                .values(
                    name=sender_name or sender.name,
                    balance=UserAccount.balance - amount,
                    total_spent=UserAccount.total_spent + amount,
                    updated_at=datetime.now(tz=UTC),
                )
                .returning(UserAccount.balance)
            )
            result = await session.execute(statement=debit_stmt)
            if result.one_or_none() is None:
                await session.rollback()
                continue

            receiver = await _get_or_create(
                session=session, user_id=receiver_id, name=receiver_name
            )
            credit_stmt = (
                update(UserAccount)
                .where(UserAccount.user_id == receiver_id)
                .values(
                    name=receiver_name or receiver.name,
                    balance=UserAccount.balance + amount,
                    total_earned=UserAccount.total_earned + amount,
                    updated_at=datetime.now(tz=UTC),
                )
            )
            await session.execute(statement=credit_stmt)
            await session.commit()
            return True


async def top_n(
    *, limit: int = 10, exclude_user_ids: tuple[int, ...] = ()
) -> list[tuple[int, str, int]]:
    """Returns accounts ordered by balance descending.

    ``exclude_user_ids`` filters out specific accounts (notably the bot's
    own house ledger row) before applying the limit, so the leaderboard
    always shows real players.

    Args:
        limit: Maximum number of accounts to return.
        exclude_user_ids: User IDs to filter out before applying the limit.

    Returns:
        `(user_id, name, balance)` tuples ordered by balance descending.
    """
    async with open_session() as session:
        stmt = select(UserAccount).order_by(desc(UserAccount.balance))
        if exclude_user_ids:
            stmt = stmt.where(UserAccount.user_id.notin_(other=exclude_user_ids))
        stmt = stmt.limit(limit=limit)
        result = await session.execute(statement=stmt)
        rows = result.scalars()
        return [(row.user_id, row.name, row.balance) for row in rows]
