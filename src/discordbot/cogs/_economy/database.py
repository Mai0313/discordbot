"""Persistent point-balance store for the economy cog.

The engine is a module-level `AsyncEngine` singleton — putting
`create_async_engine()` on a per-instance `cached_property` would leak the
connection pool, dialect cache, and inspector cache for every Discord
interaction (the same lesson `cogs/log_msg.py` captures for the sync engine
it still uses for pandas `to_sql`).

We use `aiosqlite` so every DB call stays on the event loop — no
`asyncio.to_thread` shim, no separate `_*_sync` helpers. Each operation
opens an `AsyncSession` bound to the current `_engine`, so tests can
monkeypatch `_engine` per-test and every subsequent call sees the swap.
"""

from typing import Final
import asyncio
from datetime import UTC, datetime

from sqlalchemy import String, Integer, DateTime, desc, select
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


async def _create_schema(engine: AsyncEngine) -> None:
    """Creates tables on first import; safe to call repeatedly."""
    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)


# Bootstrap the schema synchronously at import time so the first user-facing
# command doesn't race against table creation. asyncio.run() is safe here
# because nothing else is on the loop yet (this module is imported before
# the bot connects to Discord).
asyncio.run(main=_create_schema(engine=_engine))


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


async def add_balance(*, user_id: int, name: str, amount: int) -> int:
    """Adds ``amount`` to the user balance; returns the new balance.

    Non-positive amounts no-op and return the existing balance so callers
    can pass raw token counts (which can be 0 for cached responses) without
    a guard.
    """
    async with AsyncSession(bind=_engine, expire_on_commit=False) as session:
        account = await _get_or_create(session=session, user_id=user_id, name=name)
        if amount > 0:
            account.balance += amount
            account.total_earned += amount
            await session.commit()
        return account.balance


async def settle_game(*, user_id: int, name: str, delta: int) -> int:
    """Applies a signed game outcome (positive = win, negative = loss).

    The bet is *not* withdrawn upfront by callers — instead they pass the
    net delta after the round resolves: ``+bet`` on a win, ``-bet`` on a
    loss, ``0`` on push, ``+round(bet * 0.5)`` on Blackjack on top of the
    win. Loss is clamped at zero so a stale session never leaves a player
    in the red.
    """
    async with AsyncSession(bind=_engine, expire_on_commit=False) as session:
        account = await _get_or_create(session=session, user_id=user_id, name=name)
        new_balance = max(account.balance + delta, 0)
        applied_delta = new_balance - account.balance
        account.balance = new_balance
        if applied_delta > 0:
            account.total_earned += applied_delta
        elif applied_delta < 0:
            account.total_spent += -applied_delta
        await session.commit()
        return account.balance


async def house_settle(*, user_id: int, name: str, delta: int) -> int:
    """Records a dealer-side settlement; balance may go negative.

    Used to track the bot's casino P&L over time. The dealer has effectively
    infinite funds (it backs every bet), so unlike `settle_game` we
    deliberately do *not* clamp at zero — a long-running losing streak
    should surface as a negative balance, with `total_earned` /
    `total_spent` accumulating gross flows in each direction.
    """
    async with AsyncSession(bind=_engine, expire_on_commit=False) as session:
        account = await _get_or_create(session=session, user_id=user_id, name=name)
        account.balance += delta
        if delta > 0:
            account.total_earned += delta
        elif delta < 0:
            account.total_spent += -delta
        await session.commit()
        return account.balance


async def get_balance(*, user_id: int) -> int:
    """Returns the current balance, or 0 if the user has never been seen."""
    async with AsyncSession(bind=_engine, expire_on_commit=False) as session:
        account = await session.get(entity=UserAccount, ident=user_id)
        if account is None:
            return 0
        return account.balance


async def get_account(*, user_id: int) -> tuple[str, int, int, int] | None:
    """Returns ``(name, balance, total_earned, total_spent)`` or ``None`` if unseen."""
    async with AsyncSession(bind=_engine, expire_on_commit=False) as session:
        account = await session.get(entity=UserAccount, ident=user_id)
        if account is None:
            return None
        return (account.name, account.balance, account.total_earned, account.total_spent)


async def transfer(
    *, sender_id: int, sender_name: str, receiver_id: int, receiver_name: str, amount: int
) -> bool:
    """Atomically moves ``amount`` from sender to receiver.

    Returns False (and rolls back) when the sender is the receiver, the
    amount is non-positive, or the sender lacks funds. Both rows are
    upserted in the same transaction so a crash mid-transfer can't
    double-credit or vanish points.
    """
    if amount <= 0 or sender_id == receiver_id:
        return False
    async with AsyncSession(bind=_engine, expire_on_commit=False) as session:
        sender = await _get_or_create(session=session, user_id=sender_id, name=sender_name)
        if sender.balance < amount:
            return False
        receiver = await _get_or_create(session=session, user_id=receiver_id, name=receiver_name)
        sender.balance -= amount
        sender.total_spent += amount
        receiver.balance += amount
        receiver.total_earned += amount
        await session.commit()
        return True


async def top_n(
    *, limit: int = 10, exclude_user_ids: tuple[int, ...] = ()
) -> list[tuple[int, str, int]]:
    """Returns up to ``limit`` accounts ordered by balance descending.

    ``exclude_user_ids`` filters out specific accounts (notably the bot's
    own house ledger row) before applying the limit, so the leaderboard
    always shows real players.
    """
    async with AsyncSession(bind=_engine, expire_on_commit=False) as session:
        stmt = select(UserAccount).order_by(desc(UserAccount.balance))
        if exclude_user_ids:
            stmt = stmt.where(UserAccount.user_id.notin_(other=exclude_user_ids))
        stmt = stmt.limit(limit=limit)
        result = await session.execute(statement=stmt)
        rows = result.scalars()
        return [(row.user_id, row.name, row.balance) for row in rows]
