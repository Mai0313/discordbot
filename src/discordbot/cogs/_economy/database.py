"""Persistent point-balance store for the economy cog.

The engine is a module-level singleton — putting `create_engine()` on a
per-instance `cached_property` would leak the connection pool, dialect cache,
and inspector cache for every Discord interaction (mirrors the same lesson
captured in `cogs/log_msg.py`).

`check_same_thread=False` is required because the sync ORM calls run inside
`asyncio.to_thread`, which dispatches them to whichever worker thread is free;
SQLAlchemy's connection pool then guarantees one connection per checkout, so
the SQLite no-cross-thread rule is still respected at the connection level.
"""

from typing import Final
import asyncio
from datetime import UTC, datetime

from sqlalchemy import Engine, String, Integer, DateTime, desc, select, create_engine
from sqlalchemy.orm import Mapped, Session, DeclarativeBase, mapped_column

_DB_URL: Final[str] = "sqlite:///data/economy.db"
_engine: Engine = create_engine(url=_DB_URL, connect_args={"check_same_thread": False})


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


_Base.metadata.create_all(bind=_engine)


def _get_or_create(session: Session, user_id: int, name: str) -> UserAccount:
    """Returns the account row for ``user_id``, creating it on first sight.

    Refreshes the cached display name when Discord shows us a new value, so
    leaderboards stay readable even after username changes.
    """
    account = session.get(entity=UserAccount, ident=user_id)
    if account is None:
        account = UserAccount(user_id=user_id, name=name or str(user_id))
        session.add(instance=account)
        session.flush()
        return account
    if name and account.name != name:
        account.name = name
    return account


def _add_balance_sync(user_id: int, name: str, amount: int) -> int:
    """Adds ``amount`` to the user balance; returns the new balance.

    Non-positive amounts no-op and return the existing balance so callers can
    pass raw token counts (which can be 0 for cached responses) without a guard.
    """
    with Session(bind=_engine) as session:
        account = _get_or_create(session=session, user_id=user_id, name=name)
        if amount > 0:
            account.balance += amount
            account.total_earned += amount
            session.commit()
        return account.balance


def _settle_game_sync(user_id: int, name: str, delta: int) -> int:
    """Applies a signed game outcome (positive = win, negative = loss).

    The bet itself is *not* withdrawn upfront by callers — instead they pass
    the net delta after the round resolves: ``+bet`` on a win, ``-bet`` on a
    loss, ``0`` on push, ``+round(bet * 0.5)`` on Blackjack on top of the win.
    Loss is clamped at zero balance so a stale session never leaves a player
    in the red.
    """
    with Session(bind=_engine) as session:
        account = _get_or_create(session=session, user_id=user_id, name=name)
        new_balance = max(account.balance + delta, 0)
        applied_delta = new_balance - account.balance
        account.balance = new_balance
        if applied_delta > 0:
            account.total_earned += applied_delta
        elif applied_delta < 0:
            account.total_spent += -applied_delta
        session.commit()
        return account.balance


def _get_balance_sync(user_id: int) -> int:
    """Returns the current balance, or 0 if the user has never been seen."""
    with Session(bind=_engine) as session:
        account = session.get(entity=UserAccount, ident=user_id)
        if account is None:
            return 0
        return account.balance


def _transfer_sync(
    sender_id: int, sender_name: str, receiver_id: int, receiver_name: str, amount: int
) -> bool:
    """Atomically moves ``amount`` from sender to receiver.

    Returns False (and rolls back) when the sender is the receiver, the amount
    is non-positive, or the sender lacks funds. Both rows are upserted in the
    same transaction so a crash mid-transfer can't double-credit or vanish
    points.
    """
    if amount <= 0 or sender_id == receiver_id:
        return False
    with Session(bind=_engine) as session:
        sender = _get_or_create(session=session, user_id=sender_id, name=sender_name)
        if sender.balance < amount:
            return False
        receiver = _get_or_create(session=session, user_id=receiver_id, name=receiver_name)
        sender.balance -= amount
        sender.total_spent += amount
        receiver.balance += amount
        receiver.total_earned += amount
        session.commit()
        return True


def _top_n_sync(limit: int) -> list[tuple[int, str, int]]:
    """Returns up to ``limit`` accounts ordered by balance descending."""
    with Session(bind=_engine) as session:
        stmt = select(UserAccount).order_by(desc(UserAccount.balance)).limit(limit=limit)
        rows = session.execute(statement=stmt).scalars()
        return [(row.user_id, row.name, row.balance) for row in rows]


async def add_balance(*, user_id: int, name: str, amount: int) -> int:
    """Async wrapper: adds ``amount`` to the user balance; returns new balance."""
    return await asyncio.to_thread(_add_balance_sync, user_id, name, amount)


async def settle_game(*, user_id: int, name: str, delta: int) -> int:
    """Async wrapper: applies a signed game outcome; returns the new balance."""
    return await asyncio.to_thread(_settle_game_sync, user_id, name, delta)


async def get_balance(*, user_id: int) -> int:
    """Async wrapper: reads the current balance (0 if missing)."""
    return await asyncio.to_thread(_get_balance_sync, user_id)


async def transfer(
    *, sender_id: int, sender_name: str, receiver_id: int, receiver_name: str, amount: int
) -> bool:
    """Async wrapper: atomically transfers ``amount`` between two users."""
    return await asyncio.to_thread(
        _transfer_sync, sender_id, sender_name, receiver_id, receiver_name, amount
    )


async def top_n(*, limit: int = 10) -> list[tuple[int, str, int]]:
    """Async wrapper: returns the top ``limit`` accounts by balance."""
    return await asyncio.to_thread(_top_n_sync, limit)
