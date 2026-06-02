"""Persistent store for the fishing mini-game (`data/fishing.db`).

Holds each angler's equipped rod (one rod at a time, with durability), bait
counts, and an append-only catch log that backs the 魚簍, 圖鑑, and leaderboards.
Wallet money lives in the economy database, never here: rod/bait purchases burn
虛擬歡樂豆 through `apply_ordered_wallet_deltas` (a permanent sink) and fish sales
mint it back through `credit_with_repayment` (a faucet), tuned so the game is a
net sink. Casting touches only this database, so it is fully atomic; the only
cross-file steps are buy (economy debit then grant) and sell (mark sold then
credit), each with a compensating reversal because writes across two SQLite
files are not atomic together.

Schema is maintained offline (no migrations); money columns use `StoredInteger`
while physically bounded counters (durability, size, bait counts) use `Integer`
so SQL ordering and aggregation stay correct.
"""

from random import Random
from typing import Any, cast
import asyncio
from datetime import datetime
from collections.abc import Sequence

import logfire
from sqlalchemy import Index, String, Boolean, Integer, DateTime, text, event, select
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from discordbot.utils.timezone import database_now as _database_now
from discordbot.typings.economy import WalletDeltaLeg
from discordbot.typings.fishing import (
    ROD_BY_KEY,
    BAIT_BY_KEY,
    FISH_CATALOG,
    Rarity,
    RodTier,
    DexEntry,
    BuyResult,
    CastResult,
    SellResult,
    LoadoutView,
    InventoryEntry,
    BiggestCatchRow,
    FishingLeaderboardRow,
)
from discordbot.cogs._games.fishing import cast_fish
from discordbot.utils.sqlite_config import configure_sqlite_connection
from discordbot.utils.stored_integer import StoredInteger
from discordbot.cogs._economy.database import (
    get_balance,
    credit_with_repayment,
    apply_ordered_wallet_deltas,
)

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/fishing.db")
_schema_ready_for: AsyncEngine | None = None
_schema_lock: asyncio.Lock | None = None
_schema_lock_loop: asyncio.AbstractEventLoop | None = None

# A single biggest-catch leaderboard scans at most this many largest catches.
_LEADERBOARD_LIMIT = 10


class Base(DeclarativeBase):
    """Base class for fishing ORM models."""


class FishingAngler(Base):
    """Per-user equipped rod and lifetime fishing counters."""

    __tablename__ = "fishing_angler"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    rod_key: Mapped[str] = mapped_column(String(length=32), default="", nullable=False)
    rod_durability: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_casts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_sell_earned: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FishingBait(Base):
    """Per-user consumable bait counts keyed by bait type."""

    __tablename__ = "fishing_bait"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bait_key: Mapped[str] = mapped_column(String(length=32), primary_key=True)
    user_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FishingCatch(Base):
    """One caught fish; kept after sale so the 圖鑑 survives selling."""

    __tablename__ = "fishing_catch"
    __table_args__ = (
        Index("ix_fishing_catch_user_sold", "user_id", "sold"),
        Index("ix_fishing_catch_user_species", "user_id", "species_key"),
        Index("ix_fishing_catch_size", "size_mm"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    species_key: Mapped[str] = mapped_column(String(length=32), nullable=False)
    rarity: Mapped[str] = mapped_column(String(length=8), nullable=False)
    size_mm: Mapped[int] = mapped_column(Integer, nullable=False)
    sell_value: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    sold: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    caught_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sold_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def _configure_sqlite_connection(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Configures SQLite for fishing storage."""
    configure_sqlite_connection(dbapi_connection=dbapi_connection)


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Configures SQLite for fishing storage."""
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


def _current_schema_lock() -> asyncio.Lock:
    """Returns a schema lock bound to the current event loop."""
    global _schema_lock, _schema_lock_loop  # noqa: PLW0603 -- loop-local singleton
    loop = asyncio.get_running_loop()
    if _schema_lock is None or _schema_lock_loop is not loop:
        _schema_lock = asyncio.Lock()
        _schema_lock_loop = loop
    return _schema_lock


def open_fishing_session() -> AsyncSession:
    """Creates an async session bound to the current fishing database engine."""
    _ensure_sqlite_hooks(engine=_engine)
    return AsyncSession(bind=_engine, expire_on_commit=False)


async def _begin_immediate(session: AsyncSession) -> None:
    """Acquires SQLite's write lock before reading state for a mutation."""
    await session.execute(statement=text("BEGIN IMMEDIATE"))


async def _ensure_schema() -> None:
    """Bootstraps fishing schema once per engine."""
    global _schema_ready_for  # noqa: PLW0603 -- module-level cache by engine identity
    _ensure_sqlite_hooks(engine=_engine)
    if _schema_ready_for is _engine:
        return
    async with _current_schema_lock():
        if _schema_ready_for is _engine:
            return
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _schema_ready_for = _engine


def _set_rod(*, angler: FishingAngler, rod: RodTier, user_name: str, now: datetime) -> None:
    """Equips a fresh rod at full durability on an existing angler row."""
    angler.rod_key = rod.key
    angler.rod_durability = rod.durability
    angler.user_name = user_name
    angler.updated_at = now


async def _get_or_create_angler(
    *, session: AsyncSession, user_id: int, user_name: str, now: datetime
) -> FishingAngler:
    """Returns the angler row for a user, creating an empty one when missing."""
    angler = await session.get(FishingAngler, user_id)
    if angler is None:
        angler = FishingAngler(
            user_id=user_id,
            user_name=user_name,
            rod_key="",
            rod_durability=0,
            total_casts=0,
            total_sell_earned=0,
            created_at=now,
            updated_at=now,
        )
        session.add(angler)
    return angler


async def get_loadout(user_id: int) -> LoadoutView:
    """Returns the angler's wallet balance, equipped rod, and bait counts."""
    await _ensure_schema()
    balance = await get_balance(user_id=user_id)
    async with open_fishing_session() as session:
        angler = await session.get(FishingAngler, user_id)
        bait_rows = (
            (await session.execute(select(FishingBait).where(FishingBait.user_id == user_id)))
            .scalars()
            .all()
        )
    baits = {row.bait_key: row.quantity for row in bait_rows if row.quantity > 0}
    if angler is None:
        return LoadoutView(
            user_id=user_id, balance=balance, rod_key="", rod_durability=0, baits=baits
        )
    return LoadoutView(
        user_id=user_id,
        balance=balance,
        rod_key=angler.rod_key,
        rod_durability=angler.rod_durability,
        baits=baits,
        total_casts=angler.total_casts,
    )


async def list_inventory(user_id: int, limit: int = 25) -> tuple[InventoryEntry, ...]:
    """Returns the angler's unsold catches, newest first."""
    await _ensure_schema()
    async with open_fishing_session() as session:
        rows = (
            (
                await session.execute(
                    select(FishingCatch)
                    .where(FishingCatch.user_id == user_id, FishingCatch.sold.is_(False))
                    .order_by(FishingCatch.id.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
    return tuple(
        InventoryEntry(
            catch_id=row.id,
            species_key=row.species_key,
            rarity=_as_rarity(row.rarity),
            size_mm=row.size_mm,
            sell_value=row.sell_value,
        )
        for row in rows
    )


async def get_dex(user_id: int) -> tuple[DexEntry, ...]:
    """Returns one dex entry per catalog species with caught counts and records."""
    await _ensure_schema()
    async with open_fishing_session() as session:
        rows = (
            await session.execute(
                select(FishingCatch.species_key, FishingCatch.size_mm).where(
                    FishingCatch.user_id == user_id
                )
            )
        ).all()
    counts: dict[str, int] = {}
    biggest: dict[str, int] = {}
    for species_key, size_mm in rows:
        counts[species_key] = counts.get(species_key, 0) + 1
        biggest[species_key] = max(biggest.get(species_key, 0), size_mm)
    return tuple(
        DexEntry(
            species_key=species.key,
            caught=species.key in counts,
            count=counts.get(species.key, 0),
            biggest_mm=biggest.get(species.key, 0),
        )
        for species in FISH_CATALOG
    )


async def buy_rod(user_id: int, user_name: str, rod_key: str, avatar_url: str = "") -> BuyResult:
    """Debits the rod cost as a sink, then equips a fresh rod at full durability."""
    rod = ROD_BY_KEY.get(rod_key)
    if rod is None:
        return BuyResult(status="unknown_item")
    await _ensure_schema()
    wallet = await apply_ordered_wallet_deltas(
        user_id=user_id,
        name=user_name,
        deltas=[WalletDeltaLeg(delta=-rod.cost, reason=f"fishing_rod:{rod_key}")],
        avatar_url=avatar_url,
    )
    if wallet is None:
        return BuyResult(
            status="insufficient", cost=rod.cost, new_balance=await get_balance(user_id=user_id)
        )
    now = _database_now()
    try:
        async with open_fishing_session() as session:
            await _begin_immediate(session)
            angler = await _get_or_create_angler(
                session=session, user_id=user_id, user_name=user_name, now=now
            )
            _set_rod(angler=angler, rod=rod, user_name=user_name, now=now)
            await session.commit()
    except Exception:
        logfire.exception("Fishing rod grant failed; refunding", user_id=user_id, rod_key=rod_key)
        await credit_with_repayment(
            user_id=user_id, name=user_name, amount=rod.cost, avatar_url=avatar_url
        )
        raise
    return BuyResult(status="ok", cost=rod.cost, new_balance=wallet.new_balance)


async def buy_bait(
    user_id: int, user_name: str, bait_key: str, quantity: int, avatar_url: str = ""
) -> BuyResult:
    """Debits the bait cost as a sink, then adds the bait to the angler's stock."""
    bait = BAIT_BY_KEY.get(bait_key)
    if bait is None:
        return BuyResult(status="unknown_item")
    if quantity <= 0:
        return BuyResult(status="invalid_quantity")
    await _ensure_schema()
    cost = bait.cost * quantity
    wallet = await apply_ordered_wallet_deltas(
        user_id=user_id,
        name=user_name,
        deltas=[WalletDeltaLeg(delta=-cost, reason=f"fishing_bait:{bait_key}x{quantity}")],
        avatar_url=avatar_url,
    )
    if wallet is None:
        return BuyResult(
            status="insufficient", cost=cost, new_balance=await get_balance(user_id=user_id)
        )
    now = _database_now()
    try:
        async with open_fishing_session() as session:
            await _begin_immediate(session)
            await _add_bait(
                session=session,
                user_id=user_id,
                user_name=user_name,
                bait_key=bait_key,
                quantity=quantity,
                now=now,
            )
            await session.commit()
    except Exception:
        logfire.exception(
            "Fishing bait grant failed; refunding", user_id=user_id, bait_key=bait_key
        )
        await credit_with_repayment(
            user_id=user_id, name=user_name, amount=cost, avatar_url=avatar_url
        )
        raise
    return BuyResult(status="ok", cost=cost, new_balance=wallet.new_balance)


async def _add_bait(  # noqa: PLR0913 -- ordered grant helper carries identity, item, and timestamp
    *,
    session: AsyncSession,
    user_id: int,
    user_name: str,
    bait_key: str,
    quantity: int,
    now: datetime,
) -> None:
    """Adds `quantity` bait of one type to the angler, creating the row when missing."""
    row = await session.get(FishingBait, (user_id, bait_key))
    if row is None:
        session.add(
            FishingBait(
                user_id=user_id,
                bait_key=bait_key,
                user_name=user_name,
                quantity=quantity,
                updated_at=now,
            )
        )
        return
    row.quantity += quantity
    row.user_name = user_name
    row.updated_at = now


async def execute_cast(user_id: int, user_name: str, bait_key: str, rng: Random) -> CastResult:
    """Consumes one bait and one durability, rolls a cast, and logs any catch.

    Touches only the fishing database, so the whole operation is atomic. No
    wallet money moves here; money only moves at buy (sink) and sell (faucet).
    """
    bait = BAIT_BY_KEY.get(bait_key)
    if bait is None:
        return CastResult(status="no_bait")
    await _ensure_schema()
    now = _database_now()
    async with open_fishing_session() as session:
        await _begin_immediate(session)
        angler = await session.get(FishingAngler, user_id)
        if angler is None or not angler.rod_key or angler.rod_durability <= 0:
            return CastResult(status="no_rod")
        rod = ROD_BY_KEY.get(angler.rod_key)
        if rod is None:
            return CastResult(status="no_rod")
        bait_row = await session.get(FishingBait, (user_id, bait_key))
        if bait_row is None or bait_row.quantity <= 0:
            return CastResult(status="no_bait")
        bait_row.quantity -= 1
        bait_row.user_name = user_name
        bait_row.updated_at = now
        angler.rod_durability -= 1
        angler.total_casts += 1
        angler.user_name = user_name
        rod_broke = angler.rod_durability <= 0
        if rod_broke:
            angler.rod_key = ""
        angler.updated_at = now
        outcome = cast_fish(rng=rng, rod=rod, bait=bait)
        catch_id: int | None = None
        if not outcome.miss and outcome.species is not None:
            catch = FishingCatch(
                user_id=user_id,
                user_name=user_name,
                species_key=outcome.species.key,
                rarity=outcome.species.rarity,
                size_mm=outcome.size_mm,
                sell_value=outcome.sell_value,
                sold=False,
                caught_at=now,
                sold_at=None,
            )
            session.add(catch)
            await session.flush()
            catch_id = catch.id
        durability_after = angler.rod_durability if not rod_broke else 0
        bait_remaining = bait_row.quantity
        await session.commit()
    return CastResult(
        status="ok",
        outcome=outcome,
        rod_key="" if rod_broke else rod.key,
        rod_durability_after=durability_after,
        rod_broke=rod_broke,
        bait_key=bait_key,
        bait_remaining=bait_remaining,
        catch_id=catch_id,
    )


async def sell_fish(
    user_id: int, user_name: str, catch_ids: Sequence[int] | None = None, avatar_url: str = ""
) -> SellResult:
    """Marks caught fish sold and credits their total value (faucet).

    Fishing rows are committed sold first, then the wallet is credited; on a
    credit failure the rows are reverted to unsold so a hard error never leaves
    the angler paid-but-still-holding or sold-but-unpaid in the common case.
    """
    await _ensure_schema()
    now = _database_now()
    total = 0
    sold_ids: list[int] = []
    async with open_fishing_session() as session:
        await _begin_immediate(session)
        statement = select(FishingCatch).where(
            FishingCatch.user_id == user_id, FishingCatch.sold.is_(False)
        )
        if catch_ids is not None:
            statement = statement.where(FishingCatch.id.in_(catch_ids))
        rows = (await session.execute(statement)).scalars().all()
        if rows:
            total = sum(row.sell_value for row in rows)
            for row in rows:
                row.sold = True
                row.sold_at = now
            angler = await session.get(FishingAngler, user_id)
            if angler is not None:
                angler.total_sell_earned += total
                angler.user_name = user_name
                angler.updated_at = now
            sold_ids = [row.id for row in rows]
            await session.commit()
    if not sold_ids:
        return SellResult(status="nothing", new_balance=await get_balance(user_id=user_id))
    try:
        credit = await credit_with_repayment(
            user_id=user_id, name=user_name, amount=total, avatar_url=avatar_url
        )
    except Exception:
        logfire.exception("Fishing sale credit failed; reverting sold rows", user_id=user_id)
        await _revert_sold(user_id=user_id, catch_ids=sold_ids, total=total)
        raise
    return SellResult(
        status="ok", sold_count=len(sold_ids), earned=total, new_balance=credit.new_balance
    )


async def _revert_sold(*, user_id: int, catch_ids: Sequence[int], total: int) -> None:
    """Compensating un-sell when the post-sale wallet credit fails."""
    now = _database_now()
    async with open_fishing_session() as session:
        await _begin_immediate(session)
        rows = (
            (await session.execute(select(FishingCatch).where(FishingCatch.id.in_(catch_ids))))
            .scalars()
            .all()
        )
        for row in rows:
            row.sold = False
            row.sold_at = None
        angler = await session.get(FishingAngler, user_id)
        if angler is not None:
            angler.total_sell_earned -= total
            angler.updated_at = now
        await session.commit()


async def leaderboard_total_earned(
    limit: int = _LEADERBOARD_LIMIT,
) -> tuple[FishingLeaderboardRow, ...]:
    """Returns the top anglers by lifetime fish-sale earnings."""
    await _ensure_schema()
    async with open_fishing_session() as session:
        rows = (
            (
                await session.execute(
                    select(FishingAngler).where(FishingAngler.total_sell_earned > 0)
                )
            )
            .scalars()
            .all()
        )
    ranked = sorted(rows, key=lambda row: row.total_sell_earned, reverse=True)[:limit]
    return tuple(
        FishingLeaderboardRow(
            user_id=row.user_id, user_name=row.user_name, value=row.total_sell_earned
        )
        for row in ranked
    )


async def leaderboard_biggest_catch(
    limit: int = _LEADERBOARD_LIMIT,
) -> tuple[BiggestCatchRow, ...]:
    """Returns the largest individual catches across all anglers."""
    await _ensure_schema()
    async with open_fishing_session() as session:
        rows = (
            (
                await session.execute(
                    select(FishingCatch).order_by(FishingCatch.size_mm.desc()).limit(limit)
                )
            )
            .scalars()
            .all()
        )
    return tuple(
        BiggestCatchRow(
            user_id=row.user_id,
            user_name=row.user_name,
            species_key=row.species_key,
            rarity=_as_rarity(row.rarity),
            size_mm=row.size_mm,
        )
        for row in rows
    )


# --- Offline maintenance helpers (scripts/manage_fishing.py) -----------------


async def grant_rod(user_id: int, user_name: str, rod_key: str) -> bool:
    """Equips a rod at full durability without charging (offline maintenance)."""
    rod = ROD_BY_KEY.get(rod_key)
    if rod is None:
        return False
    await _ensure_schema()
    now = _database_now()
    async with open_fishing_session() as session:
        await _begin_immediate(session)
        angler = await _get_or_create_angler(
            session=session, user_id=user_id, user_name=user_name, now=now
        )
        _set_rod(angler=angler, rod=rod, user_name=user_name, now=now)
        await session.commit()
    return True


async def grant_bait(user_id: int, user_name: str, bait_key: str, quantity: int) -> bool:
    """Adds bait without charging (offline maintenance)."""
    if BAIT_BY_KEY.get(bait_key) is None or quantity <= 0:
        return False
    await _ensure_schema()
    now = _database_now()
    async with open_fishing_session() as session:
        await _begin_immediate(session)
        await _add_bait(
            session=session,
            user_id=user_id,
            user_name=user_name,
            bait_key=bait_key,
            quantity=quantity,
            now=now,
        )
        await session.commit()
    return True


async def reset_user(user_id: int) -> None:
    """Deletes all fishing state for one user (offline maintenance)."""
    await _ensure_schema()
    async with open_fishing_session() as session:
        await _begin_immediate(session)
        angler = await session.get(FishingAngler, user_id)
        if angler is not None:
            await session.delete(angler)
        bait_rows = (
            (await session.execute(select(FishingBait).where(FishingBait.user_id == user_id)))
            .scalars()
            .all()
        )
        for row in bait_rows:
            await session.delete(row)
        catch_rows = (
            (await session.execute(select(FishingCatch).where(FishingCatch.user_id == user_id)))
            .scalars()
            .all()
        )
        for row in catch_rows:
            await session.delete(row)
        await session.commit()


def _as_rarity(value: str) -> Rarity:
    """Narrows a stored rarity string to the Rarity literal (defaults to N)."""
    if value in ("N", "R", "SR", "SSR", "UR"):
        return cast("Rarity", value)
    return "N"
