"""Persistent store and settlement service for the fishing mini-game.

State lives in `data/database/games.db` (shared file, fishing-owned engine and
tables); wallet cash stays in the economy database. Catalog rows (grades,
species, gear) are the source of truth and are seeded offline; runtime never
seeds them.

Two operations cross databases. A purchase debits (burns) the wallet first, then
grants gear in games.db, refunding on a grant failure. A cast consumes bait and
durability and logs the catch in games.db first, then credits the payout in the
economy database; a payout that fails after the catch is logged is reported as
deferred rather than rolled back, which only ever deflates further. Hard crashes
between the two file commits are an accepted non-atomicity.
"""

from random import Random, SystemRandom
from typing import Any, Final
import asyncio
from datetime import datetime
from contextlib import AbstractAsyncContextManager

import logfire
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
    delete,
    select,
)
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from discordbot.utils.timezone import database_now as _database_now
from discordbot.typings.economy import WalletDeltaLeg
from discordbot.typings.fishing import (
    MAX_BAIT_PER_PURCHASE,
    FISHING_MAX_SINGLE_CATCH,
    GearType,
    GearView,
    CatchRoll,
    FishGrade,
    CastResult,
    CastStatus,
    GearUpsert,
    CatchLogView,
    BaitStackView,
    PurchaseResult,
    AnglerStateView,
    FishSpeciesView,
    FishingPanelData,
    FishSpeciesUpsert,
    FishGradeConfigView,
    FishGradeConfigUpsert,
)
from discordbot.cogs._fishing.catch import roll_catch
from discordbot.utils.asyncio_locks import LoopLocalLock, KeyedLockManager
from discordbot.utils.sqlite_config import ensure_sqlite_hooks, configure_sqlite_connection
from discordbot.utils.stored_integer import StoredInteger, stored_int_to_text
from discordbot.cogs._economy.database import (
    get_balance,
    credit_with_repayment,
    apply_ordered_wallet_deltas,
)

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/games.db")
_schema_ready_for: AsyncEngine | None = None
_schema_lock = LoopLocalLock()
_angler_locks = KeyedLockManager[int]()
_PRODUCTION_RNG: Final[SystemRandom] = SystemRandom()


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


def _current_schema_lock() -> asyncio.Lock:
    """Returns the schema bootstrap lock bound to the current event loop."""
    return _schema_lock.get()


def _angler_lock(user_id: int) -> AbstractAsyncContextManager[None]:
    """Returns a per-user fishing mutation lock bound to the current event loop."""
    return _angler_locks.hold(key=user_id)


def open_fishing_session() -> AsyncSession:
    """Creates an async session bound to the current fishing database engine."""
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    return AsyncSession(bind=_engine, expire_on_commit=False)


async def _begin_immediate(session: AsyncSession) -> None:
    """Acquires SQLite's write lock before reading state for a mutation plan."""
    await session.execute(statement=text("BEGIN IMMEDIATE"))


async def _ensure_schema() -> None:
    """Bootstraps the fishing schema once per engine."""
    global _schema_ready_for  # noqa: PLW0603 -- module-level cache by engine identity
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    if _schema_ready_for is _engine:
        return
    async with _current_schema_lock():
        if _schema_ready_for is _engine:
            return
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _schema_ready_for = _engine


def _stored_integer_desc_order(column: Any) -> tuple[Any, ...]:  # noqa: ANN401 -- SQLAlchemy columns are generic expressions
    """Returns ORDER BY terms for descending numeric order over decimal text."""
    sign = func.discordbot_int_compare_text(column, stored_int_to_text(value=0))
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


class Base(DeclarativeBase):
    """Base class for fishing ORM models."""

    pass


class FishGradeConfig(Base):
    """Tunable per-grade roll weight and display metadata. One row per grade."""

    __tablename__ = "fish_grade_config"

    grade: Mapped[str] = mapped_column(String(length=8), primary_key=True)
    weight: Mapped[int] = mapped_column(Integer, nullable=False)
    color: Mapped[int] = mapped_column(Integer, nullable=False)
    emoji: Mapped[str] = mapped_column(String(length=32), nullable=False)
    label: Mapped[str] = mapped_column(String(length=32), nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FishSpecies(Base):
    """Tunable fish species catalog row."""

    __tablename__ = "fish_species"
    __table_args__ = (Index("ix_fish_species_grade", "grade"),)

    species_id: Mapped[str] = mapped_column(String(length=32), primary_key=True)
    name: Mapped[str] = mapped_column(String(length=64), nullable=False)
    grade: Mapped[str] = mapped_column(String(length=8), nullable=False)
    emoji: Mapped[str] = mapped_column(String(length=32), nullable=False)
    intra_grade_weight: Mapped[int] = mapped_column(Integer, nullable=False)
    base_value: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    size_min_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    size_max_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    image_key: Mapped[str] = mapped_column(String(length=64), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FishingGear(Base):
    """Tunable rod and bait catalog row, discriminated by `gear_type`."""

    __tablename__ = "fishing_gear"
    __table_args__ = (Index("ix_fishing_gear_type_tier", "gear_type", "tier"),)

    gear_id: Mapped[str] = mapped_column(String(length=32), primary_key=True)
    gear_type: Mapped[str] = mapped_column(String(length=8), nullable=False)
    name: Mapped[str] = mapped_column(String(length=64), nullable=False)
    emoji: Mapped[str] = mapped_column(String(length=32), nullable=False)
    tier: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    price: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    rarity_shift_bps: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    durability: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    value_bonus_bps: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AnglerState(Base):
    """Per-user rod, durability, and lifetime fishing stats. One row per user."""

    __tablename__ = "angler_state"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    rod_id: Mapped[str] = mapped_column(String(length=32), default="", nullable=False)
    durability_remaining: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_casts: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_catch_value: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    total_spent_on_gear: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    best_catch_value: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class BaitInventory(Base):
    """Per-user bait counts keyed by (user_id, bait_id)."""

    __tablename__ = "bait_inventory"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bait_id: Mapped[str] = mapped_column(String(length=32), primary_key=True)
    user_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    quantity: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class CatchLog(Base):
    """Append-only catch record powering the leaderboard and history."""

    __tablename__ = "catch_log"
    __table_args__ = (
        Index("ix_catch_log_value", "value"),
        Index("ix_catch_log_user_created", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    species_id: Mapped[str] = mapped_column(String(length=32), nullable=False)
    species_name: Mapped[str] = mapped_column(String(length=64), nullable=False)
    grade: Mapped[str] = mapped_column(String(length=8), nullable=False)
    emoji: Mapped[str] = mapped_column(String(length=32), nullable=False)
    size_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    base_value: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    value: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    rod_id: Mapped[str] = mapped_column(String(length=32), default="", nullable=False)
    bait_id: Mapped[str] = mapped_column(String(length=32), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _grade_view(row: FishGradeConfig) -> FishGradeConfigView:
    """Projects an ORM grade config into a typed view."""
    return FishGradeConfigView(
        grade=FishGrade(row.grade),
        weight=row.weight,
        color=row.color,
        emoji=row.emoji,
        label=row.label,
        order_index=row.order_index,
    )


def _species_view(row: FishSpecies) -> FishSpeciesView:
    """Projects an ORM species row into a typed view."""
    return FishSpeciesView(
        species_id=row.species_id,
        name=row.name,
        grade=FishGrade(row.grade),
        emoji=row.emoji,
        intra_grade_weight=row.intra_grade_weight,
        base_value=row.base_value,
        size_min_bps=row.size_min_bps,
        size_max_bps=row.size_max_bps,
        image_key=row.image_key,
    )


def _gear_view(row: FishingGear) -> GearView:
    """Projects an ORM gear row into a typed view."""
    return GearView(
        gear_id=row.gear_id,
        gear_type=GearType(row.gear_type),
        name=row.name,
        emoji=row.emoji,
        tier=row.tier,
        price=row.price,
        rarity_shift_bps=row.rarity_shift_bps,
        durability=row.durability,
        value_bonus_bps=row.value_bonus_bps,
    )


def _angler_view(
    angler: AnglerState | None, user_id: int, rod: GearView | None
) -> AnglerStateView:
    """Projects an ORM angler row into a typed view, defaulting to an empty angler."""
    if angler is None:
        return AnglerStateView(user_id=user_id)
    return AnglerStateView(
        user_id=angler.user_id,
        user_name=angler.user_name,
        rod=rod,
        durability_remaining=angler.durability_remaining,
        total_casts=angler.total_casts,
        total_catch_value=angler.total_catch_value,
        total_spent_on_gear=angler.total_spent_on_gear,
        best_catch_value=angler.best_catch_value,
    )


def _catch_log_view(row: CatchLog) -> CatchLogView:
    """Projects an ORM catch row into a typed view."""
    return CatchLogView(
        user_id=row.user_id,
        user_name=row.user_name or str(row.user_id),
        species_id=row.species_id,
        species_name=row.species_name,
        grade=FishGrade(row.grade),
        emoji=row.emoji,
        size_bps=row.size_bps,
        value=row.value,
        created_at=row.created_at,
    )


async def _load_gear_map(session: AsyncSession) -> dict[str, FishingGear]:
    """Loads every gear row keyed by id for one session."""
    result = await session.execute(statement=select(FishingGear))
    return {row.gear_id: row for row in result.scalars()}


async def list_grade_configs() -> tuple[FishGradeConfigView, ...]:
    """Lists grade configs ordered by rarity rank."""
    await _ensure_schema()
    async with open_fishing_session() as session:
        result = await session.execute(
            statement=select(FishGradeConfig).order_by(FishGradeConfig.order_index.asc())
        )
        return tuple(_grade_view(row=row) for row in result.scalars())


async def get_grade_config_map() -> dict[FishGrade, FishGradeConfigView]:
    """Returns grade configs keyed by grade for display lookups."""
    return {config.grade: config for config in await list_grade_configs()}


async def list_fish_species() -> tuple[FishSpeciesView, ...]:
    """Lists fish species ordered by grade then identifier."""
    await _ensure_schema()
    async with open_fishing_session() as session:
        result = await session.execute(
            statement=select(FishSpecies).order_by(
                FishSpecies.grade.asc(), FishSpecies.species_id.asc()
            )
        )
        return tuple(_species_view(row=row) for row in result.scalars())


async def list_gear() -> tuple[GearView, ...]:
    """Lists all gear ordered by type then tier."""
    await _ensure_schema()
    async with open_fishing_session() as session:
        result = await session.execute(
            statement=select(FishingGear).order_by(
                FishingGear.gear_type.asc(), FishingGear.tier.asc(), FishingGear.gear_id.asc()
            )
        )
        return tuple(_gear_view(row=row) for row in result.scalars())


async def upsert_grade_config(
    config: FishGradeConfigUpsert, now: datetime | None = None
) -> FishGradeConfigView:
    """Creates or updates one grade config from a maintenance payload."""
    await _ensure_schema()
    effective_now = now or _database_now()
    async with open_fishing_session() as session:
        existing = await session.get(entity=FishGradeConfig, ident=config.grade.value)
        if existing is None:
            existing = FishGradeConfig(
                grade=config.grade.value,
                weight=config.weight,
                color=config.color,
                emoji=config.emoji,
                label=config.label,
                order_index=config.order_index,
                updated_at=effective_now,
            )
            session.add(instance=existing)
        else:
            existing.weight = config.weight
            existing.color = config.color
            existing.emoji = config.emoji
            existing.label = config.label
            existing.order_index = config.order_index
            existing.updated_at = effective_now
        await session.commit()
        return _grade_view(row=existing)


async def upsert_fish_species(
    species: FishSpeciesUpsert, now: datetime | None = None
) -> FishSpeciesView:
    """Creates or updates one fish species from a maintenance payload."""
    await _ensure_schema()
    effective_now = now or _database_now()
    async with open_fishing_session() as session:
        existing = await session.get(entity=FishSpecies, ident=species.species_id)
        if existing is None:
            existing = FishSpecies(
                species_id=species.species_id,
                name=species.name,
                grade=species.grade.value,
                emoji=species.emoji,
                intra_grade_weight=species.intra_grade_weight,
                base_value=species.base_value,
                size_min_bps=species.size_min_bps,
                size_max_bps=species.size_max_bps,
                image_key=species.image_key,
                created_at=effective_now,
                updated_at=effective_now,
            )
            session.add(instance=existing)
        else:
            existing.name = species.name
            existing.grade = species.grade.value
            existing.emoji = species.emoji
            existing.intra_grade_weight = species.intra_grade_weight
            existing.base_value = species.base_value
            existing.size_min_bps = species.size_min_bps
            existing.size_max_bps = species.size_max_bps
            existing.image_key = species.image_key
            existing.updated_at = effective_now
        await session.commit()
        return _species_view(row=existing)


async def upsert_gear(gear: GearUpsert, now: datetime | None = None) -> GearView:
    """Creates or updates one gear item from a maintenance payload."""
    await _ensure_schema()
    effective_now = now or _database_now()
    async with open_fishing_session() as session:
        existing = await session.get(entity=FishingGear, ident=gear.gear_id)
        if existing is None:
            existing = FishingGear(
                gear_id=gear.gear_id,
                gear_type=gear.gear_type.value,
                name=gear.name,
                emoji=gear.emoji,
                tier=gear.tier,
                price=gear.price,
                rarity_shift_bps=gear.rarity_shift_bps,
                durability=gear.durability,
                value_bonus_bps=gear.value_bonus_bps,
                created_at=effective_now,
                updated_at=effective_now,
            )
            session.add(instance=existing)
        else:
            existing.gear_type = gear.gear_type.value
            existing.name = gear.name
            existing.emoji = gear.emoji
            existing.tier = gear.tier
            existing.price = gear.price
            existing.rarity_shift_bps = gear.rarity_shift_bps
            existing.durability = gear.durability
            existing.value_bonus_bps = gear.value_bonus_bps
            existing.updated_at = effective_now
        await session.commit()
        return _gear_view(row=existing)


async def get_angler_state(user_id: int) -> AnglerStateView:
    """Returns the angler's rod and lifetime fishing state."""
    await _ensure_schema()
    async with open_fishing_session() as session:
        angler = await session.get(entity=AnglerState, ident=user_id)
        rod: GearView | None = None
        if angler is not None and angler.rod_id:
            rod_row = await session.get(entity=FishingGear, ident=angler.rod_id)
            rod = _gear_view(row=rod_row) if rod_row is not None else None
        return _angler_view(angler=angler, user_id=user_id, rod=rod)


async def _latest_catch_view(session: AsyncSession, user_id: int) -> CatchLogView | None:
    """Returns the angler's most recent catch, if any."""
    result = await session.execute(
        statement=select(CatchLog)
        .where(CatchLog.user_id == user_id)
        .order_by(CatchLog.created_at.desc(), CatchLog.id.desc())
        .limit(1)
    )
    row = result.scalars().first()
    return _catch_log_view(row=row) if row is not None else None


async def get_fishing_panel(user_id: int) -> FishingPanelData:
    """Aggregates balance, angler state, owned bait, and last catch for the panel."""
    await _ensure_schema()
    balance = await get_balance(user_id=user_id)
    async with open_fishing_session() as session:
        angler_row = await session.get(entity=AnglerState, ident=user_id)
        gear_map = await _load_gear_map(session=session)
        rod: GearView | None = None
        if angler_row is not None and angler_row.rod_id and angler_row.rod_id in gear_map:
            rod = _gear_view(row=gear_map[angler_row.rod_id])
        angler = _angler_view(angler=angler_row, user_id=user_id, rod=rod)
        bait_result = await session.execute(
            statement=select(BaitInventory).where(BaitInventory.user_id == user_id)
        )
        baits: list[BaitStackView] = []
        for bait_row in bait_result.scalars():
            if bait_row.quantity <= 0:
                continue
            gear = gear_map.get(bait_row.bait_id)
            baits.append(
                BaitStackView(
                    bait_id=bait_row.bait_id,
                    name=gear.name if gear is not None else bait_row.bait_id,
                    emoji=gear.emoji if gear is not None else "🎣",
                    quantity=bait_row.quantity,
                )
            )
        baits.sort(
            key=lambda stack: (
                gear_map[stack.bait_id].tier if stack.bait_id in gear_map else 99,
                stack.bait_id,
            )
        )
        last_catch = await _latest_catch_view(session=session, user_id=user_id)
    return FishingPanelData(
        balance=balance, angler=angler, baits=tuple(baits), last_catch=last_catch
    )


async def _get_or_create_angler_in_session(
    session: AsyncSession, user_id: int, name: str, now: datetime
) -> AnglerState:
    """Loads the angler row, creating an empty one when absent."""
    angler = await session.get(entity=AnglerState, ident=user_id)
    if angler is None:
        # Set every column explicitly: SQLAlchemy `default=` only applies at INSERT
        # flush time, so the StoredInteger/Integer fields would read as None until
        # then and break the in-place arithmetic below.
        angler = AnglerState(
            user_id=user_id,
            user_name=name,
            rod_id="",
            durability_remaining=0,
            total_casts=0,
            total_catch_value=0,
            total_spent_on_gear=0,
            best_catch_value=0,
            updated_at=now,
        )
        session.add(instance=angler)
    return angler


async def _grant_gear_in_session(  # noqa: PLR0913 -- gear grant needs identity, gear, quantity, cost, and time
    session: AsyncSession,
    user_id: int,
    name: str,
    gear: GearView,
    quantity: int,
    total_cost: int,
    now: datetime,
) -> None:
    """Grants a purchased rod or bait and bumps the angler's lifetime gear spend."""
    angler = await _get_or_create_angler_in_session(
        session=session, user_id=user_id, name=name, now=now
    )
    angler.user_name = name
    angler.total_spent_on_gear = angler.total_spent_on_gear + total_cost
    angler.updated_at = now
    if gear.gear_type == GearType.ROD:
        angler.rod_id = gear.gear_id
        angler.durability_remaining = gear.durability
        return
    bait = await session.get(entity=BaitInventory, ident=(user_id, gear.gear_id))
    if bait is None:
        bait = BaitInventory(
            user_id=user_id,
            bait_id=gear.gear_id,
            user_name=name,
            quantity=quantity,
            updated_at=now,
        )
        session.add(instance=bait)
        return
    bait.quantity = bait.quantity + quantity
    bait.user_name = name
    bait.updated_at = now


async def purchase_gear(
    user_id: int, name: str, gear_id: str, quantity: int = 1, avatar_url: str = ""
) -> PurchaseResult:
    """Buys a rod or bait, burning the wallet first then granting in games.db.

    Rods are bought exactly one at a time and replace any current rod. Bait stacks
    by the requested quantity, which must be positive and within
    `MAX_BAIT_PER_PURCHASE`. Quantity semantics are enforced here so the economy
    invariant never depends on the view layer. A grant failure after the wallet
    debit triggers a best-effort refund so the player is not charged for nothing.
    """
    await _ensure_schema()
    async with open_fishing_session() as session:
        gear_row = await session.get(entity=FishingGear, ident=gear_id)
        gear = _gear_view(row=gear_row) if gear_row is not None else None
    if gear is None:
        return PurchaseResult(success=False, gear_id=gear_id, reason="unknown_gear")
    if quantity < 1:
        return PurchaseResult(
            success=False, gear_id=gear_id, gear_type=gear.gear_type, reason="invalid_quantity"
        )
    if gear.gear_type == GearType.BAIT:
        if quantity > MAX_BAIT_PER_PURCHASE:
            return PurchaseResult(
                success=False, gear_id=gear_id, gear_type=gear.gear_type, reason="invalid_quantity"
            )
        units = quantity
    else:
        units = 1
    total_cost = gear.price * units
    wallet = await apply_ordered_wallet_deltas(
        user_id=user_id,
        name=name,
        deltas=[WalletDeltaLeg(delta=-total_cost, reason=f"fishing:buy:{gear_id}")],
        avatar_url=avatar_url,
    )
    if wallet is None:
        return PurchaseResult(
            success=False,
            gear_id=gear_id,
            gear_type=gear.gear_type,
            total_cost=total_cost,
            new_balance=await get_balance(user_id=user_id),
            reason="insufficient",
        )
    try:
        async with _angler_lock(user_id=user_id), open_fishing_session() as session:
            await _begin_immediate(session=session)
            await _grant_gear_in_session(
                session=session,
                user_id=user_id,
                name=name,
                gear=gear,
                quantity=units,
                total_cost=total_cost,
                now=_database_now(),
            )
            await session.commit()
    # Broad on purpose: any games.db failure after the economy.db debit must reach
    # the refund below rather than propagate and leave the player charged.
    except Exception as exc:
        logfire.warn(
            "Fishing gear grant failed after wallet debit; refunding",
            user_id=user_id,
            gear_id=gear_id,
            amount=total_cost,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
        refund = await apply_ordered_wallet_deltas(
            user_id=user_id,
            name=name,
            deltas=[WalletDeltaLeg(delta=total_cost, reason=f"fishing:buy_refund:{gear_id}")],
            avatar_url=avatar_url,
        )
        if refund is None:
            logfire.error(
                "Fishing purchase refund failed; manual repair needed",
                user_id=user_id,
                gear_id=gear_id,
                amount=total_cost,
            )
        return PurchaseResult(
            success=False,
            gear_id=gear_id,
            gear_type=gear.gear_type,
            total_cost=total_cost,
            new_balance=await get_balance(user_id=user_id),
            reason="grant_failed",
        )
    return PurchaseResult(
        success=True,
        gear_id=gear_id,
        gear_type=gear.gear_type,
        quantity=units,
        total_cost=total_cost,
        new_balance=wallet.new_balance,
    )


def _build_cast_log(  # noqa: PLR0913 -- one catch row needs identity, roll, gear ids, and time
    user_id: int, name: str, roll: CatchRoll, rod_id: str, bait_id: str, now: datetime
) -> CatchLog:
    """Builds the persisted catch row for one successful cast."""
    return CatchLog(
        user_id=user_id,
        user_name=name,
        species_id=roll.species_id,
        species_name=roll.species_name,
        grade=roll.grade.value,
        emoji=roll.emoji,
        size_bps=roll.size_bps,
        base_value=roll.base_value,
        value=roll.value,
        rod_id=rod_id,
        bait_id=bait_id,
        created_at=now,
    )


async def settle_cast(  # noqa: PLR0913 -- a cast needs identity, bait, avatar, rng, and time
    user_id: int,
    name: str,
    bait_id: str,
    avatar_url: str = "",
    rng: Random | None = None,
    now: datetime | None = None,
) -> CastResult:
    """Consumes bait and durability, rolls a catch, then credits the payout.

    The bait, durability, and catch log commit to games.db first; the payout is
    then credited to the economy wallet. A payout credit that fails after the
    catch is logged returns `PAYOUT_DEFERRED` rather than rolling back.
    """
    await _ensure_schema()
    effective_rng = rng or _PRODUCTION_RNG
    effective_now = now or _database_now()
    async with _angler_lock(user_id=user_id), open_fishing_session() as session:
        await _begin_immediate(session=session)
        angler = await session.get(entity=AnglerState, ident=user_id)
        if angler is None or not angler.rod_id:
            return CastResult(status=CastStatus.NO_ROD)
        if angler.durability_remaining <= 0:
            return CastResult(status=CastStatus.BROKEN_ROD)
        rod_row = await session.get(entity=FishingGear, ident=angler.rod_id)
        if rod_row is None:
            return CastResult(status=CastStatus.NO_ROD)
        bait_row = await session.get(entity=FishingGear, ident=bait_id)
        if bait_row is None or GearType(bait_row.gear_type) != GearType.BAIT:
            return CastResult(status=CastStatus.NO_BAIT)
        bait_inv = await session.get(entity=BaitInventory, ident=(user_id, bait_id))
        if bait_inv is None or bait_inv.quantity <= 0:
            return CastResult(status=CastStatus.NO_BAIT)
        grade_configs = [
            _grade_view(row=row)
            for row in (await session.execute(statement=select(FishGradeConfig))).scalars()
        ]
        species = [
            _species_view(row=row)
            for row in (await session.execute(statement=select(FishSpecies))).scalars()
        ]
        roll = roll_catch(
            rng=effective_rng,
            grade_configs=grade_configs,
            species=species,
            rod=_gear_view(row=rod_row),
            bait=_gear_view(row=bait_row),
            max_value=FISHING_MAX_SINGLE_CATCH,
        )
        bait_inv.quantity = bait_inv.quantity - 1
        bait_remaining = bait_inv.quantity
        bait_inv.updated_at = effective_now
        angler.durability_remaining = angler.durability_remaining - 1
        durability_remaining = angler.durability_remaining
        # Keep rod_id set on a break so the broken rod stays visible until the
        # player buys a replacement; the next cast then hits the BROKEN_ROD guard.
        rod_broke = durability_remaining <= 0
        angler.total_casts = angler.total_casts + 1
        angler.total_catch_value = angler.total_catch_value + roll.value
        angler.best_catch_value = max(angler.best_catch_value, roll.value)
        angler.user_name = name
        angler.updated_at = effective_now
        session.add(
            instance=_build_cast_log(
                user_id=user_id,
                name=name,
                roll=roll,
                rod_id=rod_row.gear_id,
                bait_id=bait_id,
                now=effective_now,
            )
        )
        await session.commit()
    try:
        credit = await credit_with_repayment(
            user_id=user_id, name=name, amount=roll.value, avatar_url=avatar_url
        )
        new_balance = credit.new_balance
        status = CastStatus.SUCCESS
    # Broad on purpose: the catch is already committed to games.db, so any economy-side
    # failure must still return a CastResult instead of dropping the player's cast UI.
    # Nothing retries this payout, so the log line is the only repair record.
    except Exception as exc:
        logfire.error(
            "Fishing payout credit failed after catch logged; manual repair needed",
            user_id=user_id,
            amount=roll.value,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
        new_balance = await get_balance(user_id=user_id)
        status = CastStatus.PAYOUT_DEFERRED
    return CastResult(
        status=status,
        roll=roll,
        payout=roll.value,
        new_balance=new_balance,
        rod_broke=rod_broke,
        durability_remaining=durability_remaining,
        bait_id=bait_id,
        bait_remaining=bait_remaining,
    )


async def fetch_top_catches(limit: int = 10) -> tuple[CatchLogView, ...]:
    """Returns the highest-value single catches across all anglers."""
    await _ensure_schema()
    async with open_fishing_session() as session:
        result = await session.execute(
            statement=select(CatchLog)
            .order_by(
                *_stored_integer_desc_order(column=CatchLog.value), CatchLog.created_at.desc()
            )
            .limit(limit)
        )
        return tuple(_catch_log_view(row=row) for row in result.scalars())


async def fetch_recent_catches(user_id: int, limit: int = 10) -> tuple[CatchLogView, ...]:
    """Returns one angler's most recent catches, newest first."""
    await _ensure_schema()
    async with open_fishing_session() as session:
        result = await session.execute(
            statement=select(CatchLog)
            .where(CatchLog.user_id == user_id)
            .order_by(CatchLog.created_at.desc(), CatchLog.id.desc())
            .limit(limit)
        )
        return tuple(_catch_log_view(row=row) for row in result.scalars())


async def reset_all_fishing() -> int:
    """Clears all per-user fishing state, leaving the tunable catalog intact.

    Used by the offline economy reset so stale rods, bait, and catch history do
    not survive a wallet deflation. Grade, species, and gear catalog rows are
    intentionally left untouched.

    Returns:
        The number of angler rows cleared.
    """
    await _ensure_schema()
    async with open_fishing_session() as session:
        angler_count = await session.scalar(
            statement=select(func.count()).select_from(AnglerState)
        )
        await session.execute(statement=delete(CatchLog))
        await session.execute(statement=delete(BaitInventory))
        await session.execute(statement=delete(AnglerState))
        await session.commit()
        return int(angler_count or 0)


__all__ = [
    "AnglerState",
    "BaitInventory",
    "Base",
    "CatchLog",
    "FishGradeConfig",
    "FishSpecies",
    "FishingGear",
    "fetch_recent_catches",
    "fetch_top_catches",
    "get_angler_state",
    "get_fishing_panel",
    "get_grade_config_map",
    "list_fish_species",
    "list_gear",
    "list_grade_configs",
    "open_fishing_session",
    "purchase_gear",
    "reset_all_fishing",
    "settle_cast",
    "upsert_fish_species",
    "upsert_gear",
    "upsert_grade_config",
]
