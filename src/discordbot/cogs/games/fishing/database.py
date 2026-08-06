"""Persistent store and settlement service for the fishing mini-game.

State lives in `data/database/games.db` — a file `cogs/games/database.py` and the message-cleanup
cog also open, each through an engine of its own, which is why the schema here is created lazily
by `_ensure_schema` rather than by a shared migration step. This module owns six tables in it: the
three tunable catalog ones (`fish_grade_config`, `fish_species`, `fishing_gear`), the two per-user
ones (`angler_state`, `bait_inventory`), and the append-only `catch_log` the leaderboard and the
history read. No ORM row leaves the module: every value handed back is one of the frozen views in
`typings/fishing.py`.

This is the cog's whole storage and settlement half. `views.py` renders and never splits a read
and a write of its own, `catch.py` holds the pure roll rules this module feeds catalog rows to and
calls, and wallet cash stays in `services/economy/database.py`, which this module spends and
collects through rather than keeping money of its own. Catalog rows are the tuning source of truth
and are written offline (`defaults.py` -> `scripts/seed_fishing.py` -> the `upsert_*` calls);
runtime never seeds them, so an unseeded database has no gear to sell and nobody who can cast.

Two operations cross databases and neither is atomic across the pair. A purchase debits (burns)
the wallet first, then grants gear in games.db, refunding on a grant failure. A cast consumes bait
and durability and logs the catch in games.db first, then credits the payout in the economy
database; a payout that fails after the catch is logged is reported as deferred rather than rolled
back, which only ever deflates further. Hard crashes between the two file commits are an accepted
non-atomicity.

Inside games.db a mutation is serialized twice over. `_angler_lock` keeps one angler's purchases
and casts from interleaving within this process, and `_begin_immediate` takes SQLite's write lock
before the reads a mutation plan is built from, so another writer on the same file cannot change
the row between that read and the write.
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
from discordbot.utils.asyncio_locks import LoopLocalLock, KeyedLockManager
from discordbot.utils.sqlite_config import ensure_sqlite_hooks, configure_sqlite_connection
from discordbot.utils.stored_integer import StoredInteger, stored_int_to_text
from discordbot.cogs.games.fishing.catch import roll_catch
from discordbot.services.economy.database import (
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
    """Applies the shared PRAGMA setup to one fishing SQLite connection.

    Takes the shared defaults whole: no foreign keys, since no table here declares one, and the
    `StoredInteger` UDFs registered, since `fetch_top_catches` orders by one of them by name.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
    """
    configure_sqlite_connection(dbapi_connection=dbapi_connection)


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """SQLAlchemy `connect` listener: configures each connection the pool opens.

    Registered at import time against the engine that existed then, which is why
    `ensure_sqlite_hooks` re-installs it on whatever `_engine` currently is.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
        _connection_record (Any): SQLAlchemy's pool record, unused.
    """
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _configure_sqlite_on_checkout(
    dbapi_connection: object, _connection_record: object, _connection_proxy: object
) -> None:
    """SQLAlchemy `checkout` listener: configures a pooled connection on its way out.

    The `connect` hook alone misses connections a test-swapped engine had already pooled before
    the listener was installed; `ensure_sqlite_hooks`'s docstring has the full reasoning.

    Args:
        dbapi_connection (object): The pooled DBAPI connection being handed out.
        _connection_record (object): SQLAlchemy's pool record, unused.
        _connection_proxy (object): SQLAlchemy's connection proxy, unused.
    """
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _current_schema_lock() -> asyncio.Lock:
    """Returns the schema bootstrap lock bound to the current event loop.

    Returns:
        The `asyncio.Lock` guarding `create_all` for the loop now running.
    """
    return _schema_lock.get()


def _angler_lock(user_id: int) -> AbstractAsyncContextManager[None]:
    """Serializes one angler's mutations, on the loop now running.

    A purchase and a cast each read the angler row and then write it, so two of them interleaving
    would lose a bait or a durability point. Keyed per user, so two anglers never wait on each
    other, and not reentrant — no path here takes it twice.

    Args:
        user_id (int): The angler whose mutations serialize.

    Returns:
        An async context manager holding that angler's lock for the duration of its body.
    """
    return _angler_locks.hold(key=user_id)


def open_fishing_session() -> AsyncSession:
    """Opens an async session on the current fishing engine, re-installing its PRAGMA listeners.

    The listeners are re-installed on every open (idempotently) because a test-swapped engine
    carries none of its own, and an unconfigured connection would run without WAL and without the
    `StoredInteger` UDFs the leaderboard's ORDER BY calls.

    Returns:
        A session bound to `_engine` with `expire_on_commit=False`, so a view built out of a
        just-committed row is still readable.
    """
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    return AsyncSession(bind=_engine, expire_on_commit=False)


async def _begin_immediate(session: AsyncSession) -> None:
    """Takes SQLite's write lock before the reads a mutation plan is built from.

    SQLite's default deferred transaction upgrades to a write lock only at the first write, by
    which point the plan has already read state another writer may have changed underneath it.

    Args:
        session (AsyncSession): The session whose transaction to start immediately.
    """
    await session.execute(statement=text("BEGIN IMMEDIATE"))


async def _ensure_schema() -> None:
    """Creates the fishing tables once per engine.

    Every public entry point awaits this first, so there is no migration step and a fresh
    `games.db` grows the tables on first use. The flag holds the engine rather than a bool, so a
    test that swaps `_engine` bootstraps again against the new file, and the double check around
    the lock keeps the settled case lock-free. The connection listeners are re-installed on every
    call for the same swapped-engine reason.
    """
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
    """Returns ORDER BY terms for descending numeric order over decimal text.

    A `StoredInteger` column sorts lexicographically in SQL, where "9" beats "10", and the
    registered UDF only yields a sign for one pair of values. So the numeric order is rebuilt out
    of five terms: sign first, then for positives longer text (more digits) before shorter and
    lexicographic within a length, and for negatives the mirror image. Doing it in SQL rather than
    in Python is what lets `fetch_top_catches` apply its `LIMIT` before any row is materialized.
    The economy ledger builds the same five terms; `utils/stored_integer.py` deliberately carries
    no numeric ORDER BY of its own.

    Args:
        column (Any): The `StoredInteger` column to order by.

    Returns:
        ORDER BY terms to splat into `order_by`, highest value first.
    """
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
    """Per-user rod, durability, and lifetime fishing stats. One row per user.

    No `guild_id`: an angler is one person across every server, like the wallet the payouts land
    in. A broken rod keeps its `rod_id` with `durability_remaining` at zero, so the panel can
    still name what broke.
    """

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
    """Append-only catch record powering the leaderboard and history.

    The species name, grade and emoji are denormalized onto the row rather than referenced, so
    retuning or retiring a catalog entry never rewrites what someone already caught.
    """

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
    """Projects an ORM grade config into a typed view.

    The stored grade text is parsed back into `FishGrade` here, so a hand-seeded row naming a
    grade the enum does not carry fails at the store's edge instead of reaching a caller.

    Args:
        row (FishGradeConfig): The mapped catalog row.

    Returns:
        The frozen view everything outside this module sees.
    """
    return FishGradeConfigView(
        grade=FishGrade(row.grade),
        weight=row.weight,
        color=row.color,
        emoji=row.emoji,
        label=row.label,
        order_index=row.order_index,
    )


def _species_view(row: FishSpecies) -> FishSpeciesView:
    """Projects an ORM species row into a typed view.

    Args:
        row (FishSpecies): The mapped catalog row.

    Returns:
        The frozen view everything outside this module sees.
    """
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
    """Projects an ORM gear row into a typed view.

    Args:
        row (FishingGear): The mapped catalog row, rod or bait.

    Returns:
        The frozen view everything outside this module sees.
    """
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
    """Projects an ORM angler row into a typed view, defaulting to an empty angler.

    Somebody who has never fished has no row at all, and the view's own defaults describe that
    state exactly, so no caller has to branch on None; `rod` is ignored on that path.

    Args:
        angler (AnglerState | None): The mapped row, or None when the user has never fished.
        user_id (int): The angler the view is for, used when there is no row.
        rod (GearView | None): The equipped rod the caller already resolved, or None.

    Returns:
        The frozen angler view.
    """
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
    """Projects an ORM catch row into a typed view.

    A blank stored name falls back to the user id rather than rendering as an empty leaderboard
    cell.

    Args:
        row (CatchLog): The mapped catch row.

    Returns:
        The frozen catch view.
    """
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
    """Loads every gear row keyed by id for one session.

    The catalog is small and a panel needs the rod plus every owned bait's row, so one read beats
    a `session.get` per stack.

    Args:
        session (AsyncSession): The open session to read through.

    Returns:
        Every gear row keyed by `gear_id`.
    """
    result = await session.execute(statement=select(FishingGear))
    return {row.gear_id: row for row in result.scalars()}


async def list_grade_configs() -> tuple[FishGradeConfigView, ...]:
    """Lists grade configs ordered by rarity rank.

    Nothing depends on that order: `get_grade_config_map` collapses the tuple into a dict, and
    the roll engine sorts by `order_index` itself before it reads a position off it.

    Returns:
        Every grade config, ascending by `order_index`.
    """
    await _ensure_schema()
    async with open_fishing_session() as session:
        result = await session.execute(
            statement=select(FishGradeConfig).order_by(FishGradeConfig.order_index.asc())
        )
        return tuple(_grade_view(row=row) for row in result.scalars())


async def get_grade_config_map() -> dict[FishGrade, FishGradeConfigView]:
    """Returns grade configs keyed by grade for display lookups.

    Returns:
        Every grade config keyed by its `FishGrade`.
    """
    return {config.grade: config for config in await list_grade_configs()}


async def list_fish_species() -> tuple[FishSpeciesView, ...]:
    """Lists fish species ordered by grade then identifier.

    The grade half of that order is the stored text's, not the rarity rank's, and no display reads
    it: the seed script keys the rows by id.

    Returns:
        Every species row, ascending by stored grade text then `species_id`.
    """
    await _ensure_schema()
    async with open_fishing_session() as session:
        result = await session.execute(
            statement=select(FishSpecies).order_by(
                FishSpecies.grade.asc(), FishSpecies.species_id.asc()
            )
        )
        return tuple(_species_view(row=row) for row in result.scalars())


async def list_gear() -> tuple[GearView, ...]:
    """Lists all gear ordered by type then tier.

    The shop re-partitions and re-sorts what it gets (`shop.py::partition_gear`), so this order
    is a stable read rather than the one a user sees.

    Returns:
        Every rod and bait row, ascending by gear type, then tier, then `gear_id`.
    """
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
    """Creates or updates one grade config from a maintenance payload.

    Offline path only, reached through `scripts/seed_fishing.py`; runtime never writes a catalog
    row. The payload's own field bounds are the whole validation, so nothing is re-checked here.

    Args:
        config (FishGradeConfigUpsert): The validated catalog payload to write.
        now (datetime | None): Timestamp to stamp the row with; defaults to the database clock.

    Returns:
        The stored row as a view.
    """
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
    """Creates or updates one fish species from a maintenance payload.

    Offline path only, like the other two upserts. An update keeps the row's original
    `created_at`, so re-seeding an existing catalog does not restate when it was written.

    Args:
        species (FishSpeciesUpsert): The validated catalog payload to write.
        now (datetime | None): Timestamp to stamp the row with; defaults to the database clock.

    Returns:
        The stored row as a view.
    """
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
    """Creates or updates one gear item from a maintenance payload.

    Offline path only, like the other two upserts. Retuning a rod's durability does not reach the
    anglers already holding it: durability is copied onto `angler_state` at purchase time, so the
    new figure only applies to the next one sold.

    Args:
        gear (GearUpsert): The validated catalog payload to write.
        now (datetime | None): Timestamp to stamp the row with; defaults to the database clock.

    Returns:
        The stored row as a view.
    """
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
    """Returns the angler's rod and lifetime fishing state.

    A `rod_id` pointing at gear that is no longer in the catalog reads back as no rod, matching the
    `NO_ROD` a cast with the same row would get, so a retired rod degrades instead of raising.

    Args:
        user_id (int): The angler to read.

    Returns:
        That angler's view, all-zero when the user has never fished.
    """
    await _ensure_schema()
    async with open_fishing_session() as session:
        angler = await session.get(entity=AnglerState, ident=user_id)
        rod: GearView | None = None
        if angler is not None and angler.rod_id:
            rod_row = await session.get(entity=FishingGear, ident=angler.rod_id)
            rod = _gear_view(row=rod_row) if rod_row is not None else None
        return _angler_view(angler=angler, user_id=user_id, rod=rod)


async def _latest_catch_view(session: AsyncSession, user_id: int) -> CatchLogView | None:
    """Returns the angler's most recent catch, if any.

    Ties on `created_at` break by descending id, so the order is total even for two casts stamped
    at the same instant.

    Args:
        session (AsyncSession): The open session to read through.
        user_id (int): The angler whose latest catch to read.

    Returns:
        The newest catch, or None when the angler has never caught anything.
    """
    result = await session.execute(
        statement=select(CatchLog)
        .where(CatchLog.user_id == user_id)
        .order_by(CatchLog.created_at.desc(), CatchLog.id.desc())
        .limit(1)
    )
    row = result.scalars().first()
    return _catch_log_view(row=row) if row is not None else None


async def get_fishing_panel(user_id: int) -> FishingPanelData:
    """Aggregates balance, angler state, owned bait, and last catch for the panel.

    The balance comes from the economy database and everything else from games.db, so the two
    halves are not one snapshot; this is a read-only render, and a cast landing between them shows
    up one refresh later. Emptied bait stacks are dropped rather than listed at zero, and a stack
    whose catalog row is gone keeps its id as its name and sorts last, so a retired bait is still
    visible to whoever holds it.

    Args:
        user_id (int): The angler the panel is for.

    Returns:
        Everything one panel render needs, with the bait stacks ordered by gear tier then id.
    """
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
    """Loads the angler row, creating an empty one when absent.

    A created row is added to the session and left uncommitted, so it lands or rolls back with
    whatever the caller's transaction does next.

    Args:
        session (AsyncSession): The open session, already inside the caller's transaction.
        user_id (int): The angler to load or create.
        name (str): Last-seen display name to store on a newly created row.
        now (datetime): Timestamp to stamp a newly created row with.

    Returns:
        The mapped angler row, pending insert when this call created it.
    """
    angler = await session.get(entity=AnglerState, ident=user_id)
    if angler is None:
        # Set every column explicitly: SQLAlchemy `default=` only applies at INSERT
        # flush time, so the StoredInteger/Integer fields would read as None until
        # then and break the callers' in-place arithmetic on them.
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
    """Grants a purchased rod or bait and bumps the angler's lifetime gear spend.

    A rod replaces whatever is equipped and resets durability to the new rod's, so the casts left
    on the old one are deliberately forfeited; bait adds to the stack it already has. Writes into
    the caller's transaction and commits nothing.

    Args:
        session (AsyncSession): The open session, already inside the caller's transaction.
        user_id (int): The buyer.
        name (str): Last-seen display name, refreshed on every row this touches.
        gear (GearView): The catalog row that was bought.
        quantity (int): Units of bait to add; a rod ignores it.
        total_cost (int): Amount already burned, added to `total_spent_on_gear`.
        now (datetime): Timestamp to stamp every row this touches with.
    """
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

    Rods are bought exactly one at a time and replace any current rod. Bait stacks by the
    requested quantity, which must be positive and within `MAX_BAIT_PER_PURCHASE`. Quantity
    semantics are enforced here so the economy invariant never depends on the view layer.

    The debit is what the whole ordering turns on: it happens before the grant, so a failure can
    only ever charge for nothing, never grant for nothing. That failure triggers a best-effort
    refund; a refund that also fails is logged at error level and left for manual repair, because
    nothing retries it.

    Args:
        user_id (int): The buyer.
        name (str): Last-seen display name, stored on both the wallet and the angler row.
        gear_id (str): Catalog id of the rod or bait to buy.
        quantity (int): Units of bait to buy; must be at least 1, and a rod costs and grants
            exactly one however many are asked for.
        avatar_url (str): Last-seen avatar URL, passed through to the wallet write.

    Returns:
        The settled outcome, carrying a `reason` of `unknown_gear`, `invalid_quantity`,
        `insufficient` or `grant_failed` when it did not succeed.
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
    """Builds the persisted catch row for one successful cast.

    Args:
        user_id (int): The angler who made the catch.
        name (str): Last-seen display name, stored so history renders without a Discord lookup.
        roll (CatchRoll): The pure roll this row records.
        rod_id (str): Catalog id of the rod used.
        bait_id (str): Catalog id of the bait consumed.
        now (datetime): Timestamp the catch is recorded at.

    Returns:
        The unsaved row, for the caller to add inside its own transaction.
    """
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

    Everything in games.db — the bait decrement, the durability decrement, the lifetime totals and
    the catch row — commits as one transaction, held under the angler's lock and SQLite's write
    lock, so two casts cannot spend the same bait. The payout is credited to the economy wallet
    only after that commit, and a credit that fails returns `PAYOUT_DEFERRED` with the balance as
    it stands and the payout still owed, rather than rolling the catch back; nothing retries it, so
    the error log is the only repair record.

    Every guard returns before the first write, so a cast that cannot run costs neither bait nor
    durability.

    Args:
        user_id (int): The angler casting.
        name (str): Last-seen display name, refreshed on the angler row and stored on the catch.
        bait_id (str): Catalog id of the bait to consume.
        avatar_url (str): Last-seen avatar URL, passed through to the wallet credit.
        rng (Random | None): Roll source; defaults to the module's `SystemRandom`, and tests inject
            a seeded `Random` for a reproducible catch.
        now (datetime | None): Timestamp for every row written; defaults to the database clock.

    Returns:
        The settled outcome: `SUCCESS` or `PAYOUT_DEFERRED` carrying the roll, or `NO_ROD` /
        `BROKEN_ROD` / `NO_BAIT` with no roll and nothing written.
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
    """Returns the highest-value single catches across all anglers.

    It ranks catches, not anglers, so one lucky angler can hold several places. The ordering is
    numeric over the decimal-text `value` column and runs in SQL, which is what lets `LIMIT` land
    before any row is materialized; equal values put the newer catch first.

    Args:
        limit (int): Most rows to return.

    Returns:
        Catches from the highest value down, at most `limit` of them.
    """
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
    """Returns one angler's most recent catches, newest first.

    Ties on `created_at` break by descending id, so a burst of casts still reads back in the order
    it happened.

    Args:
        user_id (int): The angler whose history to read.
        limit (int): Most rows to return.

    Returns:
        That angler's catches, newest first, at most `limit` of them.
    """
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

    The offline wipe half of the store, so that stale rods, bait, and catch history do not survive
    a wallet deflation: anglers, bait and catches go, while the grade, species and gear rows stay,
    and a reset therefore needs no re-seed. It touches games.db alone, so wallets remain the
    economy reset's own business. Nothing under `src/` calls it today — the offline economy-reset
    script that did has since been removed — and `tests/test_fishing_db.py` is what still exercises
    it.

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
