"""Persistent store and settlement engine for the simulated stock market.

Owns `data/database/stock.db` and every row in it: `stock_profile` (one virtual company, its
simulation knobs and its latest quote), `stock_position` (a user's long and short side by side),
`stock_operation` plus `stock_trade_leg` (the lifecycle row and the ordered legs that are the only
audit trail there is), `stock_price_tick` (the materialized price history) and `stock_news` (the
fictional headlines that move it). Cash is not here: balances stay in `economy.db` and every money
movement goes through `services/economy/database.py`. `cogs/stock/` renders what this module
returns and `cogs/economy/cog.py` reads a portfolio out of it; neither ever touches an ORM model,
splits a price / wallet / position read, or writes a leg of its own.

There is no market loop. A symbol advances lazily when someone interacts with it
(`advance_market_in_session`), materializing one tick per `STOCK_TICK_SECONDS` boundary up to
`MAX_TICKS_PER_INTERACTION`, so a quiet symbol just replays its backlog on the next command. The
price formula itself is pure and lives in `market.py`; this module feeds it stored state (news
impulses, decayed order flow, the profile's knobs) and persists the result. A tick write is an
insert-once on `(symbol, created_at)`, so two concurrent advances over the same boundary agree on
one price instead of racing.

Settlement has exactly one entry point, `settle_stock_operation`, and that is where the operation
lifecycle comes from. `stock.db` and `economy.db` cannot commit together, so the order is fixed:
the PENDING operation and its legs commit here first, the wallet legs apply next, and only then is
the position written and the operation marked APPLIED. A failure between those steps parks the
operation at RECONCILE_REQUIRED rather than rolling back or retrying — a non-final row keeps
reserving float in `_market_exposures` and blocks that user's next trade on the symbol through
`_blocking_operation`, so two databases out of step stop the feature for one user instead of being
papered over. `list_reconciliation_operations` is what an operator reads to clear it.

Wallet legs are gross, never netted: a cover expands into a collateral credit, a short-entry credit
and then the cover debit, in that order. The order is load-bearing, because the economy side
rejects a debit it cannot cover in full at that point in the sequence, which is what lets a cover
be paid out of proceeds the spendable balance never held; the split is what keeps the economy's
`total_earned - total_spent == balance` invariant describing real flow.

Concurrency is per key and every primitive is loop-local (`utils/asyncio_locks.py`), since a
module-level `asyncio.Lock` cannot outlive the event loop it first bound to: `_operation_locks`
serializes one user's submissions on one symbol, `_market_locks` serializes tick advancement per
symbol, `_news_generation_lock` admits one due-news sweep at a time and `_news_provider_semaphore`
bounds how many provider calls that sweep runs at once. Anything that reads state a mutation plan
depends on opens with `BEGIN IMMEDIATE`, so SQLite's write lock is taken before the read rather
than after the plan is built.

The engine and its schema flag are module-level on purpose: tests monkeypatch `_engine` onto a
`tmp_path`, and `_schema_ready_for` compares by engine identity so a swapped engine bootstraps its
own schema. There are no migrations and nothing seeds a company — `upsert_stock_profile` is the
operator's offline write path, and a fresh database is an empty market.
"""

from __future__ import annotations

from time import monotonic
import uuid
from random import Random, SystemRandom
from typing import TYPE_CHECKING, Any, Final, cast
import asyncio
from datetime import datetime, timedelta

import logfire
from pydantic import Field, BaseModel, ConfigDict
from sqlalchemy import Index, String, Integer, DateTime, or_, func, text, event, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.dialects.sqlite import insert

from discordbot.typings.stock import (
    STOCK_HISTORY_DAYS,
    STOCK_BPS_DENOMINATOR,
    STOCK_INDIVIDUAL_OWNERSHIP_CAP_BPS,
    StockAction,
    StockNewsView,
    StockMarketQuote,
    StockProfileView,
    StockPositionView,
    StockTradeLegType,
    StockTradeLegView,
    StockGeneratedNews,
    StockPortfolioView,
    StockPriceTickView,
    StockProfileUpsert,
    StockDetailViewData,
    StockOperationStatus,
    StockSupplyAuditView,
    StockPortfolioHolding,
    StockSettlementResult,
    StockNewsGenerationContext,
    StockParticipantPositionView,
    StockReconciliationOperation,
)
from discordbot.utils.currency import cash_ceil, cash_floor
from discordbot.utils.timezone import database_now as _database_now
from discordbot.typings.economy import WalletDeltaLeg
from discordbot.utils.number_text import share_quantity_text
from discordbot.utils.asyncio_locks import LoopLocalLock, KeyedLockManager, LoopLocalSemaphore
from discordbot.utils.sqlite_config import ensure_sqlite_hooks, configure_sqlite_connection
from discordbot.utils.stored_integer import StoredInteger
from discordbot.services.stock.market import (
    DAILY_PRICE_LIMIT_BPS,
    NEWS_SENTIMENT_DECAY_BPS,
    NEWS_SENTIMENT_LIMIT_BPS,
    NEWS_SENTIMENT_DECAY_SECONDS,
    as_taipei,
    clamp_bps,
    format_price,
    tick_boundary,
    decay_news_sentiment,
    execution_price_cents,
    apply_daily_price_limit,
    pressure_from_order_flow,
    tick_boundaries_to_apply,
    calculate_next_price_cents,
)
from discordbot.services.stock.prompts import (
    STOCK_NEWS_BEARISH_FALLBACK_TEMPLATES,
    STOCK_NEWS_BULLISH_FALLBACK_TEMPLATES,
    STOCK_NEWS_NEUTRAL_FALLBACK_TEMPLATES,
)
from discordbot.services.economy.database import get_balance, apply_ordered_wallet_deltas

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager
    from collections.abc import Callable, Awaitable

    from sqlalchemy.engine import CursorResult

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/stock.db")
_schema_ready_for: AsyncEngine | None = None
_schema_lock = LoopLocalLock()
_operation_locks = KeyedLockManager[tuple[int, str]]()
_market_locks = KeyedLockManager[str]()
_news_generation_lock = LoopLocalLock()
_news_provider_semaphore = LoopLocalSemaphore(capacity_provider=lambda: _NEWS_PROVIDER_CONCURRENCY)
_PRODUCTION_RNG: Final[SystemRandom] = SystemRandom()
_NEWS_PROVIDER_CONCURRENCY: Final[int] = 4
_STOCK_PORTFOLIO_CACHE_TTL_SECONDS: Final[float] = 5.0
_ORDER_FLOW_LOOKBACK = timedelta(hours=24)
_NEWS_SENTIMENT_LOOKBACK = timedelta(
    seconds=NEWS_SENTIMENT_DECAY_SECONDS
    * (NEWS_SENTIMENT_LIMIT_BPS // NEWS_SENTIMENT_DECAY_BPS + 1)
)
_FINAL_OPERATION_STATUSES: Final[tuple[str, ...]] = (
    StockOperationStatus.APPLIED.value,
    StockOperationStatus.FAILED.value,
)
type _StockPortfolioCacheKey = tuple[int, int]
_stock_portfolio_cache: dict[_StockPortfolioCacheKey, tuple[float, StockPortfolioView]] = {}


def invalidate_stock_portfolio_cache(user_id: int | None = None) -> None:
    """Drops cached portfolio views so the next read rebuilds them.

    Every write that can change a valuation calls this: a profile upsert and a position reset drop
    the whole map, a finalized trade drops only its owner. A tick advance deliberately does not, so
    a valuation goes stale on the entry's own short TTL instead of invalidating on every quote.

    Args:
        user_id (int | None): Owner whose entry to drop, or None to clear every engine's entries.
    """
    if user_id is None:
        _stock_portfolio_cache.clear()
        return
    engine_id = id(_engine)
    _stock_portfolio_cache.pop((engine_id, user_id), None)


def _cached_stock_portfolio(user_id: int) -> StockPortfolioView | None:
    """Returns a cached portfolio while its short TTL holds, evicting it once it does not.

    The key carries the engine's identity, so a test that monkeypatches `_engine` onto a `tmp_path`
    can never be served the previous engine's rows.

    Args:
        user_id (int): Owner whose cached portfolio to look up.

    Returns:
        The cached view, or None when there is none or it has aged out.
    """
    cache_key: _StockPortfolioCacheKey = (id(_engine), user_id)
    cached = _stock_portfolio_cache.get(cache_key)
    if cached is None:
        return None
    cached_at, portfolio = cached
    if monotonic() - cached_at > _STOCK_PORTFOLIO_CACHE_TTL_SECONDS:
        _stock_portfolio_cache.pop(cache_key, None)
        return None
    return portfolio


def _cache_stock_portfolio(portfolio: StockPortfolioView) -> StockPortfolioView:
    """Stores one portfolio view in the short process cache and hands it straight back.

    Args:
        portfolio (StockPortfolioView): The freshly built view to cache.

    Returns:
        The same view, so the caller can cache and return in one expression.
    """
    _stock_portfolio_cache[(id(_engine), portfolio.user_id)] = (monotonic(), portfolio)
    return portfolio


class Base(DeclarativeBase):
    """Base class for stock ORM models."""

    pass


class StockProfile(Base):
    """One virtual company: its simulation knobs, its daily anchors and its latest quote.

    Maintained offline through `upsert_stock_profile`; nothing in the runtime creates a row. The
    quote here trails `stock_price_tick` — an advance writes the tick first and only copies the
    price back when it moved.
    """

    __tablename__ = "stock_profile"

    symbol: Mapped[str] = mapped_column(String(length=16), primary_key=True)
    name: Mapped[str] = mapped_column(String(length=128), nullable=False)
    category: Mapped[str] = mapped_column(String(length=64), nullable=False)
    price_cents: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    previous_close_price_cents: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    day_open_price_cents: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    total_shares: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    float_shares: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    base_volatility_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    volatility_amplifier_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    liquidity_shares: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    fair_value_cents: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    mean_reversion_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    max_tick_change_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    news_cadence_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StockPosition(Base):
    """Per-user long and short position, keyed on `(symbol, user_id)`.

    `version` counts writes and is never read back for optimistic locking; serialization comes from
    the per-user operation lock instead.
    """

    __tablename__ = "stock_position"

    symbol: Mapped[str] = mapped_column(String(length=16), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    long_shares: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    long_cost_basis: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    short_shares: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    short_entry_value: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    short_collateral: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    realized_pnl: Mapped[int] = mapped_column(StoredInteger(), default=0, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StockOperation(Base):
    """Lifecycle row for one cross-database stock operation.

    Written before the wallet moves, so a row sitting outside `_FINAL_OPERATION_STATUSES` is what
    reserves the float its legs would consume and blocks that user's next trade on the symbol.
    """

    __tablename__ = "stock_operation"
    __table_args__ = (
        Index("ix_stock_operation_user_symbol_created", "user_id", "symbol", "created_at"),
        Index("ix_stock_operation_symbol_created", "symbol", "created_at"),
    )

    operation_id: Mapped[str] = mapped_column(String(length=36), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(length=16), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    requested_action: Mapped[str] = mapped_column(String(length=16), nullable=False)
    status: Mapped[str] = mapped_column(String(length=32), nullable=False)
    failure_reason: Mapped[str] = mapped_column(String(length=512), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StockTradeLeg(Base):
    """One ordered leg produced by a stock operation, and the whole audit trail there is.

    Each leg keeps its own slipped `price_cents` and its own deltas, so legs are never netted into
    a single movement; there is no transaction table above this.
    """

    __tablename__ = "stock_trade_leg"
    __table_args__ = (
        Index("ix_stock_trade_leg_operation_order", "operation_id", "leg_order"),
        Index("ix_stock_trade_leg_symbol_created", "symbol", "created_at"),
        Index("ix_stock_trade_leg_user_symbol_created", "user_id", "symbol", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operation_id: Mapped[str] = mapped_column(String(length=36), nullable=False)
    leg_order: Mapped[int] = mapped_column(Integer, nullable=False)
    symbol: Mapped[str] = mapped_column(String(length=16), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    leg_type: Mapped[str] = mapped_column(String(length=32), nullable=False)
    shares: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    price_cents: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    wallet_delta: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    basis_delta: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    collateral_delta: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    realized_pnl_delta: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StockPriceTick(Base):
    """Materialized price tick, stamped at its boundary rather than at write time.

    `(symbol, created_at)` is unique, which is what makes a lazy replay of an already-priced
    boundary an insert-once rather than a duplicate point.
    """

    __tablename__ = "stock_price_tick"
    __table_args__ = (
        Index("ix_stock_price_tick_symbol_created", "symbol", "created_at", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(length=16), nullable=False)
    price_cents: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StockNews(Base):
    """One fictional headline, whose sentiment fires once into a lazy tick.

    `id` is `<symbol>-<cadence bucket>`, so a bucket holds at most one headline and a second
    generation in the same bucket collides instead of stacking another impulse onto the price.
    """

    __tablename__ = "stock_news"
    __table_args__ = (Index("ix_stock_news_symbol_created", "symbol", "created_at"),)

    id: Mapped[str] = mapped_column(String(length=64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(length=16), nullable=False)
    headline: Mapped[str] = mapped_column(String(length=256), nullable=False)
    sentiment_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(length=32), default="template", nullable=False)
    model: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class _StockOperationPlan(StockSettlementResult):
    """A settlement that passed validation, before any row is written.

    Adds no field to `StockSettlementResult`; the subclass exists so the plan builders can return
    either an accepted plan or a rejection in the same slot, and settlement can tell them apart by
    type rather than by trusting `success`.
    """


class _StockExecutionSnapshot(BaseModel):
    """Submit-time state needed to cap a requested quantity.

    Frozen and read-only: the quantity searches walk it repeatedly, so it is gathered once under
    the market lock rather than re-read per candidate size.
    """

    model_config = ConfigDict(frozen=True)

    action: StockAction = Field(..., description="Requested buy/cover or short/sell action.")
    price_cents: int = Field(..., description="Submit-time reference quote price in cents.")
    liquidity_shares: int = Field(
        ..., description="Per-stock liquidity used for execution slippage."
    )
    max_order_impact_bps: int = Field(
        ..., description="Maximum per-leg price impact in basis points."
    )
    wallet_balance: int = Field(..., description="User's wallet cash available at submit time.")
    position: StockPositionView = Field(
        ..., description="User's current long/short position snapshot."
    )
    available_long_shares: int = Field(
        ..., description="Float shares still openable as long market-wide."
    )
    available_short_shares: int = Field(
        ..., description="Float shares still borrowable for shorting."
    )
    available_individual_long_shares: int = Field(
        ..., description="New long shares the user can open before the 49% ownership cap."
    )


class _StockMarketExposure(BaseModel):
    """Aggregate market exposure for one symbol."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Stock symbol the exposure totals apply to.")
    long_shares: int = Field(..., description="Aggregate long shares held plus pending opens.")
    short_shares: int = Field(
        ..., description="Aggregate short shares borrowed plus pending opens."
    )
    available_long_shares: int = Field(..., description="Float shares still openable as long.")
    available_short_shares: int = Field(
        ..., description="Float shares still borrowable for shorting."
    )


class _StockOrderFlowSummary(BaseModel):
    """Recent order-flow summary for stock news context."""

    model_config = ConfigDict(frozen=True)

    buy_side_shares: int = Field(
        default=0, description="Recent buy-side share volume in the window."
    )
    sell_side_shares: int = Field(
        default=0, description="Recent sell-side share volume in the window."
    )
    pressure_bps: int = Field(
        default=0, description="Decayed net order-flow pressure in basis points."
    )


def _configure_sqlite_connection(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Applies the project's standard SQLite PRAGMAs to one stock connection.

    Takes the shared defaults unchanged: no foreign keys, and the `StoredInteger` UDFs left on,
    which the money and share columns need to compare and add as integers rather than as text. The
    two listeners below differ only in the signature SQLAlchemy hands them, so both land here.

    Args:
        dbapi_connection (Any): The DBAPI connection to configure.
    """
    configure_sqlite_connection(dbapi_connection=dbapi_connection)


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Configures each connection the engine newly opens.

    The decorator binds this to whichever engine existed at import time, so a test-swapped engine
    only gets it once `ensure_sqlite_hooks` re-installs it.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
        _connection_record (Any): SQLAlchemy's pool record, unused.
    """
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _configure_sqlite_on_checkout(
    dbapi_connection: object, _connection_record: object, _connection_proxy: object
) -> None:
    """Configures a pooled connection on its way out of the pool.

    A connection the pool already held when a test swapped `_engine` will never fire `connect`
    again, so this is the listener that reaches it.

    Args:
        dbapi_connection (object): The pooled DBAPI connection being checked out.
        _connection_record (object): SQLAlchemy's pool record, unused.
        _connection_proxy (object): SQLAlchemy's connection proxy, unused.
    """
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _current_schema_lock() -> asyncio.Lock:
    """Returns the schema bootstrap lock, rebound if the event loop changed.

    Returns:
        The lock guarding the one-time `create_all` for this loop.
    """
    return _schema_lock.get()


def _current_news_generation_lock() -> asyncio.Lock:
    """Returns the due-news sweep lock, rebound if the event loop changed.

    Returns:
        The lock admitting one `ensure_due_stock_news` sweep at a time.
    """
    return _news_generation_lock.get()


def _current_news_provider_semaphore() -> asyncio.Semaphore:
    """Returns the news provider limiter, rebound if the event loop changed.

    Returns:
        The semaphore bounding concurrent provider calls to `_NEWS_PROVIDER_CONCURRENCY`.
    """
    return _news_provider_semaphore.get()


def _operation_lock(user_id: int, symbol: str) -> AbstractAsyncContextManager[None]:
    """Returns the lock serializing one user's submissions on one symbol.

    Held across the whole of `settle_stock_operation`, wallet legs included, so a second
    submission cannot plan against a position the first has already spent.

    Args:
        user_id (int): Discord user submitting the operation.
        symbol (str): Already-normalized ticker symbol.

    Returns:
        An async context manager holding that key's lock for the current event loop.
    """
    return _operation_locks.hold(key=(user_id, symbol))


def _market_lock(symbol: str) -> AbstractAsyncContextManager[None]:
    """Returns the lock serializing tick advancement for one symbol.

    Args:
        symbol (str): Ticker symbol, upper-cased here so a mixed-case caller shares the key.

    Returns:
        An async context manager holding that symbol's lock for the current event loop.
    """
    return _market_locks.hold(key=symbol.upper())


def open_stock_session() -> AsyncSession:
    """Opens an async session on the current stock engine, re-installing its PRAGMA listeners.

    The listeners are re-installed on every open (idempotently) because a test-swapped engine
    carries none of its own, and an unconfigured connection would run without WAL.

    Returns:
        A session bound to `_engine` with `expire_on_commit=False`, so a view built from a
        committed row is still readable.
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
    """Creates the stock tables once per engine.

    The double-check around the lock keeps the common case lock-free, and the flag stores the
    engine itself rather than a bool so a test that swaps `_engine` bootstraps its own file
    instead of inheriting the previous engine's "ready".
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


def _profile_view(profile: StockProfile) -> StockProfileView:
    """Projects an ORM profile into the frozen view the cogs read.

    Args:
        profile (StockProfile): The loaded profile row.

    Returns:
        The profile as a `StockProfileView`, minus `created_at`, which nothing renders.
    """
    return StockProfileView(
        symbol=profile.symbol,
        name=profile.name,
        category=profile.category,
        price_cents=profile.price_cents,
        previous_close_price_cents=profile.previous_close_price_cents,
        day_open_price_cents=profile.day_open_price_cents,
        total_shares=profile.total_shares,
        float_shares=profile.float_shares,
        base_volatility_bps=profile.base_volatility_bps,
        volatility_amplifier_bps=profile.volatility_amplifier_bps,
        liquidity_shares=profile.liquidity_shares,
        fair_value_cents=profile.fair_value_cents,
        mean_reversion_bps=profile.mean_reversion_bps,
        max_tick_change_bps=profile.max_tick_change_bps,
        news_cadence_hours=profile.news_cadence_hours,
        updated_at=profile.updated_at,
    )


def _position_view(
    position: StockPosition | None, symbol: str, user_id: int, user_name: str = ""
) -> StockPositionView:
    """Projects an ORM position into a typed view, inventing an empty one where there is no row.

    A user who has never traded a symbol has no row, and the detail view still has to render, so
    absence becomes an all-zero position rather than None.

    Args:
        position (StockPosition | None): The loaded position row, or None when the user has none.
        symbol (str): Ticker symbol the view is for.
        user_id (int): Discord user the view is for.
        user_name (str): Display name to fall back on when the row stored none.

    Returns:
        The position as a `StockPositionView`.
    """
    if position is None:
        return StockPositionView(symbol=symbol, user_id=user_id, user_name=user_name)
    return StockPositionView(
        symbol=position.symbol,
        user_id=position.user_id,
        user_name=position.user_name or user_name,
        long_shares=position.long_shares,
        long_cost_basis=position.long_cost_basis,
        short_shares=position.short_shares,
        short_entry_value=position.short_entry_value,
        short_collateral=position.short_collateral,
        realized_pnl=position.realized_pnl,
    )


def _participant_position_view(position: StockPosition) -> StockParticipantPositionView:
    """Projects a position into the summary every viewer of the stock sees.

    Sizes and realized P&L only; cost basis, entry value and collateral stay private to the owner.

    Args:
        position (StockPosition): The loaded position row.

    Returns:
        The public summary, with the user id as text when no name was ever stored.
    """
    return StockParticipantPositionView(
        user_id=position.user_id,
        user_name=position.user_name or str(position.user_id),
        long_shares=position.long_shares,
        short_shares=position.short_shares,
        realized_pnl=position.realized_pnl,
    )


def _trade_leg_view(leg: StockTradeLeg, user_name: str = "") -> StockTradeLegView:
    """Projects an ORM trade leg into a typed view.

    Args:
        leg (StockTradeLeg): The loaded leg row.
        user_name (str): Name to fall back on when the leg row stored none.

    Returns:
        The leg as a `StockTradeLegView`, naming the trader by the leg's own name, then the
        fallback, then the user id as text, so a row always renders.
    """
    return StockTradeLegView(
        operation_id=leg.operation_id,
        leg_order=leg.leg_order,
        symbol=leg.symbol,
        user_id=leg.user_id,
        user_name=leg.user_name or user_name or str(leg.user_id),
        leg_type=StockTradeLegType(leg.leg_type),
        shares=leg.shares,
        price_cents=leg.price_cents,
        wallet_delta=leg.wallet_delta,
        basis_delta=leg.basis_delta,
        collateral_delta=leg.collateral_delta,
        realized_pnl_delta=leg.realized_pnl_delta,
        created_at=leg.created_at,
    )


def _news_view(news: StockNews) -> StockNewsView:
    """Projects an ORM news row into a typed view.

    Args:
        news (StockNews): The loaded news row.

    Returns:
        The headline as a `StockNewsView`, keeping the `source` / `model` provenance the refresh
        logic reads back.
    """
    return StockNewsView(
        symbol=news.symbol,
        headline=news.headline,
        sentiment_bps=news.sentiment_bps,
        source=news.source,
        model=news.model,
        expires_at=news.expires_at,
        created_at=news.created_at,
    )


def _tick_view(tick: StockPriceTick) -> StockPriceTickView:
    """Projects an ORM tick row into a typed view.

    Args:
        tick (StockPriceTick): The loaded tick row.

    Returns:
        The tick as a `StockPriceTickView`, dropping the surrogate id the chart never reads.
    """
    return StockPriceTickView(
        symbol=tick.symbol, price_cents=tick.price_cents, created_at=tick.created_at
    )


def _quote_from_profile(profile: StockProfile, pressure_bps: int) -> StockMarketQuote:
    """Builds a quote from the latest profile row.

    The change pair is measured against the previous close rather than the day open, so it reads
    the way a real ticker does across a day rollover. A profile whose previous close is zero
    reports no change instead of dividing by it.

    Args:
        profile (StockProfile): The advanced profile row.
        pressure_bps (int): Recent order-flow pressure, computed by the caller.

    Returns:
        The quote the market board and the detail header render.
    """
    change_cents = profile.price_cents - profile.previous_close_price_cents
    change_bps = (
        change_cents * 10_000 // profile.previous_close_price_cents
        if profile.previous_close_price_cents > 0
        else 0
    )
    return StockMarketQuote(
        profile=_profile_view(profile=profile),
        change_cents=change_cents,
        change_bps=change_bps,
        pressure_bps=pressure_bps,
    )


async def upsert_stock_profile(
    profile: StockProfileUpsert, now: datetime | None = None
) -> StockProfileView:
    """Creates or retunes one virtual company from an operator-authored payload.

    The offline maintenance path, and the only way a company comes into existence; nothing in the
    runtime calls it. A create seeds both daily anchors from `price_cents` and writes the current
    boundary's tick; a retune leaves the anchors alone and rewrites that boundary's tick only when
    the price actually changed, so a knob-only edit does not disturb the chart. Every holding's
    valuation can move here, so the whole portfolio cache is dropped rather than one owner's.

    Args:
        profile (StockProfileUpsert): The company's fields as the operator wants them.
        now (datetime | None): Timestamp to write and to bucket the tick into, defaulting to now.

    Returns:
        The stored profile as a view.

    Raises:
        ValueError: The symbol is empty once stripped.
    """
    await _ensure_schema()
    effective_now = now or _database_now()
    normalized_symbol = profile.symbol.strip().upper()
    if not normalized_symbol:
        msg = "Stock symbol cannot be empty"
        raise ValueError(msg)
    async with open_stock_session() as session:
        existing = await session.get(entity=StockProfile, ident=normalized_symbol)
        if existing is None:
            existing = StockProfile(
                symbol=normalized_symbol,
                name=profile.name,
                category=profile.category,
                price_cents=profile.price_cents,
                previous_close_price_cents=profile.price_cents,
                day_open_price_cents=profile.price_cents,
                total_shares=profile.total_shares,
                float_shares=profile.float_shares,
                base_volatility_bps=profile.base_volatility_bps,
                volatility_amplifier_bps=profile.volatility_amplifier_bps,
                liquidity_shares=profile.liquidity_shares,
                fair_value_cents=profile.fair_value_cents,
                mean_reversion_bps=profile.mean_reversion_bps,
                max_tick_change_bps=profile.max_tick_change_bps,
                news_cadence_hours=profile.news_cadence_hours,
                created_at=effective_now,
                updated_at=effective_now,
            )
            session.add(instance=existing)
            await session.flush()
            await _insert_price_tick_or_existing(
                session=session,
                symbol=normalized_symbol,
                price_cents=profile.price_cents,
                created_at=tick_boundary(dt=effective_now),
            )
        else:
            existing.name = profile.name
            existing.category = profile.category
            existing.total_shares = profile.total_shares
            existing.float_shares = profile.float_shares
            existing.base_volatility_bps = profile.base_volatility_bps
            existing.volatility_amplifier_bps = profile.volatility_amplifier_bps
            existing.liquidity_shares = profile.liquidity_shares
            existing.fair_value_cents = profile.fair_value_cents
            existing.mean_reversion_bps = profile.mean_reversion_bps
            existing.max_tick_change_bps = profile.max_tick_change_bps
            existing.news_cadence_hours = profile.news_cadence_hours
            if existing.price_cents != profile.price_cents:
                existing.price_cents = profile.price_cents
                await _upsert_price_tick(
                    session=session,
                    symbol=normalized_symbol,
                    price_cents=profile.price_cents,
                    created_at=tick_boundary(dt=effective_now),
                )
            existing.updated_at = effective_now
        await session.commit()
        invalidate_stock_portfolio_cache()
        return _profile_view(profile=existing)


async def list_stock_profiles() -> tuple[StockProfileView, ...]:
    """Lists every company as stored, without advancing a tick.

    Returns:
        Every profile in symbol order, at whatever price the last advance left it.
    """
    await _ensure_schema()
    async with open_stock_session() as session:
        result = await session.execute(
            statement=select(StockProfile).order_by(StockProfile.symbol.asc())
        )
        return tuple(_profile_view(profile=profile) for profile in result.scalars())


async def list_stock_supply_audit() -> tuple[StockSupplyAuditView, ...]:
    """Reports issued supply against aggregate exposure, for an operator retuning a company.

    Advances nothing, so it describes the market as stored. The remaining capacity already has the
    non-final operations' opens subtracted out of it, which is why their count rides alongside:
    non-zero means the figure is provisional and may come back if those operations fail.

    Returns:
        One audit row per company, in symbol order.
    """
    await _ensure_schema()
    async with open_stock_session() as session:
        result = await session.execute(
            statement=select(StockProfile).order_by(StockProfile.symbol.asc())
        )
        profiles = tuple(result.scalars())
        exposures = await _market_exposures(session=session, profiles=profiles)
        symbols = tuple(profile.symbol for profile in profiles)
        non_final_counts: dict[str, int] = {}
        if symbols:
            count_result = await session.execute(
                statement=select(StockOperation.symbol, func.count(StockOperation.operation_id))
                .where(
                    StockOperation.symbol.in_(symbols),
                    StockOperation.status.notin_(_FINAL_OPERATION_STATUSES),
                )
                .group_by(StockOperation.symbol)
            )
            non_final_counts = {symbol: int(count) for symbol, count in count_result.all()}
        audits: list[StockSupplyAuditView] = []
        for profile in profiles:
            exposure = exposures[profile.symbol]
            audits.append(
                StockSupplyAuditView(
                    symbol=profile.symbol,
                    name=profile.name,
                    price_cents=profile.price_cents,
                    total_shares=profile.total_shares,
                    float_shares=profile.float_shares,
                    long_shares=exposure.long_shares,
                    short_shares=exposure.short_shares,
                    available_long_shares=exposure.available_long_shares,
                    available_short_shares=exposure.available_short_shares,
                    liquidity_shares=profile.liquidity_shares,
                    non_final_operations=non_final_counts.get(profile.symbol, 0),
                )
            )
        return tuple(audits)


async def ensure_due_stock_news(
    news_provider: (
        Callable[[StockNewsGenerationContext], Awaitable[StockGeneratedNews | None]] | None
    ) = None,
    symbols: tuple[str, ...] | None = None,
    now: datetime | None = None,
) -> None:
    """Fills in the headlines whose cadence has come round, preferring the provider's.

    Serialized process-wide, because two interactions arriving together would otherwise both find
    the same symbol due. Provider calls run concurrently under `_news_provider_semaphore` and,
    deliberately, with no session open — an LLM call is slow enough that holding a write
    transaction across it would stall every other trade on the file. A provider that raises or
    answers with a blank headline degrades to the deterministic templates built from the same
    market context, so the symbol always ends up with a headline.

    Passing a provider also reopens a symbol that already has a template headline inside its
    cadence, which the per-bucket insert then upgrades in place; nothing ever downgrades an AI
    headline back to a template.

    Args:
        news_provider (Callable[[StockNewsGenerationContext], Awaitable[StockGeneratedNews | None]] | None): Producer for one symbol's headline, or None to use the templates alone.
        symbols (tuple[str, ...] | None): Symbols to consider, upper-cased here; None means all.
        now (datetime | None): Timestamp deciding what is due, defaulting to now.
    """
    await _ensure_schema()
    effective_now = now or _database_now()
    normalized_symbols = tuple(symbol.upper() for symbol in symbols) if symbols else None
    async with _current_news_generation_lock():
        due_contexts = await _due_stock_news_contexts(
            normalized_symbols=normalized_symbols,
            now=effective_now,
            allow_template_upgrade=news_provider is not None,
        )
        if not due_contexts:
            return

        async def generate_row(
            context: StockNewsGenerationContext,
        ) -> tuple[StockNewsGenerationContext, StockGeneratedNews]:
            """Generates one news row without holding a database transaction.

            Returns:
                The context paired with its headline, which is always produced.
            """
            generated: StockGeneratedNews | None = None
            if news_provider is not None:
                async with _current_news_provider_semaphore():
                    try:
                        generated = await news_provider(context)
                    # Broad on purpose: a provider is best-effort, so anything it raises degrades
                    # to the deterministic templates rather than leaving the symbol without news.
                    except Exception:
                        logfire.warn(
                            "Stock news provider failed; using deterministic fallback",
                            symbol=context.profile.symbol,
                            _exc_info=True,
                        )
            if generated is None or not generated.headline.strip():
                generated = _fallback_generated_news(context=context, now=effective_now)
            return context, generated

        rows = await asyncio.gather(*(generate_row(context=context) for context in due_contexts))

        async with open_stock_session() as session:
            for context, generated in rows:
                await _insert_generated_news(
                    session=session,
                    profile=context.profile,
                    generated=generated,
                    now=effective_now,
                )
            await session.commit()


async def _due_stock_news_contexts(
    normalized_symbols: tuple[str, ...] | None, now: datetime, allow_template_upgrade: bool = False
) -> tuple[StockNewsGenerationContext, ...]:
    """Decides which symbols are due a headline and gathers the context each producer needs.

    A symbol is due when it has never had a headline, when its cadence has elapsed, or when a
    provider is available and the latest headline is still a template. The cadence is floored at
    one hour, so a profile carrying zero cannot make every read due.

    Args:
        normalized_symbols (tuple[str, ...] | None): Upper-cased symbols to consider, or None
            for every profile.
        now (datetime): Timestamp the cadence is measured against.
        allow_template_upgrade (bool): Whether a template headline inside its cadence counts as
            due, which is what lets an available provider replace one.

    Returns:
        One context per due symbol, empty when nothing is due.
    """
    async with open_stock_session() as session:
        statement = select(StockProfile)
        if normalized_symbols:
            statement = statement.where(StockProfile.symbol.in_(normalized_symbols))
        result = await session.execute(statement=statement.order_by(StockProfile.symbol.asc()))
        profiles = tuple(result.scalars())
        profile_symbols = tuple(profile.symbol for profile in profiles)
        latest_news_by_symbol: dict[str, tuple[datetime, str, str, int]] = {}
        if profile_symbols:
            latest_news_subquery = (
                select(StockNews.symbol, func.max(StockNews.created_at).label("latest_created_at"))
                .where(StockNews.symbol.in_(profile_symbols))
                .group_by(StockNews.symbol)
                .subquery()
            )
            latest_result = await session.execute(
                statement=select(
                    StockNews.symbol,
                    StockNews.created_at,
                    StockNews.source,
                    StockNews.headline,
                    StockNews.sentiment_bps,
                ).join(
                    latest_news_subquery,
                    (StockNews.symbol == latest_news_subquery.c.symbol)
                    & (StockNews.created_at == latest_news_subquery.c.latest_created_at),
                )
            )
            latest_news_by_symbol = {
                symbol: (latest_at, source, headline, sentiment_bps)
                for symbol, latest_at, source, headline, sentiment_bps in latest_result.all()
                if latest_at is not None
            }

        due_profiles: list[StockProfile] = []
        for profile in profiles:
            latest_news = latest_news_by_symbol.get(profile.symbol)
            cadence = timedelta(hours=max(profile.news_cadence_hours, 1))
            if latest_news is None:
                due_profiles.append(profile)
                continue
            latest_news_at, latest_news_source, _headline, _sentiment_bps = latest_news
            if as_taipei(dt=now) - as_taipei(dt=latest_news_at) < cadence and (
                not allow_template_upgrade or latest_news_source != "template"
            ):
                continue
            due_profiles.append(profile)
        if not due_profiles:
            return ()
        return await _stock_news_generation_contexts(
            session=session,
            profiles=tuple(due_profiles),
            latest_news_by_symbol=latest_news_by_symbol,
            now=now,
        )


async def _stock_news_generation_contexts(
    session: AsyncSession,
    profiles: tuple[StockProfile, ...],
    latest_news_by_symbol: dict[str, tuple[datetime, str, str, int]],
    now: datetime,
) -> tuple[StockNewsGenerationContext, ...]:
    """Builds the market picture a news producer is told about, batched over every due symbol.

    Order flow and recent sentiment are fetched once for the whole batch rather than per symbol,
    since a market-wide sweep would otherwise run two queries per company.

    Args:
        session (AsyncSession): Open session to read order flow and news through.
        profiles (tuple[StockProfile, ...]): The due profiles, already selected.
        latest_news_by_symbol (dict[str, tuple[datetime, str, str, int]]): Per symbol, the latest
            headline's `(created_at, source, headline, sentiment_bps)`, as the caller read it.
        now (datetime): Timestamp the flow and sentiment windows end at.

    Returns:
        One context per profile, in the order given.
    """
    symbols = tuple(profile.symbol for profile in profiles)
    flow_summaries = await _order_flow_summaries_for_symbols(
        session=session,
        symbols=symbols,
        at=now,
        liquidity_by_symbol={profile.symbol: profile.liquidity_shares for profile in profiles},
    )
    news_rows_by_symbol = await _news_rows_by_symbol_for_context(
        session=session, symbols=symbols, now=now
    )
    lookback_hours = max(int(_ORDER_FLOW_LOOKBACK.total_seconds() // 3600), 1)
    contexts: list[StockNewsGenerationContext] = []
    for profile in profiles:
        flow = flow_summaries.get(profile.symbol, _StockOrderFlowSummary())
        latest_news = latest_news_by_symbol.get(profile.symbol)
        latest_news_headline = ""
        latest_news_sentiment_bps = 0
        if latest_news is not None:
            _latest_at, _latest_source, latest_news_headline, latest_news_sentiment_bps = (
                latest_news
            )
        change_cents = profile.price_cents - profile.previous_close_price_cents
        change_bps = (
            change_cents * 10_000 // profile.previous_close_price_cents
            if profile.previous_close_price_cents > 0
            else 0
        )
        contexts.append(
            StockNewsGenerationContext(
                profile=_profile_view(profile=profile),
                change_cents=change_cents,
                change_bps=change_bps,
                pressure_bps=flow.pressure_bps,
                buy_side_shares=flow.buy_side_shares,
                sell_side_shares=flow.sell_side_shares,
                net_order_shares=flow.buy_side_shares - flow.sell_side_shares,
                recent_news_sentiment_bps=_decayed_news_sentiment_for_context(
                    news_rows=tuple(news_rows_by_symbol.get(profile.symbol, ())), at=now
                ),
                latest_news_headline=latest_news_headline,
                latest_news_sentiment_bps=latest_news_sentiment_bps,
                lookback_hours=lookback_hours,
            )
        )
    return tuple(contexts)


async def _order_flow_summaries_for_symbols(
    session: AsyncSession,
    symbols: tuple[str, ...],
    at: datetime,
    liquidity_by_symbol: dict[str, int],
) -> dict[str, _StockOrderFlowSummary]:
    """Summarizes recent order flow for several symbols in one query.

    Only legs of APPLIED operations count, so a pending or reconcile-parked operation never shows
    up as traded volume.

    Args:
        session (AsyncSession): Open session to read trade legs through.
        symbols (tuple[str, ...]): Symbols to summarize.
        at (datetime): End of the `_ORDER_FLOW_LOOKBACK` window.
        liquidity_by_symbol (dict[str, int]): Each symbol's liquidity depth, which the pressure
            figure is scaled against; a symbol missing from it is treated as having none.

    Returns:
        One summary per requested symbol, zeroed where nothing traded.
    """
    if not symbols:
        return {}
    since = at - _ORDER_FLOW_LOOKBACK
    result = await session.execute(
        statement=select(
            StockTradeLeg.symbol,
            StockTradeLeg.leg_type,
            StockTradeLeg.shares,
            StockTradeLeg.created_at,
        )
        .join(StockOperation, StockOperation.operation_id == StockTradeLeg.operation_id)
        .where(
            StockTradeLeg.symbol.in_(symbols),
            StockTradeLeg.created_at >= since,
            StockTradeLeg.created_at <= at,
            StockOperation.status == StockOperationStatus.APPLIED.value,
        )
    )
    rows_by_symbol: dict[str, list[tuple[str, int, datetime]]] = {symbol: [] for symbol in symbols}
    for symbol, leg_type, shares, created_at in result.all():
        rows_by_symbol.setdefault(symbol, []).append((leg_type, shares, created_at))
    return {
        symbol: _order_flow_summary_from_rows(
            pressure_rows=tuple(rows), at=at, liquidity_shares=liquidity_by_symbol.get(symbol, 0)
        )
        for symbol, rows in rows_by_symbol.items()
    }


def _order_flow_summary_from_rows(
    pressure_rows: tuple[tuple[str, int, datetime], ...], at: datetime, liquidity_shares: int
) -> _StockOrderFlowSummary:
    """Splits prefetched legs into buy-side and sell-side volume and their net pressure.

    The two volumes are raw sums over the window while `pressure_bps` is time-decayed, so a
    producer sees both how much traded and how much of it still counts.

    Args:
        pressure_rows (tuple[tuple[str, int, datetime], ...]): Legs as `(leg_type, shares,
            created_at)`.
        at (datetime): End of the window, which the decay ages toward.
        liquidity_shares (int): Liquidity depth the pressure is scaled against.

    Returns:
        The order-flow summary for one symbol.
    """
    buy_side_shares = 0
    sell_side_shares = 0
    for leg_type, shares, _created_at in pressure_rows:
        if leg_type in (StockTradeLegType.OPEN_LONG.value, StockTradeLegType.COVER_SHORT.value):
            buy_side_shares += shares
        else:
            sell_side_shares += shares
    return _StockOrderFlowSummary(
        buy_side_shares=buy_side_shares,
        sell_side_shares=sell_side_shares,
        pressure_bps=_recent_pressure_bps_from_rows(
            pressure_rows=pressure_rows, at=at, liquidity_shares=liquidity_shares
        ),
    )


async def _news_rows_by_symbol_for_context(
    session: AsyncSession, symbols: tuple[str, ...], now: datetime
) -> dict[str, tuple[StockNews, ...]]:
    """Fetches the still-counting news rows for several symbols in one query.

    Bounded by `_NEWS_SENTIMENT_LOOKBACK`, which is how long a headline can still carry any decayed
    sentiment at all, and skips rows that have already expired.

    Args:
        session (AsyncSession): Open session to read news through.
        symbols (tuple[str, ...]): Symbols to fetch for.
        now (datetime): End of the sentiment window.

    Returns:
        One entry per requested symbol, newest first, empty where nothing is in window.
    """
    if not symbols:
        return {}
    result = await session.execute(
        statement=select(StockNews)
        .where(
            StockNews.symbol.in_(symbols),
            StockNews.created_at <= now,
            StockNews.created_at >= now - _NEWS_SENTIMENT_LOOKBACK,
            or_(StockNews.expires_at.is_(None), StockNews.expires_at >= now),
        )
        .order_by(StockNews.created_at.desc())
    )
    rows_by_symbol: dict[str, list[StockNews]] = {symbol: [] for symbol in symbols}
    for news in result.scalars():
        rows_by_symbol.setdefault(news.symbol, []).append(news)
    return {symbol: tuple(rows) for symbol, rows in rows_by_symbol.items()}


async def _insert_generated_news(
    session: AsyncSession, profile: StockProfileView, generated: StockGeneratedNews, now: datetime
) -> None:
    """Files one generated headline into its cadence bucket, upgrading a template in place.

    The id is `<symbol>-<bucket>`, so a bucket holds one headline and a second generation collides
    with the first instead of stacking a second impulse onto the same ticks. The conflict clause is
    deliberately one-way — it rewrites only a `template` row with an `ai` one, so a provider that
    arrives late improves the bucket while a template refresh can never undo it. `sentiment_bps`
    is clamped here rather than trusted, since a producer may ask for anything.

    Args:
        session (AsyncSession): Open session; the caller commits.
        profile (StockProfileView): The symbol the headline belongs to, and its cadence.
        generated (StockGeneratedNews): What the producer returned.
        now (datetime): Creation stamp, and the anchor for `expires_at`.
    """
    bucket = _stock_news_bucket(profile=profile, now=now)
    source = generated.source or "template"
    insert_statement = insert(StockNews).values(
        id=f"{profile.symbol.lower()}-{bucket}",
        symbol=profile.symbol,
        headline=generated.headline.strip()[:256],
        sentiment_bps=clamp_bps(
            value=generated.sentiment_bps,
            lower=-NEWS_SENTIMENT_LIMIT_BPS,
            upper=NEWS_SENTIMENT_LIMIT_BPS,
        ),
        source=source,
        model=generated.model,
        expires_at=now + _NEWS_SENTIMENT_LOOKBACK,
        created_at=now,
    )
    await session.execute(
        statement=insert_statement.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "headline": insert_statement.excluded.headline,
                "sentiment_bps": insert_statement.excluded.sentiment_bps,
                "source": insert_statement.excluded.source,
                "model": insert_statement.excluded.model,
                "expires_at": insert_statement.excluded.expires_at,
                "created_at": insert_statement.excluded.created_at,
            },
            where=(StockNews.source == "template") & (insert_statement.excluded.source == "ai"),
        )
    )


def _stock_news_bucket(profile: StockProfileView, now: datetime) -> int:
    """Returns which cadence window a timestamp falls in, as a whole number of periods.

    Args:
        profile (StockProfileView): The symbol whose cadence sets the window, floored at an hour.
        now (datetime): Timestamp to bucket, read in Asia/Taipei.

    Returns:
        The bucket index, which is what makes a news row's id stable within one cadence period.
    """
    cadence_seconds = max(profile.news_cadence_hours, 1) * 60 * 60
    return int(as_taipei(dt=now).timestamp()) // cadence_seconds


def _fallback_generated_news(
    context: StockNewsGenerationContext, now: datetime
) -> StockGeneratedNews:
    """Picks a template headline for a symbol, without an LLM.

    The seed mixes the symbol, the cadence bucket and the batch's own order-flow figures, so the
    choice is reproducible within a bucket but does not repeat the same line for every company.

    Args:
        context (StockNewsGenerationContext): The market picture, which also chooses the tone.
        now (datetime): Timestamp deciding the cadence bucket.

    Returns:
        A headline marked `source="template"`, which an AI refresh may later replace.
    """
    profile = context.profile
    bucket = _stock_news_bucket(profile=profile, now=now)
    templates = _fallback_templates_for_context(context=context)
    seed = (
        sum(ord(char) for char in profile.symbol)
        + bucket
        + context.buy_side_shares
        + context.sell_side_shares
        + abs(context.pressure_bps) * 7
    )
    headline_template, sentiment_bps = templates[seed % len(templates)]
    return StockGeneratedNews(
        headline=headline_template.format(
            name=profile.name, symbol=profile.symbol, category=profile.category
        ),
        sentiment_bps=sentiment_bps,
        source="template",
    )


def _fallback_templates_for_context(
    context: StockNewsGenerationContext,
) -> tuple[tuple[str, int], ...]:
    """Chooses the bullish, bearish or neutral template set from the market context.

    Reads the same context the LLM prompt gets, so a fallback headline still fits the tape rather
    than contradicting a chart the user is looking at.

    Args:
        context (StockNewsGenerationContext): The market picture for one symbol.

    Returns:
        The `(headline template, sentiment_bps)` pairs to pick from.
    """
    signal_bps = (
        context.change_bps // 2 + context.pressure_bps + context.recent_news_sentiment_bps // 3
    )
    if signal_bps >= 50:
        return STOCK_NEWS_BULLISH_FALLBACK_TEMPLATES
    if signal_bps <= -50:
        return STOCK_NEWS_BEARISH_FALLBACK_TEMPLATES
    return STOCK_NEWS_NEUTRAL_FALLBACK_TEMPLATES


async def _latest_tick(session: AsyncSession, symbol: str) -> StockPriceTick | None:
    """Reads the most recent materialized tick for a stock.

    Args:
        session (AsyncSession): Open session to read through.
        symbol (str): Ticker symbol to read.

    Returns:
        The newest tick, or None for a symbol that has never been advanced.
    """
    result = await session.execute(
        statement=select(StockPriceTick)
        .where(StockPriceTick.symbol == symbol)
        .order_by(StockPriceTick.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _insert_price_tick_or_existing(
    session: AsyncSession, symbol: str, price_cents: int, created_at: datetime
) -> int:
    """Claims a boundary's price, or reports the price already stored there.

    This is what makes lazy advancement idempotent: the unique index on `(symbol, created_at)` lets
    the loser of a concurrent advance read back the winner's price and carry on from it, so two
    interactions racing over the same backlog converge on one price history instead of forking.

    Args:
        session (AsyncSession): Open session; the caller commits.
        symbol (str): Ticker symbol the tick belongs to.
        price_cents (int): Price to store if this call is the one that inserts.
        created_at (datetime): The tick boundary, not the wall clock.

    Returns:
        The price now persisted at that boundary, which is `price_cents` only when this call won.
    """
    result = cast(
        "CursorResult[Any]",
        await session.execute(
            statement=insert(StockPriceTick)
            .values(symbol=symbol, price_cents=price_cents, created_at=created_at)
            .on_conflict_do_nothing(index_elements=["symbol", "created_at"])
        ),
    )
    if result.rowcount:
        return price_cents
    existing = await session.execute(
        statement=select(StockPriceTick.price_cents)
        .where(StockPriceTick.symbol == symbol, StockPriceTick.created_at == created_at)
        .order_by(StockPriceTick.id.desc())
        .limit(1)
    )
    return existing.scalar_one()


async def _upsert_price_tick(
    session: AsyncSession, symbol: str, price_cents: int, created_at: datetime
) -> None:
    """Writes an operator's price onto a boundary, replacing whatever is there.

    The maintenance twin of `_insert_price_tick_or_existing`, and the one place a stored tick is
    overwritten: a hand-set price has to reach the chart, so here the newest write wins.

    Args:
        session (AsyncSession): Open session; the caller commits.
        symbol (str): Ticker symbol the tick belongs to.
        price_cents (int): Price the operator set.
        created_at (datetime): The tick boundary to write onto.
    """
    await session.execute(
        statement=insert(StockPriceTick)
        .values(symbol=symbol, price_cents=price_cents, created_at=created_at)
        .on_conflict_do_update(
            index_elements=["symbol", "created_at"], set_={"price_cents": price_cents}
        )
    )


async def _news_rows_for_boundaries(
    session: AsyncSession, symbol: str, boundaries: tuple[datetime, ...]
) -> tuple[StockNews, ...]:
    """Fetches every news row that can influence a chosen run of boundaries, in one query.

    Reaching back a further `_NEWS_SENTIMENT_LOOKBACK` before the first boundary is what lets the
    whole backlog be priced without a query per tick.

    Args:
        session (AsyncSession): Open session to read news through.
        symbol (str): Ticker symbol being advanced.
        boundaries (tuple[datetime, ...]): The boundaries about to be priced, ascending.

    Returns:
        The in-window, unexpired news rows newest first, empty when there are no boundaries.
    """
    if not boundaries:
        return ()
    result = await session.execute(
        statement=select(StockNews)
        .where(
            StockNews.symbol == symbol,
            StockNews.created_at <= boundaries[-1],
            StockNews.created_at >= boundaries[0] - _NEWS_SENTIMENT_LOOKBACK,
            or_(StockNews.expires_at.is_(None), StockNews.expires_at >= boundaries[0]),
        )
        .order_by(StockNews.created_at.desc())
    )
    return tuple(result.scalars())


def _decayed_news_sentiment_for_context(news_rows: tuple[StockNews, ...], at: datetime) -> int:
    """Sums the decayed sentiment still hanging over a symbol, for the news prompt alone.

    Deliberately not what the price reads: a headline lands on the price once, as an impulse at its
    own boundary. This decayed figure exists so a producer knows how much news mood is already
    priced in, and feeding it back into the formula would apply the same headline every tick.

    Args:
        news_rows (tuple[StockNews, ...]): Candidate news rows; future and expired ones are
            skipped here rather than by the query.
        at (datetime): The moment to decay toward.

    Returns:
        The summed decayed sentiment, clamped to `NEWS_SENTIMENT_LIMIT_BPS`.
    """
    sentiment = 0
    for news in news_rows:
        if as_taipei(dt=news.created_at) > as_taipei(dt=at):
            continue
        if news.expires_at is not None and as_taipei(dt=news.expires_at) < as_taipei(dt=at):
            continue
        elapsed_seconds = max(
            int((tick_boundary(dt=at) - tick_boundary(dt=news.created_at)).total_seconds()), 0
        )
        sentiment += decay_news_sentiment(
            sentiment_bps=news.sentiment_bps, elapsed_seconds=elapsed_seconds
        )
    return clamp_bps(
        value=sentiment, lower=-NEWS_SENTIMENT_LIMIT_BPS, upper=NEWS_SENTIMENT_LIMIT_BPS
    )


def _news_impulse_by_boundary(
    news_rows: tuple[StockNews, ...], applied_boundaries: tuple[datetime, ...]
) -> dict[datetime, int]:
    """Maps each applied tick boundary to its one-shot news sentiment sum.

    Each news row contributes its clamped sentiment exactly once, at the first applied boundary at
    or after its own tick boundary. News whose tick boundary falls before every applied boundary is
    skipped, its impulse having already landed on a previous lazy advance. Compression matters
    here: when a backlog drops boundaries, a headline that fired into a dropped one is carried
    forward to the next surviving boundary rather than lost.

    Args:
        news_rows (tuple[StockNews, ...]): News rows in window for this advance.
        applied_boundaries (tuple[datetime, ...]): The boundaries actually being priced.

    Returns:
        The one-shot sentiment sum per boundary, empty when either input is.
    """
    if not applied_boundaries or not news_rows:
        return {}
    sorted_boundaries = sorted(applied_boundaries)
    impulse: dict[datetime, int] = dict.fromkeys(applied_boundaries, 0)
    earliest = sorted_boundaries[0]
    for news in news_rows:
        news_boundary = tick_boundary(dt=news.created_at)
        if news_boundary < earliest:
            continue
        target = next(b for b in sorted_boundaries if b >= news_boundary)
        impulse[target] += clamp_bps(
            value=news.sentiment_bps,
            lower=-NEWS_SENTIMENT_LIMIT_BPS,
            upper=NEWS_SENTIMENT_LIMIT_BPS,
        )
    return impulse


async def _recent_pressure_bps(
    session: AsyncSession, symbol: str, at: datetime, liquidity_shares: int
) -> int:
    """Reads one symbol's current order-flow pressure straight from the database.

    The single-shot twin of `_pressure_rows_for_boundaries`, used for the figure a quote carries
    rather than for pricing a run of boundaries.

    Args:
        session (AsyncSession): Open session to read trade legs through.
        symbol (str): Ticker symbol to measure.
        at (datetime): End of the `_ORDER_FLOW_LOOKBACK` window.
        liquidity_shares (int): Liquidity depth the pressure is scaled against.

    Returns:
        Net pressure in basis points, bounded by `PRESSURE_LIMIT_BPS`.
    """
    since = at - _ORDER_FLOW_LOOKBACK
    result = await session.execute(
        statement=select(StockTradeLeg.leg_type, StockTradeLeg.shares, StockTradeLeg.created_at)
        .join(StockOperation, StockOperation.operation_id == StockTradeLeg.operation_id)
        .where(
            StockTradeLeg.symbol == symbol,
            StockTradeLeg.created_at >= since,
            StockTradeLeg.created_at <= at,
            StockOperation.status == StockOperationStatus.APPLIED.value,
        )
    )
    return _recent_pressure_bps_from_rows(
        pressure_rows=tuple(result.tuples().all()), at=at, liquidity_shares=liquidity_shares
    )


async def _pressure_rows_for_boundaries(
    session: AsyncSession, symbol: str, boundaries: tuple[datetime, ...]
) -> tuple[tuple[str, int, datetime], ...]:
    """Fetches every trade leg that can influence a chosen run of boundaries, in one query.

    Reaching back a further `_ORDER_FLOW_LOOKBACK` before the first boundary is what lets each
    boundary recompute its own decayed pressure in memory instead of querying per tick. Only legs
    of APPLIED operations count, so a pending one never moves the price it is waiting on.

    Args:
        session (AsyncSession): Open session to read trade legs through.
        symbol (str): Ticker symbol being advanced.
        boundaries (tuple[datetime, ...]): The boundaries about to be priced, ascending.

    Returns:
        Legs as `(leg_type, shares, created_at)`, empty when there are no boundaries.
    """
    if not boundaries:
        return ()
    result = await session.execute(
        statement=select(StockTradeLeg.leg_type, StockTradeLeg.shares, StockTradeLeg.created_at)
        .join(StockOperation, StockOperation.operation_id == StockTradeLeg.operation_id)
        .where(
            StockTradeLeg.symbol == symbol,
            StockTradeLeg.created_at >= boundaries[0] - _ORDER_FLOW_LOOKBACK,
            StockTradeLeg.created_at <= boundaries[-1],
            StockOperation.status == StockOperationStatus.APPLIED.value,
        )
    )
    return tuple((leg_type, shares, created_at) for leg_type, shares, created_at in result.all())


def _recent_pressure_bps_from_rows(
    pressure_rows: tuple[tuple[str, int, datetime], ...], at: datetime, liquidity_shares: int
) -> int:
    """Computes decayed net order-flow pressure from legs already in memory.

    A leg's weight falls linearly to zero across the lookback window, so pressure fades on its own
    rather than dropping off a cliff when a trade leaves the window. Legs outside `[at -
    lookback, at]` are ignored here, which is what lets one prefetched batch serve every boundary
    of a backlog.

    Args:
        pressure_rows (tuple[tuple[str, int, datetime], ...]): Legs as `(leg_type, shares,
            created_at)`.
        at (datetime): The moment pressure is measured at.
        liquidity_shares (int): Liquidity depth the net flow is scaled against.

    Returns:
        Net pressure in basis points, bounded by `PRESSURE_LIMIT_BPS`.
    """
    since = at - _ORDER_FLOW_LOOKBACK
    net_shares = 0.0
    at_taipei = as_taipei(dt=at)
    since_taipei = as_taipei(dt=since)
    total_seconds = _ORDER_FLOW_LOOKBACK.total_seconds()
    for leg_type, shares, created_at in pressure_rows:
        created_at_taipei = as_taipei(dt=created_at)
        if created_at_taipei < since_taipei or created_at_taipei > at_taipei:
            continue
        age_seconds = max((at_taipei - created_at_taipei).total_seconds(), 0)
        remaining_seconds = max(total_seconds - age_seconds, 0)
        if remaining_seconds <= 0:
            continue
        decayed_shares = shares * remaining_seconds / total_seconds
        if leg_type in (StockTradeLegType.OPEN_LONG.value, StockTradeLegType.COVER_SHORT.value):
            net_shares += decayed_shares
        else:
            net_shares -= decayed_shares
    return pressure_from_order_flow(net_shares=net_shares, liquidity_shares=liquidity_shares)


async def advance_market_in_session(
    session: AsyncSession,
    symbol: str,
    now: datetime | None = None,
    rng: Random | None = None,
    begin_immediate: bool = True,
) -> StockMarketQuote:
    """Prices every tick boundary a symbol still owes, up to now.

    The whole simulation loop, run on demand instead of on a timer. News and trade legs for the
    entire run are fetched once up front, then each boundary is priced from the previous boundary's
    persisted price — persisted, not computed, so an advance that loses the insert race adopts the
    winner's price and the two agree. A boundary that crosses the Asia/Taipei date first rolls the
    previous close to the price the day ended on, so the daily limit bands the new day against the
    right anchor, and then stamps the day open from what actually landed.

    Mutates the profile and inserts ticks in the caller's transaction without committing, so a
    caller settling a trade can advance the market and write its own rows atomically. The caller
    must hold that symbol's `_market_lock`.

    Args:
        session (AsyncSession): Open session; the caller commits.
        symbol (str): Ticker symbol to advance, already normalized.
        now (datetime | None): Advance target, defaulting to now.
        rng (Random | None): Volatility source, defaulting to `SystemRandom`; tests inject a
            seeded one to make a run reproducible.
        begin_immediate (bool): Whether to take SQLite's write lock here. Settlement passes False
            because it has already begun its own immediate transaction.

    Returns:
        The quote after the advance, carrying pressure measured at `now`.

    Raises:
        ValueError: No profile exists for the symbol.
    """
    if begin_immediate:
        await _begin_immediate(session=session)
    effective_now = now or _database_now()
    effective_rng = rng or _PRODUCTION_RNG
    profile_result = await session.execute(
        statement=select(StockProfile).where(StockProfile.symbol == symbol)
    )
    profile = profile_result.scalar_one_or_none()
    if profile is None:
        msg = f"Unknown stock symbol: {symbol}"
        raise ValueError(msg)

    latest_tick = await _latest_tick(session=session, symbol=symbol)
    if latest_tick is None:
        latest_tick_at = tick_boundary(dt=effective_now)
        current_price = await _insert_price_tick_or_existing(
            session=session,
            symbol=symbol,
            price_cents=profile.price_cents,
            created_at=latest_tick_at,
        )
        previous_tick_at = latest_tick_at
    else:
        current_price = latest_tick.price_cents
        previous_tick_at = latest_tick.created_at
    boundaries = tick_boundaries_to_apply(latest_tick_at=previous_tick_at, now=effective_now)
    news_rows = await _news_rows_for_boundaries(
        session=session, symbol=symbol, boundaries=boundaries
    )
    pressure_rows = await _pressure_rows_for_boundaries(
        session=session, symbol=symbol, boundaries=boundaries
    )
    news_impulse = _news_impulse_by_boundary(news_rows=news_rows, applied_boundaries=boundaries)
    for boundary in boundaries:
        news_sentiment = news_impulse.get(boundary, 0)
        pressure_bps = _recent_pressure_bps_from_rows(
            pressure_rows=pressure_rows, at=boundary, liquidity_shares=profile.liquidity_shares
        )
        next_price = calculate_next_price_cents(
            previous_price_cents=current_price,
            news_sentiment_bps=news_sentiment,
            pressure_bps=pressure_bps,
            base_volatility_bps=profile.base_volatility_bps,
            volatility_amplifier_bps=profile.volatility_amplifier_bps,
            fair_value_cents=profile.fair_value_cents,
            mean_reversion_strength_bps=profile.mean_reversion_bps,
            max_tick_change_bps=profile.max_tick_change_bps,
            rng=effective_rng,
        )
        rolls_over_day = as_taipei(dt=boundary).date() != as_taipei(dt=previous_tick_at).date()
        if rolls_over_day:
            profile.previous_close_price_cents = current_price
        next_price = apply_daily_price_limit(
            price_cents=next_price,
            previous_close_cents=profile.previous_close_price_cents,
            limit_bps=DAILY_PRICE_LIMIT_BPS,
        )
        current_price = await _insert_price_tick_or_existing(
            session=session, symbol=symbol, price_cents=next_price, created_at=boundary
        )
        if rolls_over_day:
            profile.day_open_price_cents = current_price
        previous_tick_at = boundary

    if current_price != profile.price_cents:
        profile.price_cents = current_price
        profile.updated_at = previous_tick_at
    pressure_bps = await _recent_pressure_bps(
        session=session, symbol=symbol, at=effective_now, liquidity_shares=profile.liquidity_shares
    )
    return _quote_from_profile(profile=profile, pressure_bps=pressure_bps)


async def list_market_quotes(
    now: datetime | None = None, rng: Random | None = None, refresh_news: bool = True
) -> tuple[StockMarketQuote, ...]:
    """Advances every symbol and returns the board the market view renders.

    Symbols are advanced one at a time, each under its own lock and committed before the next, so
    the board never holds every symbol's write lock at once.

    Args:
        now (datetime | None): Advance target, defaulting to now.
        rng (Random | None): Volatility source, defaulting to `SystemRandom`.
        refresh_news (bool): Whether to run the due-news sweep first. False is for a caller that
            has already swept, or one that must not pay for it.

    Returns:
        One quote per company, in symbol order.
    """
    if refresh_news:
        await ensure_due_stock_news(now=now)
    await _ensure_schema()
    async with open_stock_session() as session:
        symbols_result = await session.execute(
            statement=select(StockProfile.symbol).order_by(StockProfile.symbol.asc())
        )
        symbols = tuple(symbols_result.scalars().all())
    quotes: list[StockMarketQuote] = []
    async with open_stock_session() as session:
        for symbol in symbols:
            async with _market_lock(symbol=symbol):
                quotes.append(
                    await advance_market_in_session(
                        session=session, symbol=symbol, now=now, rng=rng
                    )
                )
                await session.commit()
    return tuple(quotes)


async def _advance_symbols_for_views(
    symbols: tuple[str, ...], now: datetime | None, rng: Random | None
) -> None:
    """Advances a set of symbols on one session, still one lock and one commit per symbol.

    Sharing the session saves a connection per symbol; the per-symbol lock and commit are what stop
    that turning into one long transaction over the whole set.

    Args:
        symbols (tuple[str, ...]): Symbols to advance.
        now (datetime | None): Advance target, defaulting to now.
        rng (Random | None): Volatility source, defaulting to `SystemRandom`.
    """
    async with open_stock_session() as session:
        for symbol in symbols:
            async with _market_lock(symbol=symbol):
                await advance_market_in_session(session=session, symbol=symbol, now=now, rng=rng)
                await session.commit()


async def _current_stock_portfolio(user_id: int) -> StockPortfolioView:
    """Values a user's non-zero holdings at whatever price is currently stored.

    Advances nothing, so the caller is responsible for having advanced the symbols it cares about
    first. Rows come back sorted by total size then symbol, which is display order rather than
    anything the totals depend on.

    Args:
        user_id (int): Owner whose portfolio to build.

    Returns:
        The portfolio, with equity and P&L summed over the holdings; all zero when there are none.
    """
    async with open_stock_session() as session:
        result = await session.execute(
            statement=select(StockPosition, StockProfile)
            .join(StockProfile, StockProfile.symbol == StockPosition.symbol)
            .where(
                StockPosition.user_id == user_id,
                or_(StockPosition.long_shares > 0, StockPosition.short_shares > 0),
            )
            .order_by(StockPosition.symbol.asc())
        )
        position_rows = list(result.all())
        position_rows.sort(
            key=lambda row: (-_position_share_total(position=row[0]), row[0].symbol)
        )
        holdings = tuple(
            _portfolio_holding_view(position=position, profile=profile)
            for position, profile in position_rows
        )
    return StockPortfolioView(
        user_id=user_id,
        holdings=holdings,
        equity_value=sum(holding.equity_value for holding in holdings),
        unrealized_pnl=sum(holding.unrealized_pnl for holding in holdings),
        realized_pnl=sum(holding.realized_pnl for holding in holdings),
    )


async def get_stock_detail(
    symbol: str,
    user_id: int,
    user_name: str = "",
    now: datetime | None = None,
    rng: Random | None = None,
) -> StockDetailViewData:
    """Gathers everything one `/stock` detail render needs, under a single market advance.

    Every stock-side read happens in one session while the symbol's lock is held, so the quote, the
    position, the recent trades and the chart cannot disagree about which tick the user is looking
    at. Only the wallet balance is fetched afterwards, from the economy database.

    Args:
        symbol (str): Ticker symbol to render.
        user_id (int): Viewer, whose own position and balance are included.
        user_name (str): Viewer's display name, used only when no name is stored on the position.
        now (datetime | None): Advance target, defaulting to now.
        rng (Random | None): Volatility source, defaulting to `SystemRandom`.

    Returns:
        The detail payload: the viewer's own quote, balance and position, plus the stock-wide
        recent trades, participant positions, news and ticks the public message shows.
    """
    await ensure_due_stock_news(symbols=(symbol,), now=now)
    await _ensure_schema()
    async with open_stock_session() as session, _market_lock(symbol=symbol):
        quote = await advance_market_in_session(session=session, symbol=symbol, now=now, rng=rng)
        position = await _get_position_view(
            session=session, symbol=symbol, user_id=user_id, user_name=user_name
        )
        recent_trades = await _recent_trade_views(session=session, symbol=symbol)
        public_positions = await _public_position_views(session=session, symbol=symbol)
        news = await _news_views(session=session, symbol=symbol)
        ticks = await _price_tick_views(session=session, symbol=symbol, now=now or _database_now())
        await session.commit()
    balance = await get_balance(user_id=user_id)
    return StockDetailViewData(
        quote=quote,
        balance=balance,
        position=position,
        recent_trades=recent_trades,
        public_positions=public_positions,
        news=news,
        ticks=ticks,
    )


async def get_stock_portfolio(
    user_id: int, now: datetime | None = None, rng: Random | None = None
) -> StockPortfolioView:
    """Advances the symbols a user holds and values their portfolio at the result.

    The economy profile embed's read, so it is served from a short process cache on the default
    call — passing `now` or `rng` means the caller wants a specific market state, so those calls
    neither read nor write that cache. A user holding nothing skips the advance entirely.

    Args:
        user_id (int): Owner whose portfolio to build.
        now (datetime | None): Advance target, defaulting to now; also disables the cache.
        rng (Random | None): Volatility source, defaulting to `SystemRandom`; also disables the
            cache.

    Returns:
        The portfolio, valued at the post-advance quotes.
    """
    if now is None and rng is None:
        cached = _cached_stock_portfolio(user_id=user_id)
        if cached is not None:
            return cached
    await _ensure_schema()
    async with open_stock_session() as session:
        symbols = await _user_position_symbols(session=session, user_id=user_id)
    if symbols:
        await ensure_due_stock_news(symbols=symbols, now=now)
        await _advance_symbols_for_views(symbols=symbols, now=now, rng=rng)
    portfolio = await _current_stock_portfolio(user_id=user_id)
    if now is None and rng is None:
        return _cache_stock_portfolio(portfolio=portfolio)
    return portfolio


async def get_stock_news(symbol: str) -> tuple[StockNewsView, ...]:
    """Refreshes a symbol's news if it is due, then returns the latest headlines.

    Advances no tick, so a headline created here only reaches the price on the next advance.

    Args:
        symbol (str): Ticker symbol to read.

    Returns:
        The most recent headlines, newest first.
    """
    await ensure_due_stock_news(symbols=(symbol,))
    await _ensure_schema()
    async with open_stock_session() as session:
        return await _news_views(session=session, symbol=symbol)


async def _get_position_view(
    session: AsyncSession, symbol: str, user_id: int, user_name: str = ""
) -> StockPositionView:
    """Reads one user's position for a symbol inside the caller's session.

    Args:
        session (AsyncSession): Open session, typically already inside the caller's transaction.
        symbol (str): Ticker symbol to read.
        user_id (int): Owner to read for.
        user_name (str): Display name to fall back on when the row stored none.

    Returns:
        The position, all-zero when the user has never traded the symbol.
    """
    result = await session.execute(
        statement=select(StockPosition).where(
            StockPosition.symbol == symbol, StockPosition.user_id == user_id
        )
    )
    return _position_view(
        position=result.scalar_one_or_none(), symbol=symbol, user_id=user_id, user_name=user_name
    )


async def _recent_trade_views(
    session: AsyncSession, symbol: str, user_id: int | None = None
) -> tuple[StockTradeLegView, ...]:
    """Reads the stock's most recent applied trade legs, newest leg of the newest operation first.

    Only APPLIED operations appear, so a pending or reconcile-parked trade is never shown as
    something that happened. The parent operation's stored name rides along as the fallback for a
    leg row that never got one.

    Args:
        session (AsyncSession): Open session to read through.
        symbol (str): Ticker symbol to read.
        user_id (int | None): Restrict to one trader, or None for every trader on the stock.

    Returns:
        Up to eight legs, newest first.
    """
    filters = [
        StockTradeLeg.symbol == symbol,
        StockOperation.status == StockOperationStatus.APPLIED.value,
    ]
    if user_id is not None:
        filters.append(StockTradeLeg.user_id == user_id)
    result = await session.execute(
        statement=select(StockTradeLeg, StockOperation.user_name)
        .join(StockOperation, StockOperation.operation_id == StockTradeLeg.operation_id)
        .where(*filters)
        .order_by(StockTradeLeg.created_at.desc(), StockTradeLeg.leg_order.desc())
        .limit(8)
    )
    return tuple(_trade_leg_view(leg=leg, user_name=user_name) for leg, user_name in result.all())


async def _public_position_views(
    session: AsyncSession, symbol: str
) -> tuple[StockParticipantPositionView, ...]:
    """Reads the largest open positions on a stock, for the public participant table.

    Ranked by combined long-plus-short size, with the most recent update breaking a tie.

    Args:
        session (AsyncSession): Open session to read through.
        symbol (str): Ticker symbol to read.

    Returns:
        Up to eight participant summaries, largest first.
    """
    result = await session.execute(
        statement=select(StockPosition)
        .where(
            StockPosition.symbol == symbol,
            or_(StockPosition.long_shares > 0, StockPosition.short_shares > 0),
        )
        .order_by(StockPosition.updated_at.desc())
    )
    positions = list(result.scalars())
    positions.sort(
        key=lambda position: (
            _position_share_total(position=position),
            as_taipei(dt=position.updated_at),
        ),
        reverse=True,
    )
    return tuple(_participant_position_view(position=position) for position in positions[:8])


async def _user_position_symbols(session: AsyncSession, user_id: int) -> tuple[str, ...]:
    """Lists the symbols a user actually holds, so a portfolio read advances only those.

    Args:
        session (AsyncSession): Open session to read through.
        user_id (int): Owner to look up.

    Returns:
        Symbols with a non-zero long or short side, in symbol order.
    """
    result = await session.execute(
        statement=select(StockPosition.symbol)
        .where(
            StockPosition.user_id == user_id,
            or_(StockPosition.long_shares > 0, StockPosition.short_shares > 0),
        )
        .order_by(StockPosition.symbol.asc())
    )
    return tuple(result.scalars())


def _position_share_total(position: StockPosition) -> int:
    """Returns a position's combined size, as the sort key for display rows.

    Long and short are added rather than netted: the ranking is about how much of the stock a
    trader is exposed to, not which way.

    Args:
        position (StockPosition): The position row to measure.

    Returns:
        Long shares plus short shares.
    """
    return position.long_shares + position.short_shares


def _portfolio_holding_view(
    position: StockPosition, profile: StockProfile
) -> StockPortfolioHolding:
    """Values one position at the current quote.

    Rounding goes against the trader on both sides — the long side floors and the cover cost ceils
    — so a portfolio can never be inflated by rounding. A short contributes its collateral plus its
    entry value minus what covering would cost, never a bare negative market value, which is what
    keeps equity comparable across a long-only and a short-heavy portfolio.

    Args:
        position (StockPosition): The position row to value.
        profile (StockProfile): Its company, for the name and the current price.

    Returns:
        The holding with its market value, equity and unrealized P&L filled in.
    """
    long_market_value = cash_floor(cents=profile.price_cents * position.long_shares)
    short_cover_cost = cash_ceil(cents=profile.price_cents * position.short_shares)
    unrealized_pnl = (
        long_market_value
        - position.long_cost_basis
        + position.short_entry_value
        - short_cover_cost
    )
    equity_value = (
        long_market_value
        + position.short_collateral
        + position.short_entry_value
        - short_cover_cost
    )
    return StockPortfolioHolding(
        symbol=position.symbol,
        name=profile.name,
        price_cents=profile.price_cents,
        long_shares=position.long_shares,
        long_cost_basis=position.long_cost_basis,
        long_market_value=long_market_value,
        short_shares=position.short_shares,
        short_entry_value=position.short_entry_value,
        short_collateral=position.short_collateral,
        short_cover_cost=short_cover_cost,
        equity_value=equity_value,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=position.realized_pnl,
    )


async def _market_exposure(session: AsyncSession, profile: StockProfile) -> _StockMarketExposure:
    """Returns one symbol's exposure, through the batched query.

    Args:
        session (AsyncSession): Open session to read through.
        profile (StockProfile): The company whose float the totals are measured against.

    Returns:
        That symbol's exposure and remaining capacity.
    """
    return (await _market_exposures(session=session, profiles=(profile,)))[profile.symbol]


async def _market_exposures(
    session: AsyncSession, profiles: tuple[StockProfile, ...]
) -> dict[str, _StockMarketExposure]:
    """Totals held and in-flight exposure per symbol, against each company's float.

    The opens of every non-final operation are counted as though they had already happened. That
    reservation is what stops two submissions racing over the last of the float: the first commits
    its PENDING legs before touching a wallet, so the second plans against a float that already has
    them subtracted, and the shares come back only if that operation ends FAILED.

    Long and short are capped separately, each against the whole float, because a borrow is not
    drawn from the same pool a long is.

    Args:
        session (AsyncSession): Open session to read through.
        profiles (tuple[StockProfile, ...]): The companies to total, which also supply the floats.

    Returns:
        Exposure keyed by symbol, empty when no profiles were given.
    """
    profile_by_symbol = {profile.symbol: profile for profile in profiles}
    if not profile_by_symbol:
        return {}
    symbols = tuple(profile_by_symbol)
    exposure_totals = {symbol: {"long": 0, "short": 0} for symbol in symbols}

    position_result = await session.execute(
        statement=select(
            StockPosition.symbol, StockPosition.long_shares, StockPosition.short_shares
        ).where(
            StockPosition.symbol.in_(symbols),
            or_(StockPosition.long_shares > 0, StockPosition.short_shares > 0),
        )
    )
    for symbol, long_shares, short_shares in position_result.all():
        exposure_totals[symbol]["long"] += long_shares
        exposure_totals[symbol]["short"] += short_shares

    pending_result = await session.execute(
        statement=select(StockTradeLeg.symbol, StockTradeLeg.leg_type, StockTradeLeg.shares)
        .join(StockOperation, StockOperation.operation_id == StockTradeLeg.operation_id)
        .where(
            StockTradeLeg.symbol.in_(symbols),
            StockTradeLeg.leg_type.in_((
                StockTradeLegType.OPEN_LONG.value,
                StockTradeLegType.OPEN_SHORT.value,
            )),
            StockOperation.status.notin_(_FINAL_OPERATION_STATUSES),
        )
    )
    for symbol, leg_type, shares in pending_result.all():
        if leg_type == StockTradeLegType.OPEN_LONG.value:
            exposure_totals[symbol]["long"] += shares
        else:
            exposure_totals[symbol]["short"] += shares

    exposures: dict[str, _StockMarketExposure] = {}
    for symbol, profile in profile_by_symbol.items():
        long_shares = exposure_totals[symbol]["long"]
        short_shares = exposure_totals[symbol]["short"]
        exposures[symbol] = _StockMarketExposure(
            symbol=symbol,
            long_shares=long_shares,
            short_shares=short_shares,
            available_long_shares=max(profile.float_shares - long_shares, 0),
            available_short_shares=max(profile.float_shares - short_shares, 0),
        )
    return exposures


async def _news_views(session: AsyncSession, symbol: str) -> tuple[StockNewsView, ...]:
    """Reads a stock's latest headlines inside the caller's session.

    Ignores `expires_at`: expiry bounds how long a headline still counts toward sentiment, not how
    long it is worth reading.

    Args:
        session (AsyncSession): Open session to read through.
        symbol (str): Ticker symbol to read.

    Returns:
        Up to five headlines, newest first.
    """
    result = await session.execute(
        statement=select(StockNews)
        .where(StockNews.symbol == symbol)
        .order_by(StockNews.created_at.desc())
        .limit(5)
    )
    return tuple(_news_view(news=news) for news in result.scalars())


async def _price_tick_views(
    session: AsyncSession, symbol: str, now: datetime
) -> tuple[StockPriceTickView, ...]:
    """Reads the tick history the 7D chart draws.

    Args:
        session (AsyncSession): Open session to read through.
        symbol (str): Ticker symbol to read.
        now (datetime): End of the `STOCK_HISTORY_DAYS` window.

    Returns:
        The window's ticks, oldest first, which is the order the chart plots them in.
    """
    since = now - timedelta(days=STOCK_HISTORY_DAYS)
    result = await session.execute(
        statement=select(StockPriceTick)
        .where(StockPriceTick.symbol == symbol, StockPriceTick.created_at >= since)
        .order_by(StockPriceTick.created_at.asc())
    )
    return tuple(_tick_view(tick=tick) for tick in result.scalars())


def _parse_quantity(
    raw_quantity: str,
    action: StockAction,
    price_cents: int,
    wallet_balance: int,
    position: StockPositionView,
) -> int:
    """Turns what the user typed into a share count.

    The `ALL` shorthand means "flatten the opposite side" whenever there is one: a buy with an open
    short closes exactly the short and opens no long, a short with a long sells exactly the long.
    Only with nothing to close does it mean the whole wallet, sized at the reference price before
    slippage — deliberately optimistic, since `_clamp_quantity_to_available` is what trims it back
    to what actually executes. Anything else goes through `int`, whose `ValueError` the caller
    turns into a format error rather than a zero.

    Args:
        raw_quantity (str): The raw modal text, tolerating thousands separators and whitespace.
        action (StockAction): Which direction was submitted.
        price_cents (int): Reference quote price, before per-leg impact.
        wallet_balance (int): Cash available at submit time.
        position (StockPositionView): The user's current position in the symbol.

    Returns:
        The requested share count, not yet capped against the market.
    """
    normalized = raw_quantity.strip().replace(",", "")
    if normalized.upper() in {"ALL", "全部", "MAX"}:
        if action == StockAction.BUY and position.short_shares > 0:
            return position.short_shares
        if action == StockAction.SHORT and position.long_shares > 0:
            return position.long_shares
        return wallet_balance * 100 // price_cents
    return int(normalized)


def _is_all_quantity(raw_quantity: str) -> bool:
    """Returns whether the raw quantity is the ALL shorthand rather than a number.

    Asked again after parsing, because an `ALL` that sizes to zero is a "nothing is executable"
    failure worth explaining, while a typed `0` is just an invalid quantity.

    Args:
        raw_quantity (str): The raw modal text.

    Returns:
        True for any accepted spelling of the shorthand.
    """
    return raw_quantity.strip().replace(",", "").upper() in {"ALL", "全部", "MAX"}


def _prorated_amount(total: int, shares: int, current_shares: int) -> int:
    """Splits a running total across a partial close, handing over everything on a full one.

    Integer division leaves dust behind on each partial close, so closing the whole position
    releases the stored total outright rather than the sum of its slices; otherwise a flattened
    position would keep a few units of basis or collateral it can never release.

    Args:
        total (int): The stored basis, entry value or collateral being released from.
        shares (int): Shares being closed.
        current_shares (int): Shares the total currently covers.

    Returns:
        The share of the total this close releases.
    """
    if shares >= current_shares:
        return total
    return total * shares // current_shares


def _buy_execution_price(
    price_cents: int, shares: int, liquidity_shares: int, max_impact_bps: int
) -> int:
    """Returns the slipped price a buy-side leg executes at, above the quote.

    Args:
        price_cents (int): Reference quote price.
        shares (int): Size of this one leg, which is what the impact scales with.
        liquidity_shares (int): The company's liquidity depth.
        max_impact_bps (int): Ceiling on the per-leg impact.

    Returns:
        The execution price in cents.
    """
    return execution_price_cents(
        reference_price_cents=price_cents,
        shares=shares,
        liquidity_shares=liquidity_shares,
        max_impact_bps=max_impact_bps,
        is_buy=True,
    )


def _sell_execution_price(
    price_cents: int, shares: int, liquidity_shares: int, max_impact_bps: int
) -> int:
    """Returns the slipped price a sell-side leg executes at, below the quote.

    Args:
        price_cents (int): Reference quote price.
        shares (int): Size of this one leg, which is what the impact scales with.
        liquidity_shares (int): The company's liquidity depth.
        max_impact_bps (int): Ceiling on the per-leg impact.

    Returns:
        The execution price in cents.
    """
    return execution_price_cents(
        reference_price_cents=price_cents,
        shares=shares,
        liquidity_shares=liquidity_shares,
        max_impact_bps=max_impact_bps,
        is_buy=False,
    )


def _buy_cost(price_cents: int, shares: int, liquidity_shares: int, max_impact_bps: int) -> int:
    """Returns what a buy-side leg costs in whole wallet units, rounded up.

    Args:
        price_cents (int): Reference quote price.
        shares (int): Size of this one leg.
        liquidity_shares (int): The company's liquidity depth.
        max_impact_bps (int): Ceiling on the per-leg impact.

    Returns:
        Cash the wallet must cover, ceiled so the rounding never favours the buyer.
    """
    execution_price = _buy_execution_price(
        price_cents=price_cents,
        shares=shares,
        liquidity_shares=liquidity_shares,
        max_impact_bps=max_impact_bps,
    )
    return cash_ceil(cents=execution_price * shares)


def _sell_proceeds(
    price_cents: int, shares: int, liquidity_shares: int, max_impact_bps: int
) -> int:
    """Returns what a sell-side leg pays out in whole wallet units, rounded down.

    Args:
        price_cents (int): Reference quote price.
        shares (int): Size of this one leg.
        liquidity_shares (int): The company's liquidity depth.
        max_impact_bps (int): Ceiling on the per-leg impact.

    Returns:
        Cash the wallet receives, floored so the rounding never favours the seller.
    """
    execution_price = _sell_execution_price(
        price_cents=price_cents,
        shares=shares,
        liquidity_shares=liquidity_shares,
        max_impact_bps=max_impact_bps,
    )
    return cash_floor(cents=execution_price * shares)


def _max_affordable_buy_shares(
    price_cents: int,
    wallet_balance: int,
    liquidity_shares: int,
    max_impact_bps: int,
    share_cap: int,
) -> int:
    """Finds the largest buy that still fits the wallet once its own slippage is counted.

    Binary search rather than division, because the cost is not linear in size: a bigger order
    slips further, so the price the last share pays depends on how many are bought with it.

    Args:
        price_cents (int): Reference quote price.
        wallet_balance (int): Cash available for this leg.
        liquidity_shares (int): The company's liquidity depth.
        max_impact_bps (int): Ceiling on the per-leg impact.
        share_cap (int): Upper bound from float and ownership limits.

    Returns:
        The largest affordable size, zero when nothing is.
    """
    if wallet_balance <= 0 or share_cap <= 0:
        return 0
    low = 0
    high = share_cap
    while low < high:
        shares = (low + high + 1) // 2
        if (
            _buy_cost(
                price_cents=price_cents,
                shares=shares,
                liquidity_shares=liquidity_shares,
                max_impact_bps=max_impact_bps,
            )
            <= wallet_balance
        ):
            low = shares
        else:
            high = shares - 1
    return low


def _max_collateralized_short_shares(price_cents: int, wallet_balance: int, share_cap: int) -> int:
    """Returns the largest short the wallet can post collateral for.

    No search is needed here: collateral is charged at the reference price, not the slipped one, so
    the requirement is linear in size. The slipped price only decides what the short is entered at.

    Args:
        price_cents (int): Reference quote price, which is what collateral is priced at.
        wallet_balance (int): Cash available to lock up.
        share_cap (int): Upper bound from remaining borrow capacity.

    Returns:
        The largest collateralizable size, zero when nothing is.
    """
    if wallet_balance <= 0 or share_cap <= 0:
        return 0
    return min(share_cap, wallet_balance * 100 // price_cents)


def _max_coverable_short_shares(
    price_cents: int,
    wallet_balance: int,
    position: StockPositionView,
    liquidity_shares: int,
    max_impact_bps: int,
) -> int:
    """Finds how much of a short the user can afford to cover right now.

    The search has to price each candidate size whole, because covering releases collateral and
    entry value prorated by the shares covered while the cover itself slips with size — so the cash
    available and the cash needed both move with the answer. This is why a cover can succeed on a
    zero spendable balance: the released proceeds pay for it.

    Args:
        price_cents (int): Reference quote price.
        wallet_balance (int): Cash available before anything is released.
        position (StockPositionView): The short being covered, with its collateral and entry value.
        liquidity_shares (int): The company's liquidity depth.
        max_impact_bps (int): Ceiling on the per-leg impact.

    Returns:
        The largest coverable size, at most the whole short.
    """
    low = 0
    high = position.short_shares
    while low < high:
        shares = (low + high + 1) // 2
        released_collateral = _prorated_amount(
            total=position.short_collateral, shares=shares, current_shares=position.short_shares
        )
        released_entry_value = _prorated_amount(
            total=position.short_entry_value, shares=shares, current_shares=position.short_shares
        )
        cover_cost = _buy_cost(
            price_cents=price_cents,
            shares=shares,
            liquidity_shares=liquidity_shares,
            max_impact_bps=max_impact_bps,
        )
        if cover_cost <= wallet_balance + released_collateral + released_entry_value:
            low = shares
        else:
            high = shares - 1
    return low


def _individual_long_cap_shares(float_shares: int) -> int:
    """Returns the ceiling on how much of one company a single user may hold long.

    Args:
        float_shares (int): The company's tradable float.

    Returns:
        `STOCK_INDIVIDUAL_OWNERSHIP_CAP_BPS` of the float, floored.
    """
    return float_shares * STOCK_INDIVIDUAL_OWNERSHIP_CAP_BPS // STOCK_BPS_DENOMINATOR


def _available_individual_long_shares(float_shares: int, position: StockPositionView) -> int:
    """Returns the headroom a user has left under the ownership cap.

    Gates opening only. A user already over the cap (a retune can lower a float underneath them)
    reads zero here and can still sell, since risk-reducing flow is never blocked.

    Args:
        float_shares (int): The company's tradable float.
        position (StockPositionView): The user's current position.

    Returns:
        New long shares the user may still open, never negative.
    """
    return max(_individual_long_cap_shares(float_shares=float_shares) - position.long_shares, 0)


def _open_long_share_cap(snapshot: _StockExecutionSnapshot) -> int:
    """Returns the binding limit on opening long: market float or this user's own cap.

    Args:
        snapshot (_StockExecutionSnapshot): Submit-time market and position state.

    Returns:
        The smaller of the two remaining allowances.
    """
    return min(snapshot.available_long_shares, snapshot.available_individual_long_shares)


def _max_executable_quantity(snapshot: _StockExecutionSnapshot) -> int:
    """Returns the largest quantity this submission could actually execute.

    Sized the way the plan builders spend it, in two legs. A short first sells the whole long,
    which funds the collateral for whatever it then borrows; a buy first covers the whole short,
    and only the cash left after that cover is what opens a long. Sizing each leg against the other
    leg's own proceeds is what lets `ALL` flatten and reverse in one submission.

    Args:
        snapshot (_StockExecutionSnapshot): Submit-time market and position state.

    Returns:
        Total shares across both legs, which is what a numeric request is clamped to.
    """
    if snapshot.action == StockAction.SHORT:
        sell_proceeds = _sell_proceeds(
            price_cents=snapshot.price_cents,
            shares=snapshot.position.long_shares,
            liquidity_shares=snapshot.liquidity_shares,
            max_impact_bps=snapshot.max_order_impact_bps,
        )
        cash_after_selling = snapshot.wallet_balance + sell_proceeds
        short_shares = _max_collateralized_short_shares(
            price_cents=snapshot.price_cents,
            wallet_balance=cash_after_selling,
            share_cap=snapshot.available_short_shares,
        )
        return snapshot.position.long_shares + short_shares

    if snapshot.position.short_shares <= 0:
        return _max_affordable_buy_shares(
            price_cents=snapshot.price_cents,
            wallet_balance=snapshot.wallet_balance,
            liquidity_shares=snapshot.liquidity_shares,
            max_impact_bps=snapshot.max_order_impact_bps,
            share_cap=_open_long_share_cap(snapshot=snapshot),
        )

    coverable_shares = _max_coverable_short_shares(
        price_cents=snapshot.price_cents,
        wallet_balance=snapshot.wallet_balance,
        position=snapshot.position,
        liquidity_shares=snapshot.liquidity_shares,
        max_impact_bps=snapshot.max_order_impact_bps,
    )
    if coverable_shares < snapshot.position.short_shares:
        return coverable_shares
    cover_cost = _buy_cost(
        price_cents=snapshot.price_cents,
        shares=snapshot.position.short_shares,
        liquidity_shares=snapshot.liquidity_shares,
        max_impact_bps=snapshot.max_order_impact_bps,
    )
    cash_after_covering = (
        snapshot.wallet_balance
        + snapshot.position.short_collateral
        + snapshot.position.short_entry_value
        - cover_cost
    )
    return snapshot.position.short_shares + _max_affordable_buy_shares(
        price_cents=snapshot.price_cents,
        wallet_balance=max(cash_after_covering, 0),
        liquidity_shares=snapshot.liquidity_shares,
        max_impact_bps=snapshot.max_order_impact_bps,
        share_cap=_open_long_share_cap(snapshot=snapshot),
    )


def _clamp_quantity_to_available(parsed_quantity: int, snapshot: _StockExecutionSnapshot) -> int:
    """Trims an over-ambitious request down to what the market can fill.

    Asking for more than is executable is treated as asking for everything, so a user who types a
    round number gets the trade rather than a rejection. A non-positive request is passed through
    untouched, so the plan builder can reject it with its own wording.

    Args:
        parsed_quantity (int): The share count the user asked for.
        snapshot (_StockExecutionSnapshot): Submit-time market and position state.

    Returns:
        The quantity to plan, at most `_max_executable_quantity`.
    """
    if parsed_quantity <= 0:
        return parsed_quantity
    return min(parsed_quantity, _max_executable_quantity(snapshot=snapshot))


def _max_quantity_error(snapshot: _StockExecutionSnapshot) -> str:
    """Names the reason nothing is executable.

    The market limits are tested before the balance one, so a user blocked by the ownership cap or
    by exhausted float is not told to add money instead.

    Args:
        snapshot (_StockExecutionSnapshot): Submit-time market and position state.

    Returns:
        The user-facing Traditional Chinese failure line.
    """
    if snapshot.action == StockAction.BUY:
        if snapshot.position.short_shares <= 0 and snapshot.available_individual_long_shares <= 0:
            return "單一玩家持股上限為 49%，目前無法再買入這檔股票"
        if snapshot.position.short_shares <= 0 and snapshot.available_long_shares <= 0:
            return "目前沒有可買入的流通股"
        return "餘額不足，無法買入或回補股票"
    if snapshot.position.long_shares <= 0 and snapshot.available_short_shares <= 0:
        return "目前沒有可借券做空的股數"
    return "餘額不足，無法賣出或建立做空部位"


def _leg_view(  # noqa: PLR0913 -- trade leg fields mirror the persisted audit row
    operation_id: str,
    leg_order: int,
    symbol: str,
    user_id: int,
    leg_type: StockTradeLegType,
    shares: int,
    price_cents: int,
    wallet_delta: int,
    basis_delta: int,
    collateral_delta: int,
    realized_pnl_delta: int,
    now: datetime,
) -> StockTradeLegView:
    """Records one planned leg, before anything is written.

    The name is left empty here and stamped onto the whole plan once settlement knows it, so the
    plan builders never need the caller's Discord identity.

    Args:
        operation_id (str): Identifier of the operation this leg belongs to.
        leg_order (int): Position within the operation, 1-based.
        symbol (str): Ticker symbol traded.
        user_id (int): Trader the leg belongs to.
        leg_type (StockTradeLegType): Which of the four atomic movements this is.
        shares (int): Shares moved by this leg.
        price_cents (int): This leg's own slipped execution price.
        wallet_delta (int): Net cash the leg moves, before the gross expansion.
        basis_delta (int): Change to long cost basis, or to short entry value.
        collateral_delta (int): Change to short collateral.
        realized_pnl_delta (int): Profit or loss the leg realizes.
        now (datetime): Creation stamp shared by every leg of the operation.

    Returns:
        The leg as a view, which is also what gets persisted.
    """
    return StockTradeLegView(
        operation_id=operation_id,
        leg_order=leg_order,
        symbol=symbol,
        user_id=user_id,
        leg_type=leg_type,
        shares=shares,
        price_cents=price_cents,
        wallet_delta=wallet_delta,
        basis_delta=basis_delta,
        collateral_delta=collateral_delta,
        realized_pnl_delta=realized_pnl_delta,
        created_at=now,
    )


def _average_leg_price(legs: tuple[StockTradeLegView, ...], fallback_price_cents: int) -> int:
    """Blends the legs' prices into the one figure a settlement summary shows.

    Summary only: each leg keeps and settles at its own price, so this is never what anything is
    charged at.

    Args:
        legs (tuple[StockTradeLegView, ...]): The operation's legs.
        fallback_price_cents (int): Price to report when the legs moved no shares.

    Returns:
        The share-weighted average execution price in cents.
    """
    total_shares = sum(leg.shares for leg in legs)
    if total_shares <= 0:
        return fallback_price_cents
    return sum(leg.price_cents * leg.shares for leg in legs) // total_shares


def _insufficient_result(  # noqa: PLR0913 -- failed results preserve the submit-time context
    symbol: str,
    action: StockAction,
    quantity: int,
    price_cents: int,
    balance: int,
    position: StockPositionView,
    error: str,
) -> StockSettlementResult:
    """Refuses a submission before anything is written.

    Carries no `operation_id` and no `status`, which is how a caller tells a rejection apart from a
    trade that started and stopped part way; the position echoed back is the one the user already
    had, unchanged.

    Args:
        symbol (str): Ticker symbol the user submitted against.
        action (StockAction): Which direction was submitted.
        quantity (int): Shares asked for, reported as zero when negative.
        price_cents (int): Quote the refusal was judged against.
        balance (int): Wallet balance at submit time, unchanged by this.
        position (StockPositionView): The user's position, unchanged by this.
        error (str): User-facing Traditional Chinese reason.

    Returns:
        A failed settlement result.
    """
    return StockSettlementResult(
        success=False,
        operation_id=None,
        symbol=symbol,
        requested_action=action,
        shares=max(quantity, 0),
        price_cents=price_cents,
        wallet_delta=0,
        balance_after=balance,
        position=position,
        legs=(),
        error=error,
    )


def _build_plan(  # noqa: PLR0913 -- settlement plan needs the current wallet and position snapshot
    operation_id: str,
    symbol: str,
    user_id: int,
    action: StockAction,
    quantity: int,
    price_cents: int,
    liquidity_shares: int,
    max_order_impact_bps: int,
    wallet_balance: int,
    position: StockPositionView,
    available_long_shares: int,
    available_short_shares: int,
    available_individual_long_shares: int,
    now: datetime,
) -> _StockOperationPlan | StockSettlementResult:
    """Turns a validated request into the ordered legs that will settle it.

    Pure: reads nothing and writes nothing, so a plan is entirely decided by the snapshot handed
    in. Dispatches on direction after rejecting a non-positive quantity, which is the one check the
    two directions share.

    Args:
        operation_id (str): Identifier to stamp on the operation and every leg.
        symbol (str): Ticker symbol traded.
        user_id (int): Trader submitting.
        action (StockAction): Which direction was submitted.
        quantity (int): Shares to plan, already clamped to what is executable.
        price_cents (int): Reference quote price.
        liquidity_shares (int): The company's liquidity depth.
        max_order_impact_bps (int): Ceiling on the per-leg impact.
        wallet_balance (int): Cash available at submit time.
        position (StockPositionView): The user's position before this operation.
        available_long_shares (int): Float still openable as long, market-wide.
        available_short_shares (int): Float still borrowable for shorting.
        available_individual_long_shares (int): This user's remaining ownership-cap headroom.
        now (datetime): Creation stamp for the legs.

    Returns:
        The plan, or a failed result carrying the reason it could not be built.
    """
    if quantity <= 0:
        return _insufficient_result(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price_cents=price_cents,
            balance=wallet_balance,
            position=position,
            error="股數必須是正整數",
        )
    if action == StockAction.BUY:
        return _build_buy_plan(
            operation_id=operation_id,
            symbol=symbol,
            user_id=user_id,
            quantity=quantity,
            price_cents=price_cents,
            liquidity_shares=liquidity_shares,
            max_order_impact_bps=max_order_impact_bps,
            wallet_balance=wallet_balance,
            position=position,
            available_long_shares=available_long_shares,
            available_individual_long_shares=available_individual_long_shares,
            now=now,
        )
    return _build_short_plan(
        operation_id=operation_id,
        symbol=symbol,
        user_id=user_id,
        quantity=quantity,
        price_cents=price_cents,
        liquidity_shares=liquidity_shares,
        max_order_impact_bps=max_order_impact_bps,
        wallet_balance=wallet_balance,
        position=position,
        available_short_shares=available_short_shares,
        now=now,
    )


def _build_buy_plan(  # noqa: PLR0913 -- buy can cover short and open long in order
    operation_id: str,
    symbol: str,
    user_id: int,
    quantity: int,
    price_cents: int,
    liquidity_shares: int,
    max_order_impact_bps: int,
    wallet_balance: int,
    position: StockPositionView,
    available_long_shares: int,
    available_individual_long_shares: int,
    now: datetime,
) -> _StockOperationPlan | StockSettlementResult:
    """Plans a buy: cover any open short first, then open long with whatever is left.

    The order is what makes one submission both close and open, and it is also what funds the
    close: the cover is affordable against the collateral and entry value it releases, so a user
    with no spendable cash can still cover. Each leg's own share count decides its slippage, so a
    cover-plus-open pays two different prices, and the running wallet total is threaded through so
    the open is judged against the cash the cover produced.

    Rejects rather than trims — a quantity past the ownership cap, past the remaining float, or
    past the wallet comes back as a failed result, because `_clamp_quantity_to_available` has
    already had its chance to size the request down.

    Args:
        operation_id (str): Identifier to stamp on the operation and every leg.
        symbol (str): Ticker symbol traded.
        user_id (int): Trader submitting.
        quantity (int): Shares to plan across both legs.
        price_cents (int): Reference quote price.
        liquidity_shares (int): The company's liquidity depth.
        max_order_impact_bps (int): Ceiling on the per-leg impact.
        wallet_balance (int): Cash available at submit time.
        position (StockPositionView): The user's position before this operation.
        available_long_shares (int): Float still openable as long, market-wide.
        available_individual_long_shares (int): This user's remaining ownership-cap headroom.
        now (datetime): Creation stamp for the legs.

    Returns:
        The plan with its ordered legs and resulting position, or a failed result.
    """
    long_shares = position.long_shares
    long_cost_basis = position.long_cost_basis
    short_shares = position.short_shares
    short_entry_value = position.short_entry_value
    short_collateral = position.short_collateral
    realized_pnl = position.realized_pnl
    remaining = quantity
    wallet_delta_total = 0
    legs: list[StockTradeLegView] = []

    if short_shares > 0 and remaining > 0:
        cover_shares = min(remaining, short_shares)
        released_collateral = _prorated_amount(
            total=short_collateral, shares=cover_shares, current_shares=short_shares
        )
        released_entry_value = _prorated_amount(
            total=short_entry_value, shares=cover_shares, current_shares=short_shares
        )
        cover_price_cents = _buy_execution_price(
            price_cents=price_cents,
            shares=cover_shares,
            liquidity_shares=liquidity_shares,
            max_impact_bps=max_order_impact_bps,
        )
        cover_cost = cash_ceil(cents=cover_price_cents * cover_shares)
        if (
            cover_cost
            > released_collateral + released_entry_value + wallet_balance + wallet_delta_total
        ):
            return _insufficient_result(
                symbol=symbol,
                action=StockAction.BUY,
                quantity=quantity,
                price_cents=price_cents,
                balance=wallet_balance,
                position=position,
                error="餘額不足，無法回補做空",
            )
        realized = released_entry_value - cover_cost
        wallet_delta = released_collateral + realized
        legs.append(
            _leg_view(
                operation_id=operation_id,
                leg_order=len(legs) + 1,
                symbol=symbol,
                user_id=user_id,
                leg_type=StockTradeLegType.COVER_SHORT,
                shares=cover_shares,
                price_cents=cover_price_cents,
                wallet_delta=wallet_delta,
                basis_delta=-released_entry_value,
                collateral_delta=-released_collateral,
                realized_pnl_delta=realized,
                now=now,
            )
        )
        short_shares -= cover_shares
        short_entry_value -= released_entry_value
        short_collateral -= released_collateral
        realized_pnl += realized
        wallet_delta_total += wallet_delta
        remaining -= cover_shares

    if remaining > 0:
        if remaining > available_individual_long_shares:
            return _insufficient_result(
                symbol=symbol,
                action=StockAction.BUY,
                quantity=quantity,
                price_cents=price_cents,
                balance=wallet_balance,
                position=position,
                error="單一玩家持股上限為 49%，目前無法再買入這檔股票",
            )
        if remaining > available_long_shares:
            return _insufficient_result(
                symbol=symbol,
                action=StockAction.BUY,
                quantity=quantity,
                price_cents=price_cents,
                balance=wallet_balance,
                position=position,
                error=f"目前可買入流通股只剩 {share_quantity_text(shares=available_long_shares)}",
            )
        open_price_cents = _buy_execution_price(
            price_cents=price_cents,
            shares=remaining,
            liquidity_shares=liquidity_shares,
            max_impact_bps=max_order_impact_bps,
        )
        cost = cash_ceil(cents=open_price_cents * remaining)
        if cost > wallet_balance + wallet_delta_total:
            return _insufficient_result(
                symbol=symbol,
                action=StockAction.BUY,
                quantity=quantity,
                price_cents=price_cents,
                balance=wallet_balance,
                position=position,
                error=f"餘額不足，需要 {cost:,} 才能買入 {share_quantity_text(shares=remaining)}",
            )
        legs.append(
            _leg_view(
                operation_id=operation_id,
                leg_order=len(legs) + 1,
                symbol=symbol,
                user_id=user_id,
                leg_type=StockTradeLegType.OPEN_LONG,
                shares=remaining,
                price_cents=open_price_cents,
                wallet_delta=-cost,
                basis_delta=cost,
                collateral_delta=0,
                realized_pnl_delta=0,
                now=now,
            )
        )
        long_shares += remaining
        long_cost_basis += cost
        wallet_delta_total -= cost

    final_position = StockPositionView(
        symbol=symbol,
        user_id=user_id,
        long_shares=long_shares,
        long_cost_basis=long_cost_basis,
        short_shares=short_shares,
        short_entry_value=short_entry_value,
        short_collateral=short_collateral,
        realized_pnl=realized_pnl,
    )
    return _StockOperationPlan(
        success=True,
        operation_id=operation_id,
        symbol=symbol,
        requested_action=StockAction.BUY,
        shares=quantity,
        price_cents=_average_leg_price(legs=tuple(legs), fallback_price_cents=price_cents),
        wallet_delta=wallet_delta_total,
        balance_after=wallet_balance + wallet_delta_total,
        position=final_position,
        legs=tuple(legs),
        status=StockOperationStatus.PENDING,
    )


def _build_short_plan(  # noqa: PLR0913 -- short can sell long and open short in order
    operation_id: str,
    symbol: str,
    user_id: int,
    quantity: int,
    price_cents: int,
    liquidity_shares: int,
    max_order_impact_bps: int,
    wallet_balance: int,
    position: StockPositionView,
    available_short_shares: int,
    now: datetime,
) -> _StockOperationPlan | StockSettlementResult:
    """Plans a short: sell any long first, then borrow and open short with the rest.

    The mirror of the buy plan, with one asymmetry that matters. Collateral is charged at the
    reference price while the short is entered at the slipped one, so a short locks up slightly
    more than it books as entry value; the difference comes back on the cover.

    Args:
        operation_id (str): Identifier to stamp on the operation and every leg.
        symbol (str): Ticker symbol traded.
        user_id (int): Trader submitting.
        quantity (int): Shares to plan across both legs.
        price_cents (int): Reference quote price.
        liquidity_shares (int): The company's liquidity depth.
        max_order_impact_bps (int): Ceiling on the per-leg impact.
        wallet_balance (int): Cash available at submit time.
        position (StockPositionView): The user's position before this operation.
        available_short_shares (int): Float still borrowable for shorting.
        now (datetime): Creation stamp for the legs.

    Returns:
        The plan with its ordered legs and resulting position, or a failed result.
    """
    long_shares = position.long_shares
    long_cost_basis = position.long_cost_basis
    short_shares = position.short_shares
    short_entry_value = position.short_entry_value
    short_collateral = position.short_collateral
    realized_pnl = position.realized_pnl
    remaining = quantity
    wallet_delta_total = 0
    legs: list[StockTradeLegView] = []

    if long_shares > 0 and remaining > 0:
        sell_shares = min(remaining, long_shares)
        released_basis = _prorated_amount(
            total=long_cost_basis, shares=sell_shares, current_shares=long_shares
        )
        sell_price_cents = _sell_execution_price(
            price_cents=price_cents,
            shares=sell_shares,
            liquidity_shares=liquidity_shares,
            max_impact_bps=max_order_impact_bps,
        )
        proceeds = cash_floor(cents=sell_price_cents * sell_shares)
        realized = proceeds - released_basis
        legs.append(
            _leg_view(
                operation_id=operation_id,
                leg_order=len(legs) + 1,
                symbol=symbol,
                user_id=user_id,
                leg_type=StockTradeLegType.SELL_LONG,
                shares=sell_shares,
                price_cents=sell_price_cents,
                wallet_delta=proceeds,
                basis_delta=-released_basis,
                collateral_delta=0,
                realized_pnl_delta=realized,
                now=now,
            )
        )
        long_shares -= sell_shares
        long_cost_basis -= released_basis
        realized_pnl += realized
        wallet_delta_total += proceeds
        remaining -= sell_shares

    if remaining > 0:
        if remaining > available_short_shares:
            return _insufficient_result(
                symbol=symbol,
                action=StockAction.SHORT,
                quantity=quantity,
                price_cents=price_cents,
                balance=wallet_balance,
                position=position,
                error=f"目前可借券做空股數只剩 {share_quantity_text(shares=available_short_shares)}",
            )
        collateral = cash_ceil(cents=price_cents * remaining)
        if collateral > wallet_balance + wallet_delta_total:
            return _insufficient_result(
                symbol=symbol,
                action=StockAction.SHORT,
                quantity=quantity,
                price_cents=price_cents,
                balance=wallet_balance,
                position=position,
                error=f"餘額不足，需要 {collateral:,} 作為做空擔保金",
            )
        short_price_cents = _sell_execution_price(
            price_cents=price_cents,
            shares=remaining,
            liquidity_shares=liquidity_shares,
            max_impact_bps=max_order_impact_bps,
        )
        entry_value = cash_floor(cents=short_price_cents * remaining)
        legs.append(
            _leg_view(
                operation_id=operation_id,
                leg_order=len(legs) + 1,
                symbol=symbol,
                user_id=user_id,
                leg_type=StockTradeLegType.OPEN_SHORT,
                shares=remaining,
                price_cents=short_price_cents,
                wallet_delta=-collateral,
                basis_delta=entry_value,
                collateral_delta=collateral,
                realized_pnl_delta=0,
                now=now,
            )
        )
        short_shares += remaining
        short_entry_value += entry_value
        short_collateral += collateral
        wallet_delta_total -= collateral

    final_position = StockPositionView(
        symbol=symbol,
        user_id=user_id,
        long_shares=long_shares,
        long_cost_basis=long_cost_basis,
        short_shares=short_shares,
        short_entry_value=short_entry_value,
        short_collateral=short_collateral,
        realized_pnl=realized_pnl,
    )
    return _StockOperationPlan(
        success=True,
        operation_id=operation_id,
        symbol=symbol,
        requested_action=StockAction.SHORT,
        shares=quantity,
        price_cents=_average_leg_price(legs=tuple(legs), fallback_price_cents=price_cents),
        wallet_delta=wallet_delta_total,
        balance_after=wallet_balance + wallet_delta_total,
        position=final_position,
        legs=tuple(legs),
        status=StockOperationStatus.PENDING,
    )


async def _blocking_operation(
    session: AsyncSession, symbol: str, user_id: int
) -> StockOperation | None:
    """Finds an unresolved operation standing in the way of a new one.

    Scoped to one user and one symbol: a stuck trade freezes only that pair, so nobody else and no
    other company is affected. Oldest first, so the reported operation is the one to clear.

    Args:
        session (AsyncSession): Open session, already inside the caller's immediate transaction.
        symbol (str): Ticker symbol being submitted against.
        user_id (int): Trader submitting.

    Returns:
        The oldest operation outside `_FINAL_OPERATION_STATUSES`, or None when the pair is clear.
    """
    result = await session.execute(
        statement=select(StockOperation)
        .where(
            StockOperation.symbol == symbol,
            StockOperation.user_id == user_id,
            StockOperation.status.notin_(_FINAL_OPERATION_STATUSES),
        )
        .order_by(StockOperation.created_at.asc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _blocked_operation_result(
    operation: StockOperation, action: StockAction, balance: int, position: StockPositionView
) -> StockSettlementResult:
    """Refuses a submission because an earlier one is still unresolved.

    The one failure whose `operation_id` and `status` name an operation other than this request, so
    the user can quote the blocker to an operator instead of just being told no.

    Args:
        operation (StockOperation): The unresolved operation standing in the way.
        action (StockAction): The direction this refused submission asked for.
        balance (int): Wallet balance at submit time, unchanged by this.
        position (StockPositionView): The user's position, unchanged by this.

    Returns:
        A failed settlement result naming the blocking operation and its status.
    """
    status = StockOperationStatus(operation.status)
    return StockSettlementResult(
        success=False,
        operation_id=operation.operation_id,
        symbol=operation.symbol,
        requested_action=action,
        shares=0,
        price_cents=0,
        wallet_delta=0,
        balance_after=balance,
        position=position,
        legs=(),
        status=status,
        error=(
            "仍有未完成的股票交易需要人工確認，"
            f"操作代碼={operation.operation_id}，狀態={status.value}"
        ),
    )


def _wallet_delta_legs_for_plan(plan: _StockOperationPlan) -> tuple[WalletDeltaLeg, ...]:
    """Expands the plan's legs into the ordered gross movements the wallet applies.

    Every leg but a cover moves its net cash once. A cover is split into three — the collateral it
    releases, the short entry value it releases, then the cover cost as a debit — and the order is
    load-bearing: the economy side rejects a debit it cannot cover in full at that point in the
    sequence, so the two credits have to land first or a cover funded by its own proceeds would be
    refused. Splitting also keeps the economy's `total_earned - total_spent == balance` invariant
    describing the real flow rather than the netted remainder. A zero movement is dropped, since it
    would only add a no-op leg to the ledger.

    Args:
        plan (_StockOperationPlan): The accepted plan whose legs to expand.

    Returns:
        The wallet legs in application order, each carrying a `stock:<operation>:<leg>` reason.
    """
    deltas: list[WalletDeltaLeg] = []
    for leg in plan.legs:
        reason_prefix = f"stock:{plan.operation_id}:{leg.leg_order}"
        if leg.leg_type != StockTradeLegType.COVER_SHORT:
            if leg.wallet_delta != 0:
                deltas.append(WalletDeltaLeg(delta=leg.wallet_delta, reason=reason_prefix))
            continue

        released_collateral = -leg.collateral_delta
        released_entry_value = -leg.basis_delta
        cover_cost = released_entry_value - leg.realized_pnl_delta
        if released_collateral:
            deltas.append(
                WalletDeltaLeg(delta=released_collateral, reason=f"{reason_prefix}:collateral")
            )
        if released_entry_value:
            deltas.append(
                WalletDeltaLeg(delta=released_entry_value, reason=f"{reason_prefix}:short_entry")
            )
        if cover_cost:
            deltas.append(WalletDeltaLeg(delta=-cover_cost, reason=f"{reason_prefix}:cover"))
    return tuple(deltas)


async def _build_submit_time_operation_plan(  # noqa: PLR0913 -- submit-time planning needs every locked snapshot input
    session: AsyncSession,
    normalized_symbol: str,
    operation_id: str,
    user_id: int,
    requested_action: StockAction,
    quantity: str,
    wallet_balance: int,
    position: StockPositionView,
    effective_now: datetime,
    rng: Random | None,
) -> _StockOperationPlan | StockSettlementResult:
    """Advances the market, then plans the request against the state that advance produced.

    Runs inside settlement's own immediate transaction, so the quote it plans against is the same
    one the trade will be written under; the advance is told not to begin a transaction of its own.
    Quantity handling happens here rather than in the plan builders because it needs the market:
    a bad quantity string is a format failure, and an over-large one is trimmed to what the
    snapshot can execute before a failure is even considered.

    Args:
        session (AsyncSession): Open session inside the caller's immediate transaction.
        normalized_symbol (str): Upper-cased ticker symbol.
        operation_id (str): Identifier to stamp on the operation and every leg.
        user_id (int): Trader submitting.
        requested_action (StockAction): Which direction was submitted.
        quantity (str): The raw modal text, still unparsed.
        wallet_balance (int): Cash read before the transaction opened.
        position (StockPositionView): The user's position, read under the same lock.
        effective_now (datetime): Advance target and creation stamp.
        rng (Random | None): Volatility source, defaulting to `SystemRandom`.

    Returns:
        The plan, or a failed result carrying the reason.

    Raises:
        ValueError: The profile vanished between the advance and the plan.
    """
    quote = await advance_market_in_session(
        session=session,
        symbol=normalized_symbol,
        now=effective_now,
        rng=rng,
        begin_immediate=False,
    )
    profile = await session.get(entity=StockProfile, ident=normalized_symbol)
    if profile is None:
        msg = f"Unknown stock symbol: {normalized_symbol}"
        raise ValueError(msg)
    exposure = await _market_exposure(session=session, profile=profile)
    available_individual_long_shares = _available_individual_long_shares(
        float_shares=profile.float_shares, position=position
    )
    try:
        parsed_quantity = _parse_quantity(
            raw_quantity=quantity,
            action=requested_action,
            price_cents=quote.profile.price_cents,
            wallet_balance=wallet_balance,
            position=position,
        )
    except ValueError:
        return _insufficient_result(
            symbol=normalized_symbol,
            action=requested_action,
            quantity=0,
            price_cents=quote.profile.price_cents,
            balance=wallet_balance,
            position=position,
            error="股數格式錯誤，請輸入正整數或 ALL",
        )
    snapshot = _StockExecutionSnapshot(
        action=requested_action,
        price_cents=quote.profile.price_cents,
        liquidity_shares=quote.profile.liquidity_shares,
        max_order_impact_bps=quote.profile.max_tick_change_bps,
        wallet_balance=wallet_balance,
        position=position,
        available_long_shares=exposure.available_long_shares,
        available_short_shares=exposure.available_short_shares,
        available_individual_long_shares=available_individual_long_shares,
    )
    requested_quantity = parsed_quantity
    parsed_quantity = _clamp_quantity_to_available(
        parsed_quantity=parsed_quantity, snapshot=snapshot
    )
    if (
        requested_quantity > 0 or _is_all_quantity(raw_quantity=quantity)
    ) and parsed_quantity <= 0:
        return _insufficient_result(
            symbol=normalized_symbol,
            action=requested_action,
            quantity=parsed_quantity,
            price_cents=quote.profile.price_cents,
            balance=wallet_balance,
            position=position,
            error=_max_quantity_error(snapshot=snapshot),
        )

    return _build_plan(
        operation_id=operation_id,
        symbol=normalized_symbol,
        user_id=user_id,
        action=requested_action,
        quantity=parsed_quantity,
        price_cents=quote.profile.price_cents,
        liquidity_shares=quote.profile.liquidity_shares,
        max_order_impact_bps=quote.profile.max_tick_change_bps,
        wallet_balance=wallet_balance,
        position=position,
        available_long_shares=exposure.available_long_shares,
        available_short_shares=exposure.available_short_shares,
        available_individual_long_shares=available_individual_long_shares,
        now=effective_now,
    )


async def settle_stock_operation(  # noqa: PLR0913 -- Service boundary returns typed validation and lifecycle failures directly
    symbol: str,
    user_id: int,
    user_name: str,
    requested_action: StockAction,
    quantity: str,
    avatar_url: str = "",
    now: datetime | None = None,
    rng: Random | None = None,
) -> StockSettlementResult:
    """Settles one buy/cover or short/sell request across both databases.

    The only way a position or a stock-side wallet movement is ever written, and the whole
    two-database dance lives here. Under the user's operation lock and the symbol's market lock, in
    order: refuse if an earlier operation is still unresolved, plan against a freshly advanced
    market, commit the PENDING operation and its legs, apply the ordered wallet legs, mark
    WALLET_APPLIED, write the position and mark APPLIED.

    Nothing rolls back and nothing retries. A wallet rejection is the clean failure — the wallet
    never moved and the operation is marked FAILED, releasing the float its legs reserved. A wallet
    that raises, or a stock-side write that fails after the wallet moved, is the dangerous one, and
    it parks the operation at RECONCILE_REQUIRED: an operator has to look, and until they do, that
    user is blocked on that symbol. Cancellation is treated as the same kind of unknown, with the
    marking shielded so the record survives the cancel before it is re-raised.

    Validation and lifecycle failures come back as a result rather than an exception, which is what
    this function's noqa claims and all it claims: an unknown symbol still raises `ValueError`
    before anything is written.

    Args:
        symbol (str): Ticker symbol, upper-cased here.
        user_id (int): Trader submitting.
        user_name (str): Display name, stamped onto the plan and stored with the rows.
        requested_action (StockAction): Which direction was submitted.
        quantity (str): The raw modal text, parsed under the lock.
        avatar_url (str): Last-seen avatar to refresh on the wallet row.
        now (datetime | None): Advance target and creation stamp, defaulting to now.
        rng (Random | None): Volatility source, defaulting to `SystemRandom`.

    Returns:
        The settled result on success, or a failed one carrying the user-facing reason and, where
        an operation was written, its id and lifecycle status.

    Raises:
        asyncio.CancelledError: Re-raised after the operation has been marked RECONCILE_REQUIRED.
    """
    normalized_symbol = symbol.upper()
    await _ensure_schema()
    async with _operation_lock(user_id=user_id, symbol=normalized_symbol):
        wallet_balance = await get_balance(user_id=user_id)
        effective_now = now or _database_now()
        operation_id = str(uuid.uuid4())
        async with open_stock_session() as session, _market_lock(symbol=normalized_symbol):
            await _begin_immediate(session=session)
            blocking_operation = await _blocking_operation(
                session=session, symbol=normalized_symbol, user_id=user_id
            )
            position = await _get_position_view(
                session=session, symbol=normalized_symbol, user_id=user_id
            )
            if blocking_operation is not None:
                return _blocked_operation_result(
                    operation=blocking_operation,
                    action=requested_action,
                    balance=wallet_balance,
                    position=position,
                )
            plan = await _build_submit_time_operation_plan(
                session=session,
                normalized_symbol=normalized_symbol,
                operation_id=operation_id,
                user_id=user_id,
                requested_action=requested_action,
                quantity=quantity,
                wallet_balance=wallet_balance,
                position=position,
                effective_now=effective_now,
                rng=rng,
            )
            # success=True is only ever built as a _StockOperationPlan; the
            # isinstance check makes that discrimination visible to type checkers.
            if not plan.success or not isinstance(plan, _StockOperationPlan):
                await session.rollback()
                return plan
            plan = plan.model_copy(
                update={
                    "position": plan.position.model_copy(update={"user_name": user_name}),
                    "legs": tuple(
                        leg.model_copy(update={"user_name": user_name}) for leg in plan.legs
                    ),
                }
            )
            await _commit_pending_operation(session=session, plan=plan, now=effective_now)

        try:
            wallet_result = await apply_ordered_wallet_deltas(
                user_id=user_id,
                name=user_name,
                avatar_url=avatar_url,
                deltas=_wallet_delta_legs_for_plan(plan=plan),
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                _mark_operation(
                    operation_id=operation_id,
                    status=StockOperationStatus.RECONCILE_REQUIRED,
                    failure_reason="wallet delta cancelled after stock operation was planned",
                )
            )
            raise
        # Broad on purpose: any wallet failure must still flip the committed operation to
        # RECONCILE_REQUIRED instead of escaping with the two databases out of step.
        except Exception as exc:
            logfire.error(
                "Stock wallet delta failed after operation was planned; manual reconciliation required",
                operation_id=operation_id,
                user_id=user_id,
                symbol=plan.symbol,
                requested_action=plan.requested_action,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            await _mark_operation(
                operation_id=operation_id,
                status=StockOperationStatus.RECONCILE_REQUIRED,
                failure_reason=f"wallet delta raised after stock operation was planned: {type(exc).__name__}",
            )
            return plan.model_copy(
                update={
                    "success": False,
                    "status": StockOperationStatus.RECONCILE_REQUIRED,
                    "error": f"交易狀態需要人工對帳，操作代碼={operation_id}",
                }
            )
        if wallet_result is None:
            await _mark_operation(
                operation_id=operation_id,
                status=StockOperationStatus.FAILED,
                failure_reason="wallet delta rejected before stock position was applied",
            )
            return plan.model_copy(
                update={
                    "success": False,
                    "status": StockOperationStatus.FAILED,
                    "error": "交易未完成，送出時餘額已不足，沒有變更股票部位",
                }
            )

        await _mark_operation(
            operation_id=operation_id,
            status=StockOperationStatus.WALLET_APPLIED,
            failure_reason="",
        )
        try:
            await _finalize_stock_side(plan=plan, now=effective_now)
        except asyncio.CancelledError:
            await asyncio.shield(
                _mark_operation(
                    operation_id=operation_id,
                    status=StockOperationStatus.RECONCILE_REQUIRED,
                    failure_reason="stock finalization cancelled after wallet side was applied",
                )
            )
            raise
        # Broad on purpose: any finalization failure must still flip the operation to
        # RECONCILE_REQUIRED instead of escaping with the wallet already moved.
        except Exception as exc:
            logfire.error(
                "Stock finalization failed after wallet was applied; manual reconciliation required",
                operation_id=operation_id,
                user_id=user_id,
                symbol=plan.symbol,
                requested_action=plan.requested_action,
                wallet_delta=plan.wallet_delta,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            await _mark_operation(
                operation_id=operation_id,
                status=StockOperationStatus.RECONCILE_REQUIRED,
                failure_reason=f"stock finalization failed after wallet side was applied: {type(exc).__name__}",
            )
            return plan.model_copy(
                update={
                    "success": False,
                    "status": StockOperationStatus.RECONCILE_REQUIRED,
                    "error": f"交易狀態需要人工對帳，操作代碼={operation_id}",
                }
            )
        return plan.model_copy(
            update={
                "status": StockOperationStatus.APPLIED,
                "balance_after": wallet_result.new_balance,
            }
        )


async def _commit_pending_operation(
    session: AsyncSession, plan: _StockOperationPlan, now: datetime
) -> None:
    """Writes the operation and its legs as PENDING, and commits before any money moves.

    Committing this early is what makes the rest recoverable: from here on there is a durable
    record of what was intended, its opens reserve float against other traders, and a crash leaves
    something an operator can read rather than a silent gap.

    Args:
        session (AsyncSession): Open session inside the caller's immediate transaction.
        plan (_StockOperationPlan): The accepted plan to persist.
        now (datetime): Creation stamp for the operation row.
    """
    session.add(
        instance=StockOperation(
            operation_id=plan.operation_id or "",
            symbol=plan.symbol,
            user_id=plan.position.user_id,
            user_name=plan.position.user_name,
            requested_action=plan.requested_action.value,
            status=StockOperationStatus.PENDING.value,
            failure_reason="",
            created_at=now,
            updated_at=now,
        )
    )
    for leg in plan.legs:
        session.add(
            instance=StockTradeLeg(
                operation_id=leg.operation_id,
                leg_order=leg.leg_order,
                symbol=leg.symbol,
                user_id=leg.user_id,
                user_name=leg.user_name,
                leg_type=leg.leg_type.value,
                shares=leg.shares,
                price_cents=leg.price_cents,
                wallet_delta=leg.wallet_delta,
                basis_delta=leg.basis_delta,
                collateral_delta=leg.collateral_delta,
                realized_pnl_delta=leg.realized_pnl_delta,
                created_at=leg.created_at,
            )
        )
    await session.flush()
    await session.commit()


async def _finalize_stock_side(plan: _StockOperationPlan, now: datetime) -> None:
    """Writes the resulting position and closes the operation out as APPLIED.

    The last step, run only once the wallet has moved. Position and status commit together, so the
    operation can never read as finished while the position it describes is missing. Drops the
    owner's cached portfolio, since their valuation just changed.

    Args:
        plan (_StockOperationPlan): The plan whose position to apply.
        now (datetime): Update stamp for the position and the operation.
    """
    async with open_stock_session() as session:
        await _write_position(session=session, position=plan.position, now=now)
        await session.execute(
            statement=update(StockOperation)
            .where(StockOperation.operation_id == plan.operation_id)
            .values(status=StockOperationStatus.APPLIED.value, updated_at=now)
        )
        await session.commit()
    invalidate_stock_portfolio_cache(user_id=plan.position.user_id)


async def _write_position(
    session: AsyncSession, position: StockPositionView, now: datetime
) -> None:
    """Writes a user's position for a symbol, creating the row if it is their first trade.

    Stores the whole position rather than a delta, because the plan already computed the end state
    from the position it read under the same lock.

    Args:
        session (AsyncSession): Open session; the caller commits.
        position (StockPositionView): The end state to store.
        now (datetime): Update stamp.
    """
    await session.execute(
        statement=insert(StockPosition)
        .values(
            symbol=position.symbol,
            user_id=position.user_id,
            user_name=position.user_name,
            long_shares=position.long_shares,
            long_cost_basis=position.long_cost_basis,
            short_shares=position.short_shares,
            short_entry_value=position.short_entry_value,
            short_collateral=position.short_collateral,
            realized_pnl=position.realized_pnl,
            version=1,
            updated_at=now,
        )
        .on_conflict_do_update(
            index_elements=["symbol", "user_id"],
            set_={
                "long_shares": position.long_shares,
                "user_name": position.user_name,
                "long_cost_basis": position.long_cost_basis,
                "short_shares": position.short_shares,
                "short_entry_value": position.short_entry_value,
                "short_collateral": position.short_collateral,
                "realized_pnl": position.realized_pnl,
                "version": StockPosition.version + 1,
                "updated_at": now,
            },
        )
    )


async def _mark_operation(
    operation_id: str, status: StockOperationStatus, failure_reason: str
) -> None:
    """Moves an operation to its next lifecycle status, in its own transaction.

    Deliberately separate from whatever it is reporting on: the failure paths call this after their
    own session is gone, and a cancelled settlement shields the call so the record outlives the
    cancel.

    Args:
        operation_id (str): The operation to update.
        status (StockOperationStatus): The status to move it to.
        failure_reason (str): Operator-facing note, empty on a healthy step.
    """
    await _ensure_schema()
    async with open_stock_session() as session:
        await session.execute(
            statement=update(StockOperation)
            .where(StockOperation.operation_id == operation_id)
            .values(status=status.value, failure_reason=failure_reason, updated_at=_database_now())
        )
        await session.commit()


async def list_reconciliation_operations() -> tuple[StockReconciliationOperation, ...]:
    """Lists every operation stuck short of a final state, with its legs, for an operator.

    Nothing repairs these automatically: the legs say what the stock side intended and
    `failure_reason` says where it stopped, but only a human can tell whether the wallet moved. An
    operation with no legs at all still appears, since a plan that got no further is exactly the
    kind that needs looking at.

    Returns:
        The unresolved operations oldest first, each with its legs in `leg_order`.
    """
    await _ensure_schema()
    async with open_stock_session() as session:
        result = await session.execute(
            statement=select(StockOperation, StockTradeLeg)
            .outerjoin(StockTradeLeg, StockTradeLeg.operation_id == StockOperation.operation_id)
            .where(StockOperation.status.notin_(_FINAL_OPERATION_STATUSES))
            .order_by(StockOperation.created_at.asc(), StockTradeLeg.leg_order.asc())
        )
        operations: list[StockOperation] = []
        legs_by_operation: dict[str, list[StockTradeLeg]] = {}
        for operation, leg in result.all():
            if operation.operation_id not in legs_by_operation:
                operations.append(operation)
                legs_by_operation[operation.operation_id] = []
            if leg is not None:
                legs_by_operation[operation.operation_id].append(leg)
        return tuple(
            StockReconciliationOperation(
                operation_id=operation.operation_id,
                status=StockOperationStatus(operation.status),
                user_id=operation.user_id,
                user_name=operation.user_name or str(operation.user_id),
                symbol=operation.symbol,
                requested_action=StockAction(operation.requested_action),
                failure_reason=operation.failure_reason,
                created_at=operation.created_at,
                updated_at=operation.updated_at,
                legs=tuple(
                    _trade_leg_view(leg=leg, user_name=operation.user_name)
                    for leg in legs_by_operation[operation.operation_id]
                ),
            )
            for operation in operations
        )


async def reset_all_positions() -> int:
    """Flattens every stock position and finalizes any non-final operation.

    Used by the offline economy reset so stale, inflated positions cannot be
    re-extracted into wallet cash after balances are deflated. Non-final
    operations (pending / wallet_applied / reconcile_required) are also forced to
    `failed`; otherwise `_blocking_operation` would keep the affected users from
    trading that symbol even though the reset claims to flatten stock state.
    Market prices in `stock_profile` are intentionally left untouched.

    Returns:
        The number of position rows affected.
    """
    await _ensure_schema()
    now = _database_now()
    async with open_stock_session() as session:
        result = cast(
            "CursorResult[Any]",
            await session.execute(
                statement=update(StockPosition).values(
                    long_shares=0,
                    long_cost_basis=0,
                    short_shares=0,
                    short_entry_value=0,
                    short_collateral=0,
                    realized_pnl=0,
                    version=StockPosition.version + 1,
                    updated_at=now,
                )
            ),
        )
        await session.execute(
            statement=update(StockOperation)
            .where(StockOperation.status.notin_(_FINAL_OPERATION_STATUSES))
            .values(
                status=StockOperationStatus.FAILED.value,
                failure_reason="economy reset",
                updated_at=now,
            )
        )
        await session.commit()
        invalidate_stock_portfolio_cache()
        return int(result.rowcount or 0)


__all__ = [
    "Base",
    "StockNews",
    "StockOperation",
    "StockPosition",
    "StockPriceTick",
    "StockProfile",
    "StockTradeLeg",
    "advance_market_in_session",
    "cash_ceil",
    "cash_floor",
    "ensure_due_stock_news",
    "format_price",
    "get_stock_detail",
    "get_stock_news",
    "get_stock_portfolio",
    "list_market_quotes",
    "list_reconciliation_operations",
    "list_stock_profiles",
    "list_stock_supply_audit",
    "open_stock_session",
    "reset_all_positions",
    "settle_stock_operation",
    "upsert_stock_profile",
]
