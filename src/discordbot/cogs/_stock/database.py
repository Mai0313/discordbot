"""Persistent store and settlement service for the simulated stock market."""

from __future__ import annotations

from time import monotonic
import uuid
from random import Random, SystemRandom
from typing import TYPE_CHECKING, Any, Final
import asyncio
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import logfire
from pydantic import BaseModel, ConfigDict
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
from discordbot.cogs._stock.market import (
    NEWS_SENTIMENT_DECAY_BPS,
    NEWS_SENTIMENT_LIMIT_BPS,
    NEWS_SENTIMENT_DECAY_SECONDS,
    as_taipei,
    clamp_bps,
    format_price,
    tick_boundary,
    decay_news_sentiment,
    execution_price_cents,
    pressure_from_order_flow,
    tick_boundaries_to_apply,
    calculate_next_price_cents,
)
from discordbot.cogs._stock.prompts import (
    STOCK_NEWS_BEARISH_FALLBACK_TEMPLATES,
    STOCK_NEWS_BULLISH_FALLBACK_TEMPLATES,
    STOCK_NEWS_NEUTRAL_FALLBACK_TEMPLATES,
)
from discordbot.utils.sqlite_config import configure_sqlite_connection
from discordbot.utils.stored_integer import StoredInteger
from discordbot.cogs._economy.database import get_balance, apply_ordered_wallet_deltas

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable, AsyncIterator

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/stock.db")
_schema_ready_for: AsyncEngine | None = None
_schema_lock: asyncio.Lock | None = None
_schema_lock_loop: asyncio.AbstractEventLoop | None = None
_operation_locks: dict[tuple[int, str], asyncio.Lock] = {}
_operation_lock_refcounts: dict[tuple[int, str], int] = {}
_operation_locks_loop: asyncio.AbstractEventLoop | None = None
_market_locks: dict[str, asyncio.Lock] = {}
_market_lock_refcounts: dict[str, int] = {}
_market_locks_loop: asyncio.AbstractEventLoop | None = None
_news_generation_lock: asyncio.Lock | None = None
_news_generation_lock_loop: asyncio.AbstractEventLoop | None = None
_news_provider_semaphore: asyncio.Semaphore | None = None
_news_provider_semaphore_loop: asyncio.AbstractEventLoop | None = None
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
    """Clears process-local stock portfolio view cache entries."""
    if user_id is None:
        _stock_portfolio_cache.clear()
        return
    engine_id = id(_engine)
    _stock_portfolio_cache.pop((engine_id, user_id), None)


def _cached_stock_portfolio(user_id: int) -> StockPortfolioView | None:
    """Returns a cached stock portfolio when its short TTL is still valid."""
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
    """Stores one stock portfolio view in the short process cache."""
    _stock_portfolio_cache[(id(_engine), portfolio.user_id)] = (monotonic(), portfolio)
    return portfolio


class Base(DeclarativeBase):
    """Base class for stock ORM models."""

    pass


class StockProfile(Base):
    """Stock profile and latest quote state."""

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
    """Per-user long and short position."""

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
    """Lifecycle row for one cross-database stock operation."""

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
    """One ordered leg produced by a stock operation."""

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
    """Materialized price tick."""

    __tablename__ = "stock_price_tick"
    __table_args__ = (
        Index("ix_stock_price_tick_symbol_created", "symbol", "created_at", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(length=16), nullable=False)
    price_cents: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StockNews(Base):
    """Stock news that can influence lazy ticks."""

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
    """Internal settlement plan before any database mutation."""


class _StockExecutionSnapshot(BaseModel):
    """Submit-time state needed to cap a requested quantity."""

    model_config = ConfigDict(frozen=True)

    action: StockAction
    price_cents: int
    liquidity_shares: int
    max_order_impact_bps: int
    wallet_balance: int
    position: StockPositionView
    available_long_shares: int
    available_short_shares: int
    available_individual_long_shares: int


class _StockMarketExposure(BaseModel):
    """Aggregate market exposure for one symbol."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    long_shares: int
    short_shares: int
    available_long_shares: int
    available_short_shares: int


class _StockOrderFlowSummary(BaseModel):
    """Recent order-flow summary for stock news context."""

    model_config = ConfigDict(frozen=True)

    buy_side_shares: int = 0
    sell_side_shares: int = 0
    pressure_bps: int = 0


def _configure_sqlite_connection(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Configures SQLite for stock storage."""
    configure_sqlite_connection(dbapi_connection=dbapi_connection)


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Configures SQLite for stock storage."""
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


def _current_news_generation_lock() -> asyncio.Lock:
    """Returns the process-local stock news generation lock for this event loop."""
    global _news_generation_lock, _news_generation_lock_loop  # noqa: PLW0603 -- loop-local singleton
    loop = asyncio.get_running_loop()
    if _news_generation_lock is None or _news_generation_lock_loop is not loop:
        _news_generation_lock = asyncio.Lock()
        _news_generation_lock_loop = loop
    return _news_generation_lock


def _current_news_provider_semaphore() -> asyncio.Semaphore:
    """Returns the stock news provider concurrency limiter for this event loop."""
    global _news_provider_semaphore, _news_provider_semaphore_loop  # noqa: PLW0603 -- loop-local singleton
    loop = asyncio.get_running_loop()
    if _news_provider_semaphore is None or _news_provider_semaphore_loop is not loop:
        _news_provider_semaphore = asyncio.Semaphore(_NEWS_PROVIDER_CONCURRENCY)
        _news_provider_semaphore_loop = loop
    return _news_provider_semaphore


@asynccontextmanager
async def _operation_lock(user_id: int, symbol: str) -> AsyncIterator[None]:
    """Returns a per-user stock operation lock bound to the current event loop."""
    global _operation_locks_loop  # noqa: PLW0603 -- loop-local lock map
    loop = asyncio.get_running_loop()
    if _operation_locks_loop is not loop:
        _operation_locks.clear()
        _operation_lock_refcounts.clear()
        _operation_locks_loop = loop
    key = (user_id, symbol)
    lock = _operation_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _operation_locks[key] = lock
    _operation_lock_refcounts[key] = _operation_lock_refcounts.get(key, 0) + 1
    acquired = False
    try:
        await lock.acquire()
        acquired = True
        yield
    finally:
        if acquired:
            lock.release()
        refcount = _operation_lock_refcounts.get(key, 1) - 1
        if refcount <= 0:
            _operation_lock_refcounts.pop(key, None)
            if _operation_locks.get(key) is lock:
                _operation_locks.pop(key, None)
        else:
            _operation_lock_refcounts[key] = refcount


@asynccontextmanager
async def _market_lock(symbol: str) -> AsyncIterator[None]:
    """Returns a per-symbol market advancement lock bound to the current event loop."""
    global _market_locks_loop  # noqa: PLW0603 -- loop-local lock map
    loop = asyncio.get_running_loop()
    if _market_locks_loop is not loop:
        _market_locks.clear()
        _market_lock_refcounts.clear()
        _market_locks_loop = loop
    key = symbol.upper()
    lock = _market_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _market_locks[key] = lock
    _market_lock_refcounts[key] = _market_lock_refcounts.get(key, 0) + 1
    acquired = False
    try:
        await lock.acquire()
        acquired = True
        yield
    finally:
        if acquired:
            lock.release()
        refcount = _market_lock_refcounts.get(key, 1) - 1
        if refcount <= 0:
            _market_lock_refcounts.pop(key, None)
            if _market_locks.get(key) is lock:
                _market_locks.pop(key, None)
        else:
            _market_lock_refcounts[key] = refcount


def open_stock_session() -> AsyncSession:
    """Creates an async session bound to the current stock database engine."""
    _ensure_sqlite_hooks(engine=_engine)
    return AsyncSession(bind=_engine, expire_on_commit=False)


async def _begin_immediate(session: AsyncSession) -> None:
    """Acquires SQLite's write lock before reading stock state for a mutation plan."""
    await session.execute(statement=text("BEGIN IMMEDIATE"))


async def _ensure_schema() -> None:
    """Bootstraps stock schema once per engine."""
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


def _profile_view(profile: StockProfile) -> StockProfileView:
    """Projects an ORM profile into a typed view."""
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
    """Projects an ORM position into a typed view."""
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
    """Projects a stock position into a public participant summary."""
    return StockParticipantPositionView(
        user_id=position.user_id,
        user_name=position.user_name or str(position.user_id),
        long_shares=position.long_shares,
        short_shares=position.short_shares,
        realized_pnl=position.realized_pnl,
    )


def _trade_leg_view(leg: StockTradeLeg, user_name: str = "") -> StockTradeLegView:
    """Projects an ORM trade leg into a typed view."""
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
    """Projects an ORM news row into a typed view."""
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
    """Projects an ORM tick row into a typed view."""
    return StockPriceTickView(
        symbol=tick.symbol, price_cents=tick.price_cents, created_at=tick.created_at
    )


def _quote_from_profile(profile: StockProfile, pressure_bps: int) -> StockMarketQuote:
    """Builds a quote from the latest profile row."""
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
    """Creates or updates a DB-owned stock profile from an explicit maintenance payload."""
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
    """Lists DB-owned stock profiles without advancing market ticks."""
    await _ensure_schema()
    async with open_stock_session() as session:
        result = await session.execute(
            statement=select(StockProfile).order_by(StockProfile.symbol.asc())
        )
        return tuple(_profile_view(profile=profile) for profile in result.scalars())


async def list_stock_supply_audit() -> tuple[StockSupplyAuditView, ...]:
    """Lists DB-owned stock supply and aggregate exposure without advancing ticks."""
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
    """Creates due stock news rows, using AI when a provider is available."""
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
            """Generates one news row without holding a database transaction."""
            generated: StockGeneratedNews | None = None
            if news_provider is not None:
                async with _current_news_provider_semaphore():
                    try:
                        generated = await news_provider(context)
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
    """Returns stock news generation contexts for profiles that need fresh news."""
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
    """Builds DB-backed market context for stock news generation."""
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
    """Returns recent order-flow summaries keyed by symbol."""
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
    """Summarizes recent order flow for stock news context."""
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
    """Returns recent news rows keyed by symbol for generation context."""
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
    """Persists one generated news row with a stable cadence-bucket ID."""
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
    """Returns the cadence bucket for one stock news row."""
    cadence_seconds = max(profile.news_cadence_hours, 1) * 60 * 60
    return int(as_taipei(dt=now).timestamp()) // cadence_seconds


def _fallback_generated_news(
    context: StockNewsGenerationContext, now: datetime
) -> StockGeneratedNews:
    """Returns deterministic fictional news for a due stock profile."""
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
    """Chooses fallback templates from the same market context used by AI news."""
    signal_bps = (
        context.change_bps // 2 + context.pressure_bps + context.recent_news_sentiment_bps // 3
    )
    if signal_bps >= 50:
        return STOCK_NEWS_BULLISH_FALLBACK_TEMPLATES
    if signal_bps <= -50:
        return STOCK_NEWS_BEARISH_FALLBACK_TEMPLATES
    return STOCK_NEWS_NEUTRAL_FALLBACK_TEMPLATES


async def _latest_tick(session: AsyncSession, symbol: str) -> StockPriceTick | None:
    """Returns the latest price tick for a stock."""
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
    """Inserts a tick once and returns the persisted price for that boundary."""
    result = await session.execute(
        statement=insert(StockPriceTick)
        .values(symbol=symbol, price_cents=price_cents, created_at=created_at)
        .on_conflict_do_nothing(index_elements=["symbol", "created_at"])
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
    """Inserts or replaces the maintenance price tick for a boundary."""
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
    """Returns news rows needed to price an already selected boundary range."""
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
    """Returns time-decayed ambient sentiment for the AI news generation prompt only."""
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

    Each news row contributes its clamped sentiment exactly once, at the first
    applied boundary at or after its own tick boundary. News whose tick boundary
    falls before every applied boundary is skipped (its impulse already landed
    on a previous lazy advance).
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
    """Returns recent buy/sell pressure from applied trade legs."""
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
        pressure_rows=tuple(result.all()), at=at, liquidity_shares=liquidity_shares
    )


async def _pressure_rows_for_boundaries(
    session: AsyncSession, symbol: str, boundaries: tuple[datetime, ...]
) -> tuple[tuple[str, int, datetime], ...]:
    """Returns trade-leg rows needed to price an already selected boundary range."""
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
    """Returns recent buy/sell pressure from prefetched trade legs."""
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
    """Advances one stock lazily to the current tick boundary."""
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
    """Returns public market quotes after lazy advancement."""
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
    """Advances several symbols with one stock session while preserving per-symbol locks."""
    async with open_stock_session() as session:
        for symbol in symbols:
            async with _market_lock(symbol=symbol):
                await advance_market_in_session(session=session, symbol=symbol, now=now, rng=rng)
                await session.commit()


async def _current_stock_portfolio(user_id: int) -> StockPortfolioView:
    """Reads a portfolio after the caller has advanced relevant symbols."""
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
    """Returns a personal stock detail view after lazy advancement."""
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
    """Returns the user's non-zero stock positions with current quote valuation."""
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
    """Returns recent news for a stock."""
    await ensure_due_stock_news(symbols=(symbol,))
    await _ensure_schema()
    async with open_stock_session() as session:
        return await _news_views(session=session, symbol=symbol)


async def _get_position_view(
    session: AsyncSession, symbol: str, user_id: int, user_name: str = ""
) -> StockPositionView:
    """Returns a position view inside the caller's stock session."""
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
    """Returns recent applied trade legs for a stock, optionally scoped to one user."""
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
    """Returns public stock-level non-zero position summaries."""
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
    """Returns symbols where the user has a non-zero position."""
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
    """Returns total long and short shares for ordering display rows."""
    return position.long_shares + position.short_shares


def _portfolio_holding_view(
    position: StockPosition, profile: StockProfile
) -> StockPortfolioHolding:
    """Projects a stock position into portfolio valuation terms."""
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
    """Returns aggregate long and short exposure against the configured float."""
    return (await _market_exposures(session=session, profiles=(profile,)))[profile.symbol]


async def _market_exposures(
    session: AsyncSession, profiles: tuple[StockProfile, ...]
) -> dict[str, _StockMarketExposure]:
    """Returns aggregate exposure by symbol, reserving shares for non-final operations."""
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
    """Returns recent news views inside the caller's stock session."""
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
    """Returns chart ticks for the last seven days."""
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
    """Parses a modal quantity at submit time."""
    normalized = raw_quantity.strip().replace(",", "")
    if normalized.upper() in {"ALL", "全部", "MAX"}:
        if action == StockAction.BUY and position.short_shares > 0:
            return position.short_shares
        if action == StockAction.SHORT and position.long_shares > 0:
            return position.long_shares
        return wallet_balance * 100 // price_cents
    return int(normalized)


def _is_all_quantity(raw_quantity: str) -> bool:
    """Returns whether the raw quantity uses the ALL shorthand."""
    return raw_quantity.strip().replace(",", "").upper() in {"ALL", "全部", "MAX"}


def _prorated_amount(total: int, shares: int, current_shares: int) -> int:
    """Returns a prorated integer basis amount, consuming dust on final close."""
    if shares >= current_shares:
        return total
    return total * shares // current_shares


def _buy_execution_price(
    price_cents: int, shares: int, liquidity_shares: int, max_impact_bps: int
) -> int:
    """Returns the execution price for a buy-side leg."""
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
    """Returns the execution price for a sell-side leg."""
    return execution_price_cents(
        reference_price_cents=price_cents,
        shares=shares,
        liquidity_shares=liquidity_shares,
        max_impact_bps=max_impact_bps,
        is_buy=False,
    )


def _buy_cost(price_cents: int, shares: int, liquidity_shares: int, max_impact_bps: int) -> int:
    """Returns integer cash needed for a buy-side leg."""
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
    """Returns integer cash received from a sell-side leg."""
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
    """Returns the largest buy-side size affordable after execution impact."""
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
    """Returns the largest short size allowed by the reference-price collateral."""
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
    """Returns how many short shares can be covered from the submit-time state."""
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
    """Returns the maximum long shares one user may hold for a stock."""
    return float_shares * STOCK_INDIVIDUAL_OWNERSHIP_CAP_BPS // STOCK_BPS_DENOMINATOR


def _available_individual_long_shares(float_shares: int, position: StockPositionView) -> int:
    """Returns how many new long shares the user can open before the ownership cap."""
    return max(_individual_long_cap_shares(float_shares=float_shares) - position.long_shares, 0)


def _open_long_share_cap(snapshot: _StockExecutionSnapshot) -> int:
    """Returns the submit-time cap for opening new long shares."""
    return min(snapshot.available_long_shares, snapshot.available_individual_long_shares)


def _max_executable_quantity(snapshot: _StockExecutionSnapshot) -> int:
    """Returns the largest quantity that can pass balance validation."""
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
    """Treats oversized numeric quantities as the submit-time executable maximum."""
    if parsed_quantity <= 0:
        return parsed_quantity
    return min(parsed_quantity, _max_executable_quantity(snapshot=snapshot))


def _max_quantity_error(snapshot: _StockExecutionSnapshot) -> str:
    """Returns the clearest validation error when nothing is executable."""
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
    """Builds an in-memory trade leg view."""
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
    """Returns a share-weighted execution price for a result summary."""
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
    """Builds a typed failed settlement result without an operation row."""
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
    """Builds ordered stock and wallet mutations from the submit-time state."""
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
    """Builds a buy/cover plan."""
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
    """Builds a short/sell plan."""
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
    """Returns the oldest non-final operation that blocks new trades."""
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
    """Builds a failed result when a previous operation needs attention first."""
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
    """Expands trade legs into ordered gross wallet movements."""
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
    """Builds a submit-time operation plan from locked market and position state."""
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
    """Settles a buy/cover or short/sell request through one service boundary."""
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
            if not plan.success:
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
        except Exception as exc:
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
        except Exception as exc:
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
    """Commits the planned operation and legs before wallet mutation."""
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
    """Applies the stock position after wallet legs have committed."""
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
    """Upserts the final position after stock-side validation."""
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
    """Updates operation status after a cross-database lifecycle step."""
    await _ensure_schema()
    async with open_stock_session() as session:
        await session.execute(
            statement=update(StockOperation)
            .where(StockOperation.operation_id == operation_id)
            .values(status=status.value, failure_reason=failure_reason, updated_at=_database_now())
        )
        await session.commit()


async def list_reconciliation_operations() -> tuple[StockReconciliationOperation, ...]:
    """Lists non-final stock operations for manual reconciliation."""
    await _ensure_schema()
    async with open_stock_session() as session:
        result = await session.execute(
            statement=select(StockOperation)
            .where(StockOperation.status.notin_(_FINAL_OPERATION_STATUSES))
            .order_by(StockOperation.created_at.asc())
        )
        operations = list(result.scalars())
        output: list[StockReconciliationOperation] = []
        for operation in operations:
            leg_result = await session.execute(
                statement=select(StockTradeLeg)
                .where(StockTradeLeg.operation_id == operation.operation_id)
                .order_by(StockTradeLeg.leg_order.asc())
            )
            output.append(
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
                        for leg in leg_result.scalars()
                    ),
                )
            )
        return tuple(output)


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
    "settle_stock_operation",
    "upsert_stock_profile",
]
