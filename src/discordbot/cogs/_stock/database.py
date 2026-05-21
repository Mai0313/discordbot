"""Persistent store and settlement service for the simulated stock market."""

from __future__ import annotations

import uuid
from random import Random, SystemRandom
from typing import Any, Final
import asyncio
from datetime import datetime, timedelta

from sqlalchemy import Index, String, Integer, DateTime, event, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.dialects.sqlite import insert

from discordbot.typings.stock import (
    BCAT_NAME,
    BCAT_SYMBOL,
    BCAT_CATEGORY,
    BCAT_TOTAL_SHARES,
    STOCK_HISTORY_DAYS,
    STOCK_TICK_SECONDS,
    BCAT_BASE_VOLATILITY_BPS,
    BCAT_INITIAL_PRICE_CENTS,
    BCAT_VOLATILITY_AMPLIFIER_BPS,
    StockAction,
    StockNewsView,
    StockMarketQuote,
    StockProfileView,
    StockPositionView,
    StockTradeLegType,
    StockTradeLegView,
    StockPriceTickView,
    StockDetailViewData,
    StockOperationStatus,
    StockSettlementResult,
    StockReconciliationOperation,
)
from discordbot.typings.economy import WalletDeltaLeg
from discordbot.cogs._stock.market import (
    TAIWAN_TIMEZONE,
    as_taipei,
    cash_ceil,
    clamp_bps,
    cash_floor,
    format_price,
    tick_boundary,
    decay_news_sentiment,
    pressure_from_volume,
    tick_boundaries_to_apply,
    calculate_next_price_cents,
)
from discordbot.cogs._economy.database import get_balance, apply_ordered_wallet_deltas

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/stock.db")
_schema_ready_for: AsyncEngine | None = None
_schema_lock: asyncio.Lock | None = None
_schema_lock_loop: asyncio.AbstractEventLoop | None = None
_operation_locks: dict[tuple[int, str], asyncio.Lock] = {}
_operation_locks_loop: asyncio.AbstractEventLoop | None = None
_PRODUCTION_RNG: Final[SystemRandom] = SystemRandom()
_RECENT_TRADE_DAYS: Final[int] = 7


class Base(DeclarativeBase):
    """Base class for stock ORM models."""

    pass


class StockProfile(Base):
    """Stock profile and latest quote state."""

    __tablename__ = "stock_profile"

    symbol: Mapped[str] = mapped_column(String(length=16), primary_key=True)
    name: Mapped[str] = mapped_column(String(length=128), nullable=False)
    category: Mapped[str] = mapped_column(String(length=64), nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_close_price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    day_open_price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    total_shares: Mapped[int] = mapped_column(Integer, nullable=False)
    base_volatility_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    volatility_amplifier_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StockPosition(Base):
    """Per-user long and short position."""

    __tablename__ = "stock_position"

    symbol: Mapped[str] = mapped_column(String(length=16), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    long_shares: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    long_cost_basis: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    short_shares: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    short_entry_value: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    short_collateral: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    realized_pnl: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
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
    leg_type: Mapped[str] = mapped_column(String(length=32), nullable=False)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    wallet_delta: Mapped[int] = mapped_column(Integer, nullable=False)
    basis_delta: Mapped[int] = mapped_column(Integer, nullable=False)
    collateral_delta: Mapped[int] = mapped_column(Integer, nullable=False)
    realized_pnl_delta: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StockPriceTick(Base):
    """Materialized price tick."""

    __tablename__ = "stock_price_tick"
    __table_args__ = (Index("ix_stock_price_tick_symbol_created", "symbol", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(length=16), nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class StockNews(Base):
    """Deterministic template news that can influence lazy ticks."""

    __tablename__ = "stock_news"
    __table_args__ = (Index("ix_stock_news_symbol_created", "symbol", "created_at"),)

    id: Mapped[str] = mapped_column(String(length=64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(length=16), nullable=False)
    headline: Mapped[str] = mapped_column(String(length=256), nullable=False)
    sentiment_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class _StockOperationPlan(StockSettlementResult):
    """Internal settlement plan before any database mutation."""


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Configures SQLite for stock storage."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def _database_now() -> datetime:
    """Returns the timestamp used for stock rows."""
    return datetime.now(tz=TAIWAN_TIMEZONE)


def _current_schema_lock() -> asyncio.Lock:
    """Returns a schema lock bound to the current event loop."""
    global _schema_lock, _schema_lock_loop  # noqa: PLW0603 -- loop-local singleton
    loop = asyncio.get_running_loop()
    if _schema_lock is None or _schema_lock_loop is not loop:
        _schema_lock = asyncio.Lock()
        _schema_lock_loop = loop
    return _schema_lock


def _operation_lock(user_id: int, symbol: str) -> asyncio.Lock:
    """Returns a per-user stock operation lock bound to the current event loop."""
    global _operation_locks_loop  # noqa: PLW0603 -- loop-local lock map
    loop = asyncio.get_running_loop()
    if _operation_locks_loop is not loop:
        _operation_locks.clear()
        _operation_locks_loop = loop
    key = (user_id, symbol)
    lock = _operation_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _operation_locks[key] = lock
    return lock


def open_stock_session() -> AsyncSession:
    """Creates an async session bound to the current stock database engine."""
    return AsyncSession(bind=_engine, expire_on_commit=False)


async def _ensure_schema() -> None:
    """Bootstraps stock schema and seed data once per engine."""
    global _schema_ready_for  # noqa: PLW0603 -- module-level cache by engine identity
    if _schema_ready_for is _engine:
        return
    async with _current_schema_lock():
        if _schema_ready_for is _engine:
            return
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            now = _database_now()
            boundary = tick_boundary(dt=now)
            await conn.execute(
                statement=insert(StockProfile)
                .values(
                    symbol=BCAT_SYMBOL,
                    name=BCAT_NAME,
                    category=BCAT_CATEGORY,
                    price_cents=BCAT_INITIAL_PRICE_CENTS,
                    previous_close_price_cents=BCAT_INITIAL_PRICE_CENTS,
                    day_open_price_cents=BCAT_INITIAL_PRICE_CENTS,
                    total_shares=BCAT_TOTAL_SHARES,
                    base_volatility_bps=BCAT_BASE_VOLATILITY_BPS,
                    volatility_amplifier_bps=BCAT_VOLATILITY_AMPLIFIER_BPS,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_nothing(index_elements=["symbol"])
            )
            await _seed_initial_tick(conn=conn, now=boundary)
            await _seed_news(conn=conn, now=now)
        _schema_ready_for = _engine


async def _seed_initial_tick(conn: Any, now: datetime) -> None:  # noqa: ANN401 -- SQLAlchemy async connection is generic here
    """Seeds the first BCAT price tick when the table is empty."""
    existing_tick = await conn.execute(
        statement=select(StockPriceTick.id).where(StockPriceTick.symbol == BCAT_SYMBOL).limit(1)
    )
    if existing_tick.scalar_one_or_none() is not None:
        return
    await conn.execute(
        statement=insert(StockPriceTick).values(
            symbol=BCAT_SYMBOL, price_cents=BCAT_INITIAL_PRICE_CENTS, created_at=now
        )
    )


async def _seed_news(conn: Any, now: datetime) -> None:  # noqa: ANN401 -- SQLAlchemy async connection is generic here
    """Seeds deterministic BCAT news rows."""
    rows = (
        ("bcat-seed-1", "BCAT 推出新的紙箱訂閱制，市場反應熱烈", 80, now - timedelta(hours=6)),
        ("bcat-seed-2", "BCAT 董事會表示將控制罐罐成本，毛利率看升", 45, now - timedelta(days=1)),
        ("bcat-seed-3", "BCAT 供應鏈短暫卡關，投資人觀望", -55, now - timedelta(days=2)),
    )
    for row_id, headline, sentiment_bps, created_at in rows:
        await conn.execute(
            statement=insert(StockNews)
            .values(
                id=row_id,
                symbol=BCAT_SYMBOL,
                headline=headline,
                sentiment_bps=sentiment_bps,
                created_at=created_at,
            )
            .on_conflict_do_nothing(index_elements=["id"])
        )


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
        base_volatility_bps=profile.base_volatility_bps,
        volatility_amplifier_bps=profile.volatility_amplifier_bps,
        updated_at=profile.updated_at,
    )


def _position_view(position: StockPosition | None, symbol: str, user_id: int) -> StockPositionView:
    """Projects an ORM position into a typed view."""
    if position is None:
        return StockPositionView(symbol=symbol, user_id=user_id)
    return StockPositionView(
        symbol=position.symbol,
        user_id=position.user_id,
        long_shares=position.long_shares,
        long_cost_basis=position.long_cost_basis,
        short_shares=position.short_shares,
        short_entry_value=position.short_entry_value,
        short_collateral=position.short_collateral,
        realized_pnl=position.realized_pnl,
    )


def _trade_leg_view(leg: StockTradeLeg) -> StockTradeLegView:
    """Projects an ORM trade leg into a typed view."""
    return StockTradeLegView(
        operation_id=leg.operation_id,
        leg_order=leg.leg_order,
        symbol=leg.symbol,
        user_id=leg.user_id,
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


async def _latest_tick(session: AsyncSession, symbol: str) -> StockPriceTick | None:
    """Returns the latest price tick for a stock."""
    result = await session.execute(
        statement=select(StockPriceTick)
        .where(StockPriceTick.symbol == symbol)
        .order_by(StockPriceTick.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _news_sentiment_bps(session: AsyncSession, symbol: str, at: datetime) -> int:
    """Returns decayed deterministic news sentiment for a tick."""
    result = await session.execute(
        statement=select(StockNews)
        .where(StockNews.symbol == symbol, StockNews.created_at <= at)
        .order_by(StockNews.created_at.desc())
        .limit(12)
    )
    sentiment = 0
    for news in result.scalars():
        ticks_elapsed = max(
            int((tick_boundary(dt=at) - tick_boundary(dt=news.created_at)).total_seconds())
            // STOCK_TICK_SECONDS,
            0,
        )
        sentiment += decay_news_sentiment(
            sentiment_bps=news.sentiment_bps, ticks_elapsed=ticks_elapsed
        )
    return clamp_bps(value=sentiment, lower=-300, upper=300)


async def _recent_pressure_bps(session: AsyncSession, symbol: str, at: datetime) -> int:
    """Returns recent buy/sell pressure from applied trade legs."""
    since = at - timedelta(days=_RECENT_TRADE_DAYS)
    result = await session.execute(
        statement=select(StockTradeLeg.leg_type, StockTradeLeg.shares)
        .join(StockOperation, StockOperation.operation_id == StockTradeLeg.operation_id)
        .where(
            StockTradeLeg.symbol == symbol,
            StockTradeLeg.created_at >= since,
            StockTradeLeg.created_at <= at,
            StockOperation.status == StockOperationStatus.APPLIED.value,
        )
    )
    buy_shares = 0
    sell_shares = 0
    for leg_type, shares in result.all():
        if leg_type in (StockTradeLegType.OPEN_LONG.value, StockTradeLegType.COVER_SHORT.value):
            buy_shares += shares
        else:
            sell_shares += shares
    return pressure_from_volume(buy_shares=buy_shares, sell_shares=sell_shares)


async def advance_market_in_session(
    session: AsyncSession, symbol: str, now: datetime | None = None, rng: Random | None = None
) -> StockMarketQuote:
    """Advances one stock lazily to the current tick boundary."""
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
        latest_tick = StockPriceTick(
            symbol=symbol,
            price_cents=profile.price_cents,
            created_at=tick_boundary(dt=effective_now),
        )
        session.add(instance=latest_tick)
        await session.flush()

    current_price = latest_tick.price_cents
    previous_tick_at = latest_tick.created_at
    for boundary in tick_boundaries_to_apply(
        latest_tick_at=latest_tick.created_at, now=effective_now
    ):
        news_sentiment = await _news_sentiment_bps(session=session, symbol=symbol, at=boundary)
        pressure_bps = await _recent_pressure_bps(session=session, symbol=symbol, at=boundary)
        next_price = calculate_next_price_cents(
            previous_price_cents=current_price,
            news_sentiment_bps=news_sentiment,
            pressure_bps=pressure_bps,
            base_volatility_bps=profile.base_volatility_bps,
            volatility_amplifier_bps=profile.volatility_amplifier_bps,
            rng=effective_rng,
        )
        if as_taipei(dt=boundary).date() != as_taipei(dt=previous_tick_at).date():
            profile.previous_close_price_cents = current_price
            profile.day_open_price_cents = next_price
        session.add(
            instance=StockPriceTick(symbol=symbol, price_cents=next_price, created_at=boundary)
        )
        current_price = next_price
        previous_tick_at = boundary

    if current_price != profile.price_cents:
        profile.price_cents = current_price
        profile.updated_at = previous_tick_at
    pressure_bps = await _recent_pressure_bps(session=session, symbol=symbol, at=effective_now)
    return _quote_from_profile(profile=profile, pressure_bps=pressure_bps)


async def list_market_quotes(
    now: datetime | None = None, rng: Random | None = None
) -> tuple[StockMarketQuote, ...]:
    """Returns public market quotes after lazy advancement."""
    await _ensure_schema()
    async with open_stock_session() as session:
        symbols_result = await session.execute(statement=select(StockProfile.symbol))
        quotes = [
            await advance_market_in_session(session=session, symbol=symbol, now=now, rng=rng)
            for symbol in symbols_result.scalars()
        ]
        await session.commit()
        return tuple(quotes)


async def get_stock_detail(
    symbol: str, user_id: int, now: datetime | None = None, rng: Random | None = None
) -> StockDetailViewData:
    """Returns a personal stock detail view after lazy advancement."""
    await _ensure_schema()
    async with open_stock_session() as session:
        quote = await advance_market_in_session(session=session, symbol=symbol, now=now, rng=rng)
        position = await _get_position_view(session=session, symbol=symbol, user_id=user_id)
        recent_trades = await _recent_trade_views(session=session, symbol=symbol, user_id=user_id)
        news = await _news_views(session=session, symbol=symbol)
        ticks = await _price_tick_views(session=session, symbol=symbol, now=now or _database_now())
        await session.commit()
    balance = await get_balance(user_id=user_id)
    return StockDetailViewData(
        quote=quote,
        balance=balance,
        position=position,
        recent_trades=recent_trades,
        news=news,
        ticks=ticks,
    )


async def get_stock_news(symbol: str) -> tuple[StockNewsView, ...]:
    """Returns recent deterministic news for a stock."""
    await _ensure_schema()
    async with open_stock_session() as session:
        return await _news_views(session=session, symbol=symbol)


async def _get_position_view(
    session: AsyncSession, symbol: str, user_id: int
) -> StockPositionView:
    """Returns a position view inside the caller's stock session."""
    result = await session.execute(
        statement=select(StockPosition).where(
            StockPosition.symbol == symbol, StockPosition.user_id == user_id
        )
    )
    return _position_view(position=result.scalar_one_or_none(), symbol=symbol, user_id=user_id)


async def _recent_trade_views(
    session: AsyncSession, symbol: str, user_id: int
) -> tuple[StockTradeLegView, ...]:
    """Returns recent applied trade legs for one user and stock."""
    result = await session.execute(
        statement=select(StockTradeLeg)
        .join(StockOperation, StockOperation.operation_id == StockTradeLeg.operation_id)
        .where(
            StockTradeLeg.symbol == symbol,
            StockTradeLeg.user_id == user_id,
            StockOperation.status == StockOperationStatus.APPLIED.value,
        )
        .order_by(StockTradeLeg.created_at.desc(), StockTradeLeg.leg_order.desc())
        .limit(8)
    )
    return tuple(_trade_leg_view(leg=leg) for leg in result.scalars())


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


def _prorated_amount(total: int, shares: int, current_shares: int) -> int:
    """Returns a prorated integer basis amount, consuming dust on final close."""
    if shares >= current_shares:
        return total
    return total * shares // current_shares


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
    wallet_balance: int,
    position: StockPositionView,
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
            wallet_balance=wallet_balance,
            position=position,
            now=now,
        )
    return _build_short_plan(
        operation_id=operation_id,
        symbol=symbol,
        user_id=user_id,
        quantity=quantity,
        price_cents=price_cents,
        wallet_balance=wallet_balance,
        position=position,
        now=now,
    )


def _build_buy_plan(  # noqa: PLR0913 -- buy can cover short and open long in order
    operation_id: str,
    symbol: str,
    user_id: int,
    quantity: int,
    price_cents: int,
    wallet_balance: int,
    position: StockPositionView,
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
        cover_cost = cash_ceil(cents=price_cents * cover_shares)
        if cover_cost > released_collateral + wallet_balance + wallet_delta_total:
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
                price_cents=price_cents,
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
        cost = cash_ceil(cents=price_cents * remaining)
        if cost > wallet_balance + wallet_delta_total:
            return _insufficient_result(
                symbol=symbol,
                action=StockAction.BUY,
                quantity=quantity,
                price_cents=price_cents,
                balance=wallet_balance,
                position=position,
                error=f"餘額不足，需要 {cost:,} 才能買入 {remaining:,} 股",
            )
        legs.append(
            _leg_view(
                operation_id=operation_id,
                leg_order=len(legs) + 1,
                symbol=symbol,
                user_id=user_id,
                leg_type=StockTradeLegType.OPEN_LONG,
                shares=remaining,
                price_cents=price_cents,
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
        price_cents=price_cents,
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
    wallet_balance: int,
    position: StockPositionView,
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
        proceeds = cash_floor(cents=price_cents * sell_shares)
        realized = proceeds - released_basis
        legs.append(
            _leg_view(
                operation_id=operation_id,
                leg_order=len(legs) + 1,
                symbol=symbol,
                user_id=user_id,
                leg_type=StockTradeLegType.SELL_LONG,
                shares=sell_shares,
                price_cents=price_cents,
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
        collateral = cash_ceil(cents=price_cents * remaining)
        if collateral > wallet_balance + wallet_delta_total:
            return _insufficient_result(
                symbol=symbol,
                action=StockAction.SHORT,
                quantity=quantity,
                price_cents=price_cents,
                balance=wallet_balance,
                position=position,
                error=f"餘額不足，需要 {collateral:,} 作為做空 collateral",
            )
        entry_value = cash_floor(cents=price_cents * remaining)
        legs.append(
            _leg_view(
                operation_id=operation_id,
                leg_order=len(legs) + 1,
                symbol=symbol,
                user_id=user_id,
                leg_type=StockTradeLegType.OPEN_SHORT,
                shares=remaining,
                price_cents=price_cents,
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
        price_cents=price_cents,
        wallet_delta=wallet_delta_total,
        balance_after=wallet_balance + wallet_delta_total,
        position=final_position,
        legs=tuple(legs),
        status=StockOperationStatus.PENDING,
    )


async def settle_stock_operation(  # noqa: PLR0913 -- Discord identity and trade request are all needed
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
        async with open_stock_session() as session:
            quote = await advance_market_in_session(
                session=session, symbol=normalized_symbol, now=effective_now, rng=rng
            )
            position = await _get_position_view(
                session=session, symbol=normalized_symbol, user_id=user_id
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
                await session.rollback()
                return _insufficient_result(
                    symbol=normalized_symbol,
                    action=requested_action,
                    quantity=0,
                    price_cents=quote.profile.price_cents,
                    balance=wallet_balance,
                    position=position,
                    error="股數格式錯誤，請輸入正整數或 ALL",
                )

            plan = _build_plan(
                operation_id=operation_id,
                symbol=normalized_symbol,
                user_id=user_id,
                action=requested_action,
                quantity=parsed_quantity,
                price_cents=quote.profile.price_cents,
                wallet_balance=wallet_balance,
                position=position,
                now=effective_now,
            )
            if not plan.success:
                await session.rollback()
                return plan
            await _commit_stock_side(session=session, plan=plan, now=effective_now)

        try:
            wallet_result = await apply_ordered_wallet_deltas(
                user_id=user_id,
                name=user_name,
                avatar_url=avatar_url,
                deltas=tuple(
                    WalletDeltaLeg(
                        delta=leg.wallet_delta, reason=f"stock:{operation_id}:{leg.leg_order}"
                    )
                    for leg in plan.legs
                ),
            )
        except Exception as exc:
            await _mark_operation(
                operation_id=operation_id,
                status=StockOperationStatus.RECONCILE_REQUIRED,
                failure_reason=f"wallet delta raised after stock side was applied: {type(exc).__name__}",
            )
            return plan.model_copy(
                update={
                    "success": False,
                    "status": StockOperationStatus.RECONCILE_REQUIRED,
                    "error": f"交易狀態需要人工 reconciliation，operation_id={operation_id}",
                }
            )
        if wallet_result is None:
            await _mark_operation(
                operation_id=operation_id,
                status=StockOperationStatus.RECONCILE_REQUIRED,
                failure_reason="wallet delta failed after stock side was applied",
            )
            return plan.model_copy(
                update={
                    "success": False,
                    "status": StockOperationStatus.RECONCILE_REQUIRED,
                    "error": f"交易狀態需要人工 reconciliation，operation_id={operation_id}",
                }
            )

        await _mark_operation(
            operation_id=operation_id,
            status=StockOperationStatus.WALLET_APPLIED,
            failure_reason="",
        )
        await _mark_operation(
            operation_id=operation_id, status=StockOperationStatus.APPLIED, failure_reason=""
        )
        return plan.model_copy(
            update={
                "status": StockOperationStatus.APPLIED,
                "balance_after": wallet_result.new_balance,
            }
        )


async def _commit_stock_side(
    session: AsyncSession, plan: _StockOperationPlan, now: datetime
) -> None:
    """Commits stock position, operation, and trade legs before wallet mutation."""
    session.add(
        instance=StockOperation(
            operation_id=plan.operation_id or "",
            symbol=plan.symbol,
            user_id=plan.position.user_id,
            requested_action=plan.requested_action.value,
            status=StockOperationStatus.PENDING.value,
            failure_reason="",
            created_at=now,
            updated_at=now,
        )
    )
    await _write_position(session=session, position=plan.position, now=now)
    for leg in plan.legs:
        session.add(
            instance=StockTradeLeg(
                operation_id=leg.operation_id,
                leg_order=leg.leg_order,
                symbol=leg.symbol,
                user_id=leg.user_id,
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
    await session.execute(
        statement=update(StockOperation)
        .where(StockOperation.operation_id == plan.operation_id)
        .values(status=StockOperationStatus.STOCK_APPLIED.value, updated_at=now)
    )
    await session.commit()


async def _write_position(
    session: AsyncSession, position: StockPositionView, now: datetime
) -> None:
    """Upserts the final position after stock-side validation."""
    await session.execute(
        statement=insert(StockPosition)
        .values(
            symbol=position.symbol,
            user_id=position.user_id,
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
            .where(
                StockOperation.status.notin_([
                    StockOperationStatus.APPLIED.value,
                    StockOperationStatus.REVERSED.value,
                ])
            )
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
                    symbol=operation.symbol,
                    requested_action=StockAction(operation.requested_action),
                    failure_reason=operation.failure_reason,
                    created_at=operation.created_at,
                    updated_at=operation.updated_at,
                    legs=tuple(_trade_leg_view(leg=leg) for leg in leg_result.scalars()),
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
    "format_price",
    "get_stock_detail",
    "get_stock_news",
    "list_market_quotes",
    "list_reconciliation_operations",
    "open_stock_session",
    "settle_stock_operation",
]
