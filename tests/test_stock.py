"""Tests for the simulated stock market service."""

import math
from random import Random
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text, delete, select, update, inspect
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from discordbot.cogs._stock import database as stock_db
from discordbot.typings.stock import (
    STOCK_TICK_SECONDS,
    STOCK_BPS_DENOMINATOR,
    STOCK_NEWS_CADENCE_HOURS,
    MAX_TICKS_PER_INTERACTION,
    STOCK_INDIVIDUAL_OWNERSHIP_CAP_BPS,
    StockAction,
    StockProfileView,
    StockTradeLegType,
    StockGeneratedNews,
    StockProfileUpsert,
    StockOperationStatus,
    StockSettlementResult,
    StockNewsGenerationContext,
)
from discordbot.typings.economy import WalletDeltaLeg, OrderedWalletDeltaResult
from discordbot.cogs._stock.chart import build_price_chart
from discordbot.cogs._stock.market import (
    TAIWAN_TIMEZONE,
    NEWS_SENTIMENT_LIMIT_BPS,
    cash_ceil,
    cash_floor,
    format_price,
    tick_boundary,
    order_impact_bps,
    decay_news_sentiment,
    execution_price_cents,
    pressure_from_order_flow,
    tick_boundaries_to_apply,
    calculate_next_price_cents,
)
from discordbot.cogs._stock.prompts import STOCK_NEWS_PROMPT, STOCK_NEWS_FALLBACK_TEMPLATES
from discordbot.cogs._economy.database import (
    UserWallet,
    open_session,
    adjust_balance,
    apply_ordered_wallet_deltas,
)

BCAT_SYMBOL = "BCAT"
BCAT_NAME = "破貓科技股份有限公司"
BCAT_CATEGORY = "科技"
BCAT_INITIAL_PRICE_CENTS = 10_000
BCAT_TOTAL_SHARES = 1_000_000
BCAT_FLOAT_SHARES = 650_000
BCAT_BASE_VOLATILITY_BPS = 70
BCAT_VOLATILITY_AMPLIFIER_BPS = 150
BCAT_LIQUIDITY_SHARES = 25_000
BCAT_FAIR_VALUE_CENTS = 10_000
BCAT_MEAN_REVERSION_BPS = 35
BCAT_MAX_TICK_CHANGE_BPS = 450
BCAT_NEWS_CADENCE_HOURS = STOCK_NEWS_CADENCE_HOURS
BCAT_INDIVIDUAL_OWNERSHIP_CAP = (
    BCAT_FLOAT_SHARES * STOCK_INDIVIDUAL_OWNERSHIP_CAP_BPS // STOCK_BPS_DENOMINATOR
)


def test_stock_news_prompt_and_fallback_templates_are_safe_and_bounded() -> None:
    """Stock news copy should stay fictional, safe, and bounded for market impact."""
    assert "fictional" in STOCK_NEWS_PROMPT
    assert "Do not claim this is real financial news" in STOCK_NEWS_PROMPT
    assert "Do not mention real people" in STOCK_NEWS_PROMPT
    assert "-180 to 180" in STOCK_NEWS_PROMPT
    assert "market context" in STOCK_NEWS_PROMPT
    assert all(
        -180 <= sentiment_bps <= 180 for _template, sentiment_bps in STOCK_NEWS_FALLBACK_TEMPLATES
    )
    assert all(
        "{name}" in template or "{symbol}" in template
        for template, _sentiment_bps in STOCK_NEWS_FALLBACK_TEMPLATES
    )


def test_stock_fallback_news_uses_absurd_templates() -> None:
    """Deterministic fallback news should match the same goofy style as AI news."""
    profile = StockProfileView(
        symbol=BCAT_SYMBOL,
        name=BCAT_NAME,
        category=BCAT_CATEGORY,
        price_cents=BCAT_INITIAL_PRICE_CENTS,
        previous_close_price_cents=BCAT_INITIAL_PRICE_CENTS,
        day_open_price_cents=BCAT_INITIAL_PRICE_CENTS,
        total_shares=BCAT_TOTAL_SHARES,
        float_shares=BCAT_FLOAT_SHARES,
        base_volatility_bps=BCAT_BASE_VOLATILITY_BPS,
        volatility_amplifier_bps=BCAT_VOLATILITY_AMPLIFIER_BPS,
        liquidity_shares=BCAT_LIQUIDITY_SHARES,
        fair_value_cents=BCAT_FAIR_VALUE_CENTS,
        mean_reversion_bps=BCAT_MEAN_REVERSION_BPS,
        max_tick_change_bps=BCAT_MAX_TICK_CHANGE_BPS,
        news_cadence_hours=BCAT_NEWS_CADENCE_HOURS,
        updated_at=datetime(2026, 1, 1),
    )
    generated = tuple(
        stock_db._fallback_generated_news(
            context=_bcat_news_context(profile=profile, change_bps=-250, pressure_bps=-80),
            now=datetime(2026, 1, 1) + timedelta(hours=BCAT_NEWS_CADENCE_HOURS * index),
        )
        for index in range(len(STOCK_NEWS_FALLBACK_TEMPLATES))
    )
    assert any("爆胎" in news.headline for news in generated)
    assert all(news.source == "template" for news in generated)
    assert all(-180 <= news.sentiment_bps <= 180 for news in generated)

    bullish = stock_db._fallback_generated_news(
        context=_bcat_news_context(profile=profile, change_bps=250, pressure_bps=80),
        now=datetime(2026, 1, 1),
    )
    bearish = stock_db._fallback_generated_news(
        context=_bcat_news_context(profile=profile, change_bps=-250, pressure_bps=-80),
        now=datetime(2026, 1, 1),
    )
    assert bullish.sentiment_bps > 0
    assert bearish.sentiment_bps < 0


def _rng(seed: int) -> Random:
    """Returns a deterministic test RNG."""
    return Random(seed)  # noqa: S311 -- deterministic market tests require seeded Random


@pytest.fixture
async def stock_empty_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Per-test SQLite file with no stock company rows."""
    stock_db_path = tmp_path / "stock.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{stock_db_path}")
    monkeypatch.setattr(stock_db, "_engine", engine)
    monkeypatch.setattr(stock_db, "_schema_ready_for", None)
    stock_db._operation_locks.clear()
    stock_db._operation_lock_refcounts.clear()
    monkeypatch.setattr(stock_db, "_operation_locks_loop", None)
    stock_db._market_locks.clear()
    stock_db._market_lock_refcounts.clear()
    monkeypatch.setattr(stock_db, "_market_locks_loop", None)
    monkeypatch.setattr(stock_db, "_news_generation_lock", None)
    monkeypatch.setattr(stock_db, "_news_generation_lock_loop", None)
    monkeypatch.setattr(stock_db, "_news_provider_semaphore", None)
    monkeypatch.setattr(stock_db, "_news_provider_semaphore_loop", None)
    yield
    await engine.dispose()


@pytest.fixture
async def stock_isolated_db(stock_empty_db: None) -> None:
    """Per-test SQLite file with one DB-managed test stock."""
    await _upsert_bcat_profile()


async def _upsert_bcat_profile(
    price_cents: int = BCAT_INITIAL_PRICE_CENTS,
    name: str = BCAT_NAME,
    category: str = BCAT_CATEGORY,
) -> StockProfileView:
    """Creates or updates the DB-owned BCAT test profile."""
    return await stock_db.upsert_stock_profile(
        profile=StockProfileUpsert(
            symbol=BCAT_SYMBOL,
            name=name,
            category=category,
            price_cents=price_cents,
            total_shares=BCAT_TOTAL_SHARES,
            float_shares=BCAT_FLOAT_SHARES,
            base_volatility_bps=BCAT_BASE_VOLATILITY_BPS,
            volatility_amplifier_bps=BCAT_VOLATILITY_AMPLIFIER_BPS,
            liquidity_shares=BCAT_LIQUIDITY_SHARES,
            fair_value_cents=BCAT_FAIR_VALUE_CENTS,
            mean_reversion_bps=BCAT_MEAN_REVERSION_BPS,
            max_tick_change_bps=BCAT_MAX_TICK_CHANGE_BPS,
            news_cadence_hours=BCAT_NEWS_CADENCE_HOURS,
        ),
        now=datetime(2026, 1, 1),
    )


async def _upsert_illiquid_profile() -> StockProfileView:
    """Creates a stock whose small orders visibly move execution price."""
    return await stock_db.upsert_stock_profile(
        profile=StockProfileUpsert(
            symbol="THIN",
            name="薄量測試股份有限公司",
            category="測試",
            price_cents=10_000,
            total_shares=1_000,
            float_shares=1_000,
            base_volatility_bps=0,
            volatility_amplifier_bps=0,
            liquidity_shares=10,
            fair_value_cents=10_000,
            mean_reversion_bps=0,
            max_tick_change_bps=1_000,
            news_cadence_hours=8,
        ),
        now=datetime(2026, 1, 1),
    )


def _bcat_news_context(
    profile: StockProfileView, change_bps: int = 0, pressure_bps: int = 0
) -> StockNewsGenerationContext:
    """Builds a deterministic BCAT news generation context."""
    return StockNewsGenerationContext(
        profile=profile,
        change_cents=profile.price_cents * change_bps // 10_000,
        change_bps=change_bps,
        pressure_bps=pressure_bps,
        buy_side_shares=0,
        sell_side_shares=0,
        net_order_shares=0,
        recent_news_sentiment_bps=0,
        lookback_hours=24,
    )


def test_stock_cash_rounding_and_price_format() -> None:
    """Prices are cent-based and cash conversion is explicit."""
    assert cash_ceil(cents=10_001) == 101
    assert cash_floor(cents=10_001) == 100
    assert format_price(price_cents=10_001) == "100.01"


def test_stock_tick_helpers_noop_and_compress_backlog() -> None:
    """Lazy ticks no-op inside one interval and compress long backlogs."""
    latest = datetime(2026, 1, 1, 0, 0)
    assert tick_boundaries_to_apply(latest_tick_at=latest, now=latest + timedelta(minutes=4)) == ()
    assert tick_boundaries_to_apply(
        latest_tick_at=latest, now=latest + timedelta(seconds=STOCK_TICK_SECONDS)
    ) == (datetime(2026, 1, 1, 0, 5, tzinfo=TAIWAN_TIMEZONE),)

    backlog = tick_boundaries_to_apply(latest_tick_at=latest, now=latest + timedelta(hours=100))
    assert len(backlog) == MAX_TICKS_PER_INTERACTION
    assert backlog[-1] == tick_boundary(dt=latest + timedelta(hours=100))
    compressed_day = tick_boundaries_to_apply(
        latest_tick_at=latest, now=latest + timedelta(hours=25)
    )
    assert datetime(2026, 1, 2, 0, 0, tzinfo=TAIWAN_TIMEZONE) in compressed_day


def test_stock_tick_boundary_treats_naive_datetime_as_taipei() -> None:
    """Naive SQLite-style datetimes are interpreted as Asia/Taipei."""
    naive = datetime(2026, 1, 1, 1, 23)
    aware = datetime(2026, 1, 1, 1, 23, tzinfo=TAIWAN_TIMEZONE)

    assert tick_boundary(dt=naive) == tick_boundary(dt=aware)


def test_stock_price_formula_is_deterministic_and_clamped() -> None:
    """The pure price formula is deterministic with seeded randomness."""
    first = calculate_next_price_cents(
        previous_price_cents=100,
        news_sentiment_bps=-20_000,
        pressure_bps=-20_000,
        base_volatility_bps=0,
        volatility_amplifier_bps=100,
        fair_value_cents=100,
        mean_reversion_strength_bps=0,
        max_tick_change_bps=500,
        rng=_rng(seed=1),
    )
    second = calculate_next_price_cents(
        previous_price_cents=100,
        news_sentiment_bps=-20_000,
        pressure_bps=-20_000,
        base_volatility_bps=0,
        volatility_amplifier_bps=100,
        fair_value_cents=100,
        mean_reversion_strength_bps=0,
        max_tick_change_bps=500,
        rng=_rng(seed=1),
    )
    assert first == second
    assert first >= 1
    assert decay_news_sentiment(sentiment_bps=500, elapsed_seconds=3 * 60 * 60) == 240
    assert decay_news_sentiment(sentiment_bps=-500, elapsed_seconds=20 * 60 * 60) == 0


def _stock_news_row(
    created_at: datetime, sentiment_bps: int, news_id: str = "test"
) -> stock_db.StockNews:
    """Builds a StockNews ORM row for impulse helper tests."""
    return stock_db.StockNews(
        id=news_id,
        symbol=BCAT_SYMBOL,
        headline="test",
        sentiment_bps=sentiment_bps,
        source="template",
        model="",
        expires_at=None,
        created_at=created_at,
    )


def test_stock_news_impulse_applies_once_at_firing_boundary() -> None:
    """Each news contributes its sentiment exactly once at its firing tick."""
    b0 = tick_boundary(dt=datetime(2026, 1, 1, 0, 0, tzinfo=TAIWAN_TIMEZONE))
    b1 = b0 + timedelta(seconds=STOCK_TICK_SECONDS)
    b2 = b1 + timedelta(seconds=STOCK_TICK_SECONDS)
    boundaries = (b0, b1, b2)
    rows = (
        _stock_news_row(created_at=b0 + timedelta(seconds=30), sentiment_bps=200, news_id="a"),
        _stock_news_row(created_at=b2, sentiment_bps=-150, news_id="b"),
    )
    impulse = stock_db._news_impulse_by_boundary(news_rows=rows, applied_boundaries=boundaries)
    assert impulse[b0] == 200
    assert impulse[b1] == 0
    assert impulse[b2] == -150


def test_stock_news_impulse_clamps_per_news_to_sentiment_limit() -> None:
    """Per-news sentiment is clamped to ±NEWS_SENTIMENT_LIMIT_BPS before summing."""
    b0 = tick_boundary(dt=datetime(2026, 1, 1, 0, 0, tzinfo=TAIWAN_TIMEZONE))
    rows = (_stock_news_row(created_at=b0, sentiment_bps=10_000, news_id="huge"),)
    impulse = stock_db._news_impulse_by_boundary(news_rows=rows, applied_boundaries=(b0,))
    assert impulse[b0] == NEWS_SENTIMENT_LIMIT_BPS


def test_stock_news_impulse_skips_pre_window_news() -> None:
    """News whose tick boundary predates all applied boundaries is dropped."""
    b0 = tick_boundary(dt=datetime(2026, 1, 1, 1, 0, tzinfo=TAIWAN_TIMEZONE))
    older = _stock_news_row(created_at=b0 - timedelta(hours=1), sentiment_bps=180, news_id="stale")
    impulse = stock_db._news_impulse_by_boundary(news_rows=(older,), applied_boundaries=(b0,))
    assert impulse[b0] == 0


def test_stock_news_impulse_routes_skipped_news_to_next_surviving_boundary() -> None:
    """Backlog-compressed boundaries still absorb news that fell in skipped ticks."""
    b_first = tick_boundary(dt=datetime(2026, 1, 1, 0, 0, tzinfo=TAIWAN_TIMEZONE))
    b_skipped = b_first + timedelta(seconds=STOCK_TICK_SECONDS)
    b_next = b_skipped + timedelta(seconds=STOCK_TICK_SECONDS)
    rows = (_stock_news_row(created_at=b_skipped, sentiment_bps=150, news_id="mid"),)
    impulse = stock_db._news_impulse_by_boundary(
        news_rows=rows, applied_boundaries=(b_first, b_next)
    )
    assert impulse[b_first] == 0
    assert impulse[b_next] == 150


def test_stock_price_formula_stays_bounded_under_impulse_news_monte_carlo() -> None:
    """Property test: with impulse news + random pressure, prices stay bounded.

    Seeded Monte Carlo using the aggressive OCTE-like profile (highest volatility
    and biggest amplifier in the production set). If a future change weakens mean
    reversion, removes impulse semantics, or widens tick bounds, the runaway
    behavior surfaces here instead of silently in production.
    """
    base_volatility_bps = 180
    volatility_amplifier_bps = 360
    fair_value_cents = 9_000
    mean_reversion_strength_bps = 55
    max_tick_change_bps = 850

    ticks_per_day = 24 * 60 * 60 // STOCK_TICK_SECONDS
    news_cadence_ticks = 4 * ticks_per_day // 24
    sim_days = 7
    trials = 50

    peak_log_ratios: list[float] = []
    final_log_ratios: list[float] = []
    for trial in range(trials):
        rng = _rng(seed=trial)
        price = fair_value_cents
        pressure_bps = 0
        peak_abs = 0.0
        for tick in range(sim_days * ticks_per_day):
            news_sentiment = 0
            if tick > 0 and tick % news_cadence_ticks == 0:
                news_sentiment = max(
                    -NEWS_SENTIMENT_LIMIT_BPS,
                    min(NEWS_SENTIMENT_LIMIT_BPS, int(rng.gauss(mu=0, sigma=120))),
                )
            pressure_bps = max(-90, min(90, pressure_bps + int(rng.gauss(mu=0, sigma=15))))
            price = calculate_next_price_cents(
                previous_price_cents=price,
                news_sentiment_bps=news_sentiment,
                pressure_bps=pressure_bps,
                base_volatility_bps=base_volatility_bps,
                volatility_amplifier_bps=volatility_amplifier_bps,
                fair_value_cents=fair_value_cents,
                mean_reversion_strength_bps=mean_reversion_strength_bps,
                max_tick_change_bps=max_tick_change_bps,
                rng=rng,
            )
            log_ratio_abs = abs(math.log(price / fair_value_cents))
            peak_abs = max(peak_abs, log_ratio_abs)
        peak_log_ratios.append(peak_abs)
        final_log_ratios.append(abs(math.log(price / fair_value_cents)))

    peak_max = max(peak_log_ratios)
    assert peak_max < math.log(30), (
        f"price ran away during simulation: peak {math.exp(peak_max):.1f}x fair_value"
    )
    sorted_finals = sorted(final_log_ratios)
    final_median = sorted_finals[trials // 2]
    assert final_median < math.log(5), (
        f"price drifted: median final {math.exp(final_median):.1f}x fair_value"
    )


def test_stock_order_flow_pressure_scales_with_liquidity() -> None:
    """Order-flow pressure uses the liquidity bucket instead of saturating on tiny flow."""
    assert pressure_from_order_flow(net_shares=0, liquidity_shares=25_000) == 0
    assert pressure_from_order_flow(net_shares=12_500, liquidity_shares=25_000) == 45
    assert pressure_from_order_flow(net_shares=25_000, liquidity_shares=25_000) == 90
    assert pressure_from_order_flow(net_shares=-50_000, liquidity_shares=25_000) == -90
    assert pressure_from_order_flow(net_shares=1_000, liquidity_shares=0) == 0


def test_stock_execution_price_uses_order_size_and_liquidity() -> None:
    """Large orders execute away from the quote, bounded by the per-stock cap."""
    assert order_impact_bps(shares=0, liquidity_shares=10, max_impact_bps=1_000) == 0
    assert order_impact_bps(shares=1, liquidity_shares=2, max_impact_bps=1) == 1
    assert order_impact_bps(shares=5, liquidity_shares=10, max_impact_bps=1_000) == 500
    assert order_impact_bps(shares=100, liquidity_shares=10, max_impact_bps=1_000) == 1_000
    assert (
        execution_price_cents(
            reference_price_cents=10_000,
            shares=10,
            liquidity_shares=10,
            max_impact_bps=1_000,
            is_buy=True,
        )
        == 11_000
    )
    assert (
        execution_price_cents(
            reference_price_cents=10_000,
            shares=10,
            liquidity_shares=10,
            max_impact_bps=1_000,
            is_buy=False,
        )
        == 9_000
    )


def test_stock_order_flow_decay_preserves_small_trade_pressure() -> None:
    """Small trades retain fractional decay before aggregate pressure conversion."""
    at = datetime(2026, 1, 2, tzinfo=TAIWAN_TIMEZONE)
    pressure_rows = (
        (StockTradeLegType.OPEN_LONG.value, 1, at - timedelta(seconds=1)),
        (StockTradeLegType.OPEN_LONG.value, 1, at - timedelta(seconds=1)),
    )

    assert (
        stock_db._recent_pressure_bps_from_rows(
            pressure_rows=pressure_rows, at=at, liquidity_shares=100
        )
        == 2
    )


async def test_stock_schema_bootstrap_does_not_seed_companies(stock_empty_db: None) -> None:
    """Schema bootstrap creates stock tables but company rows are DB-managed."""
    quotes = await stock_db.list_market_quotes(now=datetime(2026, 1, 1), rng=_rng(seed=1))

    assert quotes == ()
    async with stock_db._engine.connect() as conn:
        column_names = await conn.run_sync(
            lambda sync_conn: {
                table_name: [
                    column["name"]
                    for column in inspect(sync_conn).get_columns(table_name=table_name)
                ]
                for table_name in (
                    "stock_profile",
                    "stock_position",
                    "stock_operation",
                    "stock_trade_leg",
                    "stock_price_tick",
                    "stock_news",
                )
            }
        )
        column_types = await conn.run_sync(
            lambda sync_conn: {
                table_name: {
                    column["name"]: column["type"].__class__.__name__.upper()
                    for column in inspect(sync_conn).get_columns(table_name=table_name)
                }
                for table_name in (
                    "stock_profile",
                    "stock_position",
                    "stock_trade_leg",
                    "stock_price_tick",
                )
            }
        )
    assert "liquidity_shares" in column_names["stock_profile"]
    assert column_names["stock_position"][:3] == ["symbol", "user_id", "user_name"]
    assert column_names["stock_operation"][1:4] == ["symbol", "user_id", "user_name"]
    assert column_names["stock_trade_leg"][3:6] == ["symbol", "user_id", "user_name"]
    assert column_types["stock_profile"]["price_cents"] == "TEXT"
    assert column_types["stock_profile"]["float_shares"] == "TEXT"
    assert column_types["stock_position"]["long_cost_basis"] == "TEXT"
    assert column_types["stock_trade_leg"]["wallet_delta"] == "TEXT"
    assert column_types["stock_price_tick"]["price_cents"] == "TEXT"
    assert "source" in column_names["stock_news"]


async def test_stock_profile_upsert_manages_database_company(stock_empty_db: None) -> None:
    """Company profile data is created and updated through the stock DB."""
    profile = await _upsert_bcat_profile()

    assert profile.symbol == BCAT_SYMBOL
    assert profile.name == BCAT_NAME
    assert profile.price_cents == BCAT_INITIAL_PRICE_CENTS
    assert profile.liquidity_shares == BCAT_LIQUIDITY_SHARES
    profiles = await stock_db.list_stock_profiles()
    assert tuple(profile.symbol for profile in profiles) == (BCAT_SYMBOL,)
    audits = await stock_db.list_stock_supply_audit()
    assert audits[0].available_long_shares == BCAT_FLOAT_SHARES
    assert audits[0].available_short_shares == BCAT_FLOAT_SHARES
    async with stock_db.open_stock_session() as session:
        tick_count = await session.scalar(
            statement=select(stock_db.StockPriceTick).where(
                stock_db.StockPriceTick.symbol == BCAT_SYMBOL
            )
        )
    assert tick_count is not None

    updated = await _upsert_bcat_profile(
        price_cents=12_345, name="資料庫貓科技", category="DB managed"
    )

    assert updated.name == "資料庫貓科技"
    assert updated.category == "DB managed"
    assert updated.price_cents == 12_345
    assert len(await stock_db.list_stock_profiles()) == 1
    async with stock_db.open_stock_session() as session:
        latest_tick = await session.scalar(
            statement=select(stock_db.StockPriceTick)
            .where(stock_db.StockPriceTick.symbol == BCAT_SYMBOL)
            .order_by(stock_db.StockPriceTick.created_at.desc())
            .limit(1)
        )
    assert latest_tick is not None
    assert latest_tick.price_cents == 12_345


async def test_stock_large_numbers_use_text_storage(stock_empty_db: None) -> None:
    """Stock money and share quantities can exceed SQLite's integer ceiling."""
    large_value = 10**30
    now = datetime(2026, 1, 1)
    profile = await stock_db.upsert_stock_profile(
        profile=StockProfileUpsert(
            symbol="BIG",
            name="超大數測試股份有限公司",
            category="測試",
            price_cents=large_value,
            total_shares=large_value * 4,
            float_shares=large_value * 3,
            base_volatility_bps=0,
            volatility_amplifier_bps=0,
            liquidity_shares=large_value,
            fair_value_cents=large_value,
            mean_reversion_bps=1,
            max_tick_change_bps=1,
            news_cadence_hours=8,
        ),
        now=now,
    )
    async with stock_db.open_stock_session() as session:
        session.add(
            instance=stock_db.StockPosition(
                symbol="BIG",
                user_id=1,
                user_name="Large",
                long_shares=large_value,
                long_cost_basis=large_value * 2,
                short_shares=large_value // 2,
                short_entry_value=large_value * 3,
                short_collateral=large_value * 4,
                realized_pnl=large_value * 5,
                version=1,
                updated_at=now,
            )
        )
        session.add(
            instance=stock_db.StockOperation(
                operation_id="large-operation",
                symbol="BIG",
                user_id=1,
                user_name="Large",
                requested_action=StockAction.BUY.value,
                status=StockOperationStatus.APPLIED.value,
                failure_reason="",
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            instance=stock_db.StockTradeLeg(
                operation_id="large-operation",
                leg_order=1,
                symbol="BIG",
                user_id=1,
                user_name="Large",
                leg_type=StockTradeLegType.OPEN_LONG.value,
                shares=large_value,
                price_cents=large_value,
                wallet_delta=-(large_value * 2),
                basis_delta=large_value * 2,
                collateral_delta=0,
                realized_pnl_delta=0,
                created_at=now,
            )
        )
        await session.commit()
        storage_result = await session.execute(
            statement=text(
                """
                SELECT
                    (SELECT typeof(price_cents) FROM stock_profile WHERE symbol = 'BIG'),
                    (SELECT typeof(float_shares) FROM stock_profile WHERE symbol = 'BIG'),
                    (SELECT typeof(long_cost_basis) FROM stock_position WHERE symbol = 'BIG'),
                    (SELECT typeof(wallet_delta) FROM stock_trade_leg WHERE operation_id = 'large-operation'),
                    (SELECT typeof(price_cents) FROM stock_price_tick WHERE symbol = 'BIG')
                """
            )
        )
        storage_row = storage_result.one()

    assert profile.price_cents == large_value
    assert storage_row == ("text", "text", "text", "text", "text")
    audit = (await stock_db.list_stock_supply_audit())[0]
    assert audit.long_shares == large_value
    assert audit.short_shares == large_value // 2
    assert audit.available_long_shares == large_value * 2


def test_stock_profile_upsert_rejects_invalid_share_structure() -> None:
    """DB-owned company payloads reject impossible share counts before persistence."""
    with pytest.raises(ValueError, match="float_shares cannot exceed total_shares"):
        StockProfileUpsert(
            symbol="TEST",
            name="Test Company",
            category="Test",
            price_cents=100,
            total_shares=100,
            float_shares=101,
            base_volatility_bps=1,
            volatility_amplifier_bps=100,
            liquidity_shares=1,
            fair_value_cents=100,
            mean_reversion_bps=1,
            max_tick_change_bps=1,
            news_cadence_hours=8,
        )


async def test_stock_due_news_uses_ai_provider_and_cadence(stock_isolated_db: None) -> None:
    """Due news uses the provider once per cadence bucket and persists metadata."""
    await stock_db.list_market_quotes(now=datetime(2026, 1, 1), rng=_rng(seed=1))
    async with stock_db.open_stock_session() as session:
        await session.execute(
            statement=delete(stock_db.StockNews).where(stock_db.StockNews.symbol == BCAT_SYMBOL)
        )
        await session.commit()
    calls = 0

    async def provider(context: StockNewsGenerationContext) -> StockGeneratedNews:
        """Returns one fake AI news item."""
        nonlocal calls
        calls += 1
        return StockGeneratedNews(
            headline=f"{context.profile.symbol} 測試新聞",
            sentiment_bps=120,
            source="ai",
            model="test-model",
        )

    await stock_db.ensure_due_stock_news(
        news_provider=provider, symbols=(BCAT_SYMBOL,), now=datetime(2026, 1, 2)
    )
    await stock_db.ensure_due_stock_news(
        news_provider=provider, symbols=(BCAT_SYMBOL,), now=datetime(2026, 1, 2, 1)
    )

    async with stock_db.open_stock_session() as session:
        news_result = await session.execute(
            statement=select(stock_db.StockNews).where(stock_db.StockNews.symbol == BCAT_SYMBOL)
        )
        news_rows = news_result.scalars().all()
    assert calls == 1
    assert len(news_rows) == 1
    expected_bucket = int(datetime(2026, 1, 2, tzinfo=TAIWAN_TIMEZONE).timestamp()) // (
        BCAT_NEWS_CADENCE_HOURS * 60 * 60
    )
    assert news_rows[0].id == f"bcat-{expected_bucket}"
    assert news_rows[0].headline == "BCAT 測試新聞"
    assert news_rows[0].source == "ai"
    assert news_rows[0].model == "test-model"


async def test_stock_news_provider_receives_market_context(stock_isolated_db: None) -> None:
    """Generated news can react to daily movement, trade flow, and recent news."""
    now = datetime(2026, 1, 2, 12, 0)
    trade_at = now - timedelta(hours=1)
    async with stock_db.open_stock_session() as session:
        profile = await session.get(stock_db.StockProfile, BCAT_SYMBOL)
        assert profile is not None
        profile.price_cents = 11_000
        profile.previous_close_price_cents = 10_000
        session.add(
            instance=stock_db.StockOperation(
                operation_id="context-test-operation",
                symbol=BCAT_SYMBOL,
                user_id=1,
                user_name="alice",
                requested_action=StockAction.BUY.value,
                status=StockOperationStatus.APPLIED.value,
                failure_reason="",
                created_at=trade_at,
                updated_at=trade_at,
            )
        )
        session.add(
            instance=stock_db.StockTradeLeg(
                operation_id="context-test-operation",
                leg_order=1,
                symbol=BCAT_SYMBOL,
                user_id=1,
                user_name="alice",
                leg_type=StockTradeLegType.OPEN_LONG.value,
                shares=BCAT_LIQUIDITY_SHARES,
                price_cents=10_000,
                wallet_delta=-25_000,
                basis_delta=25_000,
                collateral_delta=0,
                realized_pnl_delta=0,
                created_at=trade_at,
            )
        )
        session.add(
            instance=stock_db.StockNews(
                id="bcat-context-template",
                symbol=BCAT_SYMBOL,
                headline="BCAT 舊新聞",
                sentiment_bps=90,
                source="template",
                model="",
                expires_at=now + timedelta(hours=1),
                created_at=now - timedelta(minutes=30),
            )
        )
        await session.commit()

    contexts: list[StockNewsGenerationContext] = []

    async def provider(context: StockNewsGenerationContext) -> StockGeneratedNews:
        """Records the market context passed to AI news."""
        contexts.append(context)
        return StockGeneratedNews(
            headline=f"{context.profile.symbol} context 測試新聞",
            sentiment_bps=120,
            source="ai",
            model="test-model",
        )

    await stock_db.ensure_due_stock_news(news_provider=provider, symbols=(BCAT_SYMBOL,), now=now)

    assert len(contexts) == 1
    context = contexts[0]
    assert context.profile.symbol == BCAT_SYMBOL
    assert context.change_bps == 1_000
    assert context.change_cents == 1_000
    assert context.buy_side_shares == BCAT_LIQUIDITY_SHARES
    assert context.sell_side_shares == 0
    assert context.net_order_shares == BCAT_LIQUIDITY_SHARES
    assert context.pressure_bps > 0
    assert context.latest_news_headline == "BCAT 舊新聞"
    assert context.recent_news_sentiment_bps > 0


async def test_stock_generated_news_upgrades_template_bucket(stock_isolated_db: None) -> None:
    """AI news can replace deterministic fallback news in the same cadence bucket."""
    profile = await _upsert_bcat_profile()
    now = datetime(year=2026, month=1, day=2)

    async with stock_db.open_stock_session() as session:
        await stock_db._insert_generated_news(
            session=session,
            profile=profile,
            generated=stock_db._fallback_generated_news(
                context=_bcat_news_context(profile=profile), now=now
            ),
            now=now,
        )
        await stock_db._insert_generated_news(
            session=session,
            profile=profile,
            generated=StockGeneratedNews(
                headline="BCAT AI 升級新聞", sentiment_bps=90, source="ai", model="test-model"
            ),
            now=now + timedelta(hours=1),
        )
        await session.commit()

        news_result = await session.execute(
            statement=select(stock_db.StockNews).where(stock_db.StockNews.symbol == BCAT_SYMBOL)
        )
        news_rows = news_result.scalars().all()

    assert len(news_rows) == 1
    assert news_rows[0].headline == "BCAT AI 升級新聞"
    assert news_rows[0].sentiment_bps == 90
    assert news_rows[0].source == "ai"
    assert news_rows[0].model == "test-model"


async def test_stock_due_news_upgrades_template_when_provider_available(
    stock_isolated_db: None,
) -> None:
    """Provider-backed refreshes can upgrade fallback news before the next cadence."""
    now = datetime(year=2026, month=1, day=2)
    await stock_db.ensure_due_stock_news(symbols=(BCAT_SYMBOL,), now=now)
    calls = 0

    async def provider(context: StockNewsGenerationContext) -> StockGeneratedNews:
        """Returns one fake AI news item."""
        nonlocal calls
        calls += 1
        return StockGeneratedNews(
            headline=f"{context.profile.symbol} provider 升級新聞",
            sentiment_bps=100,
            source="ai",
            model="test-model",
        )

    await stock_db.ensure_due_stock_news(
        news_provider=provider, symbols=(BCAT_SYMBOL,), now=now + timedelta(hours=1)
    )

    async with stock_db.open_stock_session() as session:
        news_result = await session.execute(
            statement=select(stock_db.StockNews).where(stock_db.StockNews.symbol == BCAT_SYMBOL)
        )
        news_rows = news_result.scalars().all()

    assert calls == 1
    assert len(news_rows) == 1
    assert news_rows[0].headline == "BCAT provider 升級新聞"
    assert news_rows[0].source == "ai"


async def test_stock_due_news_serializes_concurrent_provider_calls(
    stock_isolated_db: None,
) -> None:
    """Concurrent news refreshes do not pay for duplicate provider calls."""
    await stock_db.list_market_quotes(now=datetime(2026, 1, 1), rng=_rng(seed=1))
    async with stock_db.open_stock_session() as session:
        await session.execute(
            statement=delete(stock_db.StockNews).where(stock_db.StockNews.symbol == BCAT_SYMBOL)
        )
        await session.commit()
    calls = 0

    async def provider(context: StockNewsGenerationContext) -> StockGeneratedNews:
        """Returns one fake AI news item after yielding to the event loop."""
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return StockGeneratedNews(
            headline=f"{context.profile.symbol} concurrent 測試新聞",
            sentiment_bps=80,
            source="ai",
            model="test-model",
        )

    await asyncio.gather(
        stock_db.ensure_due_stock_news(
            news_provider=provider, symbols=(BCAT_SYMBOL,), now=datetime(2026, 1, 2)
        ),
        stock_db.ensure_due_stock_news(
            news_provider=provider, symbols=(BCAT_SYMBOL,), now=datetime(2026, 1, 2)
        ),
    )

    async with stock_db.open_stock_session() as session:
        news_result = await session.execute(
            statement=select(stock_db.StockNews).where(stock_db.StockNews.symbol == BCAT_SYMBOL)
        )
        news_rows = news_result.scalars().all()
    assert calls == 1
    assert len(news_rows) == 1
    assert news_rows[0].headline == "BCAT concurrent 測試新聞"
    assert news_rows[0].source == "ai"


async def test_stock_day_rollover_updates_open_and_previous_close(stock_isolated_db: None) -> None:
    """Crossing Asia/Taipei midnight updates previous close and day open."""
    await stock_db.list_market_quotes(now=datetime(2026, 1, 1, 12, 0), rng=_rng(seed=1))
    latest = datetime(2026, 1, 1, 23, 55)
    async with stock_db.open_stock_session() as session:
        await session.execute(
            statement=delete(stock_db.StockPriceTick).where(
                stock_db.StockPriceTick.symbol == BCAT_SYMBOL
            )
        )
        session.add(
            instance=stock_db.StockPriceTick(
                symbol=BCAT_SYMBOL, created_at=latest, price_cents=10_000
            )
        )
        await session.execute(
            statement=update(stock_db.StockProfile)
            .where(stock_db.StockProfile.symbol == BCAT_SYMBOL)
            .values(
                price_cents=10_000, previous_close_price_cents=10_000, day_open_price_cents=10_000
            )
        )
        await session.commit()

    quotes = await stock_db.list_market_quotes(now=datetime(2026, 1, 2, 1, 0), rng=_rng(seed=0))

    bcat_quote = next(quote for quote in quotes if quote.profile.symbol == BCAT_SYMBOL)
    assert bcat_quote.profile.previous_close_price_cents == 10_000
    assert bcat_quote.profile.day_open_price_cents > 0


async def test_stock_compressed_day_rollover_materializes_midnight(
    stock_isolated_db: None,
) -> None:
    """Compressed backlogs keep the actual midnight boundary for day-open pricing."""
    await _set_bcat_price(price_cents=10_000)

    quotes = await stock_db.list_market_quotes(now=datetime(2026, 1, 2, 1, 0), rng=_rng(seed=0))

    midnight = datetime(2026, 1, 2, 0, 0, tzinfo=TAIWAN_TIMEZONE)
    previous_close_at = datetime(2026, 1, 1, 23, 55, tzinfo=TAIWAN_TIMEZONE)
    async with stock_db.open_stock_session() as session:
        midnight_tick = await session.execute(
            statement=select(stock_db.StockPriceTick.price_cents).where(
                stock_db.StockPriceTick.symbol == BCAT_SYMBOL,
                stock_db.StockPriceTick.created_at == midnight,
            )
        )
        previous_close_tick = await session.execute(
            statement=select(stock_db.StockPriceTick.price_cents).where(
                stock_db.StockPriceTick.symbol == BCAT_SYMBOL,
                stock_db.StockPriceTick.created_at == previous_close_at,
            )
        )

    bcat_quote = next(quote for quote in quotes if quote.profile.symbol == BCAT_SYMBOL)
    assert bcat_quote.profile.day_open_price_cents == midnight_tick.scalar_one()
    assert bcat_quote.profile.previous_close_price_cents == previous_close_tick.scalar_one()


async def test_stock_day_rollover_uses_persisted_boundary_price(
    stock_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Day open follows the stored tick when a concurrent writer wins the boundary."""
    await _set_bcat_price(price_cents=10_000)
    original_insert_tick = stock_db._insert_price_tick_or_existing
    midnight = datetime(2026, 1, 2, 0, 0, tzinfo=TAIWAN_TIMEZONE)
    persisted_open_price = 12_345

    async def insert_tick_after_concurrent_writer(
        session: AsyncSession, symbol: str, price_cents: int, created_at: datetime
    ) -> int:
        if created_at == midnight:
            session.add(
                instance=stock_db.StockPriceTick(
                    symbol=symbol, price_cents=persisted_open_price, created_at=created_at
                )
            )
            await session.flush()
        return await original_insert_tick(
            session=session, symbol=symbol, price_cents=price_cents, created_at=created_at
        )

    monkeypatch.setattr(
        stock_db, "_insert_price_tick_or_existing", insert_tick_after_concurrent_writer
    )

    quotes = await stock_db.list_market_quotes(now=datetime(2026, 1, 2, 1, 0), rng=_rng(seed=0))

    bcat_quote = next(quote for quote in quotes if quote.profile.symbol == BCAT_SYMBOL)
    assert bcat_quote.profile.day_open_price_cents == persisted_open_price


async def test_stock_concurrent_market_advancement_writes_one_tick_per_boundary(
    stock_isolated_db: None,
) -> None:
    """Concurrent quote refreshes do not fork price history at the same tick."""
    await stock_db.list_market_quotes(now=datetime(2026, 1, 1), rng=_rng(seed=1))

    await asyncio.gather(
        stock_db.list_market_quotes(now=datetime(2026, 1, 1, 2), rng=_rng(seed=1)),
        stock_db.list_market_quotes(now=datetime(2026, 1, 1, 2), rng=_rng(seed=1)),
    )

    async with stock_db.open_stock_session() as session:
        result = await session.execute(
            statement=select(stock_db.StockPriceTick.created_at).where(
                stock_db.StockPriceTick.symbol == BCAT_SYMBOL
            )
        )
    tick_boundaries = result.scalars().all()
    assert len(tick_boundaries) == len(set(tick_boundaries))


async def test_stock_market_advancement_starts_write_transaction(
    stock_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quote refreshes use a SQLite write transaction before market reads."""
    original_begin_immediate = stock_db._begin_immediate
    calls = 0

    async def begin_immediate(session: AsyncSession) -> None:
        nonlocal calls
        calls += 1
        await original_begin_immediate(session=session)

    monkeypatch.setattr(stock_db, "_begin_immediate", begin_immediate)

    quotes = await stock_db.list_market_quotes(now=datetime(2026, 1, 1), rng=_rng(seed=1))

    assert calls == len(quotes)


async def test_stock_buy_long_debits_wallet_and_writes_ledger(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Buying long debits cash, opens position, and writes operation + leg."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)

    result = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="3",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert result.success
    assert result.balance_after == 700
    assert result.position.long_shares == 3
    assert result.position.long_cost_basis == 300
    assert result.legs[0].leg_type == StockTradeLegType.OPEN_LONG
    async with stock_db.open_stock_session() as session:
        operation = await session.get(stock_db.StockOperation, result.operation_id)
        assert operation is not None
        assert operation.status == StockOperationStatus.APPLIED.value
        assert operation.user_name == "alice"
        leg_result = await session.execute(statement=select(stock_db.StockTradeLeg))
        leg = leg_result.scalar_one()
        assert leg.user_name == "alice"


async def test_stock_trade_refreshes_last_seen_user_name(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """A stable user ID can update its stored display name on later trades."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)
    await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    result = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice_renamed",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1, 0, 1),
        rng=_rng(seed=1),
    )

    assert result.success
    assert result.position.user_name == "alice_renamed"
    detail = await stock_db.get_stock_detail(symbol=BCAT_SYMBOL, user_id=1)
    assert detail.position.user_name == "alice_renamed"
    async with stock_db.open_stock_session() as session:
        position = await session.get(stock_db.StockPosition, (BCAT_SYMBOL, 1))
        assert position is not None
        assert position.user_name == "alice_renamed"
        operation = await session.get(stock_db.StockOperation, result.operation_id)
        assert operation is not None
        assert operation.user_name == "alice_renamed"
    async with open_session() as session:
        wallet = await session.get(UserWallet, 1)
        assert wallet is not None
        assert wallet.name == "alice_renamed"


async def test_stock_detail_shows_stock_level_trades_and_positions(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Stock detail exposes public trade history and non-zero positions across users."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)
    await adjust_balance(user_id=2, name="bob", delta=1_000)
    await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=2,
        user_name="bob",
        requested_action=StockAction.SHORT,
        quantity="1",
        now=datetime(2026, 1, 1, 0, 1),
        rng=_rng(seed=1),
    )

    detail = await stock_db.get_stock_detail(symbol=BCAT_SYMBOL, user_id=3, user_name="carol")

    assert detail.position.user_name == "carol"
    assert {trade.user_name for trade in detail.recent_trades} == {"alice", "bob"}
    assert {position.user_name for position in detail.public_positions} == {"alice", "bob"}
    assert any(position.long_shares == 1 for position in detail.public_positions)
    assert any(position.short_shares == 1 for position in detail.public_positions)


async def test_stock_portfolio_lists_non_zero_positions_and_values(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Stock portfolio values long and short positions from current quotes."""
    await _upsert_illiquid_profile()
    await adjust_balance(user_id=1, name="alice", delta=10_000)
    await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="2",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    await stock_db.settle_stock_operation(
        symbol="THIN",
        user_id=1,
        user_name="alice",
        requested_action=StockAction.SHORT,
        quantity="3",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    portfolio = await stock_db.get_stock_portfolio(
        user_id=1, now=datetime(2026, 1, 1), rng=_rng(seed=1)
    )
    holdings = {holding.symbol: holding for holding in portfolio.holdings}
    bcat = holdings[BCAT_SYMBOL]
    thin = holdings["THIN"]

    assert set(holdings) == {BCAT_SYMBOL, "THIN"}
    assert bcat.long_shares == 2
    assert bcat.long_market_value == cash_floor(cents=bcat.price_cents * bcat.long_shares)
    assert bcat.equity_value == bcat.long_market_value
    assert thin.short_shares == 3
    assert thin.short_cover_cost == cash_ceil(cents=thin.price_cents * thin.short_shares)
    assert thin.unrealized_pnl == thin.short_entry_value - thin.short_cover_cost
    assert (
        thin.equity_value == thin.short_collateral + thin.short_entry_value - thin.short_cover_cost
    )
    assert portfolio.equity_value == bcat.equity_value + thin.equity_value
    assert portfolio.unrealized_pnl == bcat.unrealized_pnl + thin.unrealized_pnl
    assert portfolio.realized_pnl == 0


async def test_stock_portfolio_short_cache_and_invalidation(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Production portfolio reads use a short cache that stock writes can clear."""
    await adjust_balance(user_id=1, name="alice", delta=1_000_000)
    await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    first = await stock_db.get_stock_portfolio(user_id=1)
    async with stock_db.open_stock_session() as session:
        await session.execute(
            statement=update(stock_db.StockPosition)
            .where(
                stock_db.StockPosition.symbol == BCAT_SYMBOL, stock_db.StockPosition.user_id == 1
            )
            .values(long_shares=9, long_cost_basis=90_000)
        )
        await session.commit()

    cached = await stock_db.get_stock_portfolio(user_id=1)
    assert cached.holdings[0].long_shares == first.holdings[0].long_shares
    stock_db.invalidate_stock_portfolio_cache(user_id=1)
    refreshed = await stock_db.get_stock_portfolio(user_id=1)
    assert refreshed.holdings[0].long_shares == 9


async def test_stock_oversized_buy_defaults_to_affordable_all(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Numeric buy requests above the spendable balance clamp to the affordable maximum."""
    await adjust_balance(user_id=1, name="alice", delta=100)

    result = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="2",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert result.success
    assert result.shares == 1
    assert result.balance_after == 0
    detail = await stock_db.get_stock_detail(symbol=BCAT_SYMBOL, user_id=1)
    assert detail.position.long_shares == 1


async def test_stock_buy_clamps_to_remaining_float(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """New long exposure cannot exceed the DB-managed floating share supply."""
    await adjust_balance(user_id=1, name="alice", delta=100_000_000)
    await adjust_balance(user_id=2, name="bob", delta=100_000_000)
    await adjust_balance(user_id=3, name="carol", delta=100_000_000)
    await adjust_balance(user_id=4, name="dave", delta=100_000_000)

    first = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity=f"{BCAT_INDIVIDUAL_OWNERSHIP_CAP:,}",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    second = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=2,
        user_name="bob",
        requested_action=StockAction.BUY,
        quantity=f"{BCAT_INDIVIDUAL_OWNERSHIP_CAP:,}",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    assert first.success
    assert second.success

    result = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=3,
        user_name="carol",
        requested_action=StockAction.BUY,
        quantity=f"{BCAT_FLOAT_SHARES + 10:,}",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert result.success
    assert result.shares == BCAT_FLOAT_SHARES - BCAT_INDIVIDUAL_OWNERSHIP_CAP * 2
    detail = await stock_db.get_stock_detail(symbol=BCAT_SYMBOL, user_id=3)
    assert detail.position.long_shares == BCAT_FLOAT_SHARES - BCAT_INDIVIDUAL_OWNERSHIP_CAP * 2
    audits = await stock_db.list_stock_supply_audit()
    bcat_audit = next(audit for audit in audits if audit.symbol == BCAT_SYMBOL)
    assert bcat_audit.available_long_shares == 0

    blocked = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=4,
        user_name="dave",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert not blocked.success
    assert "流通股" in blocked.error


async def test_stock_buy_clamps_to_individual_ownership_cap(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """A single user cannot open long exposure above 49% of floating shares."""
    await adjust_balance(user_id=1, name="alice", delta=100_000_000)
    await adjust_balance(user_id=2, name="bob", delta=100_000_000)

    numeric = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity=f"{BCAT_FLOAT_SHARES:,}",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    all_in = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=2,
        user_name="bob",
        requested_action=StockAction.BUY,
        quantity="ALL",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert numeric.success
    assert numeric.shares == BCAT_INDIVIDUAL_OWNERSHIP_CAP
    assert all_in.success
    assert all_in.shares == BCAT_INDIVIDUAL_OWNERSHIP_CAP

    blocked = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert not blocked.success
    assert "49%" in blocked.error


async def test_stock_large_buy_uses_execution_slippage(
    stock_empty_db: None, economy_isolated_db: None
) -> None:
    """Buy-side settlement stores the execution price after liquidity impact."""
    await _upsert_illiquid_profile()
    await adjust_balance(user_id=1, name="alice", delta=2_000)

    result = await stock_db.settle_stock_operation(
        symbol="THIN",
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="10",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert result.success
    assert result.price_cents == 11_000
    assert result.legs[0].price_cents == 11_000
    assert result.legs[0].wallet_delta == -1_100
    assert result.balance_after == 900


async def test_stock_zero_affordable_buy_leaves_stock_untouched(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Oversized numeric requests still fail with the real root cause when nothing is executable."""
    result = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert not result.success
    assert "餘額不足" in result.error
    detail = await stock_db.get_stock_detail(symbol=BCAT_SYMBOL, user_id=1)
    assert detail.position.long_shares == 0
    async with stock_db.open_stock_session() as session:
        legs = await session.execute(statement=select(stock_db.StockTradeLeg))
        assert legs.scalars().all() == []


async def test_stock_long_round_trip_uses_integer_basis(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Same-price long round trips expose ceil/floor spread and consume dust."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)
    await _set_bcat_price(price_cents=10_001)

    buy = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    sell = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.SHORT,
        quantity="ALL",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert buy.balance_after == 899
    assert sell.balance_after == 999
    assert sell.position.long_shares == 0
    assert sell.position.long_cost_basis == 0
    assert sell.position.realized_pnl == -1


async def test_stock_short_round_trip_uses_collateral_and_integer_entry(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Shorting locks collateral and same-price cover reflects spread."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)
    await _set_bcat_price(price_cents=10_001)

    opened = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.SHORT,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    covered = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="ALL",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert opened.balance_after == 899
    assert opened.position.short_collateral == 101
    assert opened.position.short_entry_value == 100
    assert covered.balance_after == 999
    assert covered.position.short_shares == 0
    assert covered.position.short_collateral == 0
    assert covered.position.short_entry_value == 0
    assert covered.position.realized_pnl == -1


async def test_stock_oversized_short_defaults_to_affordable_all(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Numeric short requests above the collateral balance clamp to the affordable maximum."""
    await adjust_balance(user_id=1, name="alice", delta=100)

    result = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.SHORT,
        quantity="2",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert result.success
    assert result.shares == 1
    assert result.balance_after == 0
    assert result.position.short_shares == 1


async def test_stock_short_clamps_to_available_borrow(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """New short exposure cannot exceed the DB-managed floating share supply."""
    await adjust_balance(user_id=1, name="alice", delta=100_000_000)

    result = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.SHORT,
        quantity=str(BCAT_FLOAT_SHARES + 10),
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert result.success
    assert result.shares == BCAT_FLOAT_SHARES
    assert result.position.short_shares == BCAT_FLOAT_SHARES
    audits = await stock_db.list_stock_supply_audit()
    bcat_audit = next(audit for audit in audits if audit.symbol == BCAT_SYMBOL)
    assert bcat_audit.available_short_shares == 0

    blocked = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.SHORT,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert not blocked.success
    assert "借券" in blocked.error


async def test_stock_pending_operations_reserve_supply(
    stock_isolated_db: None, economy_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-final operation legs reserve float and borrow capacity until reconciliation."""
    await _upsert_illiquid_profile()
    await adjust_balance(user_id=1, name="alice", delta=200_000)
    await adjust_balance(user_id=2, name="bob", delta=1_000)
    await adjust_balance(user_id=3, name="carol", delta=100_000_000)
    await adjust_balance(user_id=4, name="dave", delta=1_000)
    await adjust_balance(user_id=5, name="erin", delta=200_000)
    await adjust_balance(user_id=6, name="frank", delta=200_000)
    original_apply = stock_db.apply_ordered_wallet_deltas

    async def fail_wallet(**_kwargs: object) -> OrderedWalletDeltaResult:
        """Simulates uncertainty after the stock operation reserves market supply."""
        raise RuntimeError("wallet unavailable")

    monkeypatch.setattr(stock_db, "apply_ordered_wallet_deltas", fail_wallet)
    first_pending_long = await stock_db.settle_stock_operation(
        symbol="THIN",
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1,000",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    second_pending_long = await stock_db.settle_stock_operation(
        symbol="THIN",
        user_id=5,
        user_name="erin",
        requested_action=StockAction.BUY,
        quantity="1,000",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    final_pending_long = await stock_db.settle_stock_operation(
        symbol="THIN",
        user_id=6,
        user_name="frank",
        requested_action=StockAction.BUY,
        quantity="1,000",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    pending_short = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=3,
        user_name="carol",
        requested_action=StockAction.SHORT,
        quantity=str(BCAT_FLOAT_SHARES),
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    monkeypatch.setattr(stock_db, "apply_ordered_wallet_deltas", original_apply)

    blocked_long = await stock_db.settle_stock_operation(
        symbol="THIN",
        user_id=2,
        user_name="bob",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    blocked_short = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=4,
        user_name="dave",
        requested_action=StockAction.SHORT,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    audits = {audit.symbol: audit for audit in await stock_db.list_stock_supply_audit()}

    assert first_pending_long.status == StockOperationStatus.RECONCILE_REQUIRED
    assert first_pending_long.shares == 490
    assert second_pending_long.status == StockOperationStatus.RECONCILE_REQUIRED
    assert second_pending_long.shares == 490
    assert final_pending_long.status == StockOperationStatus.RECONCILE_REQUIRED
    assert final_pending_long.shares == 20
    assert pending_short.status == StockOperationStatus.RECONCILE_REQUIRED
    assert not blocked_long.success
    assert "流通股" in blocked_long.error
    assert not blocked_short.success
    assert "借券" in blocked_short.error
    assert audits["THIN"].long_shares == 1_000
    assert audits["THIN"].available_long_shares == 0
    assert audits["THIN"].non_final_operations == 3
    assert audits[BCAT_SYMBOL].short_shares == BCAT_FLOAT_SHARES
    assert audits[BCAT_SYMBOL].available_short_shares == 0
    assert audits[BCAT_SYMBOL].non_final_operations == 1


async def test_stock_cover_can_use_withheld_short_entry_value(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Cover can consume withheld short proceeds when spendable balance is zero."""
    await adjust_balance(user_id=1, name="alice", delta=100)
    await _set_bcat_price(price_cents=10_000)
    opened = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.SHORT,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    await _set_bcat_price(price_cents=20_000)

    covered = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="ALL",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert opened.balance_after == 0
    assert covered.success
    assert covered.balance_after == 0
    assert covered.wallet_delta == 0
    assert covered.position.short_shares == 0
    assert covered.position.short_collateral == 0
    assert covered.position.short_entry_value == 0
    assert covered.position.realized_pnl == -100
    async with open_session() as session:
        wallet = await session.get(UserWallet, 1)
        assert wallet is not None
        assert wallet.total_earned == 300
        assert wallet.total_spent == 300


async def test_stock_compound_operation_uses_ordered_wallet_legs(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Cover plus open-long writes ordered legs and preserves gross invariant."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)
    opened = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.SHORT,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    assert opened.success

    result = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="2",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert [leg.leg_type for leg in result.legs] == [
        StockTradeLegType.COVER_SHORT,
        StockTradeLegType.OPEN_LONG,
    ]
    async with open_session() as session:
        wallet = await session.get(UserWallet, 1)
        assert wallet is not None
        assert wallet.total_earned - wallet.total_spent == wallet.balance
        assert wallet.balance == result.balance_after


async def test_stock_concurrent_trades_do_not_reuse_stale_position(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Same-user same-stock submissions are serialized by the service lock."""
    await adjust_balance(user_id=1, name="alice", delta=100)

    results = await asyncio_gather_stock_buys()

    assert sum(result.success for result in results) == 1
    detail = await stock_db.get_stock_detail(symbol=BCAT_SYMBOL, user_id=1)
    assert detail.position.long_shares == 1
    assert stock_db._operation_locks == {}
    assert stock_db._operation_lock_refcounts == {}


async def asyncio_gather_stock_buys() -> tuple[StockSettlementResult, StockSettlementResult]:
    """Runs two concurrent buys for the concurrency test."""
    return await asyncio.gather(
        stock_db.settle_stock_operation(
            symbol=BCAT_SYMBOL,
            user_id=1,
            user_name="alice",
            requested_action=StockAction.BUY,
            quantity="1",
            now=datetime(2026, 1, 1),
            rng=_rng(seed=1),
        ),
        stock_db.settle_stock_operation(
            symbol=BCAT_SYMBOL,
            user_id=1,
            user_name="alice",
            requested_action=StockAction.BUY,
            quantity="1",
            now=datetime(2026, 1, 1),
            rng=_rng(seed=1),
        ),
    )


async def test_stock_reconciliation_helper_lists_non_final_operations(
    stock_isolated_db: None, economy_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wallet-side uncertainty is surfaced as a reconciliation operation."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)

    async def fail_wallet(**_kwargs: object) -> OrderedWalletDeltaResult:
        """Simulates a wallet-side failure after stock commit."""
        raise RuntimeError("wallet unavailable")

    monkeypatch.setattr(stock_db, "apply_ordered_wallet_deltas", fail_wallet)
    result = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert not result.success
    assert result.status == StockOperationStatus.RECONCILE_REQUIRED
    pending = await stock_db.list_reconciliation_operations()
    assert len(pending) == 1
    assert pending[0].operation_id == result.operation_id
    assert pending[0].user_name == "alice"
    assert pending[0].legs[0].wallet_delta == -100
    assert pending[0].legs[0].user_name == "alice"
    detail = await stock_db.get_stock_detail(symbol=BCAT_SYMBOL, user_id=1)
    assert detail.position.long_shares == 0


async def test_stock_reconciliation_blocks_later_trades(
    stock_isolated_db: None, economy_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-final operation blocks more trading for the same user and symbol."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)
    original_apply = stock_db.apply_ordered_wallet_deltas

    async def fail_wallet(**_kwargs: object) -> OrderedWalletDeltaResult:
        """Simulates uncertain wallet application."""
        raise RuntimeError("wallet unavailable")

    monkeypatch.setattr(stock_db, "apply_ordered_wallet_deltas", fail_wallet)
    first = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )
    monkeypatch.setattr(stock_db, "apply_ordered_wallet_deltas", original_apply)

    second = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert first.status == StockOperationStatus.RECONCILE_REQUIRED
    assert not second.success
    assert second.operation_id == first.operation_id
    assert "未完成" in second.error


async def test_stock_wallet_cancellation_marks_reconciliation(
    stock_isolated_db: None, economy_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelled wallet application leaves an explicit reconciliation marker."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)

    async def cancel_wallet(**_kwargs: object) -> OrderedWalletDeltaResult:
        """Simulates cancellation while the wallet operation is in flight."""
        raise asyncio.CancelledError

    monkeypatch.setattr(stock_db, "apply_ordered_wallet_deltas", cancel_wallet)

    with pytest.raises(asyncio.CancelledError):
        await stock_db.settle_stock_operation(
            symbol=BCAT_SYMBOL,
            user_id=1,
            user_name="alice",
            requested_action=StockAction.BUY,
            quantity="1",
            now=datetime(2026, 1, 1),
            rng=_rng(seed=1),
        )

    pending = await stock_db.list_reconciliation_operations()
    assert len(pending) == 1
    assert pending[0].status == StockOperationStatus.RECONCILE_REQUIRED
    assert "cancelled" in pending[0].failure_reason


async def test_stock_wallet_reject_does_not_apply_position(
    stock_isolated_db: None, economy_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full-debit wallet rejection finalizes as failed without a stock position."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)

    async def reject_wallet(**_kwargs: object) -> None:
        """Simulates a wallet race that makes the debit impossible."""
        return

    monkeypatch.setattr(stock_db, "apply_ordered_wallet_deltas", reject_wallet)
    result = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert not result.success
    assert result.status == StockOperationStatus.FAILED
    detail = await stock_db.get_stock_detail(symbol=BCAT_SYMBOL, user_id=1)
    assert detail.position.long_shares == 0
    assert await stock_db.list_reconciliation_operations() == ()
    async with stock_db.open_stock_session() as session:
        operation = await session.get(stock_db.StockOperation, result.operation_id)
        assert operation is not None
        assert operation.status == StockOperationStatus.FAILED.value


async def test_stock_success_records_wallet_applied_before_final_status(
    stock_isolated_db: None, economy_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful cross-DB operations pass through the wallet_applied lifecycle."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)
    statuses: list[StockOperationStatus] = []
    original_mark_operation = stock_db._mark_operation

    async def record_mark_operation(
        operation_id: str, status: StockOperationStatus, failure_reason: str
    ) -> None:
        """Records lifecycle updates while preserving the real stock update."""
        statuses.append(status)
        await original_mark_operation(
            operation_id=operation_id, status=status, failure_reason=failure_reason
        )

    monkeypatch.setattr(stock_db, "_mark_operation", record_mark_operation)

    result = await stock_db.settle_stock_operation(
        symbol=BCAT_SYMBOL,
        user_id=1,
        user_name="alice",
        requested_action=StockAction.BUY,
        quantity="1",
        now=datetime(2026, 1, 1),
        rng=_rng(seed=1),
    )

    assert result.success
    assert statuses == [StockOperationStatus.WALLET_APPLIED]
    async with stock_db.open_stock_session() as session:
        operation = await session.get(stock_db.StockOperation, result.operation_id)
        assert operation is not None
        assert operation.status == StockOperationStatus.APPLIED.value


async def test_ordered_wallet_deltas_do_not_touch_casino_counters(
    economy_isolated_db: None,
) -> None:
    """Stock wallet legs use gross totals without casino side effects."""
    await adjust_balance(user_id=1, name="alice", delta=1_000)
    result = await apply_ordered_wallet_deltas(
        user_id=1, name="alice", deltas=(WalletDeltaLeg(delta=-100), WalletDeltaLeg(delta=80))
    )

    assert result is not None
    assert result.new_balance == 980
    async with open_session() as session:
        wallet = await session.get(UserWallet, 1)
        assert wallet is not None
        assert wallet.total_earned - wallet.total_spent == wallet.balance


def test_stock_chart_generates_non_empty_image_with_too_few_ticks() -> None:
    """Chart rendering works with one tick."""
    image = build_price_chart(
        ticks=(
            stock_db.StockPriceTickView(
                symbol=BCAT_SYMBOL,
                price_cents=BCAT_INITIAL_PRICE_CENTS,
                created_at=datetime(2026, 1, 1),
            ),
        )
    )
    assert image.startswith(b"\x89PNG")
    assert len(image) > 100


async def _set_bcat_price(price_cents: int) -> None:
    """Pins BCAT to a deterministic price for settlement tests."""
    await stock_db.list_market_quotes(now=datetime(2026, 1, 1), rng=_rng(seed=1))
    async with stock_db.open_stock_session() as session:
        now = datetime(2026, 1, 1)
        await session.execute(
            statement=update(stock_db.StockProfile)
            .where(stock_db.StockProfile.symbol == BCAT_SYMBOL)
            .values(
                price_cents=price_cents,
                previous_close_price_cents=price_cents,
                day_open_price_cents=price_cents,
                updated_at=now,
            )
        )
        await session.execute(
            statement=update(stock_db.StockPriceTick)
            .where(stock_db.StockPriceTick.symbol == BCAT_SYMBOL)
            .values(price_cents=price_cents, created_at=now)
        )
        await session.commit()
