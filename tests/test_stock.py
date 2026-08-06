"""Pins the simulated stock market: the price formula, its store, and cross-database settlement.

Covers `services/stock/market.py` (the pure arithmetic: the price formula, tick boundaries and
the guardrails), `services/stock/database.py` (`stock.db`, lazy tick advancement, the news cadence
and the single settlement entry point), the news copy in `services/stock/prompts.py` and the chart
in `cogs/stock/chart.py`. Cash is not in either module, so the settlement tests pair
`stock_isolated_db` with `economy_isolated_db` and read the wallet back out of
`services/economy/database.py`.

What is pinned here, and why each is worth pinning:

- The guardrails. A company profile is operator-written data with no ceiling of its own, so the
  caps are the only thing between a mis-tuned row and a money printer: the global per-tick change
  ceiling, the Taiwan-style daily band around the previous close, the volatility scale, the
  order-flow pressure limit and the per-order slippage cap. They were set by offline simulation,
  which is why the seeded Monte Carlo run exists — it is what notices a later change that weakens
  mean reversion, drops impulse semantics or widens the bounds.
- News as a one-shot impulse. Each headline contributes its clamped sentiment exactly once, at the
  first applied boundary at or after its own, and a compressed backlog carries a headline out of a
  dropped boundary instead of losing it. The decayed value is a different mechanic entirely and
  only ever feeds the prompt writing the next headline.
- Lazy advancement. There is no market loop, so a symbol replays its backlog on the next
  interaction: nothing moves inside one interval, compression keeps the Asia/Taipei midnight the
  daily band is anchored on, and the insert-once tick write is what makes two concurrent advances
  agree on one price history instead of forking it.
- The two-database seam. `stock.db` and `economy.db` cannot commit together, so settlement's
  lifecycle is the whole design: a wallet failure parks the operation at RECONCILE_REQUIRED, where
  it keeps reserving float and blocks that user's next trade on the symbol, a cancellation still
  records that marker, and a rejected debit finalizes as failed with no position written. Rolling
  back or silently retrying is what would leave a position with no cash behind it.
- Integer money. Prices are cents and shares are whole, so a same-price round trip is expected to
  cost the ceil/floor spread rather than break even, and the wallet legs stay gross so the
  economy's `total_earned - total_spent == balance` invariant keeps describing real flow.
- The supply limits, which are this market's own anti-inflation levers: new long exposure clamps
  to remaining float and again to one user's share of it, a short clamps to borrow capacity, and
  an oversized numeric request clamps to what the balance affords instead of failing outright.
- Decimal-text storage. Money and share columns are `StoredInteger` text to escape SQLite's 64-bit
  ceiling, so the schema tests assert TEXT affinity and a 10**30 round trip proves `typeof()`
  never drifted to REAL.

Nothing seeds a company, so the fixtures are local rather than shared: `stock_empty_db` swaps the
module-level engine onto a `tmp_path` file that bootstraps its own schema on the first call, and
`stock_isolated_db` adds the BCAT profile the `BCAT_*` constants below describe. A test that needs
a symbol thin enough for one small order to move the price creates `THIN` itself.
"""

import math
from random import Random
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text, delete, select, update, inspect
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

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
from discordbot.services.stock import database as stock_db
from discordbot.utils.currency import cash_ceil, cash_floor
from discordbot.typings.economy import WalletDeltaLeg, OrderedWalletDeltaResult
from discordbot.cogs.stock.chart import build_price_chart
from discordbot.services.stock.market import (
    TAIWAN_TIMEZONE,
    PRESSURE_LIMIT_BPS,
    DAILY_PRICE_LIMIT_BPS,
    NEWS_SENTIMENT_LIMIT_BPS,
    GLOBAL_MAX_TICK_CHANGE_BPS,
    format_price,
    tick_boundary,
    order_impact_bps,
    decay_news_sentiment,
    execution_price_cents,
    apply_daily_price_limit,
    pressure_from_order_flow,
    tick_boundaries_to_apply,
    calculate_next_price_cents,
    effective_volatility_width_bps,
)
from discordbot.services.stock.prompts import STOCK_NEWS_PROMPT, STOCK_NEWS_FALLBACK_TEMPLATES
from discordbot.services.economy.database import (
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
    """Both headline sources stay fictional, name nothing real, and are bounded to ±180 bps.

    Sentiment is a price input, so a template written outside the range the prompt states would
    move the market harder than any headline the model can write.
    """
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
    """The template fallback keeps the AI copy's style and takes its sign from the market."""
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
    """Builds the seeded generator every price path in this file is driven by.

    The formula draws randomness from an injected `Random` and nowhere else, so one seed replays
    a tick exactly; that is what lets these tests assert figures instead of ranges.

    Returns:
        A generator that produces the same draw sequence for the same seed.
    """
    return Random(seed)  # noqa: S311 -- deterministic market tests require seeded Random


@pytest.fixture
async def stock_empty_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Points the stock store at a throwaway SQLite file holding no company rows.

    `_schema_ready_for` is cleared alongside the engine because it caches readiness by engine
    identity, so the fresh file bootstraps its own tables on the first call instead of inheriting
    the previous engine's "ready".
    """
    stock_db_path = tmp_path / "stock.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{stock_db_path}")
    monkeypatch.setattr(stock_db, "_engine", engine)
    monkeypatch.setattr(stock_db, "_schema_ready_for", None)
    # The operation / market / news locks are loop-local helpers that rebuild on the
    # per-test event loop, so they need no manual reset here.
    yield
    await engine.dispose()


@pytest.fixture
async def stock_isolated_db(stock_empty_db: None) -> None:
    """Adds the BCAT company to the empty store, since nothing seeds one at bootstrap."""
    await _upsert_bcat_profile()


async def _upsert_bcat_profile(
    price_cents: int = BCAT_INITIAL_PRICE_CENTS,
    name: str = BCAT_NAME,
    category: str = BCAT_CATEGORY,
) -> StockProfileView:
    """Writes the BCAT test company through the operator's own upsert path.

    Everything but the three arguments is pinned to the `BCAT_*` constants, so a re-upsert changes
    only what the caller named.

    Returns:
        The stored profile, as the upsert hands it back.
    """
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
    """Writes THIN, a company thin enough that a ten-share order visibly moves its fill price.

    Volatility, the amplifier and mean reversion are all zeroed, so the only thing left moving a
    number in these tests is the slippage or the supply limit being measured.

    Returns:
        The stored profile, as the upsert hands it back.
    """
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
    """Builds the market picture a BCAT headline is generated from.

    Only daily movement and order-flow pressure vary; the flow counters and recent sentiment stay
    at zero, so the bullish / bearish / neutral template set turns on the two arguments alone.

    Returns:
        A context ready to hand to the template picker or a fake provider.
    """
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
    """A cent price converts to whole cash by an explicit ceiling or floor, never a bare round."""
    assert cash_ceil(cents=10_001) == 101
    assert cash_floor(cents=10_001) == 100
    assert format_price(price_cents=10_001) == "100.01"


def test_stock_tick_helpers_noop_and_compress_backlog() -> None:
    """Nothing advances inside one interval, and a compressed backlog still keeps midnight.

    The day rollover is what the daily price band is anchored on, so dropping it while trimming a
    backlog to `MAX_TICKS_PER_INTERACTION` would silently widen the band for that day.
    """
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
    """A naive SQLite-style datetime is read as Asia/Taipei, not as the container's local time."""
    naive = datetime(2026, 1, 1, 1, 23)
    aware = datetime(2026, 1, 1, 1, 23, tzinfo=TAIWAN_TIMEZONE)

    assert tick_boundary(dt=naive) == tick_boundary(dt=aware)


def test_stock_price_formula_is_deterministic_and_clamped() -> None:
    """Seeded ticks replay exactly, prices never reach zero, and news sentiment decays linearly."""
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


def test_effective_volatility_width_is_scaled_below_raw() -> None:
    """The global volatility scale shrinks the raw per-company width toward realism.

    Neither profile knob has a ceiling of its own, so this scale is the only thing holding the
    random component near real-market magnitudes.
    """
    raw_width = 180 * 360 // 100
    scaled = effective_volatility_width_bps(base_volatility_bps=180, volatility_amplifier_bps=360)
    assert 0 < scaled < raw_width
    assert effective_volatility_width_bps(base_volatility_bps=0, volatility_amplifier_bps=360) == 0


def test_apply_daily_price_limit_clamps_to_band_around_previous_close() -> None:
    """A tick price is bounded to a Taiwan-style band around the previous close, or passed on."""
    assert (
        apply_daily_price_limit(price_cents=12_000, previous_close_cents=10_000, limit_bps=1_000)
        == 11_000
    )
    assert (
        apply_daily_price_limit(price_cents=8_000, previous_close_cents=10_000, limit_bps=1_000)
        == 9_000
    )
    assert (
        apply_daily_price_limit(price_cents=10_500, previous_close_cents=10_000, limit_bps=1_000)
        == 10_500
    )
    assert (
        apply_daily_price_limit(price_cents=99_999, previous_close_cents=0, limit_bps=1_000)
        == 99_999
    )


def test_global_per_tick_ceiling_caps_change_below_company_limit() -> None:
    """A huge per-company max_tick_change is still bounded by the global ceiling."""
    previous = 10_000
    next_price = calculate_next_price_cents(
        previous_price_cents=previous,
        news_sentiment_bps=NEWS_SENTIMENT_LIMIT_BPS,
        pressure_bps=PRESSURE_LIMIT_BPS,
        base_volatility_bps=0,
        volatility_amplifier_bps=0,
        fair_value_cents=previous,
        mean_reversion_strength_bps=0,
        max_tick_change_bps=850,
        rng=_rng(seed=1),
    )
    assert next_price <= previous * (10_000 + GLOBAL_MAX_TICK_CHANGE_BPS) // 10_000
    assert DAILY_PRICE_LIMIT_BPS == 1_000


def _stock_news_row(
    created_at: datetime, sentiment_bps: int, news_id: str = "test"
) -> stock_db.StockNews:
    """Builds one `stock_news` row for the impulse tests, with no database behind it.

    Returns:
        An unattached ORM row carrying the creation stamp and sentiment the helper reads.
    """
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
    """News whose tick boundary predates all applied boundaries is dropped.

    Its impulse already landed on an earlier lazy advance, so re-applying it would price the same
    headline twice.
    """
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
    """A week of impulse news and random pressure never runs the price away from fair value.

    Seeded Monte Carlo over the most aggressive profile in the production set (highest volatility,
    biggest amplifier), because the guardrails were set by offline simulation and no single-tick
    assertion can show what a long run of same-sign ticks does. A change that weakens mean
    reversion, drops impulse semantics or widens the tick bounds surfaces here rather than as
    quiet inflation in production.
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
    assert (
        pressure_from_order_flow(net_shares=12_500, liquidity_shares=25_000)
        == PRESSURE_LIMIT_BPS // 2
    )
    assert (
        pressure_from_order_flow(net_shares=25_000, liquidity_shares=25_000) == PRESSURE_LIMIT_BPS
    )
    assert (
        pressure_from_order_flow(net_shares=-50_000, liquidity_shares=25_000)
        == -PRESSURE_LIMIT_BPS
    )
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
    """Per-leg decay stays fractional, so two one-share trades still register as pressure.

    Rounding each leg to an integer first would floor both to zero and lose the flow entirely.
    """
    at = datetime(2026, 1, 2, tzinfo=TAIWAN_TIMEZONE)
    pressure_rows = (
        (StockTradeLegType.OPEN_LONG.value, 1, at - timedelta(seconds=1)),
        (StockTradeLegType.OPEN_LONG.value, 1, at - timedelta(seconds=1)),
    )

    assert (
        stock_db._recent_pressure_bps_from_rows(
            pressure_rows=pressure_rows, at=at, liquidity_shares=100
        )
        == 1
    )


async def test_stock_schema_bootstrap_does_not_seed_companies(stock_empty_db: None) -> None:
    """Bootstrap creates the tables, in decimal-text affinity, and seeds no company at all.

    A fresh database is an empty market on purpose: companies are operator data written offline,
    so anything that seeds one from runtime code would ship a market nobody tuned.
    """
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
    """The operator upsert creates a company, updates it in place, and marks a tick each time."""
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
    """Money and share columns round-trip past SQLite's 64-bit ceiling as decimal text.

    `typeof()` is read back on every one of them, because an affinity that drifted to REAL would
    keep working on ordinary numbers and start silently losing precision on large ones.
    """
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
    """An impossible share structure is refused by the payload model, before anything persists."""
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
    """A provider is called once per cadence bucket, and its source and model land on the row."""
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
    """The context handed to a provider carries the day's move, the trade flow and the last news.

    That is what lets a headline agree with the chart a reader is looking at rather than
    contradict it; the deterministic templates pick their tone off the same figures.
    """
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
        """Captures the context the engine built, then answers like a real provider.

        Returns:
            One fake AI news item.
        """
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
    """A bucket holds one headline, and an AI one written into it replaces a template in place.

    The upgrade is deliberately one-way, so a late provider improves the bucket while a template
    refresh can never undo it, and neither stacks a second impulse onto the same ticks.
    """
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
    """Passing a provider reopens a bucket a template already filled, before the next cadence."""
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
    """Two refreshes arriving together find the symbol due once, so only one provider call is paid.

    Without the process-wide serialization each interaction would bill its own LLM call for the
    same bucket, and one of the two answers would then be discarded by the per-bucket insert.
    """
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
    """Crossing Asia/Taipei midnight rolls the previous close and opens a new day."""
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
    """A compressed backlog still materializes the real midnight tick the day open is read from.

    Both figures are read back off the persisted ticks rather than recomputed, since a rollover
    that landed on the nearest surviving boundary instead would anchor the daily band on the
    wrong price.
    """
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
    """Day open follows the stored tick when a concurrent writer wins the boundary.

    The loser of that race has to carry on from the winner's price, or two interactions over one
    backlog would leave the day anchored on two different opens.
    """
    await _set_bcat_price(price_cents=10_000)
    original_insert_tick = stock_db._insert_price_tick_or_existing
    midnight = datetime(2026, 1, 2, 0, 0, tzinfo=TAIWAN_TIMEZONE)
    persisted_open_price = 12_345

    async def insert_tick_after_concurrent_writer(
        session: AsyncSession, symbol: str, price_cents: int, created_at: datetime
    ) -> int:
        """Plants another writer's midnight tick just before the real insert runs.

        Returns:
            Whatever the real insert reports is now stored at that boundary.
        """
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
    """Concurrent quote refreshes write one tick per boundary instead of forking the history."""
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
    """Every symbol's advance takes SQLite's write lock before it reads the state it plans from.

    A deferred transaction upgrades only at the first write, by which point another writer may
    already have moved the rows the plan was built on.
    """
    original_begin_immediate = stock_db._begin_immediate
    calls = 0

    async def begin_immediate(session: AsyncSession) -> None:
        """Counts the write-lock openings while still starting the real transaction."""
        nonlocal calls
        calls += 1
        await original_begin_immediate(session=session)

    monkeypatch.setattr(stock_db, "_begin_immediate", begin_immediate)

    quotes = await stock_db.list_market_quotes(now=datetime(2026, 1, 1), rng=_rng(seed=1))

    assert calls == len(quotes)


async def test_stock_buy_long_debits_wallet_and_writes_ledger(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """A long buy debits the wallet, opens the position, and leaves an applied operation and leg.

    The legs are the only audit trail there is, since there is no transaction table on either
    side of the seam.
    """
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
    """A later trade refreshes the display name stored beside the user id, everywhere it is kept.

    The UI shows stored names rather than resolving ids, so a rename that only reached one table
    would leave the position, the operation, the legs and the wallet disagreeing about a person.
    """
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
    """A stock's detail view is public: every trader's recent trades and non-zero positions."""
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
    """A portfolio values a long at the floor and a short at the ceiling of the current quote.

    Rounding runs against the holder on both sides, so the equity a portfolio reports can never
    exceed what closing the position would actually pay out.
    """
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
    """A portfolio read is served from a short-lived cache until a stock write invalidates it.

    The position is changed behind the cache's back, which is how a stale read can be told apart
    from a fresh one that happens to agree.
    """
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
    """A numeric buy above the spendable balance fills what it can afford instead of failing."""
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
    """Aggregate long exposure stops at the floating supply, and the next buyer is refused.

    Float is what makes the market finite; without the clamp any balance could keep opening long
    exposure against shares the company never issued.
    """
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
    """One user's long exposure stops at 49% of float, whether asked for numerically or as ALL."""
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
    """Settlement stores the price the order actually filled at, not the quote it was shown.

    The leg, the wallet delta and the reported price all come from the slipped figure, so a large
    order cannot be settled at a price only a small one could have had.
    """
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
    """A buy nobody can afford fails naming the balance, and leaves no position or leg behind.

    The clamp-to-affordable path must not swallow the reason and report an empty fill as a
    success, or the two databases disagree about an operation that never happened.
    """
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
    """A same-price long round trip costs the ceil/floor spread and clears the basis to zero.

    Cents-to-cash rounds against the trader on both legs, so buying and selling at one price is
    expected to lose the dust rather than break even; a round trip that broke even would mean the
    rounding could be farmed.
    """
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
    """Opening a short locks collateral, and a same-price cover returns all but the spread."""
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
    """A numeric short beyond the collateral the balance covers opens what it can instead."""
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
    """Short borrow stops at the floating supply, and the next borrower is refused.

    The same float bounds both sides, so a short cannot borrow shares the company never issued
    any more than a long can buy them.
    """
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
    """An operation stuck short of a final state keeps holding the float and borrow it opened.

    Reserving is what stops the same shares being sold twice while the two databases are out of
    step; the capacity comes back only when the operation is reconciled or fails.

    The individual ownership cap is why three pending longs share THIN's float: one user cannot
    take more than 49% of it, so the last of them is left the 20-share remainder.
    """
    await _upsert_illiquid_profile()
    await adjust_balance(user_id=1, name="alice", delta=200_000)
    await adjust_balance(user_id=2, name="bob", delta=1_000)
    await adjust_balance(user_id=3, name="carol", delta=100_000_000)
    await adjust_balance(user_id=4, name="dave", delta=1_000)
    await adjust_balance(user_id=5, name="erin", delta=200_000)
    await adjust_balance(user_id=6, name="frank", delta=200_000)
    original_apply = stock_db.apply_ordered_wallet_deltas

    async def fail_wallet(**_kwargs: object) -> OrderedWalletDeltaResult:
        """Leaves the wallet side uncertain after the stock side has reserved supply.

        Raises:
            RuntimeError: Always, standing in for an unreachable economy database.
        """
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
    """A cover is paid out of its own collateral and proceeds, with a spendable balance of zero.

    This is what the gross, ordered wallet legs buy: the collateral and entry credits land before
    the cover debit, so the economy side can cover in full at that point in the sequence even
    though the balance never held the money.
    """
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
    """A buy that both covers a short and opens a long writes the two legs in that order.

    The legs are never netted back to one wallet movement, which is what keeps the economy's
    `total_earned - total_spent == balance` identity describing real flow.
    """
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
    """Two buys from one user on one stock serialize, and the key leaves no lock behind.

    The second submission has to re-read the position the first wrote, or both would price
    against the same stale row and spend the balance twice. `is_empty` is the other half: an
    unbounded key space would leak an entry per user and symbol if a lock outlived its holders.
    """
    await adjust_balance(user_id=1, name="alice", delta=100)

    results = await asyncio_gather_stock_buys()

    assert sum(result.success for result in results) == 1
    detail = await stock_db.get_stock_detail(symbol=BCAT_SYMBOL, user_id=1)
    assert detail.position.long_shares == 1
    assert stock_db._operation_locks.is_empty


async def asyncio_gather_stock_buys() -> tuple[StockSettlementResult, StockSettlementResult]:
    """Submits the same one-share buy twice at once, against a balance that affords one.

    Returns:
        Both settlement results, in submission order.
    """
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
    """A wallet failure after the stock commit parks the operation where an operator can see it.

    Nothing rolls back and nothing retries, because whether the debit landed is exactly what is
    unknown; the operation, its legs and the names on both are what the operator reads instead.
    """
    await adjust_balance(user_id=1, name="alice", delta=1_000)

    async def fail_wallet(**_kwargs: object) -> OrderedWalletDeltaResult:
        """Fails the wallet application after the stock side has already committed.

        Raises:
            RuntimeError: Always, standing in for an unreachable economy database.
        """
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
    """A parked operation stops that user trading that symbol, even once the wallet is healthy.

    The refusal names the blocking operation rather than opening a second one on top of it, so
    the seam stops the feature for one user instead of quietly stacking uncertainty.
    """
    await adjust_balance(user_id=1, name="alice", delta=1_000)
    original_apply = stock_db.apply_ordered_wallet_deltas

    async def fail_wallet(**_kwargs: object) -> OrderedWalletDeltaResult:
        """Fails the first operation's wallet application, then is swapped back out.

        Raises:
            RuntimeError: Always, standing in for an unreachable economy database.
        """
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
    """A cancel in flight still records the marker before the cancellation propagates.

    Cancellation is the one path that could return without writing anything, and it leaves the
    same "did the debit land" question behind as a failure does.
    """
    await adjust_balance(user_id=1, name="alice", delta=1_000)

    async def cancel_wallet(**_kwargs: object) -> OrderedWalletDeltaResult:
        """Cancels while the wallet application is in flight.

        Raises:
            CancelledError: Always, standing in for a cancelled settlement task.
        """
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
    """A wallet that answers "no" finalizes the operation as failed, with no position written.

    A rejection is a definite answer, unlike a raise, so nothing is left for an operator to
    reconcile and the reservation goes back to the market immediately.
    """
    await adjust_balance(user_id=1, name="alice", delta=1_000)

    async def reject_wallet(**_kwargs: object) -> None:
        """Refuses the debit outright, as a wallet race that emptied the balance would."""
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
    """A healthy operation is marked wallet_applied before the position is written.

    That status marks the window where the money has moved and the position has not, which is
    what a failure there is recorded against; skipping straight to applied would leave an
    operator unable to tell that window from one where nothing happened at all.
    """
    await adjust_balance(user_id=1, name="alice", delta=1_000)
    statuses: list[StockOperationStatus] = []
    original_mark_operation = stock_db._mark_operation

    async def record_mark_operation(
        operation_id: str, status: StockOperationStatus, failure_reason: str
    ) -> None:
        """Records each lifecycle step while still performing the real update."""
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
    """Stock wallet legs move gross lifetime totals and go nowhere near the casino counters.

    Settling a trade is not gambling, so it must not feed the daily casino loss the casino
    settlement helpers maintain; the balance and the two lifetime totals are all that move.
    """
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
    """A single tick still renders a real PNG rather than failing on a line with no segment.

    A freshly upserted company has exactly one tick, so this is the first chart anyone sees.
    """
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
    """Advances the market once, then pins BCAT's quote and its whole tick history to one price.

    The close and day open are pinned alongside it, so the daily band cannot clamp the next
    advance and a settlement test's arithmetic is exact rather than approximate.
    """
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


async def test_reset_all_positions_flattens_and_zeros_pnl(stock_empty_db: None) -> None:
    """The offline economy reset flattens every position, zeroes P&L and bumps the row version.

    Prices are deliberately left alone, so a deflated balance cannot be topped back up out of a
    position that still values at the old scale.
    """
    del stock_empty_db
    # The first call is what bootstraps the schema; the position can only be seeded after it.
    assert await stock_db.reset_all_positions() == 0
    async with stock_db.open_stock_session() as session:
        session.add(
            stock_db.StockPosition(
                symbol="BCAT",
                user_id=1,
                user_name="alice",
                long_shares=1_000,
                long_cost_basis=5_000,
                short_shares=200,
                short_entry_value=400,
                short_collateral=600,
                realized_pnl=999_999,
                version=3,
                updated_at=datetime(2026, 1, 1),
            )
        )
        await session.commit()

    affected = await stock_db.reset_all_positions()

    assert affected == 1
    async with stock_db.open_stock_session() as session:
        row = (
            await session.execute(
                select(stock_db.StockPosition).where(stock_db.StockPosition.user_id == 1)
            )
        ).scalar_one()
    assert (row.long_shares, row.long_cost_basis) == (0, 0)
    assert (row.short_shares, row.short_entry_value, row.short_collateral) == (0, 0, 0)
    assert row.realized_pnl == 0
    assert row.version == 4


async def test_reset_all_positions_finalizes_non_final_operations(stock_empty_db: None) -> None:
    """The reset also finalizes non-final operations, or they would go on blocking their owners.

    A reset that flattened positions but left a pending operation behind would claim the stock
    state was cleared while `_blocking_operation` kept refusing that user on that symbol.
    """
    del stock_empty_db
    assert await stock_db.reset_all_positions() == 0
    async with stock_db.open_stock_session() as session:
        session.add(
            stock_db.StockOperation(
                operation_id="op-pending",
                symbol="BCAT",
                user_id=1,
                user_name="alice",
                requested_action="buy",
                status=stock_db.StockOperationStatus.PENDING.value,
                failure_reason="",
                created_at=datetime(2026, 1, 1),
                updated_at=datetime(2026, 1, 1),
            )
        )
        await session.commit()

    await stock_db.reset_all_positions()

    async with stock_db.open_stock_session() as session:
        op = (
            await session.execute(
                select(stock_db.StockOperation).where(
                    stock_db.StockOperation.operation_id == "op-pending"
                )
            )
        ).scalar_one()
    assert op.status == stock_db.StockOperationStatus.FAILED.value
    assert op.failure_reason == "economy reset"
