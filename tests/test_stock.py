"""Tests for the simulated stock market service."""

from random import Random
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select, update, inspect
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.cogs._stock import database as stock_db
from discordbot.typings.stock import (
    BCAT_NAME,
    BCAT_SYMBOL,
    BCAT_INITIAL_PRICE_CENTS,
    MAX_TICKS_PER_INTERACTION,
    StockAction,
    StockTradeLegType,
    StockOperationStatus,
    StockSettlementResult,
)
from discordbot.typings.economy import WalletDeltaLeg, OrderedWalletDeltaResult
from discordbot.cogs._stock.chart import build_price_chart
from discordbot.cogs._stock.market import (
    TAIWAN_TIMEZONE,
    cash_ceil,
    cash_floor,
    format_price,
    tick_boundary,
    decay_news_sentiment,
    tick_boundaries_to_apply,
    calculate_next_price_cents,
)
from discordbot.cogs._economy.database import (
    UserWallet,
    open_session,
    adjust_balance,
    apply_ordered_wallet_deltas,
)


def _rng(seed: int) -> Random:
    """Returns a deterministic test RNG."""
    return Random(seed)  # noqa: S311 -- deterministic market tests require seeded Random


@pytest.fixture
async def stock_isolated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Per-test SQLite file with the stock schema."""
    stock_db_path = tmp_path / "stock.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{stock_db_path}")
    monkeypatch.setattr(stock_db, "_engine", engine)
    monkeypatch.setattr(stock_db, "_schema_ready_for", None)
    stock_db._operation_locks.clear()
    stock_db._market_locks.clear()
    yield
    await engine.dispose()


def test_stock_cash_rounding_and_price_format() -> None:
    """Prices are cent-based and cash conversion is explicit."""
    assert cash_ceil(cents=10_001) == 101
    assert cash_floor(cents=10_001) == 100
    assert format_price(price_cents=10_001) == "100.01"


def test_stock_tick_helpers_noop_and_compress_backlog() -> None:
    """Lazy ticks no-op inside one interval and compress long backlogs."""
    latest = datetime(2026, 1, 1, 0, 0)
    assert (
        tick_boundaries_to_apply(latest_tick_at=latest, now=latest + timedelta(minutes=59)) == ()
    )

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
        rng=_rng(seed=1),
    )
    second = calculate_next_price_cents(
        previous_price_cents=100,
        news_sentiment_bps=-20_000,
        pressure_bps=-20_000,
        base_volatility_bps=0,
        volatility_amplifier_bps=100,
        rng=_rng(seed=1),
    )
    assert first == second
    assert first >= 1
    assert decay_news_sentiment(sentiment_bps=500, ticks_elapsed=3) == 240
    assert decay_news_sentiment(sentiment_bps=-500, ticks_elapsed=20) == 0


async def test_stock_schema_seeds_bcat(stock_isolated_db: None) -> None:
    """Schema bootstrap creates stock tables and seeds BCAT."""
    quotes = await stock_db.list_market_quotes(now=datetime(2026, 1, 1), rng=_rng(seed=1))

    assert len(quotes) == 1
    assert quotes[0].profile.symbol == BCAT_SYMBOL
    assert quotes[0].profile.name == BCAT_NAME
    assert quotes[0].profile.price_cents == BCAT_INITIAL_PRICE_CENTS
    news = await stock_db.get_stock_news(symbol=BCAT_SYMBOL)
    assert news
    async with stock_db._engine.connect() as conn:
        column_names = await conn.run_sync(
            lambda sync_conn: {
                table_name: [
                    column["name"]
                    for column in inspect(sync_conn).get_columns(table_name=table_name)
                ]
                for table_name in ("stock_position", "stock_operation", "stock_trade_leg")
            }
        )
    assert column_names["stock_position"][:3] == ["symbol", "user_id", "user_name"]
    assert column_names["stock_operation"][1:4] == ["symbol", "user_id", "user_name"]
    assert column_names["stock_trade_leg"][3:6] == ["symbol", "user_id", "user_name"]


async def test_stock_day_rollover_updates_open_and_previous_close(stock_isolated_db: None) -> None:
    """Crossing Asia/Taipei midnight updates previous close and day open."""
    await stock_db.list_market_quotes(now=datetime(2026, 1, 1, 12, 0), rng=_rng(seed=1))
    latest = datetime(2026, 1, 1, 23, 0)
    async with stock_db.open_stock_session() as session:
        await session.execute(
            statement=update(stock_db.StockPriceTick)
            .where(stock_db.StockPriceTick.symbol == BCAT_SYMBOL)
            .values(created_at=latest, price_cents=10_000)
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

    assert quotes[0].profile.previous_close_price_cents == 10_000
    assert quotes[0].profile.day_open_price_cents > 0


async def test_stock_compressed_day_rollover_materializes_midnight(
    stock_isolated_db: None,
) -> None:
    """Compressed backlogs keep the actual midnight boundary for day-open pricing."""
    await _set_bcat_price(price_cents=10_000)

    quotes = await stock_db.list_market_quotes(now=datetime(2026, 1, 2, 1, 0), rng=_rng(seed=0))

    midnight = datetime(2026, 1, 2, 0, 0, tzinfo=TAIWAN_TIMEZONE)
    previous_close_at = datetime(2026, 1, 1, 23, 0, tzinfo=TAIWAN_TIMEZONE)
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

    assert quotes[0].profile.day_open_price_cents == midnight_tick.scalar_one()
    assert quotes[0].profile.previous_close_price_cents == previous_close_tick.scalar_one()


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


async def test_stock_insufficient_buy_leaves_stock_untouched(
    stock_isolated_db: None, economy_isolated_db: None
) -> None:
    """Insufficient funds do not mutate position or trade ledger."""
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
