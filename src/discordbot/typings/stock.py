"""Shared vocabulary for the simulated stock market: its tuning constants, enums and read models.

`services/stock/` owns the SQLAlchemy rows in `stock.db` and the settlement that writes them;
`cogs/stock/` owns the one public `/stock` message that renders them. Neither layer may import
the other, so every value crossing that seam is declared here, and no view ever touches an ORM
model or recomputes a price: a number missing from these models is a number the UI cannot show.
Most of what follows is a frozen read projection the service builds and the views only read. The
three that travel the other way say so in their own docstrings: `StockProfileUpsert` is the
operator-authored write payload, `StockNewsGenerationContext` is handed out to the cog's news
producer, and `StockGeneratedNews` is what comes back from it.

Money lives in two different units and the field name carries the rule. `*_cents` is
cent-denominated and covers exactly the quote side: `price_cents`, `previous_close_price_cents`,
`day_open_price_cents`, `fair_value_cents` and `change_cents`. Every other money field (cost
basis, entry value, collateral, realized and unrealized P&L, wallet deltas, balances) is whole
units of the economy wallet's currency, since that is what `user_wallet` holds. The service
converts per leg through `utils/currency.py::cash_ceil` / `cash_floor` and always rounds against
the trader (a buy ceils, a sell floors), so the rounding cannot be farmed.

The constants are what the service, the views and the tests all agree on:

- `STOCK_TICK_SECONDS` is the lazy tick period. There is no market loop; an interaction advances
  a symbol to the tick boundary at or before now, so a quiet channel simply replays its backlog
  on the next command.
- `MAX_TICKS_PER_INTERACTION` is one day of those ticks, bounding that catch-up. A longer gap is
  compressed rather than replayed one tick at a time, preserving the Asia/Taipei day-rollover
  boundaries so the daily price limit still lands where it should.
- `STOCK_HISTORY_DAYS` is the window the tick query and the 7D chart share.
- `STOCK_ACTION_TIMEOUT_SECONDS` is the idle life of the opener-only `/stock` message.
- `STOCK_NEWS_CADENCE_HOURS` is a documented default that nothing in `src/` reads. There is no
  seed script: a profile is created by a direct DB edit through `upsert_stock_profile`, whose
  `news_cadence_hours` is required with no default, and the refresh logic reads that per-profile
  value and never this constant. Only the tests take it.
- `STOCK_INDIVIDUAL_OWNERSHIP_CAP_BPS` over `STOCK_BPS_DENOMINATOR` is the share of float one
  user may hold long. It gates opening only — selling and covering are always allowed.
"""

from enum import StrEnum
from typing import Self, Final
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict, model_validator

STOCK_TICK_SECONDS: Final[int] = 5 * 60
MAX_TICKS_PER_INTERACTION: Final[int] = 12 * 24
STOCK_HISTORY_DAYS: Final[int] = 7
STOCK_ACTION_TIMEOUT_SECONDS: Final[int] = 180
STOCK_NEWS_CADENCE_HOURS: Final[int] = 4
STOCK_INDIVIDUAL_OWNERSHIP_CAP_BPS: Final[int] = 4_900
STOCK_BPS_DENOMINATOR: Final[int] = 10_000


class StockAction(StrEnum):
    """The two directions a user can submit; each closes the opposite side before opening.

    BUY covers any open short before opening long, SHORT sells any long before borrowing, so one
    submission both closes and opens and there is no separate close command. `StockTradeLegType`
    is what that expands into.
    """

    BUY = "buy"
    SHORT = "short"


class StockOperationStatus(StrEnum):
    """Lifecycle of one operation across `stock.db` and `economy.db`.

    The two databases cannot commit together, so the operation row is written PENDING before the
    wallet moves, advanced to WALLET_APPLIED once it has, and only then finalized to APPLIED.
    FAILED is the ordinary refusal, where the wallet rejected the delta and nothing moved.
    RECONCILE_REQUIRED is the deliberate dead end for a crash or cancellation between the two
    writes: settlement neither rolls back nor silently retries, it parks the operation for
    `list_reconciliation_operations` and blocks that user's further trades on the symbol.
    APPLIED and FAILED are the only final states, which is what `_blocking_operation` keys on.
    """

    PENDING = "pending"
    WALLET_APPLIED = "wallet_applied"
    APPLIED = "applied"
    FAILED = "failed"
    RECONCILE_REQUIRED = "reconcile_required"


class StockTradeLegType(StrEnum):
    """The atomic legs one operation expands into, in `leg_order`.

    Each leg carries its own execution price because order-size slippage is applied per leg, so
    the legs are never netted into a single wallet movement.
    """

    OPEN_LONG = "open_long"
    SELL_LONG = "sell_long"
    OPEN_SHORT = "open_short"
    COVER_SHORT = "cover_short"


class StockProfileView(BaseModel):
    """One virtual company: its simulation knobs and its current quote.

    The row behind this is maintained offline and is the source of truth for what exists on the
    market. `previous_close_price_cents` and `day_open_price_cents` both roll at the Asia/Taipei
    day boundary, but only the previous close anchors anything: the daily change display is
    measured against it and the Taiwan-style daily price limit bands around it. The day open is
    seeded on create, rewritten at each rollover and projected here, then read by nothing in
    `src/`.

    Only `max_tick_change_bps` is tightened against a global guardrail in
    `services/stock/market.py`, by a `min` with `GLOBAL_MAX_TICK_CHANGE_BPS`. `base_volatility_bps`
    and `volatility_amplifier_bps` are scaled by `MARKET_VOLATILITY_SCALE_BPS` with no ceiling of
    their own, and `mean_reversion_bps` has no global counterpart at all. What contains a wild
    profile is therefore the post-hoc per-tick clamp plus `apply_daily_price_limit`, not a
    per-knob bound: raising a knob does widen the raw move, it just cannot survive those two.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Virtual company ticker symbol.")
    name: str = Field(..., description="Display name of the virtual company.")
    category: str = Field(..., description="Market category the company belongs to.")
    price_cents: int = Field(..., description="Latest quote price in cents.")
    previous_close_price_cents: int = Field(
        ..., description="Previous trading-day close price in cents."
    )
    day_open_price_cents: int = Field(..., description="Current trading-day open price in cents.")
    total_shares: int = Field(..., description="Total issued share count.")
    float_shares: int = Field(..., description="Tradable float share count.")
    base_volatility_bps: int = Field(
        ..., description="Baseline per-tick volatility in basis points."
    )
    volatility_amplifier_bps: int = Field(
        ..., description="Additional volatility amplifier in basis points."
    )
    liquidity_shares: int = Field(
        ..., description="Liquidity depth in shares used for order-size slippage."
    )
    fair_value_cents: int = Field(..., description="Mean-reversion fair value anchor in cents.")
    mean_reversion_bps: int = Field(
        ..., description="Per-tick mean-reversion pull toward fair value in basis points."
    )
    max_tick_change_bps: int = Field(
        ..., description="Cap on price change per tick in basis points."
    )
    news_cadence_hours: int = Field(
        ..., description="Minimum hours between news refreshes for this symbol."
    )
    updated_at: datetime = Field(..., description="Timestamp of the latest profile update.")


class StockProfileUpsert(BaseModel):
    """The write side of a profile, carrying only what an operator may set by hand.

    Companies are seeded and retuned offline, never from runtime code, and there is no schema
    migration mechanism, so the bounds live on the fields and reject a bad payload before it
    reaches the row. The daily anchors and timestamps are absent on purpose: they are derived,
    and an upsert seeds them from `price_cents` on create rather than accepting them.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(
        min_length=1,
        max_length=16,
        description="Virtual company ticker symbol.",
        examples=["ACME"],
    )
    name: str = Field(
        min_length=1,
        max_length=128,
        description="Display name of the virtual company.",
        examples=["Acme Corp"],
    )
    category: str = Field(
        min_length=1,
        max_length=64,
        description="Market category the company belongs to.",
        examples=["tech"],
    )
    price_cents: int = Field(ge=1, description="Initial quote price in cents.", examples=[10000])
    total_shares: int = Field(ge=1, description="Total issued share count.", examples=[1000000])
    float_shares: int = Field(ge=0, description="Tradable float share count.", examples=[500000])
    base_volatility_bps: int = Field(
        ge=0, description="Baseline per-tick volatility in basis points.", examples=[50]
    )
    volatility_amplifier_bps: int = Field(
        ge=0, description="Additional volatility amplifier in basis points.", examples=[20]
    )
    liquidity_shares: int = Field(
        ge=1,
        description="Liquidity depth in shares used for order-size slippage.",
        examples=[10000],
    )
    fair_value_cents: int = Field(
        ge=1, description="Mean-reversion fair value anchor in cents.", examples=[10000]
    )
    mean_reversion_bps: int = Field(
        ge=0,
        description="Per-tick mean-reversion pull toward fair value in basis points.",
        examples=[10],
    )
    max_tick_change_bps: int = Field(
        ge=1, description="Cap on price change per tick in basis points.", examples=[300]
    )
    news_cadence_hours: int = Field(
        ge=1, description="Minimum hours between news refreshes for this symbol.", examples=[4]
    )

    @model_validator(mode="after")
    def validate_share_structure(self) -> Self:
        """Rejects a float larger than the issue it is drawn from.

        The check spans two fields, so it cannot be a `Field` bound. `float_shares` is what caps
        aggregate long exposure and short borrow capacity, so a float above `total_shares` would
        hand the market more tradable supply than the company ever issued.

        Returns:
            The validated model, unchanged.

        Raises:
            ValueError: `float_shares` exceeds `total_shares`.
        """
        if self.float_shares > self.total_shares:
            msg = "float_shares cannot exceed total_shares"
            raise ValueError(msg)
        return self


class StockPositionView(BaseModel):
    """One user's holding in one symbol, long and short tracked side by side.

    A short keeps its entry value and its collateral as separate running totals rather than one
    net figure, because covering releases both prorated by the shares covered; netting them
    would lose the split and mis-settle a partial cover. A user with no row is projected as an
    all-zero position rather than None, so the detail view always has something to render.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Virtual company ticker symbol.")
    user_id: int = Field(..., description="Discord user ID owning the position.")
    user_name: str = Field(default="", description="Stored display name of the position owner.")
    long_shares: int = Field(default=0, description="Number of shares held long.")
    long_cost_basis: int = Field(
        default=0, description="Aggregate cost basis of the long position in wallet units."
    )
    short_shares: int = Field(default=0, description="Number of shares held short.")
    short_entry_value: int = Field(
        default=0, description="Aggregate entry value of the short position in wallet units."
    )
    short_collateral: int = Field(
        default=0, description="Collateral reserved against the short position in wallet units."
    )
    realized_pnl: int = Field(
        default=0, description="Realized profit and loss for this position in wallet units."
    )


class StockParticipantPositionView(BaseModel):
    """What every viewer of a stock learns about one other trader.

    Sizes and realized P&L only: cost basis, short entry value and collateral stay in
    `StockPositionView`, which the viewer sees for themselves alone. Unrealized P&L is in neither
    model, since the detail render carries no valuation; it exists on `StockPortfolioHolding` and
    `StockPortfolioView`, which only the economy profile embed reads. `user_name` falls back to
    the id as text when no name was ever stored, so a row always renders.
    """

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(..., description="Discord user ID of the participant.")
    user_name: str = Field(..., description="Stored display name of the participant.")
    long_shares: int = Field(..., description="Number of shares the participant holds long.")
    short_shares: int = Field(..., description="Number of shares the participant holds short.")
    realized_pnl: int = Field(
        ..., description="Realized profit and loss for the participant in wallet units."
    )


class StockTradeLegView(BaseModel):
    """One persisted leg of an operation, and the whole audit trail there is.

    There is no transaction table above this: the deltas a leg carries are the record of how a
    position and a wallet reached their current values, which is why a leg is stored rather than
    netted with its siblings and why `price_cents` is the leg's own slipped execution price, not
    the quote at submit time.
    """

    model_config = ConfigDict(frozen=True)

    operation_id: str = Field(..., description="Parent stock operation identifier.")
    leg_order: int = Field(..., description="Ordering index of this leg within the operation.")
    symbol: str = Field(..., description="Virtual company ticker symbol.")
    user_id: int = Field(..., description="Discord user ID the leg belongs to.")
    user_name: str = Field(default="", description="Stored display name of the leg owner.")
    leg_type: StockTradeLegType = Field(..., description="Type of atomic trade leg.")
    shares: int = Field(..., description="Share quantity executed in this leg.")
    price_cents: int = Field(..., description="Per-leg execution price in cents.")
    wallet_delta: int = Field(..., description="Wallet balance change applied by this leg.")
    basis_delta: int = Field(
        ..., description="Cost-basis change applied by this leg in wallet units."
    )
    collateral_delta: int = Field(
        ..., description="Short collateral change applied by this leg in wallet units."
    )
    realized_pnl_delta: int = Field(
        ..., description="Realized profit and loss change applied by this leg in wallet units."
    )
    created_at: datetime = Field(..., description="Timestamp the leg was created.")


class StockNewsView(BaseModel):
    """One fictional news item as stored, with the provenance the refresh logic reads back.

    `source` is `"template"` for a deterministic fallback headline and `"ai"` when the LLM
    answered, with `model` naming which one. That is not decoration: a template row is the one
    kind an `"ai"` refresh may overwrite inside the same cadence bucket, and it never happens the
    other way round. `expires_at` bounds only how long the item still counts toward decayed
    sentiment; the detail view shows the latest few whatever it says.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Virtual company ticker symbol.")
    headline: str = Field(..., description="News headline text.")
    sentiment_bps: int = Field(..., description="News sentiment impulse in basis points.")
    source: str = Field(
        default="template", description="Origin of the news item, such as template or model."
    )
    model: str = Field(default="", description="Model identifier that generated the news, if any.")
    expires_at: datetime | None = Field(
        default=None, description="Timestamp after which the news is stale, if set."
    )
    created_at: datetime = Field(..., description="Timestamp the news item was created.")


class StockGeneratedNews(BaseModel):
    """A freshly produced headline, before the store clamps and files it.

    The one return shape both producers share, so a caller never learns whether the LLM answered
    or the deterministic templates did. `sentiment_bps` is whatever the producer asked for; the
    store is what clamps it into the sentiment limit, so nothing here is trusted as final.
    """

    model_config = ConfigDict(frozen=True)

    headline: str = Field(..., description="Generated news headline text.")
    sentiment_bps: int = Field(
        ..., description="Generated news sentiment impulse in basis points."
    )
    source: str = Field(
        ..., description="Origin of the generated news, such as template or model."
    )
    model: str = Field(default="", description="Model identifier that generated the news, if any.")


class StockNewsGenerationContext(BaseModel):
    """Everything a news producer is told about a symbol that has become due for a headline.

    Built once per due symbol and handed to whichever producer runs, so the deterministic
    templates pick from the same market state the LLM prompt describes and a fallback headline
    still fits the tape. `recent_news_sentiment_bps` is the decayed ambient value, which exists
    for this prompt alone — the price formula applies a news impulse once at its own boundary
    and never reads the decayed figure.
    """

    model_config = ConfigDict(frozen=True)

    profile: StockProfileView = Field(
        ..., description="Stock profile and latest quote for context."
    )
    change_cents: int = Field(..., description="Daily price change in cents.")
    change_bps: int = Field(..., description="Daily price change in basis points.")
    pressure_bps: int = Field(
        ..., description="Recent decayed order-flow pressure in basis points."
    )
    buy_side_shares: int = Field(..., description="Recent buy-side order flow in shares.")
    sell_side_shares: int = Field(..., description="Recent sell-side order flow in shares.")
    net_order_shares: int = Field(..., description="Net order flow in shares.")
    recent_news_sentiment_bps: int = Field(
        ..., description="Existing decayed news sentiment in basis points."
    )
    latest_news_headline: str = Field(default="", description="Most recent news headline, if any.")
    latest_news_sentiment_bps: int = Field(
        default=0, description="Sentiment of the most recent news in basis points."
    )
    lookback_hours: int = Field(
        ..., description="Lookback window in hours used to compute the context."
    )


class StockPriceTickView(BaseModel):
    """One point on the price history, stamped at its tick boundary rather than at write time.

    `(symbol, created_at)` is unique, so the boundary stamp is what lets a lazy catch-up replay a
    boundary without duplicating a point. The immutable tuple of these doubles as the 7D chart's
    render cache key, which is why nothing mutates one in place.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Virtual company ticker symbol.")
    price_cents: int = Field(..., description="Quote price at this tick in cents.")
    created_at: datetime = Field(..., description="Timestamp of the price tick.")


class StockMarketQuote(BaseModel):
    """A profile plus three numbers derived at read time, of which the UI renders two.

    The change pair is measured against the previous close, not the day open, so it reads the way
    a real ticker does across a day rollover. `change_bps` and `pressure_bps` are what the market
    board and the detail header show; `change_cents` is projected alongside them and read nowhere
    in `src/`, which is why `_market_board_spec` leaves it out of the cache key. Every field but
    `profile` is derived per read, which is why that PNG caches on a digest of the pixel-affecting
    fields of these rows rather than on the symbol.
    """

    model_config = ConfigDict(frozen=True)

    profile: StockProfileView = Field(..., description="Stock profile and latest quote.")
    change_cents: int = Field(..., description="Daily price change in cents.")
    change_bps: int = Field(..., description="Daily price change in basis points.")
    pressure_bps: int = Field(..., description="Recent order-flow pressure in basis points.")


class StockDetailViewData(BaseModel):
    """Everything one `/stock` detail render needs, gathered under a single market advance.

    Every stock-side field is read in one session while the symbol's lock is held, so the quote,
    the position and the chart cannot disagree about which tick the user is looking at; only
    `balance` comes from the economy DB afterwards. `position` and `balance` are the viewer's
    own, while `recent_trades` and `public_positions` are stock-wide — they name other traders,
    and the message rendering them is public.
    """

    model_config = ConfigDict(frozen=True)

    quote: StockMarketQuote = Field(..., description="Market quote for the selected stock.")
    balance: int = Field(..., description="Viewing user's wallet cash balance.")
    position: StockPositionView = Field(
        ..., description="Viewing user's current position in the stock."
    )
    recent_trades: tuple[StockTradeLegView, ...] = Field(
        ..., description="Recent applied trade legs for the stock, across all participants."
    )
    public_positions: tuple[StockParticipantPositionView, ...] = Field(
        default=(), description="Public position summaries for all participants."
    )
    news: tuple[StockNewsView, ...] = Field(..., description="Recent news items for the stock.")
    ticks: tuple[StockPriceTickView, ...] = Field(
        ..., description="Recent price ticks for the 7D chart."
    )


class StockPortfolioHolding(BaseModel):
    """One non-zero position valued at the current quote.

    The valuation fields are computed per read rather than stored, so a holding is only as fresh
    as the tick advance that produced it. `short_cover_cost` is what covering would cost right
    now, and a short contributes its collateral plus its entry value minus that cost to
    `equity_value`, never a plain negative market value.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Virtual company ticker symbol.")
    name: str = Field(..., description="Display name of the virtual company.")
    price_cents: int = Field(..., description="Latest quote price in cents.")
    long_shares: int = Field(..., description="Number of shares held long.")
    long_cost_basis: int = Field(
        ..., description="Aggregate cost basis of the long position in wallet units."
    )
    long_market_value: int = Field(
        ..., description="Current market value of the long position in wallet units."
    )
    short_shares: int = Field(..., description="Number of shares held short.")
    short_entry_value: int = Field(
        ..., description="Aggregate entry value of the short position in wallet units."
    )
    short_collateral: int = Field(
        ..., description="Collateral reserved against the short position in wallet units."
    )
    short_cover_cost: int = Field(
        ..., description="Current cost to cover the short position in wallet units."
    )
    equity_value: int = Field(..., description="Net equity value of this holding in wallet units.")
    unrealized_pnl: int = Field(
        ..., description="Unrealized profit and loss for this holding in wallet units."
    )
    realized_pnl: int = Field(
        ..., description="Realized profit and loss for this holding in wallet units."
    )


class StockPortfolioView(BaseModel):
    """A user's whole market exposure, summed from their non-zero holdings.

    Its consumer is the economy profile embed rather than `/stock`, which is exactly why the
    shape lives at this layer instead of inside either cog. The totals are plain sums of
    `holdings`, so an empty portfolio is zeroes rather than absent. A process cache serves repeat
    reads of the default call only, and is skipped outright when `now` or `rng` is passed. Three
    writes invalidate it (a profile upsert, a finalized trade and a position reset), so a stale
    portfolio cannot outlive a trade; a tick advance does not, so a valuation instead goes stale
    on the cache's own short TTL.
    """

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(..., description="Discord user ID owning the portfolio.")
    holdings: tuple[StockPortfolioHolding, ...] = Field(
        ..., description="Current stock holdings for the user."
    )
    equity_value: int = Field(
        ..., description="Total net equity value across all holdings in wallet units."
    )
    unrealized_pnl: int = Field(
        ..., description="Total unrealized profit and loss across holdings in wallet units."
    )
    realized_pnl: int = Field(
        ..., description="Total realized profit and loss across holdings in wallet units."
    )


class StockSettlementResult(BaseModel):
    """The single return shape of `settle_stock_operation`, success or not.

    Validation and lifecycle failures come back as a result rather than as an exception, which is
    what the function's own noqa claims and all it claims: an unknown symbol still raises
    `ValueError` before anything is written, and a cancellation is re-raised once the operation
    has been stamped RECONCILE_REQUIRED. Within that, `success` is what a caller branches on and
    `error` carries the user-facing text.

    `operation_id` and `status` are None only for a rejection made before any row was written,
    which echoes the submit-time `position` back unchanged. A submission refused because an
    earlier operation is still unresolved is the one case where that pair names an operation
    other than this request. A RECONCILE_REQUIRED `status` means the two databases are out of
    step and the user stays blocked on that symbol until an operator clears it.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(..., description="Whether the settlement completed successfully.")
    operation_id: str | None = Field(
        ..., description="Identifier of the settled operation, if created."
    )
    symbol: str = Field(..., description="Virtual company ticker symbol.")
    requested_action: StockAction = Field(
        ..., description="Stock operation family requested by the user."
    )
    shares: int = Field(..., description="Share quantity settled.")
    price_cents: int = Field(..., description="Execution price in cents.")
    wallet_delta: int = Field(..., description="Net wallet balance change from the settlement.")
    balance_after: int = Field(..., description="Wallet cash balance after settlement.")
    position: StockPositionView = Field(..., description="Resulting position after settlement.")
    legs: tuple[StockTradeLegView, ...] = Field(
        ..., description="Trade legs produced by the settlement."
    )
    status: StockOperationStatus | None = Field(
        default=None, description="Final operation lifecycle status, if known."
    )
    error: str = Field(default="", description="Error message when the settlement fails.")


class StockReconciliationOperation(BaseModel):
    """An operation stuck short of a final state, with its legs, for an operator to judge.

    Nothing repairs these automatically: the legs say what the stock side intended and
    `failure_reason` says where it stopped, but only a human can tell whether the wallet moved,
    so the fix is an offline decision rather than a retry the bot could make.
    """

    model_config = ConfigDict(frozen=True)

    operation_id: str = Field(..., description="Stock operation identifier.")
    status: StockOperationStatus = Field(..., description="Current operation lifecycle status.")
    user_id: int = Field(..., description="Discord user ID that initiated the operation.")
    user_name: str = Field(..., description="Stored display name of the operation owner.")
    symbol: str = Field(..., description="Virtual company ticker symbol.")
    requested_action: StockAction = Field(
        ..., description="Stock operation family requested by the user."
    )
    failure_reason: str = Field(..., description="Recorded reason the operation did not finalize.")
    created_at: datetime = Field(..., description="Timestamp the operation was created.")
    updated_at: datetime = Field(..., description="Timestamp of the latest operation update.")
    legs: tuple[StockTradeLegView, ...] = Field(
        ..., description="Trade legs associated with the operation."
    )


class StockSupplyAuditView(BaseModel):
    """Per-symbol supply against aggregate exposure, for an operator retuning a company.

    Read without advancing any tick, so it reports the market as stored rather than as it would
    be after a lazy catch-up. The two `available_*` figures already reserve the shares that
    `non_final_operations` might still consume, which is why that count is carried alongside
    them: a non-zero one means the remaining capacity is provisional.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Virtual company ticker symbol.")
    name: str = Field(..., description="Display name of the virtual company.")
    price_cents: int = Field(..., description="Latest quote price in cents.")
    total_shares: int = Field(..., description="Total issued share count.")
    float_shares: int = Field(..., description="Tradable float share count.")
    long_shares: int = Field(..., description="Aggregate long exposure in shares.")
    short_shares: int = Field(..., description="Aggregate short borrow in shares.")
    available_long_shares: int = Field(..., description="Remaining long capacity in shares.")
    available_short_shares: int = Field(
        ..., description="Remaining short borrow capacity in shares."
    )
    liquidity_shares: int = Field(
        ..., description="Liquidity depth in shares used for order-size slippage."
    )
    non_final_operations: int = Field(
        ..., description="Count of operations not yet in a final state."
    )


__all__ = [
    "MAX_TICKS_PER_INTERACTION",
    "STOCK_ACTION_TIMEOUT_SECONDS",
    "STOCK_BPS_DENOMINATOR",
    "STOCK_HISTORY_DAYS",
    "STOCK_INDIVIDUAL_OWNERSHIP_CAP_BPS",
    "STOCK_NEWS_CADENCE_HOURS",
    "STOCK_TICK_SECONDS",
    "StockAction",
    "StockDetailViewData",
    "StockGeneratedNews",
    "StockMarketQuote",
    "StockNewsGenerationContext",
    "StockNewsView",
    "StockOperationStatus",
    "StockParticipantPositionView",
    "StockPortfolioHolding",
    "StockPortfolioView",
    "StockPositionView",
    "StockPriceTickView",
    "StockProfileUpsert",
    "StockProfileView",
    "StockReconciliationOperation",
    "StockSettlementResult",
    "StockSupplyAuditView",
    "StockTradeLegType",
    "StockTradeLegView",
]
