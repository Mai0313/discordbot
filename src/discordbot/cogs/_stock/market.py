"""Pure market helpers for the simulated stock market."""

from random import Random
from datetime import UTC, datetime, timezone, timedelta

from discordbot.typings.stock import STOCK_TICK_SECONDS, MAX_TICKS_PER_INTERACTION

TAIWAN_TIMEZONE = timezone(offset=timedelta(hours=8), name="Asia/Taipei")
NEWS_SENTIMENT_DECAY_BPS = 20
NEWS_SENTIMENT_DECAY_SECONDS = 60 * 60
NEWS_SENTIMENT_LIMIT_BPS = 300
PRESSURE_LIMIT_BPS = 90


def cash_ceil(cents: int) -> int:
    """Converts cents to integer cash with a ceiling."""
    return (cents + 99) // 100


def cash_floor(cents: int) -> int:
    """Converts cents to integer cash with a floor."""
    return cents // 100


def format_price(price_cents: int) -> str:
    """Formats a cent-denominated stock price."""
    return f"{price_cents // 100:,}.{price_cents % 100:02d}"


def clamp_bps(value: int, lower: int, upper: int) -> int:
    """Clamps basis-point values."""
    return max(lower, min(upper, value))


def as_taipei(dt: datetime) -> datetime:
    """Returns `dt` interpreted in Asia/Taipei."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TAIWAN_TIMEZONE)
    return dt.astimezone(tz=TAIWAN_TIMEZONE)


def tick_boundary(dt: datetime) -> datetime:
    """Returns the tick-boundary timestamp at or before `dt`."""
    taipei_dt = as_taipei(dt=dt)
    seconds = int(taipei_dt.timestamp())
    boundary = seconds - (seconds % STOCK_TICK_SECONDS)
    return datetime.fromtimestamp(boundary, tz=UTC).astimezone(tz=TAIWAN_TIMEZONE)


def decay_news_sentiment(sentiment_bps: int, elapsed_seconds: int) -> int:
    """Applies linear time-based news sentiment decay and clamps the result."""
    clamped = clamp_bps(
        value=sentiment_bps, lower=-NEWS_SENTIMENT_LIMIT_BPS, upper=NEWS_SENTIMENT_LIMIT_BPS
    )
    decay_bps = max(elapsed_seconds, 0) * NEWS_SENTIMENT_DECAY_BPS // NEWS_SENTIMENT_DECAY_SECONDS
    remaining = max(abs(clamped) - decay_bps, 0)
    return remaining if clamped >= 0 else -remaining


def pressure_from_order_flow(net_shares: float, liquidity_shares: int) -> int:
    """Converts decayed net order flow into bounded price pressure."""
    if liquidity_shares <= 0 or net_shares == 0:
        return 0
    return clamp_bps(
        value=round(net_shares * PRESSURE_LIMIT_BPS / liquidity_shares),
        lower=-PRESSURE_LIMIT_BPS,
        upper=PRESSURE_LIMIT_BPS,
    )


def order_impact_bps(shares: int, liquidity_shares: int, max_impact_bps: int) -> int:
    """Converts an order size into bounded execution impact."""
    if shares <= 0 or liquidity_shares <= 0 or max_impact_bps <= 0:
        return 0
    raw_impact, remainder = divmod(shares * max_impact_bps, liquidity_shares)
    if remainder * 2 >= liquidity_shares:
        raw_impact += 1
    return clamp_bps(value=raw_impact, lower=0, upper=max_impact_bps)


def execution_price_cents(
    reference_price_cents: int,
    shares: int,
    liquidity_shares: int,
    max_impact_bps: int,
    is_buy: bool,
) -> int:
    """Returns a side-adjusted execution price for one order leg."""
    reference_price = max(reference_price_cents, 1)
    impact_bps = order_impact_bps(
        shares=shares, liquidity_shares=liquidity_shares, max_impact_bps=max_impact_bps
    )
    if is_buy:
        return max(reference_price * (10_000 + impact_bps) + 9_999, 1) // 10_000
    return max(reference_price * max(10_000 - impact_bps, 1) // 10_000, 1)


def mean_reversion_bps(
    previous_price_cents: int, fair_value_cents: int, mean_reversion_strength_bps: int
) -> int:
    """Returns the bounded fair-value pull for the next tick."""
    if previous_price_cents <= 0 or fair_value_cents <= 0 or mean_reversion_strength_bps <= 0:
        return 0
    fair_value_gap_bps = (fair_value_cents - previous_price_cents) * 10_000 // previous_price_cents
    return fair_value_gap_bps * mean_reversion_strength_bps // 10_000


def calculate_next_price_cents(  # noqa: PLR0913 -- pure price formula takes every market factor explicitly
    previous_price_cents: int,
    news_sentiment_bps: int,
    pressure_bps: int,
    base_volatility_bps: int,
    volatility_amplifier_bps: int,
    fair_value_cents: int,
    mean_reversion_strength_bps: int,
    max_tick_change_bps: int,
    rng: Random,
) -> int:
    """Calculates the next price using deterministic inputs plus seeded randomness."""
    volatility_width = max(base_volatility_bps * volatility_amplifier_bps // 100, 0)
    random_bps = rng.randint(-volatility_width, volatility_width) if volatility_width else 0
    raw_change_bps = (
        random_bps
        + clamp_bps(
            value=news_sentiment_bps,
            lower=-NEWS_SENTIMENT_LIMIT_BPS,
            upper=NEWS_SENTIMENT_LIMIT_BPS,
        )
        + clamp_bps(value=pressure_bps, lower=-PRESSURE_LIMIT_BPS, upper=PRESSURE_LIMIT_BPS)
        + mean_reversion_bps(
            previous_price_cents=previous_price_cents,
            fair_value_cents=fair_value_cents,
            mean_reversion_strength_bps=mean_reversion_strength_bps,
        )
    )
    change_limit = max(max_tick_change_bps, 1)
    change_bps = clamp_bps(value=raw_change_bps, lower=-change_limit, upper=change_limit)
    next_price = previous_price_cents * (10_000 + change_bps) // 10_000
    return max(next_price, 1)


def _tick_boundaries_between(
    latest_boundary: datetime, current_boundary: datetime
) -> list[datetime]:
    """Returns all tick boundaries between two normalized endpoints."""
    boundaries: list[datetime] = []
    boundary = latest_boundary + timedelta(seconds=STOCK_TICK_SECONDS)
    while boundary <= current_boundary:
        boundaries.append(boundary)
        boundary += timedelta(seconds=STOCK_TICK_SECONDS)
    return boundaries


def _compressed_required_boundaries(
    latest_boundary: datetime, current_boundary: datetime, boundaries: list[datetime]
) -> set[datetime]:
    """Returns boundaries that must survive compression for day rollover correctness."""
    selected = {current_boundary}
    boundary_set = set(boundaries)
    previous_boundary = latest_boundary
    for boundary in boundaries:
        if as_taipei(dt=boundary).date() != as_taipei(dt=previous_boundary).date():
            selected.add(boundary)
            if previous_boundary in boundary_set:
                selected.add(previous_boundary)
        previous_boundary = boundary
    return selected


def _fill_compressed_boundaries(
    boundaries: list[datetime], selected: set[datetime]
) -> tuple[datetime, ...]:
    """Fills a compressed boundary set with even sampling plus recent fallbacks."""
    total_steps = len(boundaries)
    for index in range(MAX_TICKS_PER_INTERACTION):
        source_index = (index + 1) * total_steps // MAX_TICKS_PER_INTERACTION - 1
        selected.add(boundaries[source_index])
        if len(selected) >= MAX_TICKS_PER_INTERACTION:
            break
    for boundary in reversed(boundaries):
        if len(selected) >= MAX_TICKS_PER_INTERACTION:
            break
        selected.add(boundary)
    return tuple(sorted(selected))


def tick_boundaries_to_apply(latest_tick_at: datetime, now: datetime) -> tuple[datetime, ...]:
    """Returns tick boundaries that should be materialized for a lazy interaction."""
    latest_boundary = tick_boundary(dt=latest_tick_at)
    current_boundary = tick_boundary(dt=now)
    if current_boundary <= latest_boundary:
        return ()

    boundaries = _tick_boundaries_between(
        latest_boundary=latest_boundary, current_boundary=current_boundary
    )

    if len(boundaries) <= MAX_TICKS_PER_INTERACTION:
        return tuple(boundaries)

    selected = _compressed_required_boundaries(
        latest_boundary=latest_boundary, current_boundary=current_boundary, boundaries=boundaries
    )
    if len(selected) >= MAX_TICKS_PER_INTERACTION:
        return tuple(sorted(selected)[-MAX_TICKS_PER_INTERACTION:])
    return _fill_compressed_boundaries(boundaries=boundaries, selected=selected)
