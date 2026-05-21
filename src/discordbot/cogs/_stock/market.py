"""Pure market helpers for the simulated stock market."""

from random import Random
from datetime import UTC, datetime, timezone, timedelta

from discordbot.typings.stock import STOCK_TICK_SECONDS, MAX_TICKS_PER_INTERACTION

TAIWAN_TIMEZONE = timezone(offset=timedelta(hours=8), name="Asia/Taipei")
NEWS_SENTIMENT_DECAY_BPS = 20
NEWS_SENTIMENT_LIMIT_BPS = 300
PRESSURE_LIMIT_BPS = 100


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
    """Returns ``dt`` interpreted in Asia/Taipei."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TAIWAN_TIMEZONE)
    return dt.astimezone(tz=TAIWAN_TIMEZONE)


def tick_boundary(dt: datetime) -> datetime:
    """Returns the tick-boundary timestamp at or before ``dt``."""
    taipei_dt = as_taipei(dt=dt)
    seconds = int(taipei_dt.timestamp())
    boundary = seconds - (seconds % STOCK_TICK_SECONDS)
    return datetime.fromtimestamp(boundary, tz=UTC).astimezone(tz=TAIWAN_TIMEZONE)


def decay_news_sentiment(sentiment_bps: int, ticks_elapsed: int) -> int:
    """Applies linear per-tick news sentiment decay and clamps the result."""
    clamped = clamp_bps(
        value=sentiment_bps, lower=-NEWS_SENTIMENT_LIMIT_BPS, upper=NEWS_SENTIMENT_LIMIT_BPS
    )
    remaining = max(abs(clamped) - ticks_elapsed * NEWS_SENTIMENT_DECAY_BPS, 0)
    return remaining if clamped >= 0 else -remaining


def pressure_from_volume(buy_shares: int, sell_shares: int) -> int:
    """Converts recent buy/sell volume into a bounded pressure value."""
    total = buy_shares + sell_shares
    if total <= 0:
        return 0
    return clamp_bps(
        value=(buy_shares - sell_shares) * PRESSURE_LIMIT_BPS // total,
        lower=-PRESSURE_LIMIT_BPS,
        upper=PRESSURE_LIMIT_BPS,
    )


def calculate_next_price_cents(  # noqa: PLR0913 -- pure price formula takes every market factor explicitly
    previous_price_cents: int,
    news_sentiment_bps: int,
    pressure_bps: int,
    base_volatility_bps: int,
    volatility_amplifier_bps: int,
    rng: Random,
) -> int:
    """Calculates the next price using deterministic inputs plus seeded randomness."""
    volatility_width = max(base_volatility_bps * volatility_amplifier_bps // 100, 0)
    random_bps = rng.randint(-volatility_width, volatility_width) if volatility_width else 0
    change_bps = (
        random_bps
        + clamp_bps(
            value=news_sentiment_bps,
            lower=-NEWS_SENTIMENT_LIMIT_BPS,
            upper=NEWS_SENTIMENT_LIMIT_BPS,
        )
        + clamp_bps(value=pressure_bps, lower=-PRESSURE_LIMIT_BPS, upper=PRESSURE_LIMIT_BPS)
    )
    next_price = previous_price_cents * (10_000 + change_bps) // 10_000
    return max(next_price, 1)


def tick_boundaries_to_apply(latest_tick_at: datetime, now: datetime) -> tuple[datetime, ...]:
    """Returns tick boundaries that should be materialized for a lazy interaction."""
    latest_boundary = tick_boundary(dt=latest_tick_at)
    current_boundary = tick_boundary(dt=now)
    if current_boundary <= latest_boundary:
        return ()

    boundaries: list[datetime] = []
    boundary = latest_boundary + timedelta(seconds=STOCK_TICK_SECONDS)
    while boundary <= current_boundary:
        boundaries.append(boundary)
        boundary += timedelta(seconds=STOCK_TICK_SECONDS)

    if len(boundaries) <= MAX_TICKS_PER_INTERACTION:
        return tuple(boundaries)

    total_steps = len(boundaries)
    compressed: list[datetime] = []
    for index in range(MAX_TICKS_PER_INTERACTION):
        source_index = (index + 1) * total_steps // MAX_TICKS_PER_INTERACTION - 1
        compressed.append(boundaries[source_index])
    if compressed[-1] != current_boundary:
        compressed[-1] = current_boundary
    return tuple(dict.fromkeys(compressed))
