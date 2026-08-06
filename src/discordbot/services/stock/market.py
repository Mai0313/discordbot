"""Pure price formula, tick arithmetic and market guardrails for the simulated stock market.

Everything here is a pure function: no session, no I/O, no clock read, no global RNG. Prices are
cent-denominated integers and every intensity is in basis points, so a tick replays exactly from
its inputs and the whole formula is testable with no database. Randomness arrives as an injected
`Random`, which is what lets `tests/test_stock.py` seed the Monte Carlo runs that pin the
guardrails below.

`services/stock/database.py` is the only caller that MOVES a price: it holds the per-symbol lock,
reads the profile, walks the boundaries this module hands it and writes the ticks. `cogs/stock/`
and `cogs/economy/embeds.py` reach in for display only (`format_price`,
`effective_volatility_width_bps`), so the number a user reads comes from the same arithmetic the
simulation ran instead of a second formula in the view layer.

Two mechanics are easy to mistake for each other. News reaches the PRICE as a one-shot impulse at
its own tick boundary (`_news_impulse_by_boundary`, in the storage layer), never as a drift over
later ticks; `decay_news_sentiment` here is the other thing entirely, the ambient value that only
feeds the prompt writing the next headline. And there is no market loop: an interaction advances a
symbol lazily, so `tick_boundaries_to_apply` hands back a whole backlog, compressed once it grows
past `MAX_TICKS_PER_INTERACTION` and compressed around the Asia/Taipei day rollovers the daily
price limit is anchored on.

The guardrails are what contains a wild per-company profile, since the profile knobs carry no
ceiling of their own; they were set by offline simulation and are re-measured, not adjusted by
eye.
"""

from random import Random
from datetime import UTC, datetime, timedelta

from discordbot.typings.stock import STOCK_TICK_SECONDS, MAX_TICKS_PER_INTERACTION
from discordbot.utils.timezone import TAIWAN_TIMEZONE, as_taipei

# The decay rate and the clamp are read together by the storage layer, which sizes its news query
# window from the two, so widening either widens that lookback instead of silently truncating it.
NEWS_SENTIMENT_DECAY_BPS = 20
NEWS_SENTIMENT_DECAY_SECONDS = 60 * 60
NEWS_SENTIMENT_LIMIT_BPS = 300
PRESSURE_LIMIT_BPS = 60

# Anti-inflation / realism guardrails for the price formula. A 5-minute tick at
# real-market magnitudes moves fractions of a percent, so the per-company random
# volatility is scaled down toward that range, every tick is hard-capped no matter
# how aggressive the per-company knob is, and each Asia/Taipei trading day is bounded
# to a Taiwan-style price limit measured against the previous close. Reducing per-tick
# swing and capping the daily move shrinks the volatility-harvesting that mints money.
# MARKET_VOLATILITY_SCALE_BPS was calibrated by offline simulation: at 800 the most
# aggressive production profile lands near 70% annualized volatility with the daily
# limit binding only a few percent of days, and calmer profiles scale down from there.
MARKET_VOLATILITY_SCALE_BPS = 800
GLOBAL_MAX_TICK_CHANGE_BPS = 200
DAILY_PRICE_LIMIT_BPS = 1_000


def format_price(price_cents: int) -> str:
    """Formats a cent-denominated stock price for display.

    Args:
        price_cents (int): The price in cents.

    Returns:
        The price in major units with a thousands separator and two decimals, e.g. `1,000.05`.
    """
    return f"{price_cents // 100:,}.{price_cents % 100:02d}"


def clamp_bps(value: int, lower: int, upper: int) -> int:
    """Clamps a basis-point value into an inclusive range.

    Args:
        value (int): The basis-point value to bound.
        lower (int): Inclusive lower bound.
        upper (int): Inclusive upper bound.

    Returns:
        `value`, bounded to `lower` and `upper`.
    """
    return max(lower, min(upper, value))


def tick_boundary(dt: datetime) -> datetime:
    """Returns the `STOCK_TICK_SECONDS` boundary at or before `dt`, in Asia/Taipei.

    The `as_taipei` normalization is what makes a naive input safe: SQLite hands datetimes back
    with no offset, and reading one as the container's local time would shift it by whole hours
    and with it the day the price limit anchors on. The result is returned in Taipei for the
    same reason, since callers compare `.date()` on it to spot a rollover.

    Args:
        dt (datetime): An aware timestamp, or a naive one already in Taipei wall time.

    Returns:
        The tick boundary at or before `dt`, as an aware Asia/Taipei datetime.
    """
    taipei_dt = as_taipei(dt=dt)
    seconds = int(taipei_dt.timestamp())
    boundary = seconds - (seconds % STOCK_TICK_SECONDS)
    return datetime.fromtimestamp(boundary, tz=UTC).astimezone(tz=TAIWAN_TIMEZONE)


def decay_news_sentiment(sentiment_bps: int, elapsed_seconds: int) -> int:
    """Fades one news row's sentiment linearly with its age, keeping its sign.

    Ambient context only. The price path fires a headline once, as an impulse at its own tick
    boundary, so this value reaches the model that writes the next headline and never the
    formula. The input is clamped before the fade, so an out-of-range row cannot outlive the
    decay window, and a negative age is read as zero elapsed rather than amplifying.

    Args:
        sentiment_bps (int): The stored sentiment of one news row, in basis points.
        elapsed_seconds (int): Age of that row at the moment being priced.

    Returns:
        The remaining sentiment in basis points, keeping the original sign and never crossing
        zero.
    """
    clamped = clamp_bps(
        value=sentiment_bps, lower=-NEWS_SENTIMENT_LIMIT_BPS, upper=NEWS_SENTIMENT_LIMIT_BPS
    )
    decay_bps = max(elapsed_seconds, 0) * NEWS_SENTIMENT_DECAY_BPS // NEWS_SENTIMENT_DECAY_SECONDS
    remaining = max(abs(clamped) - decay_bps, 0)
    return remaining if clamped >= 0 else -remaining


def pressure_from_order_flow(net_shares: float, liquidity_shares: int) -> int:
    """Converts decayed net order flow into bounded per-tick price pressure.

    Scaled so that net flow equal to the symbol's whole liquidity depth saturates the limit,
    which is what keeps a thin symbol from reading every ordinary order as a shock. A symbol
    with no declared liquidity produces no pressure at all rather than dividing by zero.

    Args:
        net_shares (float): Time-decayed shares bought minus sold; negative is sell-side.
        liquidity_shares (int): The symbol's liquidity depth in shares.

    Returns:
        Pressure in basis points, bounded to plus or minus `PRESSURE_LIMIT_BPS`.
    """
    if liquidity_shares <= 0 or net_shares == 0:
        return 0
    return clamp_bps(
        value=round(net_shares * PRESSURE_LIMIT_BPS / liquidity_shares),
        lower=-PRESSURE_LIMIT_BPS,
        upper=PRESSURE_LIMIT_BPS,
    )


def order_impact_bps(shares: int, liquidity_shares: int, max_impact_bps: int) -> int:
    """Converts one order leg's size into the slippage it pays, as a share of liquidity depth.

    The `divmod` plus the half-remainder bump is integer round-half-up: truncating would let an
    order worth less than a full basis point of the book fill at the untouched quote.

    Args:
        shares (int): Size of the order leg in shares.
        liquidity_shares (int): The symbol's liquidity depth in shares.
        max_impact_bps (int): Ceiling on the slippage this leg may pay.

    Returns:
        Impact in basis points between 0 and `max_impact_bps`, and 0 when any input is
        non-positive.
    """
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
    """Applies order-size slippage to a quote and returns the price one leg actually fills at.

    The rounding runs against the trader on both sides, a buy ceiling and a sell floor, so
    splitting one order into many legs cannot farm the rounding. Each leg of an operation is
    priced through here on its own and stored with its own `price_cents`; the legs are never
    netted back to the quote.

    Args:
        reference_price_cents (int): The quote to price against, read as at least 1 cent.
        shares (int): Size of this leg in shares.
        liquidity_shares (int): The symbol's liquidity depth in shares.
        max_impact_bps (int): Ceiling on the slippage this leg may pay.
        is_buy (bool): True prices the leg up, False prices it down.

    Returns:
        The execution price in cents, never below 1.
    """
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
    """Returns the pull toward fair value that the next tick carries, in basis points.

    The pull is a fraction of the gap rather than a fixed step, so it fades as the price closes
    on the anchor and a strength of 10_000 would land exactly on it. Nothing is clamped here:
    what bounds an aggressive profile is the per-tick ceiling in `calculate_next_price_cents`.
    The strength is the profile's own `mean_reversion_bps` column, renamed in the signature
    because the return value already carries that name.

    Args:
        previous_price_cents (int): The price this tick starts from.
        fair_value_cents (int): The anchor the price is pulled toward.
        mean_reversion_strength_bps (int): Fraction of the gap to close, in basis points.

    Returns:
        The signed pull in basis points, and 0 when any input is non-positive.
    """
    if previous_price_cents <= 0 or fair_value_cents <= 0 or mean_reversion_strength_bps <= 0:
        return 0
    fair_value_gap_bps = (fair_value_cents - previous_price_cents) * 10_000 // previous_price_cents
    return fair_value_gap_bps * mean_reversion_strength_bps // 10_000


def effective_volatility_width_bps(base_volatility_bps: int, volatility_amplifier_bps: int) -> int:
    """Returns the half-width of the per-tick random move, after the global volatility scale.

    Despite its `_bps` name the amplifier divides by 100, so it reads as a percentage of the
    base; the product is then cut by `MARKET_VOLATILITY_SCALE_BPS`. Neither profile knob has a
    ceiling of its own, so that scale is the only thing holding the random component near
    real-market magnitudes. `cogs/stock/presentation.py` renders the band from this same
    function, so the width a user is shown is the one the simulation drew from.

    Args:
        base_volatility_bps (int): The profile's baseline per-tick volatility in basis points.
        volatility_amplifier_bps (int): The profile's amplifier, as a percentage of the base.

    Returns:
        The symmetric half-width in basis points, never negative.
    """
    raw_width = base_volatility_bps * volatility_amplifier_bps // 100
    return max(raw_width * MARKET_VOLATILITY_SCALE_BPS // 10_000, 0)


def apply_daily_price_limit(price_cents: int, previous_close_cents: int, limit_bps: int) -> int:
    """Clamps a tick price into a Taiwan-style daily band around the previous close.

    The second half of the anti-inflation guardrail: the per-tick ceiling bounds one move, this
    bounds a whole Asia/Taipei trading day, which is what stops a run of same-sign ticks from
    harvesting volatility. The caller rolls `previous_close_cents` when it crosses a day
    boundary, so this function has no notion of when the day changed. A symbol with no recorded
    close, or a disabled limit, passes the price through.

    Args:
        price_cents (int): The tick price to bound.
        previous_close_cents (int): The previous trading day's close, in cents.
        limit_bps (int): Half-width of the band, in basis points.

    Returns:
        The bounded price in cents, never below 1.
    """
    if previous_close_cents <= 0 or limit_bps <= 0:
        return max(price_cents, 1)
    upper = previous_close_cents * (10_000 + limit_bps) // 10_000
    lower = max(previous_close_cents * (10_000 - limit_bps) // 10_000, 1)
    return max(min(price_cents, upper), lower)


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
    """Draws one tick's price from the profile knobs, the market state and injected randomness.

    The move is the sum of four terms, a symmetric random draw plus the clamped news impulse,
    the clamped order-flow pressure and the mean-reversion pull, and that sum is bounded by the
    tighter of the profile's own per-tick cap and `GLOBAL_MAX_TICK_CHANGE_BPS`, so a wild profile
    widens the raw move but cannot ship it. `news_sentiment_bps` is the one-shot impulse firing
    at THIS boundary, not a decayed running total. The daily band is deliberately left out: it
    needs the previous close, which the caller rolls as it walks the boundaries, so
    `apply_daily_price_limit` runs on the result.

    Randomness comes from `rng` and nowhere else, so a seed replays the tick exactly, which is
    what makes the guardrails measurable offline.

    Args:
        previous_price_cents (int): The price this tick starts from.
        news_sentiment_bps (int): News impulse firing at this boundary, in basis points.
        pressure_bps (int): Order-flow pressure at this boundary, in basis points.
        base_volatility_bps (int): The profile's baseline per-tick volatility in basis points.
        volatility_amplifier_bps (int): The profile's amplifier, as a percentage of the base.
        fair_value_cents (int): The mean-reversion anchor in cents.
        mean_reversion_strength_bps (int): Fraction of the fair-value gap to close, in basis
            points.
        max_tick_change_bps (int): The profile's own per-tick cap in basis points.
        rng (Random): Source of the random component; seeded in tests.

    Returns:
        The next price in cents, never below 1.
    """
    volatility_width = effective_volatility_width_bps(
        base_volatility_bps=base_volatility_bps, volatility_amplifier_bps=volatility_amplifier_bps
    )
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
    change_limit = max(min(max_tick_change_bps, GLOBAL_MAX_TICK_CHANGE_BPS), 1)
    change_bps = clamp_bps(value=raw_change_bps, lower=-change_limit, upper=change_limit)
    next_price = previous_price_cents * (10_000 + change_bps) // 10_000
    return max(next_price, 1)


def _tick_boundaries_between(
    latest_boundary: datetime, current_boundary: datetime
) -> list[datetime]:
    """Lists every tick boundary after `latest_boundary` up to and including `current_boundary`.

    Both endpoints must already be normalized through `tick_boundary`: the walk steps by exactly
    one tick, so an unaligned end would never be reached.

    Args:
        latest_boundary (datetime): The boundary already applied; excluded from the result.
        current_boundary (datetime): The newest boundary to apply; included.

    Returns:
        The boundaries in ascending order, empty when there is nothing new to apply.
    """
    boundaries: list[datetime] = []
    boundary = latest_boundary + timedelta(seconds=STOCK_TICK_SECONDS)
    while boundary <= current_boundary:
        boundaries.append(boundary)
        boundary += timedelta(seconds=STOCK_TICK_SECONDS)
    return boundaries


def _compressed_required_boundaries(
    latest_boundary: datetime, current_boundary: datetime, boundaries: list[datetime]
) -> set[datetime]:
    """Picks the boundaries a compressed backlog is not allowed to drop.

    Compression may thin the middle of a long backlog, but not across a day change: the last
    boundary of a day carries the close that the next day's price limit bands around, and the
    first boundary of the new day is where the caller rolls that close and the day open, so
    losing either would anchor the band on the wrong price. The newest boundary is always kept
    so the replay still ends where the caller asked. `latest_boundary` only seeds the day
    comparison and is never selected, since it has already been applied.

    Args:
        latest_boundary (datetime): The boundary already applied, used only as the day anchor.
        current_boundary (datetime): The newest boundary, always kept.
        boundaries (list[datetime]): The full uncompressed backlog, ascending.

    Returns:
        The subset that must survive compression, unordered.
    """
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
    """Tops a required-boundary set up to `MAX_TICKS_PER_INTERACTION` and orders it.

    The even sample spans the whole backlog so a compressed replay still walks the period rather
    than only its tail, and it can come up short when a sampled boundary was already required;
    the leftover slots are then filled from the newest end, which is where the prices a user is
    about to see come from. `selected` is mutated in place.

    Args:
        boundaries (list[datetime]): The full uncompressed backlog, ascending.
        selected (set[datetime]): The boundaries compression must keep.

    Returns:
        The filled set in ascending order, at most `MAX_TICKS_PER_INTERACTION` long.
    """
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
    """Returns the tick boundaries one lazy interaction should materialize.

    Nothing ticks a symbol on a schedule, so a quiet symbol catches up only when someone touches
    it and the backlog can be arbitrarily long. Up to `MAX_TICKS_PER_INTERACTION` boundaries
    replay one by one; past that the backlog is compressed around the day rollovers the daily
    price limit is anchored on. A backlog long enough that those rollovers alone overflow the cap
    keeps the most recent window, dropping the oldest days rather than the newest.

    Args:
        latest_tick_at (datetime): Timestamp of the last applied tick, normalized here.
        now (datetime): The moment to catch up to, normalized here.

    Returns:
        The boundaries in ascending order, empty when `now` is still inside the applied tick.
    """
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
