"""Monte Carlo the simulated stock price formula to calibrate realism guardrails.

Drives the production price path (``calculate_next_price_cents`` plus the
Taiwan-style ``apply_daily_price_limit``) over many trading days and reports the
realized daily volatility, how often the daily price limit binds, and the
terminal drift away from fair value. Use it to re-measure ``MARKET_VOLATILITY_SCALE_BPS``,
``GLOBAL_MAX_TICK_CHANGE_BPS``, and ``DAILY_PRICE_LIMIT_BPS`` after any change.

Usage::

    uv run python scripts/simulate_stock_market.py
    uv run python scripts/simulate_stock_market.py --base-volatility-bps 180 --volatility-amplifier-bps 360
"""

import math
from random import Random
import argparse
from statistics import mean, median, pstdev
from collections.abc import Sequence

from rich.table import Table
from rich.console import Console

from discordbot.cogs._stock import market
from discordbot.typings.stock import STOCK_TICK_SECONDS
from discordbot.cogs._stock.market import (
    PRESSURE_LIMIT_BPS,
    DAILY_PRICE_LIMIT_BPS,
    GLOBAL_MAX_TICK_CHANGE_BPS,
    apply_daily_price_limit,
    calculate_next_price_cents,
    effective_volatility_width_bps,
)

console = Console()

TICKS_PER_DAY = 24 * 60 * 60 // STOCK_TICK_SECONDS


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description="Monte Carlo the simulated stock price formula.")
    parser.add_argument("--base-volatility-bps", type=int, default=180)
    parser.add_argument("--volatility-amplifier-bps", type=int, default=360)
    parser.add_argument("--max-tick-change-bps", type=int, default=850)
    parser.add_argument("--mean-reversion-bps", type=int, default=55)
    parser.add_argument("--fair-value-cents", type=int, default=10_000)
    parser.add_argument("--initial-price-cents", type=int, default=10_000)
    parser.add_argument("--news-sentiment-sigma", type=int, default=120)
    parser.add_argument("--news-cadence-hours", type=int, default=4)
    parser.add_argument("--pressure-sigma", type=int, default=15)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--volatility-scale-bps",
        type=int,
        default=None,
        help="Override MARKET_VOLATILITY_SCALE_BPS for this run only.",
    )
    return parser.parse_args(args=argv)


def _simulate_trial(args: argparse.Namespace, rng: Random) -> tuple[list[float], int, int, float]:
    """Runs one trial and returns daily returns, limit-up days, limit-down days, terminal ratio."""
    price = args.initial_price_cents
    previous_close = price
    news_cadence_ticks = max(args.news_cadence_hours * TICKS_PER_DAY // 24, 1)
    pressure_bps = 0
    daily_returns: list[float] = []
    limit_up_days = 0
    limit_down_days = 0
    for day in range(args.days):
        hit_up = False
        hit_down = False
        for tick in range(TICKS_PER_DAY):
            news_sentiment = 0
            if (day * TICKS_PER_DAY + tick) % news_cadence_ticks == 0:
                news_sentiment = int(rng.gauss(mu=0, sigma=args.news_sentiment_sigma))
            pressure_bps = max(
                -PRESSURE_LIMIT_BPS,
                min(
                    PRESSURE_LIMIT_BPS,
                    pressure_bps + int(rng.gauss(mu=0, sigma=args.pressure_sigma)),
                ),
            )
            raw_next = calculate_next_price_cents(
                previous_price_cents=price,
                news_sentiment_bps=news_sentiment,
                pressure_bps=pressure_bps,
                base_volatility_bps=args.base_volatility_bps,
                volatility_amplifier_bps=args.volatility_amplifier_bps,
                fair_value_cents=args.fair_value_cents,
                mean_reversion_strength_bps=args.mean_reversion_bps,
                max_tick_change_bps=args.max_tick_change_bps,
                rng=rng,
            )
            capped = apply_daily_price_limit(
                price_cents=raw_next,
                previous_close_cents=previous_close,
                limit_bps=DAILY_PRICE_LIMIT_BPS,
            )
            hit_up = hit_up or capped < raw_next
            hit_down = hit_down or capped > raw_next
            price = capped
        daily_returns.append(price / previous_close - 1.0)
        limit_up_days += int(hit_up)
        limit_down_days += int(hit_down)
        previous_close = price
    return daily_returns, limit_up_days, limit_down_days, price / args.fair_value_cents


def _report(args: argparse.Namespace) -> None:
    """Runs all trials and prints aggregate statistics."""
    all_daily_returns: list[float] = []
    limit_up_days = 0
    limit_down_days = 0
    terminal_ratios: list[float] = []
    for trial in range(args.trials):
        returns, up_days, down_days, terminal_ratio = _simulate_trial(
            args=args,
            rng=Random(args.seed + trial),  # noqa: S311 -- deterministic sim
        )
        all_daily_returns.extend(returns)
        limit_up_days += up_days
        limit_down_days += down_days
        terminal_ratios.append(terminal_ratio)

    total_days = args.trials * args.days
    daily_std = pstdev(all_daily_returns)
    width_bps = effective_volatility_width_bps(
        base_volatility_bps=args.base_volatility_bps,
        volatility_amplifier_bps=args.volatility_amplifier_bps,
    )

    table = Table(title="Simulated stock market statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Effective per-tick volatility width", f"±{width_bps / 100:.2f}%")
    table.add_row("Global per-tick change ceiling", f"±{GLOBAL_MAX_TICK_CHANGE_BPS / 100:.2f}%")
    table.add_row("Daily price limit", f"±{DAILY_PRICE_LIMIT_BPS / 100:.2f}%")
    table.add_row("Volatility scale", f"{market.MARKET_VOLATILITY_SCALE_BPS / 100:.1f}%")
    table.add_row("Mean daily return", f"{mean(all_daily_returns) * 100:+.3f}%")
    table.add_row("Daily return std", f"{daily_std * 100:.2f}%")
    table.add_row("Annualized volatility", f"{daily_std * math.sqrt(365) * 100:.1f}%")
    table.add_row("Limit-up day rate", f"{limit_up_days / total_days * 100:.2f}%")
    table.add_row("Limit-down day rate", f"{limit_down_days / total_days * 100:.2f}%")
    table.add_row("Median terminal / fair value", f"{median(terminal_ratios):.2f}x")
    table.add_row("Max terminal / fair value", f"{max(terminal_ratios):.2f}x")
    console.print(table)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the stock market simulation CLI."""
    args = _parse_args(argv=argv)
    if args.volatility_scale_bps is not None:
        market.MARKET_VOLATILITY_SCALE_BPS = args.volatility_scale_bps
    _report(args=args)


if __name__ == "__main__":
    main()
