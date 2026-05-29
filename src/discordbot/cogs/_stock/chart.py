"""Pillow chart rendering for stock detail views."""

from io import BytesIO
from functools import lru_cache

from PIL import Image, ImageDraw

from discordbot.typings.stock import StockPriceTickView
from discordbot.cogs._stock.market import format_price

_WIDTH = 900
_HEIGHT = 360
_PADDING = 44
_LINE_COLOR = (87, 242, 135)
_AXIS_COLOR = (96, 103, 122)
_TEXT_COLOR = (230, 233, 239)
_BACKGROUND = (32, 34, 37)
_GRID_COLOR = (55, 59, 66)


def build_price_chart(ticks: tuple[StockPriceTickView, ...]) -> bytes:
    """Returns a cached 7D price chart PNG for immutable tick rows."""
    return _render_price_chart(ticks=ticks)


def invalidate_stock_chart_cache() -> None:
    """Clears process-local stock chart images."""
    _render_price_chart.cache_clear()


@lru_cache(maxsize=128)
def _render_price_chart(ticks: tuple[StockPriceTickView, ...]) -> bytes:
    """Renders a simple non-empty 7D price chart PNG."""
    image = Image.new(mode="RGB", size=(_WIDTH, _HEIGHT), color=_BACKGROUND)
    draw = ImageDraw.Draw(im=image)
    draw.rectangle(
        xy=(_PADDING, _PADDING, _WIDTH - _PADDING, _HEIGHT - _PADDING),
        outline=_AXIS_COLOR,
        width=2,
    )
    for index in range(1, 4):
        y = _PADDING + (_HEIGHT - 2 * _PADDING) * index // 4
        draw.line(xy=(_PADDING, y, _WIDTH - _PADDING, y), fill=_GRID_COLOR, width=1)

    points = _chart_points(ticks=ticks)
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse(xy=(x - 4, y - 4, x + 4, y + 4), fill=_LINE_COLOR)
    elif points:
        draw.line(xy=points, fill=_LINE_COLOR, width=4, joint="curve")
        for x, y in points[-6:]:
            draw.ellipse(xy=(x - 3, y - 3, x + 3, y + 3), fill=_LINE_COLOR)

    if ticks:
        prices = [tick.price_cents for tick in ticks]
        high = max(prices)
        low = min(prices)
        latest = prices[-1]
        draw.text(xy=(16, 12), text=f"7D HIGH {format_price(price_cents=high)}", fill=_TEXT_COLOR)
        draw.text(xy=(16, 32), text=f"7D LOW {format_price(price_cents=low)}", fill=_TEXT_COLOR)
        draw.text(
            xy=(16, _HEIGHT - 32),
            text=f"LAST {format_price(price_cents=latest)}",
            fill=_TEXT_COLOR,
        )
    else:
        draw.text(xy=(16, 16), text="NO TICKS", fill=_TEXT_COLOR)

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _chart_points(ticks: tuple[StockPriceTickView, ...]) -> list[tuple[int, int]]:
    """Converts ticks into chart coordinates."""
    if not ticks:
        return []
    prices = [tick.price_cents for tick in ticks]
    low = min(prices)
    high = max(prices)
    span = max(high - low, 1)
    plot_width = _WIDTH - 2 * _PADDING
    plot_height = _HEIGHT - 2 * _PADDING
    if len(ticks) == 1:
        return [(_WIDTH // 2, _HEIGHT // 2)]
    points: list[tuple[int, int]] = []
    for index, tick in enumerate(ticks):
        x = _PADDING + plot_width * index // (len(ticks) - 1)
        y = _HEIGHT - _PADDING - (tick.price_cents - low) * plot_height // span
        points.append((x, y))
    return points
