"""Pillow renderer for the 7D price line chart attached to one symbol's detail view.

It is the stock cog's second image builder and stays apart from `presentation.py` for two
reasons: that file draws the market board and every embed off `StockMarketQuote` rows, keyed on a
digest of pixel-affecting board fields, while this one draws a single symbol's price history off
`StockPriceTickView` rows and caches on the tick tuple itself; and every label here is ASCII, so
it needs none of the CJK font loading `utils/pil_text.py` exists for and draws with PIL's default
bitmap font instead.

`build_price_chart` is the only entry point. `views.py::edit_stock_detail` renders the ticks
`services/stock/database.py` read for the `STOCK_HISTORY_DAYS` window, wraps the bytes in a fresh
`BytesIO` per send and references them from the embed as `attachment://<symbol>_7d.png`; the
bytes are process-cached and therefore shared, so a caller must never reuse one stream.

The plot is deliberately minimal. x is the tick INDEX rather than elapsed time, so an uneven gap
between boundaries is invisible and `created_at` never reaches the drawing code; only the 7D
high, low and latest price are labelled, and there is no axis text.
"""

from io import BytesIO
from functools import lru_cache

from PIL import Image, ImageDraw

from discordbot.typings.stock import StockPriceTickView
from discordbot.services.stock.market import format_price

_WIDTH = 900
_HEIGHT = 360
_PADDING = 44
_LINE_COLOR = (87, 242, 135)
_AXIS_COLOR = (96, 103, 122)
_TEXT_COLOR = (230, 233, 239)
_BACKGROUND = (32, 34, 37)
_GRID_COLOR = (55, 59, 66)


def build_price_chart(ticks: tuple[StockPriceTickView, ...]) -> bytes:
    """Renders one symbol's 7D price chart as PNG bytes.

    Entry point over the process-cached renderer, so an unchanged tick tuple costs no Pillow work
    and hands back the same bytes object to every caller.

    Args:
        ticks (tuple[StockPriceTickView, ...]): The window's ticks, oldest first. An empty tuple
            renders a "NO TICKS" placeholder instead of failing.

    Returns:
        A complete PNG image, never empty.
    """
    return _render_price_chart(ticks=ticks)


# The cache key is the immutable tick tuple, so any quote/tick change yields a new
# key and a fresh render; stale entries can never be served and need no invalidation.
@lru_cache(maxsize=128)
def _render_price_chart(ticks: tuple[StockPriceTickView, ...]) -> bytes:
    """Draws the frame, gridlines, price line and the high/low/last labels.

    A single tick has no segment to draw, so it lands as one dot at the plot centre; a longer
    series marks only its last few points, which is what makes the recent end readable at this
    width. The labels go through PIL's default bitmap font, which is why every one of them is
    ASCII and positioned outside the plotted rectangle.

    Args:
        ticks (tuple[StockPriceTickView, ...]): The window's ticks, oldest first. Hashable and
            frozen, since this tuple is the cache key.

    Returns:
        A complete PNG image, never empty.
    """
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
    """Maps ticks onto pixel coordinates inside the padded plot area.

    The series is scaled to its own high and low rather than to an absolute price range, so the
    chart shows shape and never a flat line for a small move. A series with no move at all would
    divide by zero, so the span floors at 1 cent and every point settles on the baseline.

    Args:
        ticks (tuple[StockPriceTickView, ...]): The window's ticks, oldest first; spread evenly
            along x by position, not by the time between them.

    Returns:
        Coordinates in tick order: empty for no ticks, and a single centre point for one tick,
        which has no span to scale against.
    """
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
