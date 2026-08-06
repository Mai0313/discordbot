"""Every surface the public `/stock` message renders: its embeds and its market board PNG.

The stock cog's rendering half. `cog.py` owns the slash command and `views.py` owns the controls
and the message edits; this file touches neither Discord nor `stock.db`. It takes the frozen read
models `typings/stock.py` declares and hands back an `Embed` or PNG bytes, so a number missing
from those models is a number nothing here can show, and nothing here recomputes one.

The market list is deliberately two artifacts. A page of CJK company names against 兆-scale market
caps does not line up in an embed's proportional text, so the table is drawn on a fixed column
grid as a PNG and referenced with `attachment://`, leaving the embed beside it carrying only the
title, the idle-cleanup notice and the page counter. Both halves take their filename from one
`market_board_filename` call, which is what keeps the reference and the upload identical.

Rendered board bytes are cached on `_MarketBoardSpec`, a frozen projection of exactly the quote
fields that reach a pixel, so a cached board cannot go stale in content: a moved price, change or
pressure is a different spec and therefore a different key, and no write path has to invalidate
anything. The rule that comes with it is the maintenance cost — a field newly drawn into a row
has to join `_MarketBoardQuote` too, or the render after it changes is served from the old key.
Those bytes are shared with every other caller of the same key, so a caller wraps them in a fresh
`BytesIO` per send rather than reusing a stream.

`chart.py` is the cog's other image builder and stays apart: it plots one symbol's price history
in ASCII in Pillow's default face, while every string here is Traditional Chinese and rides the
CJK font chain in `utils/pil_text.py`, which is also where the measuring and anchoring live.

Three unit vocabularies meet here and each has exactly one renderer, so no number is formatted
inline. Cent-denominated quote prices go through `services/stock/market.py::format_price`,
wallet-unit money through `services/economy/presentation.py` (`amount_code` / `currency_text`,
which is why a stock balance reads like an economy one), and basis points through
`signed_percent`. The volatility band is read from `effective_volatility_width_bps`, the same
function the price formula draws its per-tick move from, so what a user is shown is what is in
force rather than the raw profile knobs.

The message is public and the detail embed's shape follows from that: the viewer's own balance
and position sit beside a participant table and a recent-trade list naming other traders. Who is
eligible for those two is the service's decision; this file only orders them and trims the tail.
"""

from io import BytesIO
from functools import cache, lru_cache

from PIL import Image, ImageDraw
from nextcord import Embed
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.stock import (
    StockAction,
    StockNewsView,
    StockMarketQuote,
    StockTradeLegType,
    StockTradeLegView,
    StockDetailViewData,
    StockOperationStatus,
    StockSettlementResult,
    StockParticipantPositionView,
)
from discordbot.typings.colors import DISCORD_RED, DISCORD_GREEN
from discordbot.utils.currency import cash_floor
from discordbot.utils.pil_text import Font, fit_text, load_font, draw_text_right
from discordbot.utils.number_text import compact_amount, share_quantity_text
from discordbot.services.stock.market import format_price, effective_volatility_width_bps
from discordbot.services.economy.presentation import CURRENCY_NAME, amount_code, currency_text

MARKET_COLOR = 0x2ECC71
DETAIL_COLOR = 0x3498DB
NEWS_COLOR = 0xF1C40F
ERROR_COLOR = DISCORD_RED
SUCCESS_COLOR = DISCORD_GREEN
DETAIL_LIST_LIMIT = 3
MARKET_BOARD_WIDTH = 1120
MARKET_BOARD_FILENAME_PREFIX = "stock_market"
_MARKET_BOARD_MARGIN = 32
_MARKET_HEADER_HEIGHT = 64
_MARKET_TABLE_HEADER_HEIGHT = 48
_MARKET_ROW_HEIGHT = 58
_MARKET_BOARD_FOOTER_HEIGHT = 28
_MARKET_BACKGROUND = (28, 31, 36)
_MARKET_SURFACE = (38, 42, 49)
_MARKET_ROW_ALT = (33, 37, 43)
_MARKET_GRID = (70, 76, 88)
_MARKET_TEXT = (234, 237, 242)
_MARKET_MUTED = (169, 177, 190)
_MARKET_POSITIVE = (87, 242, 135)
_MARKET_NEGATIVE = (237, 66, 69)
_MARKET_NEUTRAL = (201, 207, 217)
_MARKET_ACCENT = (88, 166, 255)
_MARKET_TAG = (246, 196, 83)
_MARKET_TABLE_LEFT = _MARKET_BOARD_MARGIN
_MARKET_TABLE_RIGHT = MARKET_BOARD_WIDTH - _MARKET_BOARD_MARGIN
_MARKET_SYMBOL_X = 52
_MARKET_COMPANY_X = 150
_MARKET_CATEGORY_X = 456
_MARKET_PRICE_RIGHT = 676
_MARKET_CHANGE_RIGHT = 802
_MARKET_PRESSURE_RIGHT = 930
_MARKET_CAP_RIGHT = 1068
_MARKET_NAME_MAX_WIDTH = 280
_MARKET_CATEGORY_MAX_WIDTH = 112


class _MarketFonts(BaseModel):
    """The five faces one market board render draws with.

    `Font` is a Pillow union pydantic cannot model, hence `arbitrary_types_allowed`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: Font = Field(..., description="Board title font.")
    header: Font = Field(..., description="Table header font.")
    symbol: Font = Field(..., description="Stock symbol font.")
    body: Font = Field(..., description="Row text font.")
    small: Font = Field(..., description="Footer and badge font.")


class _MarketBoardQuote(BaseModel):
    """One row's worth of a quote: the fields that reach a pixel, and nothing else.

    Frozen so it can travel inside the render cache's key. What is absent is absent on purpose —
    a field the board does not draw would make an unchanged board a cache miss.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="Stock ticker symbol.")
    name: str = Field(..., description="Company display name.")
    category: str = Field(..., description="Company category label.")
    price_cents: int = Field(..., description="Quoted price in cents.")
    total_shares: int = Field(..., description="Total issued shares for market cap.")
    change_bps: int = Field(..., description="Price change in basis points.")
    pressure_bps: int = Field(..., description="Order-flow pressure in basis points.")


class _MarketBoardSpec(BaseModel):
    """Everything one board page is drawn from, and the render cache's key.

    Frozen so it can be that key. The rows travel inside it, which is what makes a cached board
    impossible to serve stale: different rows are a different spec.
    """

    model_config = ConfigDict(frozen=True)

    quotes: tuple[_MarketBoardQuote, ...] = Field(..., description="Quotes for the rendered page.")
    page_index: int = Field(..., description="Zero-based page index.")
    page_size: int = Field(..., description="Rows per page.")


def market_board_filename(page_index: int) -> str:
    """Returns the attachment filename for one page of the market board.

    The number in the name is one-based while the parameter is zero-based, and a negative index
    clamps to the first page, so the name is always well formed. `views.py` derives the embed's
    `attachment://` reference and the uploaded `File` from a single call, which is what keeps the
    two spellings identical.

    Args:
        page_index (int): Zero-based page index; anything below zero renders as page 1.

    Returns:
        A name such as `stock_market_1.png`.
    """
    normalized_page = max(page_index, 0)
    return f"{MARKET_BOARD_FILENAME_PREFIX}_{normalized_page + 1}.png"


def signed_percent(bps: int) -> str:
    """Formats basis points as a signed percent.

    Every basis-point figure a user sees goes through this, so a daily change, an order-flow
    pressure and a news sentiment all read alike. The sign is unconditional, so zero renders as
    `+0.00%` rather than bare.

    Args:
        bps (int): The basis-point value.

    Returns:
        Display text such as `+1.23%` or `-0.50%`.
    """
    return f"{bps / 100:+.2f}%"


def volatility_text(base_volatility_bps: int, volatility_amplifier_bps: int) -> str:
    """Formats a profile's per-tick volatility band as a symmetric percentage.

    The width comes from `effective_volatility_width_bps`, the same function the price formula
    draws its random move from, so the band shown is the one in force rather than the profile's
    raw knobs: the amplifier reads as a percentage of the base and the product is then cut by the
    global volatility scale.

    Args:
        base_volatility_bps (int): The profile's baseline per-tick volatility in basis points.
        volatility_amplifier_bps (int): The profile's amplifier, as a percentage of the base.

    Returns:
        Display text such as `±0.40%/tick`.
    """
    width_bps = effective_volatility_width_bps(
        base_volatility_bps=base_volatility_bps, volatility_amplifier_bps=volatility_amplifier_bps
    )
    return f"±{width_bps / 100:.2f}%/tick"


def build_market_embed(
    quotes: tuple[StockMarketQuote, ...],
    page_index: int = 0,
    page_size: int = 25,
    board_filename: str | None = None,
) -> Embed:
    """Builds the market list embed that frames the board attachment.

    It carries no rows itself: the table is the PNG, and the quotes are paged here only to work
    out the counter, so this and `build_market_board_image` have to be called with the same paging
    arguments or the footer describes a page the image is not. Given no `board_filename` the embed
    simply carries no image, for a caller that is not attaching one. The idle-cleanup window in
    the footer is written out rather than read from `STOCK_ACTION_TIMEOUT_SECONDS`, so retiming
    the view means retiming this line too.

    Args:
        quotes (tuple[StockMarketQuote, ...]): Every quote on the market, not just this page's.
        page_index (int): Zero-based page to describe; bounded into range.
        page_size (int): Rows per page, matching the board's own paging.
        board_filename (str | None): Name of the board attachment to reference, or None for an
            embed with no image.

    Returns:
        The market list embed, its footer carrying a page counter once there is more than one
        page.
    """
    title = "📈 模擬股市"
    description = "### 市場列表\n選擇股票後會在這則公開訊息更新股票明細。"
    if not quotes:
        description = "### 市場列表\n目前沒有可用的股票。"
    page_count, normalized_page, _page_quotes = _market_page(
        quotes=quotes, page_index=page_index, page_size=page_size
    )
    embed = Embed(title=title, description=description, color=MARKET_COLOR)
    if board_filename is not None:
        embed.set_image(url=f"attachment://{board_filename}")
    footer = "這則股票訊息 180 秒無互動後會自動清理"
    if page_count > 1:
        footer += f" · 第 {normalized_page + 1}/{page_count} 頁"
    embed.set_footer(text=footer)
    return embed


def build_market_board_image(
    quotes: tuple[StockMarketQuote, ...], page_index: int = 0, page_size: int = 25
) -> bytes:
    """Renders one page of the market list as a PNG board.

    Entry point over the process-cached renderer. It narrows the quotes to the fields that reach a
    pixel first, so two reads of an unchanged market hit the same key however much of the rest of
    the quote moved between them. The bytes are shared with every caller of that key, so a caller
    must wrap them in a fresh stream per send and never mutate them.

    Args:
        quotes (tuple[StockMarketQuote, ...]): Every quote on the market, not just this page's.
        page_index (int): Zero-based page to draw; bounded into range.
        page_size (int): Rows per page.

    Returns:
        PNG bytes, freshly rendered or served from the render cache.
    """
    return _build_market_board_image_cached(
        spec=_market_board_spec(quotes=quotes, page_index=page_index, page_size=page_size)
    )


def _market_board_spec(
    quotes: tuple[StockMarketQuote, ...], page_index: int, page_size: int
) -> _MarketBoardSpec:
    """Projects the quotes down to what the board draws, which is also the render cache's key.

    What is left out is deliberately not invalidating: `change_cents` is projected onto every
    quote and read nowhere in `src/`, and the profile's simulation knobs never reach a row. The
    other direction is the rule to keep — a field newly drawn into a row must be added here too,
    or a change to it keeps hitting the previous key.

    Args:
        quotes (tuple[StockMarketQuote, ...]): Every quote on the market, not just this page's.
        page_index (int): Zero-based page the spec describes.
        page_size (int): Rows per page.

    Returns:
        The frozen render spec.
    """
    return _MarketBoardSpec(
        quotes=tuple(
            _MarketBoardQuote(
                symbol=quote.profile.symbol,
                name=quote.profile.name,
                category=quote.profile.category,
                price_cents=quote.profile.price_cents,
                total_shares=quote.profile.total_shares,
                change_bps=quote.change_bps,
                pressure_bps=quote.pressure_bps,
            )
            for quote in quotes
        ),
        page_index=page_index,
        page_size=page_size,
    )


# The cache key is the frozen spec, carrying every pixel-affecting quote field, so any
# change to one yields a new key and a fresh render; stale entries are never served.
@lru_cache(maxsize=128)
def _build_market_board_image_cached(spec: _MarketBoardSpec) -> bytes:
    """Draws one board page and encodes it, behind the process render cache.

    Only the height varies: it grows with the row count, so a short last page is a shorter image
    rather than a padded one. An empty page still reserves a row, which is what the placeholder
    line is drawn into.

    Args:
        spec (_MarketBoardSpec): The page to draw, and the key it is cached under.

    Returns:
        The encoded PNG bytes.
    """
    page_count, normalized_page, page_quotes = _market_page(
        quotes=spec.quotes, page_index=spec.page_index, page_size=spec.page_size
    )
    row_count = max(len(page_quotes), 1)
    height = (
        _MARKET_BOARD_MARGIN * 2
        + _MARKET_HEADER_HEIGHT
        + _MARKET_TABLE_HEADER_HEIGHT
        + row_count * _MARKET_ROW_HEIGHT
        + _MARKET_BOARD_FOOTER_HEIGHT
    )
    image = Image.new(mode="RGB", size=(MARKET_BOARD_WIDTH, height), color=_MARKET_BACKGROUND)
    draw = ImageDraw.Draw(im=image)
    fonts = _market_fonts()
    _draw_market_header(
        draw=draw,
        fonts=fonts,
        quote_count=len(spec.quotes),
        page_index=normalized_page,
        page_count=page_count,
    )
    table_top = _MARKET_BOARD_MARGIN + _MARKET_HEADER_HEIGHT
    _draw_market_table_header(draw=draw, fonts=fonts, y=table_top)
    if page_quotes:
        for index, quote in enumerate(page_quotes):
            y = table_top + _MARKET_TABLE_HEADER_HEIGHT + index * _MARKET_ROW_HEIGHT
            _draw_market_row(draw=draw, fonts=fonts, quote=quote, row_index=index, y=y)
    else:
        _draw_empty_market_row(draw=draw, fonts=fonts, y=table_top + _MARKET_TABLE_HEADER_HEIGHT)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _market_page[MarketPageRow](
    quotes: tuple[MarketPageRow, ...], page_index: int, page_size: int
) -> tuple[int, int, tuple[MarketPageRow, ...]]:
    """Bounds a page request and slices the rows it names.

    Generic because the embed pages `StockMarketQuote` rows while the renderer pages the narrowed
    `_MarketBoardQuote` ones, and both have to land on the same page for the same arguments.
    Nothing here rejects a bad request: a page size under one is read as one and an out-of-range
    index is pulled back to the nearest real page, so an empty market reports one page rather than
    none.

    Args:
        quotes (tuple[MarketPageRow, ...]): Every row, not just this page's.
        page_index (int): Requested zero-based page.
        page_size (int): Requested rows per page.

    Returns:
        The page count, the bounded page index, and that page's rows.
    """
    safe_page_size = max(page_size, 1)
    page_count = max((len(quotes) + safe_page_size - 1) // safe_page_size, 1)
    normalized_page = min(max(page_index, 0), page_count - 1)
    start = normalized_page * safe_page_size
    return page_count, normalized_page, quotes[start : start + safe_page_size]


@cache
def _market_fonts() -> _MarketFonts:
    """Loads the board's five faces, once per process.

    The sizes never vary, so the faces are opened from disk once and shared by every render; the
    consequence is that a font installed or replaced there only takes effect after a restart.
    `load_font` degrades instead of raising, so a container with no CJK face renders boxes rather
    than failing the command.

    Returns:
        The shared font set; callers must treat it as read-only.
    """
    return _MarketFonts(
        title=load_font(size=34, bold=True),
        header=load_font(size=20, bold=True),
        symbol=load_font(size=28, bold=True),
        body=load_font(size=24, bold=False),
        small=load_font(size=16, bold=False),
    )


def _draw_market_header(
    draw: ImageDraw.ImageDraw,
    fonts: _MarketFonts,
    quote_count: int,
    page_index: int,
    page_count: int,
) -> None:
    """Draws the board title, the market summary line and the currency note.

    Every offset is taken from the top margin and none is measured, so this block occupies exactly
    `_MARKET_HEADER_HEIGHT` and the table below can start from that constant. The count is the
    whole market's rather than the page's, which is what the page indicator beside it qualifies.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        fonts (_MarketFonts): Faces for the title and the summary line.
        quote_count (int): Number of quotes on the whole market.
        page_index (int): Zero-based page index, rendered one-based.
        page_count (int): Total page count; the indicator is omitted at one.
    """
    x = _MARKET_BOARD_MARGIN
    y = _MARKET_BOARD_MARGIN
    draw.text(xy=(x, y), text="市場看板", font=fonts.title, fill=_MARKET_TEXT)
    summary = f"{quote_count:,} 檔股票"
    if page_count > 1:
        summary = f"{summary} · 第 {page_index + 1}/{page_count} 頁"
    draw.text(xy=(x, y + 40), text=summary, font=fonts.small, fill=_MARKET_MUTED)
    draw_text_right(
        draw=draw,
        text=f"單位: {CURRENCY_NAME}",
        xy=(_MARKET_CAP_RIGHT, y + 40),
        font=fonts.small,
        fill=_MARKET_MUTED,
    )


def _draw_market_table_header(draw: ImageDraw.ImageDraw, fonts: _MarketFonts, y: int) -> None:
    """Draws the column header strip at `y`.

    Uses the same column constants and the same right anchors as `_draw_market_row`, which is the
    only thing keeping a header over its own cells.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        fonts (_MarketFonts): Faces for the header row.
        y (int): Top edge of the strip.
    """
    draw.rectangle(
        xy=(_MARKET_TABLE_LEFT, y, _MARKET_TABLE_RIGHT, y + _MARKET_TABLE_HEADER_HEIGHT),
        fill=_MARKET_SURFACE,
    )
    baseline = y + 14
    draw.text(xy=(_MARKET_SYMBOL_X, baseline), text="代碼", font=fonts.header, fill=_MARKET_MUTED)
    draw.text(xy=(_MARKET_COMPANY_X, baseline), text="公司", font=fonts.header, fill=_MARKET_MUTED)
    draw.text(
        xy=(_MARKET_CATEGORY_X, baseline), text="分類", font=fonts.header, fill=_MARKET_MUTED
    )
    draw_text_right(
        draw=draw,
        text="股價",
        xy=(_MARKET_PRICE_RIGHT, baseline),
        font=fonts.header,
        fill=_MARKET_MUTED,
    )
    draw_text_right(
        draw=draw,
        text="今日",
        xy=(_MARKET_CHANGE_RIGHT, baseline),
        font=fonts.header,
        fill=_MARKET_MUTED,
    )
    draw_text_right(
        draw=draw,
        text="買賣壓力",
        xy=(_MARKET_PRESSURE_RIGHT, baseline),
        font=fonts.header,
        fill=_MARKET_MUTED,
    )
    draw_text_right(
        draw=draw,
        text="市值",
        xy=(_MARKET_CAP_RIGHT, baseline),
        font=fonts.header,
        fill=_MARKET_MUTED,
    )


def _draw_market_row(
    draw: ImageDraw.ImageDraw,
    fonts: _MarketFonts,
    quote: _MarketBoardQuote,
    row_index: int,
    y: int,
) -> None:
    """Draws one quote's row: its banding, its cells and the divider under it.

    Market cap is derived here rather than read, and floored out of cents into wallet units the
    way the service values a position, so the board never shows a sub-unit it rounded up. The
    company name and the category are trimmed against their own column widths before anything is
    drawn, so a long name pushes no cell to the right, and the change and pressure cells take
    their colour from the sign of the figure in them.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        fonts (_MarketFonts): Faces for the row's cells.
        quote (_MarketBoardQuote): The row's quote fields.
        row_index (int): Zero-based index on the page, which alternates the row background.
        y (int): Top edge of the row.
    """
    row_color = _MARKET_SURFACE if row_index % 2 == 0 else _MARKET_ROW_ALT
    draw.rectangle(
        xy=(_MARKET_TABLE_LEFT, y, _MARKET_TABLE_RIGHT, y + _MARKET_ROW_HEIGHT), fill=row_color
    )
    draw.line(
        xy=(
            _MARKET_TABLE_LEFT,
            y + _MARKET_ROW_HEIGHT,
            _MARKET_TABLE_RIGHT,
            y + _MARKET_ROW_HEIGHT,
        ),
        fill=_MARKET_GRID,
        width=1,
    )
    market_cap = cash_floor(cents=quote.price_cents * quote.total_shares)
    name = fit_text(draw=draw, text=quote.name, font=fonts.body, max_width=_MARKET_NAME_MAX_WIDTH)
    category = fit_text(
        draw=draw, text=quote.category, font=fonts.small, max_width=_MARKET_CATEGORY_MAX_WIDTH
    )
    draw.text(
        xy=(_MARKET_SYMBOL_X, y + 14), text=quote.symbol, font=fonts.symbol, fill=_MARKET_ACCENT
    )
    draw.text(xy=(_MARKET_COMPANY_X, y + 8), text=name, font=fonts.body, fill=_MARKET_TEXT)
    _draw_tag(draw=draw, text=category, xy=(_MARKET_CATEGORY_X, y + 19), font=fonts.small)
    draw_text_right(
        draw=draw,
        text=format_price(price_cents=quote.price_cents),
        xy=(_MARKET_PRICE_RIGHT, y + 12),
        font=fonts.body,
        fill=_MARKET_TEXT,
    )
    draw_text_right(
        draw=draw,
        text=signed_percent(bps=quote.change_bps),
        xy=(_MARKET_CHANGE_RIGHT, y + 12),
        font=fonts.body,
        fill=_metric_color(bps=quote.change_bps),
    )
    draw_text_right(
        draw=draw,
        text=signed_percent(bps=quote.pressure_bps),
        xy=(_MARKET_PRESSURE_RIGHT, y + 12),
        font=fonts.body,
        fill=_metric_color(bps=quote.pressure_bps),
    )
    draw_text_right(
        draw=draw,
        text=compact_amount(amount=market_cap),
        xy=(_MARKET_CAP_RIGHT, y + 12),
        font=fonts.body,
        fill=_MARKET_TEXT,
    )


def _draw_empty_market_row(draw: ImageDraw.ImageDraw, fonts: _MarketFonts, y: int) -> None:
    """Draws the placeholder row shown when the page holds no quotes.

    Takes a full `_MARKET_ROW_HEIGHT` because the image height already reserved a row for it, so
    an empty board keeps the table's proportions instead of collapsing onto its header.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        fonts (_MarketFonts): Faces for the placeholder line.
        y (int): Top edge of the row.
    """
    draw.rectangle(
        xy=(_MARKET_TABLE_LEFT, y, _MARKET_TABLE_RIGHT, y + _MARKET_ROW_HEIGHT),
        fill=_MARKET_SURFACE,
    )
    draw.text(
        xy=(_MARKET_SYMBOL_X, y + 16),
        text="目前沒有可用的股票",
        font=fonts.body,
        fill=_MARKET_MUTED,
    )


def _draw_tag(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: Font) -> None:
    """Draws the category pill in a row.

    The pill is measured rather than fixed: it is the rendered text's width plus a padding either
    side, so it hugs a short category instead of filling the column. Keeping the label inside
    `_MARKET_CATEGORY_MAX_WIDTH` is the caller's job, since nothing here re-checks the fit.

    Args:
        draw (ImageDraw.ImageDraw): Canvas to draw on.
        text (str): Category label, already trimmed to its column.
        xy (tuple[int, int]): Top-left corner of the pill.
        font (Font): Face for the label.
    """
    x, y = xy
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    width = bbox[2] - bbox[0] + 16
    draw.rounded_rectangle(
        xy=(x, y, x + width, y + 20), radius=8, fill=(67, 57, 35), outline=(94, 78, 44)
    )
    draw.text(xy=(x + 8, y + 1), text=text, font=font, fill=_MARKET_TAG)


def _metric_color(bps: int) -> tuple[int, int, int]:
    """Returns the cell colour for a signed basis-point metric.

    Only the sign is read, and a flat metric gets a third neutral colour rather than either
    direction's, so an untraded symbol does not read as a move.

    Args:
        bps (int): The basis-point value being coloured.

    Returns:
        The RGB colour for a rise, a fall or no change.
    """
    if bps > 0:
        return _MARKET_POSITIVE
    if bps < 0:
        return _MARKET_NEGATIVE
    return _MARKET_NEUTRAL


def build_stock_detail_embed(detail: StockDetailViewData, chart_filename: str) -> Embed:
    """Builds the detail embed for one symbol, around its 7D chart attachment.

    The message this goes into is public and so is all of it: the viewer's own balance and
    position sit next to a participant table and a recent-trade list naming other traders. Both of
    those lists are cut to `DETAIL_LIST_LIMIT` here, on top of the cut the service already made,
    so a busy stock does not turn two fields into the whole embed.

    Args:
        detail (StockDetailViewData): One gathered detail read: the quote, the viewer's position
            and balance, and the stock-wide participants, trades and ticks.
        chart_filename (str): Name of the 7D chart attachment to reference.

    Returns:
        The detail embed, referencing `chart_filename` as its image.
    """
    profile = detail.quote.profile
    market_cap = cash_floor(cents=profile.price_cents * profile.total_shares)
    description = (
        f"## {profile.symbol} · {profile.name}\n"
        f"### `{format_price(price_cents=profile.price_cents)}` "
        f"({signed_percent(bps=detail.quote.change_bps)})\n"
        f"分類 `{profile.category}` · 波動設定 "
        f"`{volatility_text(base_volatility_bps=profile.base_volatility_bps, volatility_amplifier_bps=profile.volatility_amplifier_bps)}`\n"
        f"市值 `{compact_amount(amount=market_cap)}` {CURRENCY_NAME}"
    )
    embed = Embed(title="📊 股票明細", description=description, color=DETAIL_COLOR)
    embed.add_field(
        name="目前操作使用者",
        value=detail.position.user_name or str(detail.position.user_id),
        inline=True,
    )
    embed.add_field(
        name="可用資金", value=currency_text(amount=detail.balance, compact=True), inline=True
    )
    embed.add_field(
        name="持股",
        value=(
            f"持股數 `{share_quantity_text(shares=detail.position.long_shares)}`\n"
            f"持股成本 {amount_code(amount=detail.position.long_cost_basis, compact=True)}"
        ),
        inline=True,
    )
    embed.add_field(
        name="做空",
        value=(
            f"做空股數 `{share_quantity_text(shares=detail.position.short_shares)}`\n"
            f"做空擔保金 {amount_code(amount=detail.position.short_collateral, compact=True)}"
        ),
        inline=True,
    )
    embed.add_field(
        name="已實現損益",
        value=amount_code(amount=detail.position.realized_pnl, signed=True, compact=True),
        inline=True,
    )
    embed.add_field(
        name="近 7 日買賣壓力", value=signed_percent(bps=detail.quote.pressure_bps), inline=True
    )
    embed.add_field(
        name="公開部位摘要", value=_position_summary_lines(detail=detail), inline=False
    )
    embed.add_field(name="近期交易", value=_recent_trade_lines(detail=detail), inline=False)
    embed.set_image(url=f"attachment://{chart_filename}")
    return embed


def build_news_embed(news: tuple[StockNewsView, ...], symbol: str) -> Embed:
    """Builds the recent-news embed for one symbol.

    Renders what it is handed in the order it arrives, newest first, each headline over its own
    sentiment. Nothing distinguishes a headline the LLM wrote from a deterministic template one:
    the reader is being shown the market's news, not its provenance.

    Args:
        news (tuple[StockNewsView, ...]): Recent news items, newest first.
        symbol (str): Ticker symbol, shown in the title.

    Returns:
        The news embed, carrying a placeholder line when the symbol has no news yet.
    """
    if news:
        lines = [
            f"**{item.headline}**\n市場情緒 `{signed_percent(bps=item.sentiment_bps)}`"
            for item in news
        ]
    else:
        lines = ["目前沒有近期新聞"]
    return Embed(title=f"📰 {symbol} 近期新聞", description="\n\n".join(lines), color=NEWS_COLOR)


def build_tutorial_embed() -> Embed:
    """Builds the tutorial embed for the public stock message.

    Static copy, and the only place a user is told the parts of a submission the UI cannot show:
    each direction closes the opposite side before opening, an overshooting quantity is clamped to
    what is executable rather than refused, and a large order pays slippage against liquidity.

    Returns:
        The tutorial embed.
    """
    return Embed(
        title="📘 模擬股市教學",
        description=(
            "`買入 / 回補做空` 會先回補既有做空，剩餘數量才建立持股。\n"
            "`做空 / 賣出持股` 會先賣出既有持股，剩餘數量才建立做空。\n"
            "選擇操作後會跳出數量視窗，可以輸入整數或 `ALL`，實際價格與部位會在送出當下重新讀取。"
            "如果輸入股數超過當下餘額、流通股或可借券上限，會自動改用可執行的最大股數。"
            "大單會依照 liquidity 產生 execution slippage。"
        ),
        color=DETAIL_COLOR,
    )


def build_action_prompt_embed(detail: StockDetailViewData) -> Embed:
    """Builds the prompt shown while an operation is being chosen.

    The price and the two position sides come from a detail read taken as the dropdown opens, and
    are a preview only: the quantity modal settles against the market as of submit, so what lands
    can be a tick away from what is shown here.

    Args:
        detail (StockDetailViewData): Fresh detail read for the symbol being operated on.

    Returns:
        The action prompt embed.
    """
    profile = detail.quote.profile
    return Embed(
        title=f"🧾 {profile.symbol} 股票操作",
        description=(
            f"股票代碼：{profile.symbol}\n"
            f"當前每股價格：{format_price(price_cents=profile.price_cents)} {CURRENCY_NAME}\n"
            f"目前持有：{share_quantity_text(shares=detail.position.long_shares)} | "
            f"目前做空：{share_quantity_text(shares=detail.position.short_shares)}\n\n"
            "請先選擇操作，接著會跳出數量視窗，可輸入股數或 `ALL`。"
        ),
        color=DETAIL_COLOR,
    )


def build_settlement_embed(result: StockSettlementResult) -> Embed:
    """Builds the outcome embed for a submitted operation, settled or not.

    A refusal that never reached a row is reported as a plain failure. One carrying an operation
    id and any status other than FAILED is reported as needing manual reconciliation instead,
    since the stock and economy databases may already disagree, and the id goes into the embed so
    an operator can find the parked operation. A success lists every leg because slippage is
    applied per leg: the price in the header is the share-weighted average across them, so the
    list is the only place the fills a trader actually got appear.

    Args:
        result (StockSettlementResult): The settlement outcome to render.

    Returns:
        The success embed, or an error embed titled for a refusal or for a reconciliation.
    """
    if not result.success:
        title = "股票交易失敗"
        if result.operation_id and result.status not in (None, StockOperationStatus.FAILED):
            title = "股票交易需要人工對帳"
        embed = Embed(title=title, description=result.error or "交易沒有完成", color=ERROR_COLOR)
        if result.operation_id:
            embed.add_field(name="操作代碼", value=f"`{result.operation_id}`", inline=False)
        return embed

    action_label = _action_label(action=result.requested_action)
    lines = [
        f"### {action_label} {result.symbol}",
        f"成交股數 `{share_quantity_text(shares=result.shares)}`",
        f"成交價 `{format_price(price_cents=result.price_cents)}`",
        f"錢包變化 {amount_code(amount=result.wallet_delta, signed=True, compact=True)}",
        f"餘額 {amount_code(amount=result.balance_after, compact=True)} {CURRENCY_NAME}",
    ]
    embed = Embed(title="股票交易完成", description="\n".join(lines), color=SUCCESS_COLOR)
    embed.add_field(name="交易明細", value=_leg_lines(legs=result.legs), inline=False)
    if result.operation_id:
        embed.set_footer(text=f"操作代碼: {result.operation_id}")
    return embed


def build_error_embed(message: str) -> Embed:
    """Builds the stock panel's error embed.

    One of several same-named builders in the tree: no cog may import a peer's helper, so the
    economy and fishing panels each carry their own rather than sharing this one.

    Args:
        message (str): The user-facing failure text to show.

    Returns:
        The error embed.
    """
    return Embed(title="股票錯誤", description=message, color=ERROR_COLOR)


def _action_label(action: StockAction) -> str:
    """Returns the user-facing label for a submitted direction.

    Both labels name two operations because a submission does two things: BUY covers any open
    short before opening long and SHORT sells any long before borrowing, so there is no separate
    close action to label.

    Args:
        action (StockAction): The requested direction; anything but BUY reads as SHORT.

    Returns:
        The label, such as `買入 / 回補做空`.
    """
    if action == StockAction.BUY:
        return "買入 / 回補做空"
    return "做空 / 賣出持股"


def _leg_type_label(leg_type: StockTradeLegType) -> str:
    """Returns the compact label for one trade leg.

    The four legs are what a submitted direction expands into, so these are the halves of
    `_action_label`'s two labels, named one at a time.

    Args:
        leg_type (StockTradeLegType): The leg to label.

    Returns:
        The label, such as `回補做空`.
    """
    labels = {
        StockTradeLegType.OPEN_LONG: "買入",
        StockTradeLegType.SELL_LONG: "賣出持股",
        StockTradeLegType.OPEN_SHORT: "做空",
        StockTradeLegType.COVER_SHORT: "回補做空",
    }
    return labels[leg_type]


def _recent_trade_lines(detail: StockDetailViewData) -> str:
    """Formats the stock's recent trades for the detail embed.

    Takes the first `DETAIL_LIST_LIMIT` in the order given, which the service already ordered
    newest first, so this trims a tail rather than ranking anything.

    Args:
        detail (StockDetailViewData): The detail read whose `recent_trades` to render.

    Returns:
        The numbered lines, or a placeholder when no applied trade is on record.
    """
    if not detail.recent_trades:
        return "尚無交易紀錄"
    return "\n".join(
        _recent_trade_line(index=index, leg=leg)
        for index, leg in enumerate(detail.recent_trades[:DETAIL_LIST_LIMIT], start=1)
    )


def _position_summary_lines(detail: StockDetailViewData) -> str:
    """Formats the public shareholder summary for the detail embed.

    Long holders only, largest first, capped at `DETAIL_LIST_LIMIT`: a participant holding nothing
    but a short is dropped outright, even though the line it would have printed carries its short
    size. Both the filter and the sort run over what the service already cut to its own top
    participants, so a long holder ranked out of that cut cannot appear here however large it is.

    Args:
        detail (StockDetailViewData): The detail read whose `public_positions` to render.

    Returns:
        The numbered lines, or a placeholder when nobody holds the stock long.
    """
    positions = sorted(
        (position for position in detail.public_positions if position.long_shares > 0),
        key=lambda position: position.long_shares,
        reverse=True,
    )
    if not positions:
        return "尚無公開部位"
    return "\n".join(
        _position_summary_line(index=index, position=position)
        for index, position in enumerate(positions[:DETAIL_LIST_LIMIT], start=1)
    )


def _position_summary_line(index: int, position: StockParticipantPositionView) -> str:
    """Formats one shareholder line, its short size and realized P&L pushed into subtext.

    Falls back to the user id as text when the participant has no stored name, so a position from
    before a name was ever recorded still renders as a row rather than a blank one.

    Args:
        index (int): One-based position in the list.
        position (StockParticipantPositionView): The participant to render.

    Returns:
        The rendered line, closing with a `-#` subtext line.
    """
    name = position.user_name or str(position.user_id)
    return (
        f"{index}. **{name}** 持股 `{share_quantity_text(shares=position.long_shares)}`\n"
        f"-# 做空 `{share_quantity_text(shares=position.short_shares)}` · 已實現損益 "
        f"{amount_code(amount=position.realized_pnl, signed=True, compact=True)}"
    )


def _leg_lines(legs: tuple[StockTradeLegView, ...]) -> str:
    """Formats every leg of one settled operation for the settlement embed.

    One line per leg carrying its own execution price, never netted, since that per-leg record is
    the whole audit trail there is. Every leg of one operation belongs to the same trader and the
    name is still repeated on each line, because the embed lands in a public message where the
    reader is usually not that trader.

    Args:
        legs (tuple[StockTradeLegView, ...]): The operation's legs, in `leg_order`.

    Returns:
        One line per leg, or `無` when the operation produced none.
    """
    lines = []
    for leg in legs:
        name = leg.user_name or str(leg.user_id)
        lines.append(
            f"{name} · #{leg.leg_order} {_leg_type_label(leg_type=leg.leg_type)} "
            f"`{share_quantity_text(shares=leg.shares)}` · 成交價 `{format_price(price_cents=leg.price_cents)}` · "
            f"錢包變化 {amount_code(amount=leg.wallet_delta, signed=True, compact=True)} · "
            f"損益 {amount_code(amount=leg.realized_pnl_delta, signed=True, compact=True)}"
        )
    return "\n".join(lines) if lines else "無"


def _recent_trade_line(index: int, leg: StockTradeLegView) -> str:
    """Formats one recent-trade line, its leg order and deltas pushed into subtext.

    Renders the same model as `_leg_lines` more compactly: this list sits inside the detail embed
    among other fields, so the name, direction and size lead and everything else drops to subtext.

    Args:
        index (int): One-based position in the list.
        leg (StockTradeLegView): The trade leg to render.

    Returns:
        The rendered line, closing with a `-#` subtext line.
    """
    name = leg.user_name or str(leg.user_id)
    return (
        f"{index}. **{name}** {_leg_type_label(leg_type=leg.leg_type)} "
        f"`{share_quantity_text(shares=leg.shares)}` @ `{format_price(price_cents=leg.price_cents)}`\n"
        f"-# #{leg.leg_order} · 錢包變化 {amount_code(amount=leg.wallet_delta, signed=True, compact=True)} · "
        f"損益 {amount_code(amount=leg.realized_pnl_delta, signed=True, compact=True)}"
    )
