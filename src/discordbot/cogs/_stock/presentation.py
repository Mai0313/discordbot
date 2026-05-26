"""Discord embed and image builders for the simulated stock market."""

from io import BytesIO
from typing import TypedDict
from functools import cache, lru_cache

from PIL import Image, ImageDraw, ImageFont
from nextcord import Embed
from pydantic import BaseModel, ConfigDict

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
from discordbot.utils.number_text import compact_amount, share_quantity_text
from discordbot.cogs._stock.market import cash_floor, format_price
from discordbot.cogs._economy.presentation import CURRENCY_NAME, amount_code, currency_text

MARKET_COLOR = 0x2ECC71
DETAIL_COLOR = 0x3498DB
NEWS_COLOR = 0xF1C40F
ERROR_COLOR = 0xED4245
SUCCESS_COLOR = 0x57F287
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
_REGULAR_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "NotoSansCJK-Regular.ttc",
    "DejaVuSans.ttf",
)
_BOLD_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "NotoSansCJK-Bold.ttc",
    "DejaVuSans-Bold.ttf",
)
type _MarketFont = ImageFont.ImageFont | ImageFont.FreeTypeFont


class _MarketFonts(TypedDict):
    """Font set used by the stock market board image."""

    title: _MarketFont
    header: _MarketFont
    symbol: _MarketFont
    body: _MarketFont
    small: _MarketFont


class _MarketBoardQuote(BaseModel):
    """Immutable quote fields needed by the market board renderer."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    name: str
    category: str
    price_cents: int
    total_shares: int
    change_bps: int
    pressure_bps: int


class _MarketBoardSpec(BaseModel):
    """Immutable market board render spec used as the process-cache key."""

    model_config = ConfigDict(frozen=True)

    quotes: tuple[_MarketBoardQuote, ...]
    page_index: int
    page_size: int


def market_board_filename(page_index: int) -> str:
    """Returns a stable market board attachment filename."""
    normalized_page = max(page_index, 0)
    return f"{MARKET_BOARD_FILENAME_PREFIX}_{normalized_page + 1}.png"


def signed_percent(bps: int) -> str:
    """Formats basis points as a signed percent."""
    return f"{bps / 100:+.2f}%"


def volatility_text(base_volatility_bps: int, volatility_amplifier_bps: int) -> str:
    """Formats stock volatility settings."""
    return f"{base_volatility_bps / 100:.2f}% x {volatility_amplifier_bps / 100:.2f}"


def build_market_embed(
    quotes: tuple[StockMarketQuote, ...],
    page_index: int = 0,
    page_size: int = 25,
    board_filename: str | None = None,
) -> Embed:
    """Builds the public market list embed."""
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
    """Returns a cached market list PNG board for immutable quote rows."""
    return _build_market_board_image_cached(
        spec=_market_board_spec(quotes=quotes, page_index=page_index, page_size=page_size)
    )


def invalidate_stock_market_board_cache(symbol: str | None = None) -> None:
    """Clears process-local market board images."""
    del symbol
    _build_market_board_image_cached.cache_clear()


def _market_board_spec(
    quotes: tuple[StockMarketQuote, ...], page_index: int, page_size: int
) -> _MarketBoardSpec:
    """Extracts only the quote fields that affect market board pixels."""
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


@lru_cache(maxsize=128)
def _build_market_board_image_cached(spec: _MarketBoardSpec) -> bytes:
    """Renders the market list as a fixed-layout PNG board."""
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
    """Returns normalized market page metadata and rows."""
    safe_page_size = max(page_size, 1)
    page_count = max((len(quotes) + safe_page_size - 1) // safe_page_size, 1)
    normalized_page = min(max(page_index, 0), page_count - 1)
    start = normalized_page * safe_page_size
    return page_count, normalized_page, quotes[start : start + safe_page_size]


@cache
def _market_fonts() -> _MarketFonts:
    """Loads the market board fonts with a bundled-system fallback chain."""
    return {
        "title": _load_font(size=34, bold=True),
        "header": _load_font(size=20, bold=True),
        "symbol": _load_font(size=28, bold=True),
        "body": _load_font(size=24, bold=False),
        "small": _load_font(size=16, bold=False),
    }


def _load_font(size: int, bold: bool) -> _MarketFont:
    """Loads a CJK-capable font when available."""
    candidates = _BOLD_FONT_CANDIDATES if bold else _REGULAR_FONT_CANDIDATES
    for candidate in candidates:
        try:
            return ImageFont.truetype(font=candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_market_header(
    draw: ImageDraw.ImageDraw,
    fonts: _MarketFonts,
    quote_count: int,
    page_index: int,
    page_count: int,
) -> None:
    """Draws the board title and summary line."""
    x = _MARKET_BOARD_MARGIN
    y = _MARKET_BOARD_MARGIN
    draw.text(xy=(x, y), text="市場看板", font=fonts["title"], fill=_MARKET_TEXT)
    summary = f"{quote_count:,} 檔股票"
    if page_count > 1:
        summary = f"{summary} · 第 {page_index + 1}/{page_count} 頁"
    draw.text(xy=(x, y + 40), text=summary, font=fonts["small"], fill=_MARKET_MUTED)
    _draw_text_right(
        draw=draw,
        text=f"單位: {CURRENCY_NAME}",
        xy=(_MARKET_CAP_RIGHT, y + 40),
        font=fonts["small"],
        fill=_MARKET_MUTED,
    )


def _draw_market_table_header(draw: ImageDraw.ImageDraw, fonts: _MarketFonts, y: int) -> None:
    """Draws fixed-width table headers."""
    draw.rectangle(
        xy=(_MARKET_TABLE_LEFT, y, _MARKET_TABLE_RIGHT, y + _MARKET_TABLE_HEADER_HEIGHT),
        fill=_MARKET_SURFACE,
    )
    baseline = y + 14
    draw.text(
        xy=(_MARKET_SYMBOL_X, baseline), text="代碼", font=fonts["header"], fill=_MARKET_MUTED
    )
    draw.text(
        xy=(_MARKET_COMPANY_X, baseline), text="公司", font=fonts["header"], fill=_MARKET_MUTED
    )
    draw.text(
        xy=(_MARKET_CATEGORY_X, baseline), text="分類", font=fonts["header"], fill=_MARKET_MUTED
    )
    _draw_text_right(
        draw=draw,
        text="股價",
        xy=(_MARKET_PRICE_RIGHT, baseline),
        font=fonts["header"],
        fill=_MARKET_MUTED,
    )
    _draw_text_right(
        draw=draw,
        text="今日",
        xy=(_MARKET_CHANGE_RIGHT, baseline),
        font=fonts["header"],
        fill=_MARKET_MUTED,
    )
    _draw_text_right(
        draw=draw,
        text="買賣壓力",
        xy=(_MARKET_PRESSURE_RIGHT, baseline),
        font=fonts["header"],
        fill=_MARKET_MUTED,
    )
    _draw_text_right(
        draw=draw,
        text="市值",
        xy=(_MARKET_CAP_RIGHT, baseline),
        font=fonts["header"],
        fill=_MARKET_MUTED,
    )


def _draw_market_row(
    draw: ImageDraw.ImageDraw,
    fonts: _MarketFonts,
    quote: _MarketBoardQuote,
    row_index: int,
    y: int,
) -> None:
    """Draws one stock quote row."""
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
    name = _fit_text(
        draw=draw, text=quote.name, font=fonts["body"], max_width=_MARKET_NAME_MAX_WIDTH
    )
    category = _fit_text(
        draw=draw, text=quote.category, font=fonts["small"], max_width=_MARKET_CATEGORY_MAX_WIDTH
    )
    draw.text(
        xy=(_MARKET_SYMBOL_X, y + 14), text=quote.symbol, font=fonts["symbol"], fill=_MARKET_ACCENT
    )
    draw.text(xy=(_MARKET_COMPANY_X, y + 8), text=name, font=fonts["body"], fill=_MARKET_TEXT)
    _draw_tag(draw=draw, text=category, xy=(_MARKET_CATEGORY_X, y + 19), font=fonts["small"])
    _draw_text_right(
        draw=draw,
        text=format_price(price_cents=quote.price_cents),
        xy=(_MARKET_PRICE_RIGHT, y + 12),
        font=fonts["body"],
        fill=_MARKET_TEXT,
    )
    _draw_text_right(
        draw=draw,
        text=signed_percent(bps=quote.change_bps),
        xy=(_MARKET_CHANGE_RIGHT, y + 12),
        font=fonts["body"],
        fill=_metric_color(bps=quote.change_bps),
    )
    _draw_text_right(
        draw=draw,
        text=signed_percent(bps=quote.pressure_bps),
        xy=(_MARKET_PRESSURE_RIGHT, y + 12),
        font=fonts["body"],
        fill=_metric_color(bps=quote.pressure_bps),
    )
    _draw_text_right(
        draw=draw,
        text=compact_amount(amount=market_cap),
        xy=(_MARKET_CAP_RIGHT, y + 12),
        font=fonts["body"],
        fill=_MARKET_TEXT,
    )


def _draw_empty_market_row(draw: ImageDraw.ImageDraw, fonts: _MarketFonts, y: int) -> None:
    """Draws an empty-state row in the market board."""
    draw.rectangle(
        xy=(_MARKET_TABLE_LEFT, y, _MARKET_TABLE_RIGHT, y + _MARKET_ROW_HEIGHT),
        fill=_MARKET_SURFACE,
    )
    draw.text(
        xy=(_MARKET_SYMBOL_X, y + 16),
        text="目前沒有可用的股票",
        font=fonts["body"],
        fill=_MARKET_MUTED,
    )


def _draw_tag(
    draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: _MarketFont
) -> None:
    """Draws a compact category tag."""
    x, y = xy
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    width = bbox[2] - bbox[0] + 16
    draw.rounded_rectangle(
        xy=(x, y, x + width, y + 20), radius=8, fill=(67, 57, 35), outline=(94, 78, 44)
    )
    draw.text(xy=(x + 8, y + 1), text=text, font=font, fill=_MARKET_TAG)


def _draw_text_right(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: _MarketFont,
    fill: tuple[int, int, int],
) -> None:
    """Draws text with its right edge anchored at x."""
    right, y = xy
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    draw.text(xy=(right - (bbox[2] - bbox[0]), y), text=text, font=font, fill=fill)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: _MarketFont, max_width: int) -> str:
    """Truncates text to fit the requested pixel width."""
    if _text_width(draw=draw, text=text, font=font) <= max_width:
        return text
    suffix = "..."
    low = 0
    high = len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        candidate = f"{text[:midpoint]}{suffix}"
        if _text_width(draw=draw, text=candidate, font=font) <= max_width:
            low = midpoint
        else:
            high = midpoint - 1
    if low == 0:
        return suffix
    return f"{text[:low]}{suffix}"


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: _MarketFont) -> int:
    """Returns rendered text width."""
    bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
    return bbox[2] - bbox[0]


def _metric_color(bps: int) -> tuple[int, int, int]:
    """Returns a metric color for signed basis points."""
    if bps > 0:
        return _MARKET_POSITIVE
    if bps < 0:
        return _MARKET_NEGATIVE
    return _MARKET_NEUTRAL


def build_stock_detail_embed(detail: StockDetailViewData, chart_filename: str) -> Embed:
    """Builds a public stock detail embed for the current interaction user."""
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
    """Builds a recent news embed for the public stock message."""
    if news:
        lines = [
            f"**{item.headline}**\n市場情緒 `{signed_percent(bps=item.sentiment_bps)}`"
            for item in news
        ]
    else:
        lines = ["目前沒有近期新聞"]
    return Embed(title=f"📰 {symbol} 近期新聞", description="\n\n".join(lines), color=NEWS_COLOR)


def build_tutorial_embed() -> Embed:
    """Builds a short tutorial embed for the public stock message."""
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
    """Builds the action selection prompt for a public stock message."""
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
    """Builds a trade success or reconciliation/failure embed."""
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
    """Builds a generic stock error embed."""
    return Embed(title="股票錯誤", description=message, color=ERROR_COLOR)


def _action_label(action: StockAction) -> str:
    """Returns the user-facing action label."""
    if action == StockAction.BUY:
        return "買入 / 回補做空"
    return "做空 / 賣出持股"


def _leg_type_label(leg_type: StockTradeLegType) -> str:
    """Returns a compact trade-leg label."""
    labels = {
        StockTradeLegType.OPEN_LONG: "買入",
        StockTradeLegType.SELL_LONG: "賣出持股",
        StockTradeLegType.OPEN_SHORT: "做空",
        StockTradeLegType.COVER_SHORT: "回補做空",
    }
    return labels[leg_type]


def _recent_trade_lines(detail: StockDetailViewData) -> str:
    """Formats recent trade legs for a detail embed."""
    if not detail.recent_trades:
        return "尚無交易紀錄"
    return "\n".join(
        _recent_trade_line(index=index, leg=leg)
        for index, leg in enumerate(detail.recent_trades[:DETAIL_LIST_LIMIT], start=1)
    )


def _position_summary_lines(detail: StockDetailViewData) -> str:
    """Formats public non-zero stock positions."""
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
    """Formats one public position summary line."""
    name = position.user_name or str(position.user_id)
    return (
        f"{index}. **{name}** 持股 `{share_quantity_text(shares=position.long_shares)}`\n"
        f"-# 做空 `{share_quantity_text(shares=position.short_shares)}` · 已實現損益 "
        f"{amount_code(amount=position.realized_pnl, signed=True, compact=True)}"
    )


def _leg_lines(legs: tuple[StockTradeLegView, ...]) -> str:
    """Formats stock trade legs."""
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
    """Formats one compact recent trade line."""
    name = leg.user_name or str(leg.user_id)
    return (
        f"{index}. **{name}** {_leg_type_label(leg_type=leg.leg_type)} "
        f"`{share_quantity_text(shares=leg.shares)}` @ `{format_price(price_cents=leg.price_cents)}`\n"
        f"-# #{leg.leg_order} · 錢包變化 {amount_code(amount=leg.wallet_delta, signed=True, compact=True)} · "
        f"損益 {amount_code(amount=leg.realized_pnl_delta, signed=True, compact=True)}"
    )
