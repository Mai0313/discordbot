"""Discord embed builders for the simulated stock market."""

from nextcord import Embed

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
from discordbot.cogs._stock.market import cash_floor, format_price
from discordbot.cogs._economy.presentation import CURRENCY_NAME, amount_code, currency_text

MARKET_COLOR = 0x2ECC71
DETAIL_COLOR = 0x3498DB
NEWS_COLOR = 0xF1C40F
ERROR_COLOR = 0xED4245
SUCCESS_COLOR = 0x57F287


def signed_percent(bps: int) -> str:
    """Formats basis points as a signed percent."""
    return f"{bps / 100:+.2f}%"


def volatility_text(base_volatility_bps: int, volatility_amplifier_bps: int) -> str:
    """Formats stock volatility settings."""
    return f"{base_volatility_bps / 100:.2f}% x {volatility_amplifier_bps / 100:.2f}"


def build_market_embed(
    quotes: tuple[StockMarketQuote, ...], page_index: int = 0, page_size: int = 25
) -> Embed:
    """Builds the public market list embed."""
    title = "📈 模擬股市"
    detail_hint = "選擇股票後會在這則公開訊息更新股票明細。"
    description_parts = ["### 市場列表", detail_hint, ""]
    page_count = max((len(quotes) + page_size - 1) // page_size, 1)
    normalized_page = min(max(page_index, 0), page_count - 1)
    page_quotes = quotes[normalized_page * page_size : (normalized_page + 1) * page_size]
    for quote in page_quotes:
        profile = quote.profile
        market_cap = cash_floor(cents=profile.price_cents * profile.total_shares)
        description_parts.append(
            f"**{profile.symbol}** · {profile.name}\n"
            f"`{format_price(price_cents=profile.price_cents)}` "
            f"({signed_percent(bps=quote.change_bps)}) · "
            f"市值 {amount_code(amount=market_cap)} {CURRENCY_NAME}"
        )
    embed = Embed(title=title, description="\n".join(description_parts), color=MARKET_COLOR)
    footer = "這則股票訊息 180 秒無互動後會自動清理"
    if page_count > 1:
        footer += f" · 第 {normalized_page + 1}/{page_count} 頁"
    embed.set_footer(text=footer)
    return embed


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
        f"市值 {amount_code(amount=market_cap)} {CURRENCY_NAME}"
    )
    embed = Embed(title="📊 股票明細", description=description, color=DETAIL_COLOR)
    embed.add_field(
        name="目前操作使用者",
        value=detail.position.user_name or str(detail.position.user_id),
        inline=True,
    )
    embed.add_field(name="可用資金", value=currency_text(amount=detail.balance), inline=True)
    embed.add_field(
        name="持股",
        value=(
            f"持股數 `{detail.position.long_shares:,}` 股\n"
            f"持股成本 {amount_code(amount=detail.position.long_cost_basis)}"
        ),
        inline=True,
    )
    embed.add_field(
        name="做空",
        value=(
            f"做空股數 `{detail.position.short_shares:,}` 股\n"
            f"做空擔保金 {amount_code(amount=detail.position.short_collateral)}"
        ),
        inline=True,
    )
    embed.add_field(
        name="已實現損益",
        value=amount_code(amount=detail.position.realized_pnl, signed=True),
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
            f"目前持有：{detail.position.long_shares:,} 股 | "
            f"目前做空：{detail.position.short_shares:,} 股\n\n"
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
        f"成交股數 `{result.shares:,}`",
        f"成交價 `{format_price(price_cents=result.price_cents)}`",
        f"錢包變化 {amount_code(amount=result.wallet_delta, signed=True)}",
        f"餘額 {amount_code(amount=result.balance_after)} {CURRENCY_NAME}",
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
    return _leg_lines(legs=detail.recent_trades[:5])


def _position_summary_lines(detail: StockDetailViewData) -> str:
    """Formats public non-zero stock positions."""
    if not detail.public_positions:
        return "尚無公開部位"
    return "\n".join(
        _position_summary_line(position=position) for position in detail.public_positions[:5]
    )


def _position_summary_line(position: StockParticipantPositionView) -> str:
    """Formats one public position summary line."""
    name = position.user_name or str(position.user_id)
    return (
        f"{name} · 持股 `{position.long_shares:,}` 股 · 做空 `{position.short_shares:,}` 股 · "
        f"損益 {amount_code(amount=position.realized_pnl, signed=True)}"
    )


def _leg_lines(legs: tuple[StockTradeLegView, ...]) -> str:
    """Formats stock trade legs."""
    lines = []
    for leg in legs:
        name = leg.user_name or str(leg.user_id)
        lines.append(
            f"{name} · #{leg.leg_order} {_leg_type_label(leg_type=leg.leg_type)} "
            f"`{leg.shares:,}` 股 · 成交價 `{format_price(price_cents=leg.price_cents)}` · "
            f"錢包變化 {amount_code(amount=leg.wallet_delta, signed=True)} · "
            f"損益 {amount_code(amount=leg.realized_pnl_delta, signed=True)}"
        )
    return "\n".join(lines) if lines else "無"
