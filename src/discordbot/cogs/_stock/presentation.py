"""Discord embed builders for the simulated stock market."""

from nextcord import Embed

from discordbot.typings.stock import (
    StockAction,
    StockNewsView,
    StockMarketQuote,
    StockTradeLegType,
    StockTradeLegView,
    StockDetailViewData,
    StockSettlementResult,
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


def build_market_embed(quotes: tuple[StockMarketQuote, ...], ephemeral: bool = False) -> Embed:
    """Builds the public or ephemeral market list embed."""
    title = "📈 模擬股市"
    if ephemeral:
        title += " · 個人列表"
    description_parts = ["### 市場列表", "選擇股票後會開啟只有你看得到的 detail view。", ""]
    for quote in quotes:
        profile = quote.profile
        market_cap = cash_floor(cents=profile.price_cents * profile.total_shares)
        description_parts.append(
            f"**{profile.symbol}** · {profile.name}\n"
            f"`{format_price(price_cents=profile.price_cents)}` "
            f"({signed_percent(bps=quote.change_bps)}) · "
            f"Market cap {amount_code(amount=market_cap)} {CURRENCY_NAME}"
        )
    embed = Embed(title=title, description="\n".join(description_parts), color=MARKET_COLOR)
    embed.set_footer(text="公開 market list 會在 3 分鐘後自動清理，個人 detail 都是 ephemeral")
    return embed


def build_stock_detail_embed(detail: StockDetailViewData, chart_filename: str) -> Embed:
    """Builds a personal stock detail embed."""
    profile = detail.quote.profile
    market_cap = cash_floor(cents=profile.price_cents * profile.total_shares)
    description = (
        f"## {profile.symbol} · {profile.name}\n"
        f"### `{format_price(price_cents=profile.price_cents)}` "
        f"({signed_percent(bps=detail.quote.change_bps)})\n"
        f"Category `{profile.category}` · Volatility "
        f"`{volatility_text(base_volatility_bps=profile.base_volatility_bps, volatility_amplifier_bps=profile.volatility_amplifier_bps)}`\n"
        f"Market cap {amount_code(amount=market_cap)} {CURRENCY_NAME}"
    )
    embed = Embed(title="📊 股票 detail", description=description, color=DETAIL_COLOR)
    embed.add_field(name="你的資金", value=currency_text(amount=detail.balance), inline=True)
    embed.add_field(
        name="持股",
        value=(
            f"Long `{detail.position.long_shares:,}` 股\n"
            f"Cost basis {amount_code(amount=detail.position.long_cost_basis)}"
        ),
        inline=True,
    )
    embed.add_field(
        name="做空",
        value=(
            f"Short `{detail.position.short_shares:,}` 股\n"
            f"Collateral {amount_code(amount=detail.position.short_collateral)}"
        ),
        inline=True,
    )
    embed.add_field(
        name="Realized PnL",
        value=amount_code(amount=detail.position.realized_pnl, signed=True),
        inline=True,
    )
    embed.add_field(
        name="7D buy/sell pressure",
        value=signed_percent(bps=detail.quote.pressure_bps),
        inline=True,
    )
    embed.add_field(name="近期交易", value=_recent_trade_lines(detail=detail), inline=False)
    embed.set_image(url=f"attachment://{chart_filename}")
    return embed


def build_news_embed(news: tuple[StockNewsView, ...], symbol: str) -> Embed:
    """Builds an ephemeral recent news embed."""
    if news:
        lines = [
            f"**{item.headline}**\nSentiment `{signed_percent(bps=item.sentiment_bps)}`"
            for item in news
        ]
    else:
        lines = ["目前沒有近期新聞"]
    return Embed(title=f"📰 {symbol} 近期新聞", description="\n\n".join(lines), color=NEWS_COLOR)


def build_tutorial_embed() -> Embed:
    """Builds a short ephemeral tutorial embed."""
    return Embed(
        title="📘 模擬股市教學",
        description=(
            "`買入 / 回補做空` 會先 cover 既有 short，剩餘數量才 open long。\n"
            "`做空 / 賣出持股` 會先 sell 既有 long，剩餘數量才 open short。\n"
            "數量可以輸入整數或 `ALL`，實際價格與部位會在 modal submit 當下重新讀取。"
        ),
        color=DETAIL_COLOR,
    )


def build_settlement_embed(result: StockSettlementResult) -> Embed:
    """Builds a trade success or reconciliation/failure embed."""
    if not result.success:
        title = "股票交易失敗"
        if result.operation_id:
            title = "股票交易需要 reconciliation"
        embed = Embed(title=title, description=result.error or "交易沒有完成", color=ERROR_COLOR)
        if result.operation_id:
            embed.add_field(name="operation_id", value=f"`{result.operation_id}`", inline=False)
        return embed

    action_label = _action_label(action=result.requested_action)
    lines = [
        f"### {action_label} {result.symbol}",
        f"成交股數 `{result.shares:,}`",
        f"成交價 `{format_price(price_cents=result.price_cents)}`",
        f"Wallet delta {amount_code(amount=result.wallet_delta, signed=True)}",
        f"餘額 {amount_code(amount=result.balance_after)} {CURRENCY_NAME}",
    ]
    embed = Embed(title="股票交易完成", description="\n".join(lines), color=SUCCESS_COLOR)
    embed.add_field(name="交易 legs", value=_leg_lines(legs=result.legs), inline=False)
    if result.operation_id:
        embed.set_footer(text=f"operation_id={result.operation_id}")
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


def _leg_lines(legs: tuple[StockTradeLegView, ...]) -> str:
    """Formats stock trade legs."""
    lines = []
    for leg in legs:
        lines.append(
            f"{leg.leg_order}. {_leg_type_label(leg_type=leg.leg_type)} "
            f"`{leg.shares:,}` 股 · wallet {amount_code(amount=leg.wallet_delta, signed=True)} · "
            f"PnL {amount_code(amount=leg.realized_pnl_delta, signed=True)}"
        )
    return "\n".join(lines) if lines else "無"
