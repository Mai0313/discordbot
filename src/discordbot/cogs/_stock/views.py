"""Interactive views for the simulated stock market."""

from io import BytesIO
from typing import cast

import nextcord
from nextcord import File, ButtonStyle, Interaction, SelectOption
from nextcord.ui import View, Modal, Button, TextInput, StringSelect

from discordbot.typings.stock import STOCK_ACTION_TIMEOUT_SECONDS, StockAction, StockMarketQuote
from discordbot.utils.avatars import guild_avatar_url
from discordbot.cogs._stock.chart import build_price_chart
from discordbot.cogs._stock.database import (
    get_stock_news,
    get_stock_detail,
    list_market_quotes,
    settle_stock_operation,
)
from discordbot.cogs._stock.presentation import (
    build_news_embed,
    build_error_embed,
    build_market_embed,
    build_tutorial_embed,
    build_settlement_embed,
    build_stock_detail_embed,
)


class StockMarketView(View):
    """Stock select and tutorial controls for the market list."""

    def __init__(self, quotes: tuple[StockMarketQuote, ...], ephemeral: bool = False) -> None:
        """Initializes market controls from quote rows."""
        super().__init__(timeout=STOCK_ACTION_TIMEOUT_SECONDS)
        self.quotes = quotes
        self.ephemeral = ephemeral
        self._select = cast("StringSelect", self.stock_select)
        self._select.options = [
            SelectOption(
                label=f"{quote.profile.symbol} · {quote.profile.name}",
                value=quote.profile.symbol,
                description=f"{quote.profile.category}",
            )
            for quote in quotes[:25]
        ] or [SelectOption(label="目前沒有股票", value="none", description="請稍後再試")]

    @nextcord.ui.string_select(
        placeholder="選擇股票",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中", value="loading")],
        custom_id="stock:select",
        row=0,
    )
    async def stock_select(self, select: StringSelect, interaction: Interaction) -> None:
        """Shows a personal detail view for the selected stock."""
        symbol = select.values[0]
        if symbol in {"loading", "none"}:
            await interaction.response.send_message(
                embed=build_error_embed(message="目前沒有可用的股票"), ephemeral=True
            )
            return
        await send_stock_detail(interaction=interaction, symbol=symbol)

    @nextcord.ui.button(
        label="教學", emoji="📘", style=ButtonStyle.secondary, custom_id="stock:tutorial", row=1
    )
    async def tutorial(self, _button: Button, interaction: Interaction) -> None:
        """Shows the stock tutorial privately."""
        await interaction.response.send_message(embed=build_tutorial_embed(), ephemeral=True)


class StockDetailView(View):
    """Personal detail controls for one stock."""

    def __init__(self, symbol: str) -> None:
        """Initializes detail controls for one symbol."""
        super().__init__(timeout=STOCK_ACTION_TIMEOUT_SECONDS)
        self.symbol = symbol

    @nextcord.ui.button(
        label="操作股票", emoji="🧾", style=ButtonStyle.primary, custom_id="stock:operate", row=0
    )
    async def operate(self, _button: Button, interaction: Interaction) -> None:
        """Opens the action selection view before the quantity modal."""
        await interaction.response.send_message(
            embed=build_tutorial_embed(), view=StockActionView(symbol=self.symbol), ephemeral=True
        )

    @nextcord.ui.button(
        label="近期新聞", emoji="📰", style=ButtonStyle.secondary, custom_id="stock:news", row=0
    )
    async def news(self, _button: Button, interaction: Interaction) -> None:
        """Shows recent deterministic news privately."""
        news = await get_stock_news(symbol=self.symbol)
        await interaction.response.send_message(
            embed=build_news_embed(news=news, symbol=self.symbol), ephemeral=True
        )

    @nextcord.ui.button(
        label="返回列表", emoji="↩️", style=ButtonStyle.secondary, custom_id="stock:back", row=1
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns an ephemeral market list instead of editing the public message."""
        quotes = await list_market_quotes()
        await interaction.response.send_message(
            embed=build_market_embed(quotes=quotes, ephemeral=True),
            view=StockMarketView(quotes=quotes, ephemeral=True),
            ephemeral=True,
        )


class StockActionView(View):
    """Action buttons shown before the quantity modal."""

    def __init__(self, symbol: str) -> None:
        """Initializes action controls for one symbol."""
        super().__init__(timeout=STOCK_ACTION_TIMEOUT_SECONDS)
        self.symbol = symbol

    @nextcord.ui.button(
        label="買入 / 回補做空",
        emoji="🟢",
        style=ButtonStyle.success,
        custom_id="stock:buy",
        row=0,
    )
    async def buy(self, _button: Button, interaction: Interaction) -> None:
        """Opens a quantity modal for buy/cover."""
        await interaction.response.send_modal(
            modal=StockQuantityModal(symbol=self.symbol, action=StockAction.BUY)
        )

    @nextcord.ui.button(
        label="做空 / 賣出持股",
        emoji="🔴",
        style=ButtonStyle.danger,
        custom_id="stock:short",
        row=0,
    )
    async def short(self, _button: Button, interaction: Interaction) -> None:
        """Opens a quantity modal for short/sell."""
        await interaction.response.send_modal(
            modal=StockQuantityModal(symbol=self.symbol, action=StockAction.SHORT)
        )


class StockPostTradeView(View):
    """Refresh control shown after a submitted stock trade."""

    def __init__(self, symbol: str) -> None:
        """Initializes post-trade controls."""
        super().__init__(timeout=STOCK_ACTION_TIMEOUT_SECONDS)
        self.symbol = symbol

    @nextcord.ui.button(
        label="刷新 detail",
        emoji="🔄",
        style=ButtonStyle.secondary,
        custom_id="stock:refresh",
        row=0,
    )
    async def refresh(self, _button: Button, interaction: Interaction) -> None:
        """Sends a fresh personal detail view."""
        await send_stock_detail(interaction=interaction, symbol=self.symbol)


class StockQuantityModal(Modal):
    """Quantity modal for stock operations."""

    def __init__(self, symbol: str, action: StockAction) -> None:
        """Initializes the modal with one TextInput."""
        super().__init__(title=f"{symbol} 股票操作")
        self.symbol = symbol
        self.action = action
        self.quantity: TextInput = TextInput(
            label="股數",
            placeholder="輸入正整數或 ALL",
            min_length=1,
            max_length=16,
            required=True,
        )
        self.add_item(item=self.quantity)

    async def callback(self, interaction: Interaction) -> None:
        """Submits the quantity to the stock settlement service."""
        await self.submit_quantity(
            interaction=interaction, raw_quantity=str(self.quantity.value or "")
        )

    async def submit_quantity(self, interaction: Interaction, raw_quantity: str) -> None:
        """Submits a raw quantity string to the stock settlement service."""
        if interaction.user is None:
            return
        await interaction.response.defer(ephemeral=True)
        avatar_url = await guild_avatar_url(
            user=interaction.user, guild=getattr(interaction, "guild", None)
        )
        result = await settle_stock_operation(
            symbol=self.symbol,
            user_id=interaction.user.id,
            user_name=interaction.user.name,
            avatar_url=avatar_url,
            requested_action=self.action,
            quantity=raw_quantity,
        )
        view = StockPostTradeView(symbol=self.symbol) if result.success else None
        await interaction.followup.send(
            embed=build_settlement_embed(result=result), view=view, ephemeral=True
        )


async def send_stock_detail(interaction: Interaction, symbol: str) -> None:
    """Sends a personal stock detail view for an interaction."""
    if interaction.user is None:
        return
    await interaction.response.defer(ephemeral=True)
    try:
        detail = await get_stock_detail(symbol=symbol, user_id=interaction.user.id)
    except ValueError:
        await interaction.followup.send(
            embed=build_error_embed(message=f"找不到股票 `{symbol}`"), ephemeral=True
        )
        return
    filename = f"{symbol.lower()}_7d.png"
    chart_bytes = build_price_chart(ticks=detail.ticks)
    await interaction.followup.send(
        embed=build_stock_detail_embed(detail=detail, chart_filename=filename),
        file=File(fp=BytesIO(chart_bytes), filename=filename),
        view=StockDetailView(symbol=symbol),
        ephemeral=True,
    )


__all__ = [
    "StockActionView",
    "StockDetailView",
    "StockMarketView",
    "StockPostTradeView",
    "StockQuantityModal",
    "send_stock_detail",
]
