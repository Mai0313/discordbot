"""Interactive views for the simulated stock market."""

from io import BytesIO
from typing import cast

import nextcord
from nextcord import File, User, Member, Message, ButtonStyle, Interaction, SelectOption
from nextcord.ui import View, Modal, Button, TextInput, StringSelect

from discordbot.typings.stock import STOCK_ACTION_TIMEOUT_SECONDS, StockAction, StockMarketQuote
from discordbot.utils.avatars import guild_avatar_url
from discordbot.cogs._stock.chart import build_price_chart
from discordbot.cogs._games.cleanup import forget_game_message
from discordbot.cogs._stock.database import (
    get_stock_news,
    get_stock_detail,
    list_market_quotes,
    settle_stock_operation,
)
from discordbot.cogs._games.interactions import send_ephemeral_notice
from discordbot.cogs._stock.presentation import (
    build_news_embed,
    build_error_embed,
    build_market_embed,
    build_tutorial_embed,
    build_settlement_embed,
    build_stock_detail_embed,
)


def require_stock_user(interaction: Interaction) -> User | Member:
    """Returns the interaction user or fails before any stock state can be written."""
    if interaction.user is None:
        raise RuntimeError("Stock interaction is missing Discord user identity")
    return interaction.user


class StockPublicView(View):
    """Base view for stock states that own the same public message."""

    def __init__(self, owner_id: int) -> None:
        """Initializes public stock controls with an idle timeout."""
        super().__init__(timeout=STOCK_ACTION_TIMEOUT_SECONDS)
        self.owner_id = owner_id
        self.message: Message | None = None

    def bind_message(self, message: Message | None) -> None:
        """Records the public message this view should update or delete."""
        self.message = message

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Allows only the user who opened this stock panel to operate it."""
        user = require_stock_user(interaction=interaction)
        if self.owner_id == user.id:
            return True
        await send_ephemeral_notice(
            interaction=interaction,
            content="這個股票面板只有發起者可以操作，請自己使用 `/stock` 開一個新的面板",
            log_message="Failed to send stock owner mismatch notice",
        )
        return False

    async def on_timeout(self) -> None:
        """Deletes the stock message after 180 seconds without interaction."""
        if self.message is None:
            return
        message_id = getattr(self.message, "id", None)
        try:
            await self.message.delete()
        except nextcord.NotFound:
            pass
        except (nextcord.Forbidden, nextcord.HTTPException):
            return
        if isinstance(message_id, int):
            await forget_game_message(message_id=message_id)


class StockMarketView(StockPublicView):
    """Stock select and tutorial controls for the market list."""

    def __init__(
        self, quotes: tuple[StockMarketQuote, ...], owner_id: int, ephemeral: bool = False
    ) -> None:
        """Initializes market controls from quote rows."""
        super().__init__(owner_id=owner_id)
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
        """Shows a public detail view for the selected stock."""
        symbol = select.values[0]
        if symbol in {"loading", "none"}:
            self.stop()
            await edit_stock_message(
                interaction=interaction,
                embed=build_error_embed(message="目前沒有可用的股票"),
                view=StockMarketView(
                    quotes=self.quotes, ephemeral=self.ephemeral, owner_id=self.owner_id
                ),
            )
            return
        self.stop()
        await send_stock_detail(interaction=interaction, symbol=symbol, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="教學", emoji="📘", style=ButtonStyle.secondary, custom_id="stock:tutorial", row=1
    )
    async def tutorial(self, _button: Button, interaction: Interaction) -> None:
        """Shows the stock tutorial in the public stock message."""
        self.stop()
        await edit_stock_message(
            interaction=interaction,
            embed=build_tutorial_embed(),
            view=StockTutorialView(owner_id=self.owner_id),
        )


class StockTutorialView(StockPublicView):
    """Tutorial controls for the public stock message."""

    def __init__(self, owner_id: int) -> None:
        """Initializes tutorial controls for the owning user."""
        super().__init__(owner_id=owner_id)

    @nextcord.ui.button(
        label="返回列表", emoji="↩️", style=ButtonStyle.secondary, custom_id="stock:tutorial:back"
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the public market list."""
        quotes = await list_market_quotes()
        self.stop()
        await edit_stock_message(
            interaction=interaction,
            embed=build_market_embed(quotes=quotes),
            view=StockMarketView(quotes=quotes, owner_id=self.owner_id),
        )


class StockDetailView(StockPublicView):
    """Personal detail controls for one stock."""

    def __init__(self, symbol: str, owner_id: int) -> None:
        """Initializes detail controls for one symbol."""
        super().__init__(owner_id=owner_id)
        self.symbol = symbol

    @nextcord.ui.button(
        label="操作股票", emoji="🧾", style=ButtonStyle.primary, custom_id="stock:operate", row=0
    )
    async def operate(self, _button: Button, interaction: Interaction) -> None:
        """Opens the action selection view before the quantity modal."""
        self.stop()
        await edit_stock_message(
            interaction=interaction,
            embed=build_tutorial_embed(),
            view=StockActionView(symbol=self.symbol, owner_id=self.owner_id),
        )

    @nextcord.ui.button(
        label="近期新聞", emoji="📰", style=ButtonStyle.secondary, custom_id="stock:news", row=0
    )
    async def news(self, _button: Button, interaction: Interaction) -> None:
        """Shows recent deterministic news in the public stock message."""
        news = await get_stock_news(symbol=self.symbol)
        self.stop()
        await edit_stock_message(
            interaction=interaction,
            embed=build_news_embed(news=news, symbol=self.symbol),
            view=StockNewsControlsView(symbol=self.symbol, owner_id=self.owner_id),
        )

    @nextcord.ui.button(
        label="返回列表", emoji="↩️", style=ButtonStyle.secondary, custom_id="stock:back", row=1
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the public market list."""
        quotes = await list_market_quotes()
        self.stop()
        await edit_stock_message(
            interaction=interaction,
            embed=build_market_embed(quotes=quotes),
            view=StockMarketView(quotes=quotes, owner_id=self.owner_id),
        )


class StockNewsControlsView(StockPublicView):
    """Navigation controls shown with a stock news embed."""

    def __init__(self, symbol: str, owner_id: int) -> None:
        """Initializes news controls for one symbol."""
        super().__init__(owner_id=owner_id)
        self.symbol = symbol

    @nextcord.ui.button(
        label="返回明細", emoji="↩️", style=ButtonStyle.secondary, custom_id="stock:news:back"
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the public stock detail view."""
        self.stop()
        await send_stock_detail(
            interaction=interaction, symbol=self.symbol, owner_id=self.owner_id
        )


class StockActionView(StockPublicView):
    """Action buttons shown before the quantity modal."""

    def __init__(self, symbol: str, owner_id: int) -> None:
        """Initializes action controls for one symbol."""
        super().__init__(owner_id=owner_id)
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
            modal=StockQuantityModal(
                symbol=self.symbol,
                action=StockAction.BUY,
                message=interaction.message,
                parent=self,
                owner_id=self.owner_id,
            )
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
            modal=StockQuantityModal(
                symbol=self.symbol,
                action=StockAction.SHORT,
                message=interaction.message,
                parent=self,
                owner_id=self.owner_id,
            )
        )

    @nextcord.ui.button(
        label="返回明細",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="stock:action:back",
        row=1,
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the public stock detail view."""
        self.stop()
        await send_stock_detail(
            interaction=interaction, symbol=self.symbol, owner_id=self.owner_id
        )


class StockPostTradeView(StockPublicView):
    """Refresh control shown after a submitted stock trade."""

    def __init__(self, symbol: str, owner_id: int) -> None:
        """Initializes post-trade controls."""
        super().__init__(owner_id=owner_id)
        self.symbol = symbol

    @nextcord.ui.button(
        label="重新整理明細",
        emoji="🔄",
        style=ButtonStyle.secondary,
        custom_id="stock:refresh",
        row=0,
    )
    async def refresh(self, _button: Button, interaction: Interaction) -> None:
        """Edits the public message into a fresh detail view."""
        self.stop()
        await send_stock_detail(
            interaction=interaction, symbol=self.symbol, owner_id=self.owner_id
        )

    @nextcord.ui.button(
        label="返回列表",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="stock:post:back",
        row=0,
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the public market list."""
        quotes = await list_market_quotes()
        self.stop()
        await edit_stock_message(
            interaction=interaction,
            embed=build_market_embed(quotes=quotes),
            view=StockMarketView(quotes=quotes, owner_id=self.owner_id),
        )


class StockQuantityModal(Modal):
    """Quantity modal for stock operations."""

    def __init__(
        self,
        symbol: str,
        action: StockAction,
        owner_id: int,
        message: Message | None = None,
        parent: StockPublicView | None = None,
    ) -> None:
        """Initializes the modal with one TextInput."""
        super().__init__(title=f"{symbol} 股票操作")
        self.symbol = symbol
        self.action = action
        self.message = message
        self.parent = parent
        self.owner_id = owner_id
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
        user = require_stock_user(interaction=interaction)
        if self.owner_id != user.id:
            await send_ephemeral_notice(
                interaction=interaction,
                content="這個股票面板只有發起者可以操作，請自己使用 `/stock` 開一個新的面板",
                log_message="Failed to send stock modal owner mismatch notice",
            )
            return
        await interaction.response.defer()
        avatar_url = await guild_avatar_url(user=user, guild=getattr(interaction, "guild", None))
        result = await settle_stock_operation(
            symbol=self.symbol,
            user_id=user.id,
            user_name=user.name,
            avatar_url=avatar_url,
            requested_action=self.action,
            quantity=raw_quantity,
        )
        if self.parent is not None:
            self.parent.stop()
        view: StockPublicView = (
            StockPostTradeView(symbol=self.symbol, owner_id=self.owner_id)
            if result.success
            else StockActionView(symbol=self.symbol, owner_id=self.owner_id)
        )
        await edit_stock_message(
            interaction=interaction,
            embed=build_settlement_embed(result=result),
            view=view,
            message=self.message,
        )


async def edit_stock_detail(interaction: Interaction, symbol: str, owner_id: int) -> None:
    """Edits the public stock message into a detail view for one interaction."""
    user = require_stock_user(interaction=interaction)
    if not interaction.response.is_done():
        await interaction.response.defer()
    try:
        detail = await get_stock_detail(symbol=symbol, user_id=user.id, user_name=user.name)
    except ValueError:
        await edit_stock_message(
            interaction=interaction,
            embed=build_error_embed(message=f"找不到股票 `{symbol}`"),
            view=None,
        )
        return
    filename = f"{symbol.lower()}_7d.png"
    chart_bytes = build_price_chart(ticks=detail.ticks)
    await edit_stock_message(
        interaction=interaction,
        embed=build_stock_detail_embed(detail=detail, chart_filename=filename),
        file=File(fp=BytesIO(chart_bytes), filename=filename),
        view=StockDetailView(symbol=symbol, owner_id=owner_id),
    )


async def edit_stock_message(
    interaction: Interaction,
    embed: nextcord.Embed,
    view: StockPublicView | None,
    file: File | None = None,
    message: Message | None = None,
) -> None:
    """Edits the original public stock message for a component or modal interaction."""
    target_message = message or interaction.message
    if view is not None:
        view.bind_message(message=target_message)
    kwargs: dict[str, object] = {"embed": embed, "view": view, "attachments": []}
    if file is not None:
        kwargs["file"] = file
    if not interaction.response.is_done():
        edited = await interaction.response.edit_message(**kwargs)
        if isinstance(edited, Message) and view is not None:
            view.bind_message(message=edited)
        return
    if target_message is not None:
        await target_message.edit(**kwargs)
        return
    await interaction.followup.send(embed=embed, view=view, file=file, wait=True)


send_stock_detail = edit_stock_detail


__all__ = [
    "StockActionView",
    "StockDetailView",
    "StockMarketView",
    "StockNewsControlsView",
    "StockPostTradeView",
    "StockPublicView",
    "StockQuantityModal",
    "StockTutorialView",
    "edit_stock_detail",
    "edit_stock_message",
    "require_stock_user",
    "send_stock_detail",
]
