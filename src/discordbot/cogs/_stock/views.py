"""Interactive views for the simulated stock market."""

from io import BytesIO
from typing import cast

import nextcord
from nextcord import File, User, Embed, Member, Message, ButtonStyle, Interaction, SelectOption
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from nextcord.ui import View, Modal, Button, TextInput, StringSelect
from nextcord.ext import commands

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
    market_board_filename,
    build_settlement_embed,
    build_market_board_image,
    build_stock_detail_embed,
    build_action_prompt_embed,
)
from discordbot.utils.owned_message_views import (
    OwnedPublicView,
    send_ephemeral_notice,
    edit_owned_public_message,
)

MARKET_PAGE_SIZE = 25
SELECT_OPTION_LABEL_LIMIT = 100


def require_stock_user(interaction: Interaction[commands.Bot]) -> User | Member:
    """Returns the interaction user or fails before any stock state can be written."""
    if interaction.user is None:
        raise RuntimeError("Stock interaction is missing Discord user identity")
    return interaction.user


def _select_option_label(symbol: str, name: str) -> str:
    """Returns a stock select label that fits Discord's option limit."""
    label = f"{symbol} · {name}"
    if len(label) <= SELECT_OPTION_LABEL_LIMIT:
        return label
    return f"{label[: SELECT_OPTION_LABEL_LIMIT - 3]}..."


def build_market_message_payload(
    quotes: tuple[StockMarketQuote, ...], page_index: int = 0
) -> tuple[Embed, File]:
    """Builds the market embed and board attachment for one page."""
    filename = market_board_filename(page_index=page_index)
    embed = build_market_embed(
        quotes=quotes, page_index=page_index, page_size=MARKET_PAGE_SIZE, board_filename=filename
    )
    board = build_market_board_image(
        quotes=quotes, page_index=page_index, page_size=MARKET_PAGE_SIZE
    )
    return embed, File(fp=BytesIO(board), filename=filename)


class StockPublicView(OwnedPublicView):
    """Base view for stock states that own one Discord message."""

    def __init__(self, owner_id: int, delete_on_timeout: bool = True) -> None:
        """Initializes stock controls with an idle timeout."""
        super().__init__(
            owner_id=owner_id,
            timeout_seconds=STOCK_ACTION_TIMEOUT_SECONDS,
            owner_mismatch_notice="這個股票面板只有發起者可以操作，請自己使用 `/stock` 開一個新的面板",
            delete_on_timeout=delete_on_timeout,
        )


class _StockQuantitySubmission(BaseModel):
    """Context shared by quantity modal submit paths."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    interaction: SkipValidation[Interaction[commands.Bot]] = Field(
        ..., description="Discord interaction that submitted the quantity."
    )
    symbol: str = Field(..., description="Stock symbol being operated on.")
    action: StockAction = Field(..., description="Requested buy/cover or short/sell action.")
    owner_id: int = Field(..., description="Discord user id allowed to operate this panel.")
    raw_quantity: str = Field(..., description="Raw quantity text from the modal or preset.")
    message: SkipValidation[Message | None] = Field(
        default=None, description="Public stock message to edit in place, if any."
    )
    parent: SkipValidation[StockPublicView | None] = Field(
        default=None, description="Originating view to stop after submission, if any."
    )


class StockMarketView(StockPublicView):
    """Stock select and tutorial controls for the market list."""

    def __init__(
        self, quotes: tuple[StockMarketQuote, ...], owner_id: int, page_index: int = 0
    ) -> None:
        """Initializes market controls from quote rows."""
        super().__init__(owner_id=owner_id)
        self.quotes = quotes
        self.page_count = max((len(quotes) + MARKET_PAGE_SIZE - 1) // MARKET_PAGE_SIZE, 1)
        self.page_index = min(max(page_index, 0), self.page_count - 1)
        page_quotes = quotes[
            self.page_index * MARKET_PAGE_SIZE : (self.page_index + 1) * MARKET_PAGE_SIZE
        ]
        self._select = cast("StringSelect[StockMarketView]", self.stock_select)
        self._select.options = [
            SelectOption(
                label=_select_option_label(symbol=quote.profile.symbol, name=quote.profile.name),
                value=quote.profile.symbol,
                description=f"{quote.profile.category}",
            )
            for quote in page_quotes
        ] or [SelectOption(label="目前沒有股票", value="none", description="請稍後再試")]
        self._previous_page = cast("Button[StockMarketView]", self.previous_page)
        self._next_page = cast("Button[StockMarketView]", self.next_page)
        self._previous_page.disabled = self.page_index <= 0
        self._next_page.disabled = self.page_index >= self.page_count - 1

    @nextcord.ui.string_select(
        placeholder="選擇股票",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中", value="loading")],
        custom_id="stock:select",
        row=0,
    )
    async def stock_select(
        self, select: StringSelect["StockMarketView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Shows a public detail view for the selected stock."""
        symbol = select.values[0]
        if symbol in {"loading", "none"}:
            self.stop()
            await edit_owned_public_message(
                interaction=interaction,
                embed=build_error_embed(message="目前沒有可用的股票"),
                view=StockMarketView(quotes=self.quotes, owner_id=self.owner_id),
            )
            return
        self.stop()
        await edit_stock_detail(interaction=interaction, symbol=symbol, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="上一頁", emoji="◀️", style=ButtonStyle.secondary, custom_id="stock:page:prev", row=1
    )
    async def previous_page(
        self, _button: Button["StockMarketView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Moves the market list to the previous page."""
        await self._show_page(interaction=interaction, page_index=self.page_index - 1)

    @nextcord.ui.button(
        label="下一頁", emoji="▶️", style=ButtonStyle.secondary, custom_id="stock:page:next", row=1
    )
    async def next_page(
        self, _button: Button["StockMarketView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Moves the market list to the next page."""
        await self._show_page(interaction=interaction, page_index=self.page_index + 1)

    @nextcord.ui.button(
        label="教學", emoji="📘", style=ButtonStyle.secondary, custom_id="stock:tutorial", row=2
    )
    async def tutorial(
        self, _button: Button["StockMarketView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Shows the stock tutorial in the public stock message."""
        self.stop()
        await edit_owned_public_message(
            interaction=interaction,
            embed=build_tutorial_embed(),
            view=StockTutorialView(owner_id=self.owner_id),
        )

    async def _show_page(self, interaction: Interaction[commands.Bot], page_index: int) -> None:
        """Edits the market list to a bounded page index."""
        self.stop()
        normalized_page = min(max(page_index, 0), self.page_count - 1)
        embed, file = build_market_message_payload(quotes=self.quotes, page_index=normalized_page)
        await edit_owned_public_message(
            interaction=interaction,
            embed=embed,
            file=file,
            view=StockMarketView(
                quotes=self.quotes, owner_id=self.owner_id, page_index=normalized_page
            ),
        )


class StockTutorialView(StockPublicView):
    """Tutorial controls for stock messages."""

    def __init__(self, owner_id: int) -> None:
        """Initializes tutorial controls for the owning user."""
        super().__init__(owner_id=owner_id)

    @nextcord.ui.button(
        label="返回列表", emoji="↩️", style=ButtonStyle.secondary, custom_id="stock:tutorial:back"
    )
    async def back(
        self, _button: Button["StockTutorialView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Returns to the market list."""
        quotes = await list_market_quotes()
        embed, file = build_market_message_payload(quotes=quotes)
        self.stop()
        await edit_owned_public_message(
            interaction=interaction,
            embed=embed,
            file=file,
            view=StockMarketView(quotes=quotes, owner_id=self.owner_id),
        )


class StockDetailView(StockPublicView):
    """Public detail controls for one stock."""

    def __init__(self, symbol: str, owner_id: int) -> None:
        """Initializes detail controls for one symbol."""
        super().__init__(owner_id=owner_id)
        self.symbol = symbol

    @nextcord.ui.button(
        label="操作股票", emoji="🧾", style=ButtonStyle.primary, custom_id="stock:operate", row=0
    )
    async def operate(
        self, _button: Button["StockDetailView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Shows action selection before opening the quantity modal."""
        self.stop()
        await edit_stock_action_prompt(
            interaction=interaction, symbol=self.symbol, owner_id=self.owner_id
        )

    @nextcord.ui.button(
        label="近期新聞", emoji="📰", style=ButtonStyle.secondary, custom_id="stock:news", row=0
    )
    async def news(
        self, _button: Button["StockDetailView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Shows recent deterministic news in the public stock message."""
        news = await get_stock_news(symbol=self.symbol)
        self.stop()
        await edit_owned_public_message(
            interaction=interaction,
            embed=build_news_embed(news=news, symbol=self.symbol),
            view=StockNewsControlsView(symbol=self.symbol, owner_id=self.owner_id),
        )

    @nextcord.ui.button(
        label="返回列表", emoji="↩️", style=ButtonStyle.secondary, custom_id="stock:back", row=1
    )
    async def back(
        self, _button: Button["StockDetailView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Returns to the public market list."""
        quotes = await list_market_quotes()
        embed, file = build_market_message_payload(quotes=quotes)
        self.stop()
        await edit_owned_public_message(
            interaction=interaction,
            embed=embed,
            file=file,
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
    async def back(
        self, _button: Button["StockNewsControlsView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Returns to the public stock detail view."""
        self.stop()
        await edit_stock_detail(
            interaction=interaction, symbol=self.symbol, owner_id=self.owner_id
        )


class StockActionView(StockPublicView):
    """Action dropdown shown before the quantity modal."""

    def __init__(self, symbol: str, owner_id: int) -> None:
        """Initializes action controls for one symbol."""
        super().__init__(owner_id=owner_id)
        self.symbol = symbol

    @nextcord.ui.string_select(
        placeholder="選擇操作",
        min_values=1,
        max_values=1,
        options=[
            SelectOption(
                label="買入",
                value=StockAction.BUY.value,
                description="買入股票，若已有做空會優先回補",
            ),
            SelectOption(
                label="放空",
                value=StockAction.SHORT.value,
                description="放空股票，若已有持股會優先賣出",
            ),
        ],
        custom_id="stock:action",
        row=0,
    )
    async def action_select(
        self, select: StringSelect["StockActionView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Opens the quantity modal for the selected operation."""
        await interaction.response.send_modal(
            modal=StockQuantityModal(
                symbol=self.symbol,
                action=StockAction(select.values[0]),
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
    async def back(
        self, _button: Button["StockActionView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Returns to the public stock detail view."""
        self.stop()
        await edit_stock_detail(
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
    async def refresh(
        self, _button: Button["StockPostTradeView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Edits the public message into a fresh detail view."""
        self.stop()
        await edit_stock_detail(
            interaction=interaction, symbol=self.symbol, owner_id=self.owner_id
        )

    @nextcord.ui.button(
        label="返回列表",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="stock:post:back",
        row=0,
    )
    async def back(
        self, _button: Button["StockPostTradeView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Returns to the public market list."""
        quotes = await list_market_quotes()
        embed, file = build_market_message_payload(quotes=quotes)
        self.stop()
        await edit_owned_public_message(
            interaction=interaction,
            embed=embed,
            file=file,
            view=StockMarketView(quotes=quotes, owner_id=self.owner_id),
        )


class StockQuantityModal(Modal):
    """Quantity modal for stock operations."""

    def __init__(
        self,
        symbol: str,
        owner_id: int,
        action: StockAction,
        message: Message | None = None,
        parent: StockPublicView | None = None,
    ) -> None:
        """Initializes the modal with one quantity input."""
        super().__init__(title=f"股票操作：{symbol}")
        self.symbol = symbol
        self.action = action
        self.message = message
        self.parent = parent
        self.owner_id = owner_id
        self.quantity: TextInput[View] = TextInput(
            label="數量",
            placeholder="請輸入股數，或輸入 ALL",
            min_length=1,
            max_length=16,
            required=True,
            row=0,
        )
        self.add_item(item=self.quantity)

    async def callback(self, interaction: Interaction[commands.Bot]) -> None:
        """Submits the quantity to the stock settlement service."""
        await submit_stock_quantity(
            submission=_StockQuantitySubmission(
                interaction=interaction,
                symbol=self.symbol,
                action=self.action,
                owner_id=self.owner_id,
                raw_quantity=str(self.quantity.value or ""),
                message=self.message,
                parent=self.parent,
            )
        )

    async def submit_quantity(
        self,
        interaction: Interaction[commands.Bot],
        raw_quantity: str,
        action: StockAction | None = None,
    ) -> None:
        """Submits a raw quantity string to the stock settlement service."""
        await submit_stock_quantity(
            submission=_StockQuantitySubmission(
                interaction=interaction,
                symbol=self.symbol,
                action=action or self.action,
                owner_id=self.owner_id,
                raw_quantity=raw_quantity,
                message=self.message,
                parent=self.parent,
            )
        )


async def edit_stock_detail(
    interaction: Interaction[commands.Bot], symbol: str, owner_id: int
) -> None:
    """Shows or edits a public stock detail view for one interaction."""
    user = require_stock_user(interaction=interaction)
    if not interaction.response.is_done():
        await interaction.response.defer()
    try:
        detail = await get_stock_detail(symbol=symbol, user_id=user.id, user_name=user.name)
    except ValueError:
        quotes = await list_market_quotes()
        await edit_owned_public_message(
            interaction=interaction,
            embed=build_error_embed(message=f"找不到股票 `{symbol}`"),
            view=StockMarketView(quotes=quotes, owner_id=owner_id),
        )
        return
    filename = f"{symbol.lower()}_7d.png"
    chart_bytes = build_price_chart(ticks=detail.ticks)
    view = StockDetailView(symbol=symbol, owner_id=owner_id)
    await edit_owned_public_message(
        interaction=interaction,
        embed=build_stock_detail_embed(detail=detail, chart_filename=filename),
        file=File(fp=BytesIO(chart_bytes), filename=filename),
        view=view,
    )


async def edit_stock_action_prompt(
    interaction: Interaction[commands.Bot], symbol: str, owner_id: int
) -> None:
    """Shows the operation dropdown with fresh stock and position context."""
    user = require_stock_user(interaction=interaction)
    if not interaction.response.is_done():
        await interaction.response.defer()
    try:
        detail = await get_stock_detail(symbol=symbol, user_id=user.id, user_name=user.name)
    except ValueError:
        quotes = await list_market_quotes()
        await edit_owned_public_message(
            interaction=interaction,
            embed=build_error_embed(message=f"找不到股票 `{symbol}`"),
            view=StockMarketView(quotes=quotes, owner_id=owner_id),
        )
        return
    await edit_owned_public_message(
        interaction=interaction,
        embed=build_action_prompt_embed(detail=detail),
        view=StockActionView(symbol=symbol, owner_id=owner_id),
    )


async def submit_stock_quantity(submission: _StockQuantitySubmission) -> None:
    """Submits a stock quantity from either a dropdown preset or modal."""
    interaction = submission.interaction
    user = require_stock_user(interaction=interaction)
    if submission.owner_id != user.id:
        await send_ephemeral_notice(
            interaction=interaction,
            content="這個股票面板只有發起者可以操作，請自己使用 `/stock` 開一個新的面板",
            log_message="Failed to send stock modal owner mismatch notice",
        )
        return
    await interaction.response.defer()
    avatar_url = await guild_avatar_url(user=user, guild=getattr(interaction, "guild", None))
    result = await settle_stock_operation(
        symbol=submission.symbol,
        user_id=user.id,
        user_name=user.name,
        avatar_url=avatar_url,
        requested_action=submission.action,
        quantity=submission.raw_quantity,
    )
    if submission.parent is not None:
        submission.parent.stop()
    view: StockPublicView = (
        StockPostTradeView(symbol=submission.symbol, owner_id=submission.owner_id)
        if result.success
        else StockActionView(symbol=submission.symbol, owner_id=submission.owner_id)
    )
    await edit_owned_public_message(
        interaction=interaction,
        embed=build_settlement_embed(result=result),
        view=view,
        message=submission.message,
    )


__all__ = [
    "StockActionView",
    "StockDetailView",
    "StockMarketView",
    "StockNewsControlsView",
    "StockPostTradeView",
    "StockPublicView",
    "StockQuantityModal",
    "StockTutorialView",
    "build_market_message_payload",
    "edit_stock_action_prompt",
    "edit_stock_detail",
    "require_stock_user",
    "submit_stock_quantity",
]
