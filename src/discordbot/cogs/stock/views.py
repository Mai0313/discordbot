"""Every state the one public `/stock` message can be in, and the transitions between them.

`cog.py` posts that message once and binds it to a `StockMarketView`; from there nothing in this
file sends a second message. A control replaces the panel's view and repaints the same message
through `edit_owned_public_message`, so the states form a loop the opener walks: the paged market
list, the tutorial, one stock's detail, that stock's news, the action dropdown, and the
post-trade result, each with its own way back.

Two rules hold across every transition. A transition builds a fresh view rather than mutating
the live one, and it calls `self.stop()` on the outgoing view first: `stop()` cancels nextcord's
idle timer, which `OwnedPublicView.on_timeout` would otherwise use to delete the very message the
replacement has just taken over. `StockActionView.action_select` is the deliberate exception —
it opens a modal instead of repainting, so it stays live while the modal is on screen (a
dismissed modal leaves the dropdown usable) and is stopped by `submit_stock_quantity` once a
submission actually lands.

The half `/games fishing` shares — the owner gate, the idle deletion, and the decision of which
Discord call repaints the message — lives in `utils/owned_message_views.py`. What stays here is
the stock-specific chrome plus `edit_stock_detail` and `edit_stock_action_prompt`, the two async
helpers every state routes through, so a symbol deleted offline between two presses degrades to
the market list with an error embed in one place instead of six.

Nothing here computes a price or draws anything: numbers come from `services/stock/database.py`,
pixels and embeds from `presentation.py` and `chart.py`, and `submit_stock_quantity` is the only
path in this file that writes. That function is also where the owner gate has to be repeated by
hand, because a modal submit is its own interaction and never runs the view's
`interaction_check`.
"""

from io import BytesIO
from typing import cast

import nextcord
from nextcord import File, User, Embed, Member, Message, ButtonStyle, Interaction, SelectOption
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from nextcord.ui import View, Modal, Button, TextInput, StringSelect
from nextcord.ext import commands

from discordbot.typings.stock import STOCK_ACTION_TIMEOUT_SECONDS, StockAction, StockMarketQuote
from discordbot.utils.avatars import guild_avatar_url
from discordbot.cogs.stock.chart import build_price_chart
from discordbot.cogs.stock.presentation import (
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
from discordbot.services.stock.database import (
    get_stock_news,
    get_stock_detail,
    list_market_quotes,
    settle_stock_operation,
)
from discordbot.utils.owned_message_views import (
    OwnedPublicView,
    send_ephemeral_notice,
    edit_owned_public_message,
)

MARKET_PAGE_SIZE = 25
SELECT_OPTION_LABEL_LIMIT = 100


def require_stock_user(interaction: Interaction[commands.Bot]) -> User | Member:
    """Returns who triggered the interaction, refusing to continue when Discord named nobody.

    nextcord types `Interaction.user` as optional, and every panel here is owned by exactly one
    id, so a missing one is raised on rather than defaulted: a panel built without an owner would
    be operable by the whole channel. Called before the market is read and before any settlement,
    so the failure costs nothing that was already written.

    Args:
        interaction (Interaction[commands.Bot]): The interaction to read identity from.

    Returns:
        The user or guild member the interaction came from.

    Raises:
        RuntimeError: The interaction carries no Discord user identity.
    """
    if interaction.user is None:
        raise RuntimeError("Stock interaction is missing Discord user identity")
    return interaction.user


def _select_option_label(symbol: str, name: str) -> str:
    """Joins a symbol and a company name into one select option label, truncating to fit.

    Company names are operator-authored with no length bound while Discord caps an option label
    at `SELECT_OPTION_LABEL_LIMIT` characters, so an over-long one is cut with an ellipsis rather
    than costing the whole market list. The symbol leads, so truncation never eats the part the
    press is keyed on.

    Args:
        symbol (str): Ticker symbol of the virtual company.
        name (str): Display name of the virtual company.

    Returns:
        A label no longer than `SELECT_OPTION_LABEL_LIMIT` characters.
    """
    label = f"{symbol} · {name}"
    if len(label) <= SELECT_OPTION_LABEL_LIMIT:
        return label
    return f"{label[: SELECT_OPTION_LABEL_LIMIT - 3]}..."


def build_market_message_payload(
    quotes: tuple[StockMarketQuote, ...], page_index: int = 0
) -> tuple[Embed, File]:
    """Builds the embed and the board attachment one market page is shown as.

    The rows are pixels rather than embed text, so the embed only points at the PNG through the
    filename both halves are built from and the pair must always be sent together. The `File`
    wraps a buffer of its own each call, since nextcord reads a `File` to the end on the request
    that sends it.

    Args:
        quotes (tuple[StockMarketQuote, ...]): The whole board, not just this page; the embed and
            the render each slice their own page out.
        page_index (int): Zero-based page to show, clamped into range by both halves.

    Returns:
        The market embed and the board PNG it references with `attachment://`.
    """
    filename = market_board_filename(page_index=page_index)
    embed = build_market_embed(
        quotes=quotes, page_index=page_index, page_size=MARKET_PAGE_SIZE, board_filename=filename
    )
    board = build_market_board_image(
        quotes=quotes, page_index=page_index, page_size=MARKET_PAGE_SIZE
    )
    return embed, File(fp=BytesIO(board), filename=filename)


class StockPublicView(OwnedPublicView):
    """Base view for every state of the public stock message.

    Exists only to pin the two things the whole flow must agree on — the idle timeout and the
    owner-mismatch wording — so a new state cannot quietly ship a different refusal line or
    outlive its siblings.
    """

    def __init__(self, owner_id: int, delete_on_timeout: bool = True) -> None:
        """Initializes stock controls with the shared idle timeout and refusal notice.

        Args:
            owner_id (int): Discord id of the user allowed to operate this panel.
            delete_on_timeout (bool): Whether going idle deletes the panel's message.
        """
        super().__init__(
            owner_id=owner_id,
            timeout_seconds=STOCK_ACTION_TIMEOUT_SECONDS,
            owner_mismatch_notice="這個股票面板只有發起者可以操作，請自己使用 `/stock` 開一個新的面板",
            delete_on_timeout=delete_on_timeout,
        )


class _StockQuantitySubmission(BaseModel):
    """One submitted quantity plus everything `submit_stock_quantity` needs to answer it.

    A modal submit arrives on its own interaction, detached from the panel it was opened from,
    so the owner id, the message to edit and the view to stop all have to be carried across
    rather than read back off the interaction.
    """

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
    """The market list: a page of stocks to pick from, its paging buttons and the tutorial.

    The panel's entry state, and where the tutorial, the detail and the post-trade result hand
    back to.
    """

    def __init__(
        self, quotes: tuple[StockMarketQuote, ...], owner_id: int, page_index: int = 0
    ) -> None:
        """Fills the select with one page of quotes and settles the two paging buttons.

        `View.__init__` rebinds each decorated callback's name to the component it built, so
        `self.stock_select` and the page buttons are `Item`s by the time they are cast here and
        mutating them is what replaces the decorator's placeholder options. A page is
        `MARKET_PAGE_SIZE` rows because that is Discord's ceiling on one select's options, so
        paging is what makes a market larger than that reachable at all.

        An empty market still ships one inert option: a select needs one, and `stock_select`
        refuses its value rather than asking the service for a stock named "none".

        Args:
            quotes (tuple[StockMarketQuote, ...]): The whole board; this view slices its own page.
            owner_id (int): Discord id of the user allowed to operate this panel.
            page_index (int): Zero-based page to show, clamped to a real page.
        """
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
        """Repaints the panel as the selected stock's detail view.

        Neither placeholder value is a symbol — "loading" is the decorator's own option and
        "none" is what an empty market ships — so both repaint the market list with an error
        instead of asking the service for a stock that cannot exist.

        Args:
            select (StringSelect["StockMarketView"]): The select carrying the chosen value.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
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
        """Moves the market list one page back.

        Args:
            _button (Button["StockMarketView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
        await self._show_page(interaction=interaction, page_index=self.page_index - 1)

    @nextcord.ui.button(
        label="下一頁", emoji="▶️", style=ButtonStyle.secondary, custom_id="stock:page:next", row=1
    )
    async def next_page(
        self, _button: Button["StockMarketView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Moves the market list one page forward.

        Args:
            _button (Button["StockMarketView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
        await self._show_page(interaction=interaction, page_index=self.page_index + 1)

    @nextcord.ui.button(
        label="教學", emoji="📘", style=ButtonStyle.secondary, custom_id="stock:tutorial", row=2
    )
    async def tutorial(
        self, _button: Button["StockMarketView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Repaints the panel as the tutorial.

        Args:
            _button (Button["StockMarketView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
        self.stop()
        await edit_owned_public_message(
            interaction=interaction,
            embed=build_tutorial_embed(),
            view=StockTutorialView(owner_id=self.owner_id),
        )

    async def _show_page(self, interaction: Interaction[commands.Bot], page_index: int) -> None:
        """Repaints the market list at another page, re-rendering its board.

        Paging replaces the view rather than editing this one's options, so the buttons of the
        new page are disabled against its own bounds. The index is clamped here as well as in the
        replacement, since the board PNG is built from it before that view exists. The rows are
        the snapshot this view was built from, so turning a page advances no tick and re-reads
        nothing.

        Args:
            interaction (Interaction[commands.Bot]): The interaction to answer.
            page_index (int): Requested page, clamped to a real one.
        """
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
    """The tutorial state: static help text with nothing but a way back."""

    def __init__(self, owner_id: int) -> None:
        """Initializes tutorial controls for the owning user.

        Args:
            owner_id (int): Discord id of the user allowed to operate this panel.
        """
        super().__init__(owner_id=owner_id)

    @nextcord.ui.button(
        label="返回列表", emoji="↩️", style=ButtonStyle.secondary, custom_id="stock:tutorial:back"
    )
    async def back(
        self, _button: Button["StockTutorialView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Repaints the panel as the market list, on quotes read fresh.

        The list is re-read rather than remembered, so returning from the tutorial advances the
        market and shows current prices.

        Args:
            _button (Button["StockTutorialView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
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
    """One stock's detail state: its quote, the viewer's position, and where to go next.

    The embed and the 7D chart it sits under are painted by `edit_stock_detail`; this view is
    only the three controls beneath them.
    """

    def __init__(self, symbol: str, owner_id: int) -> None:
        """Initializes detail controls for one symbol.

        Args:
            symbol (str): Ticker symbol this panel is showing.
            owner_id (int): Discord id of the user allowed to operate this panel.
        """
        super().__init__(owner_id=owner_id)
        self.symbol = symbol

    @nextcord.ui.button(
        label="操作股票", emoji="🧾", style=ButtonStyle.primary, custom_id="stock:operate", row=0
    )
    async def operate(
        self, _button: Button["StockDetailView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Repaints the panel as the action dropdown.

        Direction is picked here rather than inside the modal, which keeps the modal down to one
        input and lets the dropdown sit next to a freshly read price and position.

        Args:
            _button (Button["StockDetailView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
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
        """Repaints the panel as this stock's recent headlines.

        Reading the news refreshes it when it is due but advances no tick, so a headline minted
        here only reaches the price on the next market advance.

        Args:
            _button (Button["StockDetailView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
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
        """Repaints the panel as the market list, on quotes read fresh.

        Args:
            _button (Button["StockDetailView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
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
    """The news state: headlines for one stock, with the way back to its detail."""

    def __init__(self, symbol: str, owner_id: int) -> None:
        """Initializes news controls for one symbol.

        Args:
            symbol (str): Ticker symbol whose headlines are showing.
            owner_id (int): Discord id of the user allowed to operate this panel.
        """
        super().__init__(owner_id=owner_id)
        self.symbol = symbol

    @nextcord.ui.button(
        label="返回明細", emoji="↩️", style=ButtonStyle.secondary, custom_id="stock:news:back"
    )
    async def back(
        self, _button: Button["StockNewsControlsView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Repaints the panel as this stock's detail.

        Args:
            _button (Button["StockNewsControlsView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
        self.stop()
        await edit_stock_detail(
            interaction=interaction, symbol=self.symbol, owner_id=self.owner_id
        )


class StockActionView(StockPublicView):
    """The action state: pick a direction, then a quantity in the modal it opens.

    Also the state a failed settlement lands back on, so a rejected quantity can be retried
    without walking in from the detail view again.
    """

    def __init__(self, symbol: str, owner_id: int) -> None:
        """Initializes action controls for one symbol.

        Args:
            symbol (str): Ticker symbol the operation will act on.
            owner_id (int): Discord id of the user allowed to operate this panel.
        """
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
        """Opens the quantity modal for the chosen direction.

        The only transition that does not stop this view or repaint the message: a dismissed
        modal has to leave a working dropdown behind. The modal carries the panel's message and
        this view instead, and `submit_stock_quantity` stops it once a submission lands.

        Args:
            select (StringSelect["StockActionView"]): The select carrying the chosen action.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
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
        """Repaints the panel as this stock's detail.

        Args:
            _button (Button["StockActionView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
        self.stop()
        await edit_stock_detail(
            interaction=interaction, symbol=self.symbol, owner_id=self.owner_id
        )


class StockPostTradeView(StockPublicView):
    """The post-trade state: the settlement receipt, with a way back to the detail or the list.

    Only a successful settlement lands here; a failed one goes back to `StockActionView` so the
    quantity can be corrected.
    """

    def __init__(self, symbol: str, owner_id: int) -> None:
        """Initializes post-trade controls.

        Args:
            symbol (str): Ticker symbol that was just traded.
            owner_id (int): Discord id of the user allowed to operate this panel.
        """
        super().__init__(owner_id=owner_id)
        self.symbol = symbol

    @nextcord.ui.button(
        label="重新整理明細",
        emoji="🔄",
        style=ButtonStyle.secondary,
        custom_id="stock:refresh",
        row=0,
    )
    async def refresh_detail(
        self, _button: Button["StockPostTradeView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Repaints the panel as this stock's detail, re-read after the trade.

        Not named `refresh`: nextcord's `View.__init__` rebinds every button callback onto its
        own name, so that spelling would shadow `View.refresh(components)` and crash the
        gateway's `MESSAGE_UPDATE` handler.

        Args:
            _button (Button["StockPostTradeView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
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
        """Repaints the panel as the market list, on quotes read fresh.

        Args:
            _button (Button["StockPostTradeView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The interaction to answer.
        """
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
    """How many shares, for a symbol and direction already chosen.

    A `Modal` is not a `View`, so it inherits no owner gate: its submit arrives on a fresh
    interaction that never runs `interaction_check`. `owner_id` is carried here for
    `submit_stock_quantity` to weigh the submitter against, and the panel's message is carried
    because a modal-submit interaction is not attached to the message the modal came from.

    The text is not parsed here. It goes to the settlement service as typed, so "ALL", a
    thousands separator and plain nonsense are all judged in one place, under the lock.
    """

    def __init__(
        self,
        symbol: str,
        owner_id: int,
        action: StockAction,
        message: Message | None = None,
        parent: StockPublicView | None = None,
    ) -> None:
        """Initializes the modal with one quantity input.

        Args:
            symbol (str): Ticker symbol the operation will act on.
            owner_id (int): Discord id of the user allowed to submit this modal.
            action (StockAction): Direction already chosen in the dropdown.
            message (Message | None): The panel message to repaint with the result.
            parent (StockPublicView | None): The view to stop once a submission lands.
        """
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
        """Hands what was typed to the shared submit path, unparsed.

        An unfilled input reads as None, which becomes the empty string here and comes back from
        the service as an ordinary format error rather than a crash.

        Args:
            interaction (Interaction[commands.Bot]): The modal-submit interaction to answer.
        """
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
        """Submits a quantity that did not come from this modal's own input.

        The same path as `callback`, minus the rendered modal, and able to override the direction
        the modal was built with. Nothing in `src/` calls it today; the tests drive settlement
        through it.

        Args:
            interaction (Interaction[commands.Bot]): The interaction to answer.
            raw_quantity (str): Quantity text, passed to the service unparsed.
            action (StockAction | None): Direction to submit instead of the modal's own, or None
                to keep it.
        """
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
    """Repaints the panel as one stock's detail, with its 7D chart.

    Every route into the detail state goes through here. An unanswered interaction is deferred
    first, because the read behind this advances the market and can outrun Discord's three-second
    response window; the repaint then takes `edit_owned_public_message`'s message-edit path
    rather than answering the press in one request.

    Profiles are maintained offline, so a symbol can disappear between the list being drawn and a
    press landing on it. That is the `ValueError`, and it degrades to the market list carrying an
    error embed rather than surfacing as a failed interaction.

    Args:
        interaction (Interaction[commands.Bot]): The interaction to answer.
        symbol (str): Ticker symbol to show.
        owner_id (int): Discord id of the user allowed to operate the replacement panel.
    """
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
    """Repaints the panel as the action dropdown, over a freshly read price and position.

    The same defer and the same vanished-symbol fallback as `edit_stock_detail`; what differs is
    that the numbers shown here are the ones the next submission will be judged against, so they
    are re-read rather than carried over from the detail view the press came from. They are still
    only a display: settlement re-reads everything under the lock.

    Args:
        interaction (Interaction[commands.Bot]): The interaction to answer.
        symbol (str): Ticker symbol the operation will act on.
        owner_id (int): Discord id of the user allowed to operate the replacement panel.
    """
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
    """Settles one submitted quantity and repaints the panel with the outcome.

    The owner gate is re-applied here by hand, and this is the one place it has to be: a modal
    submit is its own interaction, so nothing ran the panel's `interaction_check` on the way in.
    Without it a stranger holding a copied or stale modal would trade on their own account and
    repaint someone else's panel with the receipt. The refusal is ephemeral and returns before
    anything is read or written.

    Nothing here validates the quantity. `settle_stock_operation` owns both the parse and the
    money, and hands back a bad quantity, an unaffordable one and an exhausted float alike as a
    failed result — so failure means the action dropdown comes back for a retry, and only success
    reaches the post-trade view. The repaint names the carried message explicitly, because a
    modal-submit interaction has none of its own to fall back on.

    Args:
        submission (_StockQuantitySubmission): The submitted quantity plus the panel context to
            answer it in.
    """
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
