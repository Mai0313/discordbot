"""Tests for the stock cog and interactive views."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from datetime import datetime

from nextcord import Embed, Locale

from discordbot.cogs import stock
from discordbot.cogs.stock import StockCogs
from discordbot.cogs._stock import views as stock_views
from discordbot.typings.stock import (
    BCAT_NAME,
    BCAT_SYMBOL,
    StockAction,
    StockMarketQuote,
    StockProfileView,
    StockPositionView,
    StockTradeLegType,
    StockTradeLegView,
    StockSettlementResult,
)
from discordbot.cogs._stock.views import (
    StockActionView,
    StockDetailView,
    StockMarketView,
    StockQuantityModal,
)

if TYPE_CHECKING:
    import pytest


class ResponseStub:
    """Minimal interaction response stub."""

    def __init__(self) -> None:
        """Initializes captured response state."""
        self.deferred = False
        self.deferred_ephemeral = False
        self.sent: list[dict[str, Any]] = []
        self.modals: list[StockQuantityModal] = []

    async def defer(self, ephemeral: bool = False) -> None:
        """Records a deferred response."""
        self.deferred = True
        self.deferred_ephemeral = ephemeral

    async def send_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records a sent response."""
        self.sent.append(kwargs)

    async def send_modal(self, modal: StockQuantityModal) -> None:
        """Records a launched modal."""
        self.modals.append(modal)

    def is_done(self) -> bool:
        """Returns whether this response has been used."""
        return self.deferred or bool(self.sent) or bool(self.modals)


class FollowupStub:
    """Minimal interaction followup stub."""

    def __init__(self) -> None:
        """Initializes captured followup payloads."""
        self.sent: list[dict[str, Any]] = []

    async def send(self, **kwargs: Any) -> MessageStub:  # noqa: ANN401 -- test double
        """Records a followup send."""
        self.sent.append(kwargs)
        return MessageStub()


class MessageStub:
    """Minimal sent message stub."""

    def __init__(self) -> None:
        """Initializes fake message identity."""
        self.id = 123
        self.channel = SimpleNamespace(id=456)


class UserStub:
    """Minimal user stub."""

    def __init__(self) -> None:
        """Initializes fake user identity."""
        self.id = 1
        self.name = "alice"
        self.display_name = "Alice"
        self.display_avatar = SimpleNamespace(url="https://example.test/avatar.png")


class InteractionStub:
    """Minimal interaction stub."""

    def __init__(self) -> None:
        """Initializes fake Discord interaction pieces."""
        self.user = UserStub()
        self.guild = None
        self.response = ResponseStub()
        self.followup = FollowupStub()
        self.message = MessageStub()


def _quote() -> StockMarketQuote:
    """Builds a deterministic market quote."""
    profile = StockProfileView(
        symbol=BCAT_SYMBOL,
        name=BCAT_NAME,
        category="迷因科技",
        price_cents=10_000,
        previous_close_price_cents=10_000,
        day_open_price_cents=10_000,
        total_shares=1_000_000,
        base_volatility_bps=70,
        volatility_amplifier_bps=150,
        updated_at=datetime(2026, 1, 1),
    )
    return StockMarketQuote(profile=profile, change_cents=0, change_bps=0, pressure_bps=0)


def test_stock_setup_is_sync_and_adds_cog_with_override() -> None:
    """The setup hook is synchronous and uses override=True."""
    calls: list[dict[str, Any]] = []

    class BotStub:
        """Bot stub with add_cog capture."""

        def add_cog(self, cog: StockCogs, override: bool = False) -> None:
            """Records add_cog arguments."""
            calls.append({"cog": cog, "override": override})

    stock.setup(BotStub())

    assert isinstance(calls[0]["cog"], StockCogs)
    assert calls[0]["override"] is True
    assert StockCogs.stock.name == "stock"
    assert StockCogs.stock.name_localizations[Locale.zh_TW] == "股票"


async def test_stock_command_sends_public_market_and_schedules_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The slash command sends public market list and schedules cleanup."""
    scheduled: list[MessageStub] = []

    async def fake_list_market_quotes() -> tuple[StockMarketQuote, ...]:
        """Returns one fake quote."""
        return (_quote(),)

    def fake_schedule(message: MessageStub, user_name: str | None = None) -> None:
        """Records cleanup scheduling."""
        scheduled.append(message)

    monkeypatch.setattr(stock, "list_market_quotes", fake_list_market_quotes)
    monkeypatch.setattr(stock, "schedule_game_message_delete", fake_schedule)
    cog = StockCogs(bot=SimpleNamespace())
    interaction = InteractionStub()

    await StockCogs.stock.callback(cog, interaction)

    assert interaction.response.deferred
    assert interaction.followup.sent[0].get("ephemeral") is not True
    assert isinstance(interaction.followup.sent[0]["view"], StockMarketView)
    assert scheduled


async def test_stock_market_select_returns_ephemeral_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting a stock opens the personal detail flow."""
    selected: list[str] = []

    async def fake_send_stock_detail(interaction: InteractionStub, symbol: str) -> None:
        """Records selected stock detail requests."""
        selected.append(symbol)
        await interaction.response.defer(ephemeral=True)

    monkeypatch.setattr(stock_views, "send_stock_detail", fake_send_stock_detail)
    view = StockMarketView(quotes=(_quote(),))
    interaction = InteractionStub()
    view.stock_select._selected_values = [BCAT_SYMBOL]

    await view.stock_select.callback(interaction)

    assert selected == [BCAT_SYMBOL]
    assert interaction.response.deferred_ephemeral


async def test_stock_detail_buttons_open_action_news_and_back_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detail buttons produce ephemeral action, news, and market-list responses."""

    async def fake_news(symbol: str) -> tuple:
        """Returns no fake news."""
        return ()

    async def fake_quotes() -> tuple[StockMarketQuote, ...]:
        """Returns one fake quote."""
        return (_quote(),)

    monkeypatch.setattr(stock_views, "get_stock_news", fake_news)
    monkeypatch.setattr(stock_views, "list_market_quotes", fake_quotes)
    view = StockDetailView(symbol=BCAT_SYMBOL)

    operate = next(
        child for child in view.children if getattr(child, "custom_id", "") == "stock:operate"
    )
    news = next(
        child for child in view.children if getattr(child, "custom_id", "") == "stock:news"
    )
    back = next(
        child for child in view.children if getattr(child, "custom_id", "") == "stock:back"
    )

    operate_interaction = InteractionStub()
    await operate.callback(operate_interaction)
    assert operate_interaction.response.sent[0]["ephemeral"] is True
    assert isinstance(operate_interaction.response.sent[0]["view"], StockActionView)

    news_interaction = InteractionStub()
    await news.callback(news_interaction)
    assert news_interaction.response.sent[0]["ephemeral"] is True
    assert "近期新聞" in news_interaction.response.sent[0]["embed"].title

    back_interaction = InteractionStub()
    await back.callback(back_interaction)
    assert back_interaction.response.sent[0]["ephemeral"] is True
    assert isinstance(back_interaction.response.sent[0]["view"], StockMarketView)


async def test_stock_action_buttons_launch_text_input_modals() -> None:
    """Action buttons launch modals with TextInput quantity only."""
    view = StockActionView(symbol=BCAT_SYMBOL)
    buy = next(child for child in view.children if getattr(child, "custom_id", "") == "stock:buy")
    short = next(
        child for child in view.children if getattr(child, "custom_id", "") == "stock:short"
    )

    buy_interaction = InteractionStub()
    await buy.callback(buy_interaction)
    short_interaction = InteractionStub()
    await short.callback(short_interaction)

    assert buy_interaction.response.modals[0].action == StockAction.BUY
    assert short_interaction.response.modals[0].action == StockAction.SHORT
    assert isinstance(buy_interaction.response.modals[0].quantity, stock_views.TextInput)


async def test_stock_modal_reports_invalid_input_root_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid modal input returns an ephemeral error embed."""

    async def fake_settle_stock_operation(**kwargs: Any) -> StockSettlementResult:  # noqa: ANN401
        """Returns the same invalid-format failure the service would return."""
        return StockSettlementResult(
            success=False,
            operation_id=None,
            symbol=kwargs["symbol"],
            requested_action=kwargs["requested_action"],
            shares=0,
            price_cents=10_000,
            wallet_delta=0,
            balance_after=100,
            position=StockPositionView(symbol=kwargs["symbol"], user_id=1),
            legs=(),
            error="股數格式錯誤，請輸入正整數或 ALL",
        )

    monkeypatch.setattr(stock_views, "settle_stock_operation", fake_settle_stock_operation)
    modal = StockQuantityModal(symbol=BCAT_SYMBOL, action=StockAction.BUY)
    interaction = InteractionStub()

    await modal.submit_quantity(interaction=interaction, raw_quantity="abc")

    assert interaction.response.deferred_ephemeral
    embed = interaction.followup.sent[0]["embed"]
    assert isinstance(embed, Embed)
    assert "股數格式錯誤" in embed.description
    assert interaction.followup.sent[0]["ephemeral"] is True


async def test_successful_stock_modal_returns_result_and_refresh_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful modal submission returns a settlement embed and refresh control."""

    async def fake_settle_stock_operation(**kwargs: Any) -> StockSettlementResult:  # noqa: ANN401
        """Returns a successful fake settlement."""
        return StockSettlementResult(
            success=True,
            operation_id="op-1",
            symbol=kwargs["symbol"],
            requested_action=kwargs["requested_action"],
            shares=1,
            price_cents=10_000,
            wallet_delta=-100,
            balance_after=900,
            position=StockPositionView(symbol=kwargs["symbol"], user_id=1, long_shares=1),
            legs=(
                StockTradeLegView(
                    operation_id="op-1",
                    leg_order=1,
                    symbol=kwargs["symbol"],
                    user_id=1,
                    leg_type=StockTradeLegType.OPEN_LONG,
                    shares=1,
                    price_cents=10_000,
                    wallet_delta=-100,
                    basis_delta=100,
                    collateral_delta=0,
                    realized_pnl_delta=0,
                    created_at=datetime(2026, 1, 1),
                ),
            ),
        )

    monkeypatch.setattr(stock_views, "settle_stock_operation", fake_settle_stock_operation)
    modal = StockQuantityModal(symbol=BCAT_SYMBOL, action=StockAction.BUY)
    interaction = InteractionStub()

    await modal.submit_quantity(interaction=interaction, raw_quantity="1")

    assert "交易完成" in interaction.followup.sent[0]["embed"].title
    assert interaction.followup.sent[0]["ephemeral"] is True
    assert interaction.followup.sent[0]["view"] is not None


def test_stock_readme_and_help_metadata_are_covered() -> None:
    """Stock command metadata stays discoverable by help/readme tests."""
    assert StockCogs.stock.description == "Open the simulated stock market."
