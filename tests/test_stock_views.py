"""Tests for the stock cog and interactive views."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import Any
import asyncio
from datetime import datetime

from PIL import Image
import pytest
from nextcord import File, Embed, Locale

from discordbot.cogs import stock
from discordbot.utils import owned_message_views
from discordbot.cogs.stock import StockCogs
from discordbot.cogs._stock import views as stock_views
from discordbot.typings.stock import (
    StockAction,
    StockMarketQuote,
    StockProfileView,
    StockPositionView,
    StockTradeLegType,
    StockTradeLegView,
    StockPriceTickView,
    StockDetailViewData,
    StockOperationStatus,
    StockSettlementResult,
    StockParticipantPositionView,
)
from discordbot.cogs._stock.chart import build_price_chart, _render_price_chart
from discordbot.cogs._stock.views import (
    StockActionView,
    StockDetailView,
    StockMarketView,
    StockPublicView,
    StockPostTradeView,
    StockQuantityModal,
)
from discordbot.cogs._stock.presentation import (
    build_market_embed,
    market_board_filename,
    build_settlement_embed,
    build_market_board_image,
    build_stock_detail_embed,
    _build_market_board_image_cached,
)

BCAT_SYMBOL = "BCAT"
BCAT_NAME = "破貓科技股份有限公司"


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

    async def edit_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records an edited response."""
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
        self.edits: list[dict[str, Any]] = []
        self.deleted = False

    async def edit(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records a message edit."""
        self.edits.append(kwargs)

    async def delete(self) -> None:
        """Records message deletion."""
        self.deleted = True


class DeletedMessageStub(MessageStub):
    """Message stub that has already been deleted remotely."""

    async def edit(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Raises the same exception nextcord emits for deleted messages."""
        response = SimpleNamespace(status=404, reason="Not Found", headers={})
        raise stock_views.nextcord.NotFound(response=response, message="missing")


class UserStub:
    """Minimal user stub."""

    def __init__(self, user_id: int = 1, name: str = "alice") -> None:
        """Initializes fake user identity."""
        self.id = user_id
        self.name = name
        self.display_name = name.title()
        self.display_avatar = SimpleNamespace(url="https://example.test/avatar.png")


class InteractionStub:
    """Minimal interaction stub."""

    def __init__(self, user_id: int | None = 1, name: str = "alice") -> None:
        """Initializes fake Discord interaction pieces."""
        self.user = UserStub(user_id=user_id, name=name) if user_id is not None else None
        self.guild = None
        self.response = ResponseStub()
        self.followup = FollowupStub()
        self.message = MessageStub()


def _quote(name: str = BCAT_NAME) -> StockMarketQuote:
    """Builds a deterministic market quote."""
    profile = StockProfileView(
        symbol=BCAT_SYMBOL,
        name=name,
        category="科技",
        price_cents=10_000,
        previous_close_price_cents=10_000,
        day_open_price_cents=10_000,
        total_shares=1_000_000,
        float_shares=650_000,
        base_volatility_bps=70,
        volatility_amplifier_bps=150,
        liquidity_shares=25_000,
        fair_value_cents=10_000,
        mean_reversion_bps=35,
        max_tick_change_bps=450,
        news_cadence_hours=8,
        updated_at=datetime(2026, 1, 1),
    )
    return StockMarketQuote(profile=profile, change_cents=0, change_bps=0, pressure_bps=0)


def _detail(long_shares: int = 0, short_shares: int = 0) -> StockDetailViewData:
    """Builds a deterministic stock detail payload."""
    return StockDetailViewData(
        quote=_quote(),
        balance=1_000_000,
        position=StockPositionView(
            symbol=BCAT_SYMBOL,
            user_id=1,
            user_name="alice",
            long_shares=long_shares,
            short_shares=short_shares,
        ),
        recent_trades=(),
        public_positions=(),
        news=(),
        ticks=(),
    )


def _stock_trade_leg(index: int, user_name: str) -> StockTradeLegView:
    """Builds one deterministic stock trade leg."""
    return StockTradeLegView(
        operation_id=f"operation-{index}",
        leg_order=index,
        symbol=BCAT_SYMBOL,
        user_id=index,
        user_name=user_name,
        leg_type=StockTradeLegType.OPEN_LONG,
        shares=index * 1000,
        price_cents=10_000 + index,
        wallet_delta=-(index * 100),
        basis_delta=index * 100,
        collateral_delta=0,
        realized_pnl_delta=index,
        created_at=datetime(2026, 1, index),
    )


def _stock_participant(
    user_id: int, user_name: str, long_shares: int, short_shares: int = 0
) -> StockParticipantPositionView:
    """Builds one deterministic public participant position."""
    return StockParticipantPositionView(
        user_id=user_id,
        user_name=user_name,
        long_shares=long_shares,
        short_shares=short_shares,
        realized_pnl=user_id * 100,
    )


def _field_value(embed: Embed, name: str) -> str:
    """Returns one embed field value by name."""
    for field in embed.fields:
        if field.name == name:
            return str(field.value)
    raise AssertionError(f"missing embed field: {name}")


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
    """The slash command sends public market list and tracks cleanup."""
    scheduled: list[MessageStub] = []
    scheduled_news_refreshes: list[object] = []

    async def fake_list_market_quotes(refresh_news: bool = True) -> tuple[StockMarketQuote, ...]:
        """Returns one fake quote."""
        assert refresh_news is False
        return (_quote(),)

    def fake_schedule_stock_news_refresh(news_ai: object) -> None:
        """Records background news refresh scheduling."""
        scheduled_news_refreshes.append(news_ai)

    async def fake_track(message: MessageStub, user_name: str | None = None) -> None:
        """Records cleanup tracking."""
        scheduled.append(message)

    monkeypatch.setattr(stock, "list_market_quotes", fake_list_market_quotes)
    monkeypatch.setattr(stock, "_schedule_stock_news_refresh", fake_schedule_stock_news_refresh)
    monkeypatch.setattr(stock, "track_public_message", fake_track)
    cog = StockCogs(bot=SimpleNamespace())
    cog.__dict__["news_ai"] = SimpleNamespace(generate=lambda _profile: None)
    interaction = InteractionStub()

    await StockCogs.stock.callback(cog, interaction)

    assert interaction.response.deferred
    assert interaction.user is not None
    assert interaction.followup.sent[0].get("ephemeral") is not True
    assert isinstance(interaction.followup.sent[0]["view"], StockMarketView)
    assert interaction.followup.sent[0]["files"][0].filename == "stock_market_1.png"
    assert interaction.followup.sent[0]["view"].message is scheduled[0]
    assert interaction.followup.sent[0]["view"].owner_id == interaction.user.id
    assert scheduled
    assert scheduled_news_refreshes == [cog.news_ai]


async def test_stock_news_background_refresh_is_deduped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent stock command refresh scheduling keeps one active task."""
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_ensure_due_stock_news(news_provider: object) -> None:
        """Blocks the active refresh so the second schedule can observe it."""
        nonlocal calls
        assert news_provider is not None
        calls += 1
        started.set()
        await release.wait()

    monkeypatch.setattr(stock, "ensure_due_stock_news", fake_ensure_due_stock_news)
    monkeypatch.setattr(stock, "_stock_news_refresh_task", None)
    monkeypatch.setattr(stock, "_stock_news_refresh_task_loop", None)
    news_ai = SimpleNamespace(generate=lambda _context: None)

    stock._schedule_stock_news_refresh(news_ai=news_ai)
    await started.wait()
    stock._schedule_stock_news_refresh(news_ai=news_ai)
    assert calls == 1

    task = stock._stock_news_refresh_task
    assert task is not None
    release.set()
    await task
    stock._schedule_stock_news_refresh(news_ai=news_ai)
    second_task = stock._stock_news_refresh_task
    assert second_task is not None
    await second_task
    assert calls == 2


async def test_stock_command_raises_when_interaction_has_no_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stock command fails loudly instead of creating an unowned public panel."""
    called = False

    async def fake_list_market_quotes() -> tuple[StockMarketQuote, ...]:
        """Records unexpected market loading."""
        nonlocal called
        called = True
        return (_quote(),)

    monkeypatch.setattr(stock, "list_market_quotes", fake_list_market_quotes)
    cog = StockCogs(bot=SimpleNamespace())
    interaction = InteractionStub(user_id=None)

    with pytest.raises(RuntimeError, match="missing Discord user identity"):
        await StockCogs.stock.callback(cog, interaction)

    assert interaction.response.deferred
    assert not called
    assert interaction.followup.sent == []


async def test_stock_public_view_rejects_non_owner_interaction() -> None:
    """Only the user who opened a public stock panel can operate its controls."""
    view = StockMarketView(quotes=(_quote(),), owner_id=1)
    intruder = InteractionStub(user_id=2, name="bob")

    assert await view.interaction_check(interaction=InteractionStub(user_id=1)) is True
    assert await view.interaction_check(interaction=intruder) is False
    assert intruder.response.sent[0]["ephemeral"] is True
    assert "只有發起者" in intruder.response.sent[0]["content"]


async def test_stock_market_select_edits_public_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting a stock edits the same public detail flow."""
    selected: list[str] = []
    owners: list[int | None] = []

    async def fake_edit_stock_detail(
        interaction: InteractionStub, symbol: str, owner_id: int | None = None
    ) -> None:
        """Records selected stock detail requests."""
        selected.append(symbol)
        owners.append(owner_id)
        await interaction.response.defer()

    monkeypatch.setattr(stock_views, "edit_stock_detail", fake_edit_stock_detail)
    view = StockMarketView(quotes=(_quote(),), owner_id=1)
    interaction = InteractionStub()
    view.stock_select._selected_values = [BCAT_SYMBOL]

    await view.stock_select.callback(interaction)

    assert selected == [BCAT_SYMBOL]
    assert interaction.user is not None
    assert owners == [interaction.user.id]
    assert interaction.response.deferred
    assert not interaction.response.deferred_ephemeral


async def test_stock_market_select_truncates_long_company_names() -> None:
    """The market dropdown keeps option labels inside Discord's limit."""
    view = StockMarketView(quotes=(_quote(name="長" * 128),), owner_id=1)
    option = view.stock_select.options[0]

    assert len(option.label) == stock_views.SELECT_OPTION_LABEL_LIMIT
    assert option.label.startswith(f"{BCAT_SYMBOL} · 長")
    assert option.label.endswith("...")


def test_stock_market_embed_uses_board_attachment_for_rows() -> None:
    """The market embed keeps tabular rows out of Markdown text."""
    filename = market_board_filename(page_index=0)
    embed = build_market_embed(quotes=(_quote(),), board_filename=filename)

    assert embed.image.url == f"attachment://{filename}"
    assert "市值" not in embed.description
    assert "100,000,000" not in embed.description


def test_stock_market_board_handles_large_market_caps() -> None:
    """The market board renders huge market caps without relying on long text rows."""
    quote = _quote().model_copy(
        update={
            "profile": _quote().profile.model_copy(
                update={"price_cents": 987_654_321, "total_shares": 123_456_789}
            ),
            "change_bps": -1234,
            "pressure_bps": 987,
        }
    )

    image = build_market_board_image(quotes=(quote,))

    assert image.startswith(b"\x89PNG")
    with Image.open(BytesIO(image)) as opened:
        assert opened.size[0] == 1120
        assert opened.size[1] > 180


def test_stock_market_board_image_cache_key_changes_with_quote_digest() -> None:
    """Market board renders are cached by immutable quote fields."""
    _build_market_board_image_cached.cache_clear()
    quote = _quote()

    build_market_board_image(quotes=(quote,))
    assert _build_market_board_image_cached.cache_info().hits == 0
    assert _build_market_board_image_cached.cache_info().misses == 1

    build_market_board_image(quotes=(quote,))
    assert _build_market_board_image_cached.cache_info().hits == 1

    changed = quote.model_copy(
        update={"profile": quote.profile.model_copy(update={"price_cents": 12_345})}
    )
    build_market_board_image(quotes=(changed,))
    assert _build_market_board_image_cached.cache_info().misses == 2


def test_stock_chart_image_cache_key_changes_with_ticks() -> None:
    """7D chart renders are cached by immutable tick rows."""
    _render_price_chart.cache_clear()
    first_ticks = (
        StockPriceTickView(
            symbol=BCAT_SYMBOL, price_cents=10_000, created_at=datetime(2026, 1, 1)
        ),
    )
    second_ticks = (
        StockPriceTickView(
            symbol=BCAT_SYMBOL, price_cents=10_001, created_at=datetime(2026, 1, 1)
        ),
    )

    build_price_chart(ticks=first_ticks)
    assert _render_price_chart.cache_info().misses == 1
    build_price_chart(ticks=first_ticks)
    assert _render_price_chart.cache_info().hits == 1
    build_price_chart(ticks=second_ticks)
    assert _render_price_chart.cache_info().misses == 2


async def test_stock_detail_buttons_edit_same_public_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detail buttons edit the original public message instead of sending followups."""

    async def fake_news(symbol: str) -> tuple:
        """Returns no fake news."""
        return ()

    async def fake_quotes() -> tuple[StockMarketQuote, ...]:
        """Returns one fake quote."""
        return (_quote(),)

    async def fake_detail(symbol: str, user_id: int, user_name: str) -> StockDetailViewData:
        """Returns fake detail for the operation panel."""
        assert symbol == BCAT_SYMBOL
        assert user_id == 1
        assert user_name == "alice"
        return _detail(long_shares=3, short_shares=2)

    monkeypatch.setattr(stock_views, "get_stock_news", fake_news)
    monkeypatch.setattr(stock_views, "list_market_quotes", fake_quotes)
    monkeypatch.setattr(stock_views, "get_stock_detail", fake_detail)
    view = StockDetailView(symbol=BCAT_SYMBOL, owner_id=1)

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
    assert operate_interaction.response.deferred
    embed = operate_interaction.message.edits[0]["embed"]
    assert isinstance(embed, Embed)
    assert "股票代碼" in embed.description
    assert "股票代碼：BCAT" in embed.description
    assert "100.00 虛擬歡樂豆" in embed.description
    assert "目前持有：3股 | 目前做空：2股" in embed.description
    assert isinstance(operate_interaction.message.edits[0]["view"], StockActionView)

    news_interaction = InteractionStub()
    await news.callback(news_interaction)
    assert "近期新聞" in news_interaction.response.sent[0]["embed"].title
    assert news_interaction.response.sent[0]["view"].owner_id == view.owner_id

    back_interaction = InteractionStub()
    await back.callback(back_interaction)
    assert isinstance(back_interaction.response.sent[0]["view"], StockMarketView)
    assert back_interaction.response.sent[0]["view"].owner_id == view.owner_id


def test_stock_detail_embed_uses_localized_user_labels() -> None:
    """The public stock detail embed avoids placeholder-like mixed UI labels."""
    embed = build_stock_detail_embed(detail=_detail(), chart_filename="chart.png")

    field_names = {field.name for field in embed.fields}
    assert "目前操作使用者" in field_names
    assert "可用資金" in field_names
    assert "目前操作 user" not in field_names
    assert "操作 user 資金" not in field_names
    assert "市值 `1億`" in embed.description


def test_stock_detail_embed_displays_large_share_counts_as_lots() -> None:
    """The public stock detail embed keeps huge share counts readable."""
    embed = build_stock_detail_embed(
        detail=_detail(long_shares=10_000_000_000_000, short_shares=1_234),
        chart_filename="chart.png",
    )

    field_values = "\n".join(str(field.value) for field in embed.fields)
    assert "持股數 `100億張`" in field_values
    assert "做空股數 `1張 234股`" in field_values


def test_stock_detail_embed_compacts_public_position_summary() -> None:
    """The public stock detail embed shows only the top three shareholders."""
    detail = _detail().model_copy(
        update={
            "public_positions": (
                _stock_participant(
                    user_id=5, user_name="short_whale", long_shares=0, short_shares=9999
                ),
                _stock_participant(user_id=4, user_name="dave", long_shares=1000),
                _stock_participant(user_id=2, user_name="bob", long_shares=2000),
                _stock_participant(user_id=6, user_name="erin", long_shares=500),
                _stock_participant(user_id=3, user_name="carol", long_shares=3000),
            )
        }
    )

    embed = build_stock_detail_embed(detail=detail, chart_filename="chart.png")

    value = _field_value(embed=embed, name="公開部位摘要")
    assert "1. **carol** 持股 `3張`" in value
    assert "2. **bob** 持股 `2張`" in value
    assert "3. **dave** 持股 `1張`" in value
    assert "-# 做空" in value
    assert "已實現損益" in value
    assert "short_whale" not in value
    assert "erin" not in value


def test_stock_detail_embed_compacts_recent_trades() -> None:
    """The public stock detail embed shows only three recent trade rows."""
    detail = _detail().model_copy(
        update={
            "recent_trades": (
                _stock_trade_leg(index=1, user_name="alice"),
                _stock_trade_leg(index=2, user_name="bob"),
                _stock_trade_leg(index=3, user_name="carol"),
                _stock_trade_leg(index=4, user_name="dave"),
            )
        }
    )

    embed = build_stock_detail_embed(detail=detail, chart_filename="chart.png")

    value = _field_value(embed=embed, name="近期交易")
    assert "1. **alice** 買入 `1張`" in value
    assert "2. **bob** 買入 `2張`" in value
    assert "3. **carol** 買入 `3張`" in value
    assert "-# #1 · 錢包變化" in value
    assert "dave" not in value


async def test_stock_action_dropdown_launches_quantity_modal() -> None:
    """Action dropdown launches one modal with only the quantity input."""
    view = StockActionView(symbol=BCAT_SYMBOL, owner_id=1)
    action_select = next(
        child for child in view.children if getattr(child, "custom_id", "") == "stock:action"
    )
    action_select._selected_values = [StockAction.SHORT.value]

    interaction = InteractionStub()
    await action_select.callback(interaction)

    assert interaction.response.modals[0].action == StockAction.SHORT
    assert interaction.response.modals[0].owner_id == view.owner_id
    assert isinstance(interaction.response.modals[0].quantity, stock_views.TextInput)
    components = interaction.response.modals[0].to_dict()["components"]
    assert [row["components"][0]["type"] for row in components] == [4]
    assert all(getattr(child, "custom_id", "") != "stock:quantity" for child in view.children)


async def test_stock_modal_rejects_non_owner_before_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copied or stale stock modal cannot submit for someone other than the panel owner."""
    calls: list[dict[str, Any]] = []

    async def fake_settle_stock_operation(**kwargs: Any) -> StockSettlementResult:  # noqa: ANN401
        """Records unexpected settlement calls."""
        calls.append(kwargs)
        return StockSettlementResult(
            success=False,
            operation_id=None,
            symbol=kwargs["symbol"],
            requested_action=kwargs["requested_action"],
            shares=0,
            price_cents=10_000,
            wallet_delta=0,
            balance_after=100,
            position=StockPositionView(symbol=kwargs["symbol"], user_id=2),
            legs=(),
            error="unexpected",
        )

    monkeypatch.setattr(stock_views, "settle_stock_operation", fake_settle_stock_operation)
    modal = StockQuantityModal(symbol=BCAT_SYMBOL, action=StockAction.BUY, owner_id=1)
    intruder = InteractionStub(user_id=2, name="bob")

    await modal.submit_quantity(interaction=intruder, raw_quantity="1")

    assert calls == []
    assert intruder.response.sent[0]["ephemeral"] is True
    assert "只有發起者" in intruder.response.sent[0]["content"]


async def test_stock_modal_reports_invalid_input_root_cause_in_public_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid modal input edits the public message with the root-cause error."""

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
    modal = StockQuantityModal(symbol=BCAT_SYMBOL, action=StockAction.BUY, owner_id=1)
    interaction = InteractionStub()

    await modal.submit_quantity(interaction=interaction, raw_quantity="abc")

    assert interaction.response.deferred
    assert not interaction.response.deferred_ephemeral
    embed = interaction.message.edits[0]["embed"]
    assert isinstance(embed, Embed)
    assert "股數格式錯誤" in embed.description
    assert isinstance(interaction.message.edits[0]["view"], StockActionView)
    assert interaction.user is not None
    assert interaction.message.edits[0]["view"].owner_id == interaction.user.id


async def test_successful_stock_modal_edits_result_and_refresh_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful modal submission edits the public message with a refresh control."""

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
            position=StockPositionView(
                symbol=kwargs["symbol"], user_id=1, user_name="alice", long_shares=1
            ),
            legs=(
                StockTradeLegView(
                    operation_id="op-1",
                    leg_order=1,
                    symbol=kwargs["symbol"],
                    user_id=1,
                    user_name="alice",
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
    modal = StockQuantityModal(symbol=BCAT_SYMBOL, action=StockAction.BUY, owner_id=1)
    interaction = InteractionStub()

    await modal.submit_quantity(interaction=interaction, raw_quantity="1")

    assert not interaction.response.deferred_ephemeral
    assert "交易完成" in interaction.message.edits[0]["embed"].title
    assert "錢包變化" in interaction.message.edits[0]["embed"].description
    assert "Wallet" not in interaction.message.edits[0]["embed"].description
    assert interaction.message.edits[0]["embed"].fields[0].name == "交易明細"
    assert isinstance(interaction.message.edits[0]["view"], StockPostTradeView)
    assert interaction.user is not None
    assert interaction.message.edits[0]["view"].owner_id == interaction.user.id


def test_failed_stock_settlement_title_does_not_depend_on_operation_id() -> None:
    """Failed stock settlements with audit IDs are not reconciliation incidents."""
    result = StockSettlementResult(
        success=False,
        operation_id="op-1",
        symbol=BCAT_SYMBOL,
        requested_action=StockAction.BUY,
        shares=1,
        price_cents=10_000,
        wallet_delta=0,
        balance_after=900,
        position=StockPositionView(symbol=BCAT_SYMBOL, user_id=1, user_name="alice"),
        legs=(),
        status=StockOperationStatus.FAILED,
        error="交易未完成，送出時餘額已不足，沒有變更股票部位",
    )

    embed = build_settlement_embed(result=result)

    assert embed.title == "股票交易失敗"
    assert embed.fields[0].name == "操作代碼"


async def test_edit_owned_public_message_recovers_when_target_was_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale panel message edit sends a public followup instead of dropping the result."""
    forgotten: list[int] = []
    tracked: list[MessageStub] = []

    async def fake_forget(message_id: int) -> None:
        """Records the stale cleanup row removal."""
        forgotten.append(message_id)

    async def fake_track(message: MessageStub, user_name: str | None = None) -> None:
        """Records the replacement cleanup row."""
        tracked.append(message)

    monkeypatch.setattr(owned_message_views, "forget_public_message", fake_forget)
    monkeypatch.setattr(owned_message_views, "track_public_message", fake_track)
    interaction = InteractionStub()
    interaction.response.deferred = True
    interaction.message = DeletedMessageStub()
    view = StockPostTradeView(symbol=BCAT_SYMBOL, owner_id=1)
    chart_file = File(fp=BytesIO(b"chart-bytes"), filename="chart.png")

    await owned_message_views.edit_owned_public_message(
        interaction=interaction,
        embed=Embed(title="股票交易完成"),
        view=view,
        file=chart_file,
        message=interaction.message,
    )

    assert interaction.followup.sent[0].get("ephemeral") is not True
    assert interaction.followup.sent[0]["view"] is view
    assert interaction.followup.sent[0]["files"][0] is not chart_file
    assert interaction.followup.sent[0]["files"][0].filename == "chart.png"
    assert interaction.followup.sent[0]["files"][0].fp.read() == b"chart-bytes"
    assert view.message is not interaction.message
    assert forgotten == [interaction.message.id]
    assert tracked == [view.message]


async def test_stock_public_view_timeout_deletes_bound_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The active stock view uses shared public-message cleanup after idle timeout."""
    deleted: list[MessageStub] = []

    async def fake_delete(message: MessageStub) -> None:
        """Records delegated public-message deletion."""
        deleted.append(message)

    monkeypatch.setattr(owned_message_views, "delete_public_message", fake_delete)
    message = MessageStub()
    view = StockPublicView(owner_id=1)
    view.bind_message(message=message)

    await view.on_timeout()

    assert deleted == [message]


def test_stock_readme_and_help_metadata_are_covered() -> None:
    """Stock command metadata stays discoverable by help/readme tests."""
    assert StockCogs.stock.description == "Open the simulated stock market."
