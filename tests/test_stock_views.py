"""Pins the `/stock` panel's ownership, its one-message discipline, and what its renders cache on.

The stock feature is a single public message. `cog.py` posts it, binds a `StockMarketView` to it
and records it for cleanup; every later screen — paging, one stock's detail, its news, the action
dropdown, the settlement receipt — repaints that same message through `edit_owned_public_message`
rather than sending a second one. Two things follow from that shape and both are held here. The
panel is operable by its opener alone, and the quantity modal is the hole in that rule: a modal
submit arrives on its own interaction that never runs the view's `interaction_check`, so
`submit_stock_quantity` re-applies the gate by hand, and a regression there lets a stranger
holding a stale modal trade on their own wallet and repaint someone else's panel with the receipt.
The message can also be gone by the time a result is ready, so an edit that 404s re-sends the
result as a public followup, drops the stale cleanup row and tracks the replacement rather than
losing the only receipt the trade has.

The two render caches are the other half. The market board PNG and the 7D chart are process-wide
`lru_cache`s that no write path invalidates, so their keys have to carry every field that reaches
a pixel; each cache test clears the cache, renders twice for a hit, then moves one such field and
asserts a miss. A key blind to a field would serve the previous board for a price that has moved.

What is left is the presentation contract. Market rows live in the board PNG rather than in
Markdown text (proportional CJK columns do not line up, and a 兆-scale market cap has nowhere to
fit), a select option label stays inside Discord's cap, the shareholder and recent-trade fields
are trimmed to three, the detail fields are labelled in Chinese rather than half-translated, and a
failed settlement carrying an operation id is titled a plain failure rather than a reconciliation
incident.

Nothing here reads `stock.db` or a wallet: every service call is monkeypatched onto the module
that resolves it — `cog.py` for the market read and the cleanup tracker, `views.py` for the
detail, news and settlement ones — so the assertions are about the panel, not about the market
simulation `tests/test_stock.py` owns. The stubs at the top stand in for the slice of a nextcord
interaction these paths touch, and the `_quote` / `_detail` builders freeze the read models at
fixed prices and timestamps so a render is reproducible.
"""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
import asyncio
from datetime import datetime

from PIL import Image
import pytest
from nextcord import File, Embed, Locale
from nextcord.ui import StringSelect

from discordbot.utils import owned_message_views
from discordbot.cogs.stock import cog as stock
from discordbot.cogs.stock import views as stock_views
from discordbot.typings.stock import (
    StockAction,
    StockNewsView,
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
from discordbot.cogs.stock.cog import StockCogs
from discordbot.cogs.stock.chart import build_price_chart, _render_price_chart
from discordbot.cogs.stock.views import (
    StockActionView,
    StockDetailView,
    StockMarketView,
    StockPublicView,
    StockPostTradeView,
    StockQuantityModal,
)
from discordbot.cogs.stock.presentation import (
    build_market_embed,
    market_board_filename,
    build_settlement_embed,
    build_market_board_image,
    build_stock_detail_embed,
    _build_market_board_image_cached,
)

from tests.helpers.casting import as_bot, as_message, as_interaction, make_not_found

if TYPE_CHECKING:
    from discordbot.cogs.stock.news import StockNewsAI

BCAT_SYMBOL = "BCAT"
BCAT_NAME = "破貓科技股份有限公司"


class ResponseStub:
    """Records which response path a callback took: defer, send, edit or modal."""

    def __init__(self) -> None:
        """Initializes captured response state."""
        self.deferred = False
        self.deferred_ephemeral = False
        self.sent: list[dict[str, Any]] = []
        self.modals: list[StockQuantityModal] = []

    async def defer(self, ephemeral: bool = False) -> None:
        """Records the defer and whether the panel would have gone private with it."""
        self.deferred = True
        self.deferred_ephemeral = ephemeral

    async def send_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records a sent response."""
        self.sent.append(kwargs)

    async def edit_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records a single-request edit, in the same list as a send."""
        self.sent.append(kwargs)

    async def send_modal(self, modal: StockQuantityModal) -> None:
        """Records a launched modal, kept whole so its fields can be read back."""
        self.modals.append(modal)

    def is_done(self) -> bool:
        """Returns whether anything already answered this interaction, as production reads it."""
        return self.deferred or bool(self.sent) or bool(self.modals)


class FollowupStub:
    """Records the followup payloads a panel falls back to sending."""

    def __init__(self) -> None:
        """Initializes captured followup payloads."""
        self.sent: list[dict[str, Any]] = []

    async def send(self, **kwargs: Any) -> MessageStub:  # noqa: ANN401 -- test double
        """Returns a fresh message stub for the send, after recording its payload."""
        self.sent.append(kwargs)
        return MessageStub()


class MessageStub:
    """Records the edits and the deletion a panel aims at the message it owns."""

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
    """Message stub standing in for a panel message Discord no longer has."""

    async def edit(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Raises the `NotFound` nextcord raises for a message that is already gone.

        Built through `make_not_found` because `NotFound` needs an aiohttp response to construct.
        """  # noqa: DOC501 -- ruff reads the raise as `make_not_found`, a builder, not a type
        raise make_not_found(message="missing")


class UserStub:
    """Minimal presser identity: the id the owner gate weighs and the name settlement stores."""

    def __init__(self, user_id: int = 1, name: str = "alice") -> None:
        """Initializes fake user identity."""
        self.id = user_id
        self.name = name
        self.display_name = name.title()
        self.display_avatar = SimpleNamespace(url="https://example.test/avatar.png")


class InteractionStub:
    """The interaction slice a panel reads: presser, guild, response, followup and its message."""

    def __init__(self, user_id: int | None = 1, name: str = "alice") -> None:
        """Builds the pieces; `user_id=None` is the identity-less interaction the panel refuses."""
        self.user = UserStub(user_id=user_id, name=name) if user_id is not None else None
        self.guild = None
        self.response = ResponseStub()
        self.followup = FollowupStub()
        self.message = MessageStub()


def _quote(name: str = BCAT_NAME) -> StockMarketQuote:
    """Builds the one quote every board and detail render here is drawn from.

    Returns:
        A flat quote — no change, no pressure, a round price and a fixed timestamp — so a render
        is byte-stable and a cache test only moves the field it is about.
    """
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
    """Builds the detail read the detail and action embeds render from.

    Returns:
        A detail whose stock-wide lists are all empty, so a test that cares about one of them
        fills only that one back in through `model_copy`.
    """
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
    """Builds one recent-trade leg, every figure derived from its index.

    Returns:
        An OPEN_LONG leg whose shares, price and deltas all follow `index`, so a rendered line
        names the leg it came from and a trimmed list is readable as a rank.
    """
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
    """Builds one row of the public shareholder table.

    Returns:
        A participant holding what was asked for, its realized P&L derived from the id so no two
        rows render alike.
    """
    return StockParticipantPositionView(
        user_id=user_id,
        user_name=user_name,
        long_shares=long_shares,
        short_shares=short_shares,
        realized_pnl=user_id * 100,
    )


def _field_value(embed: Embed, name: str) -> str:
    """Returns one embed field's value, looked up by the heading a user reads.

    Raises:
        AssertionError: No field on the embed carries that name, which means the field was
            renamed or dropped rather than that its contents changed.
    """
    for field in embed.fields:
        if field.name == name:
            return str(field.value)
    raise AssertionError(f"missing embed field: {name}")


def test_stock_setup_is_sync_and_adds_cog_with_override() -> None:
    """`setup` is sync, adds the cog with `override=True`, and `/stock` keeps its zh_TW name."""
    calls: list[dict[str, Any]] = []

    class BotStub:
        """Bot stub capturing what `setup` registers."""

        def add_cog(self, cog: StockCogs, override: bool = False) -> None:
            """Records add_cog arguments."""
            calls.append({"cog": cog, "override": override})

    stock.setup(bot=as_bot(fake=BotStub()))

    assert isinstance(calls[0]["cog"], StockCogs)
    assert calls[0]["override"] is True
    assert StockCogs.stock.name == "stock"
    assert StockCogs.stock.name_localizations[Locale.zh_TW] == "股票"


async def test_stock_command_sends_public_market_and_schedules_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/stock` posts the public board, binds and tracks that message, owned by its opener."""
    scheduled: list[MessageStub] = []
    scheduled_news_refreshes: list[object] = []

    async def fake_list_market_quotes(refresh_news: bool = True) -> tuple[StockMarketQuote, ...]:
        """Returns one quote, pinning that a writer-backed command leaves the inline sweep off."""
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
    cog = StockCogs(bot=as_bot(fake=SimpleNamespace()))
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
    """A `/stock` landing during a news sweep is dropped, and the slot frees when it ends."""
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
    news_ai = cast("StockNewsAI", SimpleNamespace(generate=lambda _context: None))

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
        """Returns one quote, recording a market read that should never happen at all."""
        nonlocal called
        called = True
        return (_quote(),)

    monkeypatch.setattr(stock, "list_market_quotes", fake_list_market_quotes)
    cog = StockCogs(bot=as_bot(fake=SimpleNamespace()))
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

    assert (
        await view.interaction_check(interaction=as_interaction(fake=InteractionStub(user_id=1)))
        is True
    )
    assert await view.interaction_check(interaction=as_interaction(fake=intruder)) is False
    assert intruder.response.sent[0]["ephemeral"] is True
    assert "只有發起者" in intruder.response.sent[0]["content"]


async def test_stock_market_select_edits_public_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    """The market select hands the picked symbol and the panel's owner to the detail path."""
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
    stock_select = cast("StringSelect[StockMarketView]", view.stock_select)
    stock_select._selected_values = [BCAT_SYMBOL]

    await stock_select.callback(interaction=as_interaction(fake=interaction))

    assert selected == [BCAT_SYMBOL]
    assert interaction.user is not None
    assert owners == [interaction.user.id]
    assert interaction.response.deferred
    assert not interaction.response.deferred_ephemeral


async def test_stock_market_select_truncates_long_company_names() -> None:
    """The market dropdown keeps option labels inside Discord's limit."""
    view = StockMarketView(quotes=(_quote(name="長" * 128),), owner_id=1)
    stock_select = cast("StringSelect[StockMarketView]", view.stock_select)
    option = stock_select.options[0]

    assert len(option.label) == stock_views.SELECT_OPTION_LABEL_LIMIT
    assert option.label.startswith(f"{BCAT_SYMBOL} · 長")
    assert option.label.endswith("...")


def test_stock_market_embed_uses_board_attachment_for_rows() -> None:
    """The market embed keeps tabular rows out of Markdown text."""
    filename = market_board_filename(page_index=0)
    embed = build_market_embed(quotes=(_quote(),), board_filename=filename)

    assert embed.image.url == f"attachment://{filename}"
    assert embed.description is not None
    assert "市值" not in embed.description
    assert "100,000,000" not in embed.description


def test_stock_market_board_handles_large_market_caps() -> None:
    """A 兆-scale market cap renders on the board's fixed width instead of stretching a row."""
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
    """The board render cache keys on the quote fields it draws, so a moved price is a miss."""
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
    """The 7D chart cache keys on the tick tuple, so one cent of movement is a fresh render."""
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
    """Operate, news and back each repaint the panel's own message instead of adding a second."""

    async def fake_news(symbol: str) -> tuple[StockNewsView, ...]:
        """Returns no news, so the embed falls back to its placeholder line."""
        return ()

    async def fake_quotes() -> tuple[StockMarketQuote, ...]:
        """Returns one fake quote."""
        return (_quote(),)

    async def fake_detail(symbol: str, user_id: int, user_name: str) -> StockDetailViewData:
        """Returns a held-both-ways detail, checking the presser's identity reached the read."""
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
    await operate.callback(as_interaction(fake=operate_interaction))
    assert operate_interaction.response.deferred
    embed = operate_interaction.message.edits[0]["embed"]
    assert isinstance(embed, Embed)
    assert "股票代碼" in embed.description
    assert "股票代碼：BCAT" in embed.description
    assert "100.00 虛擬歡樂豆" in embed.description
    assert "目前持有：3股 | 目前做空：2股" in embed.description
    assert isinstance(operate_interaction.message.edits[0]["view"], StockActionView)

    news_interaction = InteractionStub()
    await news.callback(as_interaction(fake=news_interaction))
    assert "近期新聞" in news_interaction.response.sent[0]["embed"].title
    assert news_interaction.response.sent[0]["view"].owner_id == view.owner_id

    back_interaction = InteractionStub()
    await back.callback(as_interaction(fake=back_interaction))
    assert isinstance(back_interaction.response.sent[0]["view"], StockMarketView)
    assert back_interaction.response.sent[0]["view"].owner_id == view.owner_id


def test_stock_detail_embed_uses_localized_user_labels() -> None:
    """The detail embed labels its fields in Chinese rather than half-translated placeholders."""
    embed = build_stock_detail_embed(detail=_detail(), chart_filename="chart.png")

    field_names = {field.name for field in embed.fields}
    assert "目前操作使用者" in field_names
    assert "可用資金" in field_names
    assert "目前操作 user" not in field_names
    assert "操作 user 資金" not in field_names
    assert embed.description is not None
    assert "市值 `1億`" in embed.description


def test_stock_detail_embed_displays_large_share_counts_as_lots() -> None:
    """The detail embed reads a 兆-scale holding back as 張 and 股 rather than raw digits."""
    embed = build_stock_detail_embed(
        detail=_detail(long_shares=10_000_000_000_000, short_shares=1_234),
        chart_filename="chart.png",
    )

    field_values = "\n".join(str(field.value) for field in embed.fields)
    assert "持股數 `100億張`" in field_values
    assert "做空股數 `1張 234股`" in field_values


def test_stock_detail_embed_compacts_public_position_summary() -> None:
    """The shareholder field keeps the top three long holders and drops a short-only one."""
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
    """The recent-trade field keeps the first three legs and drops the tail."""
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
    """The action dropdown opens a modal of one quantity input, carrying the panel's owner id."""
    view = StockActionView(symbol=BCAT_SYMBOL, owner_id=1)
    child = next(
        child for child in view.children if getattr(child, "custom_id", "") == "stock:action"
    )
    assert isinstance(child, StringSelect)
    child._selected_values = [StockAction.SHORT.value]

    interaction = InteractionStub()
    await child.callback(interaction=as_interaction(fake=interaction))

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
        """Returns a failure after recording a settlement the owner gate should have refused."""
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

    await modal.submit_quantity(interaction=as_interaction(fake=intruder), raw_quantity="1")

    assert calls == []
    assert intruder.response.sent[0]["ephemeral"] is True
    assert "只有發起者" in intruder.response.sent[0]["content"]


async def test_stock_modal_reports_invalid_input_root_cause_in_public_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected quantity repaints the public message with the service's reason and a retry."""

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

    await modal.submit_quantity(interaction=as_interaction(fake=interaction), raw_quantity="abc")

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
    """A settled quantity repaints the public message with the receipt and the post-trade view."""

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

    await modal.submit_quantity(interaction=as_interaction(fake=interaction), raw_quantity="1")

    assert not interaction.response.deferred_ephemeral
    assert "交易完成" in interaction.message.edits[0]["embed"].title
    assert "錢包變化" in interaction.message.edits[0]["embed"].description
    assert "Wallet" not in interaction.message.edits[0]["embed"].description
    assert interaction.message.edits[0]["embed"].fields[0].name == "交易明細"
    assert isinstance(interaction.message.edits[0]["view"], StockPostTradeView)
    assert interaction.user is not None
    assert interaction.message.edits[0]["view"].owner_id == interaction.user.id


def test_failed_stock_settlement_title_does_not_depend_on_operation_id() -> None:
    """A FAILED settlement is titled a plain failure even though it carries an operation id."""
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
    """An edit onto a deleted panel message re-sends the result publicly and re-mints its file."""
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
        interaction=as_interaction(fake=interaction),
        embed=Embed(title="股票交易完成"),
        view=view,
        file=chart_file,
        message=as_message(fake=interaction.message),
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
    """An idle panel deletes its bound message through the shared cleanup, not `Message.delete`."""
    deleted: list[MessageStub] = []

    async def fake_delete(message: MessageStub) -> None:
        """Records delegated public-message deletion."""
        deleted.append(message)

    monkeypatch.setattr(owned_message_views, "delete_public_message", fake_delete)
    message = MessageStub()
    view = StockPublicView(owner_id=1)
    view.bind_message(message=as_message(fake=message))

    await view.on_timeout()

    assert deleted == [message]


def test_stock_readme_and_capability_metadata_are_covered() -> None:
    """`/stock` still declares the English picker description the docs are written around."""
    assert StockCogs.stock.description == "Open the simulated stock market."
