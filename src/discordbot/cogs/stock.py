"""Slash command entry point for the simulated stock market."""

import asyncio
from functools import cached_property

import logfire
import nextcord
from nextcord import Locale, Interaction
from pydantic import ValidationError
from nextcord.ext import commands

from discordbot.utils.llm import create_litellm_client
from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs._stock.news import StockNewsAI
from discordbot.cogs._stock.views import (
    StockMarketView,
    require_stock_user,
    build_market_message_payload,
)
from discordbot.cogs._stock.database import list_market_quotes, ensure_due_stock_news
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import track_public_message

_stock_news_refresh_task: asyncio.Task[None] | None = None
_stock_news_refresh_task_loop: asyncio.AbstractEventLoop | None = None


class StockCogs(commands.Cog):
    """Provides the simulated stock market slash command."""

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the stock cog."""
        self.bot = bot
        self.runtime_models = RuntimeModelCatalog()

    @cached_property
    def news_ai(self) -> StockNewsAI | None:
        """Optional AI news generator using the runtime OpenAI-compatible endpoint."""
        try:
            config = LLMConfig()
        except ValidationError:
            return None
        return StockNewsAI(
            client=create_litellm_client(config=config), model=self.runtime_models.fast_model
        )

    @nextcord.slash_command(
        name="stock",
        description="Open the simulated stock market.",
        name_localizations={Locale.zh_TW: "股票", Locale.ja: "株式"},
        description_localizations={
            Locale.zh_TW: "開啟模擬股票市場",
            Locale.ja: "シミュレーション株式市場を開きます。",
        },
        nsfw=False,
    )
    async def stock(self, interaction: Interaction) -> None:
        """Shows the public stock market list."""
        await interaction.response.defer()
        user = require_stock_user(interaction=interaction)
        news_ai = self.news_ai
        if news_ai is not None:
            _schedule_stock_news_refresh(news_ai=news_ai)
        quotes = await list_market_quotes(refresh_news=news_ai is None)
        view = StockMarketView(quotes=quotes, owner_id=user.id)
        embed, file = build_market_message_payload(quotes=quotes)
        message = await interaction.followup.send(
            embed=embed,
            view=view,
            wait=True,
            **embed_spacer_payload(
                embeds=[embed], is_edit=False, target=interaction, extra_files=[file]
            ),
        )
        view.bind_message(message=message)
        await track_public_message(message=message, user_name=user.name)


def _schedule_stock_news_refresh(news_ai: StockNewsAI) -> None:
    """Starts a background stock news refresh without delaying the market UI."""
    global _stock_news_refresh_task, _stock_news_refresh_task_loop  # noqa: PLW0603 -- process task de-dupe
    loop = asyncio.get_running_loop()
    if _stock_news_refresh_task_loop is not loop:
        _stock_news_refresh_task = None
        _stock_news_refresh_task_loop = loop
    if _stock_news_refresh_task is not None and not _stock_news_refresh_task.done():
        return
    task = asyncio.create_task(ensure_due_stock_news(news_provider=news_ai.generate))
    _stock_news_refresh_task = task
    task.add_done_callback(_finish_stock_news_refresh)


def _finish_stock_news_refresh(task: asyncio.Task[None]) -> None:
    """Clears the active background refresh slot and logs failures."""
    global _stock_news_refresh_task  # noqa: PLW0603 -- process task de-dupe
    if _stock_news_refresh_task is task:
        _stock_news_refresh_task = None
    _log_stock_news_refresh_failure(task=task)


def _log_stock_news_refresh_failure(task: asyncio.Task[None]) -> None:
    """Logs unexpected background stock news refresh failures."""
    try:
        task.result()
    except Exception:
        logfire.warn("Background stock news refresh failed", _exc_info=True)


def setup(bot: commands.Bot) -> None:
    """Adds the stock cog to the bot."""
    bot.add_cog(StockCogs(bot), override=True)
