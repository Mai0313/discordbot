"""Slash command entry point for the simulated stock market."""

import asyncio
from functools import cached_property

from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import Locale, Interaction
from pydantic import ValidationError
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs._stock.news import StockNewsAI
from discordbot.cogs._stock.views import (
    StockMarketView,
    require_stock_user,
    build_market_message_payload,
)
from discordbot.cogs._stock.database import list_market_quotes, ensure_due_stock_news
from discordbot.utils.message_cleanup import track_public_message


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
            client=AsyncOpenAI(base_url=config.base_url, api_key=config.api_key),
            model=self.runtime_models.fast_model,
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
        message = await interaction.followup.send(embed=embed, file=file, view=view, wait=True)
        view.bind_message(message=message)
        await track_public_message(message=message, user_name=user.name)


def _schedule_stock_news_refresh(news_ai: StockNewsAI) -> None:
    """Starts a background stock news refresh without delaying the market UI."""
    task = asyncio.create_task(ensure_due_stock_news(news_provider=news_ai.generate))
    task.add_done_callback(_log_stock_news_refresh_failure)


def _log_stock_news_refresh_failure(task: asyncio.Task[None]) -> None:
    """Logs unexpected background stock news refresh failures."""
    try:
        task.result()
    except Exception:
        logfire.warn("Background stock news refresh failed", _exc_info=True)


def setup(bot: commands.Bot) -> None:
    """Adds the stock cog to the bot."""
    bot.add_cog(StockCogs(bot), override=True)
