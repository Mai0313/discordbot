"""Slash command entry point for the simulated stock market."""

import nextcord
from nextcord import Locale, Interaction
from nextcord.ext import commands

from discordbot.cogs._stock.views import StockMarketView, require_stock_user
from discordbot.cogs._games.cleanup import track_game_message
from discordbot.cogs._stock.database import list_market_quotes
from discordbot.cogs._stock.presentation import build_market_embed


class StockCogs(commands.Cog):
    """Provides the simulated stock market slash command."""

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the stock cog."""
        self.bot = bot

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
        quotes = await list_market_quotes()
        view = StockMarketView(quotes=quotes, owner_id=user.id)
        message = await interaction.followup.send(
            embed=build_market_embed(quotes=quotes), view=view, wait=True
        )
        view.bind_message(message=message)
        await track_game_message(message=message, user_name=user.name)


def setup(bot: commands.Bot) -> None:
    """Adds the stock cog to the bot."""
    bot.add_cog(StockCogs(bot), override=True)
