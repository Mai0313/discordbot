"""Slash command entry point for the simulated stock market."""

import nextcord
from nextcord import Locale, Interaction
from nextcord.ext import commands

from discordbot.cogs._stock.views import StockMarketView
from discordbot.cogs._games.cleanup import schedule_game_message_delete
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
        quotes = await list_market_quotes()
        message = await interaction.followup.send(
            embed=build_market_embed(quotes=quotes), view=StockMarketView(quotes=quotes), wait=True
        )
        user_name = interaction.user.name if interaction.user is not None else None
        schedule_game_message_delete(message=message, user_name=user_name)


def setup(bot: commands.Bot) -> None:
    """Adds the stock cog to the bot."""
    bot.add_cog(StockCogs(bot), override=True)
