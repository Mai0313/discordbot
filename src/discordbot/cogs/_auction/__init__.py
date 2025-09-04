from nextcord.ext import commands

from .cog import AuctionCogs


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(AuctionCogs(bot))