"""拍賣系統 - 主要入口文件"""

from nextcord.ext import commands

from ._auction import AuctionCogs


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(AuctionCogs(bot))