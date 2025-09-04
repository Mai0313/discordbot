"""
拍賣系統主文件

此文件現在只負責導入和設置拍賣系統模組。
所有功能已經拆分到 _auction 資料夾中的各個模組。
"""

from nextcord.ext import commands
from ._auction import AuctionCogs


async def setup(bot: commands.Bot) -> None:
    """設置拍賣系統 Cog"""
    bot.add_cog(AuctionCogs(bot))