from src.cogs.fun import FunCogs
from nextcord.ext import commands


def test_roll_dice_in_range():
    bot = commands.Bot(command_prefix="!")
    cog = FunCogs(bot)
    result = cog._roll_dice(6)
    assert 1 <= result <= 6
