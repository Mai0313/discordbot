import random

import nextcord
from nextcord import Interaction, Locale, SlashOption
from nextcord.ext import commands


class FunCogs(commands.Cog):
    """Collection of small fun commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    def _roll_dice(self, sides: int) -> int:
        """Return a random number between 1 and `sides`."""
        return random.randint(1, sides)

    @nextcord.slash_command(
        name="dice",
        description="Roll a dice with the given number of sides.",
        name_localizations={Locale.zh_TW: "擲骰子", Locale.ja: "サイコロ"},
        description_localizations={
            Locale.zh_TW: "擲一顆有指定面數的骰子。",
            Locale.ja: "指定された面数のサイコロを振ります。",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def dice(
        self,
        interaction: Interaction,
        sides: int = SlashOption(
            description="Number of sides",
            required=False,
            default=6,
            min_value=2,
            max_value=100,
        ),
    ) -> None:
        """Roll a dice and reply with the result."""
        await interaction.response.defer()
        result = self._roll_dice(sides)
        await interaction.followup.send(f"🎲 You rolled **{result}** (1-{sides})")


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(FunCogs(bot), override=True)
