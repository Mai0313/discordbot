"""Localized, category-driven help command.

The command opens an ephemeral overview with a select menu. Picking a category
edits that same message in place to show the category's full command list, so
the landing view stays short instead of dumping every command at once.
"""

import nextcord
from nextcord import Locale, Interaction
from nextcord.ext import commands

from discordbot.cogs._help.views import HelpView


class HelpCogs(commands.Cog):
    """Provides the localized help slash command.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the HelpCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot

    @nextcord.slash_command(
        name="help",
        description="Show a guide on how to use this bot.",
        name_localizations={Locale.zh_TW: "使用說明", Locale.ja: "ヘルプ"},
        description_localizations={
            Locale.zh_TW: "顯示機器人的使用指南",
            Locale.ja: "ボットの使い方ガイドを表示します。",
        },
        nsfw=False,
    )
    async def help(self, interaction: Interaction) -> None:
        """Shows a category-driven guide on how to use this bot.

        Args:
            interaction: The interaction that triggered the command.
        """
        view = HelpView(
            locale=interaction.locale,
            requester_name=interaction.user.display_name,
            requester_avatar_url=interaction.user.display_avatar.url,
        )
        await interaction.response.send_message(
            embed=view.initial_embed(), view=view, ephemeral=True
        )
        view.bind_origin(interaction=interaction)


def setup(bot: commands.Bot) -> None:
    """Adds the HelpCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(HelpCogs(bot), override=True)
