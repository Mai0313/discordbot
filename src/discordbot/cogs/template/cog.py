"""Small utility cog for ping and simple message-trigger reactions."""

import nextcord
from nextcord import Embed, Locale, Message, Interaction
from nextcord.ext import commands

from discordbot.typings.colors import DISCORD_GREEN
from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.utils.discord_embeds import embed_spacer_payload


class TemplateCogs(commands.Cog):
    """Provides simple message reactions and the ping slash command.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the TemplateCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and handles specific triggers.

        Args:
            message: The message that was sent.
        """
        if message.author.bot:
            return

        if message.content.lower() == "debug":
            await message.add_reaction("🤬")
        if message.content.lower() == "可愛捏":
            await message.add_reaction("↖️")
        if message.content.lower() == "可爱捏":
            await message.add_reaction("↖️")

    @nextcord.slash_command(
        name="ping",
        description="Check the bot's response time.",
        name_localizations={Locale.zh_TW: "延遲測試", Locale.ja: "ピングテスト"},
        description_localizations={
            Locale.zh_TW: "測試機器人的回應時間",
            Locale.ja: "ボットの応答速度をテストします。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def ping(self, interaction: Interaction[commands.Bot]) -> None:
        """Checks the bot's response time.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        bot_latency = round(self.bot.latency * 1000, 2)

        embed = Embed(
            title=":ping_pong: Pong!", color=DISCORD_GREEN, timestamp=nextcord.utils.utcnow()
        )
        embed.add_field(name="Bot Latency", value=f"`{bot_latency}ms`")
        user = interaction.user
        if user is not None:
            embed.set_footer(
                text=f"Requested by {user.display_name}", icon_url=user.display_avatar.url
            )

        await interaction.followup.send(
            embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction)
        )


def setup(bot: commands.Bot) -> None:
    """Adds the TemplateCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(TemplateCogs(bot), override=True)
