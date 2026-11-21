"""Small utility cog for ping and simple message-trigger reactions."""

import nextcord
from nextcord import Embed, Locale, Message, Interaction
from nextcord.ext import commands

from discordbot.typings.colors import DISCORD_GREEN
from discordbot.utils.reactions import update_reaction
from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.utils.discord_embeds import embed_spacer_payload


class TemplateCogs(commands.Cog):
    """Provides simple message reactions and the ping slash command.

    Attributes:
        bot (commands.Bot): The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the cog.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Adds trigger reactions to matching non-bot messages.

        Args:
            message (Message): The message that was sent.
        """
        if message.author.bot:
            return

        trigger_reactions = {"debug": "🤬", "可愛捏": "↖️", "可爱捏": "↖️"}
        emoji = trigger_reactions.get(message.content.lower())
        if emoji is None:
            return
        await update_reaction(message=message, bot_user=None, emoji=emoji)

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
        """Replies with the bot's gateway latency.

        Args:
            interaction (Interaction[commands.Bot]): The interaction that triggered the command.
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
    """Adds `TemplateCogs` to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(TemplateCogs(bot), override=True)
