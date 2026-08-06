"""Discord surface for `/ping` and the fixed message-trigger reactions.

Despite the directory name this is a real, always-loaded cog rather than a scaffold to copy or
delete: `cogs/template/` holds a `cog.py`, so `DiscordBot._load_cogs_sync` takes it like any
other cog, and `/ping` is listed in `gen_reply/capabilities.md` as a user-facing command.

It registers one slash command and one listener, and nothing else. `/ping` reports the gateway
heartbeat latency in an embed, with name and description localized for `zh_TW` and `ja` like
every user-facing command here. `on_message` adds one fixed reaction to each of a handful of
exact phrases sent by a human. Neither half reads configuration, writes storage, or sits behind
a kill-switch, which is why the two share one file instead of each getting a cog of its own.
"""

import nextcord
from nextcord import Embed, Locale, Message, Interaction
from nextcord.ext import commands

from discordbot.utils.discord_embeds import embed_spacer_payload


class TemplateCogs(commands.Cog):
    """Provides simple message reactions and the ping slash command.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot):
        """Builds the cog with the bot handle `/ping` reads its latency from.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Adds a fixed reaction to the few exact phrases this cog watches for.

        The comparison is against the whole lowercased content, so a phrase buried in a longer
        sentence is not a trigger. Any bot's message is skipped, this bot's own replies included.
        As a cog listener it is dispatched alongside `DiscordBot.on_message` and every other
        cog's listener rather than replacing one, so nothing here can swallow a message.

        Args:
            message (Message): The message that was just sent in a visible channel.
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
    )
    async def ping(self, interaction: Interaction[commands.Bot]) -> None:
        """Answers with the bot's current gateway latency in an embed.

        The figure is nextcord's websocket heartbeat latency, not the round trip of this
        interaction, so it describes the connection rather than how long this command took. The
        embed goes out through `embed_spacer_payload`, which pins it to one rendered width so
        the card does not resize from one invocation to the next as the number changes.

        Args:
            interaction (Interaction[commands.Bot]): The `/ping` invocation to answer.
        """
        await interaction.response.defer()
        bot_latency = round(self.bot.latency * 1000, 2)

        embed = Embed(title=":ping_pong: Pong!", color=0x00FF00, timestamp=nextcord.utils.utcnow())
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
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(TemplateCogs(bot), override=True)
