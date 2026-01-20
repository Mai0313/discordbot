import nextcord
from nextcord import Locale, Interaction
from nextcord.ext import commands


class TemplateCogs(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message: nextcord.Message) -> None:
        # 忽略來自機器人的訊息
        if message.author.bot:
            return

        # 如果訊息內容是 "debug"，對該訊息按讚
        if message.content.lower() == "debug":
            await message.add_reaction(":face_with_symbols_over_mouth:")

    @nextcord.slash_command(
        name="ping",
        description="Check the bot's response time.",
        name_localizations={Locale.zh_TW: "延遲測試", Locale.ja: "ピングテスト"},
        description_localizations={
            Locale.zh_TW: "測試機器人的回應時間。",
            Locale.ja: "ボットの応答速度をテストします。",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def ping(self, interaction: Interaction) -> None:
        await interaction.response.defer()
        bot_latency = round(self.bot.latency * 1000, 2)  # 取得 API 延遲

        embed = nextcord.Embed(
            title=":ping_pong: Pong!",
            color=0x00FF00,  # 綠色
            timestamp=nextcord.utils.utcnow(),
        )
        embed.add_field(name="Bot Latency", value=f"`{bot_latency}ms`")
        embed.set_footer(
            text=f"Requested by {interaction.user.display_name}",
            icon_url=interaction.user.display_avatar.url,
        )

        await interaction.followup.send(embed=embed)


# 註冊 Cog
async def setup(bot: commands.Bot) -> None:
    bot.add_cog(TemplateCogs(bot), override=True)
