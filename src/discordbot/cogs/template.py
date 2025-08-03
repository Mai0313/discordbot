import time

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
            await message.add_reaction("🤬")

        # # 繼續處理其他命令
        # await self.bot.process_commands(message)

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
        start_time = time.time()  # 紀錄開始時間
        await interaction.response.defer()  # 先延遲回應，讓我們測試傳輸時間
        end_time = time.time()  # 紀錄結束時間

        latency = round((end_time - start_time) * 1000, 2)  # 計算毫秒延遲
        api_latency = round(self.bot.latency * 1000, 2)  # 取得 API 延遲

        # 根據使用者的語言來顯示不同的embed訊息
        locale_titles = {Locale.zh_TW: "🏓 延遲測試", Locale.ja: "🏓 ピングテスト"}

        locale_fields = {
            Locale.zh_TW: [
                ("訊息延遲", f"`{latency}ms`", True),
                ("API 延遲", f"`{api_latency}ms`", True),
            ],
            Locale.ja: [
                ("メッセージ遅延", f"`{latency}ms`", True),
                ("API遅延", f"`{api_latency}ms`", True),
            ],
        }

        # 預設為英文
        title = "🏓 Pong!"
        fields = [
            ("Message Latency", f"`{latency}ms`", True),
            ("API Latency", f"`{api_latency}ms`", True),
        ]

        # 如果使用者的語言有對應的本地化訊息，就使用該語言
        if interaction.locale in locale_titles:
            title = locale_titles[interaction.locale]
            fields = locale_fields[interaction.locale]

        # 建立 embed
        embed = nextcord.Embed(
            title=title,
            color=0x00FF00,  # 綠色
            timestamp=nextcord.utils.utcnow(),
        )

        for field_name, field_value, inline in fields:
            embed.add_field(name=field_name, value=field_value, inline=inline)

        embed.set_footer(
            text=f"Requested by {interaction.user.display_name}",
            icon_url=interaction.user.display_avatar.url,
        )

        await interaction.followup.send(embed=embed)


# 註冊 Cog
async def setup(bot: commands.Bot) -> None:
    bot.add_cog(TemplateCogs(bot), override=True)
