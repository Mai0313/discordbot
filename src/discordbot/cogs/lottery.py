import nextcord
from nextcord import Locale, Interaction
from nextcord.ext import commands

from discordbot.sdk.yt_chat import YoutubeStream

from ._lottery.state import add_participant, get_participants
from ._lottery.views import LotteryMethodSelectionView
from ._lottery.embeds import add_participants_field
from ._lottery.models import LotteryData, LotteryParticipant


class LotteryCog(commands.Cog):
    """抽獎功能 Cog，負責註冊指令與協調各組件。"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @nextcord.slash_command(
        name="lottery",
        description="抽獎功能主選單",
        name_localizations={Locale.zh_TW: "抽獎", Locale.ja: "抽選"},
        description_localizations={
            Locale.zh_TW: "創建和管理抽獎活動",
            Locale.ja: "抽選イベントの作成と管理",
        },
        dm_permission=False,
    )
    async def lottery_main(self, interaction: Interaction) -> None:
        """顯示抽獎建立精靈。"""
        view = LotteryMethodSelectionView(self)
        embed = nextcord.Embed(title="🧰 抽獎建立精靈", color=0x00FF00)
        embed.add_field(
            name="步驟 1",
            value="從下方選擇報名方式（Discord 按鈕 或 YouTube 關鍵字）",
            inline=False,
        )
        embed.add_field(name="步驟 2", value="系統將開啟表單讓你填寫標題與描述", inline=False)

        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    async def fetch_youtube_participants(self, lottery_data: LotteryData) -> list[str]:
        """抓取 YouTube 報名名單並同步到系統參與者列表。"""
        yt_stream = YoutubeStream(url=lottery_data.youtube_url)
        registered_accounts = yt_stream.get_registered_accounts(lottery_data.youtube_keyword)
        for account in registered_accounts:
            participant = LotteryParticipant(id=account, name=account, source="youtube")
            add_participant(lottery_data.lottery_id, participant)
        return registered_accounts

    def build_status_embed(self, lottery_data: LotteryData) -> nextcord.Embed:
        """建立抽獎狀態 Embed。"""
        participants = get_participants(lottery_data.lottery_id)
        embed = nextcord.Embed(title="📊 抽獎活動狀態", color=0x0099FF)
        embed.add_field(name="活動標題", value=lottery_data.title, inline=False)
        embed.add_field(name="活動描述", value=lottery_data.description or "無", inline=False)
        embed.add_field(name="每次抽出", value=f"{lottery_data.draw_count} 人", inline=True)
        embed.add_field(name="發起人", value=lottery_data.creator_name, inline=True)
        if lottery_data.youtube_url:
            embed.add_field(name="YouTube直播", value=lottery_data.youtube_url, inline=False)
        if lottery_data.youtube_keyword:
            embed.add_field(name="報名關鍵字", value=lottery_data.youtube_keyword, inline=True)
        if participants:
            add_participants_field(embed, participants)
        else:
            embed.add_field(name="參與者", value="目前沒有參與者", inline=False)
        return embed


async def setup(bot: commands.Bot) -> None:
    """Register the lottery cog with the bot."""
    bot.add_cog(LotteryCog(bot), override=True)
