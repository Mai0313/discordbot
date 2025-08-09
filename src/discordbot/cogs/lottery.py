import asyncio
import secrets
from datetime import datetime
from collections import defaultdict

import nextcord
from nextcord import User, Locale, Member, Interaction
from pydantic import BaseModel
from nextcord.ext import commands

from discordbot.sdk.yt_chat import YoutubeStream

# 全局變數來存儲抽獎數據（使用 defaultdict 自動初始化）
# guild_id -> LotteryData（每個伺服器同時間僅允許一個活動）
active_lotteries: dict[int, "LotteryData"] = {}
# lottery_id -> LotteryData（用於依 ID 直接查找，避免掃描）
lotteries_by_id: dict[int, "LotteryData"] = {}
# lottery_id -> 參與者列表
lottery_participants: defaultdict[int, list["LotteryParticipant"]] = defaultdict(list)
# lottery_id -> 中獎者列表
lottery_winners: defaultdict[int, list["LotteryParticipant"]] = defaultdict(list)
# 簡單的ID生成器
next_lottery_id = 1


class LotteryParticipant(BaseModel):
    """抽獎參與者數據類"""

    id: str  # Discord用戶ID或YouTube名稱
    name: str  # 顯示名稱
    source: str  # "discord" 或 "youtube"


class LotteryData(BaseModel):
    """抽獎活動數據模型"""

    lottery_id: int
    guild_id: int
    title: str
    description: str
    creator_id: int
    creator_name: str
    created_at: datetime
    is_active: bool
    registration_method: str  # "reaction" 或 "youtube"
    youtube_url: str | None = None
    youtube_keyword: str | None = None
    reaction_emoji: str = "🎉"
    reaction_message_id: int | None = None


# 簡化的數據操作函數（替代數據庫）


def create_lottery(lottery_data: dict) -> int:
    """創建抽獎活動"""
    global next_lottery_id
    lottery_id = next_lottery_id
    next_lottery_id += 1

    # 創建LotteryData對象
    lottery = LotteryData(
        lottery_id=lottery_id,
        guild_id=lottery_data["guild_id"],
        title=lottery_data["title"],
        description=lottery_data.get("description", ""),
        creator_id=lottery_data["creator_id"],
        creator_name=lottery_data["creator_name"],
        created_at=datetime.now(),
        is_active=True,
        registration_method=lottery_data["registration_method"],
        youtube_url=lottery_data.get("youtube_url"),
        youtube_keyword=lottery_data.get("youtube_keyword"),
        reaction_emoji=lottery_data.get("reaction_emoji", "🎉"),
        reaction_message_id=lottery_data.get("reaction_message_id"),
    )

    # 存儲到全局變數
    active_lotteries[lottery_data["guild_id"]] = lottery
    lotteries_by_id[lottery_id] = lottery
    # defaultdict 會自動初始化空列表，無需手動設置

    return lottery_id


def update_reaction_message_id(lottery_id: int, message_id: int) -> None:
    """更新反應消息ID（透過 ID 直接更新目前活動）"""
    lottery = lotteries_by_id.get(lottery_id)
    if lottery is not None:
        lottery.reaction_message_id = message_id


def get_active_lottery(guild_id: int) -> "LotteryData | None":
    """獲取活躍的抽獎活動"""
    return active_lotteries.get(guild_id)


def add_participant(lottery_id: int, participant: LotteryParticipant) -> bool:
    """添加參與者，返回是否成功添加（防止跨平台重複報名和平台不匹配）"""
    # 首先驗證參與者來源是否與抽獎註冊方式匹配
    lottery_data = lotteries_by_id.get(lottery_id)

    if not lottery_data:
        return False  # 抽獎不存在

    # 檢查平台匹配：Discord用戶只能參與reaction抽獎，YouTube用戶只能參與youtube抽獎
    if participant.source == "discord" and lottery_data.registration_method != "reaction":
        return False  # Discord用戶嘗試參與非reaction抽獎

    if participant.source == "youtube" and lottery_data.registration_method != "youtube":
        return False  # YouTube用戶嘗試參與非youtube抽獎

    # 檢查用戶是否已經以其他方式報名
    # defaultdict 自動創建空列表
    for existing in lottery_participants[lottery_id]:
        if existing.id == participant.id:
            # 來源相同允許（重複操作），來源不同不允許（跨平台重複報名）
            return existing.source == participant.source

    # 添加新參與者
    lottery_participants[lottery_id].append(participant)
    return True


def get_participants(lottery_id: int) -> list[LotteryParticipant]:
    """獲取所有參與者"""
    return lottery_participants[lottery_id]  # defaultdict 自動返回空列表


def add_winner(lottery_id: int, participant: LotteryParticipant) -> None:
    """記錄中獎者"""
    lottery_winners[lottery_id].append(participant)  # defaultdict 自動創建空列表


def remove_participant(lottery_id: int, participant_id: str, source: str) -> None:
    """移除參與者"""
    # defaultdict 保證列表存在，直接操作即可
    lottery_participants[lottery_id] = [
        p
        for p in lottery_participants[lottery_id]
        if not (p.id == participant_id and p.source == source)
    ]


def reset_lottery_participants(lottery_id: int) -> None:
    """重置抽獎參與者（清除中獎記錄，恢復所有參與者）"""
    lottery_winners[lottery_id].clear()  # defaultdict 保證列表存在，直接清空


def close_lottery(lottery_id: int) -> None:
    """關閉抽獎活動"""
    # 找到並關閉抽獎
    for guild_id, lottery in list(active_lotteries.items()):
        if lottery.lottery_id == lottery_id:
            lottery.is_active = False
            del active_lotteries[guild_id]
            break
    lotteries_by_id.pop(lottery_id, None)


def split_participants_by_source(
    participants: list["LotteryParticipant"],
) -> tuple[list["LotteryParticipant"], list["LotteryParticipant"]]:
    discord_users = [p for p in participants if p.source == "discord"]
    youtube_users = [p for p in participants if p.source == "youtube"]
    return discord_users, youtube_users


def add_participants_fields_to_embed(
    embed: nextcord.Embed, participants: list["LotteryParticipant"]
) -> None:
    discord_users, youtube_users = split_participants_by_source(participants)

    if discord_users:
        discord_names_str = ", ".join([user.name for user in discord_users])
        embed.add_field(
            name=f"Discord 參與者 ({len(discord_users)} 人)", value=discord_names_str, inline=False
        )

    if youtube_users:
        youtube_names_str = ", ".join([user.name for user in youtube_users])
        embed.add_field(
            name=f"YouTube 參與者 ({len(youtube_users)} 人)", value=youtube_names_str, inline=False
        )

    embed.add_field(name="總參與人數", value=f"**{len(participants)}** 人", inline=False)


class LotteryCreateModal(nextcord.ui.Modal):
    """創建抽獎活動的表單"""

    def __init__(self, registration_method: str):
        super().__init__(title="創建抽獎活動")
        self.registration_method = registration_method

        self.title_input = nextcord.ui.TextInput(
            label="抽獎標題", placeholder="請輸入抽獎活動標題...", max_length=100, required=True
        )
        self.add_item(self.title_input)

        self.description_input = nextcord.ui.TextInput(
            label="抽獎描述",
            placeholder="請輸入抽獎活動描述...",
            style=nextcord.TextInputStyle.paragraph,
            max_length=1000,
            required=False,
        )
        self.add_item(self.description_input)

        if registration_method == "youtube":
            self.youtube_url_input = nextcord.ui.TextInput(
                label="YouTube 直播網址", placeholder="請輸入YouTube直播網址...", required=True
            )
            self.add_item(self.youtube_url_input)

            self.keyword_input = nextcord.ui.TextInput(
                label="報名關鍵字", placeholder="請輸入YouTube聊天室報名關鍵字...", required=True
            )
            self.add_item(self.keyword_input)

    async def callback(self, interaction: Interaction) -> None:
        """處理表單提交"""
        await interaction.response.defer()

        try:
            # 檢查是否在伺服器中執行命令
            if interaction.guild is None:
                await interaction.followup.send(
                    "❌ 抽獎功能只能在伺服器中使用，不支援私人訊息!", ephemeral=True
                )
                return

            # 檢查是否已有活躍的抽獎
            active_lottery = get_active_lottery(interaction.guild.id)
            if active_lottery:
                await interaction.followup.send(
                    f"目前已有活躍的抽獎活動：**{active_lottery.title}**", ephemeral=True
                )
                return

            lottery_data = {
                "guild_id": interaction.guild.id,
                "title": self.title_input.value,
                "description": self.description_input.value or "",
                "creator_id": interaction.user.id,
                "creator_name": interaction.user.display_name,
                "registration_method": self.registration_method,
            }

            if hasattr(self, "youtube_url_input") and self.youtube_url_input.value:
                lottery_data["youtube_url"] = self.youtube_url_input.value
            if hasattr(self, "keyword_input") and self.keyword_input.value:
                lottery_data["youtube_keyword"] = self.keyword_input.value

            # 直接創建抽獎活動
            lottery_id = create_lottery(lottery_data)
            lottery_data["lottery_id"] = lottery_id

            # 創建回應embed
            embed = nextcord.Embed(title="🎉 抽獎活動已創建!", color=0x00FF00)
            embed.add_field(name="活動標題", value=lottery_data["title"], inline=False)
            embed.add_field(
                name="活動描述", value=lottery_data["description"] or "無", inline=False
            )
            embed.add_field(
                name="註冊方式", value=lottery_data["registration_method"], inline=True
            )

            if lottery_data["registration_method"] == "reaction":
                embed.add_field(
                    name="Discord報名方式", value="對此訊息加上 🎉 表情符號即可報名", inline=False
                )
            elif lottery_data["registration_method"] == "youtube":
                embed.add_field(
                    name="YouTube直播", value=lottery_data["youtube_url"], inline=False
                )
                embed.add_field(
                    name="報名關鍵字",
                    value=f"在聊天室發送包含「{lottery_data['youtube_keyword']}」的訊息",
                    inline=False,
                )

            embed.add_field(name="使用說明", value="使用 `/lottery start` 開始抽獎", inline=False)

            message = await interaction.followup.send(embed=embed, wait=True)

            # 如果是反應抽獎，添加反應並記錄消息ID
            if lottery_data["registration_method"] == "reaction":
                await message.add_reaction("🎉")
                update_reaction_message_id(lottery_id, message.id)

        except Exception as e:
            await interaction.followup.send(f"創建抽獎活動時發生錯誤：{e!s}", ephemeral=True)


class LotterySpinView(nextcord.ui.View):
    """抽獎轉盤視圖"""

    def __init__(self, lottery_data: LotteryData, participants: list[LotteryParticipant]):
        super().__init__(timeout=300)
        self.lottery_data = lottery_data
        self.participants = participants

    @nextcord.ui.button(label="🎰 開始抽獎", style=nextcord.ButtonStyle.primary)
    async def spin_lottery(self, button: nextcord.ui.Button, interaction: Interaction) -> None:
        """執行抽獎轉盤動畫"""
        if interaction.user.id != self.lottery_data.creator_id:
            await interaction.response.send_message("只有抽獎發起人可以開始抽獎!", ephemeral=True)
            return

        if not self.participants:
            await interaction.response.send_message("沒有參與者，無法進行抽獎!", ephemeral=True)
            return

        await interaction.response.defer()

        # 創建轉盤動畫
        embed = nextcord.Embed(title="🎰 抽獎進行中...", color=0x00FF00)
        embed.add_field(name="參與人數", value=f"{len(self.participants)} 人", inline=False)

        # 轉盤動畫序列
        spin_emojis = ["🎰", "🎲", "🎯", "🎪", "🎭", "🎨", "🎼", "🎵"]

        await interaction.followup.send(embed=embed)

        # 轉盤動畫效果
        for _i in range(15):
            emoji = secrets.choice(spin_emojis)
            embed = nextcord.Embed(title=f"{emoji} 抽獎進行中...", color=0x00FF00)
            embed.add_field(name="參與人數", value=f"{len(self.participants)} 人", inline=False)
            embed.add_field(name="⏳", value="正在隨機選擇中獎者...", inline=False)
            await interaction.edit_original_message(embed=embed)
            await asyncio.sleep(0.3)

        # 選擇中獎者
        winner = secrets.choice(self.participants)
        self.participants.remove(winner)

        # 記錄中獎者
        add_winner(self.lottery_data.lottery_id, winner)

        # 顯示結果
        result_embed = nextcord.Embed(title="🎉 恭喜中獎!", color=0xFFD700)
        result_embed.add_field(name="中獎者", value=f"**{winner.name}**", inline=False)
        result_embed.add_field(
            name="來源", value="Discord" if winner.source == "discord" else "YouTube", inline=True
        )
        result_embed.add_field(
            name="剩餘參與者", value=f"{len(self.participants)} 人", inline=True
        )

        if not self.participants:
            result_embed.add_field(name="⚠️", value="所有參與者都已抽完!", inline=False)
            button.disabled = True
            self.stop()

        await interaction.edit_original_message(
            embed=result_embed, view=self if self.participants else None
        )

    @nextcord.ui.button(label="📊 查看參與者", style=nextcord.ButtonStyle.secondary)
    async def view_participants(
        self, button: nextcord.ui.Button, interaction: Interaction
    ) -> None:
        """查看所有參與者"""
        if not self.participants:
            await interaction.response.send_message("目前沒有參與者", ephemeral=True)
            return

        embed = nextcord.Embed(title="📊 抽獎參與者名單", color=0x0099FF)
        add_participants_fields_to_embed(embed, self.participants)

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @nextcord.ui.button(label="🔄 重新抽獎", style=nextcord.ButtonStyle.secondary, emoji="🔄")
    async def reset_lottery(self, button: nextcord.ui.Button, interaction: Interaction) -> None:
        """重新抽獎（重置中獎記錄，恢復所有參與者）"""
        if interaction.user.id != self.lottery_data.creator_id:
            await interaction.response.send_message("只有抽獎發起人可以重新抽獎!", ephemeral=True)
            return

        # 重置中獎記錄
        reset_lottery_participants(self.lottery_data.lottery_id)

        # 重新獲取所有參與者（恢復被移除的中獎者）
        all_participants = get_participants(self.lottery_data.lottery_id)
        self.participants = all_participants

        embed = nextcord.Embed(title="🔄 抽獎已重置!", color=0x00FF00)
        embed.add_field(name="活動", value=self.lottery_data.title, inline=False)
        embed.add_field(name="恢復參與者", value=f"{len(self.participants)} 人", inline=True)
        embed.add_field(name="狀態", value="所有參與者已恢復，可以重新開始抽獎", inline=False)

        await interaction.response.edit_message(embed=embed, view=self)

    @nextcord.ui.button(label="❌ 結束抽獎", style=nextcord.ButtonStyle.danger)
    async def close_lottery(self, button: nextcord.ui.Button, interaction: Interaction) -> None:
        """結束抽獎活動"""
        if interaction.user.id != self.lottery_data.creator_id:
            await interaction.response.send_message("只有抽獎發起人可以結束抽獎!", ephemeral=True)
            return

        close_lottery(self.lottery_data.lottery_id)

        embed = nextcord.Embed(title="🔒 抽獎活動已結束", color=0xFF0000)
        embed.add_field(name="活動", value=self.lottery_data.title, inline=False)
        embed.add_field(name="發起人", value=self.lottery_data.creator_name, inline=True)

        # 禁用所有按鈕
        for item in self.children:
            item.disabled = True

        await interaction.response.edit_message(embed=embed, view=self)
        self.stop()


class LotteryCog(commands.Cog):
    """抽獎功能Cog"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # -------- Helper methods (避免重複邏輯) --------
    async def _ensure_no_active_lottery(self, interaction: Interaction) -> bool:
        active_lottery = get_active_lottery(interaction.guild.id)
        if active_lottery:
            await interaction.response.send_message(
                f"目前已有活躍的抽獎活動：**{active_lottery.title}**", ephemeral=True
            )
            return False
        return True

    async def _get_active_lottery_or_reply(self, interaction: Interaction) -> "LotteryData | None":
        active_lottery = get_active_lottery(interaction.guild.id)
        if not active_lottery:
            await interaction.response.send_message("目前沒有活躍的抽獎活動", ephemeral=True)
            return None
        return active_lottery

    async def _open_create_modal(self, interaction: Interaction, method: str) -> None:
        if not await self._ensure_no_active_lottery(interaction):
            return
        modal = LotteryCreateModal(method)
        await interaction.response.send_modal(modal)

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
        """抽獎功能主選單"""
        pass

    @lottery_main.subcommand(name="create_reaction", description="創建Discord表情符號抽獎")
    async def create_reaction_lottery(self, interaction: Interaction) -> None:
        """創建Discord表情符號抽獎"""
        await self._open_create_modal(interaction, "reaction")

    @lottery_main.subcommand(name="create_youtube", description="創建YouTube聊天室關鍵字抽獎")
    async def create_youtube_lottery(self, interaction: Interaction) -> None:
        """創建YouTube聊天室關鍵字抽獎"""
        await self._open_create_modal(interaction, "youtube")

    @lottery_main.subcommand(name="start", description="開始抽獎")
    async def start_lottery(self, interaction: Interaction) -> None:
        """開始抽獎"""
        active_lottery = await self._get_active_lottery_or_reply(interaction)
        if not active_lottery:
            return

        if interaction.user.id != active_lottery.creator_id:
            await interaction.response.send_message("只有抽獎發起人可以開始抽獎!", ephemeral=True)
            return

        # 如果是YouTube抽獎，需要先獲取YouTube參與者，這需要時間所以先defer
        if active_lottery.registration_method == "youtube":
            await interaction.response.defer()
            await self._fetch_youtube_participants(active_lottery, interaction)

        # 獲取所有參與者
        participants = get_participants(active_lottery.lottery_id)

        if not participants:
            if active_lottery.registration_method == "youtube":
                await interaction.followup.send("沒有參與者，無法開始抽獎!", ephemeral=True)
            else:
                await interaction.response.send_message(
                    "沒有參與者，無法開始抽獎!", ephemeral=True
                )
            return

        # 創建抽獎轉盤界面
        view = LotterySpinView(active_lottery, participants)

        embed = nextcord.Embed(title="🎰 抽獎控制台", color=0x00FF00)
        embed.add_field(name="活動", value=active_lottery.title, inline=False)
        embed.add_field(name="參與人數", value=f"{len(participants)} 人", inline=True)
        embed.add_field(name="註冊方式", value=active_lottery.registration_method, inline=True)

        if active_lottery.registration_method == "youtube":
            await interaction.followup.send(embed=embed, view=view)
        else:
            await interaction.response.send_message(embed=embed, view=view)

    @lottery_main.subcommand(name="status", description="查看抽獎狀態")
    async def lottery_status(self, interaction: Interaction) -> None:
        """查看抽獎狀態"""
        await interaction.response.defer(ephemeral=True)

        active_lottery = get_active_lottery(interaction.guild.id)
        if not active_lottery:
            await interaction.followup.send("目前沒有活躍的抽獎活動", ephemeral=True)
            return

        participants = get_participants(active_lottery.lottery_id)

        embed = nextcord.Embed(title="📊 抽獎活動狀態", color=0x0099FF)
        embed.add_field(name="活動標題", value=active_lottery.title, inline=False)
        embed.add_field(name="活動描述", value=active_lottery.description or "無", inline=False)
        embed.add_field(name="發起人", value=active_lottery.creator_name, inline=True)
        embed.add_field(name="註冊方式", value=active_lottery.registration_method, inline=True)
        embed.add_field(name="目前參與人數", value=f"{len(participants)} 人", inline=True)

        if active_lottery.youtube_url:
            embed.add_field(name="YouTube直播", value=active_lottery.youtube_url, inline=False)
        if active_lottery.youtube_keyword:
            embed.add_field(name="報名關鍵字", value=active_lottery.youtube_keyword, inline=True)

        # 顯示參與者名單（完整顯示所有參與者）
        if participants:
            add_participants_fields_to_embed(embed, participants)
        else:
            embed.add_field(name="參與者", value="目前沒有參與者", inline=False)

        await interaction.followup.send(embed=embed, ephemeral=True)

    async def _fetch_youtube_participants(
        self, lottery_data: LotteryData, interaction: Interaction
    ) -> None:
        """從YouTube聊天室獲取參與者"""
        try:
            if not lottery_data.youtube_url or not lottery_data.youtube_keyword:
                return

            await interaction.followup.send("正在從YouTube聊天室獲取參與者...", ephemeral=True)

            yt_stream = YoutubeStream(url=lottery_data.youtube_url)
            registered_accounts = yt_stream.get_registered_accounts(lottery_data.youtube_keyword)

            for account_name in registered_accounts:
                participant = LotteryParticipant(
                    id=account_name,  # YouTube使用顯示名稱作為ID
                    name=account_name,
                    source="youtube",
                )
                add_participant(lottery_data.lottery_id, participant)

            await interaction.followup.send(
                f"已從YouTube聊天室獲取 {len(registered_accounts)} 位參與者", ephemeral=True
            )

        except Exception as e:
            await interaction.followup.send(f"獲取YouTube參與者時發生錯誤：{e!s}", ephemeral=True)

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: nextcord.Reaction, user: Member | User) -> None:
        """處理Discord反應報名"""
        lottery = _get_reaction_lottery_or_none(reaction)
        if lottery is None:
            return

        if isinstance(user, (Member, User)) and not getattr(user, "bot", False):
            participant = LotteryParticipant(
                id=str(user.id), name=user.display_name, source="discord"
            )
            success = add_participant(lottery.lottery_id, participant)
            if not success:
                await reaction.remove(user)

    @commands.Cog.listener()
    async def on_reaction_remove(self, reaction: nextcord.Reaction, user: Member | User) -> None:
        """處理Discord反應取消報名"""
        lottery = _get_reaction_lottery_or_none(reaction)
        if lottery is None:
            return

        if isinstance(user, (Member, User)) and not getattr(user, "bot", False):
            remove_participant(lottery.lottery_id, str(user.id), "discord")


def _get_reaction_lottery_or_none(reaction: nextcord.Reaction) -> "LotteryData | None":
    if str(reaction.emoji) != "🎉":
        return None
    if reaction.message.guild is None:
        return None
    active_lottery = get_active_lottery(reaction.message.guild.id)
    if not active_lottery or active_lottery.registration_method != "reaction":
        return None
    if active_lottery.reaction_message_id != reaction.message.id:
        return None
    return active_lottery


async def setup(bot: commands.Bot) -> None:
    """Register the reply generation cog with the bot.

    Args:
        bot (commands.Bot): The bot instance to which the cog will be added.
    """
    bot.add_cog(LotteryCog(bot), override=True)
