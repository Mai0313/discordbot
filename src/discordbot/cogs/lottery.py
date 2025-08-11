import secrets
from datetime import datetime
import contextlib
from collections import defaultdict

import nextcord
from nextcord import User, Locale, Member, Interaction
from pydantic import BaseModel
from nextcord.ext import commands

from discordbot.sdk.yt_chat import YoutubeStream

# 全局變數來存儲抽獎數據（使用 defaultdict 自動初始化）
# lottery_id -> LotteryData（用於依 ID 直接查找）
lotteries_by_id: dict[int, "LotteryData"] = {}
# guild_id -> 當前活躍的 LotteryData（相容舊版測試）
active_lotteries: dict[int, "LotteryData"] = {}
# lottery_id -> 參與者列表
lottery_participants: defaultdict[int, list["LotteryParticipant"]] = defaultdict(list)
# lottery_id -> 中獎者列表
lottery_winners: defaultdict[int, list["LotteryParticipant"]] = defaultdict(list)
# message_id -> lottery_id（建立訊息對應抽獎）
message_to_lottery_id: dict[int, int] = {}
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
    # 每次抽出人數（預設 1）
    draw_count: int = 1


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
        draw_count=max(1, int(lottery_data.get("draw_count", 1) or 1)),
    )

    # 存儲到全局變數（允許同時存在多個抽獎）
    lotteries_by_id[lottery_id] = lottery
    # 註冊到活躍清單（相容舊版測試）
    active_lotteries[lottery.guild_id] = lottery
    # defaultdict 會自動初始化空列表，無需手動設置

    return lottery_id


def update_reaction_message_id(lottery_id: int, message_id: int) -> None:
    """更新反應消息ID（透過 ID 直接更新目前活動）"""
    lottery = lotteries_by_id.get(lottery_id)
    if lottery is not None:
        lottery.reaction_message_id = message_id
        message_to_lottery_id[message_id] = lottery_id


def get_lottery_by_message_id(message_id: int) -> "LotteryData | None":
    """由建立訊息ID獲取抽獎活動資料。"""
    lottery_id = message_to_lottery_id.get(message_id)
    return lotteries_by_id.get(lottery_id) if lottery_id is not None else None


def get_active_lottery(guild_id: int) -> "LotteryData | None":
    """取得伺服器當前活躍抽獎（供相容測試使用）。"""
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
    lottery = lotteries_by_id.pop(lottery_id, None)
    if lottery is not None:
        lottery.is_active = False
        if lottery.reaction_message_id is not None:
            message_to_lottery_id.pop(lottery.reaction_message_id, None)
        # 清除活躍清單中的映射（若尚存在）
        if active_lotteries.get(lottery.guild_id) is lottery:
            active_lotteries.pop(lottery.guild_id, None)


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

    # 顯示總參與人數（相容測試期待）
    embed.add_field(name="總參與人數", value=f"{len(participants)} 人", inline=False)


class LotteryCreateModal(nextcord.ui.Modal):
    """創建抽獎活動的表單"""

    def __init__(self, registration_method: str):
        super().__init__(title="創建抽獎活動")
        self.registration_method = registration_method

        self.title_input = nextcord.ui.TextInput(
            label="抽獎標題", placeholder="請輸入抽獎活動標題...", max_length=100, required=True
        )
        self.add_item(self.title_input)

        # 每次抽出人數
        self.draw_count_input = nextcord.ui.TextInput(
            label="每次抽出人數", placeholder="預設 1", required=False, max_length=3
        )
        self.add_item(self.draw_count_input)

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

            # 允許同一伺服器同時存在多個抽獎

            lottery_data = {
                "guild_id": interaction.guild.id,
                "title": self.title_input.value,
                "description": self.description_input.value or "",
                "creator_id": interaction.user.id,
                "creator_name": interaction.user.display_name,
                "registration_method": self.registration_method,
            }

            # 解析每次抽出人數
            try:
                if self.draw_count_input.value:
                    dc_val = int(str(self.draw_count_input.value).strip())
                    lottery_data["draw_count"] = dc_val if 1 <= dc_val <= 100 else 1
                else:
                    lottery_data["draw_count"] = 1
            except Exception:
                lottery_data["draw_count"] = 1

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
            embed.add_field(
                name="每次抽出人數", value=f"{lottery_data['draw_count']} 人", inline=True
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

            # 說明改為以反應操作：✅ 開始、📊 狀態
            embed.add_field(
                name="使用說明",
                value=(
                    "主持人加上 ✅ 以開始抽獎。按下下方『📊 狀態』按鈕可僅自己查看；"
                    "若使用 📊 反應也會私訊給你。\n"
                    "若為 Discord 表情報名，參與者對此訊息加上 🎉 即可報名。"
                ),
                inline=False,
            )

            message = await interaction.followup.send(
                embed=embed, view=LotteryStatusView(), wait=True
            )

            # 記錄建立訊息ID，並在訊息上添加控制用反應
            update_reaction_message_id(lottery_id, message.id)

            # 報名用 🎉（僅 reaction 模式）
            if lottery_data["registration_method"] == "reaction":
                await message.add_reaction("🎉")
            # 開始用 ✅、狀態用 📊、重新抽用 🔄（兩種模式皆可）
            await message.add_reaction("✅")
            await message.add_reaction("📊")
            await message.add_reaction("🔄")

        except Exception as e:
            await interaction.followup.send(f"創建抽獎活動時發生錯誤：{e!s}", ephemeral=True)


class LotteryMethodSelectionView(nextcord.ui.View):
    """先選擇報名方式的視圖，之後再開啟表單"""

    def __init__(self, cog: "LotteryCog"):
        super().__init__(timeout=300)
        self.cog = cog

    @nextcord.ui.select(
        placeholder="選擇報名方式...",
        options=[
            nextcord.SelectOption(
                label="Discord 表情符號",
                value="reaction",
                emoji="🎉",
                description="對訊息加上 🎉 表情即可報名",
            ),
            nextcord.SelectOption(
                label="YouTube 關鍵字",
                value="youtube",
                emoji="▶️",
                description="在聊天室輸入關鍵字報名",
            ),
        ],
        min_values=1,
        max_values=1,
    )
    async def method_select(self, select: nextcord.ui.Select, interaction: Interaction) -> None:
        # 依選擇開啟相對應的建立表單
        selected_method = select.values[0]
        modal = LotteryCreateModal(selected_method)
        await interaction.response.send_modal(modal)


class LotterySpinView(nextcord.ui.View):
    """保留視圖類別以相容，但目前不再顯示控制台或動畫。"""

    def __init__(self, lottery_data: LotteryData, participants: list[LotteryParticipant]):
        super().__init__(timeout=1)
        self.lottery_data = lottery_data
        self.participants = participants


class LotteryStatusView(nextcord.ui.View):
    """提供『📊 狀態』按鈕，回覆使用者 ephemeral 狀態訊息。"""

    def __init__(self) -> None:
        # 使用無限 timeout 以提升持久度（非持久視圖）
        super().__init__(timeout=None)

    @nextcord.ui.button(label="狀態", emoji="📊", style=nextcord.ButtonStyle.secondary)
    async def show_status(  # type: ignore[override]
        self, button: nextcord.ui.Button, interaction: Interaction
    ) -> None:
        lottery = get_lottery_by_message_id(interaction.message.id)
        if lottery is None:
            await interaction.response.send_message("找不到對應的抽獎活動。", ephemeral=True)
            return

        # 取得 Cog 並呼叫其內部的狀態建構器
        cog = interaction.client.get_cog("LotteryCog")
        try:
            embed = cog._build_status_embed(lottery)  # type: ignore[attr-defined]
        except Exception:
            embed = nextcord.Embed(title="📊 抽獎活動狀態", description="狀態載入失敗", color=0x0099FF)
        await interaction.response.send_message(embed=embed, ephemeral=True)

class LotteryCog(commands.Cog):
    """抽獎功能Cog"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # -------- Helper methods (避免重複邏輯) --------
    async def _ensure_no_active_lottery(self, interaction: Interaction) -> bool:
        # 允許多個抽獎，直接通過
        return True

    async def _get_active_lottery_or_reply(self, interaction: Interaction) -> "LotteryData | None":
        # 此方法不再使用（保留以兼容可能的調用）
        await interaction.response.send_message("目前沒有活躍的抽獎活動", ephemeral=True)
        return None

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
        """抽獎功能主選單：直接顯示建立精靈面板（下拉選擇報名方式，送出後開啟表單）。"""
        view = LotteryMethodSelectionView(self)
        embed = nextcord.Embed(title="🧰 抽獎建立精靈", color=0x00FF00)
        embed.add_field(
            name="步驟 1",
            value="從下方選擇報名方式（Discord 表情符號 或 YouTube 關鍵字）",
            inline=False,
        )
        embed.add_field(name="步驟 2", value="系統將開啟表單讓你填寫標題與描述", inline=False)

        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    # 移除子指令：create/start/status，改以主指令+反應操作完成

    async def _fetch_youtube_participants_simple(self, lottery_data: LotteryData) -> int:
        """從 YouTube 聊天室抓取參與者，返回新增人數。"""
        if not lottery_data.youtube_url or not lottery_data.youtube_keyword:
            return 0
        yt_stream = YoutubeStream(url=lottery_data.youtube_url)
        registered_accounts = yt_stream.get_registered_accounts(lottery_data.youtube_keyword)
        added = 0
        for account_name in registered_accounts:
            participant = LotteryParticipant(id=account_name, name=account_name, source="youtube")
            if add_participant(lottery_data.lottery_id, participant):
                added += 1
        return added

    async def _send_spin_panel_to_channel(
        self, lottery_data: LotteryData, channel: nextcord.abc.Messageable
    ) -> None:
        """不再發控制台與動畫；保留函式以兼容舊呼叫。"""
        # 僅在無參與者時做簡短提示
        participants = get_participants(lottery_data.lottery_id)
        if not participants:
            await channel.send("沒有參與者，無法開始抽獎!")
            return

    def _build_status_embed(self, lottery_data: LotteryData) -> nextcord.Embed:
        participants = get_participants(lottery_data.lottery_id)
        embed = nextcord.Embed(title="📊 抽獎活動狀態", color=0x0099FF)
        embed.add_field(name="活動標題", value=lottery_data.title, inline=False)
        embed.add_field(name="活動描述", value=lottery_data.description or "無", inline=False)
        embed.add_field(name="發起人", value=lottery_data.creator_name, inline=True)
        embed.add_field(
            name="每次抽出人數", value=f"{getattr(lottery_data, 'draw_count', 1)} 人", inline=True
        )
        # 移除註冊方式與目前參與人數，避免版面冗長
        if lottery_data.youtube_url:
            embed.add_field(name="YouTube直播", value=lottery_data.youtube_url, inline=False)
        if lottery_data.youtube_keyword:
            embed.add_field(name="報名關鍵字", value=lottery_data.youtube_keyword, inline=True)
        if participants:
            add_participants_fields_to_embed(embed, participants)
        else:
            embed.add_field(name="參與者", value="目前沒有參與者", inline=False)
        return embed

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
        """處理 Discord 反應：報名 🎉、開始 ✅、狀態 📊"""
        if getattr(user, "bot", False):
            return

        # 僅處理與抽獎建立訊息綁定的反應
        lottery = get_lottery_by_message_id(reaction.message.id)
        if lottery is None:
            return

        emoji_str = str(reaction.emoji)

        # 1) 報名（僅 reaction 模式）
        if emoji_str == "🎉" and lottery.registration_method == "reaction":
            if isinstance(user, (Member, User)):
                participant = LotteryParticipant(
                    id=str(user.id), name=user.display_name, source="discord"
                )
                success = add_participant(lottery.lottery_id, participant)
                if not success:
                    await reaction.remove(user)
            return

        # 2) 開始（僅發起人）
        if emoji_str == "✅" and isinstance(user, (Member, User)):
            if user.id != lottery.creator_id:
                # 非發起人點了 ✅，直接移除避免誤觸
                with contextlib.suppress(Exception):
                    await reaction.remove(user)
                return

            # YouTube 模式：安靜地抓取參與者（不再發提示訊息）
            if lottery.registration_method == "youtube":
                await self._fetch_youtube_participants_simple(lottery)

            participants = get_participants(lottery.lottery_id)
            if not participants:
                await reaction.message.channel.send("沒有參與者，無法開始抽獎!")
                with contextlib.suppress(Exception):
                    await reaction.remove(user)
                return

            # 直接抽出並公告（支援一次抽出多位）
            draw_count = getattr(lottery, "draw_count", 1) or 1
            k = min(int(draw_count), len(participants))
            winners: list[LotteryParticipant] = []
            for _ in range(k):
                winner = secrets.choice(participants)
                participants.remove(winner)
                add_winner(lottery.lottery_id, winner)
                winners.append(winner)

            result_embed = nextcord.Embed(title="🎉 恭喜中獎!", color=0xFFD700)
            result_embed.add_field(name="活動", value=lottery.title, inline=False)
            if len(winners) == 1:
                w = winners[0]
                result_embed.add_field(name="中獎者", value=f"**{w.name}**", inline=False)
                result_embed.add_field(
                    name="來源",
                    value=("Discord" if w.source == "discord" else "YouTube"),
                    inline=True,
                )
            else:
                winners_str = ", ".join([
                    f"{w.name}{' (DC)' if w.source == 'discord' else ' (YT)'}" for w in winners
                ])
                result_embed.add_field(
                    name=f"中獎者（{len(winners)} 人）", value=winners_str, inline=False
                )
            result_embed.add_field(name="剩餘參與者", value=f"{len(participants)} 人", inline=True)

            await reaction.message.channel.send(embed=result_embed)

            with contextlib.suppress(Exception):
                await reaction.remove(user)
            return

        # 2.5) 重新抽獎（僅發起人）
        if emoji_str == "🔄" and isinstance(user, (Member, User)):
            if user.id != lottery.creator_id:
                with contextlib.suppress(Exception):
                    await reaction.remove(user)
                return

            # 彙總舊活動所有參與者（包含先前抽中的中獎者），去重後導入新活動
            previous_participants = list(get_participants(lottery.lottery_id))
            previous_winners = list(lottery_winners.get(lottery.lottery_id, []))
            combined = previous_participants + previous_winners
            unique_map: dict[tuple[str, str], LotteryParticipant] = {}
            for p in combined:
                unique_map[(p.id, p.source)] = p
            restored_participants = list(unique_map.values())

            # 建立新的抽獎（沿用舊設定）
            new_lottery_data = {
                "guild_id": lottery.guild_id,
                "title": lottery.title,
                "description": lottery.description,
                "creator_id": lottery.creator_id,
                "creator_name": lottery.creator_name,
                "registration_method": lottery.registration_method,
                "youtube_url": lottery.youtube_url,
                "youtube_keyword": lottery.youtube_keyword,
                "reaction_emoji": lottery.reaction_emoji,
                "draw_count": getattr(lottery, "draw_count", 1) or 1,
            }

            new_lottery_id = create_lottery(new_lottery_data)
            new_lottery = lotteries_by_id[new_lottery_id]

            # 將舊活動的人員全部恢復到新活動
            if restored_participants:
                lottery_participants[new_lottery_id] = list(restored_participants)

            # 發送新的建立訊息與控制反應
            embed = nextcord.Embed(title="🎉 抽獎活動已重新建立!", color=0x00FF00)
            embed.add_field(name="活動標題", value=new_lottery.title, inline=False)
            embed.add_field(name="活動描述", value=new_lottery.description or "無", inline=False)
            embed.add_field(name="註冊方式", value=new_lottery.registration_method, inline=True)
            embed.add_field(
                name="每次抽出人數",
                value=f"{getattr(new_lottery, 'draw_count', 1)} 人",
                inline=True,
            )

            if new_lottery.registration_method == "reaction":
                embed.add_field(
                    name="Discord報名方式", value="對此訊息加上 🎉 表情符號即可報名", inline=False
                )
            elif new_lottery.registration_method == "youtube":
                if new_lottery.youtube_url:
                    embed.add_field(
                        name="YouTube直播", value=new_lottery.youtube_url, inline=False
                    )
                if new_lottery.youtube_keyword:
                    embed.add_field(
                        name="報名關鍵字",
                        value=f"在聊天室發送包含「{new_lottery.youtube_keyword}」的訊息",
                        inline=False,
                    )

            embed.add_field(
                name="使用說明",
                value=(
                    "主持人加上 ✅ 以開始抽獎；🔄 可重新建立一個全新抽獎；任何人加上 📊 可查看狀態。\n"
                    "若為 Discord 表情報名，參與者對此訊息加上 🎉 即可報名。"
                ),
                inline=False,
            )

            new_message = await reaction.message.channel.send(
                embed=embed, view=LotteryStatusView()
            )
            update_reaction_message_id(new_lottery_id, new_message.id)

            if new_lottery.registration_method == "reaction":
                await new_message.add_reaction("🎉")
            await new_message.add_reaction("✅")
            await new_message.add_reaction("📊")
            await new_message.add_reaction("🔄")

            # 關閉舊活動並移除主持人觸發反應
            close_lottery(lottery.lottery_id)
            with contextlib.suppress(Exception):
                await reaction.remove(user)
            return

        # 3) 狀態（任何人）
        if emoji_str == "📊":
            embed = self._build_status_embed(lottery)
            # 盡量以私訊傳送，避免洗頻；若使用者關閉私訊則退回公開顯示
            try:
                if isinstance(user, (Member, User)):
                    await user.send(embed=embed)
                else:
                    await reaction.message.channel.send(embed=embed)
            except Exception:
                await reaction.message.channel.send(embed=embed)
            with contextlib.suppress(Exception):
                await reaction.remove(user)
            return

    @commands.Cog.listener()
    async def on_reaction_remove(self, reaction: nextcord.Reaction, user: Member | User) -> None:
        """處理Discord反應取消報名"""
        lottery = _get_reaction_lottery_or_none(reaction)
        if lottery is None:
            return

        if isinstance(user, (Member, User)) and not getattr(user, "bot", False):
            remove_participant(lottery.lottery_id, str(user.id), "discord")


def _get_reaction_lottery_or_none(reaction: nextcord.Reaction) -> "LotteryData | None":
    # 僅用於處理 🎉 報名/取消報名 的訊息對應
    if str(reaction.emoji) != "🎉":
        return None
    lottery = get_lottery_by_message_id(reaction.message.id)
    # 驗證伺服器一致，避免跨伺服器誤判
    guild = getattr(reaction.message, "guild", None)
    if lottery is not None and getattr(guild, "id", None) == lottery.guild_id:
        return lottery
    return None


async def setup(bot: commands.Bot) -> None:
    """Register the reply generation cog with the bot.

    Args:
        bot (commands.Bot): The bot instance to which the cog will be added.
    """
    bot.add_cog(LotteryCog(bot), override=True)
