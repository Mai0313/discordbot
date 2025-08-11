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
        reaction_message_id=lottery_data.get("reaction_message_id"),
        draw_count=max(1, int(lottery_data.get("draw_count", 1) or 1)),
    )

    # 存儲到全局變數（允許同時存在多個抽獎）
    lotteries_by_id[lottery_id] = lottery
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


def close_lottery(lottery_id: int) -> None:
    """關閉抽獎活動"""
    lottery = lotteries_by_id.pop(lottery_id, None)
    if lottery is not None:
        lottery.is_active = False
        if lottery.reaction_message_id is not None:
            message_to_lottery_id.pop(lottery.reaction_message_id, None)
        # 清除活躍清單中的映射（舊版相容：已移除 active_lotteries）


def add_participants_fields_to_embed(
    embed: nextcord.Embed, participants: list["LotteryParticipant"]
) -> None:
    discord_users = [p for p in participants if p.source == "discord"]
    youtube_users = [p for p in participants if p.source == "youtube"]

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


def add_participants_ids_fields_to_embed(
    embed: nextcord.Embed, participants: list["LotteryParticipant"]
) -> None:
    """在建立訊息上顯示參與者的 ID 清單。"""
    discord_names = [p.name for p in participants if p.source == "discord"]
    youtube_names = [p.name for p in participants if p.source == "youtube"]

    if discord_names:
        embed.add_field(
            name=f"Discord 參與者ID ({len(discord_names)} 人)",
            value=", ".join(discord_names),
            inline=False,
        )
    if youtube_names:
        embed.add_field(
            name=f"YouTube 參與者ID ({len(youtube_names)} 人)",
            value=", ".join(youtube_names),
            inline=False,
        )


def build_creation_embed(lottery_data: "LotteryData") -> nextcord.Embed:
    """建立或更新『抽獎活動已創建』訊息的 Embed（包含參與者ID清單）。"""
    embed = nextcord.Embed(title="🎉 抽獎活動已創建!", color=0x00FF00)
    embed.add_field(name="活動標題", value=lottery_data.title, inline=False)
    embed.add_field(name="活動描述", value=lottery_data.description or "無", inline=False)
    # 顯示更友善的註冊方式文案（Discord 改為按鈕，不再鼓勵使用表情）
    registration_label = (
        "Discord 按鈕" if lottery_data.registration_method == "reaction" else "YouTube 關鍵字"
    )
    embed.add_field(name="註冊方式", value=registration_label, inline=True)
    embed.add_field(
        name="每次抽出人數", value=f"{getattr(lottery_data, 'draw_count', 1)} 人", inline=True
    )
    if lottery_data.registration_method == "youtube":
        if lottery_data.youtube_url:
            embed.add_field(name="YouTube直播", value=str(lottery_data.youtube_url), inline=False)
        if lottery_data.youtube_keyword:
            embed.add_field(
                name="報名關鍵字", value=str(lottery_data.youtube_keyword), inline=True
            )

    # 附加參與者ID清單
    participants = get_participants(lottery_data.lottery_id)
    if participants:
        add_participants_ids_fields_to_embed(embed, participants)
    else:
        embed.add_field(name="參與者", value="目前沒有參與者", inline=False)
    return embed


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
            lottery = lotteries_by_id[lottery_id]

            # 建立回應 embed（含參與者ID清單）
            embed = build_creation_embed(lottery)
            message = await interaction.followup.send(
                embed=embed,
                view=LotteryControlView(registration_method=lottery.registration_method),
                wait=True,
            )

            # 記錄建立訊息ID；控制改用按鈕
            update_reaction_message_id(lottery_id, message.id)
            # 不再自動添加任何控制反應；全部透過按鈕進行。

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
                label="Discord 按鈕",
                value="reaction",
                emoji="🎉",
                description="按下『報名』按鈕即可參加（不再使用表情反應）",
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


class JoinLotteryButton(nextcord.ui.Button):
    """『🎉 報名』按鈕（Discord/Reaction 模式）"""

    def __init__(self) -> None:
        super().__init__(label="報名", emoji="🎉", style=nextcord.ButtonStyle.primary)

    async def callback(self, interaction: Interaction) -> None:
        lottery = get_lottery_by_message_id(interaction.message.id)
        if lottery is None:
            await interaction.response.send_message("找不到對應的抽獎活動。", ephemeral=True)
            return
        if lottery.registration_method != "reaction":
            await interaction.response.send_message("此抽獎不支援以按鈕報名。", ephemeral=True)
            return

        user = interaction.user
        if not isinstance(user, (Member, User)):
            await interaction.response.send_message("僅限伺服器成員可報名。", ephemeral=True)
            return

        existing = any(
            p.id == str(user.id) and p.source == "discord"
            for p in get_participants(lottery.lottery_id)
        )
        participant = LotteryParticipant(id=str(user.id), name=user.display_name, source="discord")
        ok = add_participant(lottery.lottery_id, participant)

        if ok and not existing:
            await interaction.response.send_message("✅ 報名成功!", ephemeral=True)
            # 更新原始建立訊息的 embed 以加入最新的參與者ID名單
            with contextlib.suppress(Exception):
                updated = build_creation_embed(lottery)
                await interaction.message.edit(embed=updated, view=self.view)
        elif ok and existing:
            await interaction.response.send_message("你已經完成報名。", ephemeral=True)
        else:
            await interaction.response.send_message(
                "此抽獎僅限其他平台報名，無法以 Discord 報名。", ephemeral=True
            )


class CancelJoinLotteryButton(nextcord.ui.Button):
    """『🚫 取消報名』按鈕（Discord/Reaction 模式）"""

    def __init__(self) -> None:
        super().__init__(label="取消報名", emoji="🚫", style=nextcord.ButtonStyle.danger)

    async def callback(self, interaction: Interaction) -> None:
        lottery = get_lottery_by_message_id(interaction.message.id)
        if lottery is None:
            await interaction.response.send_message("找不到對應的抽獎活動。", ephemeral=True)
            return

        user = interaction.user
        if not isinstance(user, (Member, User)):
            await interaction.response.send_message("僅限伺服器成員可取消。", ephemeral=True)
            return

        before = len(get_participants(lottery.lottery_id))
        remove_participant(lottery.lottery_id, str(user.id), "discord")
        after = len(get_participants(lottery.lottery_id))

        if after < before:
            await interaction.response.send_message("已取消你的報名。", ephemeral=True)
            # 同步更新建立訊息
            with contextlib.suppress(Exception):
                updated = build_creation_embed(lottery)
                await interaction.message.edit(embed=updated, view=self.view)
        else:
            await interaction.response.send_message("你尚未報名。", ephemeral=True)


class LotteryControlView(nextcord.ui.View):
    """抽獎控制面板：🎉 報名、✅ 開始、📊 狀態（ephemeral）、🔄 重新建立。"""

    def __init__(self, registration_method: str | None = None) -> None:
        super().__init__(timeout=None)
        # 動態加入自定義按鈕（僅 Discord/Reaction 模式）
        if registration_method == "reaction":
            self.add_item(JoinLotteryButton())
            self.add_item(CancelJoinLotteryButton())

    @nextcord.ui.button(label="開始抽獎", emoji="✅", style=nextcord.ButtonStyle.success)
    async def start_draw(self, button: nextcord.ui.Button, interaction: Interaction) -> None:
        lottery = get_lottery_by_message_id(interaction.message.id)
        if lottery is None:
            await interaction.response.send_message("找不到對應的抽獎活動。", ephemeral=True)
            return

        if interaction.user.id != lottery.creator_id:
            await interaction.response.send_message("只有主持人可以開始抽獎。", ephemeral=True)
            return

        # 如為 YouTube 模式，先抓取參與者
        cog: LotteryCog = interaction.client.get_cog("LotteryCog")
        if lottery.registration_method == "youtube":
            await cog.fetch_youtube_participants(lottery)

        participants = get_participants(lottery.lottery_id)
        if not participants:
            await interaction.response.send_message("沒有參與者，無法開始抽獎!", ephemeral=True)
            return

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
        else:
            winners_str = ", ".join([
                f"{w.name}{' (DC)' if w.source == 'discord' else ' (YT)'}" for w in winners
            ])
            result_embed.add_field(
                name=f"中獎者（{len(winners)} 人）", value=winners_str, inline=False
            )
        remaining_names = ", ".join([p.name for p in participants]) or "無"
        result_embed.add_field(name="剩餘參與者", value=remaining_names, inline=False)

        # 公開公告結果
        await interaction.response.send_message(embed=result_embed)

    @nextcord.ui.button(label="狀態", emoji="📊", style=nextcord.ButtonStyle.secondary)
    async def show_status(self, button: nextcord.ui.Button, interaction: Interaction) -> None:
        lottery = get_lottery_by_message_id(interaction.message.id)
        if lottery is None:
            await interaction.response.send_message("找不到對應的抽獎活動。", ephemeral=True)
            return
        cog: LotteryCog = interaction.client.get_cog("LotteryCog")
        try:
            embed = cog.build_status_embed(lottery)
        except Exception:
            embed = nextcord.Embed(
                title="📊 抽獎活動狀態", description="狀態載入失敗", color=0x0099FF
            )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @nextcord.ui.button(label="重新建立", emoji="🔄", style=nextcord.ButtonStyle.primary)
    async def recreate_lottery(self, button: nextcord.ui.Button, interaction: Interaction) -> None:
        lottery = get_lottery_by_message_id(interaction.message.id)
        if lottery is None:
            await interaction.response.send_message("找不到對應的抽獎活動。", ephemeral=True)
            return

        if interaction.user.id != lottery.creator_id:
            await interaction.response.send_message("只有主持人可以重新建立抽獎。", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        previous_participants = list(get_participants(lottery.lottery_id))
        previous_winners = list(lottery_winners.get(lottery.lottery_id, []))
        combined = previous_participants + previous_winners
        unique_map: dict[tuple[str, str], LotteryParticipant] = {}
        for p in combined:
            unique_map[(p.id, p.source)] = p
        restored_participants = list(unique_map.values())

        new_lottery_data = {
            "guild_id": lottery.guild_id,
            "title": lottery.title,
            "description": lottery.description,
            "creator_id": lottery.creator_id,
            "creator_name": lottery.creator_name,
            "registration_method": lottery.registration_method,
            "youtube_url": lottery.youtube_url,
            "youtube_keyword": lottery.youtube_keyword,
            "draw_count": getattr(lottery, "draw_count", 1) or 1,
        }

        new_lottery_id = create_lottery(new_lottery_data)
        new_lottery = lotteries_by_id[new_lottery_id]

        if restored_participants:
            lottery_participants[new_lottery_id] = list(restored_participants)

        embed = build_creation_embed(new_lottery)

        # 發送新的控制面板訊息
        channel = getattr(interaction, "channel", None)
        if channel is not None:
            new_message = await channel.send(
                embed=embed,
                view=LotteryControlView(registration_method=new_lottery.registration_method),
            )
        else:
            new_message = await interaction.followup.send(
                embed=embed,
                view=LotteryControlView(registration_method=new_lottery.registration_method),
                wait=True,
            )
        update_reaction_message_id(new_lottery_id, new_message.id)

        # 關閉舊活動
        close_lottery(lottery.lottery_id)

        await interaction.followup.send("已重新建立新的抽獎。", ephemeral=True)


class LotteryCog(commands.Cog):
    """抽獎功能Cog"""

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

    async def fetch_youtube_participants(self, lottery_data: LotteryData) -> int:
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

    def build_status_embed(self, lottery_data: LotteryData) -> nextcord.Embed:
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


async def setup(bot: commands.Bot) -> None:
    """Register the reply generation cog with the bot.

    Args:
        bot (commands.Bot): The bot instance to which the cog will be added.
    """
    bot.add_cog(LotteryCog(bot), override=True)
