from typing import TYPE_CHECKING, Optional, cast
import secrets
import contextlib

import nextcord
from nextcord import Interaction

from .state import (
    add_winner,
    get_lottery,
    get_winners,
    close_lottery,
    create_lottery,
    add_participant,
    get_participants,
    set_participants,
    remove_participant,
    get_lottery_by_message_id,
    update_control_message_id,
)
from .embeds import build_creation_embed
from .modals import LotteryCreateModal
from .models import LotteryParticipant

if TYPE_CHECKING:  # pragma: no cover
    from discordbot.cogs.lottery import LotteryCog


def _get_lottery_cog(interaction: Interaction) -> Optional["LotteryCog"]:
    cog = interaction.client.get_cog("LotteryCog")
    if cog is None:
        return None
    return cast("LotteryCog", cog)


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
                value="discord",
                emoji="🎉",
                description="按下『報名』按鈕即可參加（不使用表情反應）",
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
        selected_method = select.values[0]
        modal = LotteryCreateModal(selected_method)
        await interaction.response.send_modal(modal)


class JoinLotteryButton(nextcord.ui.Button):
    """『🎉 報名』按鈕（Discord 模式）"""

    def __init__(self) -> None:
        super().__init__(label="報名", emoji="🎉", style=nextcord.ButtonStyle.primary)

    async def callback(self, interaction: Interaction) -> None:
        lottery = get_lottery_by_message_id(interaction.message.id)
        if lottery is None:
            await interaction.response.send_message("找不到對應的抽獎活動。", ephemeral=True)
            return
        if lottery.registration_method != "discord":
            await interaction.response.send_message("此抽獎不支援以按鈕報名。", ephemeral=True)
            return

        user = interaction.user
        participant = LotteryParticipant(id=str(user.id), name=user.display_name, source="discord")
        ok = add_participant(lottery.lottery_id, participant)

        if ok:
            await interaction.response.send_message("✅ 報名成功!", ephemeral=True)
            with contextlib.suppress(Exception):
                updated = build_creation_embed(lottery)
                await interaction.message.edit(embed=updated, view=self.view)
        else:
            await interaction.response.send_message(
                "無法加入此抽獎（可能已中獎或活動限制）。", ephemeral=True
            )


class CancelJoinLotteryButton(nextcord.ui.Button):
    """『🚫 取消報名』按鈕（Discord 模式）"""

    def __init__(self) -> None:
        super().__init__(label="取消報名", emoji="🚫", style=nextcord.ButtonStyle.danger)

    async def callback(self, interaction: Interaction) -> None:
        lottery = get_lottery_by_message_id(interaction.message.id)
        if lottery is None:
            await interaction.response.send_message("找不到對應的抽獎活動。", ephemeral=True)
            return

        user = interaction.user
        before = len(get_participants(lottery.lottery_id))
        remove_participant(lottery.lottery_id, str(user.id), "discord")
        after = len(get_participants(lottery.lottery_id))

        if after < before:
            await interaction.response.send_message("已取消你的報名。", ephemeral=True)
            with contextlib.suppress(Exception):
                updated = build_creation_embed(lottery)
                await interaction.message.edit(embed=updated, view=self.view)
        else:
            await interaction.response.send_message("已處理。", ephemeral=True)


class UpdateYoutubeParticipantsButton(nextcord.ui.Button):
    """『🔁 更新參與者』按鈕（YouTube 模式，僅主持人可用）"""

    def __init__(self) -> None:
        super().__init__(label="更新參與者", emoji="🔁", style=nextcord.ButtonStyle.secondary)

    async def callback(self, interaction: Interaction) -> None:
        lottery = get_lottery_by_message_id(interaction.message.id)
        if lottery is None:
            await interaction.response.send_message("找不到對應的抽獎活動。", ephemeral=True)
            return

        if lottery.registration_method != "youtube":
            await interaction.response.send_message(
                "此抽獎不是 YouTube 模式，無需更新。", ephemeral=True
            )
            return

        if interaction.user.id != lottery.creator_id:
            await interaction.response.send_message("只有主持人可以更新參與者。", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        cog = _get_lottery_cog(interaction)
        if cog is None:
            await interaction.followup.send("抽獎功能尚未載入，請稍後再試。", ephemeral=True)
            return

        before = len([p for p in get_participants(lottery.lottery_id) if p.source == "youtube"])
        try:
            await cog.fetch_youtube_participants(lottery)
        except Exception as exc:  # pragma: no cover - best effort
            await interaction.followup.send(f"更新參與者失敗：{exc!s}", ephemeral=True)
            return

        after = len([p for p in get_participants(lottery.lottery_id) if p.source == "youtube"])
        added = max(0, after - before)

        with contextlib.suppress(Exception):
            updated = build_creation_embed(lottery)
            await interaction.message.edit(embed=updated, view=self.view)

        await interaction.followup.send(
            f"已更新參與者：新增 {added} 人；YouTube 總計 {after} 人。", ephemeral=True
        )


class LotteryControlView(nextcord.ui.View):
    """抽獎控制面板：🎉 報名、✅ 開始、📊 狀態（ephemeral）、🔄 重新建立。"""

    def __init__(self, registration_method: str | None = None) -> None:
        super().__init__(timeout=None)
        if registration_method == "discord":
            self.add_item(JoinLotteryButton())
            self.add_item(CancelJoinLotteryButton())
        elif registration_method == "youtube":
            self.add_item(UpdateYoutubeParticipantsButton())

    @nextcord.ui.button(label="開始抽獎", emoji="✅", style=nextcord.ButtonStyle.success)
    async def start_draw(self, button: nextcord.ui.Button, interaction: Interaction) -> None:
        lottery = get_lottery_by_message_id(interaction.message.id)
        if lottery is None:
            await interaction.response.send_message("找不到對應的抽獎活動。", ephemeral=True)
            return

        if interaction.user.id != lottery.creator_id:
            await interaction.response.send_message("只有主持人可以開始抽獎。", ephemeral=True)
            return

        cog = _get_lottery_cog(interaction)
        if cog is None:
            await interaction.response.send_message(
                "抽獎功能尚未載入，請稍後再試。", ephemeral=True
            )
            return

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

        await interaction.response.send_message(embed=result_embed)

    @nextcord.ui.button(label="狀態", emoji="📊", style=nextcord.ButtonStyle.secondary)
    async def show_status(self, button: nextcord.ui.Button, interaction: Interaction) -> None:
        lottery = get_lottery_by_message_id(interaction.message.id)
        if lottery is None:
            await interaction.response.send_message("找不到對應的抽獎活動。", ephemeral=True)
            return
        cog = _get_lottery_cog(interaction)
        if cog is None:
            fallback = nextcord.Embed(
                title="📊 抽獎活動狀態", description="抽獎功能尚未載入。", color=0x0099FF
            )
            await interaction.response.send_message(embed=fallback, ephemeral=True)
            return

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
        previous_winners = list(get_winners(lottery.lottery_id))
        combined = previous_participants + previous_winners
        unique_map: dict[tuple[str, str], LotteryParticipant] = {}
        for participant in combined:
            unique_map[(participant.id, participant.source)] = participant
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
        new_lottery = get_lottery(new_lottery_id)
        if new_lottery is None:
            await interaction.followup.send("重新建立抽獎時發生錯誤。", ephemeral=True)
            return

        if restored_participants:
            set_participants(new_lottery_id, restored_participants)

        embed = build_creation_embed(new_lottery)

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
        update_control_message_id(new_lottery_id, new_message.id)

        close_lottery(lottery.lottery_id)

        await interaction.followup.send("已重新建立新的抽獎。", ephemeral=True)
