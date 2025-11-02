import contextlib

import nextcord
from nextcord import Interaction

from .state import get_lottery, create_lottery, update_control_message_id
from .embeds import build_creation_embed


class LotteryCreateModal(nextcord.ui.Modal):
    """創建抽獎活動的表單"""

    def __init__(self, registration_method: str):
        super().__init__(title="創建抽獎活動")
        self.registration_method = registration_method

        self.title_input = nextcord.ui.TextInput(
            label="抽獎標題", placeholder="請輸入抽獎活動標題...", max_length=100, required=True
        )
        self.add_item(self.title_input)

        self.draw_count_input = nextcord.ui.TextInput(
            label="每次抽出人數",
            default_value="1",
            placeholder="預設 1",
            max_length=3,
            required=False,
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
            if interaction.guild is None:
                await interaction.followup.send(
                    "❌ 抽獎功能只能在伺服器中使用，不支援私人訊息!", ephemeral=True
                )
                return

            lottery_data: dict[str, object] = {
                "guild_id": interaction.guild.id,
                "title": self.title_input.value,
                "description": self.description_input.value or "",
                "creator_id": interaction.user.id,
                "creator_name": interaction.user.display_name,
                "registration_method": self.registration_method,
            }

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

            lottery_id = create_lottery(lottery_data)
            lottery = get_lottery(lottery_id)
            if lottery is None:
                await interaction.followup.send("創建抽獎活動時發生錯誤。", ephemeral=True)
                return

            # Import locally to avoid circular dependency
            from .views import LotteryControlView  # noqa: PLC0415

            if lottery.registration_method == "youtube":
                cog = interaction.client.get_cog("LotteryCog")
                if cog is not None:
                    with contextlib.suppress(Exception):
                        await cog.fetch_youtube_participants(lottery)

            embed = build_creation_embed(lottery)
            message = await interaction.followup.send(
                embed=embed,
                view=LotteryControlView(registration_method=lottery.registration_method),
                wait=True,
            )

            update_control_message_id(lottery_id, message.id)

        except Exception as exc:  # pragma: no cover - best effort error path
            await interaction.followup.send(f"創建抽獎活動時發生錯誤：{exc!s}", ephemeral=True)
