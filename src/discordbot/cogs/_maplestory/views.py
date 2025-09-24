import nextcord
from nextcord import Interaction, SelectOption
from nextcord.ui import View, Select

from .embeds import create_monster_embed, create_item_source_embed
from .service import MapleStoryService


class MapleDropSearchView(View):
    """楓之谷掉落物品搜尋的互動式介面"""

    def __init__(
        self,
        service: MapleStoryService,
        search_type: str,
        query: str,
        *,
        timeout: float | None = 300,
    ) -> None:
        super().__init__(timeout=timeout)
        self.service = service
        self.search_type = search_type
        self.query = query

    @nextcord.ui.select(
        placeholder="選擇要查看的結果...",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中...", value="loading")],
    )
    async def select_result(self, select: Select, interaction: Interaction) -> None:
        await interaction.response.defer()

        selected_value = select.values[0]
        if selected_value == "loading":
            await interaction.followup.send("請先選擇有效的結果。", ephemeral=True)
            return

        if self.search_type == "monster":
            monster = self.service.get_monster(selected_value)
            if monster:
                embed = create_monster_embed(monster)
                await interaction.followup.edit_message(
                    interaction.message.id, embed=embed, view=None
                )
        elif self.search_type == "item":
            monsters = self.service.get_monsters_by_item(selected_value)
            if monsters:
                embed = create_item_source_embed(selected_value, monsters)
                await interaction.followup.edit_message(
                    interaction.message.id, embed=embed, view=None
                )

    def set_options(self, options: list[SelectOption]) -> None:
        self.select_result.options = options[:25]
