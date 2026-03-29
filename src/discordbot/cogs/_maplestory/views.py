from __future__ import annotations

from typing import TYPE_CHECKING

import nextcord
from nextcord import Interaction, SelectOption
from nextcord.ui import View, Select

from .embeds import (
    create_map_embed,
    create_npc_embed,
    create_quest_embed,
    create_scroll_embed,
    create_monster_embed,
    create_equipment_embed,
    create_item_source_embed,
)

if TYPE_CHECKING:
    from .service import MapleStoryService


class MapleDropSearchView(View):
    """Interactive select menu for Artale search results."""

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

    def _translate(self, category: str, name: str) -> str:
        return self.service.translate(category, name)

    @nextcord.ui.select(
        placeholder="選擇要查看的結果...",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中...", value="loading")],
    )
    async def select_result(self, select: Select, interaction: Interaction) -> None:
        await interaction.response.defer()

        selected = select.values[0]
        if selected == "loading":
            await interaction.followup.send("請先選擇有效的結果。", ephemeral=True)
            return

        embed = None
        tr = self._translate

        if self.search_type == "monster":
            monster = self.service.get_monster(selected)
            if monster:
                embed = create_monster_embed(monster, translate=tr)

        elif self.search_type == "item":
            monsters = self.service.get_monsters_by_drop(selected)
            if monsters:
                embed = create_item_source_embed(selected, monsters, translate=tr)

        elif self.search_type == "equipment":
            equip = self.service.get_equipment(selected)
            if equip:
                embed = create_equipment_embed(equip, translate=tr)

        elif self.search_type == "scroll":
            results = self.service.search_scrolls_by_name(selected)
            match = next((s for s in results if s.name == selected), None)
            if match:
                embed = create_scroll_embed(match, translate=tr)

        elif self.search_type == "npc":
            results = self.service.search_npcs_by_name(selected)
            match = next((n for n in results if n.name == selected), None)
            if match:
                embed = create_npc_embed(match, translate=tr)

        elif self.search_type == "quest":
            results = self.service.search_quests_by_name(selected)
            match = next((q for q in results if q.name == selected), None)
            if match:
                embed = create_quest_embed(match, translate=tr)

        elif self.search_type == "map":
            results = self.service.search_maps_by_name(selected)
            match = next((m for m in results if m.name == selected), None)
            if match:
                embed = create_map_embed(match, translate=tr)

        if embed:
            await interaction.followup.edit_message(interaction.message.id, embed=embed, view=None)

    def set_options(self, options: list[SelectOption]) -> None:
        self.select_result.options = options[:25]
