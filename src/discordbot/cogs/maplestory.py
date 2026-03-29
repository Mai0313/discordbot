from pathlib import Path

import nextcord
from nextcord import Embed, Interaction, Locale, SelectOption, SlashOption
from nextcord.ext import commands

from ._maplestory.embeds import (
    build_stats_embed,
    create_equipment_embed,
    create_item_source_embed,
    create_map_embed,
    create_monster_embed,
    create_npc_embed,
    create_quest_embed,
    create_scroll_embed,
)
from ._maplestory.service import DEFAULT_DATA_DIR, MapleStoryService
from ._maplestory.views import MapleDropSearchView

_NOT_FOUND_COLOR = 0xFFAA00
_MULTI_COLOR = 0x00AAFF
_ERROR_COLOR = 0xFF0000


class MapleStoryCogs(commands.Cog):
    """楓之谷 Artale 資料查詢"""

    def __init__(self, bot: commands.Bot, data_dir: Path = DEFAULT_DATA_DIR):
        self.bot = bot
        self.data_dir = data_dir
        self.service = MapleStoryService.from_directory(data_dir)

    def _ensure_data(self) -> bool:
        if self.service.has_data():
            return True
        self.service.reload(self.data_dir)
        return self.service.has_data()

    def _translate(self, category: str, name: str) -> str:
        return self.service.translate(category, name)

    async def _send_error(self, interaction: Interaction) -> None:
        embed = Embed(title=":x: 錯誤", description="無法載入資料，請聯絡管理員。", color=_ERROR_COLOR)
        await interaction.followup.send(embed=embed)

    async def _send_not_found(self, interaction: Interaction, kind: str, query: str) -> None:
        embed = Embed(
            title=":mag: 搜尋結果",
            description=f"找不到名稱包含「{query}」的{kind}。",
            color=_NOT_FOUND_COLOR,
        )
        await interaction.followup.send(embed=embed)

    # ── /maple_monster ──────────────────────────────────────────────

    @nextcord.slash_command(
        name="maple_monster",
        description="Search for monster information in MapleStory",
        name_localizations={Locale.zh_TW: "楓之谷怪物", Locale.ja: "メイプルモンスター"},
        description_localizations={
            Locale.zh_TW: "搜尋楓之谷怪物資訊",
            Locale.ja: "メイプルストーリーのモンスター情報を検索",
        },
    )
    async def maple_monster(
        self,
        interaction: Interaction,
        name: str = SlashOption(
            name="name",
            description="Monster name to search",
            name_localizations={Locale.zh_TW: "名稱", Locale.ja: "名前"},
            description_localizations={
                Locale.zh_TW: "要搜尋的怪物名稱",
                Locale.ja: "検索するモンスターの名前",
            },
            required=True,
        ),
    ) -> None:
        await interaction.response.defer()
        if not self._ensure_data():
            return await self._send_error(interaction)

        results = self.service.search_monsters_by_name(name)
        if not results:
            return await self._send_not_found(interaction, "怪物", name)

        if len(results) == 1:
            embed = create_monster_embed(results[0], translate=self._translate)
            return await interaction.followup.send(embed=embed)

        embed = Embed(
            title=":mag: 搜尋結果",
            description=f"找到 {len(results)} 個相關怪物，請選擇：",
            color=_MULTI_COLOR,
        )
        view = MapleDropSearchView(self.service, "monster", name)
        view.set_options([
            SelectOption(
                label=m.display_name, description=f"Lv.{m.level}", value=m.name
            )
            for m in results
        ])
        await interaction.followup.send(embed=embed, view=view)

    # ── /maple_equip ────────────────────────────────────────────────

    @nextcord.slash_command(
        name="maple_equip",
        description="Search for equipment in MapleStory",
        name_localizations={Locale.zh_TW: "楓之谷裝備", Locale.ja: "メイプル装備"},
        description_localizations={
            Locale.zh_TW: "搜尋楓之谷裝備資訊",
            Locale.ja: "メイプルストーリーの装備情報を検索",
        },
    )
    async def maple_equip(
        self,
        interaction: Interaction,
        name: str = SlashOption(
            name="name",
            description="Equipment name to search",
            name_localizations={Locale.zh_TW: "名稱", Locale.ja: "名前"},
            description_localizations={
                Locale.zh_TW: "要搜尋的裝備名稱",
                Locale.ja: "検索する装備の名前",
            },
            required=True,
        ),
    ) -> None:
        await interaction.response.defer()
        if not self._ensure_data():
            return await self._send_error(interaction)

        results = self.service.search_equipment_by_name(name)
        if not results:
            return await self._send_not_found(interaction, "裝備", name)

        if len(results) == 1:
            embed = create_equipment_embed(results[0], translate=self._translate)
            return await interaction.followup.send(embed=embed)

        embed = Embed(
            title=":mag: 搜尋結果",
            description=f"找到 {len(results)} 個相關裝備，請選擇：",
            color=_MULTI_COLOR,
        )
        view = MapleDropSearchView(self.service, "equipment", name)
        view.set_options([
            SelectOption(
                label=e.display_name,
                description=f"Lv.{e.level} | {self._translate('eqType', e.type)}",
                value=e.name,
            )
            for e in results
        ])
        await interaction.followup.send(embed=embed, view=view)

    # ── /maple_scroll ───────────────────────────────────────────────

    @nextcord.slash_command(
        name="maple_scroll",
        description="Search for scrolls in MapleStory",
        name_localizations={Locale.zh_TW: "楓之谷捲軸", Locale.ja: "メイプル巻物"},
        description_localizations={
            Locale.zh_TW: "搜尋楓之谷捲軸資訊",
            Locale.ja: "メイプルストーリーの巻物情報を検索",
        },
    )
    async def maple_scroll(
        self,
        interaction: Interaction,
        name: str = SlashOption(
            name="name",
            description="Scroll name to search",
            name_localizations={Locale.zh_TW: "名稱", Locale.ja: "名前"},
            description_localizations={
                Locale.zh_TW: "要搜尋的捲軸名稱",
                Locale.ja: "検索する巻物の名前",
            },
            required=True,
        ),
    ) -> None:
        await interaction.response.defer()
        if not self._ensure_data():
            return await self._send_error(interaction)

        results = self.service.search_scrolls_by_name(name)
        if not results:
            return await self._send_not_found(interaction, "捲軸", name)

        if len(results) == 1:
            embed = create_scroll_embed(results[0], translate=self._translate)
            return await interaction.followup.send(embed=embed)

        embed = Embed(
            title=":mag: 搜尋結果",
            description=f"找到 {len(results)} 個相關捲軸，請選擇：",
            color=_MULTI_COLOR,
        )
        view = MapleDropSearchView(self.service, "scroll", name)
        view.set_options([
            SelectOption(
                label=s.display_name,
                description=self._translate("eqType", s.type),
                value=s.name,
            )
            for s in results
        ])
        await interaction.followup.send(embed=embed, view=view)

    # ── /maple_npc ──────────────────────────────────────────────────

    @nextcord.slash_command(
        name="maple_npc",
        description="Search for NPCs in MapleStory",
        name_localizations={Locale.zh_TW: "楓之谷NPC", Locale.ja: "メイプルNPC"},
        description_localizations={
            Locale.zh_TW: "搜尋楓之谷 NPC 資訊",
            Locale.ja: "メイプルストーリーのNPC情報を検索",
        },
    )
    async def maple_npc(
        self,
        interaction: Interaction,
        name: str = SlashOption(
            name="name",
            description="NPC name to search",
            name_localizations={Locale.zh_TW: "名稱", Locale.ja: "名前"},
            description_localizations={
                Locale.zh_TW: "要搜尋的 NPC 名稱",
                Locale.ja: "検索するNPCの名前",
            },
            required=True,
        ),
    ) -> None:
        await interaction.response.defer()
        if not self._ensure_data():
            return await self._send_error(interaction)

        results = self.service.search_npcs_by_name(name)
        if not results:
            return await self._send_not_found(interaction, "NPC", name)

        if len(results) == 1:
            embed = create_npc_embed(results[0], translate=self._translate)
            return await interaction.followup.send(embed=embed)

        embed = Embed(
            title=":mag: 搜尋結果",
            description=f"找到 {len(results)} 個相關 NPC，請選擇：",
            color=_MULTI_COLOR,
        )
        view = MapleDropSearchView(self.service, "npc", name)
        view.set_options([
            SelectOption(label=n.display_name, description=n.type, value=n.name)
            for n in results
        ])
        await interaction.followup.send(embed=embed, view=view)

    # ── /maple_quest ────────────────────────────────────────────────

    @nextcord.slash_command(
        name="maple_quest",
        description="Search for quests in MapleStory",
        name_localizations={Locale.zh_TW: "楓之谷任務", Locale.ja: "メイプルクエスト"},
        description_localizations={
            Locale.zh_TW: "搜尋楓之谷任務資訊",
            Locale.ja: "メイプルストーリーのクエスト情報を検索",
        },
    )
    async def maple_quest(
        self,
        interaction: Interaction,
        name: str = SlashOption(
            name="name",
            description="Quest name to search",
            name_localizations={Locale.zh_TW: "名稱", Locale.ja: "名前"},
            description_localizations={
                Locale.zh_TW: "要搜尋的任務名稱",
                Locale.ja: "検索するクエストの名前",
            },
            required=True,
        ),
    ) -> None:
        await interaction.response.defer()
        if not self._ensure_data():
            return await self._send_error(interaction)

        results = self.service.search_quests_by_name(name)
        if not results:
            return await self._send_not_found(interaction, "任務", name)

        if len(results) == 1:
            embed = create_quest_embed(results[0], translate=self._translate)
            return await interaction.followup.send(embed=embed)

        embed = Embed(
            title=":mag: 搜尋結果",
            description=f"找到 {len(results)} 個相關任務，請選擇：",
            color=_MULTI_COLOR,
        )
        view = MapleDropSearchView(self.service, "quest", name)
        view.set_options([
            SelectOption(
                label=q.display_name,
                description=f"Lv.{q.lv_lower} | {q.frequency}",
                value=q.name,
            )
            for q in results
        ])
        await interaction.followup.send(embed=embed, view=view)

    # ── /maple_map ──────────────────────────────────────────────────

    @nextcord.slash_command(
        name="maple_map",
        description="Search for maps in MapleStory",
        name_localizations={Locale.zh_TW: "楓之谷地圖", Locale.ja: "メイプルマップ"},
        description_localizations={
            Locale.zh_TW: "搜尋楓之谷地圖資訊",
            Locale.ja: "メイプルストーリーのマップ情報を検索",
        },
    )
    async def maple_map(
        self,
        interaction: Interaction,
        name: str = SlashOption(
            name="name",
            description="Map name to search",
            name_localizations={Locale.zh_TW: "名稱", Locale.ja: "名前"},
            description_localizations={
                Locale.zh_TW: "要搜尋的地圖名稱",
                Locale.ja: "検索するマップの名前",
            },
            required=True,
        ),
    ) -> None:
        await interaction.response.defer()
        if not self._ensure_data():
            return await self._send_error(interaction)

        results = self.service.search_maps_by_name(name)
        if not results:
            return await self._send_not_found(interaction, "地圖", name)

        if len(results) == 1:
            embed = create_map_embed(results[0], translate=self._translate)
            return await interaction.followup.send(embed=embed)

        embed = Embed(
            title=":mag: 搜尋結果",
            description=f"找到 {len(results)} 個相關地圖，請選擇：",
            color=_MULTI_COLOR,
        )
        view = MapleDropSearchView(self.service, "map", name)
        view.set_options([
            SelectOption(
                label=m.display_name,
                description=self._translate("region", m.region),
                value=m.name,
            )
            for m in results
        ])
        await interaction.followup.send(embed=embed, view=view)

    # ── /maple_item ─────────────────────────────────────────────────

    @nextcord.slash_command(
        name="maple_item",
        description="Search for item drop sources in MapleStory",
        name_localizations={Locale.zh_TW: "楓之谷物品", Locale.ja: "メイプルアイテム"},
        description_localizations={
            Locale.zh_TW: "搜尋楓之谷物品的掉落來源",
            Locale.ja: "メイプルストーリーのアイテムドロップ元を検索",
        },
    )
    async def maple_item(
        self,
        interaction: Interaction,
        name: str = SlashOption(
            name="name",
            description="Item name to search",
            name_localizations={Locale.zh_TW: "名稱", Locale.ja: "名前"},
            description_localizations={
                Locale.zh_TW: "要搜尋的物品名稱",
                Locale.ja: "検索するアイテムの名前",
            },
            required=True,
        ),
    ) -> None:
        await interaction.response.defer()
        if not self._ensure_data():
            return await self._send_error(interaction)

        items_found = self.service.search_items_by_name(name)
        if not items_found:
            return await self._send_not_found(interaction, "物品", name)

        if len(items_found) == 1:
            item = items_found[0]
            monsters = self.service.get_monsters_by_drop(item)
            embed = create_item_source_embed(item, monsters, translate=self._translate)
            return await interaction.followup.send(embed=embed)

        embed = Embed(
            title=":mag: 搜尋結果",
            description=f"找到 {len(items_found)} 個相關物品，請選擇：",
            color=_MULTI_COLOR,
        )
        view = MapleDropSearchView(self.service, "item", name)
        view.set_options([
            SelectOption(
                label=self._translate("equipment", item)
                if self._translate("equipment", item) != item
                else self._translate("misc", item),
                description=self.service.get_item_type(item),
                value=item,
            )
            for item in items_found
        ])
        await interaction.followup.send(embed=embed, view=view)

    # ── /maple_stats ────────────────────────────────────────────────

    @nextcord.slash_command(
        name="maple_stats",
        description="Get MapleStory database statistics",
        name_localizations={Locale.zh_TW: "楓之谷統計", Locale.ja: "メイプル統計"},
        description_localizations={
            Locale.zh_TW: "顯示楓之谷資料庫統計資訊",
            Locale.ja: "メイプルストーリーデータベース統計を表示",
        },
    )
    async def maple_stats(self, interaction: Interaction) -> None:
        await interaction.response.defer()
        if not self._ensure_data():
            return await self._send_error(interaction)

        stats = self.service.get_stats()
        embed = build_stats_embed(stats)
        await interaction.followup.send(embed=embed)


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(MapleStoryCogs(bot))
