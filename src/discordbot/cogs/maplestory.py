from pathlib import Path

import nextcord
from nextcord import Embed, Locale, Interaction, SelectOption
from nextcord.ext import commands

from ._maplestory.views import MapleDropSearchView
from ._maplestory.embeds import build_stats_embed, create_monster_embed, create_item_source_embed
from ._maplestory.service import DEFAULT_DATA_PATH, MapleStoryService


class MapleStoryCogs(commands.Cog):
    """楓之谷相關功能"""

    def __init__(self, bot: commands.Bot, data_path: Path = DEFAULT_DATA_PATH):
        self.bot = bot
        self.data_path = data_path
        self.service = MapleStoryService.from_file(data_path)

    def _ensure_data_loaded(self) -> bool:
        if self.service.has_data():
            return True
        self.service.reload(self.data_path)
        return self.service.has_data()

    @nextcord.slash_command(
        name="maple_monster",
        description="Search for monster drop information in MapleStory",
        name_localizations={Locale.zh_TW: "楓之谷怪物", Locale.ja: "メイプルモンスター"},
        description_localizations={
            Locale.zh_TW: "搜尋楓之谷怪物的掉落資訊",
            Locale.ja: "メイプルストーリーのモンスタードロップ情報を検索",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def maple_monster(
        self,
        interaction: Interaction,
        monster_name: str = nextcord.SlashOption(
            name="monster_name",
            description="Name of the monster to search",
            name_localizations={Locale.zh_TW: "怪物名稱", Locale.ja: "モンスター名"},
            description_localizations={
                Locale.zh_TW: "要搜尋的怪物名稱",
                Locale.ja: "検索するモンスターの名前",
            },
            required=True,
        ),
    ) -> None:
        await interaction.response.defer()

        if not self._ensure_data_loaded():
            embed = Embed(
                title="❌ 錯誤", description="無法載入怪物資料，請聯絡管理員。", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        monsters_found = self.service.search_monsters_by_name(monster_name)
        if not monsters_found:
            embed = Embed(
                title="🔍 搜尋結果",
                description=f"找不到名稱包含「{monster_name}」的怪物。",
                color=0xFFAA00,
            )
            await interaction.followup.send(embed=embed)
            return

        if len(monsters_found) == 1:
            embed = create_monster_embed(monsters_found[0])
            await interaction.followup.send(embed=embed)
            return

        embed = Embed(
            title="🔍 搜尋結果",
            description=f"找到 {len(monsters_found)} 個相關怪物，請選擇：",
            color=0x00AAFF,
        )
        view = MapleDropSearchView(self.service, "monster", monster_name)
        options: list[SelectOption] = []
        for monster in monsters_found:
            level = monster.get("attributes", {}).get("level", "?")
            options.append(
                SelectOption(
                    label=monster.get("name"), description=f"Lv.{level}", value=monster.get("name")
                )
            )
        view.set_options(options)
        await interaction.followup.send(embed=embed, view=view)

    @nextcord.slash_command(
        name="maple_item",
        description="Search for item drop sources in MapleStory",
        name_localizations={Locale.zh_TW: "楓之谷物品", Locale.ja: "メイプルアイテム"},
        description_localizations={
            Locale.zh_TW: "搜尋楓之谷物品的掉落來源",
            Locale.ja: "メイプルストーリーのアイテムドロップ元を検索",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def maple_item(
        self,
        interaction: Interaction,
        item_name: str = nextcord.SlashOption(
            name="item_name",
            description="Name of the item to search",
            name_localizations={Locale.zh_TW: "物品名稱", Locale.ja: "アイテム名"},
            description_localizations={
                Locale.zh_TW: "要搜尋的物品名稱",
                Locale.ja: "検索するアイテムの名前",
            },
            required=True,
        ),
    ) -> None:
        await interaction.response.defer()

        if not self._ensure_data_loaded():
            embed = Embed(
                title="❌ 錯誤", description="無法載入怪物資料，請聯絡管理員。", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        items_found = self.service.search_items_by_name(item_name)
        if not items_found:
            embed = Embed(
                title="🔍 搜尋結果",
                description=f"找不到名稱包含「{item_name}」的物品。",
                color=0xFFAA00,
            )
            await interaction.followup.send(embed=embed)
            return

        if len(items_found) == 1:
            item = items_found[0]
            monsters_with_item = self.service.get_monsters_by_item(item)
            embed = create_item_source_embed(item, monsters_with_item)
            await interaction.followup.send(embed=embed)
            return

        embed = Embed(
            title="🔍 搜尋結果",
            description=f"找到 {len(items_found)} 個相關物品，請選擇：",
            color=0x00AAFF,
        )
        view = MapleDropSearchView(self.service, "item", item_name)
        options: list[SelectOption] = []
        for item in items_found:
            item_type = self.service.get_item_type(item)
            options.append(SelectOption(label=item, description=item_type, value=item))
        view.set_options(options)
        await interaction.followup.send(embed=embed, view=view)

    @nextcord.slash_command(
        name="maple_stats",
        description="Get MapleStory database statistics",
        name_localizations={Locale.zh_TW: "楓之谷統計", Locale.ja: "メイプル統計"},
        description_localizations={
            Locale.zh_TW: "顯示楓之谷資料庫統計資訊",
            Locale.ja: "メイプルストーリーデータベース統計を表示",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def maple_stats(self, interaction: Interaction) -> None:
        await interaction.response.defer()

        if not self._ensure_data_loaded():
            embed = Embed(
                title="❌ 錯誤", description="無法載入怪物資料，請聯絡管理員。", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        stats = self.service.get_stats()
        embed = build_stats_embed(stats)
        await interaction.followup.send(embed=embed)


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(MapleStoryCogs(bot))
