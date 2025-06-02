import os
import json
from typing import Any

import logfire
import nextcord
from nextcord import Embed, Locale, Interaction, SelectOption
from nextcord.ui import View, Select
from nextcord.ext import commands

# 怪物屬性格式模板
MONSTER_ATTR_TEMPLATE = """
**等級**: {level}
**HP**: {hp}
**MP**: {mp}
**經驗值**: {exp}
**迴避**: {evasion}
**物理防禦**: {pdef}
**魔法防禦**: {mdef}
**命中需求**: {accuracy_required}
"""

# 基本統計格式模板
BASIC_STATS_TEMPLATE = """
**怪物總數**: {total_monsters}
**物品總數**: {total_items}
**地圖總數**: {total_maps}
"""


class MapleDropSearchView(View):
    """楓之谷掉落物品搜尋的互動式介面"""

    def __init__(self, monsters_data: list[dict[str, Any]], search_type: str, query: str):
        super().__init__(timeout=300)
        self.monsters_data = monsters_data
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

        if self.search_type == "monster":
            # 搜尋怪物的掉落物品
            monster = next((m for m in self.monsters_data if m["name"] == selected_value), None)
            if monster:
                embed = self.create_monster_embed(monster)
                await interaction.followup.edit_message(
                    interaction.message.id, embed=embed, view=None
                )
        elif self.search_type == "item":
            # 搜尋物品的掉落來源
            monsters_with_item = []
            for monster in self.monsters_data:
                if any(drop["name"] == selected_value for drop in monster.get("drops", [])):
                    monsters_with_item.append(monster)

            if monsters_with_item:
                embed = self.create_item_source_embed(selected_value, monsters_with_item)
                await interaction.followup.edit_message(
                    interaction.message.id, embed=embed, view=None
                )

    def create_monster_embed(self, monster: dict[str, Any]) -> Embed:
        """創建怪物資訊的 Embed"""
        embed = Embed(title=f"🐲 {monster['name']}", description="怪物詳細資訊", color=0x00FF00)

        # 添加怪物圖片
        if monster.get("image"):
            embed.set_thumbnail(url=monster["image"])

        # 怪物屬性
        attrs: dict[str, str] = monster.get("attributes", {})
        attr_text = MONSTER_ATTR_TEMPLATE.format(
            level=attrs.get("level", "N/A"),
            hp=attrs.get("hp", "N/A"),
            mp=attrs.get("mp", "N/A"),
            exp=attrs.get("exp", "N/A"),
            evasion=attrs.get("evasion", "N/A"),
            pdef=attrs.get("pdef", "N/A"),
            mdef=attrs.get("mdef", "N/A"),
            accuracy_required=attrs.get("accuracy_required", "N/A"),
        )
        embed.add_field(name="📊 屬性", value=attr_text, inline=True)

        # 出現地圖
        maps = monster.get("maps", [])
        if maps:
            maps_text = "\n".join([f"• {map_name}" for map_name in maps])
            embed.add_field(name="🗺️ 出現地圖", value=maps_text, inline=True)

        # 掉落物品
        drops = monster.get("drops", [])
        if drops:
            # 分類掉落物品
            equipment = [drop for drop in drops if drop.get("type") == "裝備"]
            consumables = [drop for drop in drops if drop.get("type") == "消耗品/素材"]

            if equipment:
                equip_text = "\n".join([f"• {item['name']}" for item in equipment])
                embed.add_field(name="⚔️ 裝備掉落", value=equip_text, inline=False)

            if consumables:
                cons_text = "\n".join([f"• {item['name']}" for item in consumables])
                embed.add_field(name="🧪 消耗品/素材", value=cons_text, inline=False)

        embed.set_footer(text="資料來源：Artale")
        return embed

    def create_item_source_embed(self, item_name: str, monsters: list[dict[str, Any]]) -> Embed:
        """創建物品掉落來源的 Embed"""
        embed = Embed(title=f"🎁 {item_name}", description="物品掉落來源", color=0x0099FF)

        # 找到第一個有此物品圖片的怪物
        item_img = None
        item_link = None
        for monster in monsters:
            for drop in monster.get("drops", []):
                if drop["name"] == item_name:
                    item_img = drop.get("img")
                    item_link = drop.get("link")
                    break
            if item_img:
                break

        if item_img:
            embed.set_thumbnail(url=item_img)

        if item_link:
            embed.add_field(name="🔗 詳細資訊", value=f"[查看詳細資料]({item_link})", inline=False)

        # 掉落來源怪物
        monster_list = []
        for monster in monsters:
            attrs = monster.get("attributes", {})
            level = attrs.get("level", "?")
            monster_list.append(f"• **{monster['name']}** (Lv.{level})")

        embed.add_field(name="🐲 掉落來源怪物", value="\n".join(monster_list), inline=False)

        embed.set_footer(text="資料來源：Artale")
        return embed


class MapleStoryCogs(commands.Cog):
    """楓之谷相關功能"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.monsters_data = self._load_monsters_data()
        # 快取常用查詢結果
        self._item_cache: dict[str, list[str]] = {}
        self._monster_cache: dict[str, list[dict[str, Any]]] = {}

    def _load_monsters_data(self) -> list[dict[str, Any]]:
        """載入怪物資料"""
        try:
            monsters_file = os.path.join("data", "monsters.json")
            with open(monsters_file, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logfire.warning(f"找不到怪物資料檔案 {monsters_file}")
            return []
        except json.JSONDecodeError as e:
            logfire.error(f"無法解析怪物資料檔案 - {e}")
            return []

    def _search_monsters_by_name_cached(self, query: str) -> tuple:
        """帶快取的怪物搜尋 (返回 tuple 以支持快取)"""
        results = self.search_monsters_by_name(query)
        return tuple(results)

    def _search_items_by_name_cached(self, query: str) -> tuple:
        """帶快取的物品搜尋 (返回 tuple 以支持快取)"""
        results = self.search_items_by_name(query)
        return tuple(results)

    def search_monsters_by_name(self, query: str) -> list[dict[str, Any]]:
        """根據名稱搜尋怪物"""
        query_lower = query.lower()
        results = []

        for monster in self.monsters_data:
            if query_lower in monster["name"].lower():
                results.append(monster)

        return results

    def search_items_by_name(self, query: str) -> list[str]:
        """根據名稱搜尋物品"""
        query_lower = query.lower()
        items_found = set()

        for monster in self.monsters_data:
            for drop in monster.get("drops", []):
                if query_lower in drop["name"].lower():
                    items_found.add(drop["name"])

        return list(items_found)

    def get_monsters_by_item(self, item_name: str) -> list[dict[str, Any]]:
        """取得掉落特定物品的怪物列表"""
        monsters_with_item = []

        for monster in self.monsters_data:
            for drop in monster.get("drops", []):
                if drop["name"] == item_name:
                    monsters_with_item.append(monster)
                    break

        return monsters_with_item

    def _get_monster_stats_summary(self, monster: dict[str, Any]) -> str:
        """獲取怪物屬性摘要"""
        attrs = monster.get("attributes", {})
        level = attrs.get("level", "?")
        hp = attrs.get("hp", "?")
        exp = attrs.get("exp", "?")
        return f"Lv.{level} | HP:{hp} | EXP:{exp}"

    def _get_popular_items(self) -> list[str]:
        """獲取熱門物品 (出現次數最多的物品)"""
        item_count: dict[str, int] = {}
        for monster in self.monsters_data:
            for drop in monster.get("drops", []):
                item_name = drop["name"]
                item_count[item_name] = item_count.get(item_name, 0) + 1

        # 按出現次數排序
        sorted_items = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items]

    @nextcord.slash_command(
        name="maple_monster",
        description="Search for monster drop information in MapleStory",
        name_localizations={
            Locale.zh_TW: "楓之谷怪物",
            Locale.zh_CN: "楓之谷怪物",
            Locale.ja: "メイプルモンスター",
        },
        description_localizations={
            Locale.zh_TW: "搜尋楓之谷怪物的掉落資訊",
            Locale.zh_CN: "搜尋楓之谷怪物的掉落資訊",
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
            name_localizations={
                Locale.zh_TW: "怪物名稱",
                Locale.zh_CN: "怪物名稱",
                Locale.ja: "モンスター名",
            },
            description_localizations={
                Locale.zh_TW: "要搜尋的怪物名稱",
                Locale.zh_CN: "要搜尋的怪物名稱",
                Locale.ja: "検索するモンスターの名前",
            },
            required=True,
        ),
    ) -> None:
        """搜尋怪物掉落資訊"""
        await interaction.response.defer()

        if not self.monsters_data:
            embed = Embed(
                title="❌ 錯誤", description="無法載入怪物資料，請聯絡管理員。", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        # 搜尋怪物
        monsters_found = list(self._search_monsters_by_name_cached(monster_name))

        if not monsters_found:
            embed = Embed(
                title="🔍 搜尋結果",
                description=f"找不到名稱包含「{monster_name}」的怪物。",
                color=0xFFAA00,
            )
            await interaction.followup.send(embed=embed)
            return

        if len(monsters_found) == 1:
            # 只有一個結果，直接顯示
            monster = monsters_found[0]
            view = MapleDropSearchView(self.monsters_data, "monster", monster_name)
            embed = view.create_monster_embed(monster)
            await interaction.followup.send(embed=embed)
        else:
            # 多個結果，使用選擇器
            embed = Embed(
                title="🔍 搜尋結果",
                description=f"找到 {len(monsters_found)} 個相關怪物，請選擇：",
                color=0x00AAFF,
            )

            view = MapleDropSearchView(self.monsters_data, "monster", monster_name)

            # 更新選擇器選項
            options = []
            for _i, monster in enumerate(monsters_found):  # Discord 限制最多25個選項
                level = monster.get("attributes", {}).get("level", "?")
                options.append(
                    SelectOption(
                        label=monster["name"], description=f"Lv.{level}", value=monster["name"]
                    )
                )

            view.select_result.options = options
            await interaction.followup.send(embed=embed, view=view)

    @nextcord.slash_command(
        name="maple_item",
        description="Search for item drop sources in MapleStory",
        name_localizations={
            Locale.zh_TW: "楓之谷物品",
            Locale.zh_CN: "楓之谷物品",
            Locale.ja: "メイプルアイテム",
        },
        description_localizations={
            Locale.zh_TW: "搜尋楓之谷物品的掉落來源",
            Locale.zh_CN: "搜尋楓之谷物品的掉落來源",
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
            name_localizations={
                Locale.zh_TW: "物品名稱",
                Locale.zh_CN: "物品名稱",
                Locale.ja: "アイテム名",
            },
            description_localizations={
                Locale.zh_TW: "要搜尋的物品名稱",
                Locale.zh_CN: "要搜尋的物品名稱",
                Locale.ja: "検索するアイテムの名前",
            },
            required=True,
        ),
    ) -> None:
        """搜尋物品掉落來源"""
        await interaction.response.defer()

        if not self.monsters_data:
            embed = Embed(
                title="❌ 錯誤", description="無法載入怪物資料，請聯絡管理員。", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        # 搜尋物品
        items_found = list(self._search_items_by_name_cached(item_name))

        if not items_found:
            embed = Embed(
                title="🔍 搜尋結果",
                description=f"找不到名稱包含「{item_name}」的物品。",
                color=0xFFAA00,
            )
            await interaction.followup.send(embed=embed)
            return

        if len(items_found) == 1:
            # 只有一個結果，直接顯示
            item = items_found[0]
            monsters_with_item = self.get_monsters_by_item(item)
            view = MapleDropSearchView(self.monsters_data, "item", item_name)
            embed = view.create_item_source_embed(item, monsters_with_item)
            await interaction.followup.send(embed=embed)
        else:
            # 多個結果，使用選擇器
            embed = Embed(
                title="🔍 搜尋結果",
                description=f"找到 {len(items_found)} 個相關物品，請選擇：",
                color=0x00AAFF,
            )

            view = MapleDropSearchView(self.monsters_data, "item", item_name)

            # 更新選擇器選項
            options = []
            for item in items_found:
                # 取得物品類型
                item_type = "未知"
                for monster in self.monsters_data:
                    for drop in monster.get("drops", []):
                        if drop["name"] == item:
                            item_type = drop.get("type", "未知")
                            break
                    if item_type != "未知":
                        break

                options.append(SelectOption(label=item, description=item_type, value=item))

            view.select_result.options = options
            await interaction.followup.send(embed=embed, view=view)

    @nextcord.slash_command(
        name="maple_stats",
        description="Get MapleStory database statistics",
        name_localizations={
            Locale.zh_TW: "楓之谷統計",
            Locale.zh_CN: "楓之谷統計",
            Locale.ja: "メイプル統計",
        },
        description_localizations={
            Locale.zh_TW: "顯示楓之谷資料庫統計資訊",
            Locale.zh_CN: "顯示楓之谷資料庫統計資訊",
            Locale.ja: "メイプルストーリーデータベース統計を表示",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def maple_stats(self, interaction: Interaction) -> None:
        """顯示資料庫統計資訊"""
        await interaction.response.defer()

        if not self.monsters_data:
            embed = Embed(
                title="❌ 錯誤", description="無法載入怪物資料，請聯絡管理員。", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        # 計算統計数據
        total_monsters = len(self.monsters_data)
        total_items = len({
            drop["name"] for monster in self.monsters_data for drop in monster.get("drops", [])
        })
        total_maps = len({
            map_name for monster in self.monsters_data for map_name in monster.get("maps", [])
        })

        # 計算等級分布
        level_counts: dict[str, int] = {}
        for monster in self.monsters_data:
            level = monster.get("attributes", {}).get("level", 0)
            level_range = f"{(level // 10) * 10}-{(level // 10) * 10 + 9}"
            level_counts[level_range] = level_counts.get(level_range, 0) + 1

        # 獲取熱門物品
        popular_items = self._get_popular_items()

        embed = Embed(
            title="📊 楓之谷資料庫統計", description="Artale 楓之谷資料庫概覽", color=0x00FF88
        )

        # 基本統計
        embed.add_field(
            name="📈 基本統計",
            value=BASIC_STATS_TEMPLATE.format(
                total_monsters=total_monsters, total_items=total_items, total_maps=total_maps
            ),
            inline=True,
        )

        # 等級分布 (顯示前5個)
        level_dist = "\n".join([
            f"**{level_range}級**: {count}隻"
            for level_range, count in sorted(level_counts.items())
        ])
        embed.add_field(name="🎯 等級分布", value=level_dist, inline=True)

        # 熱門掉落物品
        popular_text = "\n".join([f"• {item}" for item in popular_items])
        embed.add_field(name="🔥 熱門掉落物品", value=popular_text, inline=False)

        embed.set_footer(text="資料來源：Artale | 使用 /maple_monster 或 /maple_item 搜尋")
        await interaction.followup.send(embed=embed)


def setup(bot: commands.Bot) -> None:
    bot.add_cog(MapleStoryCogs(bot))
