from collections.abc import Iterable

import nextcord

from .models import Monster, MapleStats
from .constants import BASIC_STATS_TEMPLATE, MONSTER_ATTR_TEMPLATE


def create_monster_embed(monster: Monster) -> nextcord.Embed:
    embed = nextcord.Embed(
        title=f"🐲 {monster.get('name')}", description="怪物詳細資訊", color=0x00FF00
    )

    image = monster.get("image")
    if image:
        embed.set_thumbnail(url=image)

    attrs = monster.get("attributes", {})
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

    maps = monster.get("maps", [])
    if maps:
        maps_text = "\n".join(f"• {name}" for name in maps)
        embed.add_field(name="🗺️ 出現地圖", value=maps_text, inline=True)

    drops = monster.get("drops", [])
    if drops:
        equipment = [drop for drop in drops if drop.get("type") == "裝備"]
        consumables = [drop for drop in drops if drop.get("type") == "消耗品/素材"]

        if equipment:
            equip_text = "\n".join(f"• {item.get('name')}" for item in equipment)
            embed.add_field(name="⚔️ 裝備掉落", value=equip_text, inline=False)

        if consumables:
            cons_text = "\n".join(f"• {item.get('name')}" for item in consumables)
            embed.add_field(name="🧪 消耗品/素材", value=cons_text, inline=False)

    embed.set_footer(text="資料來源：Artale")
    return embed


def create_item_source_embed(item_name: str, monsters: Iterable[Monster]) -> nextcord.Embed:
    monsters_list = list(monsters)
    embed = nextcord.Embed(title=f"🎁 {item_name}", description="物品掉落來源", color=0x0099FF)

    item_img = None
    item_link = None
    for monster in monsters_list:
        for drop in monster.get("drops", []):
            if drop.get("name") == item_name:
                item_img = drop.get("img")
                item_link = drop.get("link")
                break
        if item_img:
            break

    if item_img:
        embed.set_thumbnail(url=item_img)

    if item_link:
        embed.add_field(name="🔗 詳細資訊", value=f"[查看詳細資料]({item_link})", inline=False)

    monster_lines: list[str] = []
    for monster in monsters_list:
        attrs = monster.get("attributes", {})
        level = attrs.get("level", "?")
        monster_lines.append(f"• **{monster.get('name')}** (Lv.{level})")

    if monster_lines:
        embed.add_field(name="🐲 掉落來源怪物", value="\n".join(monster_lines), inline=False)

    embed.set_footer(text="資料來源：Artale")
    return embed


def build_stats_embed(stats: MapleStats) -> nextcord.Embed:
    embed = nextcord.Embed(
        title="📊 楓之谷資料庫統計", description="Artale 楓之谷資料庫概覽", color=0x00FF88
    )

    embed.add_field(
        name="📈 基本統計",
        value=BASIC_STATS_TEMPLATE.format(
            total_monsters=stats.total_monsters,
            total_items=stats.total_items,
            total_maps=stats.total_maps,
        ),
        inline=True,
    )

    if stats.level_distribution:
        level_dist = "\n".join(
            f"**{level_range}級**: {count}隻"
            for level_range, count in sorted(stats.level_distribution.items())
        )
        embed.add_field(name="🎯 等級分布", value=level_dist, inline=True)

    if stats.popular_items:
        popular_text = "\n".join(f"• {item}" for item in stats.popular_items)
        embed.add_field(name="🔥 熱門掉落物品", value=popular_text, inline=False)

    embed.set_footer(text="資料來源：Artale | 使用 /maple_monster 或 /maple_item 搜尋")
    return embed
