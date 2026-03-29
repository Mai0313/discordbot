from __future__ import annotations

from collections.abc import Iterable

from nextcord import Embed

from .constants import BASIC_STATS_TEMPLATE, MONSTER_ATTR_TEMPLATE
from .models import (
    Equipment,
    MapEntry,
    MapleStats,
    Monster,
    NPC,
    Quest,
    Scroll,
)

SITE = "https://www.artalemaplestory.com"


def _truncate(text: str, limit: int = 1024) -> str:
    return text[: limit - 3] + "..." if len(text) > limit else text


def _translate_map_name(name: str, translate: callable) -> str:
    """Translate composite map names like 'Amherst > Weapon Store'."""
    parts = [translate("maps", p.strip()) for p in name.split(" > ")]
    return " > ".join(parts)


# ── Monster ─────────────────────────────────────────────────────────


def create_monster_embed(monster: Monster, *, translate: callable = str) -> Embed:
    embed = Embed(
        title=f"\U0001f432 {monster.display_name}",
        description=f"Lv. {monster.level}",
        url=f"{SITE}/zh/monsters/{monster.name.lower().replace(' ', '-')}",
        color=0x00FF00,
    )
    embed.set_thumbnail(
        url=f"{SITE}/images/monsters/{monster.name.lower().replace(' ', '-')}.gif"
    )

    meso = monster.drops.meso_range
    attr_text = MONSTER_ATTR_TEMPLATE.format(
        level=monster.level,
        hp=monster.hp,
        mp=monster.mp,
        exp=monster.exp,
        weapon_def=monster.def_stats.weapon,
        magic_def=monster.def_stats.magic,
        avoidability=monster.def_stats.avoidability,
        accuracy_required=monster.accuracy.required,
        meso_range=f"{meso[0]:,} ~ {meso[1]:,}" if len(meso) == 2 else "N/A",
    )
    embed.add_field(name="\U0001f4ca 屬性", value=attr_text, inline=True)

    if monster.modifiers:
        mod_text = ", ".join(translate("modifiers", m) for m in monster.modifiers)
        embed.add_field(name="\U0001f300 屬性抗性", value=mod_text, inline=True)

    if monster.region_to_maps_list:
        lines: list[str] = []
        for region in monster.region_to_maps_list:
            region_zh = translate("region", region.region)
            map_names = [_translate_map_name(m, translate) for m in region.maps[:5]]
            lines.append(f"**{region_zh}**")
            lines.extend(f"• {n}" for n in map_names)
            if len(region.maps) > 5:
                lines.append(f"  ⋯ 等 {len(region.maps)} 個地圖")
        embed.add_field(
            name="\U0001f5fa\ufe0f 出現地圖", value=_truncate("\n".join(lines)), inline=False
        )

    drops = monster.drops
    if drops.equipment_items:
        text = "\n".join(
            f"• {translate('equipment', d.name)}" for d in drops.equipment_items[:10]
        )
        embed.add_field(name="\u2694\ufe0f 裝備掉落", value=_truncate(text), inline=True)

    consumables = drops.useable_items + drops.misc_items
    if consumables:
        text = "\n".join(
            f"• {translate('useable', d.name) if d in drops.useable_items else translate('misc', d.name)}"
            for d in consumables[:10]
        )
        embed.add_field(name="\U0001f9ea 消耗/素材", value=_truncate(text), inline=True)

    if drops.scrolls:
        text = "\n".join(
            f"• {translate('scrolls', d.name)}" for d in drops.scrolls[:10]
        )
        embed.add_field(name="\U0001f4dc 捲軸掉落", value=_truncate(text), inline=True)

    if monster.quests:
        text = "\n".join(
            f"• {translate('quests', q.name)} (Lv.{q.level})" for q in monster.quests[:5]
        )
        embed.add_field(name="\U0001f4cb 相關任務", value=_truncate(text), inline=False)

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── Equipment ───────────────────────────────────────────────────────


def create_equipment_embed(equip: Equipment, *, translate: callable = str) -> Embed:
    slug = equip.name.lower().replace(" ", "-")
    type_slug = equip.type.lower().replace(" ", "-")

    embed = Embed(
        title=f"\u2694\ufe0f {equip.display_name}",
        description=f"Lv. {equip.level} | {translate('eqType', equip.type)}",
        url=f"{SITE}/zh/equipment/{type_slug}/{slug}",
        color=0xFF9900,
    )
    embed.set_thumbnail(url=f"{SITE}/images/equipment/{type_slug}/{slug}.webp")

    # Stats
    stats = equip.stats.non_zero_stats()
    if stats:
        lines = [f"**{label}**: {sv.middle}" for label, sv in stats]
        if equip.stats.upgrade_slots is not None:
            lines.append(f"**Upgrade Slots**: {equip.stats.upgrade_slots}")
        if equip.attack_speed:
            lines.append(f"**Attack Speed**: {equip.attack_speed}")
        embed.add_field(name="\U0001f4ca 屬性", value="\n".join(lines), inline=True)

    # Requirements
    req = equip.equipment_restriction
    if req.has_requirements():
        lines = []
        if req.str_req:
            lines.append(f"STR: {req.str_req}")
        if req.dex:
            lines.append(f"DEX: {req.dex}")
        if req.int_req:
            lines.append(f"INT: {req.int_req}")
        if req.luk:
            lines.append(f"LUK: {req.luk}")
        embed.add_field(name="\U0001f4cf 需求", value="\n".join(lines), inline=True)

    # Jobs
    if equip.jobs:
        job_text = ", ".join(translate("job", j) for j in equip.jobs)
        embed.add_field(name="\U0001f464 職業", value=job_text, inline=False)

    # Acquisition
    acq = equip.acquisition
    if acq.monsters:
        text = "\n".join(
            f"• {translate('monsters', m.name)} (Lv.{m.level})"
            for m in acq.monsters[:8]
        )
        embed.add_field(name="\U0001f432 怪物掉落", value=_truncate(text), inline=True)

    if acq.npcs:
        text = "\n".join(
            f"• {translate('npcs', n.name)} ({n.price:,} 楓幣)"
            for n in acq.npcs[:8]
        )
        embed.add_field(name="\U0001f6d2 NPC 商店", value=_truncate(text), inline=True)

    if acq.quests:
        text = "\n".join(
            f"• {translate('quests', q.name)} (Lv.{q.level})"
            for q in acq.quests[:5]
        )
        embed.add_field(name="\U0001f4cb 任務獎勵", value=_truncate(text), inline=False)

    # Tags
    tags: list[str] = []
    if equip.tradeable:
        tags.append(equip.tradeable)
    if equip.event:
        tags.append("EVENT")
    if equip.unavailable:
        tags.append("UNAVAILABLE")
    if tags:
        embed.add_field(name="\U0001f3f7\ufe0f 標籤", value=" | ".join(tags), inline=False)

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── Scroll ──────────────────────────────────────────────────────────


def create_scroll_embed(scroll: Scroll, *, translate: callable = str) -> Embed:
    embed = Embed(
        title=f"\U0001f4dc {scroll.display_name}",
        description=f"適用: {translate('eqType', scroll.type)}" if scroll.type else "",
        color=0x9966FF,
    )

    if scroll.stats:
        stat_names = {"str": "STR", "dex": "DEX", "int": "INT", "luk": "LUK",
                      "hp": "HP", "mp": "MP", "atk": "ATK", "matk": "M.ATK",
                      "def": "DEF", "mdef": "M.DEF", "accuracy": "Accuracy",
                      "avoidability": "Avoidability", "speed": "Speed", "jump": "Jump"}
        lines = [f"**{stat_names.get(k, k)}**: +{v}" for k, v in scroll.stats.items()]
        embed.add_field(name="\U0001f4ca 屬性加成", value="\n".join(lines), inline=True)

    acq = scroll.acquisition
    if acq.monsters:
        text = "\n".join(
            f"• {translate('monsters', m.name)} (Lv.{m.level})"
            for m in acq.monsters[:10]
        )
        embed.add_field(name="\U0001f432 怪物掉落", value=_truncate(text), inline=True)

    if acq.npcs:
        text = "\n".join(
            f"• {translate('npcs', n.name)} ({n.price:,} 楓幣)"
            for n in acq.npcs[:5]
        )
        embed.add_field(name="\U0001f6d2 NPC 商店", value=_truncate(text), inline=True)

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── NPC ─────────────────────────────────────────────────────────────


def create_npc_embed(npc: NPC, *, translate: callable = str) -> Embed:
    npc_type_zh = translate("npcType", npc.type) if npc.type else ""
    embed = Embed(
        title=f"\U0001f464 {npc.display_name}",
        description=npc_type_zh,
        color=0x00CCFF,
    )

    if npc.region_to_maps_list:
        lines: list[str] = []
        for region in npc.region_to_maps_list:
            region_zh = translate("region", region.region)
            lines.append(f"**{region_zh}**: {', '.join(_translate_map_name(m, translate) for m in region.maps)}")
        embed.add_field(
            name="\U0001f5fa\ufe0f 位置", value=_truncate("\n".join(lines)), inline=False
        )

    if npc.equipment_items:
        text = "\n".join(
            f"• {translate('equipment', i.name.split('/')[-1])} ({i.price:,} 楓幣)"
            for i in npc.equipment_items[:10]
        )
        embed.add_field(name="\u2694\ufe0f 販售裝備", value=_truncate(text), inline=True)

    if npc.useable_items:
        text = "\n".join(
            f"• {translate('useable', i.name.split('/')[-1])} ({i.price:,} 楓幣)"
            for i in npc.useable_items[:10]
        )
        embed.add_field(name="\U0001f9ea 販售消耗品", value=_truncate(text), inline=True)

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── Quest ───────────────────────────────────────────────────────────

FREQ_ZH = {
    "one-time": "一次性",
    "daily": "每日",
    "12hr": "每12小時",
    "6hr": "每6小時",
    "2hr": "每2小時",
    "1hr": "每1小時",
    "exchange": "交換",
}


def create_quest_embed(quest: Quest, *, translate: callable = str) -> Embed:
    freq = FREQ_ZH.get(quest.frequency, quest.frequency)
    level_text = f"Lv. {quest.lv_lower}"
    if quest.lv_upper:
        level_text += f" ~ {quest.lv_upper}"

    embed = Embed(
        title=f"\U0001f4cb {quest.display_name}",
        description=f"{level_text} | {freq}",
        color=0xFFCC00,
    )

    for i, step in enumerate(quest.steps[:3], 1):
        lines: list[str] = []
        if step.start_npc:
            lines.append(f"**NPC**: {translate('npcs', step.start_npc)}")
        if step.monsters_to_hunt:
            for target in step.monsters_to_hunt[:5]:
                lines.append(f"• 狩獵 {translate('monsters', target.name)} x{target.quantity}")
        if step.items_to_collect:
            for cat, items in step.items_to_collect.items():
                for item in items[:3]:
                    name = item.get("name", "")
                    qty = item.get("quantity", 0)
                    lines.append(f"• 收集 {translate('misc', name)} x{qty}")
        if step.reward:
            reward_parts: list[str] = []
            if "exp" in step.reward:
                reward_parts.append(f"EXP: {step.reward['exp']:,}")
            if "fame" in step.reward:
                reward_parts.append(f"Fame: {step.reward['fame']}")
            if reward_parts:
                lines.append(f"**獎勵**: {' | '.join(reward_parts)}")

        if lines:
            embed.add_field(
                name=f"步驟 {i}" if len(quest.steps) > 1 else "任務內容",
                value=_truncate("\n".join(lines)),
                inline=False,
            )

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── Map ─────────────────────────────────────────────────────────────


def create_map_embed(map_entry: MapEntry, *, translate: callable = str) -> Embed:
    region_zh = translate("region", map_entry.region)
    embed = Embed(
        title=f"\U0001f5fa\ufe0f {map_entry.display_name}",
        description=f"Region: {region_zh}",
        color=0x33CC33,
    )

    if map_entry.monsters:
        text = "\n".join(
            f"• {translate('monsters', m.name)} (Lv.{m.level})"
            for m in map_entry.monsters[:10]
        )
        embed.add_field(name="\U0001f432 怪物", value=_truncate(text), inline=True)

    if map_entry.npcs:
        text = "\n".join(
            f"• {translate('npcs', n.name)}" for n in map_entry.npcs[:10]
        )
        embed.add_field(name="\U0001f464 NPC", value=_truncate(text), inline=True)

    if map_entry.hidden:
        embed.add_field(name="\U0001f510 隱藏地圖", value="是", inline=True)

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── Item Source ─────────────────────────────────────────────────────


def create_item_source_embed(
    item_name: str,
    monsters: Iterable[Monster],
    *,
    translate: callable = str,
) -> Embed:
    monsters_list = list(monsters)
    item_zh = (
        translate("equipment", item_name)
        or translate("scrolls", item_name)
        or translate("useable", item_name)
        or translate("misc", item_name)
    )
    display = item_zh if item_zh != item_name else item_name

    embed = Embed(
        title=f"\U0001f381 {display}", description="物品掉落來源", color=0x0099FF
    )

    lines: list[str] = []
    for monster in monsters_list[:15]:
        lines.append(f"• **{monster.display_name}** (Lv.{monster.level})")
    if lines:
        embed.add_field(
            name="\U0001f432 掉落來源怪物", value="\n".join(lines), inline=False
        )

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── Stats ───────────────────────────────────────────────────────────


def build_stats_embed(stats: MapleStats) -> Embed:
    embed = Embed(
        title="\U0001f4ca 楓之谷資料庫統計",
        description="Artale 楓之谷資料庫概覽",
        color=0x00FF88,
    )
    embed.add_field(
        name="\U0001f4c8 資料總覽",
        value=BASIC_STATS_TEMPLATE.format(
            total_monsters=stats.total_monsters,
            total_equipment=stats.total_equipment,
            total_scrolls=stats.total_scrolls,
            total_useable=stats.total_useable,
            total_npcs=stats.total_npcs,
            total_quests=stats.total_quests,
            total_maps=stats.total_maps,
            total_misc=stats.total_misc,
        ),
        inline=True,
    )

    if stats.level_distribution:
        level_dist = "\n".join(
            f"**{r}級**: {c}隻"
            for r, c in sorted(stats.level_distribution.items())
        )
        embed.add_field(name="\U0001f3af 等級分布", value=level_dist, inline=True)

    if stats.popular_items:
        popular_text = "\n".join(f"• {item}" for item in stats.popular_items[:15])
        embed.add_field(
            name="\U0001f525 熱門掉落物品", value=popular_text, inline=False
        )

    embed.set_footer(
        text="資料來源：Artale | 使用 /maple_monster 或 /maple_item 搜尋"
    )
    return embed
