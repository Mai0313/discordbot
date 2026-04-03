from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from nextcord import Embed

from .constants import BASIC_STATS_TEMPLATE, MONSTER_ATTR_TEMPLATE

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .models import (
        NPC,
        Quest,
        Scroll,
        Monster,
        MapEntry,
        Equipment,
        QuestStep,
        MapleStats,
        AcquisitionNPC,
        AcquisitionQuest,
        AcquisitionMonster,
    )

SITE = "https://www.artalemaplestory.com"


class TranslateFn(Protocol):
    def __call__(self, category: str, name: str) -> str: ...


def _identity(category: str, name: str) -> str:
    return name


def _truncate(text: str, limit: int = 1024) -> str:
    return text[: limit - 3] + "..." if len(text) > limit else text


def _translate_map_name(name: str, translate: TranslateFn) -> str:
    """Translate composite map names like 'Amherst > Weapon Store'."""
    parts = [translate(category="maps", name=p.strip()) for p in name.split(" > ")]
    return " > ".join(parts)


def _add_acquisition_fields(
    embed: Embed,
    acq_monsters: Sequence[AcquisitionMonster],
    acq_npcs: Sequence[AcquisitionNPC],
    acq_quests: Sequence[AcquisitionQuest],
    translate: TranslateFn,
) -> None:
    """Add monster/NPC/quest acquisition fields to an embed."""
    if acq_monsters:
        text = "\n".join(
            f"• {translate(category='monsters', name=m.name)} (Lv.{m.level})"
            for m in acq_monsters[:10]
        )
        embed.add_field(name="\U0001f432 怪物掉落", value=_truncate(text), inline=True)

    if acq_npcs:
        text = "\n".join(
            f"• {translate(category='npcs', name=n.name)} ({n.price:,} 楓幣)" for n in acq_npcs[:8]
        )
        embed.add_field(name="\U0001f6d2 NPC 商店", value=_truncate(text), inline=True)

    if acq_quests:
        text = "\n".join(
            f"• {translate(category='quests', name=q.name)} (Lv.{q.level})" for q in acq_quests[:5]
        )
        embed.add_field(name="\U0001f4cb 任務獎勵", value=_truncate(text), inline=False)


# ── Monster ─────────────────────────────────────────────────────────


def create_monster_embed(monster: Monster, *, translate: TranslateFn = _identity) -> Embed:
    embed = Embed(
        title=f"\U0001f432 {monster.display_name}",
        description=f"Lv. {monster.level}",
        url=f"{SITE}/zh/monsters/{monster.name.lower().replace(' ', '-')}",
        color=0x00FF00,
    )
    embed.set_thumbnail(url=f"{SITE}/images/monsters/{monster.name.lower().replace(' ', '-')}.gif")

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
        mod_text = ", ".join(translate(category="modifiers", name=m) for m in monster.modifiers)
        embed.add_field(name="\U0001f300 屬性抗性", value=mod_text, inline=True)

    if monster.region_to_maps_list:
        lines: list[str] = []
        for region in monster.region_to_maps_list:
            region_zh = translate(category="region", name=region.region)
            map_names = [_translate_map_name(name=m, translate=translate) for m in region.maps[:5]]
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
            f"• {translate(category='equipment', name=d.name)}" for d in drops.equipment_items[:10]
        )
        embed.add_field(name="\u2694\ufe0f 裝備掉落", value=_truncate(text), inline=True)

    consumables = drops.useable_items + drops.misc_items
    if consumables:
        text = "\n".join(
            f"• {translate(category='useable', name=d.name) if d in drops.useable_items else translate(category='misc', name=d.name)}"
            for d in consumables[:10]
        )
        embed.add_field(name="\U0001f9ea 消耗/素材", value=_truncate(text), inline=True)

    if drops.scrolls:
        text = "\n".join(
            f"• {translate(category='scrolls', name=d.name)}" for d in drops.scrolls[:10]
        )
        embed.add_field(name="\U0001f4dc 捲軸掉落", value=_truncate(text), inline=True)

    if monster.quests:
        text = "\n".join(
            f"• {translate(category='quests', name=q.name)} (Lv.{q.level})"
            for q in monster.quests[:5]
        )
        embed.add_field(name="\U0001f4cb 相關任務", value=_truncate(text), inline=False)

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── Equipment ───────────────────────────────────────────────────────


def _add_equip_stats(embed: Embed, equip: Equipment) -> None:
    stats = equip.stats.non_zero_stats()
    if not stats:
        return
    lines = [f"**{label}**: {sv.middle}" for label, sv in stats]
    if equip.stats.upgrade_slots is not None:
        lines.append(f"**Upgrade Slots**: {equip.stats.upgrade_slots}")
    if equip.attack_speed:
        lines.append(f"**Attack Speed**: {equip.attack_speed}")
    embed.add_field(name="\U0001f4ca 屬性", value="\n".join(lines), inline=True)


def _add_equip_requirements(embed: Embed, equip: Equipment) -> None:
    req = equip.equipment_restriction
    if not req.has_requirements():
        return
    parts = []
    for label, val in [
        ("STR", req.str_req),
        ("DEX", req.dex),
        ("INT", req.int_req),
        ("LUK", req.luk),
    ]:
        if val:
            parts.append(f"{label}: {val}")
    embed.add_field(name="\U0001f4cf 需求", value="\n".join(parts), inline=True)


def _add_equip_tags(embed: Embed, equip: Equipment) -> None:
    tags = [
        t
        for t in [
            equip.tradeable,
            "EVENT" if equip.event else "",
            "UNAVAILABLE" if equip.unavailable else "",
        ]
        if t
    ]
    if tags:
        embed.add_field(name="\U0001f3f7\ufe0f 標籤", value=" | ".join(tags), inline=False)


def create_equipment_embed(equip: Equipment, *, translate: TranslateFn = _identity) -> Embed:
    slug = equip.name.lower().replace(" ", "-")
    type_slug = equip.type.lower().replace(" ", "-")

    embed = Embed(
        title=f"\u2694\ufe0f {equip.display_name}",
        description=f"Lv. {equip.level} | {translate(category='eqType', name=equip.type)}",
        url=f"{SITE}/zh/equipment/{type_slug}/{slug}",
        color=0xFF9900,
    )
    embed.set_thumbnail(url=f"{SITE}/images/equipment/{type_slug}/{slug}.webp")

    _add_equip_stats(embed=embed, equip=equip)
    _add_equip_requirements(embed=embed, equip=equip)

    if equip.jobs:
        job_text = ", ".join(translate(category="job", name=j) for j in equip.jobs)
        embed.add_field(name="\U0001f464 職業", value=job_text, inline=False)

    acq = equip.acquisition
    _add_acquisition_fields(
        embed=embed,
        acq_monsters=acq.monsters,
        acq_npcs=acq.npcs,
        acq_quests=acq.quests,
        translate=translate,
    )
    _add_equip_tags(embed=embed, equip=equip)

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── Scroll ──────────────────────────────────────────────────────────

_STAT_LABELS = {
    "str": "STR",
    "dex": "DEX",
    "int": "INT",
    "luk": "LUK",
    "hp": "HP",
    "mp": "MP",
    "atk": "ATK",
    "matk": "M.ATK",
    "def": "DEF",
    "mdef": "M.DEF",
    "accuracy": "Accuracy",
    "avoidability": "Avoidability",
    "speed": "Speed",
    "jump": "Jump",
}


def create_scroll_embed(scroll: Scroll, *, translate: TranslateFn = _identity) -> Embed:
    embed = Embed(
        title=f"\U0001f4dc {scroll.display_name}",
        description=f"適用: {translate(category='eqType', name=scroll.type)}"
        if scroll.type
        else "",
        color=0x9966FF,
    )

    if scroll.stats:
        lines = [f"**{_STAT_LABELS.get(k, k)}**: +{v}" for k, v in scroll.stats.items()]
        embed.add_field(name="\U0001f4ca 屬性加成", value="\n".join(lines), inline=True)

    acq = scroll.acquisition
    _add_acquisition_fields(
        embed=embed,
        acq_monsters=acq.monsters,
        acq_npcs=acq.npcs,
        acq_quests=acq.quests,
        translate=translate,
    )

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── NPC ─────────────────────────────────────────────────────────────


def create_npc_embed(npc: NPC, *, translate: TranslateFn = _identity) -> Embed:
    npc_type_zh = translate(category="npcType", name=npc.type) if npc.type else ""
    embed = Embed(title=f"\U0001f464 {npc.display_name}", description=npc_type_zh, color=0x00CCFF)

    if npc.region_to_maps_list:
        lines: list[str] = []
        for region in npc.region_to_maps_list:
            region_zh = translate(category="region", name=region.region)
            lines.append(
                f"**{region_zh}**: {', '.join(_translate_map_name(name=m, translate=translate) for m in region.maps)}"
            )
        embed.add_field(
            name="\U0001f5fa\ufe0f 位置", value=_truncate("\n".join(lines)), inline=False
        )

    if npc.equipment_items:
        text = "\n".join(
            f"• {translate(category='equipment', name=i.name.split('/')[-1])} ({i.price:,} 楓幣)"
            for i in npc.equipment_items[:10]
        )
        embed.add_field(name="\u2694\ufe0f 販售裝備", value=_truncate(text), inline=True)

    if npc.useable_items:
        text = "\n".join(
            f"• {translate(category='useable', name=i.name.split('/')[-1])} ({i.price:,} 楓幣)"
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


def _format_quest_step(step: QuestStep, translate: TranslateFn) -> list[str]:
    """Format a single quest step into display lines."""
    lines: list[str] = []
    if step.start_npc:
        lines.append(f"**NPC**: {translate(category='npcs', name=step.start_npc)}")
    for target in step.monsters_to_hunt[:5]:
        lines.append(
            f"• 狩獵 {translate(category='monsters', name=target.name)} x{target.quantity}"
        )
    for items in step.items_to_collect.values():
        for item in items[:3]:
            lines.append(f"• 收集 {translate(category='misc', name=item.name)} x{item.quantity}")
    reward = step.reward
    parts: list[str] = []
    if reward.exp:
        parts.append(f"EXP: {reward.exp:,}")
    if reward.fame:
        parts.append(f"Fame: {reward.fame}")
    if parts:
        lines.append(f"**獎勵**: {' | '.join(parts)}")
    return lines


def create_quest_embed(quest: Quest, *, translate: TranslateFn = _identity) -> Embed:
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
        lines = _format_quest_step(step=step, translate=translate)
        if lines:
            embed.add_field(
                name=f"步驟 {i}" if len(quest.steps) > 1 else "任務內容",
                value=_truncate("\n".join(lines)),
                inline=False,
            )

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── Map ─────────────────────────────────────────────────────────────


def create_map_embed(map_entry: MapEntry, *, translate: TranslateFn = _identity) -> Embed:
    region_zh = translate(category="region", name=map_entry.region)
    embed = Embed(
        title=f"\U0001f5fa\ufe0f {map_entry.display_name}",
        description=f"Region: {region_zh}",
        color=0x33CC33,
    )

    if map_entry.monsters:
        text = "\n".join(
            f"• {translate(category='monsters', name=m.name)} (Lv.{m.level})"
            for m in map_entry.monsters[:10]
        )
        embed.add_field(name="\U0001f432 怪物", value=_truncate(text), inline=True)

    if map_entry.npcs:
        text = "\n".join(
            f"• {translate(category='npcs', name=n.name)}" for n in map_entry.npcs[:10]
        )
        embed.add_field(name="\U0001f464 NPC", value=_truncate(text), inline=True)

    if map_entry.hidden:
        embed.add_field(name="\U0001f510 隱藏地圖", value="是", inline=True)

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── Item Source ─────────────────────────────────────────────────────


def create_item_source_embed(
    item_name: str, monsters: Iterable[Monster], *, translate: TranslateFn = _identity
) -> Embed:
    monsters_list = list(monsters)
    item_zh = (
        translate(category="equipment", name=item_name)
        or translate(category="scrolls", name=item_name)
        or translate(category="useable", name=item_name)
        or translate(category="misc", name=item_name)
    )
    display = item_zh if item_zh != item_name else item_name

    embed = Embed(title=f"\U0001f381 {display}", description="物品掉落來源", color=0x0099FF)

    lines: list[str] = []
    for monster in monsters_list[:15]:
        lines.append(f"• **{monster.display_name}** (Lv.{monster.level})")
    if lines:
        embed.add_field(name="\U0001f432 掉落來源怪物", value="\n".join(lines), inline=False)

    embed.set_footer(text="資料來源：Artale")
    return embed


# ── Stats ───────────────────────────────────────────────────────────


def build_stats_embed(stats: MapleStats) -> Embed:
    embed = Embed(
        title="\U0001f4ca 楓之谷資料庫統計", description="Artale 楓之谷資料庫概覽", color=0x00FF88
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
            f"**{r}級**: {c}隻" for r, c in sorted(stats.level_distribution.items())
        )
        embed.add_field(name="\U0001f3af 等級分布", value=level_dist, inline=True)

    if stats.popular_items:
        popular_text = "\n".join(f"• {item}" for item in stats.popular_items[:15])
        embed.add_field(name="\U0001f525 熱門掉落物品", value=popular_text, inline=False)

    embed.set_footer(text="資料來源：Artale | 使用 /maple_monster 或 /maple_item 搜尋")
    return embed
