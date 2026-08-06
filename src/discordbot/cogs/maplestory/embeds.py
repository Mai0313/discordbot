"""Discord embed builders for every `/maplestory` result.

One builder per entity the cog can look up (monster, equipment, scroll, NPC, quest, map, the
drop sources of an item) plus `build_stats_embed` for the loaded data set's summary. They are
pure: a model in, an `Embed` out, no Interaction and no service. That is what lets `cog.py` and
`views.py` render the same result from two places — the command answers a single hit, the select
menu re-renders whichever hit the user picked — and what lets the tests assert on fields with no
Discord in the loop.

The data files are English and the surface is Traditional Chinese, so each builder takes a
`TranslateFn` (`MapleStoryService.translate` in production, `_identity` by default so a builder
stays callable with no data loaded) and translates names at render time; the headings, footers
and units written here are the only Chinese the module owns. An entity's own `display_name`
already prefers its Chinese name and is used as-is, while the site url and thumbnail path are
slugged from the English `name`, which is what `SITE` keys its pages on.

Discord's limits are honoured here rather than at the send: `_truncate` keeps each field value
under the 1024-character cap, and every list is cut to a fixed head (10 drops, 8 shop NPCs, 15
drop sources, 3 quest steps, ...) so a densely populated entity degrades to a prefix rather than
tripping a Discord 400. A section whose source list is empty is skipped entirely, so a sparse
entity yields a short embed instead of blank fields.
"""

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
    """Call shape of the name translator every builder renders through.

    `MapleStoryService.translate` satisfies it, and `views.py` types its resolvers against it so
    the builders never have to import the service.
    """

    def __call__(self, category: str, name: str) -> str:
        """Translates one MapleStory name within a translation category.

        Both implementations reachable here return the source name on a miss, so a caller cannot
        tell an untranslated name from one that translates to itself.

        Args:
            category (str): Translation table to look up, e.g. `monsters`, `equipment`, `region`.
            name (str): Source name as stored in the data files.

        Returns:
            The translated name, or the source name when the table has no entry for it.
        """
        ...


def _identity(category: str, name: str) -> str:
    """Returns the name unchanged, ignoring the category.

    The default translator, so every builder stays callable without a loaded service.

    Args:
        category (str): Translation category, unused.
        name (str): Source name.

    Returns:
        `name`, unchanged.
    """
    return name


def _truncate(text: str, limit: int = 1024) -> str:
    """Truncates text to an embed field's value limit, marking the cut with an ellipsis.

    The ellipsis is counted inside `limit`, so the result never exceeds it and never 400s the
    send. The default is Discord's 1024-character cap for a field value.

    Args:
        text (str): Text destined for an embed field value.
        limit (int): Maximum length of the returned string.

    Returns:
        `text` unchanged, or its first `limit - 3` characters followed by an ellipsis.
    """
    return text[: limit - 3] + "..." if len(text) > limit else text


def _translate_map_name(name: str, translate: TranslateFn) -> str:
    """Translates a composite map name such as `Amherst > Weapon Store` segment by segment.

    The data stores a map as a ` > `-joined path while the `maps` table is keyed on the
    individual names, so translating the joined string whole would always miss.

    Args:
        name (str): Map name, possibly a ` > `-joined path.
        translate (TranslateFn): Translator applied to each segment.

    Returns:
        The path with every segment translated and rejoined with ` > `.
    """
    parts = [translate(category="maps", name=p.strip()) for p in name.split(" > ")]
    return " > ".join(parts)


def _add_acquisition_fields(
    embed: Embed,
    acq_monsters: Sequence[AcquisitionMonster],
    acq_npcs: Sequence[AcquisitionNPC],
    acq_quests: Sequence[AcquisitionQuest],
    translate: TranslateFn,
) -> None:
    """Adds the drop / shop / quest-reward fields shared by the equipment and scroll embeds.

    Mutates `embed` in place, skipping any source that is empty. Each list is cut (10 monsters,
    8 NPCs, 5 quests) before the field cap applies.

    Args:
        embed (Embed): Embed to add the fields to.
        acq_monsters (Sequence[AcquisitionMonster]): Monsters that drop the item.
        acq_npcs (Sequence[AcquisitionNPC]): NPCs that sell it, rendered with their price.
        acq_quests (Sequence[AcquisitionQuest]): Quests that hand it out as a reward.
        translate (TranslateFn): Translator for the monster, NPC and quest names.
    """
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


def create_monster_embed(monster: Monster, translate: TranslateFn = _identity) -> Embed:
    """Builds the `/maplestory monster` embed: stats, resistances, spawn maps, drops and quests.

    Spawn maps list the first 5 of each region and append that region's total count, so a
    monster that spawns everywhere still shows how much was left out. Everything else is a plain
    head (10 items per drop kind, 5 quests). A monster whose data carries no meso range shows
    `N/A` rather than an empty stat line.

    Args:
        monster (Monster): The monster to render.
        translate (TranslateFn): Translator for the modifier, region, map, drop and quest names.

    Returns:
        An embed titled with the monster's display name, linking to its page on the site.
    """
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

    # Useable and misc drops share one field but not one translation table, so the table is
    # chosen per item.
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
    """Adds the equipment stat field, listing only the stats with a non-zero middle value.

    Mutates `embed` in place. Upgrade slots and the attack-speed label ride the same field
    although neither is part of `non_zero_stats` — and the label is `Equipment.attack_speed`,
    not the unrelated `EquipmentStats.attack_speed` number. Both are lost when the item has no
    non-zero stat at all, since the whole field is skipped then.

    Args:
        embed (Embed): Embed to add the field to.
        equip (Equipment): The equipment whose stats are rendered.
    """
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
    """Adds the stat-requirement field for equipment that requires anything.

    Mutates `embed` in place; an item with every requirement at zero gets no field. The stat
    labels are fixed English, matching the stat field above.

    Args:
        embed (Embed): Embed to add the field to.
        equip (Equipment): The equipment whose requirements are rendered.
    """
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
    """Adds the tag field: the tradeability label plus the EVENT and UNAVAILABLE markers.

    Mutates `embed` in place. The markers are raw data labels rather than translated text, and
    an item carrying none of the three gets no field.

    Args:
        embed (Embed): Embed to add the field to.
        equip (Equipment): The equipment whose flags are rendered.
    """
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


def create_equipment_embed(equip: Equipment, translate: TranslateFn = _identity) -> Embed:
    """Builds the `/maplestory equip` embed: stats, requirements, jobs, sources and tags.

    Each `_add_equip_*` helper adds nothing when its own source is empty, so the section order
    is fixed but the section set is not. Both the page url and the thumbnail are slugged from
    the English `type` and `name`, which is how the site addresses an item.

    Args:
        equip (Equipment): The equipment to render.
        translate (TranslateFn): Translator for the equipment type, job and source names.

    Returns:
        An embed titled with the equipment's display name, linking to its page on the site.
    """
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


def create_scroll_embed(scroll: Scroll, translate: TranslateFn = _identity) -> Embed:
    """Builds the `/maplestory scroll` embed: the stat bonuses and where the scroll comes from.

    A scroll with no applicable equipment type gets an empty description rather than a dangling
    prefix. Bonus keys are rendered through `_STAT_LABELS` and fall back to the raw key, so a
    stat the map has not caught up with is still shown. Unlike the monster and equipment embeds
    this one carries no site url and no thumbnail.

    Args:
        scroll (Scroll): The scroll to render.
        translate (TranslateFn): Translator for the applicable equipment type and source names.

    Returns:
        An embed titled with the scroll's display name.
    """
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


def create_npc_embed(npc: NPC, translate: TranslateFn = _identity) -> Embed:
    """Builds the `/maplestory npc` embed: where the NPC stands and what it sells.

    Locations are grouped by region with every map path translated segment by segment. Shop
    entries store their item path-qualified (`equipment/Wooden Sword`), so only the segment
    after the last `/` is translated and shown; both shop lists are cut to 10 entries. The NPC's
    scroll and misc stock, its quests and its crafting recipes are not rendered.

    Args:
        npc (NPC): The NPC to render.
        translate (TranslateFn): Translator for the NPC type, region, map and item names.

    Returns:
        An embed titled with the NPC's display name.
    """
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
    """Renders one quest step as display lines: start NPC, hunt targets, items, reward.

    Caps each part (5 hunt targets, 3 items per collection group) so one crowded step cannot eat
    a whole field. Collected items are looked up in the `misc` table whatever category the data
    keyed them under, and only EXP and fame are taken from the reward. An empty result is the
    caller's signal to add no field at all for this step.

    Args:
        step (QuestStep): The quest step to render.
        translate (TranslateFn): Translator for the NPC, monster and item names.

    Returns:
        The step's display lines, empty when the step carries nothing worth showing.
    """
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


def create_quest_embed(quest: Quest, translate: TranslateFn = _identity) -> Embed:
    """Builds the `/maplestory quest` embed: level range, frequency and the first steps.

    Only the first 3 steps are rendered and the rest are dropped without a notice. A multi-step
    quest numbers its fields, a single-step one gets one unnumbered field instead. A frequency
    the `FREQ_ZH` map does not know falls through as its raw label rather than blank.

    Args:
        quest (Quest): The quest to render.
        translate (TranslateFn): Translator for the names inside each step.

    Returns:
        An embed titled with the quest's display name.
    """
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


def create_map_embed(map_entry: MapEntry, translate: TranslateFn = _identity) -> Embed:
    """Builds the `/maplestory map` embed: region, resident monsters and NPCs.

    Both rosters are cut to 10 entries. The hidden-map field is added only for a hidden map, so
    its absence is the "not hidden" state rather than an omission.

    Args:
        map_entry (MapEntry): The map to render.
        translate (TranslateFn): Translator for the region, monster and NPC names.

    Returns:
        An embed titled with the map's display name.
    """
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
    item_name: str, monsters: Iterable[Monster], translate: TranslateFn = _identity
) -> Embed:
    """Builds the `/maplestory item` embed: which monsters drop one item.

    Lists the first 15 monsters by their own display name, which already prefers the Chinese
    name and needs no lookup. An empty `monsters` still yields the embed, just with no drop
    field.

    The item name itself is looked up across four tables, but a translator miss returns the
    source name rather than an empty string, so the `equipment` lookup always wins and the three
    fallbacks only run for a name that translates to nothing. An item whose Chinese name lives
    in one of the other tables is therefore shown in English.

    Args:
        item_name (str): Source name of the item, as stored in the data files.
        monsters (Iterable[Monster]): Monsters that drop it, consumed once into a list.
        translate (TranslateFn): Translator for the item name.

    Returns:
        An embed titled with the item name, translated when the equipment table knows it.
    """
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
    """Builds the `/maplestory stats` embed: category totals, level spread and popular drops.

    The only builder that takes no translator: every value here is a count or a data-file name,
    so nothing is looked up. Level ranges are sorted as strings, which orders `100-109` ahead of
    `20-29`. Popular items are cut to 15 and shown exactly as stored, in English.

    Args:
        stats (MapleStats): Counts and distributions taken from the loaded data set.

    Returns:
        An embed summarizing what the service currently has loaded.
    """
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

    embed.set_footer(text="資料來源：Artale | 使用 /maplestory monster 或 /maplestory item 搜尋")
    return embed
