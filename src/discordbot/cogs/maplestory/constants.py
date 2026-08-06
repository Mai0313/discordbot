"""Traditional Chinese embed field bodies for the `/maplestory` lookups.

`embeds.py` is the cog's only consumer and assembles every other field value inline; these two
are the multi-line ones, kept here so the user-facing wording reads on its own rather than
buried in the embed-assembly code. Each is a single field VALUE, not a whole embed:
`MONSTER_ATTR_TEMPLATE` fills the 屬性 field of `create_monster_embed`, `BASIC_STATS_TEMPLATE`
the 資料總覽 field of `build_stats_embed`.

Both are filled with `str.format()`, so a placeholder rename surfaces as a `KeyError` at the
call site rather than as anything a type checker reports; keep the names in step with the two
`.format()` calls in `embeds.py`. The backslash after the opening quotes keeps the first label
flush, since a leading newline would render as a blank line above the field body.

`meso_range` is the one placeholder that is not a number: the call site renders it as
`1,234 ~ 5,678`, or as `N/A` when the monster carries no pair, so that fallback stays out of
the template.
"""

MONSTER_ATTR_TEMPLATE = """\
**等級**: {level}
**HP**: {hp:,}
**MP**: {mp:,}
**EXP**: {exp:,}
**物防**: {weapon_def}
**魔防**: {magic_def}
**迴避**: {avoidability}
**命中需求**: {accuracy_required}
**楓幣**: {meso_range}"""

BASIC_STATS_TEMPLATE = """\
**怪物**: {total_monsters}
**裝備**: {total_equipment}
**捲軸**: {total_scrolls}
**消耗品**: {total_useable}
**NPC**: {total_npcs}
**任務**: {total_quests}
**地圖**: {total_maps}
**其它物品**: {total_misc}"""
