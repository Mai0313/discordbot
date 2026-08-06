"""The pydantic shapes the local Artale JSON export validates into.

`service.py` hands each file in `./data/maplestory` to one of the top-level models here
(`monsters.json` to `Monster`, `equipment.json` to `Equipment`, and likewise for scrolls, useable
items, NPCs, quests, maps and misc items), and every builder in `embeds.py` reads that result
rather than the raw dicts. The shapes sit apart from `service.py` because its loader is generic
over `type[T]`, so this file is the only place the field names exist; keeping them here lets
`embeds.py` and `tests/test_maplestory.py` name a type without importing the loader.

The export is scraped rather than authored, so the models are written to survive it:

- `_Base` ignores unknown keys, because the files already carry fields nothing here models
  (`reborn` and `description` on a monster, `fromMapHint` / `toMapHint` on a map). A key added
  upstream must never fail the load.
- Only the identifying name is required. Everything else defaults, because the exporter omits a
  key instead of writing an empty value: 20 of 371 monsters carry no `nameZh`, 1431 of 1840
  equipment rows no `attackSpeed`, and `boss` appears on 9 quests out of 426. A default here
  means "the export said nothing", not "the game says zero".
- The JSON is camelCase, so most aliases are only that (`nameZh`, `regionToMapsList`,
  `equipmentItems`). Three carry one for a second reason: `str`, `int` and `def` are a Python
  builtin or keyword, and land as `str_stat` / `str_req`, `int_stat` / `int_req` and `def_stats`
  / `def_stat`. `populate_by_name` is on so code and tests can build a model by either spelling.

`MapleStats` is the one model with no file behind it: `MapleStoryService.get_stats` computes it
for `/maplestory stats`, which is why all of its fields are required.

What is modelled is the export's shape, not the subset the eight `/maplestory` subcommands render
today. `Acquisition.craftings`, `NPC.recipes`, `MapEntry.from_map` / `to_map` / `to_region` and
the whole of `Useable` past its name reach no embed yet, and are kept so a new surface does not
have to reopen the files to find out what is in them.

The handful of properties and methods are the readings the embed builders would otherwise each
redo: the Chinese-name fallback (`display_name`), the flattened map list (`all_maps`), the
non-meso drops (`MonsterDrops.all_items`) and the two equipment reductions
(`EquipmentStats.non_zero_stats`, `EquipmentRestriction.has_requirements`). Nothing here reads a
file or reaches Discord. Only a record's own `nameZh` travels inline; a name that belongs to
another record is translated through `MapleStoryService.translate` against `translations.json`.
"""

from __future__ import annotations

from pydantic import Field, BaseModel, ConfigDict


class _Base(BaseModel):
    """Shared config for every model here: ignore unknown keys, accept either field spelling.

    `extra="ignore"` is what keeps a scraped export loadable, since the files carry keys nothing
    here models. `populate_by_name` lets a caller build a model with the python field name while
    the JSON supplies the camelCase alias.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ── Shared sub-models ───────────────────────────────────────────────


class RegionMaps(_Base):
    """Represents maps within a region.

    How both `Monster` and `NPC` carry their locations; `all_maps` on either flattens the
    grouping away, while `create_monster_embed` / `create_npc_embed` keep it as a heading.

    Attributes:
        region: Region name.
        maps: Map names in the region.
    """

    region: str = Field(..., description="Region name.")
    maps: list[str] = Field(default_factory=list, description="Map names in the region.")


class AcquisitionMonster(_Base):
    """Represents a monster that an item can be acquired from.

    Attributes:
        name: Monster name.
        level: Monster level.
    """

    name: str = Field(..., description="Monster name.")
    level: int = Field(default=0, description="Monster level.")


class AcquisitionNPC(_Base):
    """Represents an NPC that an item can be acquired from.

    Attributes:
        name: NPC name.
        price: Item price from the NPC.
    """

    name: str = Field(..., description="NPC name.")
    price: int = Field(default=0, description="Item price from the NPC.")


class AcquisitionQuest(_Base):
    """Represents a quest that an item can be acquired from.

    Attributes:
        name: Quest name.
        level: Quest level.
    """

    name: str = Field(..., description="Quest name.")
    level: int = Field(default=0, description="Quest level.")


class CraftingMaterial(_Base):
    """Represents a material required for crafting.

    Attributes:
        item: Material item name.
        quantity: Required quantity.
    """

    item: str = Field(default="", description="Material item name.")
    quantity: int = Field(default=0, description="Required quantity.")


class CraftingRecipe(_Base):
    """Represents a crafting recipe.

    Listed from two sides: as `Acquisition.craftings` under a craftable item and as `NPC.recipes`
    under a crafting NPC. Rare in the export (one entry each) and rendered by neither embed, so
    `npc` and `output` default rather than being required.

    Attributes:
        npc: NPC name associated with the recipe.
        output: Crafted output name.
        materials: Materials required by the recipe.
    """

    npc: str = Field(default="", description="NPC name associated with the recipe.")
    output: str = Field(default="", description="Crafted output name.")
    materials: list[CraftingMaterial] = Field(
        default_factory=list, description="Materials required by the recipe."
    )


class Acquisition(_Base):
    """Represents all ways an item can be acquired.

    Hangs off `Equipment`, `Scroll`, `Useable` and `MiscItem` alike. `_add_acquisition_fields`
    renders the first three lists and skips `craftings`, so an empty `Acquisition` (which is what
    an item with no known source validates into) costs the embed nothing.

    Attributes:
        monsters: Monster acquisition entries.
        npcs: NPC acquisition entries.
        quests: Quest acquisition entries.
        craftings: Crafting recipe entries.
    """

    monsters: list[AcquisitionMonster] = Field(
        default_factory=list, description="Monster acquisition entries."
    )
    npcs: list[AcquisitionNPC] = Field(
        default_factory=list, description="NPC acquisition entries."
    )
    quests: list[AcquisitionQuest] = Field(
        default_factory=list, description="Quest acquisition entries."
    )
    craftings: list[CraftingRecipe] = Field(
        default_factory=list, description="Crafting recipe entries."
    )


# ── Monster ─────────────────────────────────────────────────────────


class DefenseStats(_Base):
    """Represents monster defense statistics.

    Attributes:
        weapon: Weapon defense value.
        magic: Magic defense value.
        avoidability: Avoidability value.
    """

    weapon: int = Field(default=0, description="Weapon defense value.")
    magic: int = Field(default=0, description="Magic defense value.")
    avoidability: int = Field(default=0, description="Avoidability value.")


class AccuracyStats(_Base):
    """Represents monster accuracy statistics.

    Both fields are -1 together on the 30 monsters the export has no accuracy data for, so
    neither can be read as a plain number; `create_monster_embed` prints `required` through
    unchanged and shows -1 for those.

    Attributes:
        required: Required accuracy value.
        decrease: Accuracy decrease value.
    """

    required: int = Field(default=0, description="Required accuracy value.")
    decrease: float = Field(default=0, description="Accuracy decrease value.")


class DropItem(_Base):
    """Represents an item dropped by a monster.

    One shape for all four drop lists, so `level` and `jobs` are empty on the roughly half of
    entries that are not equipment. The name is the drop's English name and the key
    `MapleStoryService.get_monsters_by_drop` matches on, not a display name.

    Attributes:
        name: Dropped item name.
        level: Dropped item level.
        type: Dropped item type.
        jobs: Jobs associated with the dropped item.
    """

    name: str = Field(..., description="Dropped item name.")
    level: int = Field(default=0, description="Dropped item level.")
    type: str = Field(default="", description="Dropped item type.")
    jobs: list[str] = Field(
        default_factory=list, description="Jobs associated with the dropped item."
    )


class MonsterDrops(_Base):
    """Represents all drops for a monster.

    The four item lists are kept apart because the embed labels them separately and
    `MapleStoryService.get_item_type` decides an item's category purely by which list it sits in.
    `meso_range` is the odd one out: a `[min, max]` pair rather than items, which is why it is
    excluded from `all_items` and why `create_monster_embed` renders anything but a two-element
    list as N/A.

    Attributes:
        equipment_items: Equipment items dropped by the monster.
        useable_items: Useable items dropped by the monster.
        scrolls: Scrolls dropped by the monster.
        misc_items: Miscellaneous items dropped by the monster.
        meso_range: Meso range dropped by the monster.
    """

    equipment_items: list[DropItem] = Field(
        default_factory=list,
        alias="equipmentItems",
        description="Equipment items dropped by the monster.",
    )
    useable_items: list[DropItem] = Field(
        default_factory=list,
        alias="useableItems",
        description="Useable items dropped by the monster.",
    )
    scrolls: list[DropItem] = Field(
        default_factory=list, alias="scrolls", description="Scrolls dropped by the monster."
    )
    misc_items: list[DropItem] = Field(
        default_factory=list,
        alias="miscItems",
        description="Miscellaneous items dropped by the monster.",
    )
    meso_range: list[int] = Field(
        default_factory=list, alias="mesoRange", description="Meso range dropped by the monster."
    )

    @property
    def all_items(self) -> list[DropItem]:
        """Returns all non-meso drop items.

        Builds a fresh list on every read, so a caller iterating it cannot disturb the four
        underlying lists. Nothing is deduped here; `MapleStoryService.search_items_by_name`
        collects into a set of its own.

        Returns:
            Equipment, useable, scroll, and miscellaneous drop items, in that order.
        """
        return self.equipment_items + self.useable_items + self.scrolls + self.misc_items


class MonsterQuest(_Base):
    """Represents a quest associated with a monster.

    Attributes:
        name: Quest name.
        level: Quest level.
    """

    name: str = Field(..., description="Quest name.")
    level: int = Field(default=0, description="Quest level.")


class Monster(_Base):
    """Represents a MapleStory monster.

    The record every item lookup leans on: `MapleStoryService.search_items_by_name`,
    `get_monsters_by_drop` and `get_item_type` all walk `drops` across the monster list rather
    than the `Acquisition` blocks the item files carry, so `/maplestory item` can only find an
    item some monster drops.
    `modifiers` are the elemental affinity tags (`STRONG FIRE`, `WEAK ICE`, `IMMUNE POISON`,
    `CAN HEAL ATTACK`), translated through the `modifiers` category.

    Attributes:
        name: Monster name.
        name_zh: Chinese monster name.
        level: Monster level.
        hp: Monster HP.
        mp: Monster MP.
        exp: Monster EXP.
        def_stats: Monster defense statistics.
        accuracy: Monster accuracy statistics.
        modifiers: Monster modifier names.
        region_to_maps_list: Regions and maps where the monster appears.
        drops: Monster drop data.
        quests: Quests associated with the monster.
    """

    name: str = Field(..., description="Monster name.")
    name_zh: str = Field(default="", alias="nameZh", description="Chinese monster name.")
    level: int = Field(default=0, description="Monster level.")
    hp: int = Field(default=0, description="Monster HP.")
    mp: int = Field(default=0, description="Monster MP.")
    exp: int = Field(default=0, description="Monster EXP.")
    def_stats: DefenseStats = Field(
        default_factory=DefenseStats, alias="def", description="Monster defense statistics."
    )
    accuracy: AccuracyStats = Field(
        default_factory=AccuracyStats, description="Monster accuracy statistics."
    )
    modifiers: list[str] = Field(default_factory=list, description="Monster modifier names.")
    region_to_maps_list: list[RegionMaps] = Field(
        default_factory=list,
        alias="regionToMapsList",
        description="Regions and maps where the monster appears.",
    )
    drops: MonsterDrops = Field(default_factory=MonsterDrops, description="Monster drop data.")
    quests: list[MonsterQuest] = Field(
        default_factory=list, description="Quests associated with the monster."
    )

    @property
    def display_name(self) -> str:
        """Returns the monster display name.

        Returns:
            The Chinese name when present, otherwise the source name.
        """
        return self.name_zh or self.name

    @property
    def all_maps(self) -> list[str]:
        """Returns every map where the monster appears.

        Drops the region grouping that `create_monster_embed` renders as headings, keeping
        `region_to_maps_list` order.

        Returns:
            Map names flattened from all region entries.
        """
        return [m for r in self.region_to_maps_list for m in r.maps]


# ── Equipment ───────────────────────────────────────────────────────


class StatValue(_Base):
    """Represents a stat value with a middle value and a range.

    `middle` is the nominal figure the embed prints and `range` the `[min, max]` a rolled item
    can land in; they are independent in the export, so `middle` is not the midpoint of `range`
    and either can be missing. A row carrying only a `range` therefore defaults `middle` to 0 and
    disappears from `EquipmentStats.non_zero_stats`.

    Attributes:
        middle: Middle stat value.
        range: Stat value range.
    """

    middle: int = Field(default=0, description="Middle stat value.")
    range: list[int] = Field(default_factory=list, description="Stat value range.")


class EquipmentStats(_Base):
    """Represents equipment statistics.

    Every stat is optional rather than zero-defaulted, so "the export listed no ATK" stays
    distinguishable from "ATK is 0". `attack_speed` and `upgrade_slots` are plain ints, not
    `StatValue`s, which is why `non_zero_stats` cannot report them and `_add_equip_stats` appends
    them itself. Note the numeric `attack_speed` here is a different field from the `FAST` /
    `NORMAL` label on `Equipment`, despite both coming from a JSON key spelled `attackSpeed`.

    Attributes:
        str_stat: STR stat value.
        dex: DEX stat value.
        int_stat: INT stat value.
        luk: LUK stat value.
        hp: HP stat value.
        mp: MP stat value.
        atk: Attack stat value.
        matk: Magic attack stat value.
        def_stat: Defense stat value.
        mdef: Magic defense stat value.
        accuracy: Accuracy stat value.
        avoidability: Avoidability stat value.
        speed: Speed stat value.
        jump: Jump stat value.
        attack_speed: Attack speed value.
        upgrade_slots: Upgrade slot count.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    str_stat: StatValue | None = Field(default=None, alias="str", description="STR stat value.")
    dex: StatValue | None = Field(default=None, description="DEX stat value.")
    int_stat: StatValue | None = Field(default=None, alias="int", description="INT stat value.")
    luk: StatValue | None = Field(default=None, description="LUK stat value.")
    hp: StatValue | None = Field(default=None, description="HP stat value.")
    mp: StatValue | None = Field(default=None, description="MP stat value.")
    atk: StatValue | None = Field(default=None, description="Attack stat value.")
    matk: StatValue | None = Field(default=None, description="Magic attack stat value.")
    def_stat: StatValue | None = Field(
        default=None, alias="def", description="Defense stat value."
    )
    mdef: StatValue | None = Field(default=None, description="Magic defense stat value.")
    accuracy: StatValue | None = Field(default=None, description="Accuracy stat value.")
    avoidability: StatValue | None = Field(default=None, description="Avoidability stat value.")
    speed: StatValue | None = Field(default=None, description="Speed stat value.")
    jump: StatValue | None = Field(default=None, description="Jump stat value.")
    attack_speed: int | None = Field(
        default=None, alias="attackSpeed", description="Attack speed value."
    )
    upgrade_slots: int | None = Field(
        default=None, alias="upgradeSlots", description="Upgrade slot count."
    )

    def non_zero_stats(self) -> list[tuple[str, StatValue]]:
        """Returns (label, value) pairs for stats with non-zero middle.

        Covers only the fourteen `StatValue` fields, so `attack_speed` and `upgrade_slots` never
        appear and `_add_equip_stats` appends those itself. Filtering on `middle` also hides a
        stat the export gave a `range` but no `middle`. The labels match `_STAT_LABELS` in
        `embeds.py`, which spells the same set out again for the scroll embed because a scroll's
        stats arrive as a plain int map.

        Returns:
            A list of tuples containing the stat label and its value.
        """
        mapping = [
            ("STR", self.str_stat),
            ("DEX", self.dex),
            ("INT", self.int_stat),
            ("LUK", self.luk),
            ("HP", self.hp),
            ("MP", self.mp),
            ("ATK", self.atk),
            ("M.ATK", self.matk),
            ("DEF", self.def_stat),
            ("M.DEF", self.mdef),
            ("Accuracy", self.accuracy),
            ("Avoidability", self.avoidability),
            ("Speed", self.speed),
            ("Jump", self.jump),
        ]
        return [(label, sv) for label, sv in mapping if sv and sv.middle]


class EquipmentRestriction(_Base):
    """Represents equipment requirements.

    The export writes this block on every equipment row, including the roughly one in four that
    require nothing at all, which is what `has_requirements` exists to tell apart from a real
    requirement.

    Attributes:
        str_req: STR requirement.
        dex: DEX requirement.
        int_req: INT requirement.
        luk: LUK requirement.
    """

    str_req: int = Field(default=0, alias="str", description="STR requirement.")
    dex: int = Field(default=0, description="DEX requirement.")
    int_req: int = Field(default=0, alias="int", description="INT requirement.")
    luk: int = Field(default=0, description="LUK requirement.")

    def has_requirements(self) -> bool:
        """Checks if there are any stat requirements.

        The gate `_add_equip_requirements` reads before adding its field, so an all-zero block
        produces no embed field rather than four zeros.

        Returns:
            True if any requirement is non-zero, False otherwise.
        """
        return any((self.str_req, self.dex, self.int_req, self.luk))


class Equipment(_Base):
    """Represents a MapleStory equipment item.

    `attack_speed` here is the `FAST` / `NORMAL` / `SLOW` label, not the number of the same name
    inside `stats`; both come from a JSON key spelled `attackSpeed`, one at each level. The four
    label-ish fields (`tradeable`, `event`, `limited_time`, `unavailable`) are written by the
    export only when they apply, so their defaults mean "ordinary" rather than "unknown".

    Attributes:
        type: Equipment type.
        name: Equipment name.
        name_zh: Chinese equipment name.
        level: Equipment level.
        equipment_restriction: Equipment stat requirements.
        stats: Equipment stat values.
        jobs: Jobs associated with the equipment.
        attack_speed: Attack speed label.
        acquisition: Acquisition data for the equipment.
        tradeable: Tradeability label.
        event: Whether the equipment is marked as an event item.
        limited_time: Whether the equipment is marked as limited time.
        unavailable: Whether the equipment is marked as unavailable.
    """

    type: str = Field(default="", description="Equipment type.")
    name: str = Field(..., description="Equipment name.")
    name_zh: str = Field(default="", alias="nameZh", description="Chinese equipment name.")
    level: int = Field(default=0, description="Equipment level.")
    equipment_restriction: EquipmentRestriction = Field(
        default_factory=EquipmentRestriction,
        alias="equipmentRestriction",
        description="Equipment stat requirements.",
    )
    stats: EquipmentStats = Field(
        default_factory=EquipmentStats, description="Equipment stat values."
    )
    jobs: list[str] = Field(
        default_factory=list, description="Jobs associated with the equipment."
    )
    attack_speed: str = Field(default="", alias="attackSpeed", description="Attack speed label.")
    acquisition: Acquisition = Field(
        default_factory=Acquisition, description="Acquisition data for the equipment."
    )
    tradeable: str = Field(default="", description="Tradeability label.")
    event: bool = Field(
        default=False, description="Whether the equipment is marked as an event item."
    )
    limited_time: bool = Field(
        default=False,
        alias="limitedTime",
        description="Whether the equipment is marked as limited time.",
    )
    unavailable: bool = Field(
        default=False, description="Whether the equipment is marked as unavailable."
    )

    @property
    def display_name(self) -> str:
        """Returns the equipment display name.

        Returns:
            The Chinese name when present, otherwise the source name.
        """
        return self.name_zh or self.name


# ── Scroll ──────────────────────────────────────────────────────────


class Scroll(_Base):
    """Represents a MapleStory scroll.

    `stats` is a flat `{"dex": 3}` map rather than the `StatValue` shape `EquipmentStats` uses,
    because a scroll grants a fixed bonus with no roll. `create_scroll_embed` puts the keys
    through `_STAT_LABELS` for display, and an unrecognised key falls through as itself rather
    than being dropped. `type` names the equipment slot the scroll applies to.

    Attributes:
        name: Scroll name.
        name_zh: Chinese scroll name.
        stats: Stat bonuses keyed by stat name.
        type: Scroll type.
        acquisition: Acquisition data for the scroll.
    """

    name: str = Field(..., description="Scroll name.")
    name_zh: str = Field(default="", alias="nameZh", description="Chinese scroll name.")
    stats: dict[str, int] = Field(
        default_factory=dict, description="Stat bonuses keyed by stat name."
    )
    type: str = Field(default="", description="Scroll type.")
    acquisition: Acquisition = Field(
        default_factory=Acquisition, description="Acquisition data for the scroll."
    )

    @property
    def display_name(self) -> str:
        """Returns the scroll display name.

        Returns:
            The Chinese name when present, otherwise the source name.
        """
        return self.name_zh or self.name


# ── Useable ─────────────────────────────────────────────────────────


class UseableStat(_Base):
    """Represents a stat value for useable items.

    A one-field wrapper because that is the export's shape (`"hp": {"amount": 50}`), not because
    a second field is expected.

    Attributes:
        amount: Amount applied by the useable item stat.
    """

    amount: int = Field(default=0, description="Amount applied by the useable item stat.")


class Useable(_Base):
    """Represents a MapleStory useable item.

    `description` is a `{"zh": ..., "en": ...}` map on the few rows that carry one at all; the
    `str` arm of the union is what the empty default occupies, not a second export shape. Every
    stat is `None` when absent rather than a zeroed `UseableStat`, so a potion that restores only
    HP carries no `mp` block.

    There is no `/maplestory useable` subcommand and no embed builder for this model.
    `useable.json` is loaded for the `/maplestory stats` count and the `MapleStoryService.useable`
    property, and a useable item named in a monster's drops is translated through
    `translations.json` instead.

    Attributes:
        name: Useable item name.
        name_zh: Chinese useable item name.
        type: Useable item type.
        description: Description data for the useable item.
        acquisition: Acquisition data for the useable item.
        hp: HP stat data.
        mp: MP stat data.
        atk: Attack stat data.
        matk: Magic attack stat data.
        def_stat: Defense stat data.
        mdef: Magic defense stat data.
        accuracy: Accuracy stat data.
        avoidability: Avoidability stat data.
        speed: Speed stat data.
        jump: Jump stat data.
    """

    name: str = Field(..., description="Useable item name.")
    name_zh: str = Field(default="", alias="nameZh", description="Chinese useable item name.")
    type: str = Field(default="", description="Useable item type.")
    description: str | dict[str, str] = Field(
        default="", description="Description data for the useable item."
    )
    acquisition: Acquisition = Field(
        default_factory=Acquisition, description="Acquisition data for the useable item."
    )
    hp: UseableStat | None = Field(default=None, description="HP stat data.")
    mp: UseableStat | None = Field(default=None, description="MP stat data.")
    atk: UseableStat | None = Field(default=None, description="Attack stat data.")
    matk: UseableStat | None = Field(default=None, description="Magic attack stat data.")
    def_stat: UseableStat | None = Field(
        default=None, alias="def", description="Defense stat data."
    )
    mdef: UseableStat | None = Field(default=None, description="Magic defense stat data.")
    accuracy: UseableStat | None = Field(default=None, description="Accuracy stat data.")
    avoidability: UseableStat | None = Field(default=None, description="Avoidability stat data.")
    speed: UseableStat | None = Field(default=None, description="Speed stat data.")
    jump: UseableStat | None = Field(default=None, description="Jump stat data.")

    @property
    def display_name(self) -> str:
        """Returns the useable item display name.

        Returns:
            The Chinese name when present, otherwise the source name.
        """
        return self.name_zh or self.name


# ── NPC ─────────────────────────────────────────────────────────────


class NPCItem(_Base):
    """Represents an item sold by an NPC.

    Every name in the export carries a category prefix (`potion/Red Potion`), unlike the bare
    names everywhere else, which is why `create_npc_embed` splits on `/` before translating.
    Stored as it arrives so a caller can still see which category the shop listed it under.

    Attributes:
        name: Sold item name.
        price: Sold item price.
    """

    name: str = Field(..., description="Sold item name.")
    price: int = Field(default=0, description="Sold item price.")


class NPC(_Base):
    """Represents a MapleStory NPC.

    Attributes:
        name: NPC name.
        name_zh: Chinese NPC name.
        type: NPC type.
        region_to_maps_list: Regions and maps where the NPC appears.
        equipment_items: Equipment items sold by the NPC.
        useable_items: Useable items sold by the NPC.
        scrolls: Scrolls sold by the NPC.
        misc_items: Miscellaneous items sold by the NPC.
        quests: Quests associated with the NPC.
        recipes: Crafting recipes associated with the NPC.
    """

    name: str = Field(..., description="NPC name.")
    name_zh: str = Field(default="", alias="nameZh", description="Chinese NPC name.")
    type: str = Field(default="", description="NPC type.")
    region_to_maps_list: list[RegionMaps] = Field(
        default_factory=list,
        alias="regionToMapsList",
        description="Regions and maps where the NPC appears.",
    )
    equipment_items: list[NPCItem] = Field(
        default_factory=list,
        alias="equipmentItems",
        description="Equipment items sold by the NPC.",
    )
    useable_items: list[NPCItem] = Field(
        default_factory=list, alias="useableItems", description="Useable items sold by the NPC."
    )
    scrolls: list[NPCItem] = Field(
        default_factory=list, alias="scrolls", description="Scrolls sold by the NPC."
    )
    misc_items: list[NPCItem] = Field(
        default_factory=list, alias="miscItems", description="Miscellaneous items sold by the NPC."
    )
    quests: list[AcquisitionQuest] = Field(
        default_factory=list, description="Quests associated with the NPC."
    )
    recipes: list[CraftingRecipe] = Field(
        default_factory=list, description="Crafting recipes associated with the NPC."
    )

    @property
    def display_name(self) -> str:
        """Returns the NPC display name.

        Returns:
            The Chinese name when present, otherwise the source name.
        """
        return self.name_zh or self.name

    @property
    def all_maps(self) -> list[str]:
        """Returns every map where the NPC appears.

        Drops the region grouping that `create_npc_embed` renders as headings, keeping
        `region_to_maps_list` order.

        Returns:
            Map names flattened from all region entries.
        """
        return [m for r in self.region_to_maps_list for m in r.maps]


# ── Quest ───────────────────────────────────────────────────────────


class HuntTarget(_Base):
    """Represents a target to hunt for a quest.

    Attributes:
        name: Hunt target name.
        quantity: Required hunt quantity.
    """

    name: str = Field(..., description="Hunt target name.")
    quantity: int = Field(default=0, description="Required hunt quantity.")


class CollectItem(_Base):
    """Represents an item to collect for a quest.

    Attributes:
        name: Collected item name.
        quantity: Required collection quantity.
    """

    name: str = Field(default="", description="Collected item name.")
    quantity: int = Field(default=0, description="Required collection quantity.")


class QuestReward(_Base):
    """Represents rewards for a quest.

    `items` takes a union because the export writes it two ways: usually one `{"scroll": [...]}`
    map, occasionally a list of such maps. Nothing renders it or `mesos` yet, so the union is
    carried rather than normalised; `_format_quest_step` shows only `exp` and `fame`.

    Attributes:
        exp: Reward EXP.
        fame: Reward fame.
        mesos: Reward mesos.
        items: Reward item data.
    """

    exp: int = Field(default=0, description="Reward EXP.")
    fame: int = Field(default=0, description="Reward fame.")
    mesos: int = Field(default=0, description="Reward mesos.")
    items: dict[str, list[CollectItem]] | list[dict[str, list[CollectItem]]] = Field(
        default_factory=dict, description="Reward item data."
    )


class QuestStep(_Base):
    """Represents a step in a quest.

    `items_to_collect` keys the item lists by category, the same grouping `QuestReward.items`
    uses. A quest holds its steps in `Quest.steps`, and `create_quest_embed` renders the first
    three as separate fields.

    Attributes:
        start_npc: NPC that starts the quest step.
        monsters_to_hunt: Monsters required by the quest step.
        items_to_collect: Items required by the quest step.
        reward: Reward data for the quest step.
    """

    start_npc: str = Field(
        default="", alias="startNPC", description="NPC that starts the quest step."
    )
    monsters_to_hunt: list[HuntTarget] = Field(
        default_factory=list,
        alias="monstersToHunt",
        description="Monsters required by the quest step.",
    )
    items_to_collect: dict[str, list[CollectItem]] = Field(
        default_factory=dict,
        alias="itemsToCollect",
        description="Items required by the quest step.",
    )
    reward: QuestReward = Field(
        default_factory=QuestReward, description="Reward data for the quest step."
    )


class Quest(_Base):
    """Represents a MapleStory quest.

    `lv_upper` defaults to `None` while `lv_lower` defaults to 0, because only 32 quests of 426
    declare an upper limit and `create_quest_embed` prints the range only when one exists.
    `frequency` carries the export's own spelling (`one-time`, `daily`, `12hr`, ...), which
    `FREQ_ZH` maps for display and falls back to verbatim on an unrecognised value.

    Attributes:
        name: Quest name.
        name_zh: Chinese quest name.
        frequency: Quest frequency label.
        lv_lower: Lower level bound.
        lv_upper: Upper level bound.
        steps: Quest steps.
        boss: Whether the quest is marked as a boss quest.
        prerequisites: Prerequisite quest names.
    """

    name: str = Field(..., description="Quest name.")
    name_zh: str = Field(default="", alias="nameZh", description="Chinese quest name.")
    frequency: str = Field(default="", description="Quest frequency label.")
    lv_lower: int = Field(default=0, alias="lvLower", description="Lower level bound.")
    lv_upper: int | None = Field(default=None, alias="lvUpper", description="Upper level bound.")
    steps: list[QuestStep] = Field(default_factory=list, description="Quest steps.")
    boss: bool = Field(default=False, description="Whether the quest is marked as a boss quest.")
    prerequisites: list[str] = Field(default_factory=list, description="Prerequisite quest names.")

    @property
    def display_name(self) -> str:
        """Returns the quest display name.

        Returns:
            The Chinese name when present, otherwise the source name.
        """
        return self.name_zh or self.name


# ── Map ─────────────────────────────────────────────────────────────


class MapNPC(_Base):
    """Represents an NPC on a map.

    One NPC's placement, not a link to the `NPC` record of the same name: `type` here is the role
    the map lists (`Weapon Seller`), and `sub_map` names the shop or room inside the map when the
    NPC is not out on it.

    Attributes:
        name: NPC name.
        type: NPC type.
        sub_map: Sub-map name.
    """

    name: str = Field(..., description="NPC name.")
    type: str = Field(default="", description="NPC type.")
    sub_map: str = Field(default="", alias="subMap", description="Sub-map name.")


class MapMonster(_Base):
    """Represents a monster on a map.

    Attributes:
        name: Monster name.
        level: Monster level.
    """

    name: str = Field(..., description="Monster name.")
    level: int = Field(default=0, description="Monster level.")


class MapEntry(_Base):
    """Represents a MapleStory map.

    `from_map` / `to_map` / `to_region` record which map this one connects to. The export pairs
    each with a Chinese `fromMapHint` / `toMapHint` describing where the entrance is, and `_Base`
    drops both, so a surface that wants to show the connection has to model the hint first.

    Attributes:
        region: Map region name.
        name: Map name.
        name_zh: Chinese map name.
        x: Map x-coordinate.
        y: Map y-coordinate.
        npcs: NPCs on the map.
        monsters: Monsters on the map.
        hidden: Whether the map is hidden.
        from_map: Source map name.
        to_map: Destination map name.
        to_region: Destination region name.
    """

    region: str = Field(default="", description="Map region name.")
    name: str = Field(..., description="Map name.")
    name_zh: str = Field(default="", alias="nameZh", description="Chinese map name.")
    x: int = Field(default=0, description="Map x-coordinate.")
    y: int = Field(default=0, description="Map y-coordinate.")
    npcs: list[MapNPC] = Field(default_factory=list, description="NPCs on the map.")
    monsters: list[MapMonster] = Field(default_factory=list, description="Monsters on the map.")
    hidden: bool = Field(default=False, description="Whether the map is hidden.")
    from_map: str = Field(default="", alias="fromMap", description="Source map name.")
    to_map: str = Field(default="", alias="toMap", description="Destination map name.")
    to_region: str = Field(default="", alias="toRegion", description="Destination region name.")

    @property
    def display_name(self) -> str:
        """Returns the map display name.

        Returns:
            The Chinese name when present, otherwise the source name.
        """
        return self.name_zh or self.name


# ── Misc Item ───────────────────────────────────────────────────────


class MiscItem(_Base):
    """Represents a miscellaneous item.

    Attributes:
        name: Miscellaneous item name.
        name_zh: Chinese miscellaneous item name.
        type: Miscellaneous item type.
        acquisition: Acquisition data for the miscellaneous item.
    """

    name: str = Field(..., description="Miscellaneous item name.")
    name_zh: str = Field(
        default="", alias="nameZh", description="Chinese miscellaneous item name."
    )
    type: str = Field(default="", description="Miscellaneous item type.")
    acquisition: Acquisition = Field(
        default_factory=Acquisition, description="Acquisition data for the miscellaneous item."
    )

    @property
    def display_name(self) -> str:
        """Returns the miscellaneous item display name.

        Returns:
            The Chinese name when present, otherwise the source name.
        """
        return self.name_zh or self.name


# ── Stats (for /maplestory stats command) ───────────────────────────


class MapleStats(_Base):
    """Represents database statistics.

    The one model here with no JSON file behind it: `MapleStoryService.get_stats` builds and
    memoizes it from the loaded categories, and `build_stats_embed` renders it for
    `/maplestory stats`. Every field is required, since a caller that computed the summary
    computed all of it. `level_distribution` is keyed by a ten-level bucket label (`"0-9"`), and
    `popular_items` is already sorted by how many monsters drop each item and cut to 20 by the
    service.

    Attributes:
        total_monsters: Total monster count.
        total_equipment: Total equipment count.
        total_scrolls: Total scroll count.
        total_useable: Total useable item count.
        total_npcs: Total NPC count.
        total_quests: Total quest count.
        total_maps: Total map count.
        total_misc: Total miscellaneous item count.
        level_distribution: Monster counts keyed by level range.
        popular_items: Popular item names.
    """

    total_monsters: int = Field(..., description="Total monster count.")
    total_equipment: int = Field(..., description="Total equipment count.")
    total_scrolls: int = Field(..., description="Total scroll count.")
    total_useable: int = Field(..., description="Total useable item count.")
    total_npcs: int = Field(..., description="Total NPC count.")
    total_quests: int = Field(..., description="Total quest count.")
    total_maps: int = Field(..., description="Total map count.")
    total_misc: int = Field(..., description="Total miscellaneous item count.")
    level_distribution: dict[str, int] = Field(
        ..., description="Monster counts keyed by level range."
    )
    popular_items: list[str] = Field(..., description="Popular item names.")
