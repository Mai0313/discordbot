"""Pydantic models for MapleStory data.

This module defines the data structures used for monsters, equipment, NPCs,
quests, maps, etc., loaded from JSON data files.
"""

from __future__ import annotations

from pydantic import Field, BaseModel, ConfigDict


class _Base(BaseModel):
    """Base model for MapleStory data, ignoring extra fields."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ── Shared sub-models ───────────────────────────────────────────────


class RegionMaps(_Base):
    """Represents maps within a region.

    Attributes:
        region: Region name.
        maps: Map names in the region.
    """

    region: str
    maps: list[str] = Field(default_factory=list)


class AcquisitionMonster(_Base):
    """Represents a monster that an item can be acquired from.

    Attributes:
        name: Monster name.
        level: Monster level.
    """

    name: str
    level: int = 0


class AcquisitionNPC(_Base):
    """Represents an NPC that an item can be acquired from.

    Attributes:
        name: NPC name.
        price: Item price from the NPC.
    """

    name: str
    price: int = 0


class AcquisitionQuest(_Base):
    """Represents a quest that an item can be acquired from.

    Attributes:
        name: Quest name.
        level: Quest level.
    """

    name: str
    level: int = 0


class CraftingMaterial(_Base):
    """Represents a material required for crafting.

    Attributes:
        item: Material item name.
        quantity: Required quantity.
    """

    item: str = ""
    quantity: int = 0


class CraftingRecipe(_Base):
    """Represents a crafting recipe.

    Attributes:
        npc: NPC name associated with the recipe.
        output: Crafted output name.
        materials: Materials required by the recipe.
    """

    npc: str = ""
    output: str = ""
    materials: list[CraftingMaterial] = Field(default_factory=list)


class Acquisition(_Base):
    """Represents all ways an item can be acquired.

    Attributes:
        monsters: Monster acquisition entries.
        npcs: NPC acquisition entries.
        quests: Quest acquisition entries.
        craftings: Crafting recipe entries.
    """

    monsters: list[AcquisitionMonster] = Field(default_factory=list)
    npcs: list[AcquisitionNPC] = Field(default_factory=list)
    quests: list[AcquisitionQuest] = Field(default_factory=list)
    craftings: list[CraftingRecipe] = Field(default_factory=list)


# ── Monster ─────────────────────────────────────────────────────────


class DefenseStats(_Base):
    """Represents monster defense statistics.

    Attributes:
        weapon: Weapon defense value.
        magic: Magic defense value.
        avoidability: Avoidability value.
    """

    weapon: int = 0
    magic: int = 0
    avoidability: int = 0


class AccuracyStats(_Base):
    """Represents monster accuracy statistics.

    Attributes:
        required: Required accuracy value.
        decrease: Accuracy decrease value.
    """

    required: int = 0
    decrease: float = 0


class DropItem(_Base):
    """Represents an item dropped by a monster.

    Attributes:
        name: Dropped item name.
        level: Dropped item level.
        type: Dropped item type.
        jobs: Jobs associated with the dropped item.
    """

    name: str
    level: int = 0
    type: str = ""
    jobs: list[str] = Field(default_factory=list)


class MonsterDrops(_Base):
    """Represents all drops for a monster.

    Attributes:
        equipment_items: Equipment items dropped by the monster.
        useable_items: Useable items dropped by the monster.
        scrolls: Scrolls dropped by the monster.
        misc_items: Miscellaneous items dropped by the monster.
        meso_range: Meso range dropped by the monster.
    """

    equipment_items: list[DropItem] = Field(default_factory=list, alias="equipmentItems")
    useable_items: list[DropItem] = Field(default_factory=list, alias="useableItems")
    scrolls: list[DropItem] = Field(default_factory=list, alias="scrolls")
    misc_items: list[DropItem] = Field(default_factory=list, alias="miscItems")
    meso_range: list[int] = Field(default_factory=list, alias="mesoRange")

    @property
    def all_items(self) -> list[DropItem]:
        """Returns all non-meso drop items.

        Returns:
            Equipment, useable, scroll, and miscellaneous drop items.
        """
        return self.equipment_items + self.useable_items + self.scrolls + self.misc_items


class MonsterQuest(_Base):
    """Represents a quest associated with a monster.

    Attributes:
        name: Quest name.
        level: Quest level.
    """

    name: str
    level: int = 0


class Monster(_Base):
    """Represents a MapleStory monster.

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

    name: str
    name_zh: str = Field(default="", alias="nameZh")
    level: int = 0
    hp: int = 0
    mp: int = 0
    exp: int = 0
    def_stats: DefenseStats = Field(default_factory=DefenseStats, alias="def")
    accuracy: AccuracyStats = Field(default_factory=AccuracyStats)
    modifiers: list[str] = Field(default_factory=list)
    region_to_maps_list: list[RegionMaps] = Field(default_factory=list, alias="regionToMapsList")
    drops: MonsterDrops = Field(default_factory=MonsterDrops)
    quests: list[MonsterQuest] = Field(default_factory=list)

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

        Returns:
            Map names flattened from all region entries.
        """
        return [m for r in self.region_to_maps_list for m in r.maps]


# ── Equipment ───────────────────────────────────────────────────────


class StatValue(_Base):
    """Represents a stat value with a middle value and a range.

    Attributes:
        middle: Middle stat value.
        range: Stat value range.
    """

    middle: int = 0
    range: list[int] = Field(default_factory=list)


class EquipmentStats(_Base):
    """Represents equipment statistics.

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

    str_stat: StatValue | None = Field(default=None, alias="str")
    dex: StatValue | None = None
    int_stat: StatValue | None = Field(default=None, alias="int")
    luk: StatValue | None = None
    hp: StatValue | None = None
    mp: StatValue | None = None
    atk: StatValue | None = None
    matk: StatValue | None = None
    def_stat: StatValue | None = Field(default=None, alias="def")
    mdef: StatValue | None = None
    accuracy: StatValue | None = None
    avoidability: StatValue | None = None
    speed: StatValue | None = None
    jump: StatValue | None = None
    attack_speed: int | None = Field(default=None, alias="attackSpeed")
    upgrade_slots: int | None = Field(default=None, alias="upgradeSlots")

    def non_zero_stats(self) -> list[tuple[str, StatValue]]:
        """Returns (label, value) pairs for stats with non-zero middle.

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

    Attributes:
        str_req: STR requirement.
        dex: DEX requirement.
        int_req: INT requirement.
        luk: LUK requirement.
    """

    str_req: int = Field(default=0, alias="str")
    dex: int = 0
    int_req: int = Field(default=0, alias="int")
    luk: int = 0

    def has_requirements(self) -> bool:
        """Checks if there are any stat requirements.

        Returns:
            True if any requirement is non-zero, False otherwise.
        """
        return any((self.str_req, self.dex, self.int_req, self.luk))


class Equipment(_Base):
    """Represents a MapleStory equipment item.

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

    type: str = ""
    name: str
    name_zh: str = Field(default="", alias="nameZh")
    level: int = 0
    equipment_restriction: EquipmentRestriction = Field(
        default_factory=EquipmentRestriction, alias="equipmentRestriction"
    )
    stats: EquipmentStats = Field(default_factory=EquipmentStats)
    jobs: list[str] = Field(default_factory=list)
    attack_speed: str = Field(default="", alias="attackSpeed")
    acquisition: Acquisition = Field(default_factory=Acquisition)
    tradeable: str = ""
    event: bool = False
    limited_time: bool = Field(default=False, alias="limitedTime")
    unavailable: bool = False

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

    Attributes:
        name: Scroll name.
        name_zh: Chinese scroll name.
        stats: Stat bonuses keyed by stat name.
        type: Scroll type.
        acquisition: Acquisition data for the scroll.
    """

    name: str
    name_zh: str = Field(default="", alias="nameZh")
    stats: dict[str, int] = Field(default_factory=dict)
    type: str = ""
    acquisition: Acquisition = Field(default_factory=Acquisition)

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

    Attributes:
        amount: Amount applied by the useable item stat.
    """

    amount: int = 0


class Useable(_Base):
    """Represents a MapleStory useable item.

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

    name: str
    name_zh: str = Field(default="", alias="nameZh")
    type: str = ""
    description: str | dict[str, str] = ""
    acquisition: Acquisition = Field(default_factory=Acquisition)
    hp: UseableStat | None = None
    mp: UseableStat | None = None
    atk: UseableStat | None = None
    matk: UseableStat | None = None
    def_stat: UseableStat | None = Field(default=None, alias="def")
    mdef: UseableStat | None = None
    accuracy: UseableStat | None = None
    avoidability: UseableStat | None = None
    speed: UseableStat | None = None
    jump: UseableStat | None = None

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

    Attributes:
        name: Sold item name.
        price: Sold item price.
    """

    name: str
    price: int = 0


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

    name: str
    name_zh: str = Field(default="", alias="nameZh")
    type: str = ""
    region_to_maps_list: list[RegionMaps] = Field(default_factory=list, alias="regionToMapsList")
    equipment_items: list[NPCItem] = Field(default_factory=list, alias="equipmentItems")
    useable_items: list[NPCItem] = Field(default_factory=list, alias="useableItems")
    scrolls: list[NPCItem] = Field(default_factory=list, alias="scrolls")
    misc_items: list[NPCItem] = Field(default_factory=list, alias="miscItems")
    quests: list[AcquisitionQuest] = Field(default_factory=list)
    recipes: list[CraftingRecipe] = Field(default_factory=list)

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

    name: str
    quantity: int = 0


class CollectItem(_Base):
    """Represents an item to collect for a quest.

    Attributes:
        name: Collected item name.
        quantity: Required collection quantity.
    """

    name: str = ""
    quantity: int = 0


class QuestReward(_Base):
    """Represents rewards for a quest.

    Attributes:
        exp: Reward EXP.
        fame: Reward fame.
        mesos: Reward mesos.
        items: Reward item data.
    """

    exp: int = 0
    fame: int = 0
    mesos: int = 0
    items: dict[str, list[CollectItem]] | list[dict[str, list[CollectItem]]] = Field(
        default_factory=dict
    )


class QuestStep(_Base):
    """Represents a step in a quest.

    Attributes:
        start_npc: NPC that starts the quest step.
        monsters_to_hunt: Monsters required by the quest step.
        items_to_collect: Items required by the quest step.
        reward: Reward data for the quest step.
    """

    start_npc: str = Field(default="", alias="startNPC")
    monsters_to_hunt: list[HuntTarget] = Field(default_factory=list, alias="monstersToHunt")
    items_to_collect: dict[str, list[CollectItem]] = Field(
        default_factory=dict, alias="itemsToCollect"
    )
    reward: QuestReward = Field(default_factory=QuestReward)


class Quest(_Base):
    """Represents a MapleStory quest.

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

    name: str
    name_zh: str = Field(default="", alias="nameZh")
    frequency: str = ""
    lv_lower: int = Field(default=0, alias="lvLower")
    lv_upper: int | None = Field(default=None, alias="lvUpper")
    steps: list[QuestStep] = Field(default_factory=list)
    boss: bool = False
    prerequisites: list[str] = Field(default_factory=list)

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

    Attributes:
        name: NPC name.
        type: NPC type.
        sub_map: Sub-map name.
    """

    name: str
    type: str = ""
    sub_map: str = Field(default="", alias="subMap")


class MapMonster(_Base):
    """Represents a monster on a map.

    Attributes:
        name: Monster name.
        level: Monster level.
    """

    name: str
    level: int = 0


class MapEntry(_Base):
    """Represents a MapleStory map.

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

    region: str = ""
    name: str
    name_zh: str = Field(default="", alias="nameZh")
    x: int = 0
    y: int = 0
    npcs: list[MapNPC] = Field(default_factory=list)
    monsters: list[MapMonster] = Field(default_factory=list)
    hidden: bool = False
    from_map: str = Field(default="", alias="fromMap")
    to_map: str = Field(default="", alias="toMap")
    to_region: str = Field(default="", alias="toRegion")

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

    name: str
    name_zh: str = Field(default="", alias="nameZh")
    type: str = ""
    acquisition: Acquisition = Field(default_factory=Acquisition)

    @property
    def display_name(self) -> str:
        """Returns the miscellaneous item display name.

        Returns:
            The Chinese name when present, otherwise the source name.
        """
        return self.name_zh or self.name


# ── Stats (for /maple_stats command) ────────────────────────────────


class MapleStats(BaseModel):
    """Represents database statistics.

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

    total_monsters: int
    total_equipment: int
    total_scrolls: int
    total_useable: int
    total_npcs: int
    total_quests: int
    total_maps: int
    total_misc: int
    level_distribution: dict[str, int]
    popular_items: list[str]
