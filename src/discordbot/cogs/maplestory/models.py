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

    Attributes:
        required: Required accuracy value.
        decrease: Accuracy decrease value.
    """

    required: int = Field(default=0, description="Required accuracy value.")
    decrease: float = Field(default=0, description="Accuracy decrease value.")


class DropItem(_Base):
    """Represents an item dropped by a monster.

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

    name: str = Field(..., description="Quest name.")
    level: int = Field(default=0, description="Quest level.")


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

    middle: int = Field(default=0, description="Middle stat value.")
    range: list[int] = Field(default_factory=list, description="Stat value range.")


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

    str_req: int = Field(default=0, alias="str", description="STR requirement.")
    dex: int = Field(default=0, description="DEX requirement.")
    int_req: int = Field(default=0, alias="int", description="INT requirement.")
    luk: int = Field(default=0, description="LUK requirement.")

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

    Attributes:
        amount: Amount applied by the useable item stat.
    """

    amount: int = Field(default=0, description="Amount applied by the useable item stat.")


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
