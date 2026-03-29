from __future__ import annotations

from typing import Any

from pydantic import Field, BaseModel, ConfigDict


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ── Shared sub-models ───────────────────────────────────────────────


class RegionMaps(_Base):
    region: str
    maps: list[str] = []


class AcquisitionMonster(_Base):
    name: str
    level: int = 0


class AcquisitionNPC(_Base):
    name: str
    price: int = 0


class AcquisitionQuest(_Base):
    name: str
    level: int = 0


class Acquisition(_Base):
    monsters: list[AcquisitionMonster] = []
    npcs: list[AcquisitionNPC] = []
    quests: list[AcquisitionQuest] = []
    craftings: list[dict[str, Any]] = []


# ── Monster ─────────────────────────────────────────────────────────


class DefenseStats(_Base):
    weapon: int = 0
    magic: int = 0
    avoidability: int = 0


class AccuracyStats(_Base):
    required: int = 0
    decrease: float = 0


class DropItem(_Base):
    name: str
    level: int = 0
    type: str = ""
    jobs: list[str] = []


class MonsterDrops(_Base):
    equipment_items: list[DropItem] = Field(default_factory=list, alias="equipmentItems")
    useable_items: list[DropItem] = Field(default_factory=list, alias="useableItems")
    scrolls: list[DropItem] = Field(default_factory=list, alias="scrolls")
    misc_items: list[DropItem] = Field(default_factory=list, alias="miscItems")
    meso_range: list[int] = Field(default_factory=list, alias="mesoRange")

    @property
    def all_items(self) -> list[DropItem]:
        return self.equipment_items + self.useable_items + self.scrolls + self.misc_items


class MonsterQuest(_Base):
    name: str
    level: int = 0


class Monster(_Base):
    name: str
    name_zh: str = Field(default="", alias="nameZh")
    level: int = 0
    hp: int = 0
    mp: int = 0
    exp: int = 0
    def_stats: DefenseStats = Field(default_factory=DefenseStats, alias="def")
    accuracy: AccuracyStats = Field(default_factory=AccuracyStats)
    modifiers: list[str] = []
    region_to_maps_list: list[RegionMaps] = Field(default_factory=list, alias="regionToMapsList")
    drops: MonsterDrops = Field(default_factory=MonsterDrops)
    quests: list[MonsterQuest] = []

    @property
    def display_name(self) -> str:
        return self.name_zh or self.name

    @property
    def all_maps(self) -> list[str]:
        return [m for r in self.region_to_maps_list for m in r.maps]


# ── Equipment ───────────────────────────────────────────────────────


class StatValue(_Base):
    middle: int = 0
    range: list[int] = Field(default_factory=list)


class EquipmentStats(_Base):
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
        """Return (label, value) pairs for stats with non-zero middle."""
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
    str_req: int = Field(default=0, alias="str")
    dex: int = 0
    int_req: int = Field(default=0, alias="int")
    luk: int = 0

    def has_requirements(self) -> bool:
        return any((self.str_req, self.dex, self.int_req, self.luk))


class Equipment(_Base):
    type: str = ""
    name: str
    name_zh: str = Field(default="", alias="nameZh")
    level: int = 0
    equipment_restriction: EquipmentRestriction = Field(
        default_factory=EquipmentRestriction, alias="equipmentRestriction"
    )
    stats: EquipmentStats = Field(default_factory=EquipmentStats)
    jobs: list[str] = []
    attack_speed: str = Field(default="", alias="attackSpeed")
    acquisition: Acquisition = Field(default_factory=Acquisition)
    tradeable: str = ""
    event: bool = False
    limited_time: bool = Field(default=False, alias="limitedTime")
    unavailable: bool = False

    @property
    def display_name(self) -> str:
        return self.name_zh or self.name


# ── Scroll ──────────────────────────────────────────────────────────


class Scroll(_Base):
    name: str
    name_zh: str = Field(default="", alias="nameZh")
    stats: dict[str, int] = {}
    type: str = ""
    acquisition: Acquisition = Field(default_factory=Acquisition)

    @property
    def display_name(self) -> str:
        return self.name_zh or self.name


# ── Useable ─────────────────────────────────────────────────────────


class UseableStat(_Base):
    amount: int = 0


class Useable(_Base):
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
        return self.name_zh or self.name


# ── NPC ─────────────────────────────────────────────────────────────


class NPCItem(_Base):
    name: str
    price: int = 0


class NPC(_Base):
    name: str
    name_zh: str = Field(default="", alias="nameZh")
    type: str = ""
    region_to_maps_list: list[RegionMaps] = Field(default_factory=list, alias="regionToMapsList")
    equipment_items: list[NPCItem] = Field(default_factory=list, alias="equipmentItems")
    useable_items: list[NPCItem] = Field(default_factory=list, alias="useableItems")
    scrolls: list[NPCItem] = Field(default_factory=list, alias="scrolls")
    misc_items: list[NPCItem] = Field(default_factory=list, alias="miscItems")
    quests: list[AcquisitionQuest] = []
    recipes: list[dict[str, Any]] = []

    @property
    def display_name(self) -> str:
        return self.name_zh or self.name

    @property
    def all_maps(self) -> list[str]:
        return [m for r in self.region_to_maps_list for m in r.maps]


# ── Quest ───────────────────────────────────────────────────────────


class HuntTarget(_Base):
    name: str
    quantity: int = 0


class QuestStep(_Base):
    start_npc: str = Field(default="", alias="startNPC")
    monsters_to_hunt: list[HuntTarget] = Field(default_factory=list, alias="monstersToHunt")
    items_to_collect: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict, alias="itemsToCollect"
    )
    reward: dict[str, Any] = {}


class Quest(_Base):
    name: str
    name_zh: str = Field(default="", alias="nameZh")
    frequency: str = ""
    lv_lower: int = Field(default=0, alias="lvLower")
    lv_upper: int | None = Field(default=None, alias="lvUpper")
    steps: list[QuestStep] = []
    boss: bool = False
    prerequisites: list[str] = []

    @property
    def display_name(self) -> str:
        return self.name_zh or self.name


# ── Map ─────────────────────────────────────────────────────────────


class MapNPC(_Base):
    name: str
    type: str = ""
    sub_map: str = Field(default="", alias="subMap")


class MapMonster(_Base):
    name: str
    level: int = 0


class MapEntry(_Base):
    region: str = ""
    name: str
    name_zh: str = Field(default="", alias="nameZh")
    x: int = 0
    y: int = 0
    npcs: list[MapNPC] = []
    monsters: list[MapMonster] = []
    hidden: bool = False
    from_map: str = Field(default="", alias="fromMap")
    to_map: str = Field(default="", alias="toMap")
    to_region: str = Field(default="", alias="toRegion")

    @property
    def display_name(self) -> str:
        return self.name_zh or self.name


# ── Misc Item ───────────────────────────────────────────────────────


class MiscItem(_Base):
    name: str
    name_zh: str = Field(default="", alias="nameZh")
    type: str = ""
    acquisition: Acquisition = Field(default_factory=Acquisition)

    @property
    def display_name(self) -> str:
        return self.name_zh or self.name


# ── Stats (for /maple_stats command) ────────────────────────────────


class MapleStats(BaseModel):
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
