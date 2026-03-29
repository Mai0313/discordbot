import json
from pathlib import Path
from typing import TypeVar

import logfire
from pydantic import BaseModel

from .models import (
    Equipment,
    MapEntry,
    MapleStats,
    MiscItem,
    Monster,
    NPC,
    Quest,
    Scroll,
    Useable,
)

DEFAULT_DATA_DIR = Path("./data/maplestory")

T = TypeVar("T", bound=BaseModel)


def _load_json(path: Path, model: type[T]) -> list[T]:
    try:
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        return [model.model_validate(item) for item in raw]
    except FileNotFoundError:
        logfire.warning("找不到資料檔案 %s", path)
    except (json.JSONDecodeError, Exception) as exc:
        logfire.error("無法載入 %s — %s", path, exc)
    return []


def _load_translations(data_dir: Path) -> dict[str, dict[str, str]]:
    path = data_dir / "translations.json"
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class MapleStoryService:
    """Encapsulates all Artale data lookups with caching."""

    def __init__(self) -> None:
        self._monsters: list[Monster] = []
        self._equipment: list[Equipment] = []
        self._scrolls: list[Scroll] = []
        self._useable: list[Useable] = []
        self._npcs: list[NPC] = []
        self._quests: list[Quest] = []
        self._maps: list[MapEntry] = []
        self._misc: list[MiscItem] = []
        self._translations: dict[str, dict[str, str]] = {}
        # Caches
        self._cache: dict[str, object] = {}

    @classmethod
    def from_directory(cls, data_dir: Path = DEFAULT_DATA_DIR) -> "MapleStoryService":
        svc = cls()
        svc._load_all(data_dir)
        return svc

    # Keep backwards compat for existing callers
    @classmethod
    def from_file(cls, file_path: Path = DEFAULT_DATA_DIR / "monsters.json") -> "MapleStoryService":
        return cls.from_directory(file_path.parent)

    def _load_all(self, data_dir: Path) -> None:
        self._monsters = _load_json(data_dir / "monsters.json", Monster)
        self._equipment = _load_json(data_dir / "equipment.json", Equipment)
        self._scrolls = _load_json(data_dir / "scrolls.json", Scroll)
        self._useable = _load_json(data_dir / "useable.json", Useable)
        self._npcs = _load_json(data_dir / "npcs.json", NPC)
        self._quests = _load_json(data_dir / "quests.json", Quest)
        self._maps = _load_json(data_dir / "maps.json", MapEntry)
        self._misc = _load_json(data_dir / "misc.json", MiscItem)
        self._translations = _load_translations(data_dir)
        self._cache.clear()

    def reload(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self._load_all(data_dir)

    def has_data(self) -> bool:
        return bool(self._monsters)

    def translate(self, category: str, name: str) -> str:
        """Translate an English name to Chinese using the translations dict."""
        return self._translations.get(category, {}).get(name, name)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def monsters(self) -> list[Monster]:
        return self._monsters

    @property
    def equipment(self) -> list[Equipment]:
        return self._equipment

    @property
    def scrolls(self) -> list[Scroll]:
        return self._scrolls

    @property
    def useable(self) -> list[Useable]:
        return self._useable

    @property
    def npcs(self) -> list[NPC]:
        return self._npcs

    @property
    def quests(self) -> list[Quest]:
        return self._quests

    @property
    def maps(self) -> list[MapEntry]:
        return self._maps

    @property
    def misc(self) -> list[MiscItem]:
        return self._misc

    # ── Monster searches ────────────────────────────────────────────

    def search_monsters_by_name(self, query: str) -> list[Monster]:
        key = f"monster:{query.lower()}"
        if key not in self._cache:
            q = query.lower()
            self._cache[key] = [
                m for m in self._monsters
                if q in m.name.lower() or q in m.name_zh.lower()
            ]
        return list(self._cache[key])  # type: ignore[arg-type]

    def get_monster(self, name: str) -> Monster | None:
        name_lower = name.lower()
        for m in self._monsters:
            if m.name.lower() == name_lower or m.name_zh.lower() == name_lower:
                return m
        return None

    def get_monsters_by_drop(self, item_name: str) -> list[Monster]:
        q = item_name.lower()
        return [
            m for m in self._monsters
            if any(d.name.lower() == q for d in m.drops.all_items)
        ]

    # ── Equipment searches ──────────────────────────────────────────

    def search_equipment_by_name(self, query: str) -> list[Equipment]:
        key = f"equip:{query.lower()}"
        if key not in self._cache:
            q = query.lower()
            self._cache[key] = [
                e for e in self._equipment
                if q in e.name.lower() or q in e.name_zh.lower()
            ]
        return list(self._cache[key])  # type: ignore[arg-type]

    def get_equipment(self, name: str) -> Equipment | None:
        name_lower = name.lower()
        for e in self._equipment:
            if e.name.lower() == name_lower or e.name_zh.lower() == name_lower:
                return e
        return None

    # ── Scroll searches ─────────────────────────────────────────────

    def search_scrolls_by_name(self, query: str) -> list[Scroll]:
        key = f"scroll:{query.lower()}"
        if key not in self._cache:
            q = query.lower()
            self._cache[key] = [
                s for s in self._scrolls
                if q in s.name.lower() or q in s.name_zh.lower()
            ]
        return list(self._cache[key])  # type: ignore[arg-type]

    # ── NPC searches ────────────────────────────────────────────────

    def search_npcs_by_name(self, query: str) -> list[NPC]:
        key = f"npc:{query.lower()}"
        if key not in self._cache:
            q = query.lower()
            self._cache[key] = [
                n for n in self._npcs
                if q in n.name.lower() or q in n.name_zh.lower()
            ]
        return list(self._cache[key])  # type: ignore[arg-type]

    # ── Quest searches ──────────────────────────────────────────────

    def search_quests_by_name(self, query: str) -> list[Quest]:
        key = f"quest:{query.lower()}"
        if key not in self._cache:
            q = query.lower()
            self._cache[key] = [
                quest for quest in self._quests
                if q in quest.name.lower() or q in quest.name_zh.lower()
            ]
        return list(self._cache[key])  # type: ignore[arg-type]

    # ── Map searches ────────────────────────────────────────────────

    def search_maps_by_name(self, query: str) -> list[MapEntry]:
        key = f"map:{query.lower()}"
        if key not in self._cache:
            q = query.lower()
            self._cache[key] = [
                m for m in self._maps
                if q in m.name.lower() or q in m.name_zh.lower()
            ]
        return list(self._cache[key])  # type: ignore[arg-type]

    # ── Cross-type item search ──────────────────────────────────────

    def search_items_by_name(self, query: str) -> list[str]:
        """Search all drop item names across monsters."""
        key = f"item:{query.lower()}"
        if key not in self._cache:
            q = query.lower()
            items_found: set[str] = set()
            for monster in self._monsters:
                for drop in monster.drops.all_items:
                    zh = self.translate("equipment", drop.name) or \
                         self.translate("scrolls", drop.name) or \
                         self.translate("useable", drop.name) or \
                         self.translate("misc", drop.name)
                    if q in drop.name.lower() or (zh and q in zh.lower()):
                        items_found.add(drop.name)
            self._cache[key] = sorted(items_found)
        return list(self._cache[key])  # type: ignore[arg-type]

    def get_item_type(self, item_name: str) -> str:
        """Determine an item's category from monster drops."""
        for monster in self._monsters:
            for drop in monster.drops.equipment_items:
                if drop.name == item_name:
                    return "裝備"
            for drop in monster.drops.scrolls:
                if drop.name == item_name:
                    return "捲軸"
            for drop in monster.drops.useable_items:
                if drop.name == item_name:
                    return "消耗品"
            for drop in monster.drops.misc_items:
                if drop.name == item_name:
                    return "其它"
        return "未知"

    # ── Stats ───────────────────────────────────────────────────────

    def get_level_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for m in self._monsters:
            start = (m.level // 10) * 10
            key = f"{start}-{start + 9}"
            dist[key] = dist.get(key, 0) + 1
        return dist

    def get_popular_items(self) -> list[str]:
        counts: dict[str, int] = {}
        for m in self._monsters:
            for drop in m.drops.all_items:
                counts[drop.name] = counts.get(drop.name, 0) + 1
        return [name for name, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)]

    def get_stats(self) -> MapleStats:
        if "stats" not in self._cache:
            self._cache["stats"] = MapleStats(
                total_monsters=len(self._monsters),
                total_equipment=len(self._equipment),
                total_scrolls=len(self._scrolls),
                total_useable=len(self._useable),
                total_npcs=len(self._npcs),
                total_quests=len(self._quests),
                total_maps=len(self._maps),
                total_misc=len(self._misc),
                level_distribution=self.get_level_distribution(),
                popular_items=self.get_popular_items()[:20],
            )
        return self._cache["stats"]  # type: ignore[return-value]
