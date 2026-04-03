from __future__ import annotations

import json
from typing import TypeVar
from pathlib import Path

import logfire
from pydantic import BaseModel

from .models import NPC, Quest, Scroll, Monster, Useable, MapEntry, MiscItem, Equipment, MapleStats

DEFAULT_DATA_DIR = Path("./data/maplestory")

T = TypeVar("T", bound=BaseModel)


def _load_json[T: BaseModel](path: Path, model: type[T]) -> list[T]:
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
        # Caches — typed per-category to avoid mypy issues with generic dict
        self._monster_cache: dict[str, list[Monster]] = {}
        self._equip_cache: dict[str, list[Equipment]] = {}
        self._scroll_cache: dict[str, list[Scroll]] = {}
        self._npc_cache: dict[str, list[NPC]] = {}
        self._quest_cache: dict[str, list[Quest]] = {}
        self._map_cache: dict[str, list[MapEntry]] = {}
        self._item_cache: dict[str, list[str]] = {}
        self._stats: MapleStats | None = None

    @classmethod
    def from_directory(cls, data_dir: Path = DEFAULT_DATA_DIR) -> MapleStoryService:
        svc = cls()
        svc._load_all(data_dir)
        return svc

    # Keep backwards compat for existing callers
    @classmethod
    def from_file(cls, file_path: Path = DEFAULT_DATA_DIR / "monsters.json") -> MapleStoryService:
        return cls.from_directory(file_path.parent)

    def _load_all(self, data_dir: Path) -> None:
        self._monsters = _load_json(path=data_dir / "monsters.json", model=Monster)
        self._equipment = _load_json(path=data_dir / "equipment.json", model=Equipment)
        self._scrolls = _load_json(path=data_dir / "scrolls.json", model=Scroll)
        self._useable = _load_json(path=data_dir / "useable.json", model=Useable)
        self._npcs = _load_json(path=data_dir / "npcs.json", model=NPC)
        self._quests = _load_json(path=data_dir / "quests.json", model=Quest)
        self._maps = _load_json(path=data_dir / "maps.json", model=MapEntry)
        self._misc = _load_json(path=data_dir / "misc.json", model=MiscItem)
        self._translations = _load_translations(data_dir)
        self._monster_cache.clear()
        self._equip_cache.clear()
        self._scroll_cache.clear()
        self._npc_cache.clear()
        self._quest_cache.clear()
        self._map_cache.clear()
        self._item_cache.clear()
        self._stats = None

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
        key = query.lower()
        if key not in self._monster_cache:
            self._monster_cache[key] = [
                m for m in self._monsters if key in m.name.lower() or key in m.name_zh.lower()
            ]
        return list(self._monster_cache[key])

    def get_monster(self, name: str) -> Monster | None:
        name_lower = name.lower()
        for m in self._monsters:
            if m.name.lower() == name_lower or m.name_zh.lower() == name_lower:
                return m
        return None

    def get_monsters_by_drop(self, item_name: str) -> list[Monster]:
        q = item_name.lower()
        return [m for m in self._monsters if any(d.name.lower() == q for d in m.drops.all_items)]

    # ── Equipment searches ──────────────────────────────────────────

    def search_equipment_by_name(self, query: str) -> list[Equipment]:
        key = query.lower()
        if key not in self._equip_cache:
            self._equip_cache[key] = [
                e for e in self._equipment if key in e.name.lower() or key in e.name_zh.lower()
            ]
        return list(self._equip_cache[key])

    def get_equipment(self, name: str) -> Equipment | None:
        name_lower = name.lower()
        for e in self._equipment:
            if e.name.lower() == name_lower or e.name_zh.lower() == name_lower:
                return e
        return None

    # ── Scroll searches ─────────────────────────────────────────────

    def search_scrolls_by_name(self, query: str) -> list[Scroll]:
        key = query.lower()
        if key not in self._scroll_cache:
            self._scroll_cache[key] = [
                s for s in self._scrolls if key in s.name.lower() or key in s.name_zh.lower()
            ]
        return list(self._scroll_cache[key])

    # ── NPC searches ────────────────────────────────────────────────

    def search_npcs_by_name(self, query: str) -> list[NPC]:
        key = query.lower()
        if key not in self._npc_cache:
            self._npc_cache[key] = [
                n for n in self._npcs if key in n.name.lower() or key in n.name_zh.lower()
            ]
        return list(self._npc_cache[key])

    # ── Quest searches ──────────────────────────────────────────────

    def search_quests_by_name(self, query: str) -> list[Quest]:
        key = query.lower()
        if key not in self._quest_cache:
            self._quest_cache[key] = [
                q for q in self._quests if key in q.name.lower() or key in q.name_zh.lower()
            ]
        return list(self._quest_cache[key])

    # ── Map searches ────────────────────────────────────────────────

    def search_maps_by_name(self, query: str) -> list[MapEntry]:
        key = query.lower()
        if key not in self._map_cache:
            self._map_cache[key] = [
                m for m in self._maps if key in m.name.lower() or key in m.name_zh.lower()
            ]
        return list(self._map_cache[key])

    # ── Cross-type item search ──────────────────────────────────────

    def search_items_by_name(self, query: str) -> list[str]:
        """Search all drop item names across monsters."""
        key = query.lower()
        if key not in self._item_cache:
            items_found: set[str] = set()
            for monster in self._monsters:
                for drop in monster.drops.all_items:
                    zh = (
                        self.translate(category="equipment", name=drop.name)
                        or self.translate(category="scrolls", name=drop.name)
                        or self.translate(category="useable", name=drop.name)
                        or self.translate(category="misc", name=drop.name)
                    )
                    if key in drop.name.lower() or (zh and key in zh.lower()):
                        items_found.add(drop.name)
            self._item_cache[key] = sorted(items_found)
        return list(self._item_cache[key])

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
        if self._stats is None:
            self._stats = MapleStats(
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
        return self._stats
