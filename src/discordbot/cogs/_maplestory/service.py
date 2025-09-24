import json
from pathlib import Path

import logfire

from .models import Monster, MapleStats

DEFAULT_DATA_PATH = Path("./data/monsters.json")


def load_monsters_data(file_path: Path = DEFAULT_DATA_PATH) -> list[Monster]:
    try:
        with file_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        logfire.warning("找不到怪物資料檔案 %s", file_path)
    except json.JSONDecodeError as exc:
        logfire.error("無法解析怪物資料檔案 - %s", exc)
    return []


class MapleStoryService:
    """Encapsulates MapleStory monster/item lookups with simple caching."""

    def __init__(self, monsters: list[Monster] | None = None):
        self._monsters: list[Monster] = monsters or []
        self._monster_cache: dict[str, list[Monster]] = {}
        self._item_cache: dict[str, list[str]] = {}
        self._popular_items: list[str] | None = None
        self._stats: MapleStats | None = None

    @classmethod
    def from_file(cls, file_path: Path = DEFAULT_DATA_PATH) -> "MapleStoryService":
        data = load_monsters_data(file_path)
        return cls(data)

    @property
    def monsters(self) -> list[Monster]:
        return self._monsters

    def reload(self, file_path: Path = DEFAULT_DATA_PATH) -> None:
        self._monsters = load_monsters_data(file_path)
        self.clear_caches()

    def clear_caches(self) -> None:
        self._monster_cache.clear()
        self._item_cache.clear()
        self._popular_items = None
        self._stats = None

    def has_data(self) -> bool:
        return bool(self._monsters)

    def search_monsters_by_name(self, query: str) -> list[Monster]:
        key = query.lower()
        if key not in self._monster_cache:
            results = [m for m in self._monsters if key in m.get("name", "").lower()]
            self._monster_cache[key] = results
        return list(self._monster_cache[key])

    def search_items_by_name(self, query: str) -> list[str]:
        key = query.lower()
        if key not in self._item_cache:
            items_found = {
                drop.get("name", "")
                for monster in self._monsters
                for drop in monster.get("drops", [])
                if key in drop.get("name", "").lower()
            }
            self._item_cache[key] = sorted(filter(None, items_found))
        return list(self._item_cache[key])

    def get_monster(self, name: str) -> Monster | None:
        name_lower = name.lower()
        for monster in self._monsters:
            if monster.get("name", "").lower() == name_lower:
                return monster
        return None

    def get_monsters_by_item(self, item_name: str) -> list[Monster]:
        item_lower = item_name.lower()
        matches: list[Monster] = []
        for monster in self._monsters:
            if any(
                drop.get("name", "").lower() == item_lower for drop in monster.get("drops", [])
            ):
                matches.append(monster)
        return matches

    def get_item_type(self, item_name: str) -> str:
        item_lower = item_name.lower()
        for monster in self._monsters:
            for drop in monster.get("drops", []):
                if drop.get("name", "").lower() == item_lower:
                    return drop.get("type", "未知")
        return "未知"

    def get_popular_items(self) -> list[str]:
        if self._popular_items is None:
            item_count: dict[str, int] = {}
            for monster in self._monsters:
                for drop in monster.get("drops", []):
                    name = drop.get("name")
                    if not name:
                        continue
                    item_count[name] = item_count.get(name, 0) + 1
            self._popular_items = [
                item
                for item, _ in sorted(item_count.items(), key=lambda entry: entry[1], reverse=True)
            ]
        return list(self._popular_items)

    def get_level_distribution(self) -> dict[str, int]:
        distribution: dict[str, int] = {}
        for monster in self._monsters:
            level = int(monster.get("attributes", {}).get("level", 0) or 0)
            range_start = (level // 10) * 10
            key = f"{range_start}-{range_start + 9}"
            distribution[key] = distribution.get(key, 0) + 1
        return distribution

    def get_stats(self) -> MapleStats:
        if self._stats is None:
            total_monsters = len(self._monsters)
            total_items = len({
                drop.get("name")
                for monster in self._monsters
                for drop in monster.get("drops", [])
                if drop.get("name")
            })
            total_maps = len({
                map_name
                for monster in self._monsters
                for map_name in monster.get("maps", [])
                if map_name
            })
            level_distribution = self.get_level_distribution()
            popular_items = self.get_popular_items()
            self._stats = MapleStats(
                total_monsters=total_monsters,
                total_items=total_items,
                total_maps=total_maps,
                level_distribution=level_distribution,
                popular_items=popular_items,
            )
        return self._stats
