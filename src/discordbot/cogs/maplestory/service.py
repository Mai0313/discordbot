"""In-memory search engine over the local Artale JSON, the non-Discord half of `/maplestory`.

`MapleStoryCogs` builds one `MapleStoryService` in its constructor, and every subcommand plus
every resolver in `views.py` reaches the data through it, so this file holds the feature's whole
data access: the eight category files under `./data/maplestory` and `translations.json` beside
them are read once into validated `models.py` shapes and searched in process. Nothing here writes
to disk or imports nextcord; the files are maintained offline by `scripts/artale_data.py`.

Loading never raises, which is the constraint that shapes `_load_json` and `_load_translations`.
`_load_all` runs from `MapleStoryCogs.__init__`, and `_load_cogs_sync` calls that synchronously
before the gateway connects, so a missing or malformed file has to leave that one category empty
rather than abort boot. It is also why the cog re-tests `has_data` on every command and calls
`reload` when it comes back false: a directory populated after the bot started is picked up
without a restart.

Searches are memoized on the query alone, so a cache is only correct as long as the list under it
does not change; `_load_all` owns loading and clearing together for that reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import logfire
from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError

from .models import NPC, Quest, Scroll, Monster, Useable, MapEntry, MiscItem, Equipment, MapleStats

DEFAULT_DATA_DIR = Path("./data/maplestory")


def _load_json[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    """Reads one category file and validates every top-level element against `model`.

    Never raises: this runs during the synchronous cog load, so an unusable file degrades to an
    empty category instead of aborting boot. The absent case is logged apart from the malformed
    one because only the second is a defect; a category nobody populated is a normal state.

    Args:
        path (Path): The JSON file to read.
        model (type[T]): The model each element is validated into.

    Returns:
        The validated models, or an empty list when the file is missing or unusable.
    """
    try:
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        return [model.model_validate(item) for item in raw]
    except FileNotFoundError as exc:
        logfire.warn(
            "maplestory data file missing, category stays empty",
            path=str(path),
            model=model.__name__,
            _exc_info=exc,
        )
    except (json.JSONDecodeError, ValidationError, OSError, UnicodeDecodeError, TypeError) as exc:
        # TypeError covers a JSON document that parses but is not a list, which
        # would otherwise escape into the synchronous cog load and kill startup.
        logfire.error(
            "failed to load maplestory data",
            path=str(path),
            model=model.__name__,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
    return []


def _load_translations(data_dir: Path) -> dict[str, dict[str, str]]:
    """Reads `translations.json` into per-category source-name to Chinese-name maps.

    A missing or unparsable file degrades to an empty mapping, which leaves every name rendered
    in its source form rather than failing the lookup that wanted it. Unlike the category files
    this is handed back as raw JSON, so nothing checks its shape.

    Args:
        data_dir (Path): The directory holding `translations.json`.

    Returns:
        Chinese names keyed by category and then by source name, empty when the file could not
        be read.
    """
    path = data_dir / "translations.json"
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        logfire.warn(
            "maplestory translations.json missing, names stay untranslated",
            path=str(path),
            _exc_info=exc,
        )
        return {}
    except json.JSONDecodeError as exc:
        logfire.error(
            "failed to parse maplestory translations.json", path=str(path), _exc_info=exc
        )
        return {}


class MapleStoryService(BaseModel):
    """Read-only lookups over one loaded Artale data directory.

    Every category property hands back the loaded list itself rather than a copy, so a caller
    must not mutate one. The `search_*_by_name` helpers are the exception: each memoizes on the
    lowercased query and returns a copy of the cached list, matching the query as a
    case-insensitive substring of either the source name or the Chinese one. The `get_*` name
    lookups take those same two names but demand an exact match and scan without caching.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _monsters: list[Monster] = PrivateAttr(default_factory=list)
    _equipment: list[Equipment] = PrivateAttr(default_factory=list)
    _scrolls: list[Scroll] = PrivateAttr(default_factory=list)
    _useable: list[Useable] = PrivateAttr(default_factory=list)
    _npcs: list[NPC] = PrivateAttr(default_factory=list)
    _quests: list[Quest] = PrivateAttr(default_factory=list)
    _maps: list[MapEntry] = PrivateAttr(default_factory=list)
    _misc: list[MiscItem] = PrivateAttr(default_factory=list)
    _translations: dict[str, dict[str, str]] = PrivateAttr(default_factory=dict)
    # Caches — typed per-category to avoid type-checker issues with generic dict
    _monster_cache: dict[str, list[Monster]] = PrivateAttr(default_factory=dict)
    _equip_cache: dict[str, list[Equipment]] = PrivateAttr(default_factory=dict)
    _scroll_cache: dict[str, list[Scroll]] = PrivateAttr(default_factory=dict)
    _npc_cache: dict[str, list[NPC]] = PrivateAttr(default_factory=dict)
    _quest_cache: dict[str, list[Quest]] = PrivateAttr(default_factory=dict)
    _map_cache: dict[str, list[MapEntry]] = PrivateAttr(default_factory=dict)
    _item_cache: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    _item_type_cache: dict[str, str] = PrivateAttr(default_factory=dict)
    _stats: MapleStats | None = PrivateAttr(default=None)

    @classmethod
    def from_directory(cls, data_dir: Path = DEFAULT_DATA_DIR) -> MapleStoryService:
        """Builds a service with the given directory already loaded.

        Args:
            data_dir (Path): Directory holding the category files and `translations.json`.

        Returns:
            A service whose categories are populated as far as the directory allowed.
        """
        svc = cls()
        svc._load_all(data_dir)
        return svc

    def _load_all(self, data_dir: Path) -> None:
        """Reloads every category and drops everything derived from the previous one.

        The clears are not housekeeping: the search caches are keyed on the query alone and
        `_stats` is a plain memo, so either one left standing would keep answering out of the
        directory that has just been replaced.

        Args:
            data_dir (Path): Directory holding the category files and `translations.json`.
        """
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
        self._item_type_cache.clear()
        self._stats = None

    def reload(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        """Re-reads a data directory in place.

        The cog calls this whenever `has_data` is false, so a directory populated after the bot
        booted becomes searchable without a restart.

        Args:
            data_dir (Path): Directory holding the category files and `translations.json`.
        """
        self._load_all(data_dir)

    def has_data(self) -> bool:
        """Whether the service holds anything worth searching.

        Reads the monster list alone as the proxy for the whole directory, since it backs the
        cross-type item search as well as its own lookups. A directory missing only, say,
        `quests.json` still passes here and that one category simply answers nothing.

        Returns:
            True when at least one monster is loaded.
        """
        return bool(self._monsters)

    def translate(self, category: str, name: str) -> str:
        """Looks a source name up in one translation category.

        An unknown category or name gives `name` back untouched, so a caller can pass every name
        through without first asking whether it is translatable. Beside the entity categories,
        `translations.json` carries enum ones (`region`, `eqType`, `job`, `npcType`, `modifiers`)
        that have no data file behind them.

        Args:
            category (str): Translation category, e.g. `monsters` or `eqType`.
            name (str): The source name to translate.

        Returns:
            The Chinese name, or `name` when the category or the name is absent.
        """
        return self._translations.get(category, {}).get(name, name)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def monsters(self) -> list[Monster]:
        """The monsters loaded from `monsters.json`.

        Returns:
            The live list, not a copy.
        """
        return self._monsters

    @property
    def equipment(self) -> list[Equipment]:
        """The equipment loaded from `equipment.json`.

        Returns:
            The live list, not a copy.
        """
        return self._equipment

    @property
    def scrolls(self) -> list[Scroll]:
        """The scrolls loaded from `scrolls.json`.

        Returns:
            The live list, not a copy.
        """
        return self._scrolls

    @property
    def useable(self) -> list[Useable]:
        """The useable items loaded from `useable.json`.

        Returns:
            The live list, not a copy.
        """
        return self._useable

    @property
    def npcs(self) -> list[NPC]:
        """The NPCs loaded from `npcs.json`.

        Returns:
            The live list, not a copy.
        """
        return self._npcs

    @property
    def quests(self) -> list[Quest]:
        """The quests loaded from `quests.json`.

        Returns:
            The live list, not a copy.
        """
        return self._quests

    @property
    def maps(self) -> list[MapEntry]:
        """The maps loaded from `maps.json`.

        Returns:
            The live list, not a copy.
        """
        return self._maps

    @property
    def misc(self) -> list[MiscItem]:
        """The miscellaneous items loaded from `misc.json`.

        Returns:
            The live list, not a copy.
        """
        return self._misc

    # ── Monster searches ────────────────────────────────────────────

    def search_monsters_by_name(self, query: str) -> list[Monster]:
        """Finds every monster whose name contains `query`.

        Args:
            query (str): Substring of the source or Chinese name, matched case-insensitively.

        Returns:
            A copy of the memoized match list.
        """
        key = query.lower()
        if key not in self._monster_cache:
            self._monster_cache[key] = [
                m for m in self._monsters if key in m.name.lower() or key in m.name_zh.lower()
            ]
        return list(self._monster_cache[key])

    def get_monster(self, name: str) -> Monster | None:
        """Finds the monster named exactly `name`.

        Args:
            name (str): Full source or Chinese name, matched case-insensitively.

        Returns:
            The first monster matching either name, or None.
        """
        name_lower = name.lower()
        for m in self._monsters:
            if m.name.lower() == name_lower or m.name_zh.lower() == name_lower:
                return m
        return None

    def get_monsters_by_drop(self, item_name: str) -> list[Monster]:
        """Finds every monster whose drop table lists `item_name`.

        The item is named in its source form, which is what `search_items_by_name` hands back: a
        drop entry carries no Chinese name of its own.

        Args:
            item_name (str): Full source name of the dropped item, matched case-insensitively.

        Returns:
            The monsters dropping that item.
        """
        q = item_name.lower()
        return [m for m in self._monsters if any(d.name.lower() == q for d in m.drops.all_items)]

    # ── Equipment searches ──────────────────────────────────────────

    def search_equipment_by_name(self, query: str) -> list[Equipment]:
        """Finds every equipment item whose name contains `query`.

        Args:
            query (str): Substring of the source or Chinese name, matched case-insensitively.

        Returns:
            A copy of the memoized match list.
        """
        key = query.lower()
        if key not in self._equip_cache:
            self._equip_cache[key] = [
                e for e in self._equipment if key in e.name.lower() or key in e.name_zh.lower()
            ]
        return list(self._equip_cache[key])

    def get_equipment(self, name: str) -> Equipment | None:
        """Finds the equipment item named exactly `name`.

        Args:
            name (str): Full source or Chinese name, matched case-insensitively.

        Returns:
            The first equipment item matching either name, or None.
        """
        name_lower = name.lower()
        for e in self._equipment:
            if e.name.lower() == name_lower or e.name_zh.lower() == name_lower:
                return e
        return None

    # ── Scroll searches ─────────────────────────────────────────────

    def search_scrolls_by_name(self, query: str) -> list[Scroll]:
        """Finds every scroll whose name contains `query`.

        Args:
            query (str): Substring of the source or Chinese name, matched case-insensitively.

        Returns:
            A copy of the memoized match list.
        """
        key = query.lower()
        if key not in self._scroll_cache:
            self._scroll_cache[key] = [
                s for s in self._scrolls if key in s.name.lower() or key in s.name_zh.lower()
            ]
        return list(self._scroll_cache[key])

    def get_scroll(self, name: str) -> Scroll | None:
        """Finds the scroll named exactly `name`.

        Args:
            name (str): Full source or Chinese name, matched case-insensitively.

        Returns:
            The first scroll matching either name, or None.
        """
        name_lower = name.lower()
        for s in self._scrolls:
            if s.name.lower() == name_lower or s.name_zh.lower() == name_lower:
                return s
        return None

    # ── NPC searches ────────────────────────────────────────────────

    def search_npcs_by_name(self, query: str) -> list[NPC]:
        """Finds every NPC whose name contains `query`.

        Args:
            query (str): Substring of the source or Chinese name, matched case-insensitively.

        Returns:
            A copy of the memoized match list.
        """
        key = query.lower()
        if key not in self._npc_cache:
            self._npc_cache[key] = [
                n for n in self._npcs if key in n.name.lower() or key in n.name_zh.lower()
            ]
        return list(self._npc_cache[key])

    def get_npc(self, name: str) -> NPC | None:
        """Finds the NPC named exactly `name`.

        Args:
            name (str): Full source or Chinese name, matched case-insensitively.

        Returns:
            The first NPC matching either name, or None.
        """
        name_lower = name.lower()
        for n in self._npcs:
            if n.name.lower() == name_lower or n.name_zh.lower() == name_lower:
                return n
        return None

    # ── Quest searches ──────────────────────────────────────────────

    def search_quests_by_name(self, query: str) -> list[Quest]:
        """Finds every quest whose name contains `query`.

        Args:
            query (str): Substring of the source or Chinese name, matched case-insensitively.

        Returns:
            A copy of the memoized match list.
        """
        key = query.lower()
        if key not in self._quest_cache:
            self._quest_cache[key] = [
                q for q in self._quests if key in q.name.lower() or key in q.name_zh.lower()
            ]
        return list(self._quest_cache[key])

    def get_quest(self, name: str) -> Quest | None:
        """Finds the quest named exactly `name`.

        Args:
            name (str): Full source or Chinese name, matched case-insensitively.

        Returns:
            The first quest matching either name, or None.
        """
        name_lower = name.lower()
        for q in self._quests:
            if q.name.lower() == name_lower or q.name_zh.lower() == name_lower:
                return q
        return None

    # ── Map searches ────────────────────────────────────────────────

    def search_maps_by_name(self, query: str) -> list[MapEntry]:
        """Finds every map whose name contains `query`.

        Args:
            query (str): Substring of the source or Chinese name, matched case-insensitively.

        Returns:
            A copy of the memoized match list.
        """
        key = query.lower()
        if key not in self._map_cache:
            self._map_cache[key] = [
                m for m in self._maps if key in m.name.lower() or key in m.name_zh.lower()
            ]
        return list(self._map_cache[key])

    def get_map(self, name: str) -> MapEntry | None:
        """Finds the map named exactly `name`.

        Args:
            name (str): Full source or Chinese name, matched case-insensitively.

        Returns:
            The first map matching either name, or None.
        """
        name_lower = name.lower()
        for m in self._maps:
            if m.name.lower() == name_lower or m.name_zh.lower() == name_lower:
                return m
        return None

    # ── Cross-type item search ──────────────────────────────────────

    def search_items_by_name(self, query: str) -> list[str]:
        """Finds drop item names across every monster's drop table.

        The one search whose subject is not a loaded category. A drop table mixes equipment,
        scrolls, useable and misc entries, so a name is the only thing the four have in common
        and the only thing this can hand back. Those entries carry no Chinese name either, so a
        Chinese query is matched against a translation instead, probed across the same four
        categories in turn and falling back to the source name when none of them knows it.

        Args:
            query (str): Substring of the source name or its translation, matched
                case-insensitively.

        Returns:
            A copy of the memoized list of matching source names, deduplicated and sorted.
        """
        key = query.lower()
        if key not in self._item_cache:
            items_found: set[str] = set()
            for monster in self._monsters:
                for drop in monster.drops.all_items:
                    zh = next(
                        (
                            translated
                            for category in ("equipment", "scrolls", "useable", "misc")
                            if (translated := self.translate(category=category, name=drop.name))
                            != drop.name
                        ),
                        drop.name,
                    )
                    if key in drop.name.lower() or (zh and key in zh.lower()):
                        items_found.add(drop.name)
            self._item_cache[key] = sorted(items_found)
        return list(self._item_cache[key])

    def get_item_type(self, item_name: str) -> str:
        """Classifies a drop item by which bucket of a monster's drop table holds it.

        The item files are not consulted, since only the drop tables carry the split: the first
        monster dropping the item decides, and within that monster the buckets are tried
        equipment, scroll, useable, misc. A miss is memoized as `未知` as well, so an item no
        monster drops is scanned once rather than on every lookup.

        Args:
            item_name (str): Full source name of the item, matched exactly.

        Returns:
            One of `裝備`, `捲軸`, `消耗品`, `其它`, or `未知` when no monster drops it.
        """
        cached = self._item_type_cache.get(item_name)
        if cached is not None:
            return cached
        item_type = "未知"
        for monster in self._monsters:
            if any(drop.name == item_name for drop in monster.drops.equipment_items):
                item_type = "裝備"
                break
            if any(drop.name == item_name for drop in monster.drops.scrolls):
                item_type = "捲軸"
                break
            if any(drop.name == item_name for drop in monster.drops.useable_items):
                item_type = "消耗品"
                break
            if any(drop.name == item_name for drop in monster.drops.misc_items):
                item_type = "其它"
                break
        self._item_type_cache[item_name] = item_type
        return item_type

    # ── Stats ───────────────────────────────────────────────────────

    def get_level_distribution(self) -> dict[str, int]:
        """Counts the loaded monsters per ten-level band.

        Recomputed on every call; `get_stats` is where the result is memoized. Bands appear in
        the order they are first met rather than sorted, so a caller rendering them sorts first.

        Returns:
            Monster counts keyed by band label, e.g. `0-9`.
        """
        dist: dict[str, int] = {}
        for m in self._monsters:
            start = (m.level // 10) * 10
            key = f"{start}-{start + 9}"
            dist[key] = dist.get(key, 0) + 1
        return dist

    def get_popular_items(self) -> list[str]:
        """Ranks every dropped item by how many monsters drop it.

        The whole ranking comes back and `get_stats` keeps the head of it. Ties hold the order
        the items were first met, since the sort is stable over the counting dict.

        Returns:
            Source item names, most-dropped first.
        """
        counts: dict[str, int] = {}
        for m in self._monsters:
            for drop in m.drops.all_items:
                counts[drop.name] = counts.get(drop.name, 0) + 1
        return [name for name, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)]

    def get_stats(self) -> MapleStats:
        """Summarises the loaded directory, computing the summary once.

        The same instance comes back on every later call and only `_load_all` drops it, so a
        caller must not mutate what it gets.

        Returns:
            Category totals plus the level distribution and the 20 most-dropped items.
        """
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
