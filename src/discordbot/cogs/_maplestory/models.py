from typing import Any

from pydantic import BaseModel

Monster = dict[str, Any]


class MapleStats(BaseModel):
    total_monsters: int
    total_items: int
    total_maps: int
    level_distribution: dict[str, int]
    popular_items: list[str]
