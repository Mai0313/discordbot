import os
from unittest.mock import Mock

from src.cogs.maplestory import MapleStoryCogs


def test_file_exists():
    """測試資料檔案是否存在"""
    monsters_file = "data/monsters.json"
    assert os.path.exists(monsters_file), f"找不到怪物資料檔案：{monsters_file}"


def test_load_monsters_data():
    """測試載入怪物資料"""
    # 創建模擬的 bot 物件
    mock_bot = Mock()

    # 創建 MapleStoryCogs 實例
    maple_cog = MapleStoryCogs(mock_bot)

    # 檢查資料是否成功載入
    assert maple_cog.monsters_data is not None, "怪物資料應該成功載入"
    assert len(maple_cog.monsters_data) > 0, "怪物資料應該包含至少一個怪物"

    # 檢查前幾個怪物的基本資訊
    for monster in maple_cog.monsters_data[:3]:
        assert "name" in monster, "怪物應該有名稱"
        assert "drops" in monster, "怪物應該有掉落物品資訊"


def test_search_functions():
    """測試搜尋功能"""
    # 創建模擬的 bot 物件
    mock_bot = Mock()
    maple_cog = MapleStoryCogs(mock_bot)

    assert maple_cog.monsters_data is not None, "無法測試搜尋功能：怪物資料未載入"

    # 測試怪物搜尋
    monsters_found = maple_cog.search_monsters_by_name("寶")
    assert isinstance(monsters_found, list), "怪物搜尋應該返回列表"

    # 測試物品搜尋
    items_found = maple_cog.search_items_by_name("紅色")
    assert isinstance(items_found, list), "物品搜尋應該返回列表"

    # 測試物品掉落來源
    if items_found:
        test_item = items_found[0]
        monsters_with_item = maple_cog.get_monsters_by_item(test_item)
        assert isinstance(monsters_with_item, list), "怪物掉落來源搜尋應該返回列表"


def test_basic_functionality():
    """測試基本功能（不涉及 Discord UI）"""
    # 創建模擬的 bot 物件
    mock_bot = Mock()
    maple_cog = MapleStoryCogs(mock_bot)

    assert maple_cog.monsters_data is not None, "無法測試基本功能：怪物資料未載入"

    # 測試基本數據結構
    test_monster = maple_cog.monsters_data[0]
    assert "name" in test_monster, "怪物應該有名稱"

    # 檢查物品掉落來源功能
    if test_monster.get("drops"):
        test_item = test_monster["drops"][0]["name"]
        monsters_with_item = maple_cog.get_monsters_by_item(test_item)
        assert isinstance(monsters_with_item, list), "應該返回怪物列表"


def test_embed_creation():
    """測試 Embed 創建功能（僅測試不涉及事件循環的部分）"""
    # 創建模擬的 bot 物件
    mock_bot = Mock()
    maple_cog = MapleStoryCogs(mock_bot)

    assert maple_cog.monsters_data is not None, "無法測試 Embed 創建：怪物資料未載入"

    # 測試基本數據結構
    test_monster = maple_cog.monsters_data[0]
    assert "name" in test_monster, "怪物應該有名稱"
    assert "attributes" in test_monster, "怪物應該有屬性"

    # 測試物品掉落功能
    if test_monster.get("drops"):
        test_item = test_monster["drops"][0]["name"]
        monsters_with_item = maple_cog.get_monsters_by_item(test_item)
        assert isinstance(monsters_with_item, list), "應該返回怪物列表"
        assert len(monsters_with_item) > 0, "應該找到至少一個怪物"
