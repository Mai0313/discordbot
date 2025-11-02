from typing import Any
from unittest.mock import Mock, patch

from discordbot.cogs.maplestory import MapleStoryCogs


def load_fake_data() -> list[dict[str, Any]]:
    return [
        {
            "name": "嫩寶",
            "image": "https://mai0313.github.io/artale-drop/image/%E5%AB%A9%E5%AF%B6.png",
            "attributes": {
                "level": 1,
                "hp": 8,
                "mp": 0,
                "exp": 3,
                "evasion": 0,
                "pdef": 0,
                "mdef": 0,
                "accuracy_required": "0 (每少1級 +0)",
            },
            "maps": ["楓之島：嫩寶狩獵場Ⅱ"],
            "drops": [
                {
                    "name": "綠髮帶",
                    "type": "裝備",
                    "link": "https://maplesaga.com/library/cn/permalink/equip/1002067",
                    "img": "https://mai0313.github.io/artale-drop/image/%E7%B6%A0%E9%AB%AE%E5%B8%B6.png",
                }
            ],
        }
    ]


@patch.object(MapleStoryCogs, "_load_monsters_data")
def test_load_monsters_data_with_patch(mock_load_data: Mock) -> None:
    # 設置 mock 的返回值
    mock_load_data.return_value = load_fake_data()

    # 創建模擬的 bot 物件
    mock_bot = Mock()

    # 創建 MapleStoryCogs 實例 (此時會使用 mocked 的 _load_monsters_data)
    maple_cog = MapleStoryCogs(mock_bot)

    # 檢查資料是否成功載入
    assert maple_cog.monsters_data is not None, "怪物資料應該成功載入"
    assert len(maple_cog.monsters_data) > 0, "怪物資料應該包含至少一個怪物"
    assert len(maple_cog.monsters_data) == 1, "應該只有一個測試怪物"

    # 檢查 mock 是否被調用
    mock_load_data.assert_called_once()

    # 檢查怪物的基本資訊
    monster = maple_cog.monsters_data[0]
    assert monster["name"] == "嫩寶", "怪物名稱應該是嫩寶"
    assert "name" in monster, "怪物應該有名稱"
    assert "drops" in monster, "怪物應該有掉落物品資訊"


def test_load_monsters_data_with_context_manager() -> None:
    # 創建模擬的 bot 物件
    mock_bot = Mock()

    # 使用 patch 上下文管理器
    with patch.object(MapleStoryCogs, "_load_monsters_data", return_value=load_fake_data()):
        # 創建 MapleStoryCogs 實例
        maple_cog = MapleStoryCogs(mock_bot)

        # 檢查資料是否成功載入
        assert maple_cog.monsters_data is not None, "怪物資料應該成功載入"
        assert len(maple_cog.monsters_data) > 0, "怪物資料應該包含至少一個怪物"
        assert maple_cog.monsters_data[0]["name"] == "嫩寶", "應該載入測試資料"


def test_load_monsters_data_direct_assignment() -> None:
    # 創建模擬的 bot 物件
    mock_bot = Mock()

    # 創建 MapleStoryCogs 實例
    with patch.object(MapleStoryCogs, "_load_monsters_data", return_value=[]):
        maple_cog = MapleStoryCogs(mock_bot)

    # 直接設置測試資料
    maple_cog.monsters_data = load_fake_data()

    # 檢查資料是否成功設置
    assert maple_cog.monsters_data is not None, "怪物資料應該成功載入"
    assert len(maple_cog.monsters_data) > 0, "怪物資料應該包含至少一個怪物"

    # 檢查怪物的基本資訊
    for monster in maple_cog.monsters_data:
        assert "name" in monster, "怪物應該有名稱"
        assert "drops" in monster, "怪物應該有掉落物品資訊"


def test_load_monsters_data() -> None:
    # 創建模擬的 bot 物件
    mock_bot = Mock()

    # 使用 patch 在創建實例時就 mock _load_monsters_data
    with patch.object(MapleStoryCogs, "_load_monsters_data", return_value=load_fake_data()):
        maple_cog = MapleStoryCogs(mock_bot)

    # 檢查資料是否成功載入
    assert maple_cog.monsters_data is not None, "怪物資料應該成功載入"
    assert len(maple_cog.monsters_data) > 0, "怪物資料應該包含至少一個怪物"

    # 檢查前幾個怪物的基本資訊
    for monster in maple_cog.monsters_data[:1]:
        assert "name" in monster, "怪物應該有名稱"
        assert "drops" in monster, "怪物應該有掉落物品資訊"


@patch.object(MapleStoryCogs, "_load_monsters_data")
def test_search_functions(mock_load_data: Mock) -> None:
    # 設置 mock 的返回值
    mock_load_data.return_value = load_fake_data()

    # 創建模擬的 bot 物件
    mock_bot = Mock()
    maple_cog = MapleStoryCogs(mock_bot)

    assert maple_cog.monsters_data is not None, "無法測試搜尋功能：怪物資料未載入"

    # 測試怪物搜尋
    monsters_found = maple_cog.search_monsters_by_name("寶")
    assert isinstance(monsters_found, list), "怪物搜尋應該返回列表"
    assert len(monsters_found) == 1, "應該找到一個包含'寶'的怪物"
    assert monsters_found[0]["name"] == "嫩寶", "應該找到嫩寶"

    # 測試物品搜尋
    items_found = maple_cog.search_items_by_name("綠")
    assert isinstance(items_found, list), "物品搜尋應該返回列表"
    assert len(items_found) == 1, "應該找到一個包含'綠'的物品"
    assert "綠髮帶" in items_found, "應該找到綠髮帶"

    # 測試物品掉落來源
    monsters_with_item = maple_cog.get_monsters_by_item("綠髮帶")
    assert isinstance(monsters_with_item, list), "怪物掉落來源搜尋應該返回列表"
    assert len(monsters_with_item) == 1, "應該有一個怪物掉落綠髮帶"
    assert monsters_with_item[0]["name"] == "嫩寶", "嫩寶應該掉落綠髮帶"


@patch.object(MapleStoryCogs, "_load_monsters_data")
def test_basic_functionality(mock_load_data: Mock) -> None:
    # 設置 mock 的返回值
    mock_load_data.return_value = load_fake_data()

    # 創建模擬的 bot 物件
    mock_bot = Mock()
    maple_cog = MapleStoryCogs(mock_bot)

    assert maple_cog.monsters_data is not None, "無法測試基本功能：怪物資料未載入"

    # 測試基本數據結構
    test_monster = maple_cog.monsters_data[0]
    assert "name" in test_monster, "怪物應該有名稱"
    assert test_monster["name"] == "嫩寶", "第一個怪物應該是嫩寶"

    # 檢查物品掉落來源功能
    if test_monster.get("drops"):
        test_item = test_monster["drops"][0]["name"]
        assert test_item == "綠髮帶", "第一個掉落物品應該是綠髮帶"

        monsters_with_item = maple_cog.get_monsters_by_item(test_item)
        assert isinstance(monsters_with_item, list), "應該返回怪物列表"
        assert len(monsters_with_item) > 0, "應該找到至少一個掉落該物品的怪物"


@patch.object(MapleStoryCogs, "_load_monsters_data")
def test_embed_creation(mock_load_data: Mock) -> None:
    """測試 Embed 創建功能（僅測試不涉及事件循環的部分）"""
    # 設置 mock 的返回值
    mock_load_data.return_value = load_fake_data()

    # 創建模擬的 bot 物件
    mock_bot = Mock()
    maple_cog = MapleStoryCogs(mock_bot)

    assert maple_cog.monsters_data is not None, "無法測試 Embed 創建：怪物資料未載入"

    # 測試基本數據結構
    test_monster = maple_cog.monsters_data[0]
    assert "name" in test_monster, "怪物應該有名稱"
    assert "attributes" in test_monster, "怪物應該有屬性"
    assert test_monster["name"] == "嫩寶", "怪物名稱應該是嫩寶"

    # 測試物品掉落功能
    if test_monster.get("drops"):
        test_item = test_monster["drops"][0]["name"]
        assert test_item == "綠髮帶", "第一個掉落物品應該是綠髮帶"

        monsters_with_item = maple_cog.get_monsters_by_item(test_item)
        assert isinstance(monsters_with_item, list), "應該返回怪物列表"
        assert len(monsters_with_item) > 0, "應該找到至少一個怪物"
        assert monsters_with_item[0]["name"] == "嫩寶", "應該找到嫩寶"


def test_edge_cases() -> None:
    """測試邊緣情況"""
    mock_bot = Mock()

    # 測試空資料的情況
    with patch.object(MapleStoryCogs, "_load_monsters_data", return_value=[]):
        maple_cog = MapleStoryCogs(mock_bot)

        # 搜尋應該返回空列表
        monsters_found = maple_cog.search_monsters_by_name("測試")
        assert monsters_found == [], "空資料時搜尋怪物應該返回空列表"

        items_found = maple_cog.search_items_by_name("測試")
        assert items_found == [], "空資料時搜尋物品應該返回空列表"

        monsters_with_item = maple_cog.get_monsters_by_item("測試物品")
        assert monsters_with_item == [], "空資料時搜尋物品來源應該返回空列表"


def test_file_not_found_scenario() -> None:
    """測試檔案不存在的情況"""
    mock_bot = Mock()

    # 模擬檔案不存在的情況
    with (
        patch("builtins.open", side_effect=FileNotFoundError),
        patch("logfire.warning") as mock_warning,
    ):
        maple_cog = MapleStoryCogs(mock_bot)

        # 應該載入空資料
        assert maple_cog.monsters_data == [], "檔案不存在時應該載入空資料"

        # 應該記錄警告
        mock_warning.assert_called_once()
