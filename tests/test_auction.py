import os
import sqlite3
from datetime import datetime, timedelta
import tempfile

import pytest

from discordbot.cogs.auction import Bid, Auction, AuctionDatabase, get_currency_display


def test_auction_model_creation():
    """測試 Auction 模型創建"""
    end_time = datetime.now() + timedelta(hours=24)

    auction = Auction(
        guild_id=123456789,
        item_name="測試物品",
        starting_price=100000,
        increment=10000,
        creator_id=123456,
        creator_name="測試者",
        end_time=end_time,
        current_price=100000,
    )

    assert auction.guild_id == 123456789
    assert auction.item_name == "測試物品"
    assert auction.starting_price == 100000
    assert auction.increment == 10000
    assert auction.creator_id == 123456
    assert auction.creator_name == "測試者"
    assert auction.current_price == 100000
    assert auction.end_time == end_time
    assert auction.is_active is True


def test_bid_model_creation():
    """測試 Bid 模型創建"""
    bid = Bid(
        auction_id=1, guild_id=123456789, bidder_id=987654321, bidder_name="出價者", amount=150000
    )

    assert bid.auction_id == 1
    assert bid.guild_id == 123456789
    assert bid.bidder_id == 987654321
    assert bid.bidder_name == "出價者"
    assert bid.amount == 150000
    assert isinstance(bid.timestamp, datetime)


def test_auction_database_initialization():
    """測試競標資料庫初始化"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        AuctionDatabase(temp_db_path)
        assert os.path.exists(temp_db_path)
    finally:
        # 清理測試檔案
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_create_and_get_auction():
    """測試創建和獲取競標"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 測試創建競標
        test_auction = Auction(
            guild_id=123456789,
            item_name="測試武器+10",
            starting_price=1000000,
            increment=100000,
            duration_hours=24,
            creator_id=123456789,
            creator_name="測試用戶",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=1000000,
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None
        assert auction_id > 0

        # 測試獲取競標
        retrieved_auction = db.get_auction(auction_id, 123456789)
        assert retrieved_auction is not None
        assert retrieved_auction.guild_id == 123456789
        assert retrieved_auction.item_name == "測試武器+10"
        assert retrieved_auction.starting_price == 1000000
        assert retrieved_auction.increment == 100000
        assert retrieved_auction.creator_id == 123456789
        assert retrieved_auction.creator_name == "測試用戶"
        assert retrieved_auction.current_price == 1000000

    finally:
        # 清理測試檔案
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_place_bid():
    """測試出價功能"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建測試競標
        test_auction = Auction(
            guild_id=123456789,
            item_name="測試武器+10",
            starting_price=1000000,
            increment=100000,
            duration_hours=24,
            creator_id=123456789,
            creator_name="測試用戶",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=1000000,
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None

        # 測試出價
        success = db.place_bid(auction_id, 987654321, "出價者", 1200000, 123456789)
        assert success is True

        # 驗證價格更新
        updated_auction = db.get_auction(auction_id, 123456789)
        assert updated_auction is not None
        assert updated_auction.current_price == 1200000
        assert updated_auction.current_bidder_id == 987654321
        assert updated_auction.current_bidder_name == "出價者"

    finally:
        # 清理測試檔案
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_invalid_bid():
    """測試無效出價"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建測試競標
        test_auction = Auction(
            guild_id=123456789,
            item_name="測試武器+10",
            starting_price=1000000,
            increment=100000,
            duration_hours=24,
            creator_id=123456789,
            creator_name="測試用戶",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=1000000,
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None

        # 測試出價金額不足（等於當前價格）
        success = db.place_bid(auction_id, 987654321, "出價者", 1000000, 123456789)
        assert success is False

        # 測試出價金額不足（小於當前價格）
        success = db.place_bid(auction_id, 987654321, "出價者", 900000, 123456789)
        assert success is False

    finally:
        # 清理測試檔案
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_get_auction_bids():
    """測試獲取出價記錄"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建測試競標
        test_auction = Auction(
            guild_id=123456789,
            item_name="測試武器+10",
            starting_price=1000000,
            increment=100000,
            duration_hours=24,
            creator_id=123456789,
            creator_name="測試用戶",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=1000000,
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None

        # 進行多次出價
        db.place_bid(auction_id, 111111111, "出價者1", 1200000, 123456789)
        db.place_bid(auction_id, 222222222, "出價者2", 1400000, 123456789)
        db.place_bid(auction_id, 333333333, "出價者3", 1600000, 123456789)

        # 測試獲取出價記錄
        bids = db.get_auction_bids(auction_id, 123456789)
        assert bids is not None
        assert len(bids) == 3

        # 驗證出價記錄按金額降序排列
        assert bids[0].amount == 1600000
        assert bids[1].amount == 1400000
        assert bids[2].amount == 1200000

    finally:
        # 清理測試檔案
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_get_active_auctions():
    """測試獲取活躍競標"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建多個測試競標
        for i in range(3):
            test_auction = Auction(
                guild_id=123456789,
                item_name=f"測試武器+{i}",
                starting_price=1000000 + i * 100000,
                increment=100000,
                duration_hours=24,
                creator_id=123456789 + i,
                creator_name=f"測試用戶{i}",
                end_time=datetime.now() + timedelta(hours=24),
                current_price=1000000 + i * 100000,
            )
            db.create_auction(test_auction)

        # 測試獲取活躍競標
        active_auctions = db.get_active_auctions(123456789)
        assert active_auctions is not None
        assert len(active_auctions) == 3

        # 驗證所有競標都是活躍的
        for auction in active_auctions:
            assert auction.is_active is True
            assert auction.end_time > datetime.now()

    finally:
        # 清理測試檔案
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_end_auction():
    """測試結束競標"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建測試競標
        test_auction = Auction(
            guild_id=123456789,
            item_name="測試武器+10",
            starting_price=1000000,
            increment=100000,
            duration_hours=24,
            creator_id=123456789,
            creator_name="測試用戶",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=1000000,
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None

        # 測試結束競標
        success = db.end_auction(auction_id, 123456789)
        assert success is True

        # 驗證競標已結束
        updated_auction = db.get_auction(auction_id, 123456789)
        assert updated_auction is not None
        assert updated_auction.is_active is False

    finally:
        # 清理測試檔案
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_expired_auction_bid():
    """測試對已過期競標出價"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建已過期的測試競標
        past_time = datetime.now() - timedelta(hours=1)
        test_auction = Auction(
            guild_id=123456789,
            item_name="過期測試武器",
            starting_price=1000000,
            increment=100000,
            duration_hours=24,
            creator_id=123456789,
            creator_name="測試用戶",
            end_time=past_time,
            current_price=1000000,
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None

        # 測試對過期競標出價應該失敗
        success = db.place_bid(auction_id, 987654321, "出價者", 1200000, 123456789)
        assert success is False

    finally:
        # 清理測試檔案
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_claim_auction_to_guild():
    """測試將未歸屬拍賣歸屬到伺服器"""
    db = AuctionDatabase()

    # 創建一個未歸屬的拍賣 (guild_id=0)
    from datetime import datetime, timedelta

    end_time = datetime.now() + timedelta(hours=24)

    auction = Auction(
        guild_id=0,  # 未歸屬
        item_name="測試物品",
        starting_price=100.0,
        increment=10.0,
        duration_hours=24,
        creator_id=12345,
        creator_name="測試用戶",
        created_at=datetime.now(),
        end_time=end_time,
        current_price=100.0,
        current_bidder_id=None,
        current_bidder_name=None,
        is_active=True,
        currency_type="楓幣",
    )

    auction_id = db.create_auction(auction)

    # 驗證拍賣存在且未歸屬
    auction = db.get_auction(auction_id, 0)
    assert auction is not None
    assert auction.guild_id == 0

    # 歸屬到伺服器1
    success = db.claim_auction_to_guild(auction_id, 999)
    assert success is True

    # 驗證拍賣已歸屬到伺服器1
    auction = db.get_auction(auction_id, 999)
    assert auction is not None
    assert auction.guild_id == 999

    # 驗證原來的guild_id=0查詢不到
    auction = db.get_auction(auction_id, 0)
    assert auction is None

    # 嘗試再次歸屬同一拍賣到其他伺服器應該失敗
    success = db.claim_auction_to_guild(auction_id, 888)
    assert success is False  # 因為已經不是guild_id=0了


def test_get_currency_display():
    """測試貨幣顯示函數"""
    # 測試已知的貨幣類型
    assert get_currency_display("楓幣") == "楓幣"
    assert get_currency_display("雪花") == "雪花"
    assert get_currency_display("台幣") == "台幣"

    # 測試未知的貨幣類型，應該返回預設值
    assert get_currency_display("不存在的貨幣") == "楓幣"
    assert get_currency_display("") == "楓幣"
    assert get_currency_display(None) == "楓幣"


def test_auction_with_currency_type():
    """測試具有不同貨幣類型的競標模型"""
    end_time = datetime.now() + timedelta(hours=24)

    # 測試雪花貨幣
    auction_snowflake = Auction(
        guild_id=123456789,
        item_name="雪花測試物品",
        starting_price=500.5,
        increment=50.25,
        creator_id=123456,
        creator_name="測試者",
        end_time=end_time,
        current_price=500.5,
        currency_type="雪花",
    )

    assert auction_snowflake.currency_type == "雪花"
    assert auction_snowflake.starting_price == 500.5
    assert auction_snowflake.increment == 50.25

    # 測試台幣貨幣
    auction_twd = Auction(
        guild_id=123456789,
        item_name="台幣測試物品",
        starting_price=1000.0,
        increment=100.0,
        creator_id=123456,
        creator_name="測試者",
        end_time=end_time,
        current_price=1000.0,
        currency_type="台幣",
    )

    assert auction_twd.currency_type == "台幣"

    # 測試預設貨幣（楓幣）
    auction_default = Auction(
        guild_id=123456789,
        item_name="預設測試物品",
        starting_price=200000.0,
        increment=10000.0,
        creator_id=123456,
        creator_name="測試者",
        end_time=end_time,
        current_price=200000.0,
    )

    assert auction_default.currency_type == "楓幣"


def test_float_price_validation():
    """測試浮點數價格驗證"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 測試小數點價格
        test_auction = Auction(
            guild_id=123456789,
            item_name="小數點測試物品",
            starting_price=99.99,
            increment=0.01,
            duration_hours=24,
            creator_id=123456789,
            creator_name="測試用戶",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=99.99,
            currency_type="台幣",
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None

        # 測試小數點出價
        success = db.place_bid(auction_id, 987654321, "出價者", 100.0, 123456789)
        assert success is True

        # 驗證價格精度
        updated_auction = db.get_auction(auction_id, 123456789)
        assert updated_auction is not None
        assert updated_auction.current_price == 100.0
        assert updated_auction.increment == 0.01

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_auction_with_guild_isolation():
    """測試伺服器隔離功能"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建兩個不同伺服器的競標
        guild1_auction = Auction(
            guild_id=111111111,
            item_name="伺服器1物品",
            starting_price=1000000,
            increment=100000,
            duration_hours=24,
            creator_id=123456789,
            creator_name="用戶1",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=1000000,
        )

        guild2_auction = Auction(
            guild_id=222222222,
            item_name="伺服器2物品",
            starting_price=2000000,
            increment=200000,
            duration_hours=24,
            creator_id=987654321,
            creator_name="用戶2",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=2000000,
        )

        auction1_id = db.create_auction(guild1_auction)
        auction2_id = db.create_auction(guild2_auction)

        assert auction1_id is not None
        assert auction2_id is not None

        # 驗證伺服器隔離 - 伺服器1只能看到自己的競標
        guild1_auctions = db.get_active_auctions(111111111)
        assert len(guild1_auctions) == 1
        assert guild1_auctions[0].item_name == "伺服器1物品"

        # 驗證伺服器隔離 - 伺服器2只能看到自己的競標
        guild2_auctions = db.get_active_auctions(222222222)
        assert len(guild2_auctions) == 1
        assert guild2_auctions[0].item_name == "伺服器2物品"

        # 驗證跨伺服器查詢失敗
        auction_cross_check = db.get_auction(auction1_id, 222222222)
        assert auction_cross_check is None

        auction_cross_check2 = db.get_auction(auction2_id, 111111111)
        assert auction_cross_check2 is None

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_bid_with_guild_isolation():
    """測試出價的伺服器隔離功能"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建測試競標
        test_auction = Auction(
            guild_id=123456789,
            item_name="隔離測試物品",
            starting_price=1000000,
            increment=100000,
            duration_hours=24,
            creator_id=123456789,
            creator_name="測試用戶",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=1000000,
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None

        # 正確的伺服器出價應該成功
        success = db.place_bid(auction_id, 987654321, "出價者", 1200000, 123456789)
        assert success is True

        # 錯誤的伺服器出價應該失敗
        success_wrong_guild = db.place_bid(auction_id, 987654321, "出價者", 1300000, 999999999)
        assert success_wrong_guild is False

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_database_migration_guild_id():
    """測試資料庫遷移：添加 guild_id 欄位"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        # 手動創建舊版本的資料庫結構（沒有 guild_id）
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE auctions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_name TEXT NOT NULL,
                    starting_price INTEGER NOT NULL,
                    increment INTEGER NOT NULL,
                    duration_hours INTEGER NOT NULL,
                    creator_id INTEGER NOT NULL,
                    creator_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP NOT NULL,
                    current_price INTEGER NOT NULL,
                    current_bidder_id INTEGER,
                    current_bidder_name TEXT,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            cursor.execute("""
                CREATE TABLE bids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    auction_id INTEGER NOT NULL,
                    bidder_id INTEGER NOT NULL,
                    bidder_name TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (auction_id) REFERENCES auctions (id)
                )
            """)

            # 插入一些測試數據
            cursor.execute(
                """
                INSERT INTO auctions (item_name, starting_price, increment, duration_hours,
                                    creator_id, creator_name, end_time, current_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    "舊版測試物品",
                    1000000,
                    100000,
                    24,
                    123456789,
                    "舊版用戶",
                    (datetime.now() + timedelta(hours=24)).isoformat(),
                    1000000,
                ),
            )

            conn.commit()

        # 初始化 AuctionDatabase 應該觸發遷移
        db = AuctionDatabase(temp_db_path)

        # 驗證遷移後的資料
        auctions = db.get_active_auctions(0)  # 舊數據應該有 guild_id = 0
        assert len(auctions) == 1
        assert auctions[0].item_name == "舊版測試物品"
        assert auctions[0].guild_id == 0

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_database_migration_real_prices():
    """測試資料庫遷移：從 INTEGER 到 REAL 價格欄位"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        # 手動創建舊版本的資料庫結構（INTEGER 價格）
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE auctions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL DEFAULT 0,
                    item_name TEXT NOT NULL,
                    starting_price INTEGER NOT NULL,
                    increment INTEGER NOT NULL,
                    duration_hours INTEGER NOT NULL,
                    creator_id INTEGER NOT NULL,
                    creator_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP NOT NULL,
                    current_price INTEGER NOT NULL,
                    current_bidder_id INTEGER,
                    current_bidder_name TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    currency_type TEXT DEFAULT '楓幣'
                )
            """)

            cursor.execute("""
                CREATE TABLE bids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    auction_id INTEGER NOT NULL,
                    guild_id INTEGER NOT NULL DEFAULT 0,
                    bidder_id INTEGER NOT NULL,
                    bidder_name TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (auction_id) REFERENCES auctions (id)
                )
            """)

            # 插入整數價格數據
            cursor.execute(
                """
                INSERT INTO auctions (guild_id, item_name, starting_price, increment, duration_hours,
                                    creator_id, creator_name, end_time, current_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    123456789,
                    "整數價格測試",
                    1000000,
                    100000,
                    24,
                    123456789,
                    "測試用戶",
                    (datetime.now() + timedelta(hours=24)).isoformat(),
                    1000000,
                ),
            )

            conn.commit()

        # 初始化 AuctionDatabase 應該觸發遷移
        db = AuctionDatabase(temp_db_path)

        # 驗證遷移後可以處理浮點数價格
        test_auction = Auction(
            guild_id=123456789,
            item_name="小數點價格測試",
            starting_price=99.99,
            increment=0.01,
            duration_hours=24,
            creator_id=987654321,
            creator_name="小數點用戶",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=99.99,
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None

        # 測試小數點出價
        success = db.place_bid(auction_id, 111111111, "小數點出價者", 100.50, 123456789)
        assert success is True

        updated_auction = db.get_auction(auction_id, 123456789)
        assert updated_auction is not None
        assert updated_auction.current_price == 100.50

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_auction_expiration_handling():
    """測試競標過期處理"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建一個即將過期的競標
        near_expiry = datetime.now() + timedelta(seconds=1)
        test_auction = Auction(
            guild_id=123456789,
            item_name="即將過期測試",
            starting_price=1000000,
            increment=100000,
            duration_hours=24,
            creator_id=123456789,
            creator_name="測試用戶",
            end_time=near_expiry,
            current_price=1000000,
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None

        # 等待過期
        import time

        time.sleep(2)

        # 測試對過期競標出價應該失敗
        success = db.place_bid(auction_id, 987654321, "出價者", 1200000, 123456789)
        assert success is False

        # 測試結束過期競標
        end_success = db.end_auction(auction_id, 123456789)
        assert end_success is True

        # 驗證競標狀態
        ended_auction = db.get_auction(auction_id, 123456789)
        assert ended_auction is not None
        assert ended_auction.is_active is False

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_bid_validation_edge_cases():
    """測試出價驗證的邊緣情況"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建測試競標
        test_auction = Auction(
            guild_id=123456789,
            item_name="邊緣測試物品",
            starting_price=1000.0,
            increment=100.0,
            duration_hours=24,
            creator_id=123456789,
            creator_name="測試用戶",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=1000.0,
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None

        # 測試精確的最低出價
        min_bid = 1000.0 + 100.0  # current_price + increment
        success = db.place_bid(auction_id, 111111111, "最低出價者", min_bid, 123456789)
        assert success is True

        # 測試略低於最低出價
        updated_auction = db.get_auction(auction_id, 123456789)
        assert updated_auction is not None
        slightly_low = updated_auction.current_price + updated_auction.increment - 0.01
        success_low = db.place_bid(auction_id, 222222222, "出價不足者", slightly_low, 123456789)
        assert success_low is False

        # 測試同一用戶重複出價
        success_repeat = db.place_bid(auction_id, 111111111, "最低出價者", 1300.0, 123456789)
        assert success_repeat is False

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_multiple_auctions_different_currencies():
    """測試不同貨幣類型的多個競標"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建不同貨幣的競標
        currencies = ["楓幣", "雪花", "台幣"]
        auction_ids = []

        for i, currency in enumerate(currencies):
            auction = Auction(
                guild_id=123456789,
                item_name=f"{currency}測試物品{i}",
                starting_price=float(1000 * (i + 1)),
                increment=float(100 * (i + 1)),
                duration_hours=24,
                creator_id=123456789 + i,
                creator_name=f"測試用戶{i}",
                end_time=datetime.now() + timedelta(hours=24),
                current_price=float(1000 * (i + 1)),
                currency_type=currency,
            )

            auction_id = db.create_auction(auction)
            auction_ids.append(auction_id)
            assert auction_id is not None

        # 驗證所有競標都存在且貨幣類型正確
        active_auctions = db.get_active_auctions(123456789)
        assert len(active_auctions) == 3

        for auction in active_auctions:
            assert auction.currency_type in currencies
            # 驗證價格為浮點數
            assert isinstance(auction.starting_price, float)
            assert isinstance(auction.increment, float)
            assert isinstance(auction.current_price, float)

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_claim_auction_invalid_scenarios():
    """測試認領拍賣的無效情況"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建已歸屬的拍賣
        assigned_auction = Auction(
            guild_id=123456789,  # 已歸屬
            item_name="已歸屬拍賣",
            starting_price=100.0,
            increment=10.0,
            duration_hours=24,
            creator_id=12345,
            creator_name="測試用戶",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=100.0,
        )

        auction_id = db.create_auction(assigned_auction)
        assert auction_id is not None

        # 嘗試認領已歸屬的拍賣應該失敗
        success = db.claim_auction_to_guild(auction_id, 999999999)
        assert success is False

        # 嘗試認領不存在的拍賣應該失敗
        success_nonexistent = db.claim_auction_to_guild(99999, 123456789)
        assert success_nonexistent is False

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_bid_history_ordering():
    """測試出價記錄的排序"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = temp_file.name

    try:
        db = AuctionDatabase(temp_db_path)

        # 創建測試競標
        test_auction = Auction(
            guild_id=123456789,
            item_name="排序測試物品",
            starting_price=1000.0,
            increment=100.0,
            duration_hours=24,
            creator_id=123456789,
            creator_name="測試用戶",
            end_time=datetime.now() + timedelta(hours=24),
            current_price=1000.0,
        )

        auction_id = db.create_auction(test_auction)
        assert auction_id is not None

        # 進行多次出價（按順序）
        bid_amounts = [1100.0, 1300.0, 1200.0, 1500.0, 1400.0]
        bidder_names = ["出價者A", "出價者B", "出價者C", "出價者D", "出價者E"]

        for i, (amount, name) in enumerate(zip(bid_amounts, bidder_names, strict=False)):
            # 只有更高的出價才會成功
            current_auction = db.get_auction(auction_id, 123456789)
            if amount > current_auction.current_price:
                success = db.place_bid(auction_id, 111111111 + i, name, amount, 123456789)
                assert success is True

        # 獲取出價記錄
        bids = db.get_auction_bids(auction_id, 123456789)
        assert bids is not None
        assert len(bids) > 0

        # 驗證記錄按金額降序排列
        for i in range(len(bids) - 1):
            assert bids[i].amount >= bids[i + 1].amount

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_auction_database_error_handling():
    """測試資料庫錯誤處理"""
    # 測試無效的資料庫路徑
    invalid_path = "/invalid/path/that/does/not/exist/test.db"

    # 這應該會因為權限不足或路徑無效而失敗
    with pytest.raises((PermissionError, FileNotFoundError)):
        AuctionDatabase(invalid_path)


def test_auction_model_field_validation():
    """測試 Auction 模型欄位驗證"""
    from pydantic import ValidationError

    end_time = datetime.now() + timedelta(hours=24)

    # 測試必填欄位
    with pytest.raises(ValidationError, match="Field required"):
        Auction(
            # 缺少必填的 guild_id
            item_name="測試物品",
            starting_price=100.0,
            increment=10.0,
            creator_id=123456,
            creator_name="測試者",
            end_time=end_time,
            current_price=100.0,
        )

    # 測試有效的模型創建
    valid_auction = Auction(
        guild_id=123456789,
        item_name="有效測試物品",
        starting_price=100.0,
        increment=10.0,
        creator_id=123456,
        creator_name="測試者",
        end_time=end_time,
        current_price=100.0,
    )

    assert valid_auction is not None
    assert valid_auction.guild_id == 123456789


def test_bid_model_field_validation():
    """測試 Bid 模型欄位驗證"""
    from pydantic import ValidationError

    # 測試必填欄位
    with pytest.raises(ValidationError, match="Field required"):
        Bid(
            # 缺少必填的 auction_id
            guild_id=123456789,
            bidder_id=987654321,
            bidder_name="出價者",
            amount=150.0,
        )

    # 測試有效的模型創建
    valid_bid = Bid(
        auction_id=1, guild_id=123456789, bidder_id=987654321, bidder_name="出價者", amount=150.0
    )

    assert valid_bid is not None
    assert valid_bid.auction_id == 1
    assert valid_bid.guild_id == 123456789
