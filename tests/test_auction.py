import os
from datetime import datetime, timedelta
import tempfile

from src.cogs.auction import Bid, Auction, AuctionDatabase


def test_auction_model_creation():
    """測試 Auction 模型創建"""
    end_time = datetime.now() + timedelta(hours=24)

    auction = Auction(
        item_name="測試物品",
        starting_price=100000,
        increment=10000,
        creator_id=123456,
        creator_name="測試者",
        end_time=end_time,
        current_price=100000,
    )

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
    bid = Bid(auction_id=1, bidder_id=987654321, bidder_name="出價者", amount=150000)

    assert bid.auction_id == 1
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
        retrieved_auction = db.get_auction(auction_id)
        assert retrieved_auction is not None
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
        success = db.place_bid(auction_id, 987654321, "出價者", 1200000)
        assert success is True

        # 驗證價格更新
        updated_auction = db.get_auction(auction_id)
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
        success = db.place_bid(auction_id, 987654321, "出價者", 1000000)
        assert success is False

        # 測試出價金額不足（小於當前價格）
        success = db.place_bid(auction_id, 987654321, "出價者", 900000)
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
        db.place_bid(auction_id, 111111111, "出價者1", 1200000)
        db.place_bid(auction_id, 222222222, "出價者2", 1400000)
        db.place_bid(auction_id, 333333333, "出價者3", 1600000)

        # 測試獲取出價記錄
        bids = db.get_auction_bids(auction_id)
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
        active_auctions = db.get_active_auctions()
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
        success = db.end_auction(auction_id)
        assert success is True

        # 驗證競標已結束
        updated_auction = db.get_auction(auction_id)
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
        success = db.place_bid(auction_id, 987654321, "出價者", 1200000)
        assert success is False

    finally:
        # 清理測試檔案
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
