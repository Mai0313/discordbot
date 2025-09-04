"""拍賣系統資料庫操作"""

import os
import sqlite3
from datetime import datetime

from .models import Auction, Bid


class AuctionDatabase:
    """競標資料庫操作類"""

    def __init__(self, db_path: str = "data/auctions.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """初始化資料庫"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 創建競標表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS auctions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    item_name TEXT NOT NULL,
                    starting_price REAL NOT NULL,
                    increment REAL NOT NULL,
                    duration_hours INTEGER NOT NULL,
                    creator_id INTEGER NOT NULL,
                    creator_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP NOT NULL,
                    current_price REAL NOT NULL,
                    current_bidder_id INTEGER,
                    current_bidder_name TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    currency_type TEXT DEFAULT '楓幣'
                )
            """)

            # 創建出價記錄表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    auction_id INTEGER NOT NULL,
                    guild_id INTEGER NOT NULL,
                    bidder_id INTEGER NOT NULL,
                    bidder_name TEXT NOT NULL,
                    amount REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (auction_id) REFERENCES auctions (id)
                )
            """)

            # 檢查並更新現有資料庫結構為 REAL 類型 (遷移支援)
            cursor.execute("PRAGMA table_info(auctions)")
            columns = {col[1]: col[2] for col in cursor.fetchall()}

            # 檢查是否需要添加 guild_id 欄位
            if "guild_id" not in columns:
                cursor.execute("ALTER TABLE auctions ADD COLUMN guild_id INTEGER DEFAULT 0")
                conn.commit()

            # 檢查是否需要添加 currency_type 欄位
            if "currency_type" not in columns:
                cursor.execute("ALTER TABLE auctions ADD COLUMN currency_type TEXT DEFAULT '楓幣'")
                conn.commit()

            # 如果價格欄位還是 INTEGER，進行遷移
            if columns.get("starting_price") == "INTEGER":
                cursor.execute("BEGIN TRANSACTION")
                try:
                    # 重新獲取更新後的欄位信息
                    cursor.execute("PRAGMA table_info(auctions)")
                    updated_columns = {col[1]: col[2] for col in cursor.fetchall()}

                    # 創建新的臨時表
                    cursor.execute("""
                        CREATE TABLE auctions_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            guild_id INTEGER NOT NULL DEFAULT 0,
                            item_name TEXT NOT NULL,
                            starting_price REAL NOT NULL,
                            increment REAL NOT NULL,
                            duration_hours INTEGER NOT NULL,
                            creator_id INTEGER NOT NULL,
                            creator_name TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            end_time TIMESTAMP NOT NULL,
                            current_price REAL NOT NULL,
                            current_bidder_id INTEGER,
                            current_bidder_name TEXT,
                            is_active BOOLEAN DEFAULT TRUE,
                            currency_type TEXT DEFAULT '楓幣'
                        )
                    """)

                    # 複製數據，考慮到 currency_type 可能不存在的情況
                    if "currency_type" in updated_columns:
                        cursor.execute("""
                            INSERT INTO auctions_new
                            SELECT id, COALESCE(guild_id, 0), item_name, CAST(starting_price AS REAL), CAST(increment AS REAL),
                                   duration_hours, creator_id, creator_name, created_at, end_time,
                                   CAST(current_price AS REAL), current_bidder_id, current_bidder_name,
                                   is_active,
                                   COALESCE(currency_type, '楓幣')
                            FROM auctions
                        """)
                    else:
                        cursor.execute("""
                            INSERT INTO auctions_new (id, guild_id, item_name, starting_price, increment,
                                                     duration_hours, creator_id, creator_name, created_at, end_time,
                                                     current_price, current_bidder_id, current_bidder_name,
                                                     is_active, currency_type)
                            SELECT id, COALESCE(guild_id, 0), item_name, CAST(starting_price AS REAL), CAST(increment AS REAL),
                                   duration_hours, creator_id, creator_name, created_at, end_time,
                                   CAST(current_price AS REAL), current_bidder_id, current_bidder_name,
                                   is_active, '楓幣'
                            FROM auctions
                        """)

                    # 刪除舊表並重命名新表
                    cursor.execute("DROP TABLE auctions")
                    cursor.execute("ALTER TABLE auctions_new RENAME TO auctions")
                    cursor.execute("COMMIT")
                except Exception:
                    cursor.execute("ROLLBACK")
                    raise

            # 檢查並更新 bids 表
            cursor.execute("PRAGMA table_info(bids)")
            bid_columns = {col[1]: col[2] for col in cursor.fetchall()}

            # 檢查是否需要添加 guild_id 欄位
            if "guild_id" not in bid_columns:
                cursor.execute("ALTER TABLE bids ADD COLUMN guild_id INTEGER DEFAULT 0")
                conn.commit()

            if bid_columns.get("amount") == "INTEGER":
                cursor.execute("BEGIN TRANSACTION")
                try:
                    # 創建新的 bids 表
                    cursor.execute("""
                        CREATE TABLE bids_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            auction_id INTEGER NOT NULL,
                            guild_id INTEGER NOT NULL DEFAULT 0,
                            bidder_id INTEGER NOT NULL,
                            bidder_name TEXT NOT NULL,
                            amount REAL NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (auction_id) REFERENCES auctions (id)
                        )
                    """)

                    # 複製數據
                    cursor.execute("""
                        INSERT INTO bids_new
                        SELECT id, auction_id, COALESCE(guild_id, 0), bidder_id, bidder_name,
                               CAST(amount AS REAL), timestamp
                        FROM bids
                    """)

                    # 刪除舊表並重命名新表
                    cursor.execute("DROP TABLE bids")
                    cursor.execute("ALTER TABLE bids_new RENAME TO bids")
                    cursor.execute("COMMIT")
                except Exception:
                    cursor.execute("ROLLBACK")
                    raise

            conn.commit()

    def create_auction(self, auction: Auction) -> int:
        """創建新競標"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO auctions (
                    guild_id, item_name, starting_price, increment, duration_hours,
                    creator_id, creator_name, end_time, current_price, currency_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    auction.guild_id,
                    auction.item_name,
                    auction.starting_price,
                    auction.increment,
                    auction.duration_hours,
                    auction.creator_id,
                    auction.creator_name,
                    auction.end_time,
                    auction.current_price,
                    auction.currency_type,
                ),
            )
            conn.commit()
            result = cursor.lastrowid
            return result if result is not None else 0

    def get_auction(self, auction_id: int, guild_id: int) -> Auction | None:
        """取得特定競標"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM auctions WHERE id = ? AND guild_id = ?", (auction_id, guild_id)
            )

            row = cursor.fetchone()

            if row:
                try:
                    currency_type = row["currency_type"]
                except (KeyError, IndexError):
                    currency_type = "楓幣"  # Default for backward compatibility

                try:
                    guild_id_from_row = row["guild_id"]
                except (KeyError, IndexError):
                    guild_id_from_row = 0  # Default for backward compatibility

                return Auction(
                    id=row["id"],
                    guild_id=guild_id_from_row,
                    item_name=row["item_name"],
                    starting_price=row["starting_price"],
                    increment=row["increment"],
                    duration_hours=row["duration_hours"],
                    creator_id=row["creator_id"],
                    creator_name=row["creator_name"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    end_time=datetime.fromisoformat(row["end_time"]),
                    current_price=row["current_price"],
                    current_bidder_id=row["current_bidder_id"],
                    current_bidder_name=row["current_bidder_name"],
                    is_active=bool(row["is_active"]),
                    currency_type=currency_type,
                )
            return None

    def get_active_auctions(self, guild_id: int) -> list[Auction]:
        """取得特定伺服器的所有活躍競標"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM auctions
                WHERE guild_id = ? AND is_active = TRUE AND end_time > datetime('now')
                ORDER BY end_time ASC
            """,
                (guild_id,),
            )

            auctions = []
            for row in cursor.fetchall():
                try:
                    currency_type = row["currency_type"]
                except (KeyError, IndexError):
                    currency_type = "楓幣"  # Default for backward compatibility

                try:
                    guild_id_from_row = row["guild_id"]
                except (KeyError, IndexError):
                    guild_id_from_row = guild_id  # Use provided guild_id as fallback

                auctions.append(
                    Auction(
                        id=row["id"],
                        guild_id=guild_id_from_row,
                        item_name=row["item_name"],
                        starting_price=row["starting_price"],
                        increment=row["increment"],
                        duration_hours=row["duration_hours"],
                        creator_id=row["creator_id"],
                        creator_name=row["creator_name"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        end_time=datetime.fromisoformat(row["end_time"]),
                        current_price=row["current_price"],
                        current_bidder_id=row["current_bidder_id"],
                        current_bidder_name=row["current_bidder_name"],
                        is_active=bool(row["is_active"]),
                        currency_type=currency_type,
                    )
                )

            return auctions

    def place_bid(
        self, auction_id: int, bidder_id: int, bidder_name: str, amount: float, guild_id: int
    ) -> bool:
        """出價"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 檢查競標是否存在且活躍
            cursor.execute(
                """
                SELECT current_price, increment, end_time, is_active, current_bidder_id, creator_id
                FROM auctions WHERE id = ? AND guild_id = ?
            """,
                (auction_id, guild_id),
            )

            auction_data = cursor.fetchone()
            if not auction_data:
                return False

            current_price, increment, end_time, is_active, current_bidder_id, creator_id = (
                auction_data
            )

            # 檢查競標是否已結束
            if not is_active or datetime.fromisoformat(end_time) <= datetime.now():
                return False

            # 檢查用戶是否為拍賣創建者
            if bidder_id == creator_id:
                return False

            # 檢查用戶是否為當前最高出價者
            if current_bidder_id is not None and bidder_id == current_bidder_id:
                return False

            # 檢查出價是否足夠（必須至少為當前價格 + 加價金額）
            min_bid = current_price + increment
            if amount < min_bid:
                return False

            # 記錄出價
            cursor.execute(
                """
                INSERT INTO bids (auction_id, guild_id, bidder_id, bidder_name, amount)
                VALUES (?, ?, ?, ?, ?)
            """,
                (auction_id, guild_id, bidder_id, bidder_name, amount),
            )

            # 更新競標當前價格
            cursor.execute(
                """
                UPDATE auctions
                SET current_price = ?, current_bidder_id = ?, current_bidder_name = ?
                WHERE id = ? AND guild_id = ?
            """,
                (amount, bidder_id, bidder_name, auction_id, guild_id),
            )

            conn.commit()
            return True

    def get_auction_bids(self, auction_id: int, guild_id: int) -> list[Bid]:
        """取得競標的所有出價記錄"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM bids WHERE auction_id = ? AND guild_id = ?
                ORDER BY amount DESC, timestamp DESC
                LIMIT 10
            """,
                (auction_id, guild_id),
            )

            bids = []
            for row in cursor.fetchall():
                try:
                    guild_id_from_row = row["guild_id"]
                except (KeyError, IndexError):
                    guild_id_from_row = guild_id  # Use provided guild_id as fallback

                bids.append(
                    Bid(
                        id=row["id"],
                        auction_id=row["auction_id"],
                        guild_id=guild_id_from_row,
                        bidder_id=row["bidder_id"],
                        bidder_name=row["bidder_name"],
                        amount=row["amount"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                    )
                )

            return bids

    def end_auction(self, auction_id: int, guild_id: int) -> bool:
        """結束競標"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE auctions SET is_active = FALSE WHERE id = ? AND guild_id = ?
            """,
                (auction_id, guild_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def claim_auction_to_guild(self, auction_id: int, guild_id: int) -> bool:
        """將未歸屬的拍賣 (guild_id=0) 歸屬到指定伺服器"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 只更新 guild_id=0 的拍賣
            cursor.execute(
                """
                UPDATE auctions SET guild_id = ? WHERE id = ? AND guild_id = 0
            """,
                (guild_id, auction_id),
            )

            # 捕獲拍賣更新的行數
            auction_updated = cursor.rowcount > 0

            # 同時更新相關的出價記錄
            if auction_updated:
                cursor.execute(
                    """
                    UPDATE bids SET guild_id = ? WHERE auction_id = ? AND guild_id = 0
                """,
                    (guild_id, auction_id),
                )

            conn.commit()
            return auction_updated