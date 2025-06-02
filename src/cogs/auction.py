import os
from typing import Optional
import sqlite3
from datetime import datetime, timedelta

import logfire
import nextcord
from nextcord import Embed, Locale, Interaction, SelectOption
from pydantic import Field, BaseModel
from nextcord.ui import View, Modal, Button, Select, TextInput
from nextcord.ext import commands


def get_currency_display(currency_type: str) -> str:
    """取得貨幣顯示文字"""
    currency_map = {"楓幣": "楓幣", "雪花": "雪花", "台幣": "台幣"}
    return currency_map.get(currency_type, "楓幣")


# Pydantic 模型
class Auction(BaseModel):
    """競標資料模型"""

    id: Optional[int] = Field(None, description="競標ID")
    item_name: str = Field(..., description="拍賣物品名稱")
    starting_price: float = Field(..., description="起標價格")
    increment: float = Field(..., description="每次加價金額")
    duration_hours: int = Field(default=24, description="競標持續時間 (小時)")
    creator_id: int = Field(..., description="創建者Discord ID")
    creator_name: str = Field(..., description="創建者Discord名稱")
    created_at: datetime = Field(default_factory=datetime.now, description="創建時間")
    end_time: datetime = Field(..., description="結束時間")
    current_price: float = Field(..., description="當前最高價")
    current_bidder_id: Optional[int] = Field(None, description="當前最高出價者ID")
    current_bidder_name: Optional[str] = Field(None, description="當前最高出價者名稱")
    is_active: bool = Field(default=True, description="是否活躍中")
    currency_type: str = Field(default="楓幣", description="貨幣類型 (楓幣、雪花或台幣)")


class Bid(BaseModel):
    """出價記錄模型"""

    id: Optional[int] = Field(None, description="出價ID")
    auction_id: int = Field(..., description="競標ID")
    bidder_id: int = Field(..., description="出價者Discord ID")
    bidder_name: str = Field(..., description="出價者Discord名稱")
    amount: float = Field(..., description="出價金額")
    timestamp: datetime = Field(default_factory=datetime.now, description="出價時間")


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

            # 如果價格欄位還是 INTEGER，進行遷移
            if columns.get("starting_price") == "INTEGER":
                cursor.execute("BEGIN TRANSACTION")
                try:
                    # 創建新的臨時表
                    cursor.execute("""
                        CREATE TABLE auctions_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

                    # 複製數據
                    cursor.execute("""
                        INSERT INTO auctions_new
                        SELECT id, item_name, CAST(starting_price AS REAL), CAST(increment AS REAL),
                               duration_hours, creator_id, creator_name, created_at, end_time,
                               CAST(current_price AS REAL), current_bidder_id, current_bidder_name,
                               is_active,
                               CASE WHEN currency_type IS NULL THEN '楓幣' ELSE currency_type END
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

            if bid_columns.get("amount") == "INTEGER":
                cursor.execute("BEGIN TRANSACTION")
                try:
                    # 創建新的 bids 表
                    cursor.execute("""
                        CREATE TABLE bids_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            auction_id INTEGER NOT NULL,
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
                        SELECT id, auction_id, bidder_id, bidder_name,
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
                    item_name, starting_price, increment, duration_hours,
                    creator_id, creator_name, end_time, current_price, currency_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
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

    def get_auction(self, auction_id: int) -> Optional[Auction]:
        """取得特定競標"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM auctions WHERE id = ?", (auction_id,))
            row = cursor.fetchone()

            if row:
                try:
                    currency_type = row["currency_type"]
                except (KeyError, IndexError):
                    currency_type = "楓幣"  # Default for backward compatibility

                return Auction(
                    id=row["id"],
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

    def get_active_auctions(self) -> list[Auction]:
        """取得所有活躍競標"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM auctions
                WHERE is_active = TRUE AND end_time > datetime('now')
                ORDER BY end_time ASC
            """)

            auctions = []
            for row in cursor.fetchall():
                try:
                    currency_type = row["currency_type"]
                except (KeyError, IndexError):
                    currency_type = "楓幣"  # Default for backward compatibility

                auctions.append(
                    Auction(
                        id=row["id"],
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

    def place_bid(self, auction_id: int, bidder_id: int, bidder_name: str, amount: float) -> bool:
        """出價"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 檢查競標是否存在且活躍
            cursor.execute(
                """
                SELECT current_price, end_time, is_active
                FROM auctions WHERE id = ?
            """,
                (auction_id,),
            )

            auction_data = cursor.fetchone()
            if not auction_data:
                return False

            current_price, end_time, is_active = auction_data

            # 檢查競標是否已結束
            if not is_active or datetime.fromisoformat(end_time) <= datetime.now():
                return False

            # 檢查出價是否足夠
            if amount <= current_price:
                return False

            # 記錄出價
            cursor.execute(
                """
                INSERT INTO bids (auction_id, bidder_id, bidder_name, amount)
                VALUES (?, ?, ?, ?)
            """,
                (auction_id, bidder_id, bidder_name, amount),
            )

            # 更新競標當前價格
            cursor.execute(
                """
                UPDATE auctions
                SET current_price = ?, current_bidder_id = ?, current_bidder_name = ?
                WHERE id = ?
            """,
                (amount, bidder_id, bidder_name, auction_id),
            )

            conn.commit()
            return True

    def get_auction_bids(self, auction_id: int) -> list[Bid]:
        """取得競標的所有出價記錄"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM bids WHERE auction_id = ?
                ORDER BY amount DESC, timestamp DESC
                LIMIT 10
            """,
                (auction_id,),
            )

            bids = []
            for row in cursor.fetchall():
                bids.append(
                    Bid(
                        id=row["id"],
                        auction_id=row["auction_id"],
                        bidder_id=row["bidder_id"],
                        bidder_name=row["bidder_name"],
                        amount=row["amount"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                    )
                )

            return bids

    def end_auction(self, auction_id: int) -> bool:
        """結束競標"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE auctions SET is_active = FALSE WHERE id = ?
            """,
                (auction_id,),
            )
            conn.commit()
            return cursor.rowcount > 0


class AuctionCurrencySelectionView(View):
    """貨幣類型選擇視圖"""

    def __init__(self):
        super().__init__(timeout=300)

    @nextcord.ui.select(
        placeholder="選擇貨幣類型...",
        options=[
            SelectOption(label="楓幣", value="楓幣", emoji="🍁", description="遊戲內楓幣"),
            SelectOption(label="雪花", value="雪花", emoji="❄️", description="雪花貨幣"),
            SelectOption(label="台幣", value="台幣", emoji="💰", description="台灣新台幣"),
        ],
        min_values=1,
        max_values=1,
    )
    async def currency_select(self, select: Select, interaction: Interaction) -> None:
        selected_currency = select.values[0]
        modal = AuctionCreateModal(currency_type=selected_currency)
        await interaction.response.send_modal(modal)


class AuctionCreateModal(Modal):
    """創建競標的模態對話框"""

    def __init__(self, currency_type: str = "楓幣"):
        super().__init__(title="創建拍賣", timeout=300)
        self.selected_currency = currency_type

        self.item_name = TextInput(
            label="物品名稱",
            placeholder="請輸入要拍賣的物品名稱...",
            required=True,
            max_length=100,
        )

        currency_display = get_currency_display(currency_type)
        self.starting_price = TextInput(
            label="起標價格",
            placeholder=f"請輸入起標價格 ({currency_display})，支援小數點...",
            required=True,
            max_length=20,
        )

        self.increment = TextInput(
            label="加價金額",
            placeholder=f"請輸入每次最少加價金額 ({currency_display})，支援小數點...",
            required=True,
            max_length=20,
        )

        self.duration = TextInput(
            label="拍賣時長 (小時)",
            placeholder="請輸入拍賣持續時間 (1-168小時)...",
            required=True,
            max_length=3,
            default_value="24",
        )

        self.add_item(self.item_name)
        self.add_item(self.starting_price)
        self.add_item(self.increment)
        self.add_item(self.duration)

    async def callback(self, interaction: Interaction) -> None:
        try:
            starting_price = float(self.starting_price.value)
            increment = float(self.increment.value)
            duration_hours = int(self.duration.value)
            currency_type = self.selected_currency

            if starting_price <= 0:
                await interaction.response.send_message("❌ 起標價格必須大於 0!", ephemeral=True)
                return

            if increment <= 0:
                await interaction.response.send_message("❌ 加價金額必須大於 0!", ephemeral=True)
                return

            if not (1 <= duration_hours <= 168):
                await interaction.response.send_message(
                    "❌ 拍賣時長必須在 1-168 小時之間!", ephemeral=True
                )
                return

            # 創建競標
            auction = Auction(
                item_name=self.item_name.value,
                starting_price=starting_price,
                increment=increment,
                duration_hours=duration_hours,
                creator_id=interaction.user.id,
                creator_name=interaction.user.display_name,
                end_time=datetime.now() + timedelta(hours=duration_hours),
                current_price=starting_price,
                currency_type=currency_type,
            )

            db = AuctionDatabase()
            auction_id = db.create_auction(auction)
            auction.id = auction_id

            # 創建競標顯示
            embed = self._create_auction_embed(auction)
            view = AuctionView(auction)

            await interaction.response.send_message(
                f"🎉 拍賣已成功創建!拍賣編號：#{auction_id}", embed=embed, view=view
            )

        except ValueError:
            await interaction.response.send_message("❌ 請輸入有效的數字格式!", ephemeral=True)
        except Exception as e:
            logfire.error(f"創建拍賣時發生錯誤: {e}")
            await interaction.response.send_message(
                "❌ 創建拍賣時發生錯誤，請稍後再試!", ephemeral=True
            )

    def _create_auction_embed(self, auction: Auction) -> Embed:
        """創建競標 Embed"""
        embed = Embed(
            title=f"🏺 {auction.item_name}", description=f"拍賣編號：#{auction.id}", color=0xFFD700
        )

        currency = get_currency_display(auction.currency_type)
        embed.add_field(
            name="💰 當前價格", value=f"{auction.current_price:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="📈 加價金額", value=f"{auction.increment:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="👤 當前領先", value=auction.current_bidder_name or "暫無出價", inline=True
        )

        embed.add_field(name="🏁 拍賣發起人", value=auction.creator_name, inline=True)

        remaining_time = auction.end_time - datetime.now()
        hours, remainder = divmod(int(remaining_time.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)

        embed.add_field(name="⏰ 剩餘時間", value=f"{hours}時{minutes}分", inline=True)

        embed.add_field(
            name="📅 結束時間", value=auction.end_time.strftime("%m/%d %H:%M"), inline=True
        )

        embed.set_footer(text="點擊下方按鈕參與競標!")
        return embed


class AuctionBidModal(Modal):
    """出價模態對話框"""

    def __init__(self, auction: Auction):
        super().__init__(title=f"競標 - {auction.item_name}", timeout=300)
        self.auction = auction

        min_bid = auction.current_price + auction.increment
        currency = get_currency_display(auction.currency_type)

        self.bid_amount = TextInput(
            label="出價金額",
            placeholder=f"最低出價：{min_bid:,.2f} {currency} (支援小數點)",
            required=True,
            max_length=20,
        )

        self.add_item(self.bid_amount)

    async def callback(self, interaction: Interaction) -> None:
        try:
            bid_amount = float(self.bid_amount.value)
            min_bid = self.auction.current_price + self.auction.increment
            currency = get_currency_display(self.auction.currency_type)

            if bid_amount < min_bid:
                await interaction.response.send_message(
                    f"❌ 出價金額必須至少為 {min_bid:,.2f} {currency}!", ephemeral=True
                )
                return

            # 檢查用戶是否為當前最高出價者
            if self.auction.current_bidder_id == interaction.user.id:
                await interaction.response.send_message(
                    "❌ 你已經是當前最高出價者了!", ephemeral=True
                )
                return

            # 檢查用戶是否為拍賣創建者
            if self.auction.creator_id == interaction.user.id:
                await interaction.response.send_message(
                    "❌ 拍賣創建者不能參與自己的拍賣!", ephemeral=True
                )
                return

            # 處理出價
            db = AuctionDatabase()
            if self.auction.id is None:
                await interaction.response.send_message("❌ 拍賣ID無效!", ephemeral=True)
                return

            success = db.place_bid(
                self.auction.id, interaction.user.id, interaction.user.display_name, bid_amount
            )

            if success:
                # 更新競標資訊
                updated_auction = db.get_auction(self.auction.id)
                if updated_auction:
                    embed = self._create_auction_embed(updated_auction)
                    view = AuctionView(updated_auction)

                    await interaction.response.edit_message(
                        content=f"🎉 出價成功! {interaction.user.mention} 出價 {bid_amount:,.2f} {currency}",
                        embed=embed,
                        view=view,
                    )
                else:
                    await interaction.response.send_message(
                        "❌ 出價失敗，請稍後再試!", ephemeral=True
                    )
            else:
                await interaction.response.send_message(
                    "❌ 出價失敗，可能有其他人同時出價了!", ephemeral=True
                )

        except ValueError:
            await interaction.response.send_message("❌ 請輸入有效的數字格式!", ephemeral=True)
        except Exception as e:
            logfire.error(f"出價時發生錯誤: {e}")
            await interaction.response.send_message(
                "❌ 出價時發生錯誤，請稍後再試!", ephemeral=True
            )

    def _create_auction_embed(self, auction: Auction) -> Embed:
        """創建競標 Embed"""
        embed = Embed(
            title=f"🏺 {auction.item_name}", description=f"拍賣編號：#{auction.id}", color=0xFFD700
        )

        currency = get_currency_display(auction.currency_type)
        embed.add_field(
            name="💰 當前價格", value=f"{auction.current_price:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="📈 加價金額", value=f"{auction.increment:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="👤 當前領先", value=auction.current_bidder_name or "暫無出價", inline=True
        )

        embed.add_field(name="🏁 拍賣發起人", value=auction.creator_name, inline=True)

        remaining_time = auction.end_time - datetime.now()
        if remaining_time.total_seconds() > 0:
            hours, remainder = divmod(int(remaining_time.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)

            embed.add_field(name="⏰ 剩餘時間", value=f"{hours}時{minutes}分", inline=True)
        else:
            embed.add_field(name="⏰ 狀態", value="已結束", inline=True)

        embed.add_field(
            name="📅 結束時間", value=auction.end_time.strftime("%m/%d %H:%M"), inline=True
        )

        embed.set_footer(text="點擊下方按鈕參與競標!")
        return embed


class AuctionView(View):
    """競標互動視圖"""

    def __init__(self, auction: Auction):
        super().__init__(timeout=None)  # 不設置超時
        self.auction = auction

    @nextcord.ui.button(label="出價", style=nextcord.ButtonStyle.green, emoji="💰")
    async def bid_button(self, button: Button, interaction: Interaction) -> None:
        # 檢查競標是否已結束
        if datetime.now() >= self.auction.end_time:
            await interaction.response.send_message("❌ 此拍賣已結束!", ephemeral=True)
            return

        modal = AuctionBidModal(self.auction)
        await interaction.response.send_modal(modal)

    @nextcord.ui.button(label="查看記錄", style=nextcord.ButtonStyle.gray, emoji="📊")
    async def history_button(self, button: Button, interaction: Interaction) -> None:
        if self.auction.id is None:
            await interaction.response.send_message("❌ 拍賣ID無效!", ephemeral=True)
            return

        db = AuctionDatabase()
        bids = db.get_auction_bids(self.auction.id)

        if not bids:
            await interaction.response.send_message("📭 此拍賣還沒有出價記錄。", ephemeral=True)
            return

        embed = Embed(
            title=f"📊 拍賣記錄 - {self.auction.item_name}",
            description=f"拍賣編號：#{self.auction.id}",
            color=0x00AAFF,
        )

        currency = get_currency_display(self.auction.currency_type)
        bid_list = []
        for i, bid in enumerate(bids, 1):
            time_str = bid.timestamp.strftime("%m/%d %H:%M")
            bid_list.append(
                f"{i}. **{bid.bidder_name}** - {bid.amount:,.2f} {currency} ({time_str})"
            )

        embed.add_field(
            name="💰 出價記錄 (前10筆)",
            value="\n".join(bid_list) if bid_list else "暫無記錄",
            inline=False,
        )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @nextcord.ui.button(label="刷新", style=nextcord.ButtonStyle.gray, emoji="🔄")
    async def refresh_button(self, button: Button, interaction: Interaction) -> None:
        if self.auction.id is None:
            await interaction.response.send_message("❌ 拍賣ID無效!", ephemeral=True)
            return

        db = AuctionDatabase()
        updated_auction = db.get_auction(self.auction.id)

        if updated_auction:
            self.auction = updated_auction
            embed = self._create_auction_embed(updated_auction)
            view = AuctionView(updated_auction)

            await interaction.response.edit_message(embed=embed, view=view)
        else:
            await interaction.response.send_message("❌ 無法載入拍賣資訊!", ephemeral=True)

    def _create_auction_embed(self, auction: Auction) -> Embed:
        """創建競標 Embed"""
        embed = Embed(
            title=f"🏺 {auction.item_name}", description=f"拍賣編號：#{auction.id}", color=0xFFD700
        )

        currency = get_currency_display(auction.currency_type)
        embed.add_field(
            name="💰 當前價格", value=f"{auction.current_price:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="📈 加價金額", value=f"{auction.increment:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="👤 當前領先", value=auction.current_bidder_name or "暫無出價", inline=True
        )

        embed.add_field(name="🏁 拍賣發起人", value=auction.creator_name, inline=True)

        remaining_time = auction.end_time - datetime.now()
        if remaining_time.total_seconds() > 0:
            hours, remainder = divmod(int(remaining_time.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)

            embed.add_field(name="⏰ 剩餘時間", value=f"{hours}時{minutes}分", inline=True)
        else:
            embed.add_field(name="⏰ 狀態", value="已結束", inline=True)

        embed.add_field(
            name="📅 結束時間", value=auction.end_time.strftime("%m/%d %H:%M"), inline=True
        )

        embed.set_footer(text="點擊下方按鈕參與競標!")
        return embed


class AuctionListView(View):
    """拍賣列表視圖"""

    def __init__(self, auctions: list[Auction]):
        super().__init__(timeout=300)
        self.auctions = auctions

        if auctions:
            options = []
            for auction in auctions:
                remaining_time = auction.end_time - datetime.now()
                hours = int(remaining_time.total_seconds() // 3600)
                currency = get_currency_display(auction.currency_type)

                description = f"當前價格: {auction.current_price:,.2f} {currency} | 剩餘: {hours}h"
                options.append(
                    SelectOption(
                        label=auction.item_name, description=description, value=str(auction.id)
                    )
                )

            self.auction_select.options = options
        else:
            self.auction_select.disabled = True

    @nextcord.ui.select(placeholder="選擇要查看的拍賣...", min_values=1, max_values=1)
    async def auction_select(self, select: Select, interaction: Interaction) -> None:
        auction_id = int(select.values[0])

        db = AuctionDatabase()
        auction = db.get_auction(auction_id)

        if auction:
            embed = self._create_auction_embed(auction)
            view = AuctionView(auction)

            await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        else:
            await interaction.response.send_message("❌ 找不到該拍賣!", ephemeral=True)

    def _create_auction_embed(self, auction: Auction) -> Embed:
        """創建競標 Embed"""
        embed = Embed(
            title=f"🏺 {auction.item_name}", description=f"拍賣編號：#{auction.id}", color=0xFFD700
        )

        currency = get_currency_display(auction.currency_type)
        embed.add_field(
            name="💰 當前價格", value=f"{auction.current_price:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="📈 加價金額", value=f"{auction.increment:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="👤 當前領先", value=auction.current_bidder_name or "暫無出價", inline=True
        )

        embed.add_field(name="🏁 拍賣發起人", value=auction.creator_name, inline=True)

        remaining_time = auction.end_time - datetime.now()
        if remaining_time.total_seconds() > 0:
            hours, remainder = divmod(int(remaining_time.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)

            embed.add_field(name="⏰ 剩餘時間", value=f"{hours}時{minutes}分", inline=True)
        else:
            embed.add_field(name="⏰ 狀態", value="已結束", inline=True)

        embed.add_field(
            name="📅 結束時間", value=auction.end_time.strftime("%m/%d %H:%M"), inline=True
        )

        embed.set_footer(text="點擊下方按鈕參與競標!")
        return embed


class AuctionCogs(commands.Cog):
    """拍賣系統功能"""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # 初始化競標資料庫
        self.auction_db = AuctionDatabase()

    @nextcord.slash_command(
        name="auction_create",
        description="Create a new item auction",
        name_localizations={
            Locale.zh_TW: "創建拍賣",
            Locale.zh_CN: "创建拍卖",
            Locale.ja: "オークション作成",
        },
        description_localizations={
            Locale.zh_TW: "創建新的物品拍賣",
            Locale.zh_CN: "创建新的物品拍卖",
            Locale.ja: "新しいアイテムオークションを作成",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def auction_create(self, interaction: Interaction) -> None:
        """創建新拍賣"""
        embed = Embed(
            title="🏺 創建拍賣", description="請先選擇拍賣使用的貨幣類型：", color=0xFFD700
        )
        view = AuctionCurrencySelectionView()
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    @nextcord.slash_command(
        name="auction_list",
        description="View active auctions",
        name_localizations={
            Locale.zh_TW: "拍賣列表",
            Locale.zh_CN: "拍卖列表",
            Locale.ja: "オークションリスト",
        },
        description_localizations={
            Locale.zh_TW: "查看進行中的拍賣列表",
            Locale.zh_CN: "查看进行中的拍卖列表",
            Locale.ja: "進行中のオークション一覧を表示",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def auction_list(self, interaction: Interaction) -> None:
        """查看拍賣列表"""
        await interaction.response.defer()

        auctions = self.auction_db.get_active_auctions()

        if not auctions:
            embed = Embed(
                title="📋 拍賣列表", description="目前沒有進行中的拍賣。", color=0xFFAA00
            )
            embed.add_field(
                name="💡 提示", value="使用 `/auction_create` 來創建新的拍賣!", inline=False
            )
            await interaction.followup.send(embed=embed)
            return

        embed = Embed(
            title="📋 進行中的拍賣",
            description=f"共有 {len(auctions)} 個拍賣進行中",
            color=0x00AAFF,
        )

        # 顯示前5個拍賣的摘要
        auction_summary = []
        for i, auction in enumerate(auctions, 1):
            remaining_time = auction.end_time - datetime.now()
            hours = int(remaining_time.total_seconds() // 3600)
            currency = get_currency_display(auction.currency_type)

            summary = (
                f"{i}. **{auction.item_name}** (#{auction.id})\n"
                f"   💰 {auction.current_price:,.2f} {currency} | ⏰ {hours}h 剩餘"
            )
            auction_summary.append(summary)

        embed.add_field(name="🏺 拍賣預覽", value="\n\n".join(auction_summary), inline=False)

        if len(auctions) > 5:
            embed.add_field(name="📝 說明", value="請使用下方選單查看詳細資訊。", inline=False)

        view = AuctionListView(auctions)
        await interaction.followup.send(embed=embed, view=view)

    @nextcord.slash_command(
        name="auction_info",
        description="View detailed information about a specific auction",
        name_localizations={
            Locale.zh_TW: "拍賣資訊",
            Locale.zh_CN: "拍卖资讯",
            Locale.ja: "オークション情報",
        },
        description_localizations={
            Locale.zh_TW: "查看特定拍賣的詳細資訊",
            Locale.zh_CN: "查看特定拍卖的详细资讯",
            Locale.ja: "特定のオークションの詳細情報を表示",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def auction_info(
        self,
        interaction: Interaction,
        auction_id: int = nextcord.SlashOption(
            name="auction_id",
            description="Auction ID to view",
            name_localizations={
                Locale.zh_TW: "拍賣編號",
                Locale.zh_CN: "拍卖编号",
                Locale.ja: "オークション番号",
            },
            description_localizations={
                Locale.zh_TW: "要查看的拍賣編號",
                Locale.zh_CN: "要查看的拍卖编号",
                Locale.ja: "表示するオークション番号",
            },
            required=True,
        ),
    ) -> None:
        """查看特定拍賣資訊"""
        await interaction.response.defer()

        auction = self.auction_db.get_auction(auction_id)

        if not auction:
            embed = Embed(
                title="❌ 錯誤", description=f"找不到編號 #{auction_id} 的拍賣。", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        embed = self._create_auction_embed(auction)
        view = AuctionView(auction)

        await interaction.followup.send(embed=embed, view=view)

    @nextcord.slash_command(
        name="auction_my",
        description="View your auctions and bids",
        name_localizations={
            Locale.zh_TW: "我的拍賣",
            Locale.zh_CN: "我的拍卖",
            Locale.ja: "マイオークション",
        },
        description_localizations={
            Locale.zh_TW: "查看你的拍賣和出價記錄",
            Locale.zh_CN: "查看你的拍卖和出价记录",
            Locale.ja: "あなたのオークションと入札記録を表示",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def auction_my(self, interaction: Interaction) -> None:
        """查看個人拍賣記錄"""
        await interaction.response.defer()

        # 取得用戶創建的拍賣
        user_auctions = []
        active_auctions = self.auction_db.get_active_auctions()

        for auction in active_auctions:
            if auction.creator_id == interaction.user.id:
                user_auctions.append(auction)

        # 取得用戶參與的拍賣 (當前最高出價者)
        leading_auctions = []
        for auction in active_auctions:
            if auction.current_bidder_id == interaction.user.id:
                leading_auctions.append(auction)

        embed = Embed(title=f"📋 {interaction.user.mention} 的拍賣記錄", color=0x9966FF)

        if user_auctions:
            auction_list = []
            for auction in user_auctions:
                remaining_time = auction.end_time - datetime.now()
                hours = int(remaining_time.total_seconds() // 3600)
                currency = get_currency_display(auction.currency_type)

                auction_list.append(
                    f"#{auction.id} **{auction.item_name}** - {auction.current_price:,.2f} {currency} ({hours}h)"
                )

            embed.add_field(name="🏺 我創建的拍賣", value="\n".join(auction_list), inline=False)

        if leading_auctions:
            leading_list = []
            for auction in leading_auctions:
                remaining_time = auction.end_time - datetime.now()
                hours = int(remaining_time.total_seconds() // 3600)
                currency = get_currency_display(auction.currency_type)

                leading_list.append(
                    f"#{auction.id} **{auction.item_name}** - {auction.current_price:,.2f} {currency} ({hours}h)"
                )

            embed.add_field(name="👑 我領先的拍賣", value="\n".join(leading_list), inline=False)

        if not user_auctions and not leading_auctions:
            embed.description = "你還沒有創建或參與任何拍賣。"
            embed.add_field(
                name="💡 開始使用",
                value="使用 `/auction_create` 創建拍賣\n使用 `/auction_list` 查看並參與拍賣",
                inline=False,
            )

        await interaction.followup.send(embed=embed)

    def _create_auction_embed(self, auction: Auction) -> Embed:
        """創建競標 Embed"""
        embed = Embed(
            title=f"🏺 {auction.item_name}", description=f"拍賣編號：#{auction.id}", color=0xFFD700
        )

        currency = get_currency_display(auction.currency_type)
        embed.add_field(
            name="💰 當前價格", value=f"{auction.current_price:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="📈 加價金額", value=f"{auction.increment:,.2f} {currency}", inline=True
        )

        embed.add_field(
            name="👤 當前領先", value=auction.current_bidder_name or "暫無出價", inline=True
        )

        embed.add_field(name="🏁 拍賣發起人", value=auction.creator_name, inline=True)

        remaining_time = auction.end_time - datetime.now()
        if remaining_time.total_seconds() > 0:
            hours, remainder = divmod(int(remaining_time.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)

            embed.add_field(name="⏰ 剩餘時間", value=f"{hours}時{minutes}分", inline=True)
        else:
            embed.add_field(name="⏰ 狀態", value="已結束", inline=True)

        embed.add_field(
            name="📅 結束時間", value=auction.end_time.strftime("%m/%d %H:%M"), inline=True
        )

        embed.set_footer(text="點擊下方按鈕參與競標!")
        return embed


def setup(bot: commands.Bot) -> None:
    bot.add_cog(AuctionCogs(bot))
