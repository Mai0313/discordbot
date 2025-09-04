"""拍賣系統資料模型"""

from datetime import datetime
from pydantic import Field, BaseModel


class Auction(BaseModel):
    """競標資料模型"""

    id: int | None = Field(None, description="競標ID")
    guild_id: int = Field(..., description="伺服器ID")
    item_name: str = Field(..., description="拍賣物品名稱")
    starting_price: float = Field(..., description="起標價格")
    increment: float = Field(..., description="每次加價金額")
    duration_hours: int = Field(default=24, description="競標持續時間 (小時)")
    creator_id: int = Field(..., description="創建者Discord ID")
    creator_name: str = Field(..., description="創建者Discord名稱")
    created_at: datetime = Field(default_factory=datetime.now, description="創建時間")
    end_time: datetime = Field(..., description="結束時間")
    current_price: float = Field(..., description="當前最高價")
    current_bidder_id: int | None = Field(None, description="當前最高出價者ID")
    current_bidder_name: str | None = Field(None, description="當前最高出價者名稱")
    is_active: bool = Field(default=True, description="是否活躍中")
    currency_type: str = Field(default="楓幣", description="貨幣類型 (楓幣、雪花或台幣)")


class Bid(BaseModel):
    """出價記錄模型"""

    id: int | None = Field(None, description="出價ID")
    auction_id: int = Field(..., description="競標ID")
    guild_id: int = Field(..., description="伺服器ID")
    bidder_id: int = Field(..., description="出價者Discord ID")
    bidder_name: str = Field(..., description="出價者Discord名稱")
    amount: float = Field(..., description="出價金額")
    timestamp: datetime = Field(default_factory=datetime.now, description="出價時間")