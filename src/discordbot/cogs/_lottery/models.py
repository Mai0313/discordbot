from datetime import datetime

from pydantic import BaseModel


class LotteryParticipant(BaseModel):
    """Representation of a lottery participant."""

    id: str  # Discord 用戶 ID 或 YouTube 名稱
    name: str  # 顯示名稱
    source: str  # "discord" 或 "youtube"


class LotteryData(BaseModel):
    """Metadata describing a single lottery event."""

    lottery_id: int
    guild_id: int
    title: str
    description: str
    creator_id: int
    creator_name: str
    created_at: datetime
    is_active: bool
    registration_method: str  # "discord" 或 "youtube"
    youtube_url: str | None = None
    youtube_keyword: str | None = None
    control_message_id: int | None = None
    draw_count: int = 1
