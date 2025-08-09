from datetime import datetime

from pydantic import Field, BaseModel


class LiveStreamingDetails(BaseModel):
    actual_start_time: datetime | None = Field(
        None, validation_alias="actualStartTime", description="實際開播時間"
    )
    scheduled_start_time: datetime | None = Field(
        None, validation_alias="scheduledStartTime", description="預定開播時間"
    )
    concurrent_viewers: str | None = Field(
        None, validation_alias="concurrentViewers", description="同時觀看人數"
    )
    active_live_chat_id: str | None = Field(
        None, validation_alias="activeLiveChatId", description="目前使用中的聊天室 ID"
    )


class VideoItem(BaseModel):
    kind: str = Field(..., description="物件類型，例如 youtube#video")
    etag: str = Field(..., description="資源的 ETag")
    id: str = Field(..., description="影片 ID")
    live_streaming_details: LiveStreamingDetails | None = Field(
        None, validation_alias="liveStreamingDetails", description="直播相關詳細資訊"
    )


class PageInfo(BaseModel):
    total_results: int = Field(..., validation_alias="totalResults", description="結果總數")
    results_per_page: int = Field(
        ..., validation_alias="resultsPerPage", description="每頁顯示結果數"
    )


class VideoListResponse(BaseModel):
    kind: str = Field(..., description="回應類型，例如 youtube#videoListResponse")
    etag: str = Field(..., description="回應的 ETag")
    items: list[VideoItem] = Field(..., description="影片列表")
    page_info: PageInfo = Field(..., validation_alias="pageInfo", description="分頁資訊")
