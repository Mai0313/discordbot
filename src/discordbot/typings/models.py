from datetime import datetime

from pydantic import Field, BaseModel


# Shared models
class PageInfo(BaseModel):
    total_results: int = Field(..., validation_alias="totalResults", description="結果總數")
    results_per_page: int = Field(
        ..., validation_alias="resultsPerPage", description="每頁顯示的結果數"
    )


# Chat-related models
class TextMessageDetails(BaseModel):
    message_text: str = Field(
        ..., validation_alias="messageText", description="實際的文字訊息內容"
    )


class Snippet(BaseModel):
    type: str = Field(..., description="訊息的類型，例如 textMessageEvent")
    live_chat_id: str = Field(..., validation_alias="liveChatId", description="聊天室 ID")
    author_channel_id: str = Field(
        ..., validation_alias="authorChannelId", description="發送者的頻道 ID"
    )
    published_at: datetime = Field(
        ..., validation_alias="publishedAt", description="訊息發布的時間"
    )
    has_display_content: bool = Field(
        ..., validation_alias="hasDisplayContent", description="是否有可顯示的內容"
    )
    display_message: str = Field(
        ..., validation_alias="displayMessage", description="聊天室中顯示的訊息文字"
    )
    text_message_details: TextMessageDetails | None = Field(
        None, validation_alias="textMessageDetails", description="包含訊息詳細資料的物件"
    )


class AuthorDetails(BaseModel):
    channel_id: str = Field(..., validation_alias="channelId", description="使用者的頻道 ID")
    channel_url: str = Field(..., validation_alias="channelUrl", description="使用者的頻道網址")
    display_name: str = Field(
        ..., validation_alias="displayName", description="使用者在聊天室顯示的名稱"
    )
    profile_image_url: str = Field(
        ..., validation_alias="profileImageUrl", description="使用者的頭像 URL"
    )
    is_verified: bool = Field(
        ..., validation_alias="isVerified", description="是否為已驗證的使用者"
    )
    is_chat_owner: bool = Field(
        ..., validation_alias="isChatOwner", description="是否為聊天室擁有者"
    )
    is_chat_sponsor: bool = Field(
        ..., validation_alias="isChatSponsor", description="是否為聊天室贊助者"
    )
    is_chat_moderator: bool = Field(
        ..., validation_alias="isChatModerator", description="是否為聊天室管理員"
    )


class LiveChatMessageItem(BaseModel):
    kind: str = Field(..., description="物件的類型，例如 youtube#liveChatMessage")
    etag: str = Field(..., description="資源的 ETag")
    id: str = Field(..., description="此訊息的唯一識別碼")
    snippet: Snippet = Field(..., description="訊息的主要內容")
    author_details: AuthorDetails | None = Field(
        default=None, validation_alias="authorDetails", description="發送者的資訊"
    )


class LiveChatMessageListResponse(BaseModel):
    kind: str = Field(..., description="回應的類型，例如 youtube#liveChatMessageListResponse")
    etag: str = Field(..., description="回應的 ETag")
    polling_interval_millis: int = Field(
        ..., validation_alias="pollingIntervalMillis", description="多久後再次查詢 API（毫秒）"
    )
    page_info: PageInfo = Field(..., validation_alias="pageInfo", description="分頁資訊")
    next_page_token: str | None = Field(
        None, validation_alias="nextPageToken", description="下一頁的分頁標記"
    )
    items: list[LiveChatMessageItem] = Field(..., description="聊天室訊息列表")


# Stream-related models
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


class VideoListResponse(BaseModel):
    kind: str = Field(..., description="回應類型，例如 youtube#videoListResponse")
    etag: str = Field(..., description="回應的 ETag")
    items: list[VideoItem] = Field(..., description="影片列表")
    page_info: PageInfo = Field(..., validation_alias="pageInfo", description="分頁資訊")


__all__ = [
    "AuthorDetails",
    "LiveChatMessageItem",
    "LiveChatMessageListResponse",
    "LiveStreamingDetails",
    "PageInfo",
    "Snippet",
    "TextMessageDetails",
    "VideoItem",
    "VideoListResponse",
]
