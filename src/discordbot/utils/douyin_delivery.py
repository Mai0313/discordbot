"""Discord delivery planning and status-line formatting for downloaded Douyin posts."""

from typing import Protocol
from pathlib import Path

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.utils.discord_embeds import DISCORD_ATTACHMENT_LIMIT
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    MediaPlan,
    MediaDeliveryPlanner,
)


class DownloadedPost(Protocol):
    """What a delivery plan reads off whatever the platform downloaded."""

    @property
    def filenames(self) -> list[Path]:
        """Local paths of the downloaded files."""
        ...

    @property
    def total_bytes(self) -> int:
        """Combined size of those files."""
        ...

    @property
    def omitted_images(self) -> int:
        """Images present in the source post but not downloaded."""
        ...


class DouyinDelivery(BaseModel):
    """Planned Douyin delivery outcome and its pre-planning total size in MiB.

    Captures `total_mb` before planning; external hosting unlinks the source files on disk.
    """

    model_config = ConfigDict(frozen=True)

    plan: MediaPlan = Field(..., description="The attach-vs-host-vs-drop outcome for the files.")
    total_mb: float = Field(
        ..., description="Combined size of the download, read before planning.", examples=[12.4]
    )


async def plan_douyin_delivery(
    planner: MediaDeliveryPlanner, result: DownloadedPost, upload_limit: int
) -> DouyinDelivery:
    """Returns the delivery plan for one downloaded Douyin post.

    Reserves `MEDIA_ENVELOPE_MARGIN` for multi-file galleries and `0` for single-file sends.

    Args:
        planner (MediaDeliveryPlanner): Planner that partitions items into attach, host, or drop.
        result (DownloadedPost): Finished Douyin download.
        upload_limit (int): Destination's attachment size ceiling in bytes.
    """
    items = [MediaItem(source=path, filename=path.name) for path in result.filenames]
    total_mb = result.total_bytes / 1024 / 1024
    plan = await planner.plan(
        items=items,
        upload_limit=upload_limit,
        envelope_margin=MEDIA_ENVELOPE_MARGIN if len(items) > 1 else 0,
    )
    return DouyinDelivery(plan=plan, total_mb=total_mb)


def douyin_delivery_lines(
    result: DownloadedPost, plan: MediaPlan, hosting_available: bool, url: str, dropped_event: str
) -> list[str]:
    """Returns subtext lines for omitted or dropped Douyin items, followed by any hosted URLs.

    Hosted URLs are appended last and unwrapped.

    Args:
        result (DownloadedPost): Finished Douyin download.
        plan (MediaPlan): Planned delivery outcome for the downloaded files.
        hosting_available (bool): Whether external media hosting is active, recorded on drop logs.
        url (str): Source post URL for drop logs.
        dropped_event (str): Caller-specific log event name when items are dropped.
    """
    lines: list[str] = []
    if result.omitted_images:
        lines.append(
            f"-# 已省略 {result.omitted_images} 張圖片 (Discord 單則訊息最多 "
            f"{DISCORD_ATTACHMENT_LIMIT} 個附件)"
        )
    if plan.dropped_items:
        logfire.warn(
            dropped_event,
            url=url,
            dropped_count=len(plan.dropped_items),
            native_count=len(plan.native),
            hosted_count=len(plan.hosted_urls),
            hosting_available=hosting_available,
        )
        lines.append(f"-# 有 {len(plan.dropped_items)} 個檔案傳送失敗")
    lines.extend(plan.hosted_urls)
    return lines
