"""How a downloaded Douyin post reaches Discord: the plan, and the lines that report it.

Kept out of the platform reader because this is the Discord side of the job — it reaches
`nextcord.File` through `utils/media_delivery.py`, and `services/platforms/` is Discord-free.
Sitting here is also why `/download_video` and the auto-expansion can share it: a cog may not
import a peer cog, so the one thing both need lives one layer down.

What it reads off a finished download is spelled as a Protocol rather than imported, for the
same layering reason in the other direction: `DouyinDownload` lives under `services/` now, and
`utils/` may not import that. Structural typing costs nothing here — `result` is a parameter,
never a pydantic field, so nothing validates against it.
"""

from typing import Protocol
from pathlib import Path

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    DISCORD_ATTACHMENT_LIMIT,
    MediaItem,
    MediaPlan,
    MediaDeliveryPlanner,
)


class DownloadedPost(Protocol):
    """The three members a delivery plan reads off whatever the platform downloaded.

    `total_bytes` has to be read before the plan runs and before a successful host moves the
    files out of the temp dir, which is why `DouyinDelivery` carries the number rather than the
    download; `DouyinDownload.total_bytes` caches its own answer for the same reason.
    """

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
    """A planned Douyin send: what goes out, and the size a refusal has to be able to quote.

    `total_mb` rides along rather than being re-read off the download, because reading it is
    order-sensitive: `DouyinDownload.total_bytes` stats the files and caches the answer, and a
    successful host moves them out of the temp dir, so the read has to happen BEFORE the plan.
    Carrying the number here is what stops a later caller re-deriving it from a deleted path —
    on exactly the oversize path that most needs it.
    """

    model_config = ConfigDict(frozen=True)

    plan: MediaPlan = Field(..., description="The attach-vs-host-vs-drop outcome for the files.")
    total_mb: float = Field(
        ..., description="Combined size of the download, read before planning.", examples=[12.4]
    )


async def plan_douyin_delivery(
    *, planner: MediaDeliveryPlanner, result: DownloadedPost, upload_limit: int
) -> DouyinDelivery:
    """Decides how one downloaded Douyin post reaches Discord.

    Shared by `/download_video` and the auto-expansion, which differ only in where the upload
    limit comes from. A gallery rides several attachments on one send and Discord measures the
    whole multipart body, so it holds back the envelope margin; a lone video is a single-file
    send and keeps the margin at 0.
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
    *,
    result: DownloadedPost,
    plan: MediaPlan,
    hosting_available: bool,
    url: str,
    dropped_event: str,
) -> list[str]:
    """The subtext lines stating what a Douyin send left out, plus any hosted URLs.

    Anything left out is said explicitly rather than silently dropped, so a user seeing a
    partial gallery knows it is partial. The two causes are reported separately because they are
    not the same problem: the attachment cap is a Discord limit nothing can change, while a
    dropped item means delivery itself failed.

    `dropped_event` is the caller's own log message rather than a shared one. That is the whole
    point of it: `/download_video` and the auto-expansion are told apart in `data/logs` by the
    event name alone, so merging them would cost the one field that says which path dropped the
    media.

    Hosted URLs come last and unwrapped: they must stay clickable and, under ~100 MiB, render
    Discord's inline player.
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
