"""Slash command cog that downloads a video, or a Douyin gallery, and sends it back."""

import asyncio

import logfire
import nextcord
from nextcord import Locale, Interaction, SlashOption, AllowedMentions
from nextcord.ext import commands

from discordbot.utils.urls import extract_first_url
from discordbot.typings.video import VideoQuality
from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.typings.timeouts import VIDEO_DOWNLOAD_TIMEOUT_SECONDS
from discordbot.utils.scratch_dir import scratch_directory
from discordbot.utils.discord_embeds import DISCORD_ATTACHMENT_LIMIT
from discordbot.utils.media_delivery import (
    MediaItem,
    upload_limit_for,
    build_media_delivery_planner,
)
from discordbot.utils.douyin_delivery import (
    DouyinDelivery,
    plan_douyin_delivery,
    douyin_delivery_lines,
)
from discordbot.services.platforms.ytdlp import (
    DownloadResult,
    VideoDownloader,
    download_with_stop_signal,
)
from discordbot.services.platforms.douyin import (
    DOUYIN_URL_RE,
    DouyinDownload,
    DouyinDownloader,
    DouyinBlockedError,
    DouyinTransferError,
    DouyinUnavailableError,
    is_douyin_url,
)

# The labels Discord shows for the `quality` option, keyed to the presets themselves so a
# relabelling cannot drift onto a value the downloaders do not answer.
QUALITY_CHOICES: dict[str, VideoQuality] = {
    "Best Quality": "best",
    "High (1080p)": "high",
    "Medium (720p)": "medium",
    "Low (480p, 540p on Douyin)": "low",
}

_DOWNLOAD_FAILED = "-# 檔案無法下載"


def douyin_failure_message(error: Exception) -> str:
    """Maps a Douyin failure to the message a user should see.

    A bot wall, a missing post, a request that never got through and a stall are kept apart on
    purpose. Reporting any of them as a deleted post is the single worst outcome this feature
    can produce: it sends someone off to re-check a link that is perfectly fine. Only
    `DouyinUnavailableError` — Douyin explicitly filtering the post out — earns that wording,
    and only `DouyinBlockedError` earns the one that says Douyin is refusing requests: a
    stalled CDN read is retryable too, but blaming a wall sends someone off to wait out
    something that was never there.

    `DouyinTooLargeError` has no branch of its own: this command arms no `max_bytes`, so nothing
    it runs can raise it.
    """
    if isinstance(error, DouyinUnavailableError):
        return "-# 這則貼文已被刪除或設為私人"
    if isinstance(error, DouyinBlockedError):
        return "-# 抖音暫時擋住了請求，請稍後再試"
    if isinstance(error, DouyinTransferError):
        return "-# 這次沒有抓到,稍後再試一次"
    if isinstance(error, TimeoutError):
        return "-# 抖音回應太慢,這次沒有抓到;稍後再試一次"
    return _DOWNLOAD_FAILED


def _file_header(file_size_mb: float, url: str) -> str:
    """The size and source lines a delivered file leads with."""
    return f"-# 檔案大小: {file_size_mb:.1f}MB\n-# 來源: <{url}>"


class VideoCogs(commands.Cog):
    """Downloads videos from slash command requests.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        media_delivery: Planner deciding which files attach and which are hosted as a URL.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the VideoCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.media_delivery = build_media_delivery_planner()

    @nextcord.slash_command(
        name="download_video",
        description="Download a video from various platforms and send it back.",
        name_localizations={Locale.zh_TW: "下載影片", Locale.ja: "動画ダウンロード"},
        description_localizations={
            Locale.zh_TW: "從多種平台下載影片並傳送 (支援 YouTube, Facebook, Instagram, X, Tiktok, 抖音 等)",
            Locale.ja: "YouTube, Facebook, Instagram, X, Tiktok, 抖音 などから動画をダウンロードして送信します。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def download_video(
        self,
        interaction: Interaction[commands.Bot],
        url: str = SlashOption(
            description="Video URL, or the share text containing it (YouTube, Instagram, X, Douyin, etc.)",
            description_localizations={
                Locale.zh_TW: "影片連結,或含有連結的分享文字 (YouTube, Instagram, X, 抖音 等)",
                Locale.ja: "動画のリンク、またはそれを含む共有テキスト (YouTube, Instagram, X, 抖音 など)",
            },
            required=True,
        ),
        quality: VideoQuality = SlashOption(
            description="Video quality (higher quality = larger file size)",
            description_localizations={
                Locale.zh_TW: "影片畫質 (畫質越高,檔案越大)",
                Locale.ja: "動画の画質 (高画質ほどファイルサイズが大きくなります)",
            },
            required=False,
            default="best",
            choices=QUALITY_CHOICES,
        ),
    ) -> None:
        """Downloads a video from various platforms and sends it back.

        Args:
            interaction: The interaction that triggered the command.
            url: The URL of the video to download.
            quality: The desired video quality.
        """
        await interaction.response.defer()
        await interaction.edit_original_message(content="-# 正在下載影片...")

        # Share buttons hand over a blob of text with the link buried in it, and pasting that
        # whole thing here is the natural thing to do. Douyin's pattern goes first because its
        # copy runs straight into Chinese with no space, where the generic rule would swallow it.
        url = extract_first_url(text=url, patterns=(DOUYIN_URL_RE,))

        upload_limit = upload_limit_for(guild=interaction.guild)

        # Douyin never reaches yt-dlp: `services/platforms/douyin.py` has why.
        if is_douyin_url(url=url):
            await self._download_douyin(
                interaction=interaction, url=url, quality=quality, upload_limit=upload_limit
            )
            return

        try:
            # The scratch directory is also what removes the downloaded file once it is sent.
            with scratch_directory(prefix="download-video-") as download_dir:
                downloader = VideoDownloader(output_folder=download_dir)
                # Bounded because yt-dlp's own `socket_timeout` is per socket and every retry
                # setting multiplies it, so a stalling host would otherwise leave the user on
                # "正在下載影片..." indefinitely.
                async with asyncio.timeout(delay=VIDEO_DOWNLOAD_TIMEOUT_SECONDS):
                    result = await download_with_stop_signal(
                        downloader=downloader, url=url, quality=quality
                    )
                try:
                    await self._deliver_download(
                        interaction=interaction, url=url, result=result, upload_limit=upload_limit
                    )
                except Exception as error:
                    # Broad on purpose: anything narrower falls through to the handler below
                    # and is logged as a download failure.
                    logfire.warn(
                        "Video delivery failed",
                        url=url,
                        error_type=type(error).__name__,
                        _exc_info=error,
                    )
                    await self._edit_quietly(interaction=interaction, content=_DOWNLOAD_FAILED)
        except Exception as error:
            # Broad on purpose: nothing answers the interaction on an error, so anything
            # escaping here would strand the user on "正在下載影片..." forever.
            logfire.warn(
                "Video download failed",
                url=url,
                quality=quality,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            await self._edit_quietly(interaction=interaction, content=_DOWNLOAD_FAILED)

    async def _deliver_download(
        self,
        interaction: Interaction[commands.Bot],
        url: str,
        result: DownloadResult,
        upload_limit: int,
    ) -> None:
        """Attaches, hosts or refuses one finished yt-dlp download.

        Args:
            interaction: The interaction the command is holding open.
            url: The source URL, appended only when the file is attached natively.
            result: The finished download, inside the caller's scratch directory.
            upload_limit: The destination's attachment ceiling.
        """
        file_size_mb = result.filename.stat().st_size / 1024 / 1024
        item = MediaItem(source=result.filename, filename=result.filename.name)
        plan = await self.media_delivery.plan(items=[item], upload_limit=upload_limit)
        if plan.native:
            await self._deliver(
                interaction=interaction, file_size_mb=file_size_mb, item=plan.native[0], url=url
            )
            return

        # Too big for native upload: host the original-quality file and post its URL, rather
        # than downgrading quality.
        if plan.hosted_urls:
            await self._deliver_url(
                interaction=interaction, file_size_mb=file_size_mb, public_url=plan.hosted_urls[0]
            )
            return

        await self._refuse_oversize(
            interaction=interaction, file_size_mb=file_size_mb, upload_limit=upload_limit
        )

    async def _download_douyin(
        self,
        interaction: Interaction[commands.Bot],
        url: str,
        quality: VideoQuality,
        upload_limit: int,
    ) -> None:
        """Downloads a Douyin video or photo post and sends it back.

        Kept separate from the yt-dlp branch because a Douyin post can be a gallery, which needs
        several attachments on one message rather than the single file the yt-dlp path delivers.

        Args:
            interaction: The interaction that triggered the command.
            url: The Douyin URL.
            quality: The desired video quality; ignored for a photo post.
            upload_limit: The destination's attachment ceiling.
        """
        with scratch_directory(prefix="download-video-douyin-") as download_dir:
            downloader = DouyinDownloader(output_folder=download_dir)
            try:
                # Capped at the attachment limit so a 48-image gallery does not download 38 files
                # that could never be sent; `omitted_images` reports what the cap left behind.
                # Bounded on wall-clock too, because a gallery costs `download_timeout` x
                # `max_retries` per file: a stalling CDN would otherwise hold this command open
                # for half an hour.
                async with asyncio.timeout(delay=VIDEO_DOWNLOAD_TIMEOUT_SECONDS):
                    result = await asyncio.to_thread(
                        downloader.download,
                        url=url,
                        quality=quality,
                        max_images=DISCORD_ATTACHMENT_LIMIT,
                    )
            except Exception as error:
                # Deliberately catches everything, not just DouyinError: this runs outside the
                # command's own try block and nothing answers the interaction on an error, so
                # anything escaping here would strand the user on "正在下載影片..." forever.
                logfire.warn(
                    "Douyin download failed",
                    url=url,
                    quality=quality,
                    error_type=type(error).__name__,
                    _exc_info=error,
                )
                await self._edit_quietly(
                    interaction=interaction, content=douyin_failure_message(error=error)
                )
                return

            try:
                delivery = await plan_douyin_delivery(
                    planner=self.media_delivery, result=result, upload_limit=upload_limit
                )
                plan = delivery.plan

                # Nothing to attach has two ways out; anything else is the normal reply.
                if not plan.native:
                    # Only a lone oversize file may collapse to the bare-URL reply, which
                    # deliberately posts nothing but the link so Discord renders the inline
                    # player. A gallery would lose every URL past the first, plus the omitted
                    # / dropped notices, so it goes through the normal reply instead.
                    if plan.hosted_urls and len(result.filenames) == 1:
                        await self._deliver_url(
                            interaction=interaction,
                            file_size_mb=delivery.total_mb,
                            public_url=plan.hosted_urls[0],
                        )
                        return
                    if not plan.hosted_urls:
                        await self._refuse_oversize(
                            interaction=interaction,
                            file_size_mb=delivery.total_mb,
                            upload_limit=upload_limit,
                        )
                        return

                await self._deliver_douyin(
                    interaction=interaction, delivery=delivery, result=result, url=url
                )
            except Exception as error:
                # Broad on purpose, for the same reason as the download step above: an escape
                # leaves the interaction unanswered and the user on the placeholder.
                logfire.warn(
                    "Douyin delivery failed",
                    url=url,
                    error_type=type(error).__name__,
                    _exc_info=error,
                )
                await self._edit_quietly(interaction=interaction, content=_DOWNLOAD_FAILED)

    async def _deliver_douyin(
        self,
        interaction: Interaction[commands.Bot],
        delivery: DouyinDelivery,
        result: DouyinDownload,
        url: str,
    ) -> None:
        """Edits the placeholder into the final Douyin response.

        The size and source lines lead because this is the command surface: someone who asked for
        a file wants to know how big it is and where it came from. Everything after them is the
        shared omitted / dropped / hosted accounting.

        Args:
            interaction: The interaction that triggered the command.
            delivery: The attach-vs-host outcome plus the size read before planning.
            result: The downloaded post, carrying the pre-cap image count.
            url: The source Douyin URL.
        """
        plan = delivery.plan
        lines = [_file_header(file_size_mb=delivery.total_mb, url=url)]
        lines.extend(
            douyin_delivery_lines(
                result=result,
                plan=plan,
                hosting_available=self.media_delivery.media_hosting.config.available,
                url=url,
                dropped_event="Douyin download dropped some media",
            )
        )

        # An all-hosted gallery reaches here with nothing to attach, and the edit carries only the
        # URLs, so the attachment list is omitted entirely rather than sent empty.
        files = [item.to_file() for item in plan.native]
        if files:
            await interaction.edit_original_message(
                content="\n".join(lines), files=files, allowed_mentions=AllowedMentions.none()
            )
            return

        await interaction.edit_original_message(
            content="\n".join(lines), allowed_mentions=AllowedMentions.none()
        )

    async def _edit_quietly(self, interaction: Interaction[commands.Bot], content: str) -> None:
        """Edits the deferred message, swallowing a failure to edit it.

        Broad on purpose: this is the last-resort reporter every failure path in the cog uses, so
        it must never raise a second exception on top of the one being reported.
        """
        try:
            await interaction.edit_original_message(content=content)
        except Exception as error:
            logfire.warn(
                "Could not report the failure to the user",
                interaction_id=interaction.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )

    async def _refuse_oversize(
        self, interaction: Interaction[commands.Bot], file_size_mb: float, upload_limit: int
    ) -> None:
        """Tells the user a file too big to attach could not be hosted either."""
        await self._edit_quietly(
            interaction=interaction,
            content=(
                f"-# 下載失敗\n檔案大小 {file_size_mb:.1f}MB，"
                f"超過上傳上限 {upload_limit / 1024 / 1024:.0f}MB"
            ),
        )

    async def _deliver(
        self,
        interaction: Interaction[commands.Bot],
        file_size_mb: float,
        item: MediaItem,
        url: str,
    ) -> None:
        """Edits the deferred placeholder into the final downloaded file response."""
        await interaction.edit_original_message(
            content=_file_header(file_size_mb=file_size_mb, url=url),
            file=item.to_file(),
            allowed_mentions=AllowedMentions.none(),
        )

    async def _deliver_url(
        self, interaction: Interaction[commands.Bot], file_size_mb: float, public_url: str
    ) -> None:
        """Edits the placeholder into a hosted-URL response for a file too big to upload.

        The hosted URL is the only link in the message so Discord renders the inline video player
        (a second URL such as the source link, even wrapped in `<>`, stops Discord from rendering
        the inline player, so the source is intentionally omitted here). Under ~100 MiB Discord
        inline-plays the link; above it the link is browser-playable.
        """
        body = f"-# 檔案大小: {file_size_mb:.1f}MB (過大，改用連結)\n{public_url}"
        await interaction.edit_original_message(
            content=body, allowed_mentions=AllowedMentions.none()
        )


def setup(bot: commands.Bot) -> None:
    """Adds the VideoCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(VideoCogs(bot), override=True)
