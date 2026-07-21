"""Slash command cog for downloading videos through yt-dlp."""

import asyncio
from pathlib import Path
import tempfile
import contextlib

import logfire
import nextcord
from nextcord import File, Locale, Interaction, SlashOption, AllowedMentions
from nextcord.ext import commands

from discordbot.utils.urls import extract_first_url
from discordbot.utils.douyin import DOUYIN_URL_RE, DouyinDownload, DouyinDownloader, is_douyin_url
from discordbot.typings.video import VideoQuality
from discordbot.utils.downloader import VideoDownloader
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    DISCORD_ATTACHMENT_LIMIT,
    MediaItem,
    MediaPlan,
    upload_limit_for,
    build_media_delivery_planner,
)
from discordbot.cogs._parse_douyin.fetch import douyin_failure_message

# The labels Discord shows for the `quality` option, keyed to the presets themselves so a
# relabelling cannot drift onto a value the downloaders do not answer.
QUALITY_CHOICES: dict[str, VideoQuality] = {
    "Best Quality": "best",
    "High (1080p)": "high",
    "Medium (720p)": "medium",
    "Low (480p)": "low",
}


class VideoCogs(commands.Cog):
    """Downloads videos from slash command requests.

    Attributes:
        bot: The Discord bot instance that owns this cog.
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
    )
    async def download_video(
        self,
        interaction: Interaction,
        url: str = SlashOption(
            description="Video URL, or the share text containing it (YouTube, Instagram, X, Douyin, etc.)",
            required=True,
        ),
        quality: VideoQuality = SlashOption(
            description="Video quality (higher quality = larger file size)",
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

        # Read the destination's real upload ceiling (boost tier raises it to 50/100 MiB);
        # a DM has no guild to query, so fall back to Discord's current non-Nitro base of 10 MiB.
        upload_limit = upload_limit_for(guild=interaction.guild)

        # Douyin is routed away from yt-dlp entirely: its extractor needs cookies, never yields a
        # photo post, and caps below the source resolution. The yt-dlp path below is untouched.
        if is_douyin_url(url=url):
            await self._handle_douyin(
                interaction=interaction, url=url, quality=quality, upload_limit=upload_limit
            )
            return

        try:
            downloader = VideoDownloader(output_folder=tempfile.gettempdir())
            result = await asyncio.to_thread(downloader.download, url=url, quality=quality)
            with result:
                file_size_mb = result.filename.stat().st_size / 1024 / 1024
                item = MediaItem(source=result.filename, filename=result.filename.name)
                plan = await self.media_delivery.plan(items=[item], upload_limit=upload_limit)
                if plan.native:
                    await self._deliver(
                        interaction=interaction,
                        file_size_mb=file_size_mb,
                        file_path=result.filename,
                        url=url,
                    )
                    return

                # Too big for native upload: host the original-quality file and post its URL,
                # rather than downgrading quality. Under ~100 MiB Discord still inline-plays the
                # link; above that it is a browser-playable link. Hosting moves the file into the
                # serve dir on a fresh upload (the `with result` exit unlink then no-ops) but leaves
                # it on a dedup hit, so the exit unlink (missing_ok) cleans it up either way.
                if plan.hosted_urls:
                    await self._deliver_url(
                        interaction=interaction,
                        file_size_mb=file_size_mb,
                        public_url=plan.hosted_urls[0],
                    )
                    return

                await interaction.edit_original_message(
                    content=f"-# 下載失敗\n檔案大小超過 {file_size_mb:.1f}MB"
                )
        except Exception:
            logfire.warn("Video download failed", _exc_info=True)
            await self._edit_quietly(interaction=interaction, content="-# 檔案無法下載")

    async def _handle_douyin(
        self, interaction: Interaction, url: str, quality: VideoQuality, upload_limit: int
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
        # A private directory per invocation, because the filenames are derived from the post id:
        # two people downloading the same post into one shared temp dir would write the same paths,
        # letting one truncate the other's file and letting either one's cleanup delete a file the
        # other is still uploading. The directory is removed once delivery finishes.
        with tempfile.TemporaryDirectory(prefix="douyin-") as download_dir:
            await self._download_and_deliver_douyin(
                interaction=interaction,
                url=url,
                quality=quality,
                upload_limit=upload_limit,
                download_dir=download_dir,
            )

    async def _download_and_deliver_douyin(
        self,
        interaction: Interaction,
        url: str,
        quality: VideoQuality,
        upload_limit: int,
        download_dir: str,
    ) -> None:
        """Runs the Douyin download and delivery inside a caller-owned download directory."""
        downloader = DouyinDownloader(output_folder=download_dir)
        try:
            # Capped at the attachment limit so a 48-image gallery does not download 38 files
            # that could never be sent; `omitted_images` reports what the cap left behind.
            result = await asyncio.to_thread(
                downloader.download, url=url, quality=quality, max_images=DISCORD_ATTACHMENT_LIMIT
            )
        except Exception as error:
            # Deliberately catches everything, not just DouyinError: this runs outside the
            # command's own try block and the bot registers no application-command error handler,
            # so anything escaping here would strand the user on "正在下載影片..." forever.
            logfire.warn("Douyin download failed", _exc_info=True)
            await self._edit_quietly(
                interaction=interaction, content=douyin_failure_message(error=error)
            )
            return

        try:
            with result:
                items = [MediaItem(source=path, filename=path.name) for path in result.filenames]
                # Read BEFORE planning, which caches it: a successful host moves the source out of
                # the temp dir, so measuring afterwards would stat a deleted path and break the very
                # oversize-to-URL fallback this number describes.
                total_mb = result.total_bytes / 1024 / 1024
                plan = await self.media_delivery.plan(
                    items=items,
                    upload_limit=upload_limit,
                    # A gallery rides several attachments on one edit, and Discord measures the
                    # whole multipart body, so hold back the envelope the way the other combined
                    # sends do. A lone video is a single-file send and keeps the margin at 0, which
                    # is what the yt-dlp branch above does.
                    envelope_margin=MEDIA_ENVELOPE_MARGIN if len(items) > 1 else 0,
                )

                # Only a lone oversize file may collapse to the bare-URL reply, which deliberately
                # posts nothing but the link so Discord renders the inline player. A gallery would
                # lose every URL past the first, plus the omitted / dropped notices, so it goes
                # through the normal reply instead.
                if not plan.native and plan.hosted_urls and len(items) == 1:
                    await self._deliver_url(
                        interaction=interaction,
                        file_size_mb=total_mb,
                        public_url=plan.hosted_urls[0],
                    )
                    return

                if not plan.native and plan.hosted_urls:
                    await self._deliver_douyin(
                        interaction=interaction, plan=plan, result=result, url=url
                    )
                    return

                if not plan.native:
                    await self._edit_quietly(
                        interaction=interaction,
                        content=f"-# 下載失敗\n檔案大小超過 {total_mb:.1f}MB",
                    )
                    return

                await self._deliver_douyin(
                    interaction=interaction, plan=plan, result=result, url=url
                )
        except Exception:
            logfire.warn("Douyin delivery failed", _exc_info=True)
            await self._edit_quietly(interaction=interaction, content="-# 檔案無法下載")

    async def _deliver_douyin(
        self, interaction: Interaction, plan: MediaPlan, result: DouyinDownload, url: str
    ) -> None:
        """Edits the placeholder into the final Douyin response.

        Anything left out is stated explicitly rather than silently dropped, so the user knows a
        gallery they see is partial. The two causes are reported separately because they are not
        the same problem: the attachment cap is a Discord limit nothing can change, while a
        dropped item means delivery itself failed.

        Args:
            interaction: The interaction that triggered the command.
            plan: The attach-vs-host outcome for the downloaded files.
            result: The downloaded post, carrying the pre-cap image count.
            url: The source Douyin URL.
        """
        total_mb = result.total_bytes / 1024 / 1024
        lines = [f"-# 檔案大小: {total_mb:.1f}MB", f"-# 來源: <{url}>"]
        if result.omitted_images:
            lines.append(
                f"-# 已省略 {result.omitted_images} 張圖片 (Discord 單則訊息最多 "
                f"{DISCORD_ATTACHMENT_LIMIT} 個附件)"
            )
        if plan.dropped_items:
            lines.append(f"-# 有 {len(plan.dropped_items)} 個檔案傳送失敗")
        # Hosted URLs must stay unwrapped to remain clickable, so they follow the subtext lines.
        lines.extend(plan.hosted_urls)

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

    async def _edit_quietly(self, interaction: Interaction, content: str) -> None:
        """Edits the deferred message, swallowing a failure to edit it."""
        with contextlib.suppress(Exception):
            await interaction.edit_original_message(content=content)

    async def _deliver(
        self, interaction: Interaction, file_size_mb: float, file_path: Path, url: str
    ) -> None:
        """Edits the deferred placeholder into the final downloaded file response."""
        body = f"-# 檔案大小: {file_size_mb:.1f}MB\n-# 來源: <{url}>"
        await interaction.edit_original_message(
            content=body,
            file=File(fp=file_path, filename=file_path.name),
            allowed_mentions=AllowedMentions.none(),
        )

    async def _deliver_url(
        self, interaction: Interaction, file_size_mb: float, public_url: str
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


# 註冊 Cog
def setup(bot: commands.Bot) -> None:
    """Adds the VideoCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(VideoCogs(bot), override=True)
