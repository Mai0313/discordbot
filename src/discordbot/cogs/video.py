"""Slash command cog for downloading videos through yt-dlp."""

import asyncio
from pathlib import Path
import tempfile
import contextlib

import logfire
import nextcord
from nextcord import File, Locale, Interaction, SlashOption, AllowedMentions
from nextcord.ext import commands

from discordbot.utils.downloader import VideoDownloader
from discordbot.utils.media_hosting import MediaHostingConfig, MediaHostingService
from discordbot.utils.discord_limits import upload_limit_for


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
        self.media_hosting = MediaHostingService(config=MediaHostingConfig())

    @nextcord.slash_command(
        name="download_video",
        description="Download a video from various platforms and send it back.",
        name_localizations={Locale.zh_TW: "下載影片", Locale.ja: "動画ダウンロード"},
        description_localizations={
            Locale.zh_TW: "從多種平台下載影片並傳送 (支援 YouTube, Facebook, Instagram, X, Tiktok 等)",
            Locale.ja: "YouTube, Facebook, Instagram, X, Tiktok などから動画をダウンロードして送信します。",
        },
        nsfw=False,
    )
    async def download_video(
        self,
        interaction: Interaction,
        url: str = SlashOption(
            description="Video URL (YouTube, Facebook Reels, Instagram, X, etc.)", required=True
        ),
        quality: str = SlashOption(
            description="Video quality (higher quality = larger file size)",
            required=False,
            default="best",
            choices={
                "Best Quality": "best",
                "High (1080p)": "high",
                "Medium (720p)": "medium",
                "Low (480p)": "low",
            },
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

        # Read the destination's real upload ceiling (boost tier raises it to 50/100 MiB);
        # a DM has no guild to query, so fall back to Discord's current non-Nitro base of 10 MiB.
        upload_limit = upload_limit_for(guild=interaction.guild)

        try:
            downloader = VideoDownloader(output_folder=tempfile.gettempdir())
            result = await asyncio.to_thread(downloader.download, url=url, quality=quality)
            with result:
                size_bytes = result.filename.stat().st_size
                file_size_mb = size_bytes / 1024 / 1024
                if size_bytes <= upload_limit:
                    await self._deliver(
                        interaction=interaction,
                        file_size_mb=file_size_mb,
                        file_path=result.filename,
                        url=url,
                    )
                    return

                # Too big for native upload: host the original-quality file and post its URL,
                # rather than downgrading quality. Under ~100 MiB Discord still inline-plays the
                # link; above that it is a browser-playable link. publish_path moves the file out
                # of the temp dir, so the `with result` exit unlink becomes a no-op.
                public_url = await asyncio.to_thread(
                    self.media_hosting.publish_path, file_path=result.filename
                )
                if public_url is not None:
                    await self._deliver_url(
                        interaction=interaction,
                        file_size_mb=file_size_mb,
                        public_url=public_url,
                        url=url,
                    )
                    return

                await interaction.edit_original_message(
                    content=f"-# 下載失敗\n檔案大小超過 {file_size_mb:.1f}MB"
                )
        except Exception:
            logfire.warn("Video download failed", _exc_info=True)
            with contextlib.suppress(Exception):
                await interaction.edit_original_message(content="-# 檔案無法下載")

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
        self, interaction: Interaction, file_size_mb: float, public_url: str, url: str
    ) -> None:
        """Edits the placeholder into a hosted-URL response for a file too big to upload.

        The hosted URL is posted unwrapped so Discord embeds it (inline player under ~100 MiB,
        a browser-playable link above); the source `url` stays wrapped in `<>` to suppress its
        own embed.
        """
        body = (
            f"-# 檔案大小: {file_size_mb:.1f}MB (過大，改用連結)\n{public_url}\n-# 來源: <{url}>"
        )
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
