"""Slash command cog for downloading videos through yt-dlp."""

import asyncio
from pathlib import Path
import contextlib

import logfire
import nextcord
from nextcord import File, Locale, Interaction, SlashOption, AllowedMentions
from nextcord.ext import commands

from discordbot.utils.downloader import VideoDownloader

# Hard-cap aligned with the unboosted Discord upload limit. The downloader
# already retries at low quality once when this is exceeded.
_DISCORD_FILE_LIMIT_BYTES = 25 * 1024 * 1024


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

        try:
            downloader = VideoDownloader(output_folder="./data/downloads")
            result = await asyncio.to_thread(downloader.download, url=url, quality=quality)
            with result:
                file_size_mb = result.filename.stat().st_size / 1024 / 1024
                if result.filename.stat().st_size <= _DISCORD_FILE_LIMIT_BYTES:
                    await self._deliver(
                        interaction=interaction,
                        file_size_mb=file_size_mb,
                        file_path=result.filename,
                        url=url,
                    )
                    return

                if quality == "low":
                    await interaction.edit_original_message(
                        content=f"-# 下載失敗\n檔案大小超過 {file_size_mb:.1f}MB"
                    )
                    return

                await interaction.edit_original_message(
                    content=f"-# 檔案過大 ({file_size_mb:.1f}MB)，正在重新下載低畫質版本..."
                )
                low_result = await asyncio.to_thread(downloader.download, url=url, quality="low")
                with low_result:
                    file_size_mb = low_result.filename.stat().st_size / 1024 / 1024
                    if low_result.filename.stat().st_size > _DISCORD_FILE_LIMIT_BYTES:
                        await interaction.edit_original_message(
                            content=f"-# 下載失敗\n檔案大小超過 {file_size_mb:.1f}MB"
                        )
                        return
                    await self._deliver(
                        interaction=interaction,
                        file_size_mb=file_size_mb,
                        file_path=low_result.filename,
                        url=url,
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


# 註冊 Cog
def setup(bot: commands.Bot) -> None:
    """Adds the VideoCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(VideoCogs(bot), override=True)
