"""Slash command cog for downloading videos through yt-dlp."""

import contextlib

import logfire
import nextcord
from nextcord import File, Locale, Interaction, SlashOption
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
        await interaction.edit_original_message(content="⏳ 正在下載影片，請稍候...")

        try:
            downloader = VideoDownloader(output_folder="./data/downloads")
            with downloader.download(url=url, quality=quality) as result:
                file_size_mb = result.filename.stat().st_size / 1024 / 1024
                if result.filename.stat().st_size <= _DISCORD_FILE_LIMIT_BYTES:
                    await self._deliver(
                        interaction=interaction,
                        file_size_mb=file_size_mb,
                        file_path=str(result.filename),
                        file_name=result.filename.name,
                    )
                    return

                if quality == "low":
                    await interaction.edit_original_message(
                        content=f":x: 下載失敗\n檔案大小超過 {file_size_mb:.1f}MB"
                    )
                    return

                await interaction.edit_original_message(
                    content=f"⚠️ 檔案過大 ({file_size_mb:.1f}MB)，正在重新下載低畫質版本..."
                )
                with downloader.download(url=url, quality="low") as low_result:
                    file_size_mb = low_result.filename.stat().st_size / 1024 / 1024
                    if low_result.filename.stat().st_size > _DISCORD_FILE_LIMIT_BYTES:
                        await interaction.edit_original_message(
                            content=f":x: 下載失敗\n檔案大小超過 {file_size_mb:.1f}MB"
                        )
                        return
                    await self._deliver(
                        interaction=interaction,
                        file_size_mb=file_size_mb,
                        file_path=str(low_result.filename),
                        file_name=low_result.filename.name,
                    )
        except Exception:
            logfire.warn("Video download failed", _exc_info=True)
            with contextlib.suppress(Exception):
                await interaction.edit_original_message(content=":x: 下載失敗\n檔案無法下載")

    async def _deliver(
        self, interaction: Interaction, file_size_mb: float, file_path: str, file_name: str
    ) -> None:
        """Sends the downloaded file as a fresh followup and collapses the placeholder.

        ``interaction.edit_original_message(content=…, file=…)`` drops
        ``content`` when a multipart file payload is attached, so we send the
        file as a separate followup and then collapse the original placeholder
        to a checkmark.
        """
        body = f"✅ 下載成功! 檔案大小: {file_size_mb:.1f}MB"
        await interaction.followup.send(content=body, file=File(fp=file_path, filename=file_name))
        with contextlib.suppress(Exception):
            await interaction.edit_original_message(content="✅")


# 註冊 Cog
def setup(bot: commands.Bot) -> None:
    """Adds the VideoCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(VideoCogs(bot), override=True)
