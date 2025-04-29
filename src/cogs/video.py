from pathlib import Path

from yt_dlp import YoutubeDL
import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands


class VideoCogs(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # 準備下載資料夾
        self.download_folder = Path("downloads")
        self.download_folder.mkdir(exist_ok=True)
        # Discord 檔案上傳大小限制 (25MB in bytes)
        self.max_file_size = 25 * 1024 * 1024

    @nextcord.slash_command(
        name="download_video",
        description="Download a video from various platforms and send it back.",
        name_localizations={
            Locale.zh_TW: "下載影片",
            Locale.zh_CN: "下载视频",
            Locale.ja: "動画ダウンロード",
        },
        description_localizations={
            Locale.zh_TW: "從多種平台下載影片並傳送 (支援 YouTube、Facebook、Instagram、X 等)。",
            Locale.zh_CN: "从多种平台下载视频并发送 (支持 YouTube、Facebook、Instagram、X 等)。",
            Locale.ja: "YouTube、Facebook、Instagram、X などから動画をダウンロードして送信します。",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def download_video(
        self,
        interaction: Interaction,
        url: str = SlashOption(
            description="Video URL (YouTube, Facebook Reels, Instagram, X, etc.)", required=True
        ),
    ) -> None:
        # 避免互動超時
        await interaction.response.defer()
        try:
            # 設定 yt_dlp 選項，輸出到 downloads 資料夾
            ydl_opts = {
                "format": "best",
                "outtmpl": str(self.download_folder / "%(title)s.%(ext)s"),
                "continuedl": True,
                "restrictfilenames": True,
            }
            # 下載並取得檔案資訊
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = Path(ydl.prepare_filename(info))

            # 檢查檔案大小是否超過 Discord 限制 (25MB)
            if filename.stat().st_size > self.max_file_size:
                await interaction.followup.send(
                    "❌ 影片檔案大小超過 25MB，無法上傳至 Discord。請嘗試較短的影片或較低畫質的來源。"
                )
                filename.unlink()  # 刪除檔案
                return

            # 傳送檔案並刪除
            await interaction.followup.send(
                file=nextcord.File(str(filename), filename=filename.name)
            )
            filename.unlink()  # 刪除檔案
        except Exception as e:
            await interaction.followup.send(f"❌ 下載失敗: {e}")


# 註冊 Cog
async def setup(bot: commands.Bot) -> None:
    bot.add_cog(VideoCogs(bot), override=True)
