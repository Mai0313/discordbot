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

        # 影片畫質對應的 yt_dlp 格式設定
        self.quality_formats = {
            "best": "best",
            "high": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "medium": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "low": "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "audio": "bestaudio/best",
        }

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
        quality: str = SlashOption(
            description="Video quality (higher quality = larger file size)",
            required=False,
            default="best",
            choices={
                "Best Quality": "best",
                "High (1080p)": "high",
                "Medium (720p)": "medium",
                "Low (480p)": "low",
                "Audio Only": "audio",
            },
        ),
    ) -> None:
        # 避免互動超時
        await interaction.response.defer()

        # 發送初始狀態訊息並保存引用
        await interaction.followup.send(f"🔄 正在下載影片，請稍候... (已選擇{quality}畫質)")

        try:
            # 獲取所選畫質的格式設定
            format_option = self.quality_formats.get(quality, "best")
            is_audio_only = quality == "audio"

            # 設定 yt_dlp 選項
            ydl_opts = {
                "format": format_option,
                "outtmpl": str(self.download_folder / "%(title).40s-%(id)s.%(ext)s"),
                "continuedl": True,
                "restrictfilenames": True,
            }

            # 如果是音訊模式，轉換成 mp3
            if is_audio_only:
                ydl_opts.update({
                    "postprocessors": [
                        {
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3",
                            "preferredquality": "192",
                        }
                    ]
                })

            # 下載並取得檔案資訊
            with YoutubeDL(ydl_opts) as ydl:
                # 更新訊息顯示正在下載
                await interaction.edit_original_message(
                    content=f"⏳ 正在下載 {url}... (已選擇{quality}畫質)"
                )
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "")
                filename = Path(ydl.prepare_filename(info))

                # 修正音訊模式下的副檔名
                if is_audio_only and filename.suffix != ".mp3":
                    filename = filename.with_suffix(".mp3")

            # 檢查檔案大小是否超過 Discord 限制 (25MB)
            if filename.stat().st_size > self.max_file_size:
                file_size_mb = filename.stat().st_size / 1024 / 1024
                await interaction.edit_original_message(
                    content=f"❌ 檔案大小超過 25MB ({file_size_mb:.1f}MB)，無法上傳至 Discord。\n"
                    f"請選擇較低的畫質選項或較短的影片。"
                )
                filename.unlink()  # 刪除檔案
                return

            # 更新訊息並傳送檔案
            file_size_mb = filename.stat().st_size / 1024 / 1024
            await interaction.edit_original_message(
                content=f"✅ 下載成功! 檔案大小: {file_size_mb:.1f}MB\n{title}",
                file=nextcord.File(str(filename), filename=filename.name),
            )
            filename.unlink()  # 刪除檔案
        except Exception as e:
            # 發生錯誤時更新原始訊息
            await interaction.edit_original_message(content=f"❌ 下載失敗: {e}")


# 註冊 Cog
async def setup(bot: commands.Bot) -> None:
    bot.add_cog(VideoCogs(bot), override=True)
