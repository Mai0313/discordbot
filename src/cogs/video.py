import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands

from src.utils.downloader import VideoDownloader


class VideoCogs(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @nextcord.slash_command(
        name="download_video",
        description="Download a video from various platforms and send it back.",
        name_localizations={
            Locale.zh_TW: "下載影片",
            Locale.zh_CN: "下载视频",
            Locale.ja: "動画ダウンロード",
        },
        description_localizations={
            Locale.zh_TW: "從多種平台下載影片並傳送 (支援 YouTube, Facebook, Instagram, X, Tiktok 等)。",
            Locale.zh_CN: "从多种平台下载视频并发送 (支持 YouTube, Facebook, Instagram, X, Tiktok 等)。",
            Locale.ja: "YouTube, Facebook, Instagram, X, Tiktok などから動画をダウンロードして送信します。",
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
        await interaction.followup.send("🔄 正在下載影片，請稍候...")

        try:
            await interaction.edit_original_message(content="⏳ 正在下載...")
            title, filename = VideoDownloader(output_folder="./data/downloads").download(
                url=url, quality=quality
            )

            # 檢查檔案大小是否超過 Discord 限制 (25MB)
            file_size_mb = filename.stat().st_size / 1024 / 1024
            if filename.stat().st_size > 25 * 1024 * 1024:
                link = f"https://mai0313.com/drive/d/share/{filename.name}"
                embed = nextcord.Embed(title=title, description=f"{file_size_mb:.1f}MB", url=link)
                await interaction.edit_original_message(content="✅ 下載成功!", embed=embed)
            else:
                await interaction.edit_original_message(
                    content=f"✅ 下載成功! 檔案大小: {file_size_mb:.1f}MB\n{title}",
                    file=nextcord.File(str(filename), filename=filename.name),
                )
        except Exception:
            embed = nextcord.Embed(
                title="操", description="自己點開來看啦白癡 你媽沒給你生手喔", url=url
            )
            await interaction.edit_original_message(content=f"❌ 下載失敗\n{url}", embed=embed)


# 註冊 Cog
async def setup(bot: commands.Bot) -> None:
    bot.add_cog(VideoCogs(bot), override=True)
