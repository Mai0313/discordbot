"""YouTube 音樂播放功能模組"""

from typing import Any, Optional
import asyncio
import logging

import yt_dlp
import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands

logger = logging.getLogger(__name__)

# 抑制 yt-dlp 的錯誤報告訊息
yt_dlp.utils.bug_reports_message = lambda: ""

# yt-dlp 配置選項
YTDL_FORMAT_OPTIONS = {
    "format": "bestaudio/best",
    "outtmpl": "%(extractor)s-%(id)s-%(title)s.%(ext)s",
    "restrictfilenames": True,
    "noplaylist": True,
    "nocheckcertificate": True,
    "ignoreerrors": False,
    "logtostderr": False,
    "quiet": True,
    "no_warnings": True,
    "default_search": "auto",
    "source_address": "0.0.0.0",  # noqa: S104
}

# FFmpeg 配置選項
FFMPEG_OPTIONS = {"options": "-vn"}

ytdl = yt_dlp.YoutubeDL(YTDL_FORMAT_OPTIONS)


class YTDLSource(nextcord.PCMVolumeTransformer):
    """YouTube 音源處理器"""

    def __init__(
        self, source: nextcord.AudioSource, *, data: dict[str, Any], volume: float = 0.5
    ) -> None:
        super().__init__(source, volume)
        self.data = data
        self.title = data.get("title")
        self.url = data.get("url")
        self.duration = data.get("duration")
        self.uploader = data.get("uploader")

    def cleanup(self) -> None:
        """清理音源資源"""
        try:
            if hasattr(self, "original") and hasattr(self.original, "cleanup"):
                self.original.cleanup()
            super().cleanup()
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")

    @classmethod
    async def from_url(
        cls, url: str, *, loop: Optional[asyncio.AbstractEventLoop] = None, stream: bool = False
    ) -> "YTDLSource":
        """從 URL 創建音源

        Args:
            url: YouTube URL 或搜尋關鍵字
            loop: 事件循環
            stream: 是否使用串流模式

        Returns:
            YTDLSource: 音源實例
        """
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, lambda: ytdl.extract_info(url, download=not stream)
        )

        if "entries" in data:
            # 如果是播放清單，取第一個項目
            data = data["entries"][0]

        filename = data["url"] if stream else ytdl.prepare_filename(data)
        return cls(nextcord.FFmpegPCMAudio(filename, **FFMPEG_OPTIONS), data=data)


class MusicCogs(commands.Cog):
    """YouTube 音樂播放功能"""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @nextcord.slash_command(
        name="join",
        description="Join your current voice channel",
        name_localizations={Locale.zh_TW: "加入", Locale.ja: "参加"},
        description_localizations={
            Locale.zh_TW: "加入你目前的語音頻道",
            Locale.ja: "現在のボイスチャンネルに参加",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def join(self, interaction: Interaction) -> None:
        """加入語音頻道"""
        await interaction.response.defer()

        # 檢查用戶是否在語音頻道中
        if not interaction.user.voice or not interaction.user.voice.channel:
            embed = nextcord.Embed(
                title="❌ 錯誤",
                description="你必須先加入一個語音頻道才能使用此指令",
                color=0xFF0000,
            )
            await interaction.followup.send(embed=embed)
            return

        target_channel = interaction.user.voice.channel

        # 如果已經連接到語音頻道，移動到新頻道
        if interaction.guild.voice_client is not None:
            if interaction.guild.voice_client.channel == target_channel:
                embed = nextcord.Embed(
                    title="❗ 提示",
                    description=f"我已經在 {target_channel.mention} 中了",
                    color=0x0099FF,
                )
            else:
                await interaction.guild.voice_client.move_to(target_channel)
                embed = nextcord.Embed(
                    title="🎵 已移動",
                    description=f"已移動到 {target_channel.mention}",
                    color=0x00FF00,
                )
        else:
            await target_channel.connect()
            embed = nextcord.Embed(
                title="🎵 已加入", description=f"已加入 {target_channel.mention}", color=0x00FF00
            )

        await interaction.followup.send(embed=embed)

    @nextcord.slash_command(
        name="play",
        description="Play music from YouTube",
        name_localizations={Locale.zh_TW: "播放", Locale.ja: "再生"},
        description_localizations={
            Locale.zh_TW: "從 YouTube 播放音樂",
            Locale.ja: "YouTubeから音楽を再生",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def play(
        self,
        interaction: Interaction,
        url: str = SlashOption(
            name="url",
            description="YouTube URL or search query",
            name_localizations={Locale.zh_TW: "網址", Locale.ja: "URL"},
            description_localizations={
                Locale.zh_TW: "YouTube 網址或搜尋關鍵字",
                Locale.ja: "YouTube または検索キーワード",
            },
            required=True,
        ),
    ) -> None:
        """播放 YouTube 音樂"""
        await interaction.response.defer()

        # 確保使用者在語音頻道中
        if not await self._ensure_voice(interaction):
            return

        try:
            # 停止當前播放
            if interaction.guild.voice_client.is_playing():
                interaction.guild.voice_client.stop()

            # 創建音源
            player = await YTDLSource.from_url(url, loop=self.bot.loop, stream=True)

            # 播放音樂
            interaction.guild.voice_client.play(
                player, after=lambda e: logger.error(f"Player error: {e}") if e else None
            )

            embed = nextcord.Embed(
                title="🎵 正在播放", description=f"**{player.title}**", color=0x00FF00
            )
            if player.uploader:
                embed.add_field(name="頻道", value=player.uploader, inline=True)
            if player.duration:
                duration_str = f"{player.duration // 60}:{player.duration % 60:02d}"
                embed.add_field(name="時長", value=duration_str, inline=True)

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"播放音樂時發生錯誤: {e}")
            embed = nextcord.Embed(
                title="❌ 播放失敗", description=f"無法播放音樂: {e!s}", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)

    @nextcord.slash_command(
        name="stream",
        description="Stream music from YouTube (no download)",
        name_localizations={Locale.zh_TW: "串流", Locale.ja: "ストリーム"},
        description_localizations={
            Locale.zh_TW: "從 YouTube 串流音樂（不下載）",
            Locale.ja: "YouTubeから音楽をストリーミング（ダウンロードなし）",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def stream(
        self,
        interaction: Interaction,
        url: str = SlashOption(
            name="url",
            description="YouTube URL or search query",
            name_localizations={Locale.zh_TW: "網址", Locale.ja: "URL"},
            description_localizations={
                Locale.zh_TW: "YouTube 網址或搜尋關鍵字",
                Locale.ja: "YouTube または検索キーワード",
            },
            required=True,
        ),
    ) -> None:
        """串流播放 YouTube 音樂"""
        await interaction.response.defer()

        # 確保使用者在語音頻道中
        if not await self._ensure_voice(interaction):
            return

        try:
            # 停止當前播放
            if interaction.guild.voice_client.is_playing():
                interaction.guild.voice_client.stop()

            # 創建串流音源
            player = await YTDLSource.from_url(url, loop=self.bot.loop, stream=True)

            # 播放音樂
            interaction.guild.voice_client.play(
                player, after=lambda e: logger.error(f"Player error: {e}") if e else None
            )

            embed = nextcord.Embed(
                title="🎵正在串流", description=f"**{player.title}**", color=0x00FF00
            )
            if player.uploader:
                embed.add_field(name="頻道", value=player.uploader, inline=True)
            if player.duration:
                duration_str = f"{player.duration // 60}:{player.duration % 60:02d}"
                embed.add_field(name="時長", value=duration_str, inline=True)

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"串流音樂時發生錯誤: {e}")
            embed = nextcord.Embed(
                title="❌ 串流失敗", description=f"無法串流音樂: {e!s}", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)

    @nextcord.slash_command(
        name="volume",
        description="Change the music volume",
        name_localizations={Locale.zh_TW: "音量", Locale.ja: "ボリューム"},
        description_localizations={Locale.zh_TW: "調整音樂音量", Locale.ja: "音楽の音量を調整"},
        dm_permission=False,
        nsfw=False,
    )
    async def volume(
        self,
        interaction: Interaction,
        volume: int = SlashOption(
            name="volume",
            description="Volume level (0-100)",
            name_localizations={Locale.zh_TW: "音量", Locale.ja: "ボリューム"},
            description_localizations={
                Locale.zh_TW: "音量等級 (0-100)",
                Locale.ja: "ボリュームレベル (0-100)",
            },
            required=True,
            min_value=0,
            max_value=100,
        ),
    ) -> None:
        """調整音量"""
        await interaction.response.defer()

        if interaction.guild.voice_client is None:
            embed = nextcord.Embed(
                title="❌ 錯誤", description="機器人未連接到語音頻道", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        # 設定音量（0-1 範圍）
        volume_level = volume / 100
        interaction.guild.voice_client.source.volume = volume_level

        embed = nextcord.Embed(
            title="🔊 音量調整", description=f"音量已設定為 {volume}%", color=0x00FF00
        )
        await interaction.followup.send(embed=embed)

    @nextcord.slash_command(
        name="stop",
        description="Stop music and disconnect from voice channel",
        name_localizations={Locale.zh_TW: "停止", Locale.ja: "停止"},
        description_localizations={
            Locale.zh_TW: "停止音樂並離開語音頻道",
            Locale.ja: "音楽を停止してボイスチャンネルから退出",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def stop(self, interaction: Interaction) -> None:
        """停止播放並離開語音頻道"""
        await interaction.response.defer()

        if interaction.guild.voice_client is None:
            embed = nextcord.Embed(
                title="❌ 錯誤", description="機器人未連接到語音頻道", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        await interaction.guild.voice_client.disconnect()
        embed = nextcord.Embed(
            title="🛑 已停止", description="已停止播放並離開語音頻道", color=0x00FF00
        )
        await interaction.followup.send(embed=embed)

    @nextcord.slash_command(
        name="pause",
        description="Pause the current music",
        name_localizations={Locale.zh_TW: "暫停", Locale.ja: "一時停止"},
        description_localizations={
            Locale.zh_TW: "暫停當前音樂",
            Locale.ja: "現在の音楽を一時停止",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def pause(self, interaction: Interaction) -> None:
        """暫停播放"""
        await interaction.response.defer()

        if (
            interaction.guild.voice_client is None
            or not interaction.guild.voice_client.is_playing()
        ):
            embed = nextcord.Embed(title="❌ 錯誤", description="目前沒有播放音樂", color=0xFF0000)
            await interaction.followup.send(embed=embed)
            return

        interaction.guild.voice_client.pause()
        embed = nextcord.Embed(title="⏸️ 已暫停", description="音樂播放已暫停", color=0x00FF00)
        await interaction.followup.send(embed=embed)

    @nextcord.slash_command(
        name="resume",
        description="Resume the paused music",
        name_localizations={Locale.zh_TW: "繼續", Locale.ja: "再開"},
        description_localizations={
            Locale.zh_TW: "繼續播放暫停的音樂",
            Locale.ja: "一時停止した音楽を再開",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def resume(self, interaction: Interaction) -> None:
        """繼續播放"""
        await interaction.response.defer()

        if (
            interaction.guild.voice_client is None
            or not interaction.guild.voice_client.is_paused()
        ):
            embed = nextcord.Embed(
                title="❌ 錯誤", description="目前沒有暫停的音樂", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        interaction.guild.voice_client.resume()
        embed = nextcord.Embed(title="▶️ 已繼續", description="音樂播放已繼續", color=0x00FF00)
        await interaction.followup.send(embed=embed)

    async def _ensure_voice(self, interaction: Interaction) -> bool:
        """確保機器人在語音頻道中

        Args:
            interaction: Discord 互動物件

        Returns:
            bool: 是否成功確保語音連接
        """
        if interaction.guild.voice_client is None:
            if interaction.user.voice:
                await interaction.user.voice.channel.connect()
            else:
                embed = nextcord.Embed(
                    title="❌ 錯誤", description="你需要先加入一個語音頻道", color=0xFF0000
                )
                await interaction.followup.send(embed=embed)
                return False
        elif interaction.guild.voice_client.is_playing():
            interaction.guild.voice_client.stop()

        return True


def setup(bot: commands.Bot) -> None:
    """設定 Cog"""
    bot.add_cog(MusicCogs(bot))
