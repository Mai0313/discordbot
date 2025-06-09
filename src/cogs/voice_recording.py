"""語音連接功能模組"""

import logging

import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands

from src.utils.voice_recorder import VoiceRecorder

logger = logging.getLogger(__name__)

# SlashOption definitions to avoid B008 ruff error
CHANNEL_OPTION = SlashOption(
    name="channel",
    description="Voice channel to join (optional, defaults to your current channel)",
    name_localizations={Locale.zh_TW: "頻道", Locale.ja: "チャンネル"},
    description_localizations={
        Locale.zh_TW: "要加入的語音頻道（可選，預設為你目前的頻道）",
        Locale.ja: "参加するボイスチャンネル（オプション、デフォルトは現在のチャンネル）",
    },
    required=False,
    default=None,
)

MAX_DURATION_OPTION = SlashOption(
    name="max_duration",
    description="Maximum connection duration in minutes (1-60, default: 5)",
    name_localizations={Locale.zh_TW: "最長時間", Locale.ja: "最大時間"},
    description_localizations={
        Locale.zh_TW: "最長連接時間（分鐘，1-60，預設: 5）",
        Locale.ja: "最大接続時間（分、1-60、デフォルト: 5）",
    },
    required=False,
    default=5,
    min_value=1,
    max_value=60,
)


class VoiceRecordingCogs(commands.Cog):
    """語音連接功能（注意：nextcord 不支援內建錄音）"""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.voice_recorders: dict[int, VoiceRecorder] = {}  # 每個伺服器一個連接管理器

    def get_recorder(self, guild_id: int) -> VoiceRecorder:
        """取得指定伺服器的語音連接管理器

        Args:
            guild_id: 伺服器 ID

        Returns:
            VoiceRecorder: 語音連接管理器實例
        """
        if guild_id not in self.voice_recorders:
            self.voice_recorders[guild_id] = VoiceRecorder()
        return self.voice_recorders[guild_id]

    @nextcord.slash_command(
        name="voice_join",
        description="Join voice channel",
        name_localizations={Locale.zh_TW: "加入語音", Locale.ja: "ボイス参加"},
        description_localizations={
            Locale.zh_TW: "加入語音頻道",
            Locale.ja: "ボイスチャンネルに参加",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def voice_join(
        self,
        interaction: Interaction,
        channel: nextcord.VoiceChannel = CHANNEL_OPTION,
        max_duration: int = MAX_DURATION_OPTION,
    ) -> None:
        """加入語音頻道"""
        await interaction.response.defer()

        # 檢查是否在伺服器中執行
        if not interaction.guild:
            embed = nextcord.Embed(
                title="❌ 錯誤", description="此指令只能在伺服器中使用!", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        # 確定目標語音頻道
        target_channel = channel
        if not target_channel:
            # 如果沒有指定頻道，嘗試使用用戶當前的語音頻道
            if (
                isinstance(interaction.user, nextcord.Member)
                and interaction.user.voice
                and interaction.user.voice.channel
            ):
                target_channel = interaction.user.voice.channel
            else:
                embed = nextcord.Embed(
                    title="❌ 錯誤",
                    description="你必須先加入一個語音頻道，或在指令中指定頻道!",
                    color=0xFF0000,
                )
                await interaction.followup.send(embed=embed)
                return

        # 檢查權限
        permissions = target_channel.permissions_for(interaction.guild.me)
        if not permissions.connect or not permissions.speak:
            embed = nextcord.Embed(
                title="❌ 權限不足", description="機器人沒有加入該語音頻道的權限!", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        # 檢查機器人是否已經在語音頻道中
        if interaction.guild.voice_client:
            if interaction.guild.voice_client.channel == target_channel:
                embed = nextcord.Embed(
                    title="⚠️ 警告", description="機器人已經在這個語音頻道中!", color=0xFFAA00
                )
                await interaction.followup.send(embed=embed)
                return
            # 離開目前的語音頻道
            await interaction.guild.voice_client.disconnect()

        try:
            # 取得語音連接管理器
            recorder = self.get_recorder(interaction.guild.id)

            # 加入語音頻道
            await recorder.join_voice_channel(target_channel)

            embed = nextcord.Embed(
                title="🎙️ 語音連接成功",
                description=f"✅ 已加入 {target_channel.mention}\n"
                f"⏱️ 最長連接時間: {max_duration} 分鐘\n"
                f"👥 頻道成員: {len(target_channel.members)} 人\n\n"
                f"**注意：nextcord 目前不支援內建錄音功能**\n"
                f"此功能僅提供語音頻道連接\n\n"
                f"使用 `/voice_stop` 離開頻道",
                color=0x00FF00,
            )
            embed.set_footer(text=f"由 {interaction.user.display_name} 啟動")

            await interaction.followup.send(embed=embed)
            logger.info(
                f"用戶 {interaction.user} 在 {interaction.guild} 加入語音頻道: {target_channel.name}"
            )

        except RuntimeError as e:
            embed = nextcord.Embed(title="❌ 錯誤", description=f"連接失敗: {e!s}", color=0xFF0000)
            await interaction.followup.send(embed=embed)
        except Exception as e:
            embed = nextcord.Embed(
                title="❌ 錯誤",
                description=f"加入語音頻道時發生錯誤:\n```{e!s}```",
                color=0xFF0000,
            )
            await interaction.followup.send(embed=embed)
            logger.error(f"語音連接啟動錯誤: {e}")

    @nextcord.slash_command(
        name="voice_stop",
        description="Leave voice channel",
        name_localizations={Locale.zh_TW: "離開語音", Locale.ja: "ボイス退出"},
        description_localizations={
            Locale.zh_TW: "離開語音頻道",
            Locale.ja: "ボイスチャンネルから退出",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def voice_stop(self, interaction: Interaction) -> None:
        """離開語音頻道"""
        await interaction.response.defer()

        # 檢查是否在伺服器中執行
        if not interaction.guild:
            embed = nextcord.Embed(
                title="❌ 錯誤", description="此指令只能在伺服器中使用!", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        # 檢查機器人是否在語音頻道中
        if not interaction.guild.voice_client:
            embed = nextcord.Embed(
                title="⚠️ 警告", description="機器人目前不在任何語音頻道中!", color=0xFFAA00
            )
            await interaction.followup.send(embed=embed)
            return

        try:
            # 取得語音連接管理器
            recorder = self.get_recorder(interaction.guild.id)

            # 取得連接時長
            duration = recorder.get_connection_duration()
            channel_name = interaction.guild.voice_client.channel.name

            # 離開語音頻道
            success = await recorder.leave_voice_channel()

            if success:
                embed = nextcord.Embed(
                    title="🎙️ 語音連接結束",
                    description=f"✅ 已離開語音頻道: {channel_name}\n"
                    f"⏱️ 連接時長: {duration // 60:02d}:{duration % 60:02d}",
                    color=0x00FF00,
                )
            else:
                embed = nextcord.Embed(
                    title="⚠️ 警告",
                    description="離開語音頻道時出現問題，但已嘗試斷開連接",
                    color=0xFFAA00,
                )

            embed.set_footer(text=f"由 {interaction.user.display_name} 停止")
            await interaction.followup.send(embed=embed)
            logger.info(f"用戶 {interaction.user} 在 {interaction.guild} 離開語音頻道")

        except Exception as e:
            embed = nextcord.Embed(
                title="❌ 錯誤",
                description=f"離開語音頻道時發生錯誤:\n```{e!s}```",
                color=0xFF0000,
            )
            await interaction.followup.send(embed=embed)
            logger.error(f"語音離開錯誤: {e}")

    @nextcord.slash_command(
        name="voice_status",
        description="Check current voice connection status",
        name_localizations={Locale.zh_TW: "語音狀態", Locale.ja: "ボイス状態"},
        description_localizations={
            Locale.zh_TW: "查看目前的語音連接狀態",
            Locale.ja: "現在のボイス接続状態を確認",
        },
        dm_permission=False,
        nsfw=False,
    )
    async def voice_status(self, interaction: Interaction) -> None:
        """查看語音連接狀態"""
        await interaction.response.defer()

        # 檢查是否在伺服器中執行
        if not interaction.guild:
            embed = nextcord.Embed(
                title="❌ 錯誤", description="此指令只能在伺服器中使用!", color=0xFF0000
            )
            await interaction.followup.send(embed=embed)
            return

        # 檢查機器人語音狀態
        voice_client = interaction.guild.voice_client
        recorder = self.get_recorder(interaction.guild.id)
        status = recorder.get_status()

        if voice_client and voice_client.is_connected():
            duration = status["duration"]
            embed = nextcord.Embed(
                title="🎙️ 語音連接狀態",
                description=f"🟢 已連接到語音頻道\n"
                f"📍 頻道: {voice_client.channel.mention}\n"
                f"⏱️ 連接時長: {duration // 60:02d}:{duration % 60:02d}\n"
                f"👥 頻道成員: {len(voice_client.channel.members)} 人\n\n"
                f"**功能說明：**\n"
                f"• nextcord 不支援內建錄音功能\n"
                f"• 此機器人僅提供語音頻道連接",
                color=0x00FF00,
            )
        else:
            embed = nextcord.Embed(
                title="🎙️ 語音連接狀態", description="⚪ 未連接到任何語音頻道", color=0x808080
            )

        embed.set_footer(text=f"查詢者: {interaction.user.display_name}")
        await interaction.followup.send(embed=embed)

    @commands.Cog.listener()
    async def on_voice_state_update(
        self, member: nextcord.Member, before: nextcord.VoiceState, after: nextcord.VoiceState
    ) -> None:
        """處理語音狀態變更事件"""
        # 如果機器人被斷開連接，自動清理連接狀態
        if member == self.bot.user and before.channel and not after.channel:
            guild_id = before.channel.guild.id
            if guild_id in self.voice_recorders:
                recorder = self.voice_recorders[guild_id]
                if recorder.is_connected:
                    try:
                        await recorder.leave_voice_channel()
                        logger.info(f"機器人被斷開連接，自動清理連接狀態 (Guild: {guild_id})")
                    except Exception as e:
                        logger.error(f"自動清理連接狀態失敗: {e}")


# 註冊 Cog
def setup(bot: commands.Bot) -> None:
    bot.add_cog(VoiceRecordingCogs(bot), override=True)
