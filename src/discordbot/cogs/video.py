import contextlib

import logfire
import nextcord
from nextcord import File, Locale, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.utils.downloader import VideoDownloader
from discordbot.cogs._economy.database import add_balance

_BASE_REWARD = 10
_REWARD_CAP = 100

# Hard-cap aligned with the unboosted Discord upload limit. The downloader
# already retries at low quality once when this is exceeded.
_DISCORD_FILE_LIMIT_BYTES = 25 * 1024 * 1024


def _reward_for(file_size_mb: float) -> int:
    """Calculates the point reward for a successful download.

    Base reward plus one point per MB downloaded, capped at ``_REWARD_CAP``
    so a single 1080p clip can't drain the leaderboard.
    """
    return min(_BASE_REWARD + round(file_size_mb), _REWARD_CAP)


async def _award_silently(*, user_id: int, name: str, amount: int) -> int | None:
    """Persists the reward and returns the new balance; logs and returns ``None`` on DB failure."""
    try:
        return await add_balance(user_id=user_id, name=name, amount=amount)
    except Exception:
        logfire.warn("Failed to award video download points", _exc_info=True)
        return None


def _success_text(file_size_mb: float, reward: int | None) -> str:
    """Builds the final message body, appending the reward suffix only on a real DB write."""
    base = f"✅ 下載成功! 檔案大小: {file_size_mb:.1f}MB"
    if reward is None:
        return base
    return f"{base} · 獲得 {reward:,} 點數"


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
            Locale.zh_TW: "從多種平台下載影片並傳送 (支援 YouTube, Facebook, Instagram, X, Tiktok 等)。",
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

        The status message (downloading / re-encoding / failure) lives on the
        deferred placeholder via ``edit_original_message``; the final video
        file is delivered as a new followup so we never have to mix
        ``content=`` and ``file=`` in a single edit — that combination
        sometimes drops the new content on Discord's side, which is why the
        reward suffix was vanishing on success.

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
        self, *, interaction: Interaction, file_size_mb: float, file_path: str, file_name: str
    ) -> None:
        """Awards points, then sends the file as a fresh followup with the reward suffix.

        We deliberately do NOT call ``edit_original_message(file=…, content=…)``
        here — that combination has dropped the ``content`` (i.e. the reward
        suffix) on the Discord side when the multipart file payload is
        attached. Sending the result as a separate followup keeps the
        ``content`` reliable, and the placeholder is collapsed away by the
        final ``edit_original_message`` call.
        """
        reward = _reward_for(file_size_mb=file_size_mb)
        awarded: int | None = None
        if interaction.user is not None:
            awarded = await _award_silently(
                user_id=interaction.user.id, name=interaction.user.name, amount=reward
            )
        # Show only what we actually persisted: omit the reward suffix when
        # the DB write failed, so we never promise points the user can't see.
        body = _success_text(
            file_size_mb=file_size_mb, reward=reward if awarded is not None else None
        )
        await interaction.followup.send(content=body, file=File(fp=file_path, filename=file_name))
        # Collapse the "downloading..." placeholder into a brief checkmark so
        # the channel doesn't keep two near-identical bot messages around.
        with contextlib.suppress(Exception):
            await interaction.edit_original_message(content="✅")


# 註冊 Cog
def setup(bot: commands.Bot) -> None:
    """Adds the VideoCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(VideoCogs(bot), override=True)
