"""語音連接工具模組"""

import time
from typing import Optional
import asyncio
import logging
from pathlib import Path

import nextcord

logger = logging.getLogger(__name__)


class VoiceRecorder:
    """語音連接管理器

    負責語音頻道連接和狀態管理
    注意：nextcord 目前不支援內建語音錄音功能
    """

    def __init__(self, output_folder: str = "./data/recordings") -> None:
        """初始化語音管理器

        Args:
            output_folder (str): 錄音文件輸出資料夾路徑（暫時保留以備將來使用）
        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.is_connected = False
        self.voice_client: Optional[nextcord.VoiceClient] = None
        self.start_time: Optional[float] = None
        self._disconnect_task: Optional[asyncio.Task] = None

    async def join_voice_channel(
        self,
        channel: nextcord.VoiceChannel,
        max_duration: int = 300,  # 預設最長連接5分鐘
    ) -> nextcord.VoiceClient:
        """加入語音頻道

        Args:
            channel: 要加入的語音頻道
            max_duration: 最長連接時間（秒）

        Returns:
            nextcord.VoiceClient: 語音客戶端
        """
        if self.is_connected:
            raise RuntimeError("已經連接到語音頻道")

        try:
            # 連接到語音頻道
            voice_client = await channel.connect()

            self.voice_client = voice_client
            self.is_connected = True
            self.start_time = time.time()

            logger.info(f"已連接到語音頻道 {channel.name} (Guild: {channel.guild.id})")

            # 設定自動斷開時間
            self._disconnect_task = asyncio.create_task(self._auto_disconnect(max_duration))

            return voice_client

        except Exception as e:
            self.is_connected = False
            self.voice_client = None
            self.start_time = None
            self._disconnect_task = None
            logger.error(f"語音頻道連接失敗: {e}")
            raise

    async def connect_to_voice(
        self,
        voice_client: nextcord.VoiceClient,
        guild_id: int,
        max_duration: int = 300,  # 預設最長連接5分鐘
    ) -> bool:
        """連接到語音頻道（使用現有的語音客戶端）

        Args:
            voice_client: Discord 語音客戶端
            guild_id: 伺服器 ID
            max_duration: 最長連接時間（秒）

        Returns:
            bool: 是否成功連接
        """
        if self.is_connected:
            raise RuntimeError("已經連接到語音頻道")

        try:
            self.voice_client = voice_client
            self.is_connected = True
            self.start_time = time.time()

            logger.info(f"已連接到語音頻道 (Guild: {guild_id})")

            # 設定自動斷開時間
            self._disconnect_task = asyncio.create_task(self._auto_disconnect(max_duration))

            return True

        except Exception as e:
            self.is_connected = False
            self.voice_client = None
            self.start_time = None
            self._disconnect_task = None
            logger.error(f"語音頻道連接失敗: {e}")
            raise
        """離開語音頻道

        Returns:
            bool: 是否成功離開
        """
        return await self.disconnect_from_voice()

    async def disconnect_from_voice(self) -> bool:
        """斷開語音頻道連接

        Returns:
            bool: 是否成功斷開
        """
        if not self.is_connected or not self.voice_client:
            return False

        try:
            # 取消自動斷開任務
            if self._disconnect_task and not self._disconnect_task.done():
                self._disconnect_task.cancel()

            # 斷開連接
            await self.voice_client.disconnect()
            self.is_connected = False
            self.voice_client = None
            self.start_time = None
            self._disconnect_task = None

            logger.info("已斷開語音頻道連接")
            return True

        except Exception as e:
            logger.error(f"語音頻道斷開失敗: {e}")
            return False

    def get_connection_duration(self) -> int:
        """取得目前連接時長（秒）

        Returns:
            int: 連接時長，若未連接則返回 0
        """
        if not self.is_connected or not self.start_time:
            return 0
        return int(time.time() - self.start_time)

    async def _auto_disconnect(self, max_duration: int) -> None:
        """自動斷開連接

        Args:
            max_duration: 最長連接時間（秒）
        """
        try:
            await asyncio.sleep(max_duration)
            if self.is_connected:
                await self.disconnect_from_voice()
                logger.info(f"連接達到最長時間 {max_duration} 秒，自動斷開")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"自動斷開失敗: {e}")

    def is_voice_connected(self) -> bool:
        """檢查是否已連接到語音頻道

        Returns:
            bool: 是否已連接
        """
        return self.is_connected and self.voice_client is not None

    def get_status(self) -> dict:
        """取得語音連接狀態資訊

        Returns:
            dict: 包含連接狀態資訊的字典
        """
        return {
            "is_connected": self.is_connected,
            "duration": self.get_connection_duration(),
            "channel": self.voice_client.channel.name if self.voice_client else None,
            "guild": self.voice_client.guild.name if self.voice_client else None,
        }
