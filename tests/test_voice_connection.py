"""語音連接功能測試"""

import time
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.utils.voice_recorder import VoiceRecorder


class TestVoiceRecorder:
    """VoiceRecorder 測試類別"""

    def setup_method(self):
        """每個測試方法前的設置"""
        self.voice_recorder = VoiceRecorder()

    def test_init(self):
        """測試初始化"""
        assert self.voice_recorder.is_connected is False
        assert self.voice_recorder.voice_client is None
        assert self.voice_recorder.start_time is None

    @pytest.mark.asyncio
    async def test_join_voice_channel_success(self):
        """測試成功加入語音頻道"""
        # 模擬語音頻道和客戶端
        mock_voice_channel = MagicMock()
        mock_voice_channel.name = "測試頻道"
        mock_voice_channel.connect = AsyncMock()
        mock_voice_client = MagicMock()
        mock_voice_channel.connect.return_value = mock_voice_client

        # 測試加入語音頻道
        result = await self.voice_recorder.join_voice_channel(mock_voice_channel)

        # 驗證結果
        assert result == mock_voice_client
        assert self.voice_recorder.is_connected is True
        assert self.voice_recorder.voice_client == mock_voice_client
        assert self.voice_recorder.start_time is not None
        mock_voice_channel.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_join_voice_channel_already_connected(self):
        """測試已經連接時嘗試加入語音頻道"""
        # 設置已連接狀態
        self.voice_recorder.is_connected = True
        self.voice_recorder.voice_client = MagicMock()

        mock_voice_channel = MagicMock()

        # 測試應該拋出例外
        with pytest.raises(RuntimeError, match="已經連接到語音頻道"):
            await self.voice_recorder.join_voice_channel(mock_voice_channel)

    @pytest.mark.asyncio
    async def test_join_voice_channel_connection_error(self):
        """測試連接失敗"""
        mock_voice_channel = MagicMock()
        mock_voice_channel.name = "測試頻道"
        mock_voice_channel.connect = AsyncMock(side_effect=Exception("連接失敗"))

        # 測試連接失敗
        with pytest.raises(Exception, match="連接失敗"):
            await self.voice_recorder.join_voice_channel(mock_voice_channel)

        # 驗證狀態未改變
        assert self.voice_recorder.is_connected is False
        assert self.voice_recorder.voice_client is None

    @pytest.mark.asyncio
    async def test_leave_voice_channel_success(self):
        """測試成功離開語音頻道"""
        # 設置已連接狀態
        mock_voice_client = AsyncMock()
        self.voice_recorder.is_connected = True
        self.voice_recorder.voice_client = mock_voice_client
        self.voice_recorder.start_time = time.time()

        # 測試離開語音頻道
        result = await self.voice_recorder.leave_voice_channel()

        # 驗證結果
        assert result is True
        assert self.voice_recorder.is_connected is False
        assert self.voice_recorder.voice_client is None
        assert self.voice_recorder.start_time is None
        mock_voice_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_leave_voice_channel_not_connected(self):
        """測試未連接時嘗試離開語音頻道"""
        # 測試離開語音頻道
        result = await self.voice_recorder.leave_voice_channel()

        # 驗證結果
        assert result is False

    @pytest.mark.asyncio
    async def test_leave_voice_channel_disconnect_error(self):
        """測試離開語音頻道時發生錯誤"""
        # 設置已連接狀態
        mock_voice_client = AsyncMock()
        mock_voice_client.disconnect.side_effect = Exception("斷開連接失敗")
        self.voice_recorder.is_connected = True
        self.voice_recorder.voice_client = mock_voice_client
        self.voice_recorder.start_time = time.time()

        # 測試離開語音頻道
        result = await self.voice_recorder.leave_voice_channel()

        # 驗證結果
        assert result is False
        mock_voice_client.disconnect.assert_called_once()

    def test_get_connection_duration_connected(self):
        """測試取得連接時長（已連接）"""
        # 設置已連接狀態
        self.voice_recorder.is_connected = True
        self.voice_recorder.start_time = time.time() - 60  # 1分鐘前

        duration = self.voice_recorder.get_connection_duration()

        # 驗證時長大約為60秒（允許小誤差）
        assert 59 <= duration <= 61

    def test_get_connection_duration_not_connected(self):
        """測試取得連接時長（未連接）"""
        duration = self.voice_recorder.get_connection_duration()
        assert duration == 0

    def test_get_status_connected(self):
        """測試取得狀態（已連接）"""
        # 設置已連接狀態
        mock_voice_client = MagicMock()
        mock_channel = MagicMock()
        mock_channel.name = "測試頻道"
        mock_voice_client.channel = mock_channel

        self.voice_recorder.is_connected = True
        self.voice_recorder.voice_client = mock_voice_client
        self.voice_recorder.start_time = time.time() - 30

        status = self.voice_recorder.get_status()

        assert status["is_connected"] is True
        assert status["channel"] == "測試頻道"
        assert 29 <= status["duration"] <= 31

    def test_get_status_not_connected(self):
        """測試取得狀態（未連接）"""
        status = self.voice_recorder.get_status()

        assert status["is_connected"] is False
        assert status["duration"] == 0
        assert status["channel"] is None


@pytest.mark.asyncio
class TestVoiceRecorderIntegration:
    """VoiceRecorder 整合測試"""

    async def test_full_connection_lifecycle(self):
        """測試完整的連接生命週期"""
        voice_recorder = VoiceRecorder()

        # 模擬語音頻道
        mock_voice_channel = MagicMock()
        mock_voice_channel.name = "整合測試頻道"
        mock_voice_channel.connect = AsyncMock()
        mock_voice_client = AsyncMock()
        mock_voice_channel.connect.return_value = mock_voice_client

        # 1. 測試加入語音頻道
        result = await voice_recorder.join_voice_channel(mock_voice_channel)
        assert result == mock_voice_client
        assert voice_recorder.is_connected is True

        # 2. 測試取得狀態
        status = voice_recorder.get_status()
        assert status["is_connected"] is True

        # 3. 等待一段時間以測試時長計算
        await asyncio.sleep(0.1)

        # 4. 測試離開語音頻道
        success = await voice_recorder.leave_voice_channel()
        assert success is True
        assert voice_recorder.is_connected is False

        # 5. 驗證最終狀態
        final_status = voice_recorder.get_status()
        assert final_status["is_connected"] is False
        assert final_status["duration"] == 0
