"""音樂播放 Cog 測試"""

from unittest.mock import Mock, AsyncMock, MagicMock, patch

import pytest
import nextcord
from src.cogs.music import MusicCogs, YTDLSource


class TestMusicCogs:
    """音樂 Cog 測試類別"""

    @pytest.fixture
    def bot(self):
        """創建模擬機器人"""
        bot = MagicMock()
        bot.loop = AsyncMock()
        return bot

    @pytest.fixture
    def music_cog(self, bot):
        """創建音樂 Cog 實例"""
        return MusicCogs(bot)

    @pytest.fixture
    def mock_interaction(self):
        """創建模擬互動"""
        interaction = MagicMock()
        interaction.response = AsyncMock()
        interaction.followup = AsyncMock()
        interaction.user = MagicMock()
        interaction.guild = MagicMock()
        interaction.guild.voice_client = None
        return interaction

    def test_music_cog_initialization(self, bot):
        """測試音樂 Cog 初始化"""
        cog = MusicCogs(bot)
        assert cog.bot == bot

    @pytest.mark.asyncio
    async def test_join_command_no_voice_channel(self, music_cog, mock_interaction):
        """測試加入指令 - 使用者不在語音頻道"""
        mock_interaction.user.voice = None

        await music_cog.join(mock_interaction)

        # 驗證有發送錯誤訊息
        mock_interaction.response.defer.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        # 檢查錯誤訊息內容
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "錯誤" in embed.title
        assert embed.color.value == 0xFF0000

    @pytest.mark.asyncio
    async def test_join_command_with_user_in_voice(self, music_cog, mock_interaction):
        """測試加入指令 - 使用者在語音頻道中"""
        mock_channel = MagicMock()
        mock_channel.mention = "#test-channel"
        mock_channel.connect = AsyncMock()

        mock_interaction.user.voice = MagicMock()
        mock_interaction.user.voice.channel = mock_channel

        await music_cog.join(mock_interaction)

        # 驗證有連接到頻道
        mock_channel.connect.assert_called_once()
        mock_interaction.response.defer.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        # 檢查成功訊息
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "已加入" in embed.title
        assert embed.color.value == 0x00FF00

    @pytest.mark.asyncio
    async def test_join_command_already_in_same_channel(self, music_cog, mock_interaction):
        """測試加入指令 - 機器人已經在相同頻道中"""
        mock_channel = MagicMock()
        mock_channel.mention = "#test-channel"

        mock_interaction.user.voice = MagicMock()
        mock_interaction.user.voice.channel = mock_channel

        mock_voice_client = MagicMock()
        mock_voice_client.channel = mock_channel
        mock_interaction.guild.voice_client = mock_voice_client

        await music_cog.join(mock_interaction)

        # 驗證沒有重新連接
        mock_interaction.response.defer.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        # 檢查提示訊息
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "提示" in embed.title
        assert embed.color.value == 0x0099FF

    @pytest.mark.asyncio
    async def test_stop_command_no_voice_client(self, music_cog, mock_interaction):
        """測試停止指令 - 機器人未連接語音頻道"""
        mock_interaction.guild.voice_client = None

        await music_cog.stop(mock_interaction)

        # 驗證有發送錯誤訊息
        mock_interaction.response.defer.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        # 檢查錯誤訊息內容
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "錯誤" in embed.title
        assert embed.color.value == 0xFF0000

    @pytest.mark.asyncio
    async def test_stop_command_with_voice_client(self, music_cog, mock_interaction):
        """測試停止指令 - 機器人已連接語音頻道"""
        mock_voice_client = AsyncMock()
        mock_interaction.guild.voice_client = mock_voice_client

        await music_cog.stop(mock_interaction)

        # 驗證有斷開連接
        mock_voice_client.disconnect.assert_called_once()
        mock_interaction.response.defer.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        # 檢查成功訊息
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "已停止" in embed.title
        assert embed.color.value == 0x00FF00

    @pytest.mark.asyncio
    async def test_volume_command_no_voice_client(self, music_cog, mock_interaction):
        """測試音量指令 - 機器人未連接語音頻道"""
        mock_interaction.guild.voice_client = None

        await music_cog.volume(mock_interaction, 50)

        # 驗證有發送錯誤訊息
        mock_interaction.response.defer.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        # 檢查錯誤訊息內容
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "錯誤" in embed.title
        assert embed.color.value == 0xFF0000

    @pytest.mark.asyncio
    async def test_volume_command_with_voice_client(self, music_cog, mock_interaction):
        """測試音量指令 - 機器人已連接語音頻道"""
        mock_voice_client = MagicMock()
        mock_source = MagicMock()
        mock_voice_client.source = mock_source
        mock_interaction.guild.voice_client = mock_voice_client

        await music_cog.volume(mock_interaction, 75)

        # 驗證音量已設定
        assert mock_source.volume == 0.75
        mock_interaction.response.defer.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        # 檢查成功訊息
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "音量調整" in embed.title
        assert "75%" in embed.description
        assert embed.color.value == 0x00FF00

    @pytest.mark.asyncio
    async def test_pause_command_no_music(self, music_cog, mock_interaction):
        """測試暫停指令 - 沒有播放音樂"""
        mock_voice_client = MagicMock()
        mock_voice_client.is_playing.return_value = False
        mock_interaction.guild.voice_client = mock_voice_client

        await music_cog.pause(mock_interaction)

        # 驗證有發送錯誤訊息
        mock_interaction.response.defer.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        # 檢查錯誤訊息內容
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "錯誤" in embed.title
        assert embed.color.value == 0xFF0000

    @pytest.mark.asyncio
    async def test_pause_command_with_music(self, music_cog, mock_interaction):
        """測試暫停指令 - 正在播放音樂"""
        mock_voice_client = MagicMock()
        mock_voice_client.is_playing.return_value = True
        mock_interaction.guild.voice_client = mock_voice_client

        await music_cog.pause(mock_interaction)

        # 驗證音樂已暫停
        mock_voice_client.pause.assert_called_once()
        mock_interaction.response.defer.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        # 檢查成功訊息
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "已暫停" in embed.title
        assert embed.color.value == 0x00FF00

    @pytest.mark.asyncio
    async def test_resume_command_no_paused_music(self, music_cog, mock_interaction):
        """測試繼續指令 - 沒有暫停的音樂"""
        mock_voice_client = MagicMock()
        mock_voice_client.is_paused.return_value = False
        mock_interaction.guild.voice_client = mock_voice_client

        await music_cog.resume(mock_interaction)

        # 驗證有發送錯誤訊息
        mock_interaction.response.defer.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        # 檢查錯誤訊息內容
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "錯誤" in embed.title
        assert embed.color.value == 0xFF0000

    @pytest.mark.asyncio
    async def test_resume_command_with_paused_music(self, music_cog, mock_interaction):
        """測試繼續指令 - 有暫停的音樂"""
        mock_voice_client = MagicMock()
        mock_voice_client.is_paused.return_value = True
        mock_interaction.guild.voice_client = mock_voice_client

        await music_cog.resume(mock_interaction)

        # 驗證音樂已繼續
        mock_voice_client.resume.assert_called_once()
        mock_interaction.response.defer.assert_called_once()
        mock_interaction.followup.send.assert_called_once()

        # 檢查成功訊息
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "已繼續" in embed.title
        assert embed.color.value == 0x00FF00

    @pytest.mark.asyncio
    async def test_ensure_voice_no_user_voice(self, music_cog, mock_interaction):
        """測試確保語音連接 - 使用者不在語音頻道"""
        mock_interaction.guild.voice_client = None
        mock_interaction.user.voice = None

        result = await music_cog._ensure_voice(mock_interaction)  # noqa: SLF001

        assert result is False
        mock_interaction.followup.send.assert_called_once()

        # 檢查錯誤訊息
        call_args = mock_interaction.followup.send.call_args
        embed = call_args[1]["embed"]
        assert "錯誤" in embed.title
        assert embed.color.value == 0xFF0000

    @pytest.mark.asyncio
    async def test_ensure_voice_user_in_voice(self, music_cog, mock_interaction):
        """測試確保語音連接 - 使用者在語音頻道"""
        mock_interaction.guild.voice_client = None
        mock_voice = MagicMock()
        mock_channel = AsyncMock()
        mock_voice.channel = mock_channel
        mock_interaction.user.voice = mock_voice

        result = await music_cog._ensure_voice(mock_interaction)  # noqa: SLF001

        assert result is True
        mock_channel.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_voice_already_connected_and_playing(self, music_cog, mock_interaction):
        """測試確保語音連接 - 機器人已連接且正在播放"""
        mock_voice_client = MagicMock()
        mock_voice_client.is_playing.return_value = True
        mock_interaction.guild.voice_client = mock_voice_client

        result = await music_cog._ensure_voice(mock_interaction)  # noqa: SLF001

        assert result is True
        mock_voice_client.stop.assert_called_once()


class TestYTDLSource:
    """YouTube 音源測試類別"""

    def test_ytdl_source_initialization(self):
        """測試 YTDL 音源初始化"""
        # 使用真實的 AudioSource mock
        with patch("nextcord.FFmpegPCMAudio"):
            mock_source = Mock(spec=nextcord.AudioSource)
            # 設定 is_opus() 返回 False，避免 PCMVolumeTransformer 錯誤
            mock_source.is_opus.return_value = False
            # 添加 cleanup 方法
            mock_source.cleanup = Mock()
            mock_data = {
                "title": "Test Song",
                "url": "https://example.com/test.mp3",
                "duration": 180,
                "uploader": "Test Channel",
            }

            ytdl_source = YTDLSource(mock_source, data=mock_data)

            assert ytdl_source.title == "Test Song"
            assert ytdl_source.url == "https://example.com/test.mp3"
            assert ytdl_source.duration == 180
            assert ytdl_source.uploader == "Test Channel"

    @pytest.mark.asyncio
    @patch("src.cogs.music.ytdl")
    @patch("nextcord.FFmpegPCMAudio")
    async def test_ytdl_source_from_url(self, mock_ffmpeg, mock_ytdl):
        """測試從 URL 創建音源"""
        mock_data = {
            "title": "Test Song",
            "url": "https://example.com/test.mp3",
            "duration": 180,
            "uploader": "Test Channel",
        }
        mock_ytdl.extract_info.return_value = mock_data
        mock_ytdl.prepare_filename.return_value = "./data/test.mp3"

        # 創建一個實際的 AudioSource mock
        mock_audio_source = Mock(spec=nextcord.AudioSource)
        # 設定 is_opus() 返回 False，避免 PCMVolumeTransformer 錯誤
        mock_audio_source.is_opus.return_value = False
        # 添加 cleanup 方法
        mock_audio_source.cleanup = Mock()
        mock_ffmpeg.return_value = mock_audio_source

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_data)

            source = await YTDLSource.from_url("https://youtube.com/watch?v=test")

            assert source.title == "Test Song"
            assert source.url == "https://example.com/test.mp3"
            assert source.duration == 180
            assert source.uploader == "Test Channel"

    @pytest.mark.asyncio
    @patch("src.cogs.music.ytdl")
    @patch("nextcord.FFmpegPCMAudio")
    async def test_ytdl_source_from_url_playlist(self, mock_ffmpeg, mock_ytdl):
        """測試從播放清單 URL 創建音源"""
        mock_data = {
            "entries": [
                {
                    "title": "First Song",
                    "url": "https://example.com/first.mp3",
                    "duration": 120,
                    "uploader": "Test Channel",
                },
                {
                    "title": "Second Song",
                    "url": "https://example.com/second.mp3",
                    "duration": 180,
                    "uploader": "Test Channel",
                },
            ]
        }
        mock_ytdl.extract_info.return_value = mock_data
        mock_ytdl.prepare_filename.return_value = "./data/first.mp3"

        # 創建一個實際的 AudioSource mock
        mock_audio_source = Mock(spec=nextcord.AudioSource)
        # 設定 is_opus() 返回 False，避免 PCMVolumeTransformer 錯誤
        mock_audio_source.is_opus.return_value = False
        # 添加 cleanup 方法
        mock_audio_source.cleanup = Mock()
        mock_ffmpeg.return_value = mock_audio_source

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_data)

            source = await YTDLSource.from_url("https://youtube.com/playlist?list=test")

            # 應該使用播放清單的第一首歌
            assert source.title == "First Song"
            assert source.url == "https://example.com/first.mp3"
            assert source.duration == 120
            assert source.uploader == "Test Channel"
