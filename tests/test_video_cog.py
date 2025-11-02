from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import pytest
import nextcord

from discordbot.cogs.video import VideoCogs


class _StatResult:
    """Mock stat result with st_size attribute."""

    st_size = 10 * 1024 * 1024  # 10MB


@pytest.mark.asyncio
@patch("discordbot.cogs.video.VideoDownloader")
async def test_download_video_happy_path(mock_downloader_cls: Mock) -> None:
    # Prepare downloader mock
    instance = Mock()

    # Use a relative path to avoid S108 temp path rule
    tmp = Path("data/downloads/test.mp4")

    class F:
        def __init__(self, p: Path) -> None:
            self._p = p

        def stat(self) -> _StatResult:
            return _StatResult()

        @property
        def name(self) -> str:
            return self._p.name

        def __str__(self) -> str:
            return str(self._p)

    instance.download.return_value = ("title", F(tmp))
    mock_downloader_cls.return_value = instance

    # interaction mocks
    interaction = Mock(spec=nextcord.Interaction)
    interaction.response = Mock()
    interaction.response.defer = AsyncMock()
    interaction.followup = Mock()
    interaction.followup.send = AsyncMock()
    interaction.edit_original_message = AsyncMock()

    cog = VideoCogs(bot=Mock())

    await cog.download_video(interaction, url="https://example.com", quality="best")

    interaction.response.defer.assert_awaited()
    interaction.followup.send.assert_awaited()
    interaction.edit_original_message.assert_awaited()
    instance.download.assert_called_once()
