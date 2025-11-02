from unittest.mock import Mock, AsyncMock

import pytest
import nextcord

from discordbot.cogs.gen_image import ImageGeneratorCogs


@pytest.mark.asyncio
async def test_graph_flow_defer_and_edit_message() -> None:
    cog = ImageGeneratorCogs(bot=Mock())

    interaction = Mock(spec=nextcord.Interaction)
    interaction.response = Mock()
    interaction.response.defer = AsyncMock()
    interaction.followup = Mock()
    interaction.followup.send = AsyncMock()
    interaction.edit_original_message = AsyncMock()

    await cog.graph(interaction, prompt="hi")

    interaction.response.defer.assert_awaited()
    interaction.followup.send.assert_awaited()
    interaction.edit_original_message.assert_awaited()
