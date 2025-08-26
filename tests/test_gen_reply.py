import asyncio
from unittest.mock import Mock, AsyncMock

import pytest
import nextcord

from discordbot.cogs.gen_reply import ReplyGeneratorCogs


def test_get_attachment_list_collects_all_types():
    cog = ReplyGeneratorCogs(bot=Mock())

    # Build fake messages
    msg = Mock(spec=nextcord.Message)
    att = Mock()
    att.url = "https://example.com/a.png"
    embed = Mock()
    embed.description = "desc"
    sticker = Mock()
    sticker.url = "https://example.com/sticker.png"

    msg.attachments = [att]
    msg.embeds = [embed]
    msg.stickers = [sticker]

    res = asyncio.run(cog._get_attachment_list([msg]))
    assert "https://example.com/a.png" in res
    assert "desc" in res
    assert "https://example.com/sticker.png" in res


@pytest.mark.asyncio
async def test_clear_memory_behavior():
    cog = ReplyGeneratorCogs(bot=Mock())

    # Prepare user memory
    user_id = 123
    cog.user_last_response_id[user_id] = "rid"

    # Fake interaction
    interaction = Mock(spec=nextcord.Interaction)
    interaction.user = Mock()
    interaction.user.id = user_id
    interaction.response = Mock()
    interaction.response.send_message = AsyncMock()

    await cog.clear_memory(interaction)
    assert user_id not in cog.user_last_response_id
    interaction.response.send_message.assert_awaited()
