from unittest.mock import Mock, AsyncMock

import pytest
import nextcord

from discordbot.cogs.template import TemplateCogs


@pytest.mark.asyncio
async def test_on_message_reacts_on_debug_word():
    mock_bot = Mock()
    cog = TemplateCogs(mock_bot)

    # Build a fake message
    message = Mock(spec=nextcord.Message)
    message.author = Mock()
    message.author.bot = False
    message.content = "DeBuG"  # case-insensitive
    message.add_reaction = AsyncMock()

    await cog.on_message(message)
    message.add_reaction.assert_awaited_once_with("🤬")


@pytest.mark.asyncio
async def test_on_message_ignores_bots():
    mock_bot = Mock()
    cog = TemplateCogs(mock_bot)

    message = Mock(spec=nextcord.Message)
    message.author = Mock()
    message.author.bot = True
    message.content = "debug"
    message.add_reaction = AsyncMock()

    await cog.on_message(message)
    message.add_reaction.assert_not_called()


@pytest.mark.asyncio
async def test_ping_command_builds_localized_embed():
    mock_bot = Mock()
    # bot.latency is used by the command; set as float seconds
    mock_bot.latency = 0.123

    cog = TemplateCogs(mock_bot)

    # Create a fake interaction with needed attributes/methods
    interaction = Mock(spec=nextcord.Interaction)
    interaction.locale = nextcord.Locale.zh_TW
    interaction.user = Mock()
    interaction.user.display_name = "tester"
    interaction.user.display_avatar = Mock()
    interaction.user.display_avatar.url = "https://example/avatar.png"

    # followup and response APIs are awaited
    interaction.response = Mock()
    interaction.response.defer = AsyncMock()
    interaction.followup = Mock()
    interaction.followup.send = AsyncMock()

    await cog.ping(interaction)

    # Ensure an embed was sent
    interaction.followup.send.assert_awaited()
