from unittest.mock import Mock, AsyncMock, patch

import pytest
import nextcord

from discordbot.cogs.summary import MessageFetcher


@pytest.mark.asyncio
async def test_format_messages_collects_embeds_and_attachments():
    mock_bot = Mock()
    cog = MessageFetcher(mock_bot)

    # Build fake messages
    m1 = Mock(spec=nextcord.Message)
    m1.content = "hello"
    m1.embeds = []
    m1.attachments = []
    m1.author = Mock()
    m1.author.name = "user1"

    m2 = Mock(spec=nextcord.Message)
    m2.content = ""
    # one embed with description
    embed = Mock()
    embed.description = "an embed desc"
    m2.embeds = [embed]
    m2.attachments = []
    m2.author = Mock()
    m2.author.name = "user2"

    m3 = Mock(spec=nextcord.Message)
    m3.content = ""
    att = Mock()
    att.url = "https://file/url.png"
    m3.attachments = [att]
    m3.embeds = []
    m3.author = Mock()
    m3.author.name = "user3"

    final_prompt, attachments = cog._format_messages([m1, m2, m3])

    assert "user1: hello" in final_prompt
    assert "嵌入內容" in final_prompt
    assert "附件" in final_prompt
    assert "an embed desc" in attachments
    assert "https://file/url.png" in attachments


@pytest.mark.asyncio
@patch("discordbot.cogs.summary.LLMSDK")
async def test_do_summarize_calls_llm(mock_llmsdk: Mock):
    mock_bot = Mock()
    cog = MessageFetcher(mock_bot)

    # Fake channel.history
    channel = Mock(spec=nextcord.TextChannel)

    msg = Mock(spec=nextcord.Message)
    msg.author = Mock()
    msg.author.bot = False
    msg.author.name = "user"
    msg.content = "content"
    msg.embeds = []
    msg.attachments = []

    async def history_iter(limit=None):
        yield msg

    channel.history = Mock(return_value=history_iter())

    # Mock LLM SDK and its methods
    instance = Mock()
    instance.model = "openai/gpt-5-mini"
    instance.prepare_response_content = AsyncMock(
        return_value=[{"type": "input_text", "text": "prepared"}]
    )

    # client.responses.create returns an object with output_text
    class R:
        output_text = "summary"

    instance.client = Mock()
    instance.client.responses = Mock()
    instance.client.responses.create = AsyncMock(return_value=R())
    mock_llmsdk.return_value = instance

    result = await cog.do_summarize(channel, history_count=1, target_user=None)
    assert result == "summary"
    instance.prepare_response_content.assert_awaited()
    instance.client.responses.create.assert_awaited()
