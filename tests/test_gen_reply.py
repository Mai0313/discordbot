from types import SimpleNamespace
from collections.abc import AsyncIterator

import pytest

from discordbot.cogs.gen_reply import ReplyGeneratorCogs


class FakeReply:
    """Provides a fake reply object that records edited content."""

    def __init__(self) -> None:
        """Initializes the fake reply with empty content."""
        self.content = ""

    async def edit(self, *, content: str) -> None:
        """Records the replacement content passed to edit."""
        self.content = content


class FakeMessage:
    """Provides a fake message object that records created replies."""

    def __init__(self) -> None:
        """Initializes the fake message with no recorded replies."""
        self.replies: list[FakeReply] = []

    async def reply(self, *, content: str) -> FakeReply:
        """Creates and records a fake reply with the requested content."""
        reply = FakeReply()
        reply.content = content
        self.replies.append(reply)
        return reply


async def _stream_events() -> AsyncIterator[SimpleNamespace]:
    yield SimpleNamespace(type="response.output_text.delta", delta="hello from stream")
    yield SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            model="gemini-pro-latest",
            usage=SimpleNamespace(input_tokens=12, output_tokens=34, output_tokens_details=None),
        ),
    )


async def test_handle_streaming_allows_missing_output_token_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: LiteLLM may return usage with output_tokens_details=null."""

    def fake_calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
        assert model_name == "gemini-pro-latest"
        assert input_tokens == 12
        assert output_tokens == 34
        return 0.0

    monkeypatch.setattr(ReplyGeneratorCogs, "_calculate_cost", staticmethod(fake_calculate_cost))

    cog = ReplyGeneratorCogs.__new__(ReplyGeneratorCogs)
    message = FakeMessage()

    result = await cog._handle_streaming(responses=_stream_events(), message=message)

    assert result == "hello from stream\n> **gemini-pro-latest** ⬆ 12 ⬇ 34 $0.00000000"
    assert message.replies[0].content == result
