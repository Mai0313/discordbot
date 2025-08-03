import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from src.discordbot.sdk.llm import LLMSDK

prompt = "Hi"


@pytest.fixture
def llm_sdk() -> LLMSDK:
    return LLMSDK()


@pytest.mark.asyncio
async def test_get_oai_reply(llm_sdk: LLMSDK) -> None:
    response = await llm_sdk.get_oai_reply(prompt=prompt)
    assert isinstance(response, ChatCompletion)


@pytest.mark.asyncio
async def test_get_oai_reply_stream(llm_sdk: LLMSDK) -> None:
    async for response in llm_sdk.get_oai_reply_stream(prompt=prompt):
        assert isinstance(response, ChatCompletionChunk)
