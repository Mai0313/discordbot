import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from discordbot.sdk.llm import LLMSDK

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
    responses = await llm_sdk.get_oai_reply_stream(prompt=prompt)
    async for response in responses:
        assert isinstance(response, ChatCompletionChunk)
