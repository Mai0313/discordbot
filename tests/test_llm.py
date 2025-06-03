import pytest
from src.sdk.llm import LLMSDK
from openai.types import ImagesResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk

prompt = "Hi"


@pytest.fixture
def llm_sdk() -> LLMSDK:
    return LLMSDK()


@pytest.mark.asyncio
async def test_get_oai_reply(llm_sdk: LLMSDK) -> None:
    response = await llm_sdk.get_oai_reply(prompt=prompt)
    assert isinstance(response, ChatCompletion)


@pytest.mark.skip(reason="This function is not implemented yet.")
@pytest.mark.asyncio
async def test_get_dalle_image(llm_sdk: LLMSDK) -> None:
    response = await llm_sdk.get_dalle_image(prompt=prompt)
    assert isinstance(response, ImagesResponse)


@pytest.mark.asyncio
async def test_get_oai_reply_stream(llm_sdk: LLMSDK) -> None:
    async for response in llm_sdk.get_oai_reply_stream(prompt=prompt):
        assert isinstance(response, ChatCompletionChunk)
