import pytest
from src.sdk.llm import LLMSDK
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

prompt = "既然從地球發射火箭那麼困難, 為何我們不直接在太空中建造火箭呢?"


@pytest.fixture
def llm_services() -> LLMSDK:
    return LLMSDK()


@pytest.mark.asyncio
async def test_get_oai_reply_stream(llm_services: LLMSDK) -> None:
    async for response in llm_services.get_oai_reply_stream(prompt=prompt):
        assert isinstance(response, ChatCompletionChunk)
