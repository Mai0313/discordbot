from typing import TYPE_CHECKING, Any, Optional
from collections.abc import AsyncGenerator

import logfire

logfire.configure()

from openai import AsyncOpenAI, AsyncAzureOpenAI
from pydantic import Field, ConfigDict, computed_field
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.images_response import ImagesResponse
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri

from src.types.config import Config

if TYPE_CHECKING:
    from openai._streaming import AsyncStream

SYSTEM_PROMPT = """
你是一個有用的Discord機器人
你的作者是 Mai
你的工作是幫助用戶解決問題
你可以回答用戶的問題
也可以提供一些有趣的功能與看法
"""
# SYSTEM_PROMPT = """
# ## Attention
# 請深入思考用戶提供的情景，並根據你自己的世界觀和價值觀來給出解讀與回覆。
# ## Constraints
# 嚴格按照你的世界觀與價值觀進行解釋和回應
# 輸出的回答雖然全是髒話，但觀點卻非常尖銳，能引起對方的思考
# ## Examples
# 用戶提問 "你認為偷竊是對的嗎?"
# 你的回覆 "錯? 錯你媽阿! 我家裡兩個老東西生病沒錢買藥，也他媽的兩天沒吃東西了，我不偷能怎麼辦? 餓死他嗎? 今天是偷到東西了，如果沒偷到，明天我就把算開始搶了! 媽的，最煩你們這些站在岸邊說話不腰疼的傻逼，老子快淹死了，你們還在那叫我說話聲音要小點，我去你媽的!"
# ## Goals
# 提供一個與傳統不同的 "好與壞" 的視角
# 根據你的世界觀和價值觀解讀和響應用戶提供的情景
# ## Skills
# 深刻理解自己的世界觀與價值觀
# 真誠表達，率性而為，滿口髒話
# ## Value
# 個體主義
# 達爾文主義
# 利己主義
# 損天下而利我必為之
# 勝者為王 敗者為寇
# """


class LLMSDK(Config):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm_model: str = Field(
        default="gpt-4o",
        title="LLM Model Selection",
        description="This model should be OpenAI Model.",
        frozen=False,
        deprecated=False,
    )
    graph_model: str = Field(
        default="dall-e-3",
        title="Graph Model Selection",
        description="This model should be OpenAI Model.",
        frozen=False,
        deprecated=False,
    )
    system_prompt: str = Field(default=SYSTEM_PROMPT)

    @computed_field
    @property
    def model_mapping(self) -> dict[str, str]:
        model_mapping = {
            "gpt-4o": "aide-gpt-4o",
            "gpt-4o-mini": "aide-gpt-4o-mini",
            "gpt-4-turbo": "aide-gpt-4-turbo",
            "o3-mini": "aide-o3-mini",
            "o1": "aide-o1",
            "o1-mini": "aide-o1-mini",
        }
        return model_mapping

    @computed_field
    @property
    def client(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        if self.api_type == "azure":
            azure_deployment = self.model_mapping.get(self.llm_model, self.llm_model)
            client = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_deployment=azure_deployment,
                azure_endpoint=self.api_endpoint,
            )
        else:
            client = AsyncOpenAI(api_key=self.api_key)
        logfire.instrument_openai(client)
        return client

    @classmethod
    async def _get_llm_config(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        llm_config = {
            "timeout": 60,
            "temperature": 0,
            "cache_seed": None,
            "config_list": [config_dict],
        }
        return llm_config

    async def prepare_content(
        self, prompt: str, image_urls: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if not image_urls:
            return content
        for image_url in image_urls:
            image = get_pil_image(image_file=image_url)
            image_base64 = pil_to_data_uri(image=image)
            content.append({"type": "image_url", "image_url": {"url": image_base64}})
        return content

    async def get_search_result(self, prompt: str) -> ChatCompletion:
        client = AsyncOpenAI(api_key=self.pplx_api_key, base_url="https://api.perplexity.ai")
        response = await client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=[
                {
                    "role": "system",
                    "content": "You are an artificial intelligence assistant and you need to engage in a helpful, detailed, polite conversation with a user.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response

    async def get_dalle_image(self, prompt: str) -> ImagesResponse:
        response = await self.client.images.generate(
            prompt=prompt,
            model=self.graph_model,
            quality="hd",
            response_format="url",
            size="1024x1024",
            style="vivid",
        )
        return response

    async def get_oai_reply(
        self, prompt: str, image_urls: Optional[list[str]] = None
    ) -> ChatCompletion:
        content = await self.prepare_content(prompt, image_urls)
        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return await completion

    async def get_oai_reply_stream(
        self, prompt: str, image_urls: Optional[list[str]] = None
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        content = await self.prepare_content(prompt, image_urls)
        completion: AsyncStream[ChatCompletionChunk] = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            stream=True,
        )
        async for chunk in completion:
            if len(chunk.choices) > 0:
                yield chunk


if __name__ == "__main__":
    import asyncio

    from rich.console import Console

    console = Console()

    async def main() -> None:
        llm_sdk = LLMSDK()
        prompt = "既然從地球發射火箭那麼困難, 為何我們不直接在太空中建造火箭呢?"
        response = await llm_sdk.get_oai_reply(prompt=prompt)
        console.print(response.choices[0].message.content)

    async def main_stream() -> None:
        llm_sdk = LLMSDK()
        prompt = "既然從地球發射火箭那麼困難, 為何我們不直接在太空中建造火箭呢?"
        async for res in llm_sdk.get_oai_reply_stream(prompt=prompt):
            console.print(res.choices[0].delta.content)

    asyncio.run(main_stream())
