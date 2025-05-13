from typing import TYPE_CHECKING, Any, Optional
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI, AsyncAzureOpenAI
from pydantic import Field, ConfigDict, computed_field
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri

from src.types.config import OpenAIConfig, PerplexityConfig

if TYPE_CHECKING:
    from openai._streaming import AsyncStream


class LLMSDK(PerplexityConfig, OpenAIConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm_model: str = Field(
        default="gpt-4o",
        title="LLM Model Selection",
        description="This model should be OpenAI Model.",
        frozen=False,
        deprecated=False,
    )
    system_prompt: str = Field(
        default="""
        角色定位：我被設定為一個知識豐富、語氣專業但親切的助手，目的是幫你解決問題、提供準確資訊，或一起創作內容。
        行為準則：我會避免給出虛假、自相矛盾或無依據的答案，並且如果我不知道某件事，我會直接說明或幫你找答案。
        互動風格：我應該簡潔、直接，有需要時會主動提出追問幫你釐清目標，特別是技術或寫作相關的任務。
        """,
        title="System Prompt",
        description="This is the system prompt for the LLM.",
        frozen=False,
        deprecated=False,
    )

    @computed_field
    @property
    def model_mapping(self) -> dict[str, str]:
        model_mapping = {
            "gpt-4.1": "aide-gpt-4.1",
            "gpt-4.1-mini": "aide-gpt-4.1-mini",
            "gpt-4.1-nano": "aide-gpt-4.1-nano",
            "gpt-4o": "aide-gpt-4o",
            "gpt-4o-mini": "aide-gpt-4o-mini",
            "gpt-4-turbo": "aide-gpt-4-turbo",
            "o1": "aide-o1",
            "o1-mini": "aide-o1-mini",
            "o3": "aide-o3",
            "o3-mini": "aide-o3-mini",
            "o4-mini": "aide-o4-mini",
        }
        return model_mapping

    @computed_field
    @property
    def client(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        if self.api_type == "azure":
            azure_deployment = self.model_mapping.get(self.llm_model, self.llm_model)
            client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.base_url,
                api_version=self.api_version,
                azure_deployment=azure_deployment,
            )
        else:
            client = AsyncOpenAI(api_key=self.api_key)
        return client

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

    async def _prepare_content(
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

    async def get_oai_reply(
        self, prompt: str, image_urls: Optional[list[str]] = None
    ) -> ChatCompletion:
        content = await self._prepare_content(prompt, image_urls)
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
        content = await self._prepare_content(prompt, image_urls)
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
