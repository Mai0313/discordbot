from typing import Any
from collections.abc import AsyncGenerator

import dotenv
from openai import AsyncOpenAI, AsyncAzureOpenAI
from pydantic import Field, ConfigDict, AliasChoices, computed_field
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic_settings import BaseSettings
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri

dotenv.load_dotenv()


class LLMSDK(BaseSettings):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    base_url: str = Field(
        ...,
        description="The base url from openai for calling models.",
        examples=["https://api.openai.com/v1", "https://xxxx.openai.azure.com"],
        validation_alias=AliasChoices("OPENAI_BASE_URL", "AZURE_OPENAI_ENDPOINT"),
        frozen=False,
        deprecated=False,
    )
    api_key: str = Field(
        ...,
        description="The api key from openai for calling models.",
        examples=["sk-proj-...", "141698ac..."],
        validation_alias=AliasChoices("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
        frozen=False,
        deprecated=False,
    )
    api_version: str | None = Field(
        default=None,
        description="The api version from openai for calling models.",
        examples=["2025-04-01-preview"],
        validation_alias=AliasChoices("OPENAI_API_VERSION"),
        frozen=False,
        deprecated=False,
    )
    model: str = Field(
        default="gpt-4.1",
        title="LLM Model Selection",
        description="This model should be OpenAI Model.",
        frozen=False,
        deprecated=False,
    )
    pplx_api_key: str = Field(
        ...,
        description="The api key from perplexity for calling models.",
        examples=["pplx-..."],
        validation_alias=AliasChoices("PERPLEXITY_API_KEY"),
        frozen=False,
        deprecated=False,
    )

    @computed_field
    @property
    def client(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        if self.api_version:
            client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.base_url,
                api_version=self.api_version,
                azure_deployment=self.model,
            )
        else:
            client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        return client

    @computed_field
    @property
    def pplx_client(self) -> AsyncOpenAI:
        pplx_client = AsyncOpenAI(api_key=self.pplx_api_key, base_url="https://api.perplexity.ai")
        return pplx_client

    async def get_search_result(self, prompt: str) -> ChatCompletion:
        response = await self.pplx_client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=[{"role": "user", "content": prompt}],
        )
        return response

    async def _prepare_content(
        self, prompt: str, image_urls: list[str] | None = None
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
        self, prompt: str, image_urls: list[str] | None = None
    ) -> ChatCompletion:
        content = await self._prepare_content(prompt=prompt, image_urls=image_urls)
        responses = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}]
        )
        return await responses

    async def get_oai_response(
        self, prompt: str, image_urls: list[str] | None = None
    ) -> ChatCompletion:
        content = await self._prepare_content(prompt=prompt, image_urls=image_urls)
        responses = self.client.responses.create(
            model=self.model, input=[{"role": "user", "content": content}]
        )
        return await responses

    async def get_oai_reply_stream(
        self, prompt: str, image_urls: list[str] | None = None
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        content = await self._prepare_content(prompt=prompt, image_urls=image_urls)
        responses = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}], stream=True
        )
        return await responses

    async def get_oai_response_stream(
        self, prompt: str, image_urls: list[str] | None = None
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        content = await self._prepare_content(prompt=prompt, image_urls=image_urls)
        responses = self.client.responses.create(
            model=self.model, input=[{"role": "user", "content": content}], stream=True
        )
        return await responses


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
        responses = await llm_sdk.get_oai_reply_stream(prompt=prompt)
        async for res in responses:
            console.print(res.choices[0].delta.content, end="")

    asyncio.run(main_stream())
