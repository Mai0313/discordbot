from typing import Any

import dotenv
from openai import AsyncOpenAI, AsyncAzureOpenAI
from pydantic import Field, ConfigDict, AliasChoices, computed_field
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
        ...,
        title="LLM Model Selection",
        description="This model should be OpenAI Model.",
        frozen=False,
        deprecated=False,
    )

    @computed_field
    @property
    def client(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        if self.api_version:
            model = self.model.split("/")[1] if "/" in self.model else self.model
            client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.base_url,
                api_version=self.api_version,
                azure_deployment=model,
            )
        else:
            client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        return client

    async def prepare_completion_content(
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

    async def prepare_response_content(
        self, prompt: str, image_urls: list[str] | None = None
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        if not image_urls:
            return content
        for image_url in image_urls:
            image = get_pil_image(image_file=image_url)
            image_base64 = pil_to_data_uri(image=image)
            content.append({"type": "input_image", "image_url": image_base64})
        return content
