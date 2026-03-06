from functools import cached_property

import dotenv
from openai import AsyncOpenAI
from pydantic import Field, ConfigDict, AliasChoices, computed_field
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class LLMSDK(BaseSettings):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    base_url: str = Field(
        ...,
        description="The base url from openai for calling models.",
        examples=["https://api.openai.com/v1"],
        validation_alias=AliasChoices("OPENAI_BASE_URL"),
        frozen=False,
        deprecated=False,
    )
    api_key: str = Field(
        ...,
        description="The api key from openai for calling models.",
        examples=["sk-proj-..."],
        validation_alias=AliasChoices("OPENAI_API_KEY"),
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
    @cached_property
    def client(self) -> AsyncOpenAI:
        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        return client
