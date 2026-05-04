import dotenv
from pydantic import Field, ConfigDict, AliasChoices
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class LLMConfig(BaseSettings):
    """Configuration settings for LLM integration, reading from environment variables.

    Attributes:
        base_url: The base URL for the OpenAI API or compatible endpoint.
        api_key: The API key for authentication.
    """

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
