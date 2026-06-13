import dotenv
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv()


class LLMConfig(BaseSettings):
    """Configuration settings for LLM integration, reading from environment variables.

    Attributes:
        base_url: The base URL for the OpenAI API or compatible endpoint.
        api_key: The API key for authentication.
        gemini_api_key: The Google AI Studio key used to upload attachments to
            the Gemini Files API directly, so uploads can be polled to ACTIVE.
    """

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)
    # All credentials default to empty so tests never have to supply env vars; a real
    # deployment provides them via .env, and an empty value fails at the API call.
    base_url: str = Field(
        default="",
        description="The base url from openai for calling models.",
        examples=["https://api.openai.com/v1"],
        validation_alias=AliasChoices("OPENAI_BASE_URL"),
    )
    api_key: str = Field(
        default="",
        description="The api key from openai for calling models.",
        examples=["sk-proj-..."],
        validation_alias=AliasChoices("OPENAI_API_KEY"),
    )
    gemini_api_key: str = Field(
        default="",
        description="The Google AI Studio key for direct Gemini Files API uploads.",
        examples=["AIza..."],
        validation_alias=AliasChoices("GEMINI_API_KEY"),
    )


__all__ = ["LLMConfig"]
