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
        anthropic_api_key: The Anthropic key used to upload attachments to the
            Anthropic Files API directly (the side-channel for Claude answer models).
        voice_reply_enabled: Kill-switch for spoken QA replies; when false the answer
            model's voice marker is still stripped but no audio clip is synthesized.
        inline_image_enabled: Kill-switch for inline generated images on QA replies; when
            false the answer model's `<image>` marker is still stripped but no image is rendered.
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
    anthropic_api_key: str = Field(
        default="",
        description="The Anthropic API key for direct Anthropic Files API uploads.",
        examples=["sk-ant-..."],
        validation_alias=AliasChoices("ANTHROPIC_API_KEY"),
    )
    voice_reply_enabled: bool = Field(
        default=True,
        description="Whether the bot may synthesize a spoken clip for fierce QA replies.",
        validation_alias=AliasChoices("VOICE_REPLY_ENABLED"),
    )
    inline_image_enabled: bool = Field(
        default=True,
        description="Whether the bot may render an inline generated image for QA replies.",
        validation_alias=AliasChoices("INLINE_IMAGE_ENABLED"),
    )


__all__ = ["LLMConfig"]
