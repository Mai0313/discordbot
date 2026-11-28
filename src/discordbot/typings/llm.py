"""Runtime LLM configuration: proxy/provider credentials from env and switches from TOML."""

from pydantic import Field, BaseModel, AliasChoices, model_validator
from pydantic_settings import BaseSettings

from discordbot.typings.config import load_bot_config


class LLMCredentials(BaseSettings):
    """LLM API credentials and endpoint URLs loaded from environment variables."""

    # Credentials default to empty for credential-less tests; missing `OPENAI_API_KEY` fails on
    # first client construction, while missing `GEMINI_API_KEY` disables or falls back off
    # direct-to-Google paths.
    base_url: str = Field(
        default="",
        description="The base url for the OpenAI API or a compatible endpoint.",
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
        description=(
            "The Google AI Studio key for every direct-to-Google path, Gemini Files API "
            "uploads among them. One key serves the whole deployment: a file is readable "
            "only by the project that uploaded it, so the key that uploads has to be the "
            "key that answers."
        ),
        examples=["AIza..."],
        validation_alias=AliasChoices("GEMINI_API_KEY"),
    )
    anthropic_api_key: str = Field(
        default="",
        description=(
            "Reserved Anthropic API key for the scaffolded direct Anthropic Files API uploader "
            "(not enabled at runtime)."
        ),
        examples=["sk-ant-..."],
        validation_alias=AliasChoices("ANTHROPIC_API_KEY"),
    )
    xai_api_key: str = Field(
        default="",
        description=(
            "Reserved xAI API key for the scaffolded direct xAI Files API uploader (not enabled "
            "at runtime)."
        ),
        examples=["xai-..."],
        validation_alias=AliasChoices("XAI_API_KEY"),
    )


class LLMConfig(BaseModel):
    """Runtime LLM settings combining env credentials with `[global.reply]` TOML switches."""

    base_url: str = Field(
        default="",
        description="The base url for the OpenAI API or a compatible endpoint.",
        examples=["https://api.openai.com/v1"],
    )
    api_key: str = Field(
        default="",
        description="The api key from openai for calling models.",
        examples=["sk-proj-..."],
    )
    gemini_api_key: str = Field(
        default="",
        description="The Google AI Studio key for every direct-to-Google path.",
        examples=["AIza..."],
    )
    anthropic_api_key: str = Field(
        default="",
        description="Reserved Anthropic API key for the scaffolded direct Files API uploader.",
        examples=["sk-ant-..."],
    )
    xai_api_key: str = Field(
        default="",
        description="Reserved xAI API key for the scaffolded direct Files API uploader.",
        examples=["xai-..."],
    )
    feedback_enabled: bool = Field(
        default=True,
        description=(
            "Whether finalized replies attach persistent thumbs-up/thumbs-down feedback buttons "
            "(plus a `🔄` regenerate button on normal gateway QA replies) and persist turn "
            "snapshots in `reply.db`; when false replies omit the buttons, skip turn snapshots, "
            "and refuse `🔄` regeneration clicks."
        ),
    )
    file_api_enabled: bool = Field(
        default=True,
        description=(
            "Whether media may reach the answer model as a provider Files API reference; when "
            "false attachments and linked-post images inline as base64 instead, while "
            "linked-post video clips are skipped. Provider-agnostic on purpose: the switch "
            "answers the same question for every uploader."
        ),
    )
    answer_streaming_enabled: bool = Field(
        default=True,
        description=(
            "Whether QA replies and media-persona follow-ups stream live preview edits onto "
            "Discord; when false they fetch complete non-streaming responses and post them once "
            "finished without live preview edits."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _populate_from_runtime_sources(cls, data: object) -> object:
        """Fills omitted fields from `LLMCredentials` and `[global.reply]` in `config.toml`."""
        if not isinstance(data, dict):
            return data
        credentials = LLMCredentials()
        reply = load_bot_config().global_config.reply
        defaults: dict[str, object] = {
            "base_url": credentials.base_url,
            "api_key": credentials.api_key,
            "gemini_api_key": credentials.gemini_api_key,
            "anthropic_api_key": credentials.anthropic_api_key,
            "xai_api_key": credentials.xai_api_key,
            "feedback_enabled": reply.feedback_enabled,
            "file_api_enabled": reply.file_api_enabled,
            "answer_streaming_enabled": reply.streaming_enabled,
        }
        return {**defaults, **data}

    @property
    def gemini_key_configured(self) -> bool:
        """Returns whether a non-blank direct Gemini API key is configured."""
        return bool(self.gemini_api_key.strip())
