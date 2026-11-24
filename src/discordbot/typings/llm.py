"""Runtime LLM configuration: the proxy credentials, the Gemini key, and the kill-switches."""

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """Configuration settings for LLM integration, reading from environment variables."""

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
    feedback_enabled: bool = Field(
        default=True,
        description=(
            "Whether finalized replies attach persistent thumbs-up/thumbs-down feedback buttons "
            "(plus a `🔄` regenerate button on normal gateway QA replies) and persist turn "
            "snapshots in `reply.db`; when false replies omit the buttons, skip turn snapshots, "
            "and refuse `🔄` regeneration clicks."
        ),
        validation_alias=AliasChoices("FEEDBACK_ENABLED"),
    )
    slow_model_blind_test_enabled: bool = Field(
        default=False,
        description=(
            "Whether QA turns randomly pick `slow_model` from "
            "`RuntimeModelCatalog.slow_model_blind_test_candidates` (allowing `🔄` regeneration "
            "across remaining untried candidates) and mask the model name as `blind-test-model` "
            "in the system prompt and reply footer (also hiding the effort label in the footer), "
            "while still recording the real answered model and effort in `reply.db` and logs; "
            "when false QA uses `slow_model`, allows one regeneration, and shows its real name "
            "and effort."
        ),
        validation_alias=AliasChoices("SLOW_MODEL_BLIND_TEST_ENABLED"),
    )
    file_api_enabled: bool = Field(
        default=True,
        description=(
            "Whether media may reach the answer model as a provider Files API reference; when "
            "false attachments inline as base64 instead and link media is not uploaded at all. "
            "Provider-agnostic on purpose: the switch answers the same question for every "
            "uploader."
        ),
        validation_alias=AliasChoices("FILE_API_ENABLED"),
    )
    answer_streaming_enabled: bool = Field(
        default=True,
        description=(
            "Whether QA replies and media-persona follow-ups stream live preview edits onto "
            "Discord; when false they fetch complete non-streaming responses and post them once "
            "finished without live preview edits."
        ),
        validation_alias=AliasChoices("ANSWER_STREAMING_ENABLED"),
    )

    @property
    def gemini_key_configured(self) -> bool:
        """Returns whether a non-blank direct Gemini API key is configured."""
        return bool(self.gemini_api_key.strip())
