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
    inline_voice_enabled: bool = Field(
        default=True,
        description=(
            "Whether the bot may synthesize a spoken clip for a reply; when false the answer "
            "model's voice marker is still stripped but no audio clip is synthesized."
        ),
        validation_alias=AliasChoices("INLINE_VOICE_ENABLED"),
    )
    inline_image_enabled: bool = Field(
        default=True,
        description=(
            "Whether the bot may render an inline generated image for QA replies; when false "
            "the `<generate-image>` marker is still stripped but no image is rendered."
        ),
        validation_alias=AliasChoices("INLINE_IMAGE_ENABLED"),
    )
    inline_music_enabled: bool = Field(
        default=True,
        description=(
            "Whether the bot may generate an inline music clip for QA replies; when false the "
            "`<generate-music>` marker is still stripped but no clip is generated."
        ),
        validation_alias=AliasChoices("INLINE_MUSIC_ENABLED"),
    )
    inline_video_enabled: bool = Field(
        default=True,
        description=(
            "Whether the bot may generate an inline video clip for QA replies; when false the "
            "`<generate-video>` marker is still stripped but no clip is generated."
        ),
        validation_alias=AliasChoices("INLINE_VIDEO_ENABLED"),
    )
    youtube_video_enabled: bool = Field(
        default=True,
        description=(
            "Whether the bot may watch a linked YouTube video via the Interactions API; when "
            "false the QA turn falls back to the Responses path, which cannot watch the video."
        ),
        validation_alias=AliasChoices("YOUTUBE_VIDEO_ENABLED"),
    )
    douyin_video_enabled: bool = Field(
        default=True,
        description=(
            "Whether the bot may upload a linked Douyin post's media for the model to read; "
            "when false the caption still rides as context but the model is told plainly that "
            "it has not seen the clip."
        ),
        validation_alias=AliasChoices("DOUYIN_VIDEO_ENABLED"),
    )
    bilibili_video_enabled: bool = Field(
        default=True,
        description=(
            "Whether the bot may upload a linked Bilibili video for the model to watch; when "
            "false the title and description still ride as context but the model is told "
            "plainly that it has not watched the clip."
        ),
        validation_alias=AliasChoices("BILIBILI_VIDEO_ENABLED"),
    )
    deep_research_enabled: bool = Field(
        default=True,
        description=(
            "Whether the bot may launch a deep-research thread from a QA marker / slash; when "
            "false the `<deep-research>` marker is still stripped but no research runs, and a "
            "restart leaves whatever was in flight alone instead of re-attaching to it."
        ),
        validation_alias=AliasChoices("DEEP_RESEARCH_ENABLED"),
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
    image_refine_prompt_enabled: bool = Field(
        default=True,
        description=(
            "Whether the prompt director refines the IMAGE-route request before generation; "
            "when false the raw user request goes straight to the image model."
        ),
        validation_alias=AliasChoices("IMAGE_REFINE_PROMPT_ENABLED"),
    )
    video_refine_prompt_enabled: bool = Field(
        default=True,
        description=(
            "Whether the prompt director refines the VIDEO-route request before generation; "
            "when false the raw user request goes straight to the video model."
        ),
        validation_alias=AliasChoices("VIDEO_REFINE_PROMPT_ENABLED"),
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

    @property
    def deep_research_available(self) -> bool:
        """Returns whether deep research is enabled and a direct Gemini API key is configured."""
        return self.deep_research_enabled and self.gemini_key_configured

    @property
    def music_available(self) -> bool:
        """Returns whether inline music is enabled and a direct Gemini API key is configured."""
        return self.inline_music_enabled and self.gemini_key_configured

    @property
    def video_available(self) -> bool:
        """Returns whether inline video is enabled and a direct Gemini API key is configured."""
        return self.inline_video_enabled and self.gemini_key_configured
