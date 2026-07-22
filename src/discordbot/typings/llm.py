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
        xai_api_key: The xAI key used to upload attachments to the xAI Files API
            directly (the side-channel for Grok answer models, which the proxy cannot route).
        inline_voice_enabled: Kill-switch for spoken QA replies; when false the answer
            model's voice marker is still stripped but no audio clip is synthesized.
        inline_image_enabled: Kill-switch for inline generated images on QA replies; when
            false the answer model's `<generate-image>` marker is still stripped but no image is rendered.
        inline_music_enabled: Kill-switch for inline generated music on QA replies; when false
            the answer model's `<generate-music>` marker is still stripped but no clip is generated.
        inline_video_enabled: Kill-switch for inline generated video on QA replies; when false
            the answer model's `<generate-video>` marker is still stripped but no clip is generated.
        youtube_video_enabled: Kill-switch for answering about a linked YouTube video via the
            Gemini Interactions API; when false the QA turn falls back to the Responses path
            (which cannot watch the video).
        douyin_video_enabled: Kill-switch for downloading a linked Douyin post's media and
            uploading it so the answer model can watch it; when false the caption still rides
            as context but the model is told plainly that it has not seen the clip.
        bilibili_video_enabled: Kill-switch for downloading a linked Bilibili video and
            uploading it so the answer model can watch it; when false the title and
            description still ride as context but the model is told plainly that it has not
            watched the clip.
        deep_research_enabled: Kill-switch for the deep-research feature; when false the QA
            answer model's `<deep-research>` marker is still stripped but no research runs.
        deep_research_max_enabled: Whether the priciest Deep Research Max tier may be picked
            from the escalation buttons; off by default so the expensive tier is opt-in.
        image_refine_prompt_enabled: Kill-switch for the IMAGE-route prompt director; when false
            the raw user request goes straight to the image model with no refinement step.
        video_refine_prompt_enabled: Kill-switch for the VIDEO-route prompt director; when false
            the raw user request goes straight to the video model with no refinement step.
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
    xai_api_key: str = Field(
        default="",
        description="The xAI API key for direct xAI Files API uploads.",
        examples=["xai-..."],
        validation_alias=AliasChoices("XAI_API_KEY"),
    )
    inline_voice_enabled: bool = Field(
        default=True,
        description="Whether the bot may synthesize a spoken clip for fierce QA replies.",
        validation_alias=AliasChoices("INLINE_VOICE_ENABLED"),
    )
    inline_image_enabled: bool = Field(
        default=True,
        description="Whether the bot may render an inline generated image for QA replies.",
        validation_alias=AliasChoices("INLINE_IMAGE_ENABLED"),
    )
    inline_music_enabled: bool = Field(
        default=True,
        description="Whether the bot may generate an inline music clip for QA replies.",
        validation_alias=AliasChoices("INLINE_MUSIC_ENABLED"),
    )
    inline_video_enabled: bool = Field(
        default=True,
        description="Whether the bot may generate an inline video clip for QA replies.",
        validation_alias=AliasChoices("INLINE_VIDEO_ENABLED"),
    )
    youtube_video_enabled: bool = Field(
        default=True,
        description="Whether the bot may watch a linked YouTube video via the Interactions API.",
        validation_alias=AliasChoices("YOUTUBE_VIDEO_ENABLED"),
    )
    douyin_video_enabled: bool = Field(
        default=True,
        description="Whether the bot may upload a linked Douyin post's media for the model to read.",
        validation_alias=AliasChoices("DOUYIN_VIDEO_ENABLED"),
    )
    bilibili_video_enabled: bool = Field(
        default=True,
        description="Whether the bot may upload a linked Bilibili video for the model to watch.",
        validation_alias=AliasChoices("BILIBILI_VIDEO_ENABLED"),
    )
    deep_research_enabled: bool = Field(
        default=True,
        description="Whether the bot may launch a deep-research thread from a QA marker / slash.",
        validation_alias=AliasChoices("DEEP_RESEARCH_ENABLED"),
    )
    deep_research_max_enabled: bool = Field(
        default=False,
        description="Whether the priciest Deep Research Max tier is offered on the escalation buttons.",
        validation_alias=AliasChoices("DEEP_RESEARCH_MAX_ENABLED"),
    )
    image_refine_prompt_enabled: bool = Field(
        default=True,
        description="Whether the prompt director refines the IMAGE-route request before generation.",
        validation_alias=AliasChoices("IMAGE_REFINE_PROMPT_ENABLED"),
    )
    video_refine_prompt_enabled: bool = Field(
        default=True,
        description="Whether the prompt director refines the VIDEO-route request before generation.",
        validation_alias=AliasChoices("VIDEO_REFINE_PROMPT_ENABLED"),
    )

    @property
    def deep_research_available(self) -> bool:
        """Whether deep research can actually run: enabled AND a direct Gemini key is configured.

        The research cog calls Google directly with `gemini_api_key`; without it a launch would
        open a thread and then fail, so the QA marker and `/deep_research` are only offered when
        both the kill-switch is on and the key is present.
        """
        return self.deep_research_enabled and bool(self.gemini_api_key.strip())

    @property
    def music_available(self) -> bool:
        """Whether inline music can actually run: enabled AND a direct Gemini key is configured.

        The music clip is generated by calling Google directly with `gemini_api_key` (the Lyria
        Interactions API); without it the render would just fail, so the QA `<generate-music>` marker is
        only advertised when both the kill-switch is on and the key is present.
        """
        return self.inline_music_enabled and bool(self.gemini_api_key.strip())

    @property
    def video_available(self) -> bool:
        """Whether inline video can actually run: enabled AND a direct Gemini key is configured.

        The video clip is generated by calling Google directly with `gemini_api_key` (the omni
        Interactions API, Gemini-only, not reachable via the proxy); without it the render would
        just fail, so the QA `<generate-video>` marker is only advertised when both the kill-switch is on
        and the key is present.
        """
        return self.inline_video_enabled and bool(self.gemini_api_key.strip())


__all__ = ["LLMConfig"]
