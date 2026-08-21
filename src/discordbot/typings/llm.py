import os
import re

import dotenv
from pydantic import Field, BaseModel, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv()

# The keys past the first are numbered rather than named, because the count is open-ended:
# a deployment adds a key by adding `GEMINI_API_KEY_4` here and its `-key4` deployments to
# the proxy, with no code change. That open end is why `LLMConfig.gemini_keys` is a property
# reading the environment rather than a field with a `validation_alias` per key.
_NUMBERED_GEMINI_KEY_RE = re.compile(r"^GEMINI_API_KEY_(\d+)$")


class GeminiKeySlot(BaseModel):
    """One configured Gemini API key, identified by the number the deployment gave it.

    The number is the whole point of this type. It ties three things that have to agree
    across the deployment: the `GEMINI_API_KEY` / `GEMINI_API_KEY_<n>` environment variable
    the key came from, the `-key<n>` LiteLLM deployment that holds the same credential, and
    the Google project whose Files API will accept a file uploaded with this key. A reply
    that mixes two of them fails outright rather than degrading, so the index travels with
    the key instead of being recomputed from a list position.

    Attributes:
        index: 1-based key number, matching both the env var suffix and the `-key<n>` alias.
        api_key: The Google AI Studio key itself.
    """

    index: int = Field(
        ...,
        description="1-based key number, matching the env var suffix and the `-key<n>` alias.",
        examples=[1, 2, 3],
    )
    api_key: str = Field(..., description="The Google AI Studio key itself.", examples=["AIza..."])


class LLMConfig(BaseSettings):
    """Configuration settings for LLM integration, reading from environment variables.

    Attributes:
        base_url: The base URL for the OpenAI API or compatible endpoint.
        api_key: The API key for authentication.
        gemini_api_key: The Google AI Studio key used to upload attachments to
            the Gemini Files API directly, so uploads can be polled to ACTIVE. Also key 1
            of the balanced set; see `gemini_keys` for the rest and for what the numbering
            is tied to.
        anthropic_api_key: The Anthropic key used to upload attachments to the
            Anthropic Files API directly (the side-channel for Claude answer models).
        xai_api_key: The xAI key used to upload attachments to the xAI Files API
            directly (the side-channel for Grok answer models, which the proxy cannot route).
        inline_voice_enabled: Kill-switch for spoken replies; when false
            the answer model's voice marker is still stripped but no audio clip is
            synthesized.
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
            answer model's `<deep-research>` marker is still stripped but no research runs, and
            a restart leaves whatever was in flight alone instead of re-attaching to it.
        image_refine_prompt_enabled: Kill-switch for the IMAGE-route prompt director; when false
            the raw user request goes straight to the image model with no refinement step.
        video_refine_prompt_enabled: Kill-switch for the VIDEO-route prompt director; when false
            the raw user request goes straight to the video model with no refinement step.
        file_api_enabled: Kill-switch for handing the answer model a provider Files API
            reference; when false attachments inline as base64 instead and link media is not
            uploaded at all. Provider-agnostic on purpose: the Gemini path is the only one
            wired today, but the switch answers the same question for every uploader.
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
        description="Whether the bot may synthesize a spoken clip for a reply.",
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
    file_api_enabled: bool = Field(
        default=True,
        description="Whether media may reach the answer model as a provider Files API reference.",
        validation_alias=AliasChoices("FILE_API_ENABLED"),
    )

    @property
    def gemini_keys(self) -> list[GeminiKeySlot]:
        """Every configured Gemini key, ordered by its number.

        Key 1 is `gemini_api_key` itself, taken off the field rather than the environment so
        a test that overrides the field is honoured. The rest come from `GEMINI_API_KEY_<n>`
        read straight from the environment, which is the only place an open-ended count can
        come from; `_1` is deliberately not one of them, since key 1 is the unsuffixed
        variable and a second spelling of the same number could only disagree with it.

        A blank value is dropped instead of occupying its number, so emptying a variable
        retires that key without renumbering the ones after it (the numbers are shared with
        the proxy's `-key<n>` deployments, so they must not shift). An unconfigured
        deployment gets an empty list.

        Returns:
            The configured keys in number order, blanks removed.
        """
        slots: list[GeminiKeySlot] = []
        if self.gemini_api_key.strip():
            slots.append(GeminiKeySlot(index=1, api_key=self.gemini_api_key.strip()))
        numbered: list[tuple[int, str]] = []
        for name, value in os.environ.items():
            matched = _NUMBERED_GEMINI_KEY_RE.match(string=name)
            if matched is None or not value.strip():
                continue
            number = int(matched.group(1))
            if number < 2:
                continue
            numbered.append((number, value.strip()))
        slots.extend(
            GeminiKeySlot(index=number, api_key=value) for number, value in sorted(numbered)
        )
        return slots

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


__all__ = ["GeminiKeySlot", "LLMConfig"]
