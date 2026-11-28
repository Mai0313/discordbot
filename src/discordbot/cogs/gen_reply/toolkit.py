"""Process-wide clients, generators, model catalog, and input caches for reply composition.

Shared across turns so `input_builder`'s rendered-part and Files API caches persist across
replies. Delivery and usage services (`media_delivery`, `usage_recorder`) remain on the cog.
"""

from typing import Literal
from functools import cached_property

from google import genai
from openai import AsyncOpenAI
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from anthropic import AsyncAnthropic
from nextcord.ext import commands
from google.genai.types import HttpOptions

from discordbot.utils.lru import BoundedLRU
from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.cogs.gen_reply.input import AttachmentCache, MessageInputBuilder
from discordbot.services.memory.writer import MemoryWriterAI
from discordbot.cogs.gen_reply.generation import (
    ImageGenerator,
    MusicGenerator,
    VideoGenerator,
    VoiceGenerator,
    PromptGenerator,
)
from discordbot.cogs.gen_reply.attachment.base import AttachmentRenderer
from discordbot.cogs.gen_reply.attachment.select import build_attachment_handler

type AttachmentBucketKey = Literal["gemini", "claude", "inline"]


class ReplyToolkit(BaseModel):
    """The clients, generators and caches every reply is composed with."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: SkipValidation[commands.Bot] = Field(
        ..., description="The Discord bot instance, passed through to the input builder."
    )
    openai_client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client."
    )
    config: LLMConfig = Field(
        ...,
        description=(
            "Runtime LLM config supplying the direct provider API keys (Google AI Studio, "
            "Anthropic) and the Files API kill-switch."
        ),
    )
    _memory_writer: MemoryWriterAI | None = PrivateAttr(default=None)
    _attachment_buckets: dict[AttachmentBucketKey, tuple[AttachmentRenderer, AttachmentCache]] = (
        PrivateAttr(default_factory=dict)
    )
    _input_builders: dict[str, MessageInputBuilder] = PrivateAttr(default_factory=dict)

    @cached_property
    def runtime_models(self) -> RuntimeModelCatalog:
        """Returns the runtime model catalog for every tier."""
        return RuntimeModelCatalog()

    @cached_property
    def gemini_client(self) -> genai.Client:
        """Returns the native Gemini client for direct-to-Google paths.

        Raises at construction when `config.gemini_api_key` is empty; callers that can run without
        a key use `gemini_client_if_configured` instead.
        """
        # `google-genai` defaults to `v1`, where `gemini-omni-*` and `lyria-*` 404 and `files.upload`
        # returns `/v1/files/` URIs that miss LiteLLM's `/v1beta/files/` pass-through prefix.
        return genai.Client(
            api_key=self.config.gemini_api_key, http_options=HttpOptions(api_version="v1beta")
        )

    @property
    def gemini_client_if_configured(self) -> genai.Client | None:
        """Returns the direct Gemini client, or None when `config.gemini_key_configured` is false."""
        if not self.config.gemini_key_configured:
            return None
        return self.gemini_client

    @cached_property
    def anthropic_client(self) -> AsyncAnthropic:
        """Returns the native Anthropic client for direct Files API uploads.

        Callers that can run without a key use `anthropic_client_if_configured` instead.
        """
        return AsyncAnthropic(api_key=self.config.anthropic_api_key)

    @property
    def anthropic_client_if_configured(self) -> AsyncAnthropic | None:
        """Returns the direct Anthropic client, or None when `config.anthropic_key_configured` is false."""
        if not self.config.anthropic_key_configured:
            return None
        return self.anthropic_client

    @cached_property
    def voice_generator(self) -> VoiceGenerator:
        """Returns the text-to-speech generator for spoken QA replies."""
        return VoiceGenerator(
            client=self.openai_client, model_name=self.runtime_models.tts_model.name
        )

    @cached_property
    def image_generator(self) -> ImageGenerator:
        """Returns the image generator and editor bound to the proxy client."""
        return ImageGenerator(
            client=self.openai_client, image_model=self.runtime_models.image_model
        )

    @cached_property
    def prompt_generator(self) -> PromptGenerator:
        """Returns the prompt refiner for the IMAGE and VIDEO routes."""
        return PromptGenerator(
            client=self.openai_client, prompt_model=self.runtime_models.fast_model
        )

    @cached_property
    def video_generator(self) -> VideoGenerator:
        """Returns the video generator bound to the direct Gemini client."""
        return VideoGenerator(
            client=self.gemini_client, video_model=self.runtime_models.video_model
        )

    @cached_property
    def music_generator(self) -> MusicGenerator:
        """Returns the music generator for `<generate-music>`, bound to the direct Gemini client."""
        return MusicGenerator(
            client=self.gemini_client, music_model=self.runtime_models.music_model
        )

    def _attachment_bucket_key(self, model: ModelSettings) -> AttachmentBucketKey:
        """Returns the provider bucket key isolating rendered attachment caches across providers."""
        if not self.config.file_api_enabled:
            return "inline"
        if model.is_gemini and self.config.gemini_key_configured:
            return "gemini"
        if model.is_claude and self.config.anthropic_key_configured:
            return "claude"
        return "inline"

    def input_builder_for(self, model: ModelSettings) -> MessageInputBuilder:
        """Returns the cached input builder matching `model`'s provider and modality set.

        Shares one `(AttachmentRenderer, AttachmentCache)` pair per provider bucket so same-provider
        candidates reuse uploaded Files API handles across history and regenerations while
        mixed-provider candidate pools never serve one provider's rendered parts to another.

        Args:
            model (ModelSettings): Selected answer model for the turn.
        """
        builder = self._input_builders.get(model.name)
        if builder is not None:
            return builder
        bucket_key = self._attachment_bucket_key(model=model)
        bucket = self._attachment_buckets.get(bucket_key)
        if bucket is None:
            handler = build_attachment_handler(
                model=model,
                gemini_client=lambda: self.gemini_client_if_configured,
                anthropic_client=lambda: self.anthropic_client_if_configured,
                file_api_enabled=self.config.file_api_enabled,
            )
            cache: AttachmentCache = BoundedLRU()
            bucket = (handler, cache)
            self._attachment_buckets[bucket_key] = bucket
        handler, cache = bucket
        builder = MessageInputBuilder(
            bot=self.bot,
            runtime_models=self.runtime_models,
            attachment_handler=handler,
            model_name=model.name,
        )
        builder.bind_attachment_cache(cache=cache)
        self._input_builders[model.name] = builder
        return builder

    @property
    def input_builder(self) -> MessageInputBuilder:
        """Returns the input builder for the primary slow model (`slow_model`)."""
        return self.input_builder_for(model=self.runtime_models.slow_model)

    @property
    def memory_writer(self) -> MemoryWriterAI:
        """Returns the memory writer service for user and server scopes."""
        bot_user_id = self.bot.user.id if self.bot.user is not None else None
        if self._memory_writer is not None and (
            self._memory_writer.bot_user_id == bot_user_id or bot_user_id is None
        ):
            return self._memory_writer
        writer = MemoryWriterAI(
            client=self.openai_client,
            model=self.runtime_models.memory_model,
            bot_user_id=bot_user_id,
        )
        if bot_user_id is not None:
            self._memory_writer = writer
        return writer
