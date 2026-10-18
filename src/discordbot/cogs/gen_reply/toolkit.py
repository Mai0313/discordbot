"""The clients, generators and caches a reply is built out of, made once and reused.

The pipeline is handed one toolkit and reads its clients, generators, model catalog and input
builder off it, so no call site has to assemble any of that for itself. What is not here is
deliberate: `media_delivery` and `usage_recorder` stay on the cog, being about delivering a
reply rather than composing one.

One toolkit serves the whole process rather than one reply. That is the point of it: the
caches inside — the input builder's rendered-part cache and the uploader's re-poll cache —
hold Gemini Files API uris, and rebuilding them per reply would re-upload the whole history
window every time.
"""

from functools import cached_property

from google import genai
from openai import AsyncOpenAI
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from nextcord.ext import commands

from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.services.memory.writer import MemoryWriterAI
from discordbot.cogs.gen_reply.generation import (
    ImageGenerator,
    MusicGenerator,
    VideoGenerator,
    VoiceGenerator,
    PromptGenerator,
)
from discordbot.cogs.gen_reply.attachment.select import build_attachment_handler


class ReplyToolkit(BaseModel):
    """The clients, generators and caches every reply is composed with."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: SkipValidation[commands.Bot] = Field(
        ..., description="The Discord bot instance, passed through to the input builder."
    )
    openai_client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client."
    )
    gemini_api_key: str = Field(
        ...,
        description=(
            "Google AI Studio key for the direct-to-Google paths, or empty when the deployment "
            "configured none. Empty leaves those paths unavailable, and the Gemini-only features "
            "gate themselves off as they already do."
        ),
    )

    @cached_property
    def runtime_models(self) -> RuntimeModelCatalog:
        """Returns the runtime model catalog for every tier."""
        return RuntimeModelCatalog()

    @cached_property
    def gemini_client(self) -> genai.Client:
        """Returns the native Gemini client for direct-to-Google paths.

        Raises at construction when `gemini_api_key` is empty; callers that can run without a key
        use `gemini_client_if_configured` instead.
        """
        return genai.Client(api_key=self.gemini_api_key)

    @property
    def gemini_client_if_configured(self) -> genai.Client | None:
        """Returns the direct Gemini client, or None when `gemini_api_key` is blank."""
        if not self.gemini_api_key.strip():
            return None
        return self.gemini_client

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

    @cached_property
    def input_builder(self) -> MessageInputBuilder:
        """Returns the Discord-message-to-Responses-API input builder and its attachment caches."""
        return MessageInputBuilder(
            bot=self.bot,
            runtime_models=self.runtime_models,
            attachment_handler=build_attachment_handler(
                model=self.runtime_models.slow_model,
                gemini_client=lambda: self.gemini_client_if_configured,
            ),
        )

    @cached_property
    def memory_writer(self) -> MemoryWriterAI:
        """Returns the memory writer service for user and server scopes."""
        return MemoryWriterAI(
            client=self.openai_client,
            model=self.runtime_models.memory_writer_model,
            bot_user_id=self.bot.user.id if self.bot.user is not None else None,
        )
