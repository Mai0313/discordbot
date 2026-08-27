"""The clients, generators and caches a reply is built out of, made once and reused.

The pipeline is handed one toolkit and reads its clients, generators, model catalog and input
builder off it, so no call site has to assemble any of that for itself. What is NOT here is
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
from discordbot.services.memory.server_prompts import (
    SERVER_PHASE2_PROMPT,
    SERVER_PHASE1_EVALUATOR_PROMPT,
)
from discordbot.cogs.gen_reply.attachment.select import build_attachment_handler


class ReplyToolkit(BaseModel):
    """The clients, generators and caches every reply is composed with.

    Attributes:
        bot: The Discord bot instance, passed through to the input builder.
        openai_client: The shared LiteLLM-proxy client.
        gemini_api_key: The Google AI Studio key for the direct-to-Google paths, or empty
            when the deployment configured none. Empty leaves those paths unavailable, and
            the Gemini-only features gate themselves off as they already do.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: SkipValidation[commands.Bot] = Field(
        ..., description="The Discord bot instance, passed through to the input builder."
    )
    openai_client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client."
    )
    gemini_api_key: str = Field(
        ..., description="Google AI Studio key for direct-to-Google paths; empty when unset."
    )

    @cached_property
    def runtime_models(self) -> RuntimeModelCatalog:
        """The model catalog every tier is read off.

        Returns:
            The runtime catalog.
        """
        return RuntimeModelCatalog()

    @cached_property
    def gemini_client(self) -> genai.Client:
        """The native Gemini client for every DIRECT-to-Google path.

        DIRECT to Google (no proxy): it serves the runtime paths the LiteLLM proxy cannot.

        - native omni video generation / editing (`interactions.create`, delivery=uri + Files
          download), for both the VIDEO route and the inline `<generate-video>` marker;
        - the inline `<generate-music>` Lyria render, also on the Interactions API;
        - Files API uploads, so a generated clip (and, through
          `gemini_client_if_configured`, a linked post's media) can be referenced by uri; and
        - the YouTube-aware QA answer turn that streams through the native Interactions API
          (the only path that can actually watch a linked video). That last swap only ever
          fires when the answer model is already Gemini, so the direct credential is always
          the right one.

        All of them forgo proxy-side cost/usage tracking, like the deep-research direct path.
        An empty key raises at construction, so a caller that is reachable without one must go
        through `gemini_client_if_configured` instead of touching this.

        Returns:
            A Gemini client for native media generation and the Interactions answer turn.
        """
        return genai.Client(api_key=self.gemini_api_key)

    @property
    def gemini_client_if_configured(self) -> genai.Client | None:
        """The direct Gemini client, or None when no key is configured.

        For the paths that stay useful without a key: a linked post still contributes its
        text, it just carries no uploaded media. Reading `gemini_client` there would raise
        before the feature's own kill-switch was ever consulted.

        Returns:
            The client, or None when this toolkit holds no key.
        """
        if not self.gemini_api_key.strip():
            return None
        return self.gemini_client

    @cached_property
    def voice_generator(self) -> VoiceGenerator:
        """The text-to-speech engine for spoken QA replies.

        Returns:
            A generator bound to the proxy client and the TTS deployment; the caller
            still gates it on `allow_voice` and `config.inline_voice_enabled`.
        """
        return VoiceGenerator(
            client=self.openai_client, model_name=self.runtime_models.tts_model.name
        )

    @cached_property
    def image_generator(self) -> ImageGenerator:
        """The image renderer shared by the IMAGE route and the `<generate-image>` marker.

        Returns:
            A generator bound to the proxy client and the image deployment; the route
            calls `render` (raises) while the inline path calls `generate` (best-effort,
            gated on `allow_image` and `config.inline_image_enabled`).
        """
        return ImageGenerator(
            client=self.openai_client, image_model=self.runtime_models.image_model
        )

    @cached_property
    def prompt_generator(self) -> PromptGenerator:
        """The prompt director for the IMAGE and VIDEO routes.

        Returns:
            A director bound to the proxy client and the grounding-capable `fast_model`; each
            `refine` call is gated by the caller's per-route flag
            (`config.image_refine_prompt_enabled` / `config.video_refine_prompt_enabled`) and
            expands the raw request before `render`, best-effort (raw prompt on disable /
            empty / error).
        """
        return PromptGenerator(
            client=self.openai_client, prompt_model=self.runtime_models.fast_model
        )

    @cached_property
    def video_generator(self) -> VideoGenerator:
        """The video renderer shared by the VIDEO route and the `<generate-video>` marker.

        Returns:
            A generator bound to the DIRECT-to-Google Gemini client and the video model
            (the Interactions API is Gemini-only, not reachable via the proxy); the route
            calls `render` (raises) while the inline path calls `generate` (best-effort, gated
            on `allow_video` and `config.video_available`).
        """
        return VideoGenerator(
            client=self.gemini_client, video_model=self.runtime_models.video_model
        )

    @cached_property
    def music_generator(self) -> MusicGenerator:
        """The music renderer for the QA-route `<generate-music>` marker.

        Returns:
            A generator bound to the DIRECT-to-Google Gemini client (Lyria runs on the
            Interactions API, not the proxy) and the music model; the inline path calls
            `generate` (best-effort, gated on `allow_music` and `config.music_available`).
        """
        return MusicGenerator(
            client=self.gemini_client, music_model=self.runtime_models.music_model
        )

    @cached_property
    def input_builder(self) -> MessageInputBuilder:
        """The Discord-message-to-Responses-API input builder.

        This is the piece that makes the toolkit worth holding onto: its rendered-part cache
        keeps the Files API uris a message's attachments were uploaded to, so a message that
        stays in the history window is uploaded once rather than once per reply.

        Returns:
            A builder bound to this bot, the runtime model catalog, and an attachment handler
            holding the direct Gemini credential.
        """
        return MessageInputBuilder(
            bot=self.bot,
            runtime_models=self.runtime_models,
            attachment_handler=build_attachment_handler(
                model_name=self.runtime_models.slow_model.name, gemini_api_key=self.gemini_api_key
            ),
        )

    @cached_property
    def memory_writer(self) -> MemoryWriterAI:
        """The per-user memory writing service.

        Returns:
            A writer bound to the proxy client and the memory deployments.
        """
        return MemoryWriterAI(
            client=self.openai_client,
            evaluate_model=self.runtime_models.memory_writer_model,
            consolidate_model=self.runtime_models.memory_writer_model,
        )

    @cached_property
    def server_memory_writer(self) -> MemoryWriterAI:
        """The per-server (bot self) memory writing service.

        Returns:
            A writer sharing the per-user models and client but driving the server-flavor
            prompts, so the bot builds community-level memory per guild through the same
            engine.
        """
        return MemoryWriterAI(
            client=self.openai_client,
            evaluate_model=self.runtime_models.memory_writer_model,
            consolidate_model=self.runtime_models.memory_writer_model,
            evaluator_prompt=SERVER_PHASE1_EVALUATOR_PROMPT,
            consolidate_prompt=SERVER_PHASE2_PROMPT,
        )


__all__ = ["ReplyToolkit"]
