"""Everything one leased Gemini key owns, built once per key and reused.

A reply leases a key and then does every Gemini call of that reply on it, because a Files API
file is readable only by the project that uploaded it and one request naming files from two
keys fails outright. This type is what makes that hold without every call site having to know
a key exists: the pipeline is handed one toolkit and reads its clients, generators, model
catalog and input builder off it.

Everything here is bound to the key. What is NOT here is deliberate: `openai_client` (the key
lives in the model name on that path, so one client serves every key), `media_delivery` and
`usage_recorder` (nothing about them is per-key) all stay on the cog.

Toolkits are cached per key for the life of the process, not per reply. That is the point of
them: the caches inside — the input builder's rendered-part cache and the uploader's
re-poll cache — hold uris that are only valid for this key, and rebuilding them every reply
would re-upload the whole history window every time.
"""

from functools import cached_property

from google import genai
from openai import AsyncOpenAI
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from nextcord.ext import commands

from discordbot.typings.llm import GeminiKeySlot
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


class GeminiKeyToolkit(BaseModel):
    """The clients, generators and caches bound to one Gemini key.

    Attributes:
        bot: The Discord bot instance, passed through to the input builder.
        openai_client: The shared proxy client; the key rides in the model name, not here.
        slot: The leased key, or None when the deployment has no Gemini key configured. None
            leaves every tier unpinned and every direct-to-Google path unavailable, which is
            exactly what the bot did before it balanced anything.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: SkipValidation[commands.Bot] = Field(
        ..., description="The Discord bot instance, passed through to the input builder."
    )
    openai_client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client; the key rides in the model name."
    )
    slot: GeminiKeySlot | None = Field(
        ..., description="The leased Gemini key, or None when none is configured."
    )

    @property
    def key_index(self) -> int | None:
        """The leased key's number, or None when unpinned. Logged, never dispatched."""
        return self.slot.index if self.slot is not None else None

    @cached_property
    def runtime_models(self) -> RuntimeModelCatalog:
        """The model catalog with every tier pinned to this key.

        Returns:
            A catalog whose tiers carry `key_index`, so each one's `deployment_name` names
            this key's proxy deployment.
        """
        return RuntimeModelCatalog(key_index=self.key_index)

    @cached_property
    def gemini_client(self) -> genai.Client:
        """The native Gemini client for every DIRECT-to-Google path on this key.

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
        return genai.Client(api_key=self.slot.api_key if self.slot is not None else "")

    @property
    def gemini_client_if_configured(self) -> genai.Client | None:
        """The direct Gemini client, or None when no key is configured.

        For the paths that stay useful without a key: a linked post still contributes its
        text, it just carries no uploaded media. Reading `gemini_client` there would raise
        before the feature's own kill-switch was ever consulted.

        Returns:
            The client, or None when this toolkit holds no key.
        """
        if self.slot is None:
            return None
        return self.gemini_client

    @cached_property
    def voice_generator(self) -> VoiceGenerator:
        """The text-to-speech engine for spoken QA replies.

        Returns:
            A generator bound to the proxy client and this key's TTS deployment; the caller
            still gates it on `allow_voice` and `config.inline_voice_enabled`.
        """
        return VoiceGenerator(
            client=self.openai_client, model_name=self.runtime_models.tts_model.deployment_name
        )

    @cached_property
    def image_generator(self) -> ImageGenerator:
        """The image renderer shared by the IMAGE route and the `<generate-image>` marker.

        Returns:
            A generator bound to the proxy client and this key's image deployment; the route
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
            A generator bound to this key's DIRECT-to-Google Gemini client and the video model
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
            A generator bound to this key's DIRECT-to-Google Gemini client (Lyria runs on the
            Interactions API, not the proxy) and the music model; the inline path calls
            `generate` (best-effort, gated on `allow_music` and `config.music_available`).
        """
        return MusicGenerator(
            client=self.gemini_client, music_model=self.runtime_models.music_model
        )

    @cached_property
    def input_builder(self) -> MessageInputBuilder:
        """The Discord-message-to-Responses-API input builder for this key.

        Per key rather than shared, and this is the piece that makes the whole design
        necessary: its rendered-part cache holds Files API uris, which only the key that
        uploaded them can read.

        Returns:
            A builder bound to this bot, this key's pinned model catalog, and an attachment
            handler holding this key's credential.
        """
        return MessageInputBuilder(
            bot=self.bot,
            runtime_models=self.runtime_models,
            attachment_handler=build_attachment_handler(
                model_name=self.runtime_models.slow_model.name,
                gemini_api_key=self.slot.api_key if self.slot is not None else "",
            ),
        )

    @cached_property
    def memory_writer(self) -> MemoryWriterAI:
        """The per-user memory writing service.

        Returns:
            A writer bound to the proxy client and this key's memory deployments.
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


__all__ = ["GeminiKeyToolkit"]
