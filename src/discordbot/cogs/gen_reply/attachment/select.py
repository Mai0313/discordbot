"""Selects the attachment renderer that matches the current answer model's provider."""

from collections.abc import Callable

from google import genai

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs.gen_reply.attachment.base import AttachmentRenderer
from discordbot.cogs.gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs.gen_reply.attachment.gemini_file_api import GeminiFileUploader

# from discordbot.cogs.gen_reply.attachment.grok_file_api import GrokFileUploader
# from discordbot.cogs.gen_reply.attachment.openai_file_api import OpenAIFileUploader
# from discordbot.cogs.gen_reply.attachment.anthropic_file_api import AnthropicFileUploader


def build_attachment_handler(
    model: ModelSettings, gemini_client: Callable[[], genai.Client | None]
) -> AttachmentRenderer:
    """Returns the attachment renderer matching the answer (slow) model's provider.

    Only Gemini resolves an uploaded Files-API URI; OpenAI / Anthropic answer models reject
    it (the proxy mistranslates it), so they inline instead. The OpenAI, Anthropic and Grok
    Files-API uploaders are scaffolded behind the commented branches below until their
    reference path is verified. This is the single place that maps an answer model to its
    attachment handling, so adding a provider changes only here.

    `gemini_client` hands back the deployment's own direct client, so the one credential a
    deployment answers on is the one it uploads with: an uploaded file is readable only by the
    project that uploaded it, so an uploader holding a different key from the deployment behind
    the answer model fails the whole request. None is the no-key case, where each attachment is
    dropped.

    `file_api_enabled` overrides the provider branch entirely: a provider whose Files API is
    refusing to resolve references costs the WHOLE reply, since the answer carries the failing
    part, so the switch trades video / audio ingestion (which `InlineRenderer` drops) for
    replies that still land. Flipping it takes a restart, like every other setting here:
    `.env` is read at import and the one production caller is a `cached_property`.
    """
    if not LLMConfig().file_api_enabled:
        return InlineRenderer()
    if model.is_gemini:
        return GeminiFileUploader(gemini_client=gemini_client)
    # if "gpt" in model.name:
    #     return OpenAIFileUploader(model_name=model.name)
    # if "claude" in model.name:
    #     return AnthropicFileUploader()
    # if "grok" in model.name:
    #     return GrokFileUploader()
    return InlineRenderer()
