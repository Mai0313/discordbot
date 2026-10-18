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

    Only Gemini resolves an uploaded Files-API URI; OpenAI and Anthropic answer models reject
    it (the proxy mistranslates it), so they inline instead. The OpenAI, Anthropic, and Grok
    Files-API uploaders are scaffolded behind the commented branches below until their
    reference path is verified.

    `file_api_enabled` overrides the provider branch: when disabled, or when `gemini_client`
    returns `None`, every provider falls back to `InlineRenderer`.

    Args:
        model (ModelSettings): Selected answer model settings.
        gemini_client (Callable[[], genai.Client | None]): Provider for the deployment's direct
            Gemini client, or `None` when no Gemini key is configured.
    """
    if not LLMConfig().file_api_enabled:
        return InlineRenderer()
    if model.is_gemini and gemini_client() is not None:
        return GeminiFileUploader(gemini_client=gemini_client)
    # if "gpt" in model.name:
    #     return OpenAIFileUploader(model_name=model.name)
    # if "claude" in model.name:
    #     return AnthropicFileUploader()
    # if "grok" in model.name:
    #     return GrokFileUploader()
    return InlineRenderer()
