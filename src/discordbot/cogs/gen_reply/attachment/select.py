"""Selects the attachment renderer that matches the current answer model's provider."""

from collections.abc import Callable

from google import genai
from anthropic import AsyncAnthropic

from discordbot.typings.models import ModelSettings
from discordbot.cogs.gen_reply.attachment.base import AttachmentRenderer
from discordbot.cogs.gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs.gen_reply.attachment.gemini_file_api import GeminiFileUploader
from discordbot.cogs.gen_reply.attachment.anthropic_file_api import AnthropicFileUploader

# from discordbot.cogs.gen_reply.attachment.grok_file_api import GrokFileUploader
# from discordbot.cogs.gen_reply.attachment.openai_file_api import OpenAIFileUploader


def build_attachment_handler(
    model: ModelSettings,
    gemini_client: Callable[[], genai.Client | None],
    anthropic_client: Callable[[], AsyncAnthropic | None],
    file_api_enabled: bool = True,
) -> AttachmentRenderer:
    """Returns the attachment renderer matching the answer (slow) model's provider.

    Gemini and Claude answer models upload via their direct Files APIs when `file_api_enabled` is
    true and their provider key is configured; OpenAI and Grok inline until their uploaders'
    reference paths are verified.

    `file_api_enabled` overrides every provider branch: when disabled, or when the provider's
    client getter returns `None`, that provider falls back to `InlineRenderer`.

    Args:
        model (ModelSettings): Selected answer model settings.
        gemini_client (Callable[[], genai.Client | None]): Provider for the deployment's direct
            Gemini client, or `None` when no Gemini key is configured.
        anthropic_client (Callable[[], AsyncAnthropic | None]): Provider for the deployment's
            direct Anthropic client, or `None` when no Anthropic key is configured.
        file_api_enabled (bool): Whether provider Files API uploads are enabled.
    """
    if not file_api_enabled:
        return InlineRenderer()
    if model.is_gemini and gemini_client() is not None:
        return GeminiFileUploader(gemini_client=gemini_client)
    if model.is_claude and anthropic_client() is not None:
        return AnthropicFileUploader(anthropic_client=anthropic_client)
    # if "gpt" in model.name:
    #     return OpenAIFileUploader(model_name=model.name)
    # if "grok" in model.name:
    #     return GrokFileUploader()
    return InlineRenderer()
