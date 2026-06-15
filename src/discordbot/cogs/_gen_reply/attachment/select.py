"""Selects the attachment renderer that matches the current answer model's provider."""

from google import genai

from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs._gen_reply.attachment.base import AttachmentRenderer
from discordbot.cogs._gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs._gen_reply.attachment.gemini_file_api import GeminiFileUploader


def build_attachment_handler(
    runtime_models: RuntimeModelCatalog, gemini_client: genai.Client | None
) -> AttachmentRenderer:
    """Returns the attachment renderer matching the answer (slow) model's provider.

    Only Gemini resolves an uploaded Files-API URI; OpenAI / Anthropic answer models reject
    it (the proxy mistranslates it), so they inline instead. This is the single place that
    maps an answer model to its attachment handling, so adding a provider changes only here.
    """
    if "gemini" in runtime_models.slow_model.name:
        return GeminiFileUploader(gemini_client=gemini_client)
    return InlineRenderer()
