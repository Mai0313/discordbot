"""Selects the attachment renderer that matches the current answer model's provider."""

from discordbot.cogs._gen_reply.attachment.base import AttachmentRenderer
from discordbot.cogs._gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs._gen_reply.attachment.gemini_file_api import GeminiFileUploader


def build_attachment_handler(model_name: str) -> AttachmentRenderer:
    """Returns the attachment renderer matching the answer (slow) model's provider.

    Only Gemini resolves an uploaded Files-API URI; OpenAI / Anthropic answer models reject
    it (the proxy mistranslates it), so they inline instead. This is the single place that
    maps an answer model to its attachment handling, so adding a provider changes only here.
    The Gemini uploader builds its own Files API client lazily, so this only needs the name.
    """
    if "gemini" in model_name:
        return GeminiFileUploader()
    return InlineRenderer()
