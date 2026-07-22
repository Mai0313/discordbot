"""Selects the attachment renderer that matches the current answer model's provider."""

from discordbot.cogs._gen_reply.attachment.base import AttachmentRenderer
from discordbot.cogs._gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs._gen_reply.attachment.gemini_file_api import GeminiFileUploader

# from discordbot.cogs._gen_reply.attachment.grok_file_api import GrokFileUploader
# from discordbot.cogs._gen_reply.attachment.openai_file_api import OpenAIFileUploader
# from discordbot.cogs._gen_reply.attachment.anthropic_file_api import AnthropicFileUploader


def build_attachment_handler(model_name: str) -> AttachmentRenderer:
    """Returns the attachment renderer matching the answer (slow) model's provider.

    Only Gemini resolves an uploaded Files-API URI; OpenAI / Anthropic answer models reject
    it (the proxy mistranslates it), so they inline instead. The OpenAI, Anthropic and Grok
    Files-API uploaders are scaffolded behind the commented branches below until their
    reference path is verified. This is the single place that maps an answer model to its
    attachment handling, so adding a provider changes only here. Each uploader builds its own
    Files API client lazily, so this only needs the name.
    """
    if "gemini" in model_name:
        return GeminiFileUploader()
    # if "gpt" in model_name:
    #     return OpenAIFileUploader(model_name=model_name)
    # if "claude" in model_name:
    #     return AnthropicFileUploader()
    # if "grok" in model_name:
    #     return GrokFileUploader()
    return InlineRenderer()
