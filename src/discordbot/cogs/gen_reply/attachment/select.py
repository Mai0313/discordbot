"""Selects the attachment renderer that matches the current answer model's provider."""

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.gen_reply.attachment.base import AttachmentRenderer
from discordbot.cogs.gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs.gen_reply.attachment.gemini_file_api import GeminiFileUploader

# from discordbot.cogs.gen_reply.attachment.grok_file_api import GrokFileUploader
# from discordbot.cogs.gen_reply.attachment.openai_file_api import OpenAIFileUploader
# from discordbot.cogs.gen_reply.attachment.anthropic_file_api import AnthropicFileUploader


def build_attachment_handler(model_name: str, gemini_api_key: str) -> AttachmentRenderer:
    """Returns the attachment renderer matching the answer (slow) model's provider.

    Only Gemini resolves an uploaded Files-API URI; OpenAI / Anthropic answer models reject
    it (the proxy mistranslates it), so they inline instead. The OpenAI, Anthropic and Grok
    Files-API uploaders are scaffolded behind the commented branches below until their
    reference path is verified. This is the single place that maps an answer model to its
    attachment handling, so adding a provider changes only here.

    `gemini_api_key` is the reply's leased key, and it has to be passed rather than read from
    the environment here: the uploaded file is readable only by the project that uploaded it,
    so an uploader on a different key from the answer's `-key<n>` deployment fails the whole
    request. An empty string is the unbalanced case (no key configured), which behaves as it
    always did — the client raises lazily and the attachment is dropped.

    `file_api_enabled` overrides the provider branch entirely: a provider whose Files API is
    refusing to resolve references costs the WHOLE reply, since the answer carries the failing
    part, so the switch trades video / audio ingestion (which `InlineRenderer` drops) for
    replies that still land. Flipping it takes a restart, like every other setting here:
    `.env` is read at import and the one production caller is a `cached_property`.
    """
    if not LLMConfig().file_api_enabled:
        return InlineRenderer()
    if "gemini" in model_name:
        return GeminiFileUploader(api_key=gemini_api_key)
    # if "gpt" in model_name:
    #     return OpenAIFileUploader(model_name=model_name)
    # if "claude" in model_name:
    #     return AnthropicFileUploader()
    # if "grok" in model_name:
    #     return GrokFileUploader()
    return InlineRenderer()
