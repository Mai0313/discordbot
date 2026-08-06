"""The single mapping from an answer model's name to the attachment renderer it can read.

`MessageInputBuilder` turns every attachment source into a content part through one injected
`AttachmentRenderer`, so the provider branch lives here instead of inside the builder: swapping
the answer model swaps the renderer, and adding a provider touches this file plus its own
renderer module. Only Gemini resolves an uploaded Files-API uri, so it gets `GeminiFileUploader`
and every other answer model falls back to `InlineRenderer`, which inlines each attachment per
type and needs no `GEMINI_API_KEY`.

The OpenAI / Anthropic / Grok branches are commented out beside their modules, which are
scaffolded rather than dead: each module's docstring lists what its reference path still needs
verified against a live model, so uncommenting a branch here is the last step of enabling one.
Do not delete them as unused code.
"""

from discordbot.cogs.gen_reply.attachment.base import AttachmentRenderer
from discordbot.cogs.gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs.gen_reply.attachment.gemini_file_api import GeminiFileUploader

# from discordbot.cogs.gen_reply.attachment.grok_file_api import GrokFileUploader
# from discordbot.cogs.gen_reply.attachment.openai_file_api import OpenAIFileUploader
# from discordbot.cogs.gen_reply.attachment.anthropic_file_api import AnthropicFileUploader


def build_attachment_handler(model_name: str) -> AttachmentRenderer:
    """Returns the attachment renderer matching the answer (slow) model's provider.

    Matched on the model string itself, since `ModelSettings` carries only the LiteLLM name and
    its effort, with no provider field to read. A non-Gemini answer model rejects a Gemini
    Files-API uri (the proxy mistranslates it), so it inlines instead. No client is injected into
    the renderer: each builds its own lazily, which is why the name is all this needs.

    Args:
        model_name (str): The answer model's LiteLLM name, i.e. `slow_model.name`.

    Returns:
        A `GeminiFileUploader` for a Gemini answer model, an `InlineRenderer` for every other.
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
