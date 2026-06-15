"""The attachment renderer strategy interface and its shared rendered-part type."""

from datetime import datetime

from nextcord import Attachment, StickerItem
from pydantic import BaseModel, ConfigDict
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

# A rendered attachment content part. The Gemini answer model reads a Files-API handle
# (input_file with a file URI); non-Gemini answer models cannot resolve that URI, so their
# attachments are inlined per type instead: images as input_image base64, PDFs as input_file
# base64 file_data, and text/code files as input_text.
type RenderedPart = ResponseInputTextParam | ResponseInputImageParam | ResponseInputFileParam


class AttachmentRenderer(BaseModel):
    """Strategy that turns one Discord attachment source into a Responses API content part.

    Each implementation owns one way to make an attachment readable by the answer model
    (Gemini Files-API upload, or per-type inline base64), so the answer model's provider is
    swapped by injecting a different renderer into `MessageInputBuilder`, not by branching
    inside it. Both methods return the rendered part plus the cache expiry the per-message
    render cache reuses it until, or None when the source is dropped (unsupported / failed).
    `cache_key` and `allow_dead_cache` drive the Gemini uploader's re-poll / dead-source
    caches; a stateless renderer accepts them for interface parity and ignores them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> tuple[RenderedPart, datetime] | None:
        """Renders an image source (attachment, sticker, or URL) to a content part."""
        raise NotImplementedError

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> tuple[RenderedPart, datetime] | None:
        """Renders a non-image file attachment to a content part."""
        raise NotImplementedError
