"""Inline attachment renderer for answer models that cannot resolve Gemini Files URIs."""

from datetime import UTC, datetime, timedelta

import logfire
from nextcord import Attachment, StickerItem
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.images import to_data_uri
from discordbot.typings.media import RenderedAttachment
from discordbot.cogs.gen_reply.attachment.base import AttachmentRenderer
from discordbot.cogs.gen_reply.attachment.loaders import (
    attachment_mime,
    load_image_bytes,
    load_attachment_bytes,
    resolve_source_filename,
)


def _inline_expiry() -> datetime:
    """Cache validity for a self-contained inlined part.

    Inlined bytes never expire, but the cache key cannot see a Discord CDN re-host of the
    same source, so the render is refreshed periodically as a cheap safety net.
    """
    return datetime.now(tz=UTC) + timedelta(hours=12)


class InlineRenderer(AttachmentRenderer):
    """Inlines attachments as base64 / text parts.

    Selected for any non-Gemini answer model, none of which can resolve a Gemini Files URI,
    and for every provider (Gemini included) while `file_api_enabled` is off. Stateless:
    every render fetches the source and embeds it directly in the request, so there is no
    upload handle to track; `allow_dead_cache` is ignored and `cache_key` only labels a
    failure log. Images inline as `input_image` base64, PDFs as base64 `input_file`, UTF-8
    files as `input_text`, and anything else is dropped.
    """

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> RenderedAttachment | None:
        loaded = await self._load_source_bytes(
            cache_key=cache_key,
            filename=resolve_source_filename(source=source, url_fallback="image.png"),
            load_data=lambda: load_image_bytes(source=source),
            allow_dead_cache=False,
        )
        if loaded is None:
            return None
        image_part = ResponseInputImageParam(
            type="input_image",
            image_url=to_data_uri(data=loaded.data, mime_type=loaded.mime_type),
            detail="auto",
        )
        return RenderedAttachment(part=image_part, expires_at=_inline_expiry())

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        mime_type = attachment_mime(attachment=attachment)
        if not mime_type:
            logfire.warn(
                "skipping attachment with unknown MIME type",
                filename=attachment.filename,
                url=attachment.url,
            )
            return None
        if mime_type.startswith(("video/", "audio/")):
            # `_inline_file_part` drops these anyway, and reaching it means downloading the
            # whole clip first. Free until `file_api_enabled` made this renderer reachable for
            # a Gemini answer model: every other provider's modality gate (`_supported_sources`,
            # keyed on the slow model) already rejects them before any renderer runs. A dropped
            # part also keeps the whole message out of the render cache, so without this the
            # clip is re-downloaded on every single reply.
            logfire.warn(
                "dropping video / audio attachment the inline renderer cannot carry",
                filename=attachment.filename,
                mime_type=mime_type,
            )
            return None
        loaded = await self._load_source_bytes(
            cache_key=cache_key,
            filename=attachment.filename,
            load_data=lambda: load_attachment_bytes(attachment=attachment),
            allow_dead_cache=False,
        )
        if loaded is None:
            return None
        return self._inline_file_part(
            filename=attachment.filename, data=loaded.data, mime_type=mime_type
        )

    def _inline_file_part(
        self, filename: str, data: bytes, mime_type: str
    ) -> RenderedAttachment | None:
        """Inlines a non-image file, or drops it.

        PDFs inline as base64 `input_file` (the one document type OpenAI / Anthropic accept
        inline); UTF-8-decodable files inline as `input_text` with a filename header; anything
        else (non-text binaries the Gemini Files path would have uploaded) is dropped.
        """
        if mime_type == "application/pdf":
            pdf_part = ResponseInputFileParam(
                type="input_file",
                filename=filename,
                file_data=to_data_uri(data=data, mime_type=mime_type),
            )
            return RenderedAttachment(part=pdf_part, expires_at=_inline_expiry())
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            logfire.warn(
                "dropping non-text, non-PDF attachment for a non-Gemini model",
                filename=filename,
                mime_type=mime_type,
            )
            return None
        text_part = ResponseInputTextParam(
            type="input_text", text=f"[attached file: {filename}]\n{text}"
        )
        return RenderedAttachment(part=text_part, expires_at=_inline_expiry())
