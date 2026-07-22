"""Inline attachment renderer for answer models that cannot resolve Gemini Files URIs."""

import base64
from datetime import UTC, datetime, timedelta

import logfire
from nextcord import Attachment, StickerItem
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.cogs._gen_reply.attachment.base import (
    RenderedPart,
    AttachmentRenderer,
    loggable_cache_key,
)
from discordbot.cogs._gen_reply.attachment.loaders import (
    attachment_mime,
    load_image_bytes,
    load_attachment_bytes,
    resolve_source_filename,
)


def _data_uri(data: bytes, mime_type: str) -> str:
    """Builds a base64 data URI for inlining bytes into a content part."""
    return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"


def _inline_expiry() -> datetime:
    """Cache validity for a self-contained inlined part.

    Inlined bytes never expire, but the cache key cannot see a Discord CDN re-host of the
    same source, so the render is refreshed periodically as a cheap safety net.
    """
    return datetime.now(tz=UTC) + timedelta(hours=12)


class InlineRenderer(AttachmentRenderer):
    """Inlines attachments as base64 / text parts (OpenAI / Anthropic answer models).

    Stateless: every render fetches the source and embeds it directly in the request, so
    there is no upload handle to track and the `cache_key` / `allow_dead_cache` re-poll
    arguments are ignored. Images inline as `input_image` base64, PDFs as base64
    `input_file`, UTF-8 files as `input_text`, and anything else is dropped.
    """

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> tuple[RenderedPart, datetime] | None:
        try:
            file_bytes, content_type = await load_image_bytes(source=source)
        except Exception as exc:
            # Broad on purpose: `load_image_bytes` spans a CDN fetch, a PIL decode and a
            # downscale re-encode, so the type is what names the failing step.
            logfire.warn(
                "failed to load image for inline render; dropping",
                filename=resolve_source_filename(source=source, url_fallback="image.png"),
                cache_key=loggable_cache_key(cache_key=cache_key),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        image_part = ResponseInputImageParam(
            type="input_image",
            image_url=_data_uri(data=file_bytes, mime_type=content_type),
            detail="auto",
        )
        return image_part, _inline_expiry()

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> tuple[RenderedPart, datetime] | None:
        mime_type = attachment_mime(attachment=attachment)
        if not mime_type:
            logfire.warn(
                "skipping attachment with unknown MIME type",
                filename=attachment.filename,
                url=attachment.url,
            )
            return None
        try:
            file_bytes, _ = await load_attachment_bytes(attachment=attachment)
        except Exception as exc:
            # Broad on purpose: `attachment.read()` surfaces nextcord HTTPException/NotFound,
            # aiohttp client errors and timeouts; all of them just drop this one part.
            logfire.warn(
                "failed to download attachment for inline render; dropping",
                filename=attachment.filename,
                url=attachment.url,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        return self._inline_file_part(
            filename=attachment.filename, data=file_bytes, mime_type=mime_type
        )

    def _inline_file_part(
        self, filename: str, data: bytes, mime_type: str
    ) -> tuple[RenderedPart, datetime] | None:
        """Inlines a non-image file, or drops it.

        PDFs inline as base64 `input_file` (the one document type OpenAI / Anthropic accept
        inline); UTF-8-decodable files inline as `input_text` with a filename header; anything
        else (non-text binaries the Gemini Files path would have uploaded) is dropped.
        """
        if mime_type == "application/pdf":
            pdf_part = ResponseInputFileParam(
                type="input_file",
                filename=filename,
                file_data=_data_uri(data=data, mime_type=mime_type),
            )
            return pdf_part, _inline_expiry()
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
        return text_part, _inline_expiry()
