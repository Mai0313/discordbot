"""Inline attachment renderer for answer models that cannot resolve Gemini Files URIs.

The non-Gemini half of the provider-aware attachment path: `select.py` picks between this and
`GeminiFileUploader` off the answer model's name. A Gemini Files uri is resolvable only by
Gemini (the LiteLLM proxy mistranslates it for anyone else), so an OpenAI / Anthropic / Grok
answer model gets its attachments embedded straight into the request instead — images as base64
`input_image`, PDFs as base64 `input_file`, UTF-8-decodable files as `input_text` under a
filename header, and everything else dropped.

Making that last drop decision here rather than in the source collector is deliberate.
`input.py::_attachment_kind` denylists only the types known to 400 and proxies everything else
as `image` / `input_file`, because MIME cannot reliably tell an unlisted binary apart from
unlisted code; the final narrowing therefore belongs where the provider's real inline capability
is known. The set that survives is smaller than what the Gemini uploader ingests (no video, no
audio, no unknown binary), which is the price of having no upload side-channel.

Kept apart from `gemini_file_api.py` because it is entirely stateless: no upload handle, no
activation poll, no pending / dead-source bookkeeping, and no `GEMINI_API_KEY`. It inherits
`AttachmentRenderer`'s caches for interface parity and never touches them, so the per-message
render cache in `input.py` is the only thing keeping a source from being re-fetched every reply.
"""

import base64
from datetime import UTC, datetime, timedelta

import logfire
from nextcord import Attachment, StickerItem
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.cogs.gen_reply.attachment.base import (
    RenderedPart,
    AttachmentRenderer,
    loggable_cache_key,
)
from discordbot.cogs.gen_reply.attachment.loaders import (
    attachment_mime,
    load_image_bytes,
    load_attachment_bytes,
    resolve_source_filename,
)


def _data_uri(data: bytes, mime_type: str) -> str:
    """Builds a base64 data URI for inlining bytes into a content part.

    `tests/test_model_media_parts.py` allowlists this helper by name: a media part whose source
    is anything but a local data URI fails that guard, so keep media parts built through here.

    Args:
        data (bytes): The raw bytes to embed.
        mime_type (str): The MIME type the part declares for those bytes.

    Returns:
        A `data:<mime_type>;base64,<payload>` URI.
    """
    return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"


def _inline_expiry() -> datetime:
    """Cache validity for a self-contained inlined part.

    Inlined bytes never expire, but the cache key cannot see a Discord CDN re-host of the
    same source, so the render is refreshed periodically as a cheap safety net.

    Returns:
        Twelve hours out; `input.py`'s render cache reuses the part until shortly before that.
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
        """Fetches and downscales an image source, then inlines it as base64 `input_image`.

        A fetch or decode failure drops this one source instead of raising: the caller gathers
        the renders without `return_exceptions`, so an escaping error would cost the whole
        message its attachments. `cache_key` only labels the failure log and `allow_dead_cache`
        is ignored, since a stateless renderer keeps no dead-source cache to consult.

        Args:
            source (Attachment | StickerItem | str): The image attachment, sticker, or URL.
            cache_key (int | str): The source's cache key; logged, never stored.
            allow_dead_cache (bool): Accepted for interface parity and ignored.

        Returns:
            The `input_image` part plus its cache expiry, or None when the source could not be
            loaded.
        """
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
        """Downloads a non-image attachment and inlines it per type, or drops it.

        An unguessable MIME type is dropped before the download, since the part shape is chosen
        entirely from that type. A download failure drops this one attachment for the same
        reason `render_image` does. Neither cache argument is read: this renderer is stateless.

        Args:
            attachment (Attachment): The non-image attachment to inline.
            cache_key (int | str): Accepted for interface parity and unused.
            allow_dead_cache (bool): Accepted for interface parity and unused.

        Returns:
            The inlined content part plus its cache expiry, or None when the attachment was
            dropped.
        """
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
        else (non-text binaries the Gemini Files path would have uploaded) is dropped. The
        decode attempt, not the declared MIME, is what settles text-vs-binary, because
        `input.py` deliberately forwards every unlisted type here rather than guessing.

        Args:
            filename (str): The attachment's filename, kept in the `input_text` header so the
                model can name the file it is reading.
            data (bytes): The downloaded file bytes.
            mime_type (str): The attachment's resolved MIME type, which selects the part shape.

        Returns:
            The `input_file` or `input_text` part plus its cache expiry, or None when the file
            is neither a PDF nor UTF-8 text.
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
