"""Shared byte loaders for attachment rendering (image fetch + downscale, mime resolution).

Used by both renderer strategies and by the IMAGE route's raw-bytes path, so the
download/downscale logic lives in one place independent of how the bytes are later consumed.
"""

import asyncio
from mimetypes import guess_type

from nextcord import Attachment, StickerItem

from discordbot.utils.images import get_image_data, shrink_image_bytes


async def load_image_bytes(source: Attachment | StickerItem | str) -> tuple[bytes, str]:
    """Fetches and downscales an image source to upload-ready bytes and MIME type.

    URL sources fetch over the network and attachments decode/re-encode, so the blocking
    work runs off the event loop. Raises on any fetch/decode failure. Callers bound their
    own concurrency (the Gemini uploader's media semaphore) or fetch a single current-turn
    image (the IMAGE route, the inline render).
    """
    if isinstance(source, str):
        file_bytes = await asyncio.to_thread(get_image_data, image_file=source)
        return file_bytes, "image/jpeg"
    if isinstance(source, Attachment):
        content_type = source.content_type or guess_type(source.filename)[0] or "image/png"
    else:
        content_type = guess_type(source.url)[0] or "image/png"
    file_bytes = await source.read()
    return await asyncio.to_thread(
        shrink_image_bytes, payload=file_bytes, content_type=content_type
    )


def resolve_source_filename(source: Attachment | StickerItem | str, *, url_fallback: str) -> str:
    """Returns the upload filename for an image source (attachment, sticker, or URL).

    A URL or embed image has no filename, so `url_fallback` (an image-extensioned name) is
    used; a downstream filename-only classifier then reads it as an image, not a document.
    An attachment keeps its real filename; a sticker synthesizes `<name>.png`.
    """
    if isinstance(source, str):
        return url_fallback
    return getattr(source, "filename", None) or f"{getattr(source, 'name', 'sticker')}.png"


def attachment_mime(attachment: Attachment) -> str:
    """Returns the bare MIME type of a file attachment, empty when unguessable."""
    content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
    return content_type.split(";")[0].strip()


async def load_attachment_bytes(attachment: Attachment) -> tuple[bytes, str]:
    """Reads a file attachment's bytes alongside its resolved MIME type."""
    return await attachment.read(), attachment_mime(attachment=attachment)
