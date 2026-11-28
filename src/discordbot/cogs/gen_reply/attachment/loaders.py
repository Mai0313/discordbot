"""Shared byte loaders for attachment rendering (image fetch + downscale, mime resolution).

The download and downscale live in one place, independent of how the bytes are later consumed.
"""

import asyncio
from pathlib import Path
from mimetypes import guess_type

from nextcord import Attachment, StickerItem

from discordbot.typings.media import LoadedMedia
from discordbot.cogs.gen_reply.images import get_image_data, shrink_image_bytes


def _load_local_image(path: Path) -> LoadedMedia:
    """Reads and downscales a local image file off the event loop."""
    content_type = guess_type(path.name)[0] or "image/jpeg"
    return shrink_image_bytes(
        payload=path.read_bytes(), content_type=content_type, filename=path.name
    )


async def load_image_bytes(source: Attachment | StickerItem | Path | str) -> LoadedMedia:
    """Returns downscaled, upload-ready bytes and MIME type for `source`.

    Runs network fetches for URL sources, local disk reads for `Path` sources, and PIL
    decode/re-encode off the event loop without bounding concurrency.

    Args:
        source (Attachment | StickerItem | Path | str): Image attachment, sticker, local path, or
            URL to load.
    """
    if isinstance(source, str):
        return await asyncio.to_thread(get_image_data, image_file=source)
    if isinstance(source, Path):
        return await asyncio.to_thread(_load_local_image, path=source)
    if isinstance(source, Attachment):
        content_type = attachment_mime(attachment=source) or "image/png"
    else:
        content_type = guess_type(source.url)[0] or "image/png"
    file_bytes = await source.read()  # noqa: ASYNC240 -- Path returned above; source is Attachment | StickerItem
    return await asyncio.to_thread(
        shrink_image_bytes,
        payload=file_bytes,
        content_type=content_type,
        filename=resolve_source_filename(source=source, url_fallback="image.png"),
    )


def resolve_source_filename(
    source: Attachment | StickerItem | Path | str, url_fallback: str
) -> str:
    """Returns the upload filename for an image attachment, sticker (`<name>.png`), path, or URL.

    Args:
        source (Attachment | StickerItem | Path | str): Image attachment, sticker, local path, or
            URL.
        url_fallback (str): Filename to use when `source` is a URL or lacks a filename.
    """
    if isinstance(source, str):
        return url_fallback
    if isinstance(source, Path):
        return source.name
    return getattr(source, "filename", None) or f"{getattr(source, 'name', 'sticker')}.png"


def attachment_mime(attachment: Attachment) -> str:
    """Returns the bare, lowercase MIME type of `attachment`, or `""` when unknown.

    Args:
        attachment (Attachment): Discord file attachment to inspect.
    """
    content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
    return content_type.split(";")[0].strip().lower()


async def load_attachment_bytes(attachment: Attachment) -> LoadedMedia:
    """Returns `attachment`'s raw bytes and resolved MIME type.

    Args:
        attachment (Attachment): Discord file attachment to read.
    """
    return LoadedMedia(
        data=await attachment.read(), mime_type=attachment_mime(attachment=attachment)
    )
