"""Shared byte loaders for attachment rendering (image fetch + downscale, MIME resolution).

Four helpers, all Discord-aware and consumer-agnostic. `load_image_bytes` turns an attachment, a
sticker or an image url into upload-ready bytes plus the MIME type that describes them;
`load_attachment_bytes` does the same for a non-image file, with no decode and no re-encode;
`attachment_mime` resolves the bare type a renderer gates on; `resolve_source_filename` names a
source that has no name of its own.

They live here rather than inside one renderer because the same bytes are consumed three
different ways: uploaded to a provider Files API and referenced by handle (`gemini_file_api.py`
and the scaffolded OpenAI / Anthropic / Grok uploaders), base64-inlined into the request
(`inline.py`), or handed to a generation model as raw pixels (`input.py`'s
`get_image_sources_with_mime` / `get_video_sources`, which the IMAGE and VIDEO routes and the
`<generate-image>` / `<generate-video>` markers read). `link_sources/threads.py` reaches the same
downscale policy for a linked post's images. One fetch-and-downscale policy serves all of them.

That policy itself is Discord-free and sits a layer below in `utils/images.py`; what this module
adds is the nextcord half — reading an `Attachment` / `StickerItem`, and deciding the MIME type
and the filename those do or do not carry.

What it deliberately does not do: no caching (every call re-fetches, which is why `streaming.py`
loads the marker source images once and shares them), no concurrency bound, no dead-source skip,
and no failure swallowing. Those belong to the callers — `base.py`'s media semaphore and
dead-source cache, `input.py`'s per-message render cache — so the loaders stay usable by a path
that has none of them.
"""

import asyncio
from mimetypes import guess_type

from nextcord import Attachment, StickerItem

from discordbot.utils.images import get_image_data, shrink_image_bytes


async def load_image_bytes(source: Attachment | StickerItem | str) -> tuple[bytes, str]:
    """Fetches one image source and downscales it to upload-ready bytes plus its MIME type.

    A str source is fetched and re-encoded to JPEG by `get_image_data`, so the `image/jpeg` label
    is asserted rather than sniffed; that holds because every caller passes an `http(s)` url. An
    attachment or sticker is read from Discord and handed to `shrink_image_bytes` under the type
    Discord declared or one guessed from its filename / url, because that type is what selects
    the GIF and in-bounds passthroughs and what labels a passthrough result. Both the url fetch
    and the PIL work block, so they run on a worker thread.

    Every fetch and decode failure propagates, and each call re-fetches, so referencing one
    source twice costs two downloads. Callers bound their own concurrency (the uploaders hold
    `base.py`'s media semaphore across the whole load plus upload) or are loading a single
    current-turn image.

    Args:
        source (Attachment | StickerItem | str): The image attachment, sticker, or image url.

    Returns:
        The image bytes and the MIME type that describes them.
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
    """Returns the upload filename for an image source (attachment, sticker, or url).

    A url or embed image carries no filename of its own, so `url_fallback` names it and must
    carry an image extension: it becomes the Files API entry's name, and a filename-only
    classifier downstream then reads the source as an image rather than a document. An
    attachment keeps its real filename; a sticker synthesizes `<name>.png`. The two nextcord
    types are told apart by attribute rather than `isinstance`, since only `Attachment` carries
    `filename` and only `StickerItem` carries `name`.

    Args:
        source (Attachment | StickerItem | str): The image attachment, sticker, or image url.
        url_fallback (str): Image-extensioned name for a url source (`image.png` on the Gemini
            path, `image.jpg` on the OpenAI / Anthropic ones).

    Returns:
        The filename to upload the source under.
    """
    if isinstance(source, str):
        return url_fallback
    return getattr(source, "filename", None) or f"{getattr(source, 'name', 'sticker')}.png"


def attachment_mime(attachment: Attachment) -> str:
    """Returns the bare MIME type of a file attachment, empty when unguessable.

    Discord's own `content_type` wins, then a guess from the filename. Any `; charset=...`
    parameter is dropped, since consumers compare the result against a bare type
    (`application/pdf` in `InlineRenderer._inline_file_part`) and hand it to a Files API as the
    entry's mime. Every renderer treats the empty string as "drop this attachment" rather than
    uploading something the answer request cannot describe.

    Args:
        attachment (Attachment): The attachment to type.

    Returns:
        The MIME type without parameters, or an empty string when neither source names one.
    """
    content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
    return content_type.split(";")[0].strip()


async def load_attachment_bytes(attachment: Attachment) -> tuple[bytes, str]:
    """Reads a file attachment's bytes alongside its resolved MIME type.

    Nothing is decoded, downscaled or size-capped here, unlike the image loader, so a large clip
    is held whole in memory and the caller is what bounds that (`get_video_sources` reads only
    the first usable video; an uploader holds a media slot across the read). The type is not
    checked either and can come back empty, so a caller that needs a usable one calls
    `attachment_mime` first and drops the attachment before paying for this read. Discord read
    failures propagate.

    Args:
        attachment (Attachment): The attachment to download.

    Returns:
        The attachment bytes and the MIME type resolved by `attachment_mime`.
    """
    return await attachment.read(), attachment_mime(attachment=attachment)
