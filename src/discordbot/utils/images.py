"""Image loading, downscaling, and data-URI helpers."""

from io import BytesIO
import base64

from PIL import Image
import logfire
import requests

from discordbot.typings.media import LoadedMedia
from discordbot.typings.timeouts import IMAGE_FETCH_TIMEOUT_SECONDS

# Gemini scales anything past 3072x3072 down server-side before the model sees it, so
# capping the longest edge locally never changes what the model consumes; it only stops
# us uploading bytes the provider would discard anyway.
_MAX_IMAGE_DIMENSION = 3072


def shrink_image_bytes(payload: bytes, content_type: str, filename: str) -> LoadedMedia:
    """Downscales an image to the provider's effective resolution and re-encodes it.

    Photos re-encode as JPEG quality 95 (near-lossless, a fraction of PNG photo
    bytes); images with transparency or an indexed palette stay PNG (alpha must
    survive, and JPEG artifacts are visible on flat-color palette graphics);
    GIFs and other animated images pass through untouched so motion context
    survives. Anything PIL cannot decode passes through unchanged, as does an
    image already within the cap: those come back byte-identical, never re-encoded.

    Args:
        payload: The original encoded image bytes.
        content_type: The image's MIME type, used to pick passthrough cases.
        filename: The source's upload filename, logged when the payload cannot be decoded.

    Returns:
        The (possibly re-encoded) image bytes and their MIME type.
    """
    unchanged = LoadedMedia(data=payload, mime_type=content_type)
    if content_type == "image/gif":
        return unchanged
    try:
        image = Image.open(fp=BytesIO(initial_bytes=payload))
        if getattr(image, "is_animated", False):
            return unchanged
        keep_png = image.mode in {"RGBA", "LA", "PA", "P"}
        within_bounds = max(image.size) <= _MAX_IMAGE_DIMENSION
        if within_bounds and (content_type == "image/jpeg" or keep_png):
            return unchanged
        image.thumbnail(
            size=(_MAX_IMAGE_DIMENSION, _MAX_IMAGE_DIMENSION), resample=Image.Resampling.LANCZOS
        )
        buffered = BytesIO()
        if keep_png:
            image.save(fp=buffered, format="PNG")
            return LoadedMedia(data=buffered.getvalue(), mime_type="image/png")
        image.convert("RGB").save(fp=buffered, format="JPEG", quality=95)
        return LoadedMedia(data=buffered.getvalue(), mime_type="image/jpeg")
    except Exception as exc:
        # Broad: an undecodable or exotic payload is sent as-is, for the API to reject.
        logfire.warn(
            "image could not be downscaled; sending it unchanged",
            filename=filename,
            content_type=content_type,
            size_bytes=len(payload),
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
        return unchanged


def get_image_data(image_file: str) -> LoadedMedia:
    """Fetches an image URL as one still, downscaled to the provider's effective resolution.

    Unlike an attachment, an animated image keeps only its first frame: linked GIFs are mostly
    GIF-picker clips of several megabytes, and a reply's history can carry several of them.
    A still with transparency stays PNG so its alpha survives; anything else becomes JPEG.

    Args:
        image_file: `http(s)://...` URL.

    Returns:
        The re-encoded still and its MIME type.

    Raises:
        requests.RequestException: The URL could not be fetched, or is not `http(s)://`.
        PIL.UnidentifiedImageError: What came back is not a decodable image, which is
            what a 404 HTML body from a dead CDN arrives as.
    """
    response = requests.get(url=image_file, timeout=IMAGE_FETCH_TIMEOUT_SECONDS)
    # An image opens on its first frame, and saving without `save_all` writes only that frame.
    image = Image.open(fp=BytesIO(initial_bytes=response.content))
    image.thumbnail(
        size=(_MAX_IMAGE_DIMENSION, _MAX_IMAGE_DIMENSION), resample=Image.Resampling.LANCZOS
    )
    buffered = BytesIO()
    if image.has_transparency_data:
        image.save(fp=buffered, format="PNG")
        return LoadedMedia(data=buffered.getvalue(), mime_type="image/png")
    image.convert("RGB").save(fp=buffered, format="JPEG", quality=95)
    return LoadedMedia(data=buffered.getvalue(), mime_type="image/jpeg")


def to_data_uri(data: bytes, mime_type: str | None = None) -> str:
    """Encodes bytes already in hand as a `data:<mime>;base64,...` URI.

    Args:
        data: The payload to inline.
        mime_type: The payload's MIME type, or None to sniff an image type from its first 12
            bytes (enough for every format recognised here), `image/jpeg` when unrecognised.

    Returns:
        A data URI carrying the MIME type and the base64-encoded payload.
    """
    if mime_type is None:
        header = data[:12]
        if header.startswith(b"\xff\xd8\xff"):
            mime_type = "image/jpeg"
        elif header.startswith(b"\x89PNG\r\n\x1a\n"):
            mime_type = "image/png"
        elif header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
            mime_type = "image/gif"
        elif header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"
    return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"
