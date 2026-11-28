"""Image loading, downscaling, and data-URI helpers."""

from io import BytesIO
import base64

from PIL import Image
import logfire
import requests

from discordbot.typings.media import LoadedMedia
from discordbot.typings.timeouts import IMAGE_FETCH_TIMEOUT_SECONDS

# Gemini downscales inputs past 3072x3072 server-side; cap locally to avoid wasted upload bytes.
_MAX_IMAGE_DIMENSION = 3072


def shrink_image_bytes(payload: bytes, content_type: str, filename: str) -> LoadedMedia:
    """Returns `payload` downscaled to the provider's effective resolution and re-encoded.

    Photos re-encode as JPEG quality 95; images with transparency or an indexed palette stay PNG;
    GIFs and other animated images pass through untouched. Undecodable payloads and images already
    within bounds pass through unchanged.

    Args:
        payload (bytes): The original encoded image bytes.
        content_type (str): The image's MIME type, used to pick passthrough cases.
        filename (str): The source's upload filename, logged when the payload cannot be decoded.
    """
    unchanged = LoadedMedia(data=payload, mime_type=content_type)
    if content_type == "image/gif":
        return unchanged
    try:
        image = Image.open(fp=BytesIO(initial_bytes=payload))
        if getattr(image, "is_animated", False):
            return unchanged
        keep_png = image.has_transparency_data or image.mode == "P"
        within_bounds = max(image.size) <= _MAX_IMAGE_DIMENSION
        if within_bounds and (
            (content_type == "image/jpeg" and image.format == "JPEG")
            or (keep_png and content_type == "image/png" and image.format == "PNG")
        ):
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
    """Returns the image at `image_file` downscaled to one still frame and its MIME type.

    Animated images keep only their first frame. Stills with transparency stay PNG; all others
    re-encode as JPEG.

    Args:
        image_file (str): `http(s)://...` URL.

    Raises:
        requests.RequestException: URL could not be fetched or is not `http(s)://`.
        PIL.UnidentifiedImageError: Response body is not a decodable image.
    """
    response = requests.get(url=image_file, timeout=IMAGE_FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    # An image opens on its first frame, and saving without `save_all` writes only that frame.
    with Image.open(fp=BytesIO(initial_bytes=response.content)) as image:
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
    """Returns `data` encoded as a `data:<mime>;base64,...` URI.

    Args:
        data (bytes): Payload to inline.
        mime_type (str | None): Payload MIME type, or None to sniff from magic bytes (defaulting
            to `image/jpeg`).
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
