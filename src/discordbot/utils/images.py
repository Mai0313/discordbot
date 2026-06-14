"""Image helpers previously sourced from autogen.agentchat.contrib.img_utils.

Inlined to drop the autogen/ag2 runtime dependency. Trimmed to the input forms
the bot actually uses: `http(s)://` URLs and `data:image/...;base64,...` URIs.
"""

from io import BytesIO
import re
import base64

from PIL import Image
import requests

_DATA_URI_RE = re.compile(pattern=r"^data:image/(?:jpg|jpeg|png|gif|bmp|webp);base64,")


def get_pil_image(image_file: str) -> Image.Image:
    """Loads an image from an `http(s)://` URL or a base64 data URI.

    Args:
        image_file: `http(s)://...` URL or `data:image/<mime>;base64,...` URI.

    Returns:
        The decoded image, converted to RGB.

    Raises:
        ValueError: `image_file` is neither an `http(s)://` URL nor a
            recognised image data URI.
    """
    if image_file.startswith(("http://", "https://")):
        # 10s caps the history-render I/O tail: a URL taking longer is almost always a
        # dead/slow CDN that would fail anyway, and a 30s wait let one such source dominate
        # the whole render. Healthy media.discordapp.net images return well under 1s.
        response = requests.get(url=image_file, timeout=10)
        image = Image.open(fp=BytesIO(initial_bytes=response.content))
    elif match := _DATA_URI_RE.match(string=image_file):
        payload = base64.b64decode(s=image_file[match.end() :])
        image = Image.open(fp=BytesIO(initial_bytes=payload))
    else:
        raise ValueError(f"Unsupported image source: {image_file[:64]!r}")
    return image.convert("RGB")


# Gemini scales anything past 3072x3072 down server-side before the model sees it, so
# capping the longest edge locally never changes what the model consumes; it only stops
# us uploading bytes the provider would discard anyway.
_MAX_IMAGE_DIMENSION = 3072


def shrink_image_bytes(payload: bytes, content_type: str) -> tuple[bytes, str]:
    """Downscales an image to the provider's effective resolution and re-encodes it.

    Photos re-encode as JPEG quality 95 (near-lossless, a fraction of PNG photo
    bytes); images with transparency or an indexed palette stay PNG (alpha must
    survive, and JPEG artifacts are visible on flat-color palette graphics);
    GIFs and other animated images pass through untouched so motion context
    survives. Anything PIL cannot decode passes through unchanged.

    Args:
        payload: The original encoded image bytes.
        content_type: The image's MIME type, used to pick passthrough cases.

    Returns:
        The (possibly re-encoded) image bytes and their MIME type.
    """
    if content_type == "image/gif":
        return payload, content_type
    try:
        image = Image.open(fp=BytesIO(initial_bytes=payload))
        if getattr(image, "is_animated", False):
            return payload, content_type
        keep_png = image.mode in {"RGBA", "LA", "PA", "P"}
        within_bounds = max(image.size) <= _MAX_IMAGE_DIMENSION
        if within_bounds and (content_type == "image/jpeg" or keep_png):
            return payload, content_type
        image.thumbnail(
            size=(_MAX_IMAGE_DIMENSION, _MAX_IMAGE_DIMENSION), resample=Image.Resampling.LANCZOS
        )
        buffered = BytesIO()
        if keep_png:
            image.save(fp=buffered, format="PNG")
            return buffered.getvalue(), "image/png"
        image.convert("RGB").save(fp=buffered, format="JPEG", quality=95)
        return buffered.getvalue(), "image/jpeg"
    except Exception:
        # An undecodable or exotic payload is sent as-is; the API rejects it the
        # same way it would have before the shrink existed.
        return payload, content_type


def get_image_data(image_file: str) -> bytes:
    """Returns the underlying bytes of an image.

    Fast path: when `image_file` is already a `data:image/<mime>;base64,...`
    URI, the embedded payload is decoded and returned as-is — no PIL
    decode/encode round trip, no format change. JPEG stays JPEG, PNG stays PNG.

    Slow path: anything else is fetched / decoded via :func:`get_pil_image`,
    downscaled to the provider's effective resolution, and re-encoded as JPEG.

    Args:
        image_file: URL or data URI.

    Returns:
        Raw image bytes.

    Raises:
        ValueError: `image_file` is not a supported URL or image data URI.
    """
    if match := _DATA_URI_RE.match(string=image_file):
        payload = image_file[match.end() :]
        return base64.b64decode(s=payload)

    image = get_pil_image(image_file=image_file)
    image.thumbnail(
        size=(_MAX_IMAGE_DIMENSION, _MAX_IMAGE_DIMENSION), resample=Image.Resampling.LANCZOS
    )
    buffered = BytesIO()
    image.save(fp=buffered, format="JPEG", quality=95)
    return buffered.getvalue()


def convert_base64_to_data_uri(base64_image: str) -> str:
    """Wraps a base64 image string in a `data:image/<mime>;base64,...` URI.

    Sniffs the MIME type from the first 12 decoded bytes (enough for every
    format we recognise). Falls back to `image/jpeg` for unknown payloads.

    Args:
        base64_image: Base64-encoded image payload without a data URI prefix.

    Returns:
        A data URI containing the detected image MIME type and original payload.
    """
    header = base64.b64decode(s=base64_image[:16])
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
    return f"data:{mime_type};base64,{base64_image}"
