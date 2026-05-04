"""Image helpers previously sourced from autogen.agentchat.contrib.img_utils.

Inlined to drop the autogen/ag2 runtime dependency. Trimmed to the input forms
the bot actually uses: ``http(s)://`` URLs and ``data:image/...;base64,...`` URIs.
"""

from io import BytesIO
import re
import base64
from typing import Literal, overload

from PIL import Image
import requests

_DATA_URI_RE = re.compile(pattern=r"^data:image/(?:jpg|jpeg|png|gif|bmp|webp);base64,")


def get_pil_image(image_file: str) -> Image.Image:
    """Loads an image from an ``http(s)://`` URL or a base64 data URI.

    Args:
        image_file: ``http(s)://...`` URL or ``data:image/<mime>;base64,...`` URI.

    Returns:
        The decoded image, converted to RGB.

    Raises:
        ValueError: ``image_file`` is neither an ``http(s)://`` URL nor a
            recognised image data URI.
    """
    if image_file.startswith(("http://", "https://")):
        response = requests.get(url=image_file, timeout=30)
        image = Image.open(fp=BytesIO(initial_bytes=response.content))
    elif match := _DATA_URI_RE.match(string=image_file):
        payload = base64.b64decode(s=image_file[match.end() :])
        image = Image.open(fp=BytesIO(initial_bytes=payload))
    else:
        raise ValueError(f"Unsupported image source: {image_file[:64]!r}")
    return image.convert("RGB")


@overload
def get_image_data(image_file: str, use_b64: Literal[True] = ...) -> str: ...
@overload
def get_image_data(image_file: str, use_b64: Literal[False]) -> bytes: ...
def get_image_data(image_file: str, use_b64: bool = True) -> bytes | str:
    """Returns the underlying bytes of an image (or base64-encoded form).

    Fast path: when ``image_file`` is already a ``data:image/<mime>;base64,...``
    URI, the embedded payload is returned as-is — no PIL decode/encode round
    trip, no format change. JPEG stays JPEG, PNG stays PNG.

    Slow path: anything else is fetched / decoded via :func:`get_pil_image` and
    re-encoded as PNG.

    Args:
        image_file: URL or data URI.
        use_b64: When ``True`` (default) return a base64 ``str``; when ``False``
            return raw ``bytes``.
    """
    if match := _DATA_URI_RE.match(string=image_file):
        payload = image_file[match.end() :]
        return payload if use_b64 else base64.b64decode(s=payload)

    image = get_pil_image(image_file=image_file)
    buffered = BytesIO()
    image.save(fp=buffered, format="PNG")
    content = buffered.getvalue()
    if use_b64:
        return base64.b64encode(s=content).decode(encoding="utf-8")
    return content


def convert_base64_to_data_uri(base64_image: str) -> str:
    """Wraps a base64 image string in a ``data:image/<mime>;base64,...`` URI.

    Sniffs the MIME type from the first 12 decoded bytes (enough for every
    format we recognise). Falls back to ``image/jpeg`` for unknown payloads.
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
