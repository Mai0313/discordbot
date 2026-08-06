"""Image normalisation on the way to an LLM provider: fetch, downscale, re-encode, wrap.

Inlined from `autogen.agentchat.contrib.img_utils` to drop the autogen/ag2 runtime dependency,
then trimmed to the two input forms the bot actually passes around: an `http(s)://` URL and a
`data:image/...;base64,...` URI. Anything else is refused rather than guessed at.

`get_pil_image` resolves one of those two forms into a PIL image; `get_image_data` returns the
bytes behind either form, handing a data URI's payload straight back and flattening to JPEG only
on the URL branch; `shrink_image_bytes` is the attachment-side one that downscales bytes already
in hand while keeping the format where the format carries something; `convert_base64_to_data_uri`
goes the other way, wrapping raw base64 into the URI a Responses `input_image` part expects.

It sits below the cogs because it is deliberately Discord-free: a caller reads an `Attachment`
and hands over plain bytes, so the attachment ingestion path (`gen_reply/attachment/loaders.py`)
and the linked-post image builders, which reach it through that same `load_image_bytes`, share
one downscale policy instead of growing two. That policy is the single `_MAX_IMAGE_DIMENSION`
cap below. The remaining importers (the generated-image persona replies, the prompt director's
grounding images, `scripts/image_dev.py`) take `convert_base64_to_data_uri` alone and never reach
the cap.

What it deliberately does not do: no caching, no download size limit, no retry, and nothing
async — `get_pil_image`, `get_image_data` and `shrink_image_bytes` block on network I/O or a PIL
decode, so async callers run those through `asyncio.to_thread`. `convert_base64_to_data_uri` does
neither and is called straight from the event loop.
"""

from io import BytesIO
import re
import base64

from PIL import Image
import requests

_DATA_URI_RE = re.compile(pattern=r"^data:image/(?:jpg|jpeg|png|gif|bmp|webp);base64,")


def get_pil_image(image_file: str) -> Image.Image:
    """Loads an image from an `http(s)://` URL or a base64 data URI.

    The result is flattened to RGB, so alpha does not survive this path; a caller that needs
    transparency intact works from the encoded bytes through `shrink_image_bytes` instead. The
    response status is not checked, so an error page surfaces as a PIL decode failure rather than
    an HTTP one; that and any `requests` transport failure propagate untouched.

    Args:
        image_file (str): An `http://` or `https://` URL, or a `data:image/<mime>;base64,...`
            URI.

    Returns:
        The decoded image, converted to RGB.

    Raises:
        ValueError: `image_file` is neither an `http(s)://` URL nor a recognised image data URI.
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

    Transparent and palette images are never converted to JPEG (alpha has to survive, and JPEG
    artifacts are visible on flat-color palette graphics); only the ones that actually downscale
    are rewritten, as PNG, while an in-bounds one is handed back under the caller's own
    `content_type`, so an in-bounds transparent WebP stays labelled `image/webp` and is never PNG
    at any point. Everything else becomes JPEG at quality 95, near-lossless and a fraction of
    PNG's photo bytes — so an already in-bounds opaque PNG is still re-encoded, which is the point
    rather than an oversight. Five shapes skip the re-encode and come back byte-identical under
    the MIME type they arrived with: a GIF and any other animated image (so motion context reaches
    the model at all), an in-bounds JPEG, an in-bounds transparent or palette image, and a payload
    PIL cannot round-trip. The two in-bounds ones are the commonest attachment shapes, so most
    images pass straight through despite the re-encode rule above. The last one is wider than a
    corrupt file: an exotic mode PIL reads but cannot write back (`PA`) also lands there, and
    since it is caught after the resize the oversized original is what gets returned.

    Args:
        payload (bytes): The original encoded image bytes.
        content_type (str): The image's MIME type, which selects the GIF and JPEG passthroughs
            and labels every passthrough result.

    Returns:
        The (possibly re-encoded) image bytes and the MIME type that now describes them.
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
    """Returns the underlying bytes of an image named by a URL or a data URI.

    A data URI short-circuits: the embedded payload is decoded and handed back as it was written,
    with no PIL round trip, so the format is preserved and the dimension cap is never applied.
    Everything else goes through `get_pil_image`, then downscales and re-encodes as JPEG, which is
    why `load_image_bytes` labels a URL source `image/jpeg` without inspecting the result.

    A source that is neither form therefore reaches `get_pil_image`'s `else` branch and leaves
    here as `ValueError`, uncaught: an unsupported source is a live caller-visible failure, not a
    silent empty result. It gets no `Raises:` section only because DOC502 reserves that for an
    exception raised in this body.

    Args:
        image_file (str): URL or data URI.

    Returns:
        The image bytes: JPEG for a fetched URL, the original encoding for a data URI.
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

    The MIME type is sniffed from the first 12 decoded bytes (16 base64 characters, which is as
    far as the magic numbers below reach) rather than taken from a caller, since the generation
    paths hand over whatever the image model returned. An unrecognised payload is labelled
    `image/jpeg` and nothing rechecks that guess.

    Args:
        base64_image (str): Base64-encoded image payload without a data URI prefix.

    Returns:
        A data URI carrying the sniffed MIME type and the payload unchanged.
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
