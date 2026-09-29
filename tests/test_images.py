from io import BytesIO
import base64

from PIL import Image
import pytest

from discordbot.utils.images import to_data_uri, shrink_image_bytes


def _encoded_bytes(size: tuple[int, int], mode: str, image_format: str) -> bytes:
    """Encodes a solid-color test image of the given size, mode, and format."""
    buffer = BytesIO()
    Image.new(mode=mode, size=size, color=0).save(fp=buffer, format=image_format)
    return buffer.getvalue()


def test_shrink_reencodes_oversized_png_as_jpeg() -> None:
    """An oversized opaque PNG is downscaled to the provider cap and becomes JPEG."""
    payload = _encoded_bytes(size=(4000, 20), mode="RGB", image_format="PNG")

    shrunk = shrink_image_bytes(payload=payload, content_type="image/png")

    assert shrunk.mime_type == "image/jpeg"
    image = Image.open(fp=BytesIO(initial_bytes=shrunk.data))
    assert max(image.size) <= 3072
    assert image.format == "JPEG"


def test_shrink_reencodes_small_png_photo_as_jpeg() -> None:
    """An in-bounds opaque PNG still re-encodes as the cheaper JPEG."""
    payload = _encoded_bytes(size=(64, 64), mode="RGB", image_format="PNG")

    shrunk = shrink_image_bytes(payload=payload, content_type="image/png")

    assert shrunk.mime_type == "image/jpeg"


def test_shrink_passes_small_jpeg_through() -> None:
    """An in-bounds JPEG passes through byte-identical."""
    payload = _encoded_bytes(size=(64, 64), mode="RGB", image_format="JPEG")

    shrunk = shrink_image_bytes(payload=payload, content_type="image/jpeg")

    assert shrunk.data == payload
    assert shrunk.mime_type == "image/jpeg"


def test_shrink_keeps_alpha_as_png() -> None:
    """An oversized transparent image downscales but stays PNG so alpha survives."""
    payload = _encoded_bytes(size=(4000, 20), mode="RGBA", image_format="PNG")

    shrunk = shrink_image_bytes(payload=payload, content_type="image/png")

    assert shrunk.mime_type == "image/png"
    image = Image.open(fp=BytesIO(initial_bytes=shrunk.data))
    assert image.mode == "RGBA"
    assert max(image.size) <= 3072


def test_shrink_passes_small_alpha_png_through() -> None:
    """An in-bounds transparent PNG passes through byte-identical."""
    payload = _encoded_bytes(size=(64, 64), mode="RGBA", image_format="PNG")

    shrunk = shrink_image_bytes(payload=payload, content_type="image/png")

    assert shrunk.data == payload
    assert shrunk.mime_type == "image/png"


def test_shrink_keeps_palette_as_png() -> None:
    """An oversized palette image downscales but stays PNG to avoid JPEG artifacts."""
    payload = _encoded_bytes(size=(4000, 20), mode="P", image_format="PNG")

    shrunk = shrink_image_bytes(payload=payload, content_type="image/png")

    assert shrunk.mime_type == "image/png"
    image = Image.open(fp=BytesIO(initial_bytes=shrunk.data))
    assert max(image.size) <= 3072


def test_shrink_passes_gif_through() -> None:
    """GIFs pass through untouched so animation survives."""
    payload = _encoded_bytes(size=(4000, 20), mode="RGB", image_format="GIF")

    shrunk = shrink_image_bytes(payload=payload, content_type="image/gif")

    assert shrunk.data == payload
    assert shrunk.mime_type == "image/gif"


def test_shrink_passes_undecodable_payload_through() -> None:
    """Bytes PIL cannot decode pass through unchanged."""
    payload = b"definitely not an image"

    shrunk = shrink_image_bytes(payload=payload, content_type="image/png")

    assert shrunk.data == payload
    assert shrunk.mime_type == "image/png"


@pytest.mark.parametrize(
    ("image_format", "mode", "mime_type"),
    [
        ("JPEG", "RGB", "image/jpeg"),
        ("PNG", "RGB", "image/png"),
        ("GIF", "P", "image/gif"),
        ("WEBP", "RGB", "image/webp"),
        ("BMP", "RGB", "image/jpeg"),
    ],
)
def test_data_uri_sniffs_an_image_type_when_none_is_given(
    image_format: str, mode: str, mime_type: str
) -> None:
    """Without a MIME type the payload's own header names it, JPEG when unrecognised."""
    payload = _encoded_bytes(size=(7, 5), mode=mode, image_format=image_format)

    uri = to_data_uri(data=payload)

    assert uri == f"data:{mime_type};base64,{base64.b64encode(payload).decode()}"


def test_data_uri_keeps_a_given_type_over_the_header() -> None:
    """A MIME type the caller knows is used as-is, even when the bytes look like an image."""
    payload = _encoded_bytes(size=(7, 5), mode="RGB", image_format="PNG")

    uri = to_data_uri(data=payload, mime_type="application/pdf")

    assert uri == f"data:application/pdf;base64,{base64.b64encode(payload).decode()}"
