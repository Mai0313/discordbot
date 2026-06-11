from io import BytesIO

from PIL import Image

from discordbot.utils.images import shrink_image_bytes


def _encoded_bytes(size: tuple[int, int], mode: str, image_format: str) -> bytes:
    """Encodes a solid-color test image of the given size, mode, and format."""
    buffer = BytesIO()
    Image.new(mode=mode, size=size, color=0).save(fp=buffer, format=image_format)
    return buffer.getvalue()


def test_shrink_reencodes_oversized_png_as_jpeg() -> None:
    """An oversized opaque PNG is downscaled to the provider cap and becomes JPEG."""
    payload = _encoded_bytes(size=(4000, 20), mode="RGB", image_format="PNG")

    shrunk, mime_type = shrink_image_bytes(payload=payload, content_type="image/png")

    assert mime_type == "image/jpeg"
    image = Image.open(fp=BytesIO(initial_bytes=shrunk))
    assert max(image.size) <= 3072
    assert image.format == "JPEG"


def test_shrink_reencodes_small_png_photo_as_jpeg() -> None:
    """An in-bounds opaque PNG still re-encodes as the cheaper JPEG."""
    payload = _encoded_bytes(size=(64, 64), mode="RGB", image_format="PNG")

    shrunk, mime_type = shrink_image_bytes(payload=payload, content_type="image/png")

    assert mime_type == "image/jpeg"


def test_shrink_passes_small_jpeg_through() -> None:
    """An in-bounds JPEG passes through byte-identical."""
    payload = _encoded_bytes(size=(64, 64), mode="RGB", image_format="JPEG")

    shrunk, mime_type = shrink_image_bytes(payload=payload, content_type="image/jpeg")

    assert shrunk == payload
    assert mime_type == "image/jpeg"


def test_shrink_keeps_alpha_as_png() -> None:
    """An oversized transparent image downscales but stays PNG so alpha survives."""
    payload = _encoded_bytes(size=(4000, 20), mode="RGBA", image_format="PNG")

    shrunk, mime_type = shrink_image_bytes(payload=payload, content_type="image/png")

    assert mime_type == "image/png"
    image = Image.open(fp=BytesIO(initial_bytes=shrunk))
    assert image.mode == "RGBA"
    assert max(image.size) <= 3072


def test_shrink_passes_small_alpha_png_through() -> None:
    """An in-bounds transparent PNG passes through byte-identical."""
    payload = _encoded_bytes(size=(64, 64), mode="RGBA", image_format="PNG")

    shrunk, mime_type = shrink_image_bytes(payload=payload, content_type="image/png")

    assert shrunk == payload
    assert mime_type == "image/png"


def test_shrink_passes_gif_through() -> None:
    """GIFs pass through untouched so animation survives."""
    payload = _encoded_bytes(size=(4000, 20), mode="RGB", image_format="GIF")

    shrunk, mime_type = shrink_image_bytes(payload=payload, content_type="image/gif")

    assert shrunk == payload
    assert mime_type == "image/gif"


def test_shrink_passes_undecodable_payload_through() -> None:
    """Bytes PIL cannot decode pass through unchanged."""
    payload = b"definitely not an image"

    shrunk, mime_type = shrink_image_bytes(payload=payload, content_type="image/png")

    assert shrunk == payload
    assert mime_type == "image/png"
