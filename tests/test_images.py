"""Pins `utils.images.shrink_image_bytes`, the one downscale-and-re-encode policy for ingestion.

Every image that reaches a model crosses this single helper (a Discord attachment or sticker via
`gen_reply/attachment/loaders.py`, and the linked-post image builders that arrive through the same
`load_image_bytes`), so what it decides per format is what the model actually sees and there is no
second policy to fall back on. The cases below are its branch table, and each one is a decision
rather than an implementation detail: an alpha (`RGBA`) or palette (`P`) image downscales but stays
PNG, since JPEG would drop transparency and put visible artifacts on flat-color graphics; anything
else opaque becomes JPEG even when it is already in bounds, because quality-95 JPEG is
near-lossless at a fraction of PNG's photo bytes; a GIF is handed back untouched so animation
reaches the model at all; and bytes PIL cannot decode come back unchanged instead of raising, so a
corrupt attachment degrades to whatever the provider makes of it rather than failing the reply
around it.

The two byte-identical passthroughs are the assertions with teeth. An in-bounds JPEG and an
in-bounds transparent PNG are the commonest attachment shapes, so a future re-encode that
"normalises everything" would quietly rewrite most uploads; comparing against the exact input bytes
is what stops that landing unnoticed. The `3072` here is `_MAX_IMAGE_DIMENSION` spelled out, and it
is a transfer-cost bound rather than a quality one: Gemini scales anything larger down server-side
anyway.

The oversized fixtures are 4000x20 because only the longest edge has to cross the cap, which keeps
every encode and decode in this file cheap. The rest of `utils/images.py` is deliberately not
pinned here: `get_pil_image` and `get_image_data` reach the network, and
`convert_base64_to_data_uri` serves the generation paths rather than this ingestion cap.
"""

from io import BytesIO

from PIL import Image

from discordbot.utils.images import shrink_image_bytes


def _encoded_bytes(size: tuple[int, int], mode: str, image_format: str) -> bytes:
    """Encodes a solid-color test image of the given size, mode, and format.

    `mode` is what selects the branch under test: `RGBA` carries alpha and `P` is a palette image,
    both of which `shrink_image_bytes` refuses to turn into JPEG.

    Returns:
        The encoded image bytes.
    """
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

    _shrunk, mime_type = shrink_image_bytes(payload=payload, content_type="image/png")

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


def test_shrink_keeps_palette_as_png() -> None:
    """An oversized palette image downscales but stays PNG to avoid JPEG artifacts."""
    payload = _encoded_bytes(size=(4000, 20), mode="P", image_format="PNG")

    shrunk, mime_type = shrink_image_bytes(payload=payload, content_type="image/png")

    assert mime_type == "image/png"
    image = Image.open(fp=BytesIO(initial_bytes=shrunk))
    assert max(image.size) <= 3072


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
