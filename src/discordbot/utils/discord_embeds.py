"""Helpers for Discord embed rendering quirks."""

from io import BytesIO
from typing import Final
from functools import cache

from PIL import Image
from nextcord import File, Embed

DEFAULT_EMBED_SPACER_FILENAME: Final[str] = "embed_spacer.png"
DEFAULT_EMBED_SPACER_WIDTH: Final[int] = 640
DEFAULT_EMBED_SPACER_HEIGHT: Final[int] = 1
_TRANSPARENT_RGBA: Final[tuple[int, int, int, int]] = (0, 0, 0, 0)


def embed_spacer_url(*, filename: str = DEFAULT_EMBED_SPACER_FILENAME) -> str:
    """Returns the attachment URL for a transparent embed spacer image."""
    return f"attachment://{filename}"


def build_embed_spacer_file(
    *,
    filename: str = DEFAULT_EMBED_SPACER_FILENAME,
    width: int = DEFAULT_EMBED_SPACER_WIDTH,
    height: int = DEFAULT_EMBED_SPACER_HEIGHT,
) -> File:
    """Builds a fresh transparent PNG upload for one Discord send or edit."""
    return File(
        fp=BytesIO(initial_bytes=_transparent_png_bytes(width=width, height=height)),
        filename=filename,
    )


def apply_embed_spacer_image(
    *, embeds: list[Embed], filename: str = DEFAULT_EMBED_SPACER_FILENAME
) -> list[Embed]:
    """Sets a transparent spacer image on embeds so Discord renders aligned widths."""
    spacer_url = embed_spacer_url(filename=filename)
    for embed in embeds:
        embed.set_image(url=spacer_url)
    return embeds


@cache
def _transparent_png_bytes(*, width: int, height: int) -> bytes:
    image = Image.new(mode="RGBA", size=(width, height), color=_TRANSPARENT_RGBA)
    buffer = BytesIO()
    image.save(fp=buffer, format="PNG", optimize=True)
    return buffer.getvalue()
