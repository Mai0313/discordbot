"""Helpers for Discord embed rendering quirks."""

from io import BytesIO
from typing import Any, Final
from functools import cache

from PIL import Image
from nextcord import File, Embed

DEFAULT_EMBED_SPACER_FILENAME: Final[str] = "embed_spacer.png"
DEFAULT_EMBED_SPACER_WIDTH: Final[int] = 640
DEFAULT_EMBED_SPACER_HEIGHT: Final[int] = 1
DISCORD_MAX_FILES_PER_MESSAGE: Final[int] = 10
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


def _embed_has_image(embed: Embed) -> bool:
    """Returns True when an embed already shows a real image via set_image."""
    return bool(embed.image and embed.image.url)


def apply_embed_spacer_image(
    *, embeds: list[Embed], filename: str = DEFAULT_EMBED_SPACER_FILENAME
) -> list[Embed]:
    """Sets a transparent spacer only on embeds without an image of their own."""
    spacer_url = embed_spacer_url(filename=filename)
    for embed in embeds:
        if not _embed_has_image(embed=embed):
            embed.set_image(url=spacer_url)
    return embeds


def embed_spacer_payload(
    *,
    embeds: list[Embed],
    is_edit: bool,
    extra_files: list[File] | None = None,
    filename: str = DEFAULT_EMBED_SPACER_FILENAME,
) -> dict[str, Any]:
    """Returns the spacer files/attachments increment to merge into a send or edit."""
    spacer_url = embed_spacer_url(filename=filename)
    needs_spacer = any(not _embed_has_image(embed=embed) for embed in embeds)
    files: list[File] = list(extra_files or [])
    if needs_spacer and len(files) < DISCORD_MAX_FILES_PER_MESSAGE:
        apply_embed_spacer_image(embeds=embeds, filename=filename)
        files.append(build_embed_spacer_file(filename=filename))
    payload: dict[str, Any] = {}
    if files:
        payload["files"] = files
    if is_edit:
        payload["attachments"] = []
    return payload


@cache
def _transparent_png_bytes(*, width: int, height: int) -> bytes:
    image = Image.new(mode="RGBA", size=(width, height), color=_TRANSPARENT_RGBA)
    buffer = BytesIO()
    image.save(fp=buffer, format="PNG", optimize=True)
    return buffer.getvalue()
