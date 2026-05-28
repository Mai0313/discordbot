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


def _embed_has_real_image(*, embed: Embed, spacer_url: str) -> bool:
    """Returns True when an embed already shows a real image via set_image."""
    image_url = embed.image.url if embed.image else None
    return bool(image_url and image_url != spacer_url)


def _target_allows_file_uploads(*, target: object | None) -> bool:
    """Returns False only when the current channel clearly denies file uploads."""
    if target is None:
        return True
    channel = getattr(target, "channel", None)
    guild = getattr(target, "guild", None) or getattr(channel, "guild", None)
    if guild is None:
        return True
    member = getattr(target, "me", None) or getattr(guild, "me", None)
    if member is None:
        client = getattr(target, "client", None) or getattr(target, "bot", None)
        user = getattr(client, "user", None)
        user_id = getattr(user, "id", None)
        get_member = getattr(guild, "get_member", None)
        if isinstance(user_id, int) and callable(get_member):
            member = get_member(user_id)
    permissions_for = getattr(channel, "permissions_for", None)
    if member is None or not callable(permissions_for):
        return True
    permissions = permissions_for(member)
    return bool(getattr(permissions, "attach_files", True))


def apply_embed_spacer_image(
    *, embeds: list[Embed], filename: str = DEFAULT_EMBED_SPACER_FILENAME
) -> list[Embed]:
    """Sets a transparent spacer only on embeds without an image of their own."""
    spacer_url = embed_spacer_url(filename=filename)
    for embed in embeds:
        if not _embed_has_real_image(embed=embed, spacer_url=spacer_url):
            embed.set_image(url=spacer_url)
    return embeds


def embed_spacer_payload(
    *,
    embeds: list[Embed],
    is_edit: bool,
    target: object | None = None,
    extra_files: list[File] | None = None,
    filename: str = DEFAULT_EMBED_SPACER_FILENAME,
) -> dict[str, Any]:
    """Returns the spacer files/attachments increment to merge into a send or edit."""
    spacer_url = embed_spacer_url(filename=filename)
    needs_spacer = any(
        not _embed_has_real_image(embed=embed, spacer_url=spacer_url) for embed in embeds
    )
    files: list[File] = list(extra_files or [])
    can_upload_spacer = _target_allows_file_uploads(target=target)
    if needs_spacer and can_upload_spacer and len(files) < DISCORD_MAX_FILES_PER_MESSAGE:
        apply_embed_spacer_image(embeds=embeds, filename=filename)
        files.append(build_embed_spacer_file(filename=filename))
    elif needs_spacer:
        for embed in embeds:
            if embed.image and embed.image.url == spacer_url:
                embed.set_image(url=None)
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
