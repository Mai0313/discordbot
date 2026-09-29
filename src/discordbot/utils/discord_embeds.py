"""Helpers for Discord embed rendering quirks."""

from io import BytesIO
from typing import Any, Final
from functools import cache

from PIL import Image
from nextcord import File, Embed, Attachment

from discordbot.utils.media_delivery import DISCORD_ATTACHMENT_LIMIT

# Discord's own ceilings on what one message may carry: its `content`, one `embed.description`,
# and every text field of every embed in it summed. Overshooting any of them makes Discord reject
# the whole send, not trim it.
DISCORD_MESSAGE_LIMIT: Final[int] = 2000
DISCORD_EMBED_DESCRIPTION_LIMIT: Final[int] = 4096
DISCORD_EMBED_TOTAL_LIMIT: Final[int] = 6000

DEFAULT_EMBED_SPACER_FILENAME: Final[str] = "embed_spacer.png"
DEFAULT_EMBED_SPACER_WIDTH: Final[int] = 640
DEFAULT_EMBED_SPACER_HEIGHT: Final[int] = 1
_TRANSPARENT_RGBA: Final[tuple[int, int, int, int]] = (0, 0, 0, 0)


def utf16_length(*, value: str) -> int:
    """Counts UTF-16 code units, the conservative reading of Discord's "characters".

    Discord's docs never define which unit its embed limits count, so an emoji is priced at the
    two units it costs on the wire rather than the one `len` sees.
    """
    return sum(2 if ord(character) > 0xFFFF else 1 for character in value)


def clip_to_utf16_limit(*, text: str, limit: int, notice: str) -> str:
    """Returns `text` within `limit` UTF-16 units, ending in `notice` when it had to cut.

    The cut is marked so a truncated post never reads as a whole one. A limit with no room for
    the notice yields the notice alone: the alternative is the negative slice a plain
    `text[: limit - len(notice)]` takes, which silently returns MORE than was asked for.
    """
    if utf16_length(value=text) <= limit:
        return text
    room = max(limit - utf16_length(value=notice), 0)
    kept: list[str] = []
    spent = 0
    for character in text:
        cost = 2 if ord(character) > 0xFFFF else 1
        if spent + cost > room:
            break
        kept.append(character)
        spent += cost
    return f"{''.join(kept)}{notice}"


def embed_spacer_url() -> str:
    """Returns the attachment URL for a transparent embed spacer image."""
    return f"attachment://{DEFAULT_EMBED_SPACER_FILENAME}"


def build_embed_spacer_file() -> File:
    """Builds a fresh transparent PNG upload for one Discord send or edit."""
    return File(
        fp=BytesIO(initial_bytes=_transparent_png_bytes()), filename=DEFAULT_EMBED_SPACER_FILENAME
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
    permissions_for = getattr(channel, "permissions_for", None)
    if member is None or not callable(permissions_for):
        return True
    permissions = permissions_for(member)
    return bool(getattr(permissions, "attach_files", True))


def apply_embed_spacer_image(*, embeds: list[Embed]) -> list[Embed]:
    """Sets a transparent spacer only on embeds without an image of their own."""
    spacer_url = embed_spacer_url()
    for embed in embeds:
        if not _embed_has_real_image(embed=embed, spacer_url=spacer_url):
            embed.set_image(url=spacer_url)
    return embeds


def _existing_spacer_attachment(*, target: object | None) -> Attachment | None:
    """Returns an already-uploaded spacer attachment on the edit target, if present."""
    message = target if hasattr(target, "attachments") else getattr(target, "message", None)
    attachments = getattr(message, "attachments", None) or ()
    for attachment in attachments:
        if getattr(attachment, "filename", None) == DEFAULT_EMBED_SPACER_FILENAME:
            return attachment
    return None


def embed_spacer_payload(
    *,
    embeds: list[Embed],
    is_edit: bool,
    target: object | None = None,
    extra_files: list[File] | None = None,
) -> dict[str, Any]:
    """Returns the spacer files/attachments increment to merge into a send or edit.

    The spacer never changes, so an edit retains an already-uploaded spacer by id
    instead of re-uploading it. Re-uploading the same spacer on every edit trips
    Discord's per-message edit attachment upload limit (error code 400009) for
    rapidly edited messages.
    """
    spacer_url = embed_spacer_url()
    files: list[File] = list(extra_files or [])
    retained: list[Attachment] = []
    if any(not _embed_has_real_image(embed=embed, spacer_url=spacer_url) for embed in embeds):
        existing_spacer = _existing_spacer_attachment(target=target) if is_edit else None
        can_upload_spacer = _target_allows_file_uploads(target=target)
        if existing_spacer is not None:
            apply_embed_spacer_image(embeds=embeds)
            retained.append(existing_spacer)
        elif can_upload_spacer and len(files) < DISCORD_ATTACHMENT_LIMIT:
            apply_embed_spacer_image(embeds=embeds)
            files.append(build_embed_spacer_file())
        else:
            for embed in embeds:
                if embed.image and embed.image.url == spacer_url:
                    embed.set_image(url=None)
    payload: dict[str, Any] = {}
    if files:
        payload["files"] = files
    if is_edit:
        payload["attachments"] = retained
    return payload


@cache
def _transparent_png_bytes() -> bytes:
    image = Image.new(
        mode="RGBA",
        size=(DEFAULT_EMBED_SPACER_WIDTH, DEFAULT_EMBED_SPACER_HEIGHT),
        color=_TRANSPARENT_RGBA,
    )
    buffer = BytesIO()
    image.save(fp=buffer, format="PNG", optimize=True)
    return buffer.getvalue()
