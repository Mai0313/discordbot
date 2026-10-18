"""Discord's per-message ceilings, and helpers for its embed rendering quirks."""

from io import BytesIO
from typing import TYPE_CHECKING, Any, Final
from functools import cache

from PIL import Image
from nextcord import File, Embed, Attachment

if TYPE_CHECKING:
    from nextcord.types.embed import Embed as EmbedData

# Discord's own ceilings on what one message may carry: its `content`, one `embed.description`,
# every text field of every embed in it summed, and how many embeds and attachments it holds.
# Overshooting any of them makes Discord reject the whole send, not trim it.
DISCORD_MESSAGE_LIMIT: Final[int] = 2000
DISCORD_EMBED_DESCRIPTION_LIMIT: Final[int] = 4096
DISCORD_EMBED_TOTAL_LIMIT: Final[int] = 6000
DISCORD_EMBED_COUNT_LIMIT: Final[int] = 10
DISCORD_ATTACHMENT_LIMIT: Final[int] = 10

DEFAULT_EMBED_SPACER_FILENAME: Final[str] = "embed_spacer.png"


def utf16_length(value: str) -> int:
    """Returns the UTF-16 code-unit length of `value`.

    Discord's docs never define which unit its embed limits count, so an emoji is priced at the
    two units it costs on the wire rather than the one `len` sees.

    Args:
        value (str): Text to measure.
    """
    return sum(2 if ord(character) > 0xFFFF else 1 for character in value)


def clip_to_utf16_limit(text: str, limit: int, notice: str) -> str:
    """Returns `text` within `limit` UTF-16 units, ending in `notice` when it had to cut.

    The cut is marked so a truncated post never reads as a whole one. A limit with no room for
    the notice yields the notice alone: a plain `text[: limit - len(notice)]` takes a negative
    slice and silently returns more than was asked for.

    Args:
        text (str): Text to clip.
        limit (int): Maximum UTF-16 code-unit length.
        notice (str): Suffix appended when `text` exceeds `limit`.
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


def embed_text_length(embed: Embed) -> int:
    """Returns the total UTF-16 code-unit length of all text-bearing fields in `embed`.

    Args:
        embed (Embed): Embed to measure against Discord's message-wide limit.
    """
    payload: EmbedData = embed.to_dict()
    text_parts = [
        value for value in (payload.get("title"), payload.get("description")) if value is not None
    ]
    if footer := payload.get("footer"):
        text_parts.append(footer["text"])
    if (author := payload.get("author")) and (name := author.get("name")):
        text_parts.append(name)
    for field in payload.get("fields", []):
        text_parts.extend((field["name"], field["value"]))

    return sum(utf16_length(value=value) for value in text_parts)


def embed_spacer_url() -> str:
    """Returns the attachment URL for a transparent embed spacer image."""
    return f"attachment://{DEFAULT_EMBED_SPACER_FILENAME}"


def build_embed_spacer_file() -> File:
    """Returns a fresh transparent PNG upload for one Discord send or edit."""
    return File(
        fp=BytesIO(initial_bytes=_transparent_png_bytes()), filename=DEFAULT_EMBED_SPACER_FILENAME
    )


def _embed_has_real_image(embed: Embed, spacer_url: str) -> bool:
    """Returns True when an embed already shows a real image via set_image."""
    image_url = embed.image.url if embed.image else None
    return bool(image_url and image_url != spacer_url)


def _target_allows_file_uploads(target: object | None) -> bool:
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


def apply_embed_spacer_image(embeds: list[Embed]) -> list[Embed]:
    """Returns `embeds` after setting a transparent spacer on each embed without an image of its own.

    Args:
        embeds (list[Embed]): Embeds to update in place.
    """
    spacer_url = embed_spacer_url()
    for embed in embeds:
        if not _embed_has_real_image(embed=embed, spacer_url=spacer_url):
            embed.set_image(url=spacer_url)
    return embeds


def _existing_spacer_attachment(target: object | None) -> Attachment | None:
    """Returns an already-uploaded spacer attachment on the edit target, if present."""
    message = target if hasattr(target, "attachments") else getattr(target, "message", None)
    attachments = getattr(message, "attachments", None) or ()
    for attachment in attachments:
        if getattr(attachment, "filename", None) == DEFAULT_EMBED_SPACER_FILENAME:
            return attachment
    return None


def embed_spacer_payload(
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

    Args:
        embeds (list[Embed]): Embeds being sent or edited.
        is_edit (bool): Whether the payload is for an edit rather than an initial send.
        target (object | None): Message or interaction used to inspect existing attachments and
            upload permissions.
        extra_files (list[File] | None): Additional files to send alongside the spacer.
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
    """Returns a 640x1 transparent PNG so spaced embeds render at a uniform width."""
    image = Image.new(mode="RGBA", size=(640, 1), color=(0, 0, 0, 0))
    buffer = BytesIO()
    image.save(fp=buffer, format="PNG", optimize=True)
    return buffer.getvalue()
