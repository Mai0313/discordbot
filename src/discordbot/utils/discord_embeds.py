"""One rendered width for every embed in a message, via a transparent spacer image.

Discord sizes an embed to its own content, so a message carrying several text embeds (the
Blackjack table's one embed per seat) renders ragged, each row only as wide as its own longest
line. A lone embed has the same problem across time rather than across rows: it changes width
as its content changes from one send or edit to the next, which is what most call sites here
are actually buying. An embed that carries an image is laid out around that image instead, so
a fully transparent 640x1 PNG set on every embed that has no image of its own pins them all to
one fixed width while showing nothing.

`embed_spacer_payload` is the entry point, and it returns only the `files` / `attachments`
increment to splat into a send or edit the caller already owns. Nothing here talks to Discord,
so reply vs followup vs `edit_message` stays the call site's decision. It also owns the three
cases where the spacer must not be uploaded: an edit whose message already carries one
(re-uploading it trips Discord error 400009, see that function), a channel that denies
`attach_files`, and a caller whose own files already fill the 10-attachment cap. In the last
two it strips the `attachment://` url back off the embeds as well, so an embed object reused
across sends never points at a file the payload does not carry.

It sits in `utils/` rather than inside one cog because the alignment has to be identical
everywhere embeds are posted together, and because the parts that are easy to get wrong (the
400009 rule, the attachment cap, the permission fallback) are worth discovering once instead
of per cog.
"""

from io import BytesIO
from typing import Any, Final
from functools import cache

from PIL import Image
from nextcord import File, Embed, Attachment

DEFAULT_EMBED_SPACER_FILENAME: Final[str] = "embed_spacer.png"
DEFAULT_EMBED_SPACER_WIDTH: Final[int] = 640
DEFAULT_EMBED_SPACER_HEIGHT: Final[int] = 1
DISCORD_MAX_FILES_PER_MESSAGE: Final[int] = 10
_TRANSPARENT_RGBA: Final[tuple[int, int, int, int]] = (0, 0, 0, 0)


def embed_spacer_url(*, filename: str = DEFAULT_EMBED_SPACER_FILENAME) -> str:
    """Builds the url an embed uses to point at the uploaded spacer.

    Args:
        filename (str): The name the spacer is uploaded under. The embed and the `File` must
            agree on it, or the image resolves to nothing.

    Returns:
        The `attachment://<filename>` url.
    """
    return f"attachment://{filename}"


def build_embed_spacer_file(
    *,
    filename: str = DEFAULT_EMBED_SPACER_FILENAME,
    width: int = DEFAULT_EMBED_SPACER_WIDTH,
    height: int = DEFAULT_EMBED_SPACER_HEIGHT,
) -> File:
    """Builds a fresh transparent PNG upload for one Discord send or edit.

    A new `File` per call is a requirement rather than tidiness: nextcord documents a `File` as
    single-use, because the request that sends it reads its buffer to the end and `File.reset`
    rewinds only inside one request's own retry loop, so a second send would start at EOF. The
    encoded bytes behind it are cached, so the only extra cost is the `BytesIO` wrapper.

    Args:
        filename (str): The name to upload under, matching what `embed_spacer_url` was built
            from.
        width (int): Spacer width in pixels, which is what widens the embeds carrying it.
        height (int): Spacer height in pixels, kept at 1 so it adds no visible vertical space.

    Returns:
        A `File` to pass in a send or edit's `files=`.
    """
    return File(
        fp=BytesIO(initial_bytes=_transparent_png_bytes(width=width, height=height)),
        filename=filename,
    )


def _embed_has_real_image(*, embed: Embed, spacer_url: str) -> bool:
    """Whether an embed already shows an image of its own.

    Only `set_image` counts: a thumbnail leaves the embed narrow, so one that has only a
    thumbnail still needs a spacer. A spacer applied by an earlier call does not count either,
    which is what lets the same embed object be re-sent.

    Args:
        embed (Embed): The embed to inspect.
        spacer_url (str): The spacer url to discount, as built by `embed_spacer_url`.

    Returns:
        True when the embed's image is a real one rather than the spacer.
    """
    image_url = embed.image.url if embed.image else None
    return bool(image_url and image_url != spacer_url)


def _target_allows_file_uploads(*, target: object | None) -> bool:
    """Whether the bot may attach a file where this send or edit is going.

    The check earns its place because uploading into a channel that denies `attach_files` costs
    the whole message, not just the alignment. It is duck-typed over `object` because a caller
    holds a `Context`, a `Message` or an `Interaction` and the bot's own member resolves
    differently on each, tried in that order: `Context.me`, then `guild.me`, then a `get_member`
    lookup on the client's user id, whose handle is `Context.bot` or `Interaction.client` since
    a `Message` carries neither. Every step that cannot answer falls through to True, a DM's
    absent guild included, so only a resolved member plus a channel whose permissions actually
    say `attach_files` is False returns False.

    Args:
        target (object | None): The context, message or interaction the payload is destined for,
            or None when the caller has nothing to check against.

    Returns:
        False only when the destination channel clearly denies `attach_files`.
    """
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
    """Sets the transparent spacer on every embed that has no image of its own.

    Mutates the embeds in place and hands the same list back for chaining. Production reaches
    it only through `embed_spacer_payload`, which calls it once it knows the spacer will really
    be in the payload, so an embed is never left pointing at a file that is not sent.

    Args:
        embeds (list[Embed]): The embeds of one message, mutated in place.
        filename (str): The spacer's upload name.

    Returns:
        The same list that was passed in.
    """
    spacer_url = embed_spacer_url(filename=filename)
    for embed in embeds:
        if not _embed_has_real_image(embed=embed, spacer_url=spacer_url):
            embed.set_image(url=spacer_url)
    return embeds


def _existing_spacer_attachment(*, target: object | None, filename: str) -> Attachment | None:
    """Finds a spacer already uploaded on the message an edit is about to rewrite.

    Handles both target shapes: a `Message` carries `attachments` itself, while an
    `Interaction` reaches them through its `.message`. Matched by filename, which is what makes
    an unchanged spacer retainable by id rather than re-uploaded.

    Args:
        target (object | None): The message or interaction being edited.
        filename (str): The spacer's upload name.

    Returns:
        The existing spacer attachment, or None when the message carries none.
    """
    message = target if hasattr(target, "attachments") else getattr(target, "message", None)
    attachments = getattr(message, "attachments", None) or ()
    for attachment in attachments:
        if getattr(attachment, "filename", None) == filename:
            return attachment
    return None


def embed_spacer_payload(
    *,
    embeds: list[Embed],
    is_edit: bool,
    target: object | None = None,
    extra_files: list[File] | None = None,
    filename: str = DEFAULT_EMBED_SPACER_FILENAME,
) -> dict[str, Any]:
    """Builds the spacer's files/attachments increment to splat into a send or edit.

    The spacer never changes, so an edit retains an already-uploaded spacer by id
    instead of re-uploading it. Re-uploading the same spacer on every edit trips
    Discord's per-message edit attachment upload limit (error code 400009) for
    rapidly edited messages such as the Blackjack table.

    `embeds` is mutated in place: the spacer url goes on when it will really be uploaded or
    retained, and comes back off when it cannot be, which is the case for a channel denying
    `attach_files` and for `extra_files` already filling Discord's 10-attachment cap. Clearing
    it matters because these embed objects are reused across sends, so a leftover url would
    point at a file this payload does not carry.

    `attachments` is emitted only for an edit, since send methods reject it, and it lists just
    the retained spacer. Discord reads that list as everything the message keeps, so an edit
    drops the caller's earlier uploads; a call site that wants one back re-uploads it through
    `extra_files`.

    Args:
        embeds (list[Embed]): The message's embeds, mutated in place.
        is_edit (bool): Whether this payload is for an edit rather than a send.
        target (object | None): The context, message or interaction being edited or sent to,
            used to find an already-uploaded spacer and to check upload permission.
        extra_files (list[File] | None): The caller's own uploads, which keep their order in
            front of the spacer and take precedence over it at the attachment cap.
        filename (str): The spacer's upload name.

    Returns:
        The keyword arguments to splat into the send or edit: `files` when anything is being
        uploaded, plus `attachments` on every edit, empty list included.
    """
    spacer_url = embed_spacer_url(filename=filename)
    needs_spacer = any(
        not _embed_has_real_image(embed=embed, spacer_url=spacer_url) for embed in embeds
    )
    files: list[File] = list(extra_files or [])
    retained: list[Attachment] = []
    existing_spacer = (
        _existing_spacer_attachment(target=target, filename=filename) if is_edit else None
    )
    can_upload_spacer = _target_allows_file_uploads(target=target)
    if needs_spacer and existing_spacer is not None:
        apply_embed_spacer_image(embeds=embeds, filename=filename)
        retained.append(existing_spacer)
    elif needs_spacer and can_upload_spacer and len(files) < DISCORD_MAX_FILES_PER_MESSAGE:
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
        payload["attachments"] = retained
    return payload


@cache
def _transparent_png_bytes(*, width: int, height: int) -> bytes:
    """Encodes a fully transparent PNG of the given size.

    Cached because the bytes are identical on every send and a Pillow encode per message edit
    would sit on the event loop. `bytes` is immutable, so one cached buffer is safe to hand to
    any number of `File` wrappers.

    Args:
        width (int): Width in pixels, half of the cache key.
        height (int): Height in pixels, the other half.

    Returns:
        The encoded PNG bytes, the same object on every call for a given size.
    """
    image = Image.new(mode="RGBA", size=(width, height), color=_TRANSPARENT_RGBA)
    buffer = BytesIO()
    image.save(fp=buffer, format="PNG", optimize=True)
    return buffer.getvalue()
