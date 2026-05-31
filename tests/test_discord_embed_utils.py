"""Tests for shared Discord embed helpers."""

from types import SimpleNamespace

from nextcord import Embed

from discordbot.utils.discord_embeds import (
    DEFAULT_EMBED_SPACER_FILENAME,
    embed_spacer_url,
    embed_spacer_payload,
    build_embed_spacer_file,
    apply_embed_spacer_image,
)


def _permission_target(*, attach_files: bool) -> SimpleNamespace:
    """Builds a Discord target stub with channel permissions."""
    member = object()
    guild = SimpleNamespace(me=member)

    class _Channel:
        def __init__(self) -> None:
            self.guild = guild

        @staticmethod
        def permissions_for(target_member: object) -> SimpleNamespace:
            assert target_member is member
            return SimpleNamespace(attach_files=attach_files)

    return SimpleNamespace(guild=guild, channel=_Channel())


def test_apply_embed_spacer_image_sets_attachment_url() -> None:
    """Spacer image helpers keep multiple embeds on the same rendered width."""
    embeds = [Embed(description="short"), Embed(description="also short")]

    result = apply_embed_spacer_image(embeds=embeds)

    assert result is embeds
    assert [embed.image.url for embed in embeds] == [embed_spacer_url(), embed_spacer_url()]


def test_build_embed_spacer_file_returns_fresh_png_upload() -> None:
    """Each send or edit gets its own Discord File object."""
    first = build_embed_spacer_file()
    second = build_embed_spacer_file()

    assert first is not second
    assert first.filename == DEFAULT_EMBED_SPACER_FILENAME
    assert first.fp.read(8) == b"\x89PNG\r\n\x1a\n"


def test_apply_embed_spacer_image_skips_embeds_with_real_image() -> None:
    """Embeds that already show a real image are never overwritten by the spacer."""
    with_image = Embed(description="has image")
    with_image.set_image(url="https://cdn.test/board.png")
    text_only = Embed(description="text only")

    apply_embed_spacer_image(embeds=[with_image, text_only])

    assert with_image.image.url == "https://cdn.test/board.png"
    assert text_only.image.url == embed_spacer_url()


def test_apply_embed_spacer_image_treats_thumbnail_as_image_less() -> None:
    """A thumbnail is not an image, so the embed still receives a spacer."""
    embed = Embed(description="thumb only")
    embed.set_thumbnail(url="https://cdn.test/avatar.png")

    apply_embed_spacer_image(embeds=[embed])

    assert embed.image.url == embed_spacer_url()


def test_embed_spacer_payload_edit_adds_spacer_file_and_clears_attachments() -> None:
    """An edit without an existing spacer uploads a fresh one and clears stale attachments."""
    embed = Embed(description="text")

    payload = embed_spacer_payload(embeds=[embed], is_edit=True)

    assert payload["attachments"] == []
    assert payload["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert embed.image.url == embed_spacer_url()


def test_embed_spacer_payload_edit_retains_existing_spacer_without_reupload() -> None:
    """An edit reuses an already-uploaded spacer by id instead of re-uploading it."""
    embed = Embed(description="text")
    spacer = SimpleNamespace(filename=DEFAULT_EMBED_SPACER_FILENAME)
    other = SimpleNamespace(filename="board.png")
    target = SimpleNamespace(attachments=[other, spacer])

    payload = embed_spacer_payload(embeds=[embed], is_edit=True, target=target)

    assert "files" not in payload
    assert payload["attachments"] == [spacer]
    assert embed.image.url == embed_spacer_url()


def test_embed_spacer_payload_edit_retains_spacer_from_interaction_message() -> None:
    """An interaction edit target resolves the spacer through its message attachments."""
    embed = Embed(description="text")
    spacer = SimpleNamespace(filename=DEFAULT_EMBED_SPACER_FILENAME)
    target = SimpleNamespace(message=SimpleNamespace(attachments=[spacer]))

    payload = embed_spacer_payload(embeds=[embed], is_edit=True, target=target)

    assert "files" not in payload
    assert payload["attachments"] == [spacer]
    assert embed.image.url == embed_spacer_url()


def test_embed_spacer_payload_send_omits_attachments() -> None:
    """A send payload never carries attachments, which send methods reject."""
    embed = Embed(description="text")

    payload = embed_spacer_payload(embeds=[embed], is_edit=False)

    assert "attachments" not in payload
    assert payload["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME


def test_embed_spacer_payload_skips_spacer_without_attach_files_permission() -> None:
    """A text-only send stays usable when the bot cannot upload attachments."""
    embed = Embed(description="text")

    payload = embed_spacer_payload(
        embeds=[embed], is_edit=False, target=_permission_target(attach_files=False)
    )

    assert payload == {}
    assert not embed.image.url


def test_embed_spacer_payload_removes_stale_spacer_without_attach_files_permission() -> None:
    """An edit without upload permission drops stale spacer attachment references."""
    embed = Embed(description="text")
    embed.set_image(url=embed_spacer_url())

    payload = embed_spacer_payload(
        embeds=[embed], is_edit=True, target=_permission_target(attach_files=False)
    )

    assert payload == {"attachments": []}
    assert not embed.image.url


def test_embed_spacer_payload_reuploads_existing_spacer_image() -> None:
    """A reused spacer embed still gets a fresh upload for the new message."""
    embed = Embed(description="text")
    embed_spacer_payload(embeds=[embed], is_edit=True)

    payload = embed_spacer_payload(embeds=[embed], is_edit=False)

    assert "attachments" not in payload
    assert payload["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert embed.image.url == embed_spacer_url()


def test_embed_spacer_payload_real_image_keeps_only_extra_files() -> None:
    """A real-image embed adds no spacer upload but preserves the caller's file."""
    embed = Embed(description="board")
    embed.set_image(url="https://cdn.test/board.png")
    real_file = build_embed_spacer_file(filename="board.png")

    payload = embed_spacer_payload(embeds=[embed], is_edit=False, extra_files=[real_file])

    assert payload["files"] == [real_file]
    assert embed.image.url == "https://cdn.test/board.png"


def test_embed_spacer_payload_is_empty_when_nothing_needed() -> None:
    """A real-image send with no extra files produces an empty payload."""
    embed = Embed(description="board")
    embed.set_image(url="https://cdn.test/board.png")

    assert embed_spacer_payload(embeds=[embed], is_edit=False) == {}


def test_embed_spacer_payload_merges_extra_files_with_spacer() -> None:
    """Mixed embeds keep the real upload first and append a single spacer upload."""
    real_file = build_embed_spacer_file(filename="video.mp4")
    with_image = Embed(description="image")
    with_image.set_image(url="https://cdn.test/photo.png")
    text_only = Embed(description="text")

    payload = embed_spacer_payload(
        embeds=[with_image, text_only], is_edit=False, extra_files=[real_file]
    )

    assert payload["files"][0] is real_file
    assert payload["files"][1].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert with_image.image.url == "https://cdn.test/photo.png"
    assert text_only.image.url == embed_spacer_url()


def test_embed_spacer_payload_skips_spacer_when_extra_files_fill_discord_limit() -> None:
    """A full file payload keeps the real files and leaves text embeds unmodified."""
    files = [build_embed_spacer_file(filename=f"video-{index}.mp4") for index in range(10)]
    embed = Embed(description="video-only post")

    payload = embed_spacer_payload(embeds=[embed], is_edit=False, extra_files=files)

    assert payload["files"] == files
    assert not embed.image.url


def test_embed_spacer_payload_removes_stale_spacer_when_file_limit_is_full() -> None:
    """A reused spacer embed drops the missing attachment reference when upload is full."""
    files = [build_embed_spacer_file(filename=f"video-{index}.mp4") for index in range(10)]
    embed = Embed(description="video-only post")
    embed.set_image(url=embed_spacer_url())

    payload = embed_spacer_payload(embeds=[embed], is_edit=False, extra_files=files)

    assert payload["files"] == files
    assert not embed.image.url
