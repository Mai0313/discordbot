"""Tests for shared Discord embed helpers."""

from nextcord import Embed

from discordbot.utils.discord_embeds import (
    DEFAULT_EMBED_SPACER_FILENAME,
    embed_spacer_url,
    build_embed_spacer_file,
    apply_embed_spacer_image,
)


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
