"""Structural assertions over Discord embeds.

Tests used to pin whole localized strings (full titles, footers, field copy),
which breaks on any wording or emoji refresh even though the behavior is
unchanged. These assert the embed's *shape* — that a named field exists, that a
title carries its category marker — and hand the field back so the caller checks
only the value that actually encodes behavior (an amount, a status).
"""

from typing import Protocol

from nextcord import Embed


class EmbedField(Protocol):
    """The field shape ``Embed.fields`` yields at runtime.

    Mirrors nextcord's TYPE_CHECKING-only ``_EmbedFieldProxy`` so this module
    never imports a private name.
    """

    name: str | None
    value: str | None
    inline: bool


def assert_embed_has_field(embed: Embed, name: str) -> EmbedField:
    """Asserts a field with the given name exists and returns it for value checks."""
    for field in embed.fields:
        if field.name == name:
            return field
    available = [field.name for field in embed.fields]
    raise AssertionError(f"embed has no field named {name!r}; fields present: {available}")


def assert_embed_title_prefix(embed: Embed, prefix: str) -> None:
    """Asserts the embed title starts with the given prefix (a category marker)."""
    title = embed.title or ""
    assert title.startswith(prefix), f"embed title {title!r} does not start with {prefix!r}"
