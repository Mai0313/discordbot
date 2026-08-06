"""Structural assertions over Discord embeds.

Tests used to pin whole localized strings (full titles, footers, field copy),
which breaks on any wording or emoji refresh even though the behavior is
unchanged. These assert the embed's *shape* — that a named field exists, that a
title carries its category marker — and hand the field back so the caller checks
only the value that actually encodes behavior (an amount, a status).

Two helpers live here: ``assert_embed_has_field`` for a named field and
``assert_embed_title_prefix`` for a title's leading marker. Both fail with the
surrounding context (the names actually present, the title actually set) so a
failure names what the embed carried rather than only what it missed. Cog smoke
tests are the caller that matters, reading economy embeds field by field; the
helpers themselves are pinned by ``tests/test_helpers.py``.
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
    """Asserts a field with the given name exists and returns it for value checks.

    The failure message lists every field name the embed carried, since the usual cause is a
    renamed or conditionally omitted field rather than a wrong value. nextcord appends fields
    without deduplicating names, so the first match wins.

    Args:
        embed (Embed): The embed to search.
        name (str): Exact field name to match.

    Returns:
        The matching field, whose ``value`` the caller then asserts on.

    Raises:
        AssertionError: The embed carries no field with that name.
    """
    for field in embed.fields:
        if field.name == name:
            return field
    available = [field.name for field in embed.fields]
    raise AssertionError(f"embed has no field named {name!r}; fields present: {available}")


def assert_embed_title_prefix(embed: Embed, prefix: str) -> None:
    """Asserts the embed title starts with the given prefix (a category marker).

    An embed that never set a title is read as an empty one, so it fails the assertion with the
    same message instead of raising ``AttributeError`` off ``None``.

    Args:
        embed (Embed): The embed whose title is checked.
        prefix (str): Leading marker the title must carry, typically the category emoji.
    """
    title = embed.title or ""
    assert title.startswith(prefix), f"embed title {title!r} does not start with {prefix!r}"
