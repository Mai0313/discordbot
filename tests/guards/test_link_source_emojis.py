"""Every registered link source has a platform emoji to mark a read with.

`ReplyPipeline` looks its marker up with a plain subscript, so a source registered without an
entry would raise mid-reply — after the builders started and before the answer streams. That
failure is invisible until someone posts that platform's link, which is why the domain is
enumerated here instead of being softened into a `.get()` at the call site.
"""

import re
from typing import get_args
from pathlib import Path

from discordbot.typings.config import EmojiConfig, load_bot_config
from discordbot.typings.emojis import LINK_SOURCE_EMOJIS, LinkSourceName
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES


def test_the_markers_name_exactly_the_registered_sources() -> None:
    """Every source the router can select has a marker, and no marker outlives its source."""
    assert set(LINK_SOURCE_EMOJIS) == {source.name for source in LINK_CONTEXT_SOURCES}


def test_the_route_vocabulary_names_exactly_the_marked_sources() -> None:
    """A name left in the vocabulary after its source is gone stays selectable and reads nothing.

    The other direction fails at import, where a source registers under a name the vocabulary
    lacks.
    """
    assert set(get_args(LinkSourceName)) == set(LINK_SOURCE_EMOJIS)


def test_a_custom_emoji_marker_is_a_well_formed_reference() -> None:
    """A malformed reference is accepted by the API and then renders as literal text.

    Static or animated is the marker's choice, as is a plain Unicode emoji; a reference that is
    neither is the bug.
    """
    for name, emoji in LINK_SOURCE_EMOJIS.items():
        if any(mark in emoji for mark in "<:>"):
            assert re.fullmatch(r"<a?:\w+:\d+>", emoji), name


def test_default_emojis_are_unicode_and_link_sources_are_distinct() -> None:
    """Default emojis need no custom Discord Application Emojis and keep link markers distinct."""
    defaults = EmojiConfig()
    for field_name, value in defaults.model_dump().items():
        assert value, field_name
        assert not any(mark in value for mark in "<:>"), field_name
    assert len(set(LINK_SOURCE_EMOJIS.values())) == len(LINK_SOURCE_EMOJIS)


def test_emoji_config_reads_toml_overrides(bot_config_isolated: Path) -> None:
    """Local `config.toml` overrides default Unicode emojis with custom app emoji references."""
    bot_config_isolated.write_text(
        data=(
            "[global.emojis]\n"
            'facebook = "<:facebook:1546179601288396810>"\n'
            'research = "<:dino:1517560319281594570>"\n'
        ),
        encoding="utf-8",
    )

    config = load_bot_config().global_config.emojis

    assert config.facebook == "<:facebook:1546179601288396810>"
    assert config.research == "<:dino:1517560319281594570>"
    assert config.done == "✅"
