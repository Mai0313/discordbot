"""Every registered link source has a platform emoji to mark a read with.

`ReplyPipeline` looks its marker up with a plain subscript, so a source registered without an
entry would raise mid-reply — after the builders started and before the answer streams. That
failure is invisible until someone posts that platform's link, which is why the domain is
enumerated here instead of being softened into a `.get()` at the call site.
"""

from discordbot.typings.emojis import LINK_SOURCE_EMOJIS
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES


def test_the_markers_name_exactly_the_registered_sources() -> None:
    """Every source the router can select has a marker, and no marker outlives its source."""
    assert set(LINK_SOURCE_EMOJIS) == {source.name for source in LINK_CONTEXT_SOURCES}


def test_every_marker_is_a_custom_emoji_reference() -> None:
    """A malformed reference is accepted by the API and then renders as literal text."""
    for name, emoji in LINK_SOURCE_EMOJIS.items():
        assert emoji.startswith("<:"), name
        assert emoji.endswith(">"), name
        assert emoji.strip("<>").split(":")[-1].isdigit(), name
