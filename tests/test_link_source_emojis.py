"""Every registered link source has a platform emoji to mark a read with.

`ReplyPipeline` looks its marker up with a plain subscript, so a source registered without an
entry would raise mid-reply — after the builders started and before the answer streams. That
failure is invisible until someone posts that platform's link, which is why the domain is
enumerated here instead of being softened into a `.get()` at the call site.
"""

from discordbot.typings.emojis import LINK_SOURCE_EMOJIS
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES


def test_every_registered_source_has_a_marker() -> None:
    """A source the router can select is a source the pipeline will try to mark."""
    registered = {source.name for source in LINK_CONTEXT_SOURCES}

    assert registered <= set(LINK_SOURCE_EMOJIS)


def test_no_marker_names_a_source_that_is_gone() -> None:
    """The other direction, so a removed source does not leave its emoji behind."""
    registered = {source.name for source in LINK_CONTEXT_SOURCES}

    assert set(LINK_SOURCE_EMOJIS) <= registered


def test_every_marker_is_a_custom_emoji_reference() -> None:
    """A malformed reference is accepted by the API and then renders as literal text."""
    for name, emoji in LINK_SOURCE_EMOJIS.items():
        assert emoji.startswith("<:"), name
        assert emoji.endswith(">"), name
        assert emoji.strip("<>").split(":")[-1].isdigit(), name
