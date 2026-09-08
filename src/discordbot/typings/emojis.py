"""Custom emoji marking which platform's link the bot just read.

Shared because two layers put the SAME marker on the same message and cannot import each
other: an expansion cog (`parse_threads`, `parse_douyin`) marks a link it expanded into the
channel, while `gen_reply` marks one it read into an answer instead, and
`tests/test_package_layering.py` forbids a cog reaching across to another for it. Bilibili
has no expansion cog at all, so the reply path is the only thing that ever marks it.

Scope is the platform markers alone. The status emoji an auto-expansion answers with moved
to `utils/expansion_placeholder.py` instead, beside the protocol that gives each of them its
meaning: they are one feature's vocabulary rather than a cross-layer marker, and the shared
restart sweep needs them anyway. `gen_reply` still spells its own three inline, since its
reactions track a reply turn rather than an expansion outcome.

The keys below are also what the expansion cogs name their `pending_expansion` rows by, so a
platform is spelled one way across the reply path, the four cogs and that table.
"""

from typing import Final

FACEBOOK_EMOJI: Final[str] = "<:facebook:1546179601288396810>"
INSTAGRAM_EMOJI: Final[str] = "<:instagram:1546181126945505340>"
THREADS_EMOJI: Final[str] = "<:threads:1546180328639434923>"
DOUYIN_EMOJI: Final[str] = "<:douyin:1546180677710385304>"
BILIBILI_EMOJI: Final[str] = "<:bilibili:1546180944615051344>"

# Keyed by `LinkContextSource.name`, so the reply path can mark whichever sources it read.
# `tests/test_link_source_emojis.py` pins every registered source to an entry here, which is
# what lets the lookup be a plain subscript: a source added without one fails the test rather
# than raising mid-reply.
LINK_SOURCE_EMOJIS: Final[dict[str, str]] = {
    "threads": THREADS_EMOJI,
    "facebook": FACEBOOK_EMOJI,
    "instagram": INSTAGRAM_EMOJI,
    "douyin": DOUYIN_EMOJI,
    "bilibili": BILIBILI_EMOJI,
}

__all__ = [
    "BILIBILI_EMOJI",
    "DOUYIN_EMOJI",
    "FACEBOOK_EMOJI",
    "INSTAGRAM_EMOJI",
    "LINK_SOURCE_EMOJIS",
    "THREADS_EMOJI",
]
