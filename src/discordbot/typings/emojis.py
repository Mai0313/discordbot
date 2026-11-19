"""Platform link-source names and custom emoji markers shared by expansion and reply paths.

`LinkSourceName` is the canonical spelling for registry entries, `pending_expansion` rows,
route outputs, and `LINK_SOURCE_EMOJIS` keys. Expansion status emojis stay beside the
expansion protocol.
"""

from typing import Final, Literal

# Plain alias rather than a `type` statement so `RouteClassification`'s JSON schema inlines
# the enum instead of hoisting it into `$defs` behind a `$ref`.
LinkSourceName = Literal["threads", "facebook", "instagram", "twitter", "douyin", "bilibili"]

FACEBOOK_EMOJI: Final[str] = "<:facebook:1546179601288396810>"
INSTAGRAM_EMOJI: Final[str] = "<:instagram:1546181126945505340>"
THREADS_EMOJI: Final[str] = "<:threads:1546180328639434923>"
TWITTER_EMOJI: Final[str] = "<:twitter:1558536389287747604>"
DOUYIN_EMOJI: Final[str] = "<:douyin:1546180677710385304>"
BILIBILI_EMOJI: Final[str] = "<:bilibili:1546180944615051344>"

LINK_SOURCE_EMOJIS: Final[dict[str, str]] = {
    "threads": THREADS_EMOJI,
    "facebook": FACEBOOK_EMOJI,
    "instagram": INSTAGRAM_EMOJI,
    "twitter": TWITTER_EMOJI,
    "douyin": DOUYIN_EMOJI,
    "bilibili": BILIBILI_EMOJI,
}
