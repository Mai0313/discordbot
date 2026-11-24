"""Platform link-source names and configurable emoji markers shared across cogs.

`LinkSourceName` is the canonical spelling for registry entries, `pending_expansion` rows,
route outputs, and `LINK_SOURCE_EMOJIS` keys. Emoji values load from `[global.emojis]` in
`src/discordbot/config.toml` merged with `./data/config.toml`, defaulting to standard Unicode
so fresh deployments work without custom Discord Application Emojis.
"""

from typing import Final, Literal

from discordbot.typings.config import load_bot_config

# Plain alias rather than a `type` statement so `RouteClassification`'s JSON schema inlines
# the enum instead of hoisting it into `$defs` behind a `$ref`.
LinkSourceName = Literal["threads", "facebook", "instagram", "twitter", "douyin", "bilibili"]

_emojis = load_bot_config().global_config.emojis

FACEBOOK_EMOJI: Final[str] = _emojis.facebook
INSTAGRAM_EMOJI: Final[str] = _emojis.instagram
THREADS_EMOJI: Final[str] = _emojis.threads
TWITTER_EMOJI: Final[str] = _emojis.twitter
DOUYIN_EMOJI: Final[str] = _emojis.douyin
BILIBILI_EMOJI: Final[str] = _emojis.bilibili

DONE_EMOJI: Final[str] = _emojis.done
FAILED_EMOJI: Final[str] = _emojis.failed
ROUTING_EMOJI: Final[str] = _emojis.routing
IMAGE_EMOJI: Final[str] = _emojis.image
VIDEO_EMOJI: Final[str] = _emojis.video
ANSWER_EMOJI: Final[str] = _emojis.answer
VOICE_EMOJI: Final[str] = _emojis.voice
YOUTUBE_EMOJI: Final[str] = _emojis.youtube
RESEARCH_EMOJI: Final[str] = _emojis.research

LINK_SOURCE_EMOJIS: Final[dict[str, str]] = {
    "threads": THREADS_EMOJI,
    "facebook": FACEBOOK_EMOJI,
    "instagram": INSTAGRAM_EMOJI,
    "twitter": TWITTER_EMOJI,
    "douyin": DOUYIN_EMOJI,
    "bilibili": BILIBILI_EMOJI,
}
