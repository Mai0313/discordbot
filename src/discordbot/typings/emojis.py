"""Platform link-source names and configurable emoji markers shared across cogs.

`LinkSourceName` is the canonical spelling for registry entries, `pending_expansion` rows,
route outputs, and `LINK_SOURCE_EMOJIS` keys. Every emoji defaults to standard Unicode so
fresh deployments work without custom Discord Application Emojis, while env vars override
each marker with custom `<:name:id>` values.
"""

from typing import Final, Literal

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings

# Plain alias rather than a `type` statement so `RouteClassification`'s JSON schema inlines
# the enum instead of hoisting it into `$defs` behind a `$ref`.
LinkSourceName = Literal["threads", "facebook", "instagram", "twitter", "douyin", "bilibili"]


class EmojiConfig(BaseSettings):
    """Discord reaction and status emojis, defaulting to Unicode with env overrides."""

    facebook: str = Field(
        default="📘",
        description="Reaction emoji marking a Facebook link.",
        examples=["📘", "<:facebook:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_FACEBOOK"),
    )
    instagram: str = Field(
        default="📷",
        description="Reaction emoji marking an Instagram link.",
        examples=["📷", "<:instagram:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_INSTAGRAM"),
    )
    threads: str = Field(
        default="🧵",
        description="Reaction emoji marking a Threads link.",
        examples=["🧵", "<:threads:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_THREADS"),
    )
    twitter: str = Field(
        default="🐦",
        description="Reaction emoji marking a Twitter/X link.",
        examples=["🐦", "<:twitter:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_TWITTER"),
    )
    douyin: str = Field(
        default="🎵",
        description="Reaction emoji marking a Douyin link.",
        examples=["🎵", "<:douyin:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_DOUYIN"),
    )
    bilibili: str = Field(
        default="📺",
        description="Reaction emoji marking a Bilibili link.",
        examples=["📺", "<:bilibili:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_BILIBILI"),
    )
    done: str = Field(
        default="✅",
        description="Reaction emoji marking a completed reply or link expansion.",
        examples=["✅", "<:greencheck:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_DONE"),
    )
    failed: str = Field(
        default="❌",
        description="Reaction emoji marking a failed reply or link expansion.",
        examples=["❌", "<:redcross:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_FAILED"),
    )
    routing: str = Field(
        default="🔀",
        description="Reaction emoji shown while routing a reply turn.",
        examples=["🔀", "<:flowchart:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_ROUTING"),
    )
    image: str = Field(
        default="🖼️",
        description="Reaction emoji shown when generating an image.",
        examples=["🖼️", "<:image:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_IMAGE"),
    )
    video: str = Field(
        default="🎬",
        description="Reaction emoji shown when generating a video.",
        examples=["🎬", "<:video:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_VIDEO"),
    )
    answer: str = Field(
        default="💬",
        description="Reaction and thinking-preview emoji shown while streaming an answer.",
        examples=["💬", "<:message:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_ANSWER"),
    )
    voice: str = Field(
        default="🗣️",
        description="Reaction emoji shown when generating an inline voice clip.",
        examples=["🗣️", "<:voice:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_VOICE"),
    )
    youtube: str = Field(
        default="▶️",
        description="Reaction emoji shown when watching a linked YouTube video.",
        examples=["▶️", "<:youtube:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_YOUTUBE"),
    )
    research: str = Field(
        default="🔬",
        description="Reaction emoji placed on the anchor message of a deep-research thread.",
        examples=["🔬", "<:dino:1234567890123456789>"],
        validation_alias=AliasChoices("EMOJI_RESEARCH"),
    )


_emojis = EmojiConfig()

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
