"""Reaction emojis and footnote symbols for reply turns.

Single source of truth for the reply turn's vocabulary, kept separate from auto-expansion marks.
"""

from typing import Final

from discordbot.typings.emojis import (
    DONE_EMOJI,
    IMAGE_EMOJI,
    VIDEO_EMOJI,
    VOICE_EMOJI,
    ANSWER_EMOJI,
    FAILED_EMOJI,
    ROUTING_EMOJI,
    YOUTUBE_EMOJI,
)

# Marks a message that asked for nothing, beside the bare `?` reply.
EMPTY_PROMPT_EMOJI: Final[str] = "❓"
# No app emoji exists for music.
MUSIC_EMOJI: Final[str] = "🎵"

# Best-effort dropped media: timed-out voice vs any other dropped media.
TIMEOUT_HINT_EMOJI: Final[str] = "⏱️"
DROPPED_HINT_EMOJI: Final[str] = "⚠️"

# Shared between the source reaction and the reply's retrying notice.
RETRY_HINT_EMOJI: Final[str] = "🔁"

# Memory footnote symbols; utils/llm_transcript.py::USAGE_FOOTER_RE matches write and forget too.
MEMORY_READ_EMOJI: Final[str] = "📖"
MEMORY_WRITE_EMOJI: Final[str] = "✏️"
MEMORY_FORGET_EMOJI: Final[str] = "🩹"

__all__ = [
    "ANSWER_EMOJI",
    "DONE_EMOJI",
    "DROPPED_HINT_EMOJI",
    "EMPTY_PROMPT_EMOJI",
    "FAILED_EMOJI",
    "IMAGE_EMOJI",
    "MEMORY_FORGET_EMOJI",
    "MEMORY_READ_EMOJI",
    "MEMORY_WRITE_EMOJI",
    "MUSIC_EMOJI",
    "RETRY_HINT_EMOJI",
    "ROUTING_EMOJI",
    "TIMEOUT_HINT_EMOJI",
    "VIDEO_EMOJI",
    "VOICE_EMOJI",
    "YOUTUBE_EMOJI",
]
