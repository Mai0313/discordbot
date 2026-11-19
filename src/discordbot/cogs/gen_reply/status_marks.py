"""Reaction emojis and footnote symbols for reply turns.

Single source of truth for the reply turn's vocabulary, kept separate from auto-expansion marks.
"""

from typing import Final

# Status chain in turn order: routing, route taken, terminal outcome.
ROUTING_EMOJI: Final[str] = "<:flowchart:1517561877973045349>"
IMAGE_EMOJI: Final[str] = "<:image:1517559727880667226>"
VIDEO_EMOJI: Final[str] = "<:video:1517560671913377842>"
ANSWER_EMOJI: Final[str] = "<:message:1517560873000898860>"
DONE_EMOJI: Final[str] = "<:greencheck:1517565102424068226>"
FAILED_EMOJI: Final[str] = "<:redcross:1517565100838355016>"

# Marks a message that asked for nothing, beside the bare `?` reply.
EMPTY_PROMPT_EMOJI: Final[str] = "❓"
VOICE_EMOJI: Final[str] = "<:voice:1517558121092878376>"
# No app emoji exists for music.
MUSIC_EMOJI: Final[str] = "🎵"
YOUTUBE_EMOJI: Final[str] = "<:youtube:1517546722535018596>"

# Best-effort dropped media: timed-out voice vs any other dropped media.
TIMEOUT_HINT_EMOJI: Final[str] = "⏱️"
DROPPED_HINT_EMOJI: Final[str] = "⚠️"

# Shared between the source reaction and the reply's retrying notice.
RETRY_HINT_EMOJI: Final[str] = "🔁"

# Memory footnote symbols; utils/llm_transcript.py::USAGE_FOOTER_RE matches write and forget too.
MEMORY_READ_EMOJI: Final[str] = "📖"
MEMORY_WRITE_EMOJI: Final[str] = "✏️"
MEMORY_FORGET_EMOJI: Final[str] = "🩹"
