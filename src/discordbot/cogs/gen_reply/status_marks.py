"""The marks a reply turn shows: its reactions on the message it answers, and its notes' symbols.

Each is named once so one symbol means one thing wherever a turn shows it. They are the reply
turn's own vocabulary: the auto-expansion marks spell theirs separately, even where the two
render the same.
"""

from typing import Final

# The status chain, in the order a turn advances it: routing, then the route it took, then how
# it ended.
ROUTING_EMOJI: Final[str] = "<:flowchart:1517561877973045349>"
IMAGE_EMOJI: Final[str] = "<:image:1517559727880667226>"
VIDEO_EMOJI: Final[str] = "<:video:1517560671913377842>"
ANSWER_EMOJI: Final[str] = "<:message:1517560873000898860>"
DONE_EMOJI: Final[str] = "<:greencheck:1517565102424068226>"
FAILED_EMOJI: Final[str] = "<:redcross:1517565100838355016>"

# Marks a spoken clip is being synthesized for the reply.
VOICE_EMOJI: Final[str] = "<:voice:1517558121092878376>"
# Marks a music clip is being rendered for the reply; there is no app emoji for music.
MUSIC_EMOJI: Final[str] = "🎵"
# Marks a reply grounded in a watched YouTube video.
YOUTUBE_EMOJI: Final[str] = "<:youtube:1517546722535018596>"

# Best-effort media the reply goes out without: a clip that ran out of time, and anything
# dropped for another reason.
TIMEOUT_HINT_EMOJI: Final[str] = "⏱️"
DROPPED_HINT_EMOJI: Final[str] = "⚠️"

# Shown on both retry surfaces, so the reaction on the source message and the notice on the
# reply read as the same event rather than two unrelated hints.
RETRY_HINT_EMOJI: Final[str] = "🔁"

# One symbol per memory action, because all three notes stack in the same corner of the same
# reply: with one shared symbol a reader cannot tell "I read this person's memory" from "I wrote
# this down about you" without parsing the whole sentence, and can mistake the read credit for
# something the bot just recorded. Each note also opens on its VERB for the same reason. Plain
# unicode rather than app emoji because uploading one is not something the bot can do; swapping
# in a custom `<:name:id>` later is a change to these three lines.
MEMORY_READ_EMOJI: Final[str] = "📖"
MEMORY_WRITE_EMOJI: Final[str] = "✏️"
MEMORY_FORGET_EMOJI: Final[str] = "🩹"
