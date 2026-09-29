"""The bot's own application emoji a reply turn puts on the message it answers.

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
# Marks a reply grounded in a watched YouTube video.
YOUTUBE_EMOJI: Final[str] = "<:youtube:1517546722535018596>"
