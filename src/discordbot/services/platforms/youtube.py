"""YouTube video URL pattern.

Matches watchable single-video forms (`watch?v=`, `youtu.be/`, `/shorts/`, `/live/`) and excludes
channel, playlist, and user pages.
"""

import re

from discordbot.utils.urls import URL_START_ANCHOR

# The 11-character video id uses the `[A-Za-z0-9_-]` YouTube id alphabet; the optional ASCII query
# tail must end on `[A-Za-z0-9_-]` so a link written mid-sentence stops cleanly before CJK or
# ASCII sentence punctuation.
YOUTUBE_URL_RE = re.compile(
    pattern=rf"{URL_START_ANCHOR}https?://(?:www\.|m\.)?"
    r"(?:"
    r"youtube\.com/(?:watch\?[A-Za-z0-9=&%_.-]*v=[A-Za-z0-9_-]{11}|(?:shorts|live)/[A-Za-z0-9_-]{11})"
    r"|youtu\.be/[A-Za-z0-9_-]{11}"
    r")"
    r"(?:[?&][A-Za-z0-9=&%_.-]*[A-Za-z0-9_-])?"
)
