"""YouTube URL detection shared by the gen_reply YouTube video-answer path."""

import re

# Single source of truth for detecting a YouTube video URL in a message. gen_reply uses it
# to decide whether the QA answer turn should ingest the linked video through the Gemini
# Interactions API (which can fetch a YouTube URL server-side, unlike the Responses bridge).
# Matches the watchable forms only (watch?v=, youtu.be/, /shorts/, /live/), never channel /
# playlist / user pages, since those carry no single video to watch. The 11-character video id
# is the [A-Za-z0-9_-] YouTube id alphabet; an optional ASCII query tail is allowed and must end
# on `[A-Za-z0-9_-]` so a link written mid-sentence stops cleanly at a non-ASCII terminator
# (e.g. zh/ja `...VIDEOID。`) instead of swallowing punctuation into the URL.
YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.|m\.)?"
    r"(?:"
    r"youtube\.com/(?:watch\?[A-Za-z0-9=&%_.-]*v=[A-Za-z0-9_-]{11}|(?:shorts|live)/[A-Za-z0-9_-]{11})"
    r"|youtu\.be/[A-Za-z0-9_-]{11}"
    r")"
    r"(?:[?&][A-Za-z0-9=&%_.-]*[A-Za-z0-9_-])?"
)
