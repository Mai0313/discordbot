"""YouTube video URL detection, the one definition the reply pipeline scans a message with.

This module owns a single compiled regex and promises one thing: find, in arbitrary message text,
a URL that names ONE watchable YouTube video. `gen_reply` is what needs it, for a reason none of
the sibling link patterns share. A YouTube link is the one linked medium the answer model fetches
itself: `gen_reply/interactions.py` hands the matched URL to the native Gemini Interactions API as
a video part and Gemini watches it server-side, so unlike Threads / Douyin / Bilibili nothing here
or downstream downloads or uploads a byte. That is also what the match is for: it swaps that one
answer turn off the LiteLLM Responses bridge, which HTTP-fetches the URL and leaves Gemini reading
the page's HTML instead of watching the video.

`cog.py` reads it through `_youtube_url_in_message` / `_find_youtube_url`, which scan the current
message and then outward along the reply-reference chain, and only once the router has set
`RouteClassification.watch_video`. The URL therefore comes out of text already posted, never out
of THIS turn's model output: the model cannot name a video by emitting a URL in the answer it is
currently producing. That is not a boundary against model-authored URLs in general — the chain
resolves any referenced message, this bot's own earlier replies included, and only the usage
footer is stripped from those, so a link the model itself wrote can come back when a user replies
to that answer. A match is not by itself the gate either: with `youtube_video_enabled` off, a
non-Gemini answer model, or no direct Gemini key, the turn stays on the Responses path and the
video simply goes unwatched.

It deliberately resolves nothing and fetches nothing. A match says the URL is SHAPED like a
watchable video page, never that the video exists, is public, or is playable from where Gemini
fetches it; the answer turn finds that out itself and degrades on its own.

Detection sits in `utils/` beside `THREADS_URL_RE`, `DOUYIN_URL_RE` and `BILIBILI_URL_RE` rather
than inside `gen_reply`, its only reader today: the link-pattern vocabulary stays in one place, so
this one is looked for where the others already are. Unlike Threads and Douyin, YouTube has no
expansion cog, so nothing yet forces the pattern out of the cog — the siblings' own reason (a cog
may not import a peer cog to reach one) is what would apply the day a second reader appears.
"""

import re

# Matches the watchable forms only (watch?v=, youtu.be/, /shorts/, /live/), never channel /
# playlist / user pages, since those carry no single video to watch. The host is anchored right
# after the scheme, so `evil.com/?x=youtube.com/watch?v=...` never matches; a suffix domain like
# `youtube.com.attacker.com/watch?v=...` is rejected instead by the `/` the branch requires
# immediately after `youtube.com`, since there the host does start right after the scheme. The
# video id is the [A-Za-z0-9_-] YouTube id alphabet, and that class is what makes a link written
# mid-sentence stop cleanly at any terminator outside it, non-ASCII (zh/ja `...VIDEOID。`) or
# ASCII (a trailing `.`); the `{11}` bounds the id, so a longer run of id-alphabet characters is
# not swallowed whole. The `watch?` branch carries its own ASCII query segment ahead of `v=`, so
# `?app=desktop&v=...` matches with no tail involved. The separate optional ASCII query tail
# (`&t=30s`, `?si=`) must END on `[A-Za-z0-9_-]`, which is what keeps trailing punctuation out of
# the URL once such a tail is present (`...?t=30.`).
YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.|m\.)?"
    r"(?:"
    r"youtube\.com/(?:watch\?[A-Za-z0-9=&%_.-]*v=[A-Za-z0-9_-]{11}|(?:shorts|live)/[A-Za-z0-9_-]{11})"
    r"|youtu\.be/[A-Za-z0-9_-]{11}"
    r")"
    r"(?:[?&][A-Za-z0-9=&%_.-]*[A-Za-z0-9_-])?"
)
