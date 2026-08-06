"""Bilibili video URL detection, the one definition every Bilibili-aware path matches against.

This module owns a single compiled regex and promises one thing: find, in arbitrary message text,
a URL that names ONE watchable Bilibili video. `gen_reply` is what needs that. A linked Bilibili
video reaches the answer model only by being downloaded with yt-dlp and uploaded to the Gemini
Files API, since Gemini cannot fetch a Bilibili page server-side the way it fetches a YouTube
link, so the match is the first gate in front of a genuinely expensive build. Only the first: the
router still has to name `bilibili`, and the download plus the upload additionally need
`bilibili_video_enabled`, a direct Gemini key and a Gemini answer model, so a selected match with
any of those missing buys only a cheap metadata probe and the text it injects.

That path reads it twice. `cog.py`'s `LINK_CONTEXT_SOURCES` entry uses it as the `url_pattern`
picking WHICH URL a router-selected Bilibili build reads (the router names sources, never URLs),
and `link_sources/bilibili.py` re-matches it against the canonical URL of a playlist-shaped
yt-dlp result, since a `b23.tv` short link is opaque until something follows it. Only that shape
is re-matched: a single-video result is the linked video whatever URL it resolved to.

It deliberately resolves nothing and fetches nothing. A match says the URL is SHAPED like a
watchable page, never that the video exists, is public, or is region-available; the builder learns
that from yt-dlp and answers with its one neutral notice either way.

Detection sits in `utils/` rather than inside the one cog reading it today, beside
`YOUTUBE_URL_RE`, `DOUYIN_URL_RE` and `THREADS_URL_RE`: a link source's URL pattern is the half a
second reader needs first, and the expansion cogs that read the Threads and Douyin ones may not
import a peer cog to reach them.
"""

import re

# Matches the watchable forms only: `/video/BV.../` and `/video/av.../` pages plus `b23.tv`
# share short links. Live rooms (`live.bilibili.com`), user spaces (`space.bilibili.com`),
# moments (`t.bilibili.com`) and `/bangumi/` pages never match, so unlike `DOUYIN_URL_RE` no
# separate post-URL guard is needed on top. A `b23.tv` short link CAN still resolve to one of
# those — and yt-dlp reads a space or collection SUCCESSFULLY as a playlist rather than
# failing — so the context builder rejects a playlist-shaped result whose resolved canonical
# URL falls outside this regex with its neutral notice (playlist-shaped only: a single
# /video/ link Bilibili redirects to /bangumi/ server-side is still the linked video).
# Bilibili has no Douyin-grade WAF economics, so the one wasted probe is acceptable.
# The host is anchored right after the scheme, so `bilibili.com.attacker.com/video/...` and
# `evil.com/?x=bilibili.com/video/...` never match. A BV id is exactly `BV` plus 10 base-62
# characters (the lookahead stops a longer token from matching truncated), an av id is digits,
# and the optional query tail must end on `[A-Za-z0-9_-]` so a link written mid-sentence stops
# cleanly at a non-ASCII terminator (e.g. zh/ja `...hEc8。`) instead of swallowing punctuation.
BILIBILI_URL_RE = re.compile(
    r"https?://"
    r"(?:"
    r"(?:www\.|m\.)?bilibili\.com/video/(?:BV[0-9A-Za-z]{10}(?![0-9A-Za-z])|av\d+)/?"
    r"|b23\.tv/[A-Za-z0-9]+"
    r")"
    r"(?:\?[A-Za-z0-9=&%_.-]*[A-Za-z0-9_-])?"
)
