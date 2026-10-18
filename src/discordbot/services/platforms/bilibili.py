"""Bilibili video URL pattern.

Matches watchable video pages (`/video/BV.../` and `/video/av.../`) and `b23.tv` short links.
Live rooms (`live.bilibili.com`), user spaces (`space.bilibili.com`), moments (`t.bilibili.com`),
and `/bangumi/` pages do not match, so no separate post-URL filter is needed. Because a `b23.tv`
short link can still redirect to a space or collection (which yt-dlp extracts as a playlist),
callers reject playlist-shaped results whose resolved canonical URL falls outside `BILIBILI_URL_RE`
while still accepting single `/video/` links that redirect server-side to `/bangumi/`.
"""

import re

from discordbot.utils.urls import URL_START_ANCHOR

# Host is anchored right after the scheme so lookalike domains (`bilibili.com.attacker.com`) and
# query strings (`evil.com/?x=bilibili.com/video/...`) cannot match. A BV id is `BV` plus 10
# base-62 characters (with a negative lookahead so longer tokens do not match truncated), an av id
# is digits, and the optional query tail ends on `[A-Za-z0-9_-]` so a link written mid-sentence
# stops before CJK or ASCII sentence punctuation.
BILIBILI_URL_RE = re.compile(
    pattern=rf"{URL_START_ANCHOR}https?://"
    r"(?:"
    r"(?:www\.|m\.)?bilibili\.com/video/(?:BV[0-9A-Za-z]{10}(?![0-9A-Za-z])|av\d+)/?"
    r"|b23\.tv/[A-Za-z0-9]+"
    r")"
    r"(?:\?[A-Za-z0-9=&%_.-]*[A-Za-z0-9_-])?"
)
