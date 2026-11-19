"""Bilibili video URL pattern.

Matches watchable video pages (`/video/BV.../` and `/video/av.../`) and `b23.tv` short links;
excludes live rooms, user spaces, moments, and `/bangumi/` pages. A `b23.tv` link can still
redirect to a space or collection, so callers reject playlist-shaped results whose resolved URL
falls outside `BILIBILI_URL_RE` while accepting `/video/` links that redirect to `/bangumi/`.
"""

import re

from discordbot.utils.urls import URL_START_ANCHOR

# Host-anchored to reject lookalike domains/queries; BV ids require exactly 10 base-62 chars
# (negative lookahead prevents truncated matches); trailing class stops before CJK/sentence marks.
BILIBILI_URL_RE = re.compile(
    pattern=rf"{URL_START_ANCHOR}https?://"
    r"(?:"
    r"(?:www\.|m\.)?bilibili\.com/video/(?:BV[0-9A-Za-z]{10}(?![0-9A-Za-z])|av\d+)/?"
    r"|b23\.tv/[A-Za-z0-9]+"
    r")"
    r"(?:\?[A-Za-z0-9=&%_.-]*[A-Za-z0-9_-])?"
)
