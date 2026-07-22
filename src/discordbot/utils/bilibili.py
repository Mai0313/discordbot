"""Bilibili video URL detection shared by the gen_reply linked-video answer path."""

import re

# Single source of truth for detecting a Bilibili video URL in a message. gen_reply uses it to
# decide whether to read the linked video into answer context (yt-dlp download, then a Gemini
# Files API upload — unlike YouTube, Gemini cannot fetch a Bilibili page server-side).
# Matches the watchable forms only: `/video/BV.../` and `/video/av.../` pages plus `b23.tv`
# share short links. Live rooms (`live.bilibili.com`), user spaces (`space.bilibili.com`),
# moments (`t.bilibili.com`) and `/bangumi/` pages never match, so unlike `DOUYIN_URL_RE` no
# separate post-URL guard is needed on top. A `b23.tv` short link CAN still resolve to one of
# those — and yt-dlp reads a space or collection SUCCESSFULLY as a playlist rather than
# failing — so the context builder checks the resolved canonical URL against this regex and
# rejects a non-video page with its neutral notice. Bilibili has no Douyin-grade WAF
# economics, so the one wasted probe is acceptable.
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
