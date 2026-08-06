"""Pins `BILIBILI_URL_RE`, the shape test deciding whether a message names a Bilibili video.

The regex is the whole admission gate. The Bilibili link source registers no `url_filter` (unlike
Douyin, whose regex matches the host and needs a second check for the path), so what this matches
is what `gen_reply` may hand to yt-dlp. It is also how the builder recognises a `b23.tv` link that
resolved somewhere else: `link_sources/bilibili.py` re-matches it against the canonical URL of a
playlist-shaped result and injects its neutral notice when it fails. Widening the regex therefore
costs twice — a non-video surface would be admitted here AND slip past that guard, putting some
other video's metadata in front of the model under the user's link.

Three things are pinned. What must match: `/video/` pages under every host form Bilibili answers
on (`www.`, `m.`, bare, either scheme), BV and av ids alike, with or without a trailing slash or a
query tail, plus the `b23.tv` links the mobile share button copies. Where a match ends: chat
writes a URL flush against Chinese punctuation, and a tail greedy enough to swallow `。` would
send yt-dlp a URL nobody linked. What must never match: Bilibili's non-video surfaces, a host that
merely contains `bilibili.com` (the downloader attaches a Bilibili `Referer` only to a real one,
and this regex is the half that decides such a URL is worth reading at all), and an id of the
wrong length.

Nothing here fetches anything, which mirrors the regex itself: a match is a claim about shape,
never about a video existing, being public or being region-available. What the builder does once
it holds the URL is `tests/test_parse_bilibili.py`.
"""

import pytest

from discordbot.utils.bilibili import BILIBILI_URL_RE


@pytest.mark.parametrize(
    argnames="url",
    argvalues=[
        "https://www.bilibili.com/video/BV1jpK86hEc8",
        "https://bilibili.com/video/BV1jpK86hEc8",
        "https://m.bilibili.com/video/BV1jpK86hEc8",
        "http://www.bilibili.com/video/BV1jpK86hEc8",
        "https://www.bilibili.com/video/BV1jpK86hEc8/",
        "https://www.bilibili.com/video/BV1jpK86hEc8?p=2&t=30",
        "https://www.bilibili.com/video/BV1jpK86hEc8/?spm_id_from=333.1007",
        "https://www.bilibili.com/video/av170001",
        "https://b23.tv/abc123X",
    ],
)
def test_bilibili_url_re_matches_watchable_forms(url: str) -> None:
    """Video pages (BV and av ids, any official host form) and share short links match."""
    match = BILIBILI_URL_RE.search(string=url)
    assert match is not None
    assert match.group(0) == url


def test_bilibili_url_re_stops_before_sentence_punctuation() -> None:
    """A link written mid-CJK-sentence sheds the trailing punctuation, not the id."""
    text = "看看這個 https://www.bilibili.com/video/BV1jpK86hEc8。很好笑"
    match = BILIBILI_URL_RE.search(string=text)
    assert match is not None
    assert match.group(0) == "https://www.bilibili.com/video/BV1jpK86hEc8"

    tail = "https://b23.tv/abc123X，對吧"
    short_match = BILIBILI_URL_RE.search(string=tail)
    assert short_match is not None
    assert short_match.group(0) == "https://b23.tv/abc123X"


@pytest.mark.parametrize(
    argnames="url",
    argvalues=[
        # Non-video Bilibili surfaces: a live room, a profile, a moment, a bangumi episode and
        # an article are none of them one ordinary video page the builder can read.
        "https://live.bilibili.com/12345",
        "https://space.bilibili.com/672328094",
        "https://t.bilibili.com/1043462527",
        "https://www.bilibili.com/bangumi/play/ep1234",
        "https://www.bilibili.com/opus/1043462527",
        # Host lookalikes must never match (mirrors the downloader's Referer guard).
        "https://bilibili.com.attacker.com/video/BV1jpK86hEc8",
        "https://evil.com/?x=bilibili.com/video/BV1jpK86hEc8",
        # Malformed ids: a BV id is exactly BV + 10 base-62 characters.
        "https://www.bilibili.com/video/BV1jpK86hEc",
        "https://www.bilibili.com/video/BV1jpK86hEc8X",
        "https://www.bilibili.com/video/xyz",
    ],
)
def test_bilibili_url_re_rejects_non_video_and_lookalike_urls(url: str) -> None:
    """Non-video surfaces, lookalike hosts and malformed ids never match."""
    assert BILIBILI_URL_RE.search(string=url) is None
