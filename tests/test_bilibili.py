"""Tests for the Bilibili video URL regex."""

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
        # Non-video Bilibili surfaces: no single anonymous-fetchable video behind them.
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
    """Live rooms, spaces, bangumi, lookalike hosts and malformed ids never match."""
    assert BILIBILI_URL_RE.search(string=url) is None
