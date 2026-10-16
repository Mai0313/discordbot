"""Tests for the YouTube URL pattern."""

import pytest

from discordbot.services.platforms.youtube import YOUTUBE_URL_RE


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        ),
        ("https://youtube.com/watch?v=jNQXAC9IVRw", "https://youtube.com/watch?v=jNQXAC9IVRw"),
        ("https://youtu.be/jNQXAC9IVRw", "https://youtu.be/jNQXAC9IVRw"),
        (
            "https://www.youtube.com/shorts/abcdefghijk",
            "https://www.youtube.com/shorts/abcdefghijk",
        ),
        ("https://www.youtube.com/live/abcdefghijk", "https://www.youtube.com/live/abcdefghijk"),
        (
            "https://m.youtube.com/watch?v=jNQXAC9IVRw&t=30s",
            "https://m.youtube.com/watch?v=jNQXAC9IVRw&t=30s",
        ),
        (
            "https://www.youtube.com/watch?app=desktop&v=jNQXAC9IVRw",
            "https://www.youtube.com/watch?app=desktop&v=jNQXAC9IVRw",
        ),
        ("看這個 https://youtu.be/jNQXAC9IVRw。很讚", "https://youtu.be/jNQXAC9IVRw"),
        ("watch https://youtu.be/jNQXAC9IVRw, then react", "https://youtu.be/jNQXAC9IVRw"),
    ],
)
def test_youtube_url_re_matches_watchable_links(text: str, expected: str) -> None:
    """The shared regex extracts a watchable YouTube URL, trailing punctuation excluded."""
    match = YOUTUBE_URL_RE.search(string=text)
    assert match is not None
    assert match.group(0) == expected


@pytest.mark.parametrize(
    "text",
    [
        "no url here at all",
        "https://www.youtube.com/playlist?list=PL123",
        "https://www.youtube.com/@channelname",
        "https://example.com/watch?v=jNQXAC9IVRw",
        "https://vimeo.com/123456789",
    ],
)
def test_youtube_url_re_rejects_non_videos(text: str) -> None:
    """Channel / playlist / non-YouTube URLs and plain text are not matched."""
    assert YOUTUBE_URL_RE.search(string=text) is None
