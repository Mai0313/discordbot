"""Tests for the cog that auto-expands Facebook links pasted into a channel.

The downloader is stubbed at `downloader_factory` and `video_downloader_factory`, so nothing here
touches the network or the parser: what these cover is the cog's own decisions — when it fires,
what it renders, how target videos attach or degrade, and what a reader sees when the post cannot
be read.
"""

import re
from typing import Any
from pathlib import Path
import threading
from collections.abc import Callable

import pytest
from yt_dlp.utils import DownloadError

from discordbot.utils.expansion_cog import VIDEO_HINT, omitted_videos_note
from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME
from discordbot.utils.file_downloads import DownloadTooLargeError
from discordbot.cogs.parse_facebook.cog import FacebookCogs
from discordbot.services.platforms.ytdlp import DownloadResult, DownloadStoppedError
from discordbot.services.platforms.facebook import FacebookOutput
from discordbot.utils.expansion_placeholder import EXPANSION_DONE_EMOJI

from tests.helpers.casting import as_message
from tests.helpers.link_sources import (
    FACEBOOK_URL,
    expand,
    facebook_post,
    guild_message,
    expansion_embeds,
    stub_conversation_cog,
)
from tests.helpers.discord_mocks import FakeDiscordMessage, expansion_payload
from tests.helpers.logfire_capture import capture_logs, capture_levels


def _media_uploads(message: FakeDiscordMessage) -> dict[str, bytes]:
    """Non-spacer attached files delivered on `message`, keyed by filename."""
    payload = expansion_payload(message=message)
    return {
        str(file.filename): file.fp.read()
        for file in payload.get("files", [])
        if str(file.filename) != DEFAULT_EMBED_SPACER_FILENAME
    }


def _stub_video_downloads(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: dict[str, bytes | Exception] | None = None,
    on_download: Callable[[str, threading.Event | None], DownloadResult] | None = None,
) -> list[dict[str, Any]]:
    """Installs a fake `VideoDownloader` factory on `FacebookCogs`, recording every call."""
    calls: list[dict[str, Any]] = []
    outcomes_map = outcomes or {}

    class _FakeVideoDownloader:
        def __init__(self, output_folder: str) -> None:
            self.output_folder = output_folder

        def download(
            self,
            url: str,
            quality: str = "best",
            stop_signal: threading.Event | None = None,
            max_items: int = 1,
            max_bytes: int | None = None,
        ) -> DownloadResult:
            calls.append({
                "url": url,
                "quality": quality,
                "stop_signal": stop_signal,
                "max_items": max_items,
                "max_bytes": max_bytes,
                "output_folder": Path(self.output_folder),
            })
            if on_download is not None:
                return on_download(url, stop_signal)
            outcome = outcomes_map.get(url, url.encode())
            if isinstance(outcome, Exception):
                raise outcome
            # Deliberately use a yt-dlp-style id filename so tests verify renaming to `video_{i}.mp4`.
            out_file = Path(self.output_folder) / f"fb_{len(calls)}.mp4"
            out_file.write_bytes(outcome)
            return DownloadResult(filenames=[out_file])

    monkeypatch.setattr(
        target=FacebookCogs, name="video_downloader_factory", value=_FakeVideoDownloader
    )
    return calls


async def test_a_pasted_link_is_expanded_into_a_card() -> None:
    """The ordinary case: one post embed carrying the body, the author and the counters."""
    message, stub = await expand(cog_type=FacebookCogs, outcome=facebook_post())

    assert stub.seen == [FACEBOOK_URL]
    assert message.suppressed
    embeds = expansion_embeds(message=message)
    assert embeds[0].description == "post body"
    assert embeds[0].author.name == "Somebody"
    assert embeds[0].url == FACEBOOK_URL
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


async def test_the_footer_names_the_group_and_the_counters() -> None:
    """The group is the part a reader cannot get from the post itself, so it leads."""
    message, _ = await expand(cog_type=FacebookCogs, outcome=facebook_post())

    footer = expansion_embeds(message=message)[0].footer.text
    assert footer is not None
    assert footer.startswith("Some Group")
    assert "1,017" in footer
    assert "40" in footer
    assert "37" in footer


async def test_a_page_post_leaves_the_group_out_of_the_footer() -> None:
    """A page post has no group, and the line must not carry a hole where one would go."""
    message, _ = await expand(cog_type=FacebookCogs, outcome=facebook_post(group_name=""))

    footer = expansion_embeds(message=message)[0].footer.text
    assert footer is not None
    assert footer == footer.lstrip(" ·")
    assert "1,017" in footer


@pytest.mark.parametrize(
    argnames=("post_url", "expected"),
    argvalues=[
        (FACEBOOK_URL, f"{FACEBOOK_URL}?comment_id=222"),
        # A `permalink.php` post URL already carries a query, so a second `?` breaks the link.
        (
            "https://www.facebook.com/permalink.php?story_fbid=1&id=2",
            "https://www.facebook.com/permalink.php?story_fbid=1&id=2&comment_id=222",
        ),
    ],
    ids=["plain", "permalink-php"],
)
async def test_the_comment_link_names_the_comment_on_the_post_url(
    post_url: str, expected: str
) -> None:
    """The comment card links to the post's URL naming the comment, whatever query it carries.

    Its author line is the commenter's name as parsed, with nothing added.
    """
    comment = FacebookOutput(comment_id="222", text="linked", author_name="C")
    message, _ = await expand(
        cog_type=FacebookCogs,
        outcome=facebook_post(url=post_url, comments=[comment], selected_comment_id="222"),
    )

    comment_embed = expansion_embeds(message=message)[-1]
    assert comment_embed.author.name == "C"
    assert comment_embed.url == expected


async def test_an_album_counts_the_videos_nothing_linked() -> None:
    """Only the first video gets a hint, so the rest would go unmentioned."""
    message, _ = await expand(
        cog_type=FacebookCogs,
        outcome=facebook_post(
            video_urls=[f"https://www.facebook.com/watch/?v={n}" for n in range(3)]
        ),
    )

    footer = expansion_embeds(message=message)[0].footer.text
    assert footer is not None
    assert "2" in re.findall(r"\d[\d,]*", footer)


async def test_a_video_post_downloads_attaches_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target video downloads via yt-dlp into a scratch clip subdir, attaches as `video_0.mp4`, and cleans up."""
    clip = "https://www.facebook.com/attn/videos/123456/"
    calls = _stub_video_downloads(monkeypatch=monkeypatch)

    message, _ = await expand(
        cog_type=FacebookCogs, outcome=facebook_post(image_urls=[], video_urls=[clip])
    )

    uploads = _media_uploads(message=message)
    assert list(uploads.keys()) == ["video_0.mp4"]
    assert uploads["video_0.mp4"] == clip.encode()
    assert VIDEO_HINT.partition("{url}")[0] not in (
        expansion_embeds(message=message)[0].description or ""
    )
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == clip
    assert call["quality"] == "best"
    assert call["max_items"] == 1
    assert isinstance(call["max_bytes"], int)
    assert call["max_bytes"] > 0
    clip_dir = call["output_folder"]
    assert clip_dir.name == "clip_0"
    assert not clip_dir.exists()
    assert not clip_dir.parent.exists()


async def test_an_oversize_video_falls_back_to_link_and_logs_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`DownloadTooLargeError` degrades to `VIDEO_HINT` at `info` with no traceback."""
    clip = "https://www.facebook.com/attn/videos/999/"
    _stub_video_downloads(
        monkeypatch=monkeypatch, outcomes={clip: DownloadTooLargeError("exceeds cap")}
    )
    records = capture_levels(monkeypatch=monkeypatch, levels=("info", "warn", "error"))

    message, _ = await expand(
        cog_type=FacebookCogs, outcome=facebook_post(image_urls=[], video_urls=[clip])
    )

    assert _media_uploads(message=message) == {}
    assert VIDEO_HINT.format(url=clip) in (expansion_embeds(message=message)[0].description or "")
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI
    assert any(
        level == "info" and "byte cap" in msg and "_exc_info" not in fields
        for level, msg, fields in records
    )
    assert not any(level in {"warn", "error"} for level, _, _ in records)


async def test_a_ytdlp_download_error_falls_back_to_link_without_refusing_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A yt-dlp `DownloadError` logs a warning with traceback and keeps the card with `VIDEO_HINT`."""
    clip = "https://www.facebook.com/attn/videos/404/"
    _stub_video_downloads(
        monkeypatch=monkeypatch, outcomes={clip: DownloadError("video unavailable")}
    )
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")

    message, _ = await expand(
        cog_type=FacebookCogs, outcome=facebook_post(image_urls=[], video_urls=[clip])
    )

    assert _media_uploads(message=message) == {}
    assert VIDEO_HINT.format(url=clip) in (expansion_embeds(message=message)[0].description or "")
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI
    assert any(
        fields.get("error_type") == "DownloadError" and "_exc_info" in fields
        for _, fields in warns
    )


async def test_partial_multi_video_failure_links_exact_undelivered_clip_when_first_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When clip 0 fails and clip 1 attaches as `video_1.mp4`, `VIDEO_HINT` targets clip 0."""
    clips = [
        "https://www.facebook.com/attn/videos/10/",
        "https://www.facebook.com/attn/videos/20/",
    ]
    _stub_video_downloads(
        monkeypatch=monkeypatch, outcomes={clips[0]: DownloadError("clip 0 failed")}
    )

    message, _ = await expand(
        cog_type=FacebookCogs, outcome=facebook_post(image_urls=[], video_urls=clips)
    )

    uploads = _media_uploads(message=message)
    embed = expansion_embeds(message=message)[0]
    description = embed.description or ""
    assert list(uploads.keys()) == ["video_1.mp4"]
    assert uploads["video_1.mp4"] == clips[1].encode()
    assert VIDEO_HINT.format(url=clips[0]) in description
    assert clips[1] not in description
    assert "部影片" not in (embed.footer.text or "")


async def test_partial_multi_video_failure_links_first_undelivered_and_counts_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When clip 0 attaches and clips 1 and 2 fail, `VIDEO_HINT` targets clip 1 and footer counts 1 more."""
    clips = [
        "https://www.facebook.com/attn/videos/100/",
        "https://www.facebook.com/attn/videos/200/",
        "https://www.facebook.com/attn/videos/300/",
    ]
    _stub_video_downloads(
        monkeypatch=monkeypatch,
        outcomes={
            clips[1]: DownloadTooLargeError("too big"),
            clips[2]: DownloadError("broken stream"),
        },
    )

    message, _ = await expand(
        cog_type=FacebookCogs, outcome=facebook_post(image_urls=[], video_urls=clips)
    )

    uploads = _media_uploads(message=message)
    embed = expansion_embeds(message=message)[0]
    description = embed.description or ""
    assert list(uploads.keys()) == ["video_0.mp4"]
    assert uploads["video_0.mp4"] == clips[0].encode()
    assert VIDEO_HINT.format(url=clips[1]) in description
    assert clips[0] not in description
    assert omitted_videos_note(count=1) in (embed.footer.text or "")


async def test_video_download_timeout_propagates_cancellation_and_signals_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `VIDEO_DOWNLOAD_TIMEOUT_SECONDS` fires, cancellation sets `stop_signal` and degrades to link."""
    clip = "https://www.facebook.com/attn/videos/777/"
    stopped = threading.Event()

    def slow_download(url: str, stop_signal: threading.Event | None) -> DownloadResult:
        del url
        assert stop_signal is not None
        if stop_signal.wait(timeout=5.0):
            stopped.set()
            raise DownloadStoppedError("stopped by caller")
        raise AssertionError("stop_signal was never set")

    _stub_video_downloads(monkeypatch=monkeypatch, on_download=slow_download)
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs, outcome=facebook_post(image_urls=[], video_urls=[clip])
    )
    cog.__dict__["VIDEO_DOWNLOAD_TIMEOUT_SECONDS"] = 0.05
    message = guild_message(content=FACEBOOK_URL)

    await cog.on_message(message=as_message(fake=message))

    assert stopped.is_set()
    assert _media_uploads(message=message) == {}
    assert VIDEO_HINT.format(url=clip) in (expansion_embeds(message=message)[0].description or "")
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI
    assert any("timed out" in msg for msg, _ in warns)


async def test_direct_timeout_error_propagates_out_of_video_download_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `TimeoutError` from the downloader is never swallowed as a per-clip error."""
    clips = ["https://www.facebook.com/attn/videos/1/", "https://www.facebook.com/attn/videos/2/"]
    calls = _stub_video_downloads(
        monkeypatch=monkeypatch, outcomes={clips[0]: TimeoutError("socket timed out")}
    )
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")

    message, _ = await expand(
        cog_type=FacebookCogs, outcome=facebook_post(image_urls=[], video_urls=clips)
    )

    assert {call["url"] for call in calls} == {clips[0]}
    assert _media_uploads(message=message) == {}
    assert VIDEO_HINT.format(url=clips[0]) in (
        expansion_embeds(message=message)[0].description or ""
    )
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI
    assert any("timed out" in msg for msg, _ in warns)
