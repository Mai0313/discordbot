"""Tests for the cog that auto-expands Douyin links pasted into a channel."""

from types import SimpleNamespace
from typing import Unpack, NoReturn
import asyncio
from pathlib import Path
import threading

import pytest

from discordbot.cogs.parse_douyin import cog as parse_douyin
from discordbot.cogs.parse_douyin.cog import DouyinCogs
from discordbot.services.platforms.douyin import DouyinMetadata
from discordbot.utils.expansion_placeholder import (
    EXPANSION_DONE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
)

from tests.helpers.casting import as_message
from tests.helpers.link_sources import (
    StubDouyinOptions,
    StubDouyinDownloader,
    stub_bot,
    hosting_planner,
    hosting_off_planner,
    stub_douyin_downloads,
    race_every_scratch_teardown,
)
from tests.helpers.discord_mocks import (
    FakeGuild,
    FakeDiscordMessage,
    expansion_payload,
    placeholder_withdrawn,
)

_URL = "https://v.douyin.com/abc123"


def _cog(**canned: Unpack[StubDouyinOptions]) -> tuple[DouyinCogs, list[StubDouyinDownloader]]:
    """Builds a cog wired to stub downloaders and a hosting-off delivery planner."""
    cog = DouyinCogs(bot=stub_bot())
    cog.media_delivery = hosting_off_planner()
    made: list[StubDouyinDownloader] = []
    cog.__dict__["downloader_factory"] = stub_douyin_downloads(made=made, **canned)
    return cog, made


def _message(content: str = _URL, filesize_limit: int = 25 * 1024 * 1024) -> FakeDiscordMessage:
    """Builds a guild message carrying a Douyin link."""
    return FakeDiscordMessage(content=content, guild=FakeGuild(filesize_limit=filesize_limit))


def _reply_body(*, message: FakeDiscordMessage) -> str:
    """Returns the delivered text, failing loudly when nothing reached the placeholder."""
    content = expansion_payload(message=message)["content"]
    assert content is not None
    return content


async def test_a_pasted_link_is_expanded_with_its_caption() -> None:
    """A plain paste attaches the clip, adds a caption card, and suppresses the raw preview."""
    cog, made = _cog()
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.suppressed
    delivered = expansion_payload(message=message)
    # By name: the caption card's embed spacer rides as a file too, so a bare non-empty check
    # would pass a card that lost its clip.
    assert "1.mp4" in [file.filename for file in delivered["files"]]
    assert delivered["embeds"][0].description == "caption"
    assert delivered["embeds"][0].author.name == "somebody"
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI
    # The scratch dir is per invocation and removed with its files once delivery finishes.
    assert not await asyncio.to_thread(Path(made[0].output_folder).exists)


async def test_a_message_without_a_link_is_ignored() -> None:
    """The listener sees every message, so a non-Douyin one must cost nothing."""
    cog, made = _cog()
    message = _message(content="just chatting")

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions == []
    assert made == []


async def test_an_oversize_clip_is_hosted_as_a_url(tmp_path: Path) -> None:
    """Too big to attach means a hosted link, exactly as `/download_video` behaves."""
    cog, _ = _cog()
    (tmp_path / "serve").mkdir()
    cog.media_delivery = hosting_planner(serve_dir=tmp_path / "serve")
    message = _message(filesize_limit=4)  # tiny ceiling -> the clip counts as oversize

    await cog.on_message(message=as_message(fake=message))

    content = _reply_body(message=message)
    assert any(line.startswith("https://media.test/") for line in content.splitlines())
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


async def test_a_capped_gallery_reports_what_it_left_out() -> None:
    """A gallery trimmed by Discord's attachment cap says so rather than silently dropping."""
    cog, _ = _cog(
        post=DouyinMetadata(aweme_id="1", title="gallery", author_name="a", is_photo=True),
        files=[(f"1_{index}.jpg", b"x" * (index + 1)) for index in range(3)],
        total_images=12,
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert "已省略 9 張圖片" in _reply_body(message=message)
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


async def test_the_parsed_post_is_handed_to_the_download() -> None:
    """The parsed post rides into the download, so the post is never resolved a second time.

    Asserting the download ran once would not catch dropping `post=`; the stub records what it
    was actually given.
    """
    cog, made = _cog()
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    (stub,) = made
    assert len(stub.download_calls) == 1
    assert stub.download_calls[0]["post"] is stub.post


async def test_a_non_post_link_is_left_alone() -> None:
    """A profile or live-room link is not a post, so it earns no reaction, reply, or request.

    The URL regex matches the host rather than the path, so without the post-shape gate the
    cog would answer a pasted profile with a warning reaction and a failure message.
    """
    for content in (
        "https://www.douyin.com/user/MS4wLjABAAAAxyz",
        "https://live.douyin.com/123456",
        "https://www.douyin.com/search/abc",
    ):
        cog, made = _cog()
        message = _message(content=content)

        await cog.on_message(message=as_message(fake=message))

        assert message.reactions == [], content
        assert message.replies == [], content
        assert made == [], content


async def test_a_non_post_link_does_not_hide_a_post_after_it() -> None:
    """A refused link is skipped, so the post after it is still expanded (#854)."""
    live_room = "https://live.douyin.com/123456"
    assert DouyinCogs.URL_PATTERN.search(string=live_room) is not None
    cog, made = _cog()
    message = _message(content=f"{live_room} 跟這篇 {_URL}")

    await cog.on_message(message=as_message(fake=message))

    (stub,) = made
    assert [call["url"] for call in stub.download_calls] == [_URL]
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


def _stall_every_read(*, cog: DouyinCogs, release: threading.Event) -> None:
    """Points the cog at a downloader whose every call holds its worker thread until `release`.

    That is how a stalling CDN read holds a thread `asyncio.to_thread` cannot cancel. The wait is
    bounded so a bound that never fires fails the test on the reaction rather than hanging it.
    """

    def stall(**kwargs: object) -> NoReturn:
        """Blocks until released, then fails the read it never finished."""
        del kwargs
        release.wait(timeout=5.0)
        raise AssertionError("should have been abandoned")

    cog.__dict__["downloader_factory"] = lambda output_folder: SimpleNamespace(
        parse_metadata=stall, download=stall
    )


async def test_a_stalled_expansion_gives_up_and_frees_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalling download must not hold the shared Douyin slot indefinitely.

    The slot is shared with the reply path, so an unbounded gallery would stall every AI reply
    about a Douyin link behind it. A timeout is reported as a retryable failure, never as a
    missing post.
    """
    monkeypatch.setattr(parse_douyin, "DOUYIN_EXPAND_TIMEOUT_SECONDS", 0.05)
    cog, _ = _cog()

    release = threading.Event()
    _stall_every_read(cog=cog, release=release)
    message = _message()

    await cog.on_message(message=as_message(fake=message))
    release.set()

    assert message.reactions[-1] == EXPANSION_RETRY_LATER_EMOJI
    assert placeholder_withdrawn(message=message)


async def test_a_raced_scratch_teardown_keeps_the_failure_the_expansion_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleanup losing a race with its own abandoned worker cannot relabel the failure.

    The timeout leaves the download thread running (`asyncio.to_thread` cannot cancel it), so the
    scratch removal walks a directory something is still writing into and can raise. Raised, that
    lands in `on_message`'s outer handler, which paints the generic ❌ over the ⏱️ the timeout
    just explained and logs a defect that never happened.
    """
    monkeypatch.setattr(parse_douyin, "DOUYIN_EXPAND_TIMEOUT_SECONDS", 0.05)
    removed = race_every_scratch_teardown(monkeypatch)
    cog, _ = _cog()

    release = threading.Event()
    _stall_every_read(cog=cog, release=release)
    message = _message()

    await cog.on_message(message=as_message(fake=message))
    release.set()

    assert removed  # the teardown really ran and really failed
    assert message.reactions[-1] == EXPANSION_RETRY_LATER_EMOJI
    assert placeholder_withdrawn(message=message)
