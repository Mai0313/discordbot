"""Tests for the cog that auto-expands Douyin links pasted into a channel."""

import time
from types import SimpleNamespace
from typing import Unpack, TypedDict
import asyncio
from pathlib import Path
import tempfile

import pytest
from nextcord import Message

from discordbot.utils import scratch_dir
from discordbot.typings.emojis import DOUYIN_EMOJI
from discordbot.cogs.parse_douyin import cog as parse_douyin
from discordbot.utils.media_delivery import MediaHostingService, MediaDeliveryPlanner
from discordbot.cogs.parse_douyin.cog import DouyinCogs
from discordbot.services.platforms.douyin import (
    DouyinError,
    DouyinDownload,
    DouyinMetadata,
    DouyinBlockedError,
    DouyinUnavailableError,
)
from discordbot.utils.expansion_placeholder import EXPANSION_RETRY_LATER_EMOJI

from tests.helpers.casting import as_bot, as_message, make_forbidden, make_media_hosting_config
from tests.helpers.discord_mocks import (
    FakeUser,
    FakeDiscordMessage,
    expansion_payload,
    placeholder_withdrawn,
)

_URL = "https://v.douyin.com/abc123"
_GREEN = "<:greencheck:1517565102424068226>"
_RED = "<:redcross:1517565100838355016>"


class _StubDownloader:
    """Stands in for DouyinDownloader, serving canned metadata and files."""

    def __init__(  # noqa: PLR0913 -- one canned outcome per stage the cog can hit
        self,
        output_folder: str,
        post: DouyinMetadata | None = None,
        files: list[tuple[str, bytes]] | None = None,
        parse_error: Exception | None = None,
        download_error: Exception | None = None,
        total_images: int = 0,
    ) -> None:
        """Records the scratch dir and the canned outcome for each stage."""
        self.output_folder = output_folder
        self.post = post or DouyinMetadata(aweme_id="1", title="caption", author_name="somebody")
        self.files = files if files is not None else [("1.mp4", b"video-bytes")]
        self.parse_error = parse_error
        self.download_error = download_error
        self.total_images = total_images
        self.download_calls = 0
        self.received_post: DouyinMetadata | None = None

    def parse_metadata(self, url: str) -> DouyinMetadata:
        """Returns the canned post, or raises the canned parse failure."""
        del url
        if self.parse_error is not None:
            raise self.parse_error
        return self.post

    def download(
        self,
        url: str,
        quality: str = "best",
        max_images: int | None = None,
        max_bytes: int | None = None,
        post: DouyinMetadata | None = None,
    ) -> DouyinDownload:
        """Writes the canned files into the scratch dir, or raises the canned failure."""
        del url, quality, max_images, max_bytes
        self.download_calls += 1
        self.received_post = post
        if self.download_error is not None:
            raise self.download_error
        written: list[Path] = []
        for name, payload in self.files:
            path = Path(self.output_folder) / name
            path.write_bytes(payload)
            written.append(path)
        source = post or self.post
        return DouyinDownload(
            title=source.title,
            is_photo=source.is_photo,
            filenames=written,
            total_images=self.total_images,
        )


class _StubOptions(TypedDict, total=False):
    """Canned per-stage outcomes a test forwards through `_cog` to the stub downloader."""

    post: DouyinMetadata | None
    files: list[tuple[str, bytes]] | None
    parse_error: Exception | None
    download_error: Exception | None
    total_images: int


def _cog(
    bot_id: int = 999, **downloader_kwargs: Unpack[_StubOptions]
) -> tuple[DouyinCogs, dict[str, _StubDownloader]]:
    """Builds a cog wired to a stub downloader and a hosting-off delivery planner."""
    cog = DouyinCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=bot_id))))
    # Explicitly disabled planner — never the no-arg default, whose config is `available` on a
    # dev box where .env enables hosting (it would write into the live serve dir).
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(config=make_media_hosting_config(enabled=False))
    )
    made: dict[str, _StubDownloader] = {}

    def factory(output_folder: str) -> _StubDownloader:
        """Records the stub so a test can assert on what it was asked to do."""
        stub = _StubDownloader(output_folder=output_folder, **downloader_kwargs)
        made["stub"] = stub
        return stub

    cog.__dict__["downloader_factory"] = factory
    return cog, made


class _DouyinMessage(FakeDiscordMessage):
    """Adds the author/content/guild fields `DouyinCogs.on_message` reads."""

    def __init__(self, author: FakeUser, content: str, guild: object) -> None:
        """Builds a message double carrying the fields the cog inspects."""
        super().__init__()
        self.author = author
        self.content = content
        self.guild = guild


def _message(content: str = _URL, filesize_limit: int = 25 * 1024 * 1024) -> _DouyinMessage:
    """Builds a guild message carrying a Douyin link."""
    return _DouyinMessage(
        author=FakeUser(bot=False),
        content=content,
        guild=SimpleNamespace(filesize_limit=filesize_limit),
    )


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
    assert delivered["files"]
    assert delivered["embeds"][0].description == "caption"
    assert delivered["embeds"][0].author.name == "somebody"
    assert message.reactions[-1] == _GREEN
    # The read marker rides beside the status chain, which only ever removes its own reaction.
    assert message.reactions[0] == DOUYIN_EMOJI
    assert all(emoji != DOUYIN_EMOJI for emoji, _ in message.removed)
    # The scratch dir is per invocation and removed with its files once delivery finishes.
    assert not await asyncio.to_thread(Path(made["stub"].output_folder).exists)


async def test_the_placeholder_is_posted_before_the_post_is_read() -> None:
    """The whole point of the placeholder: the reply slot is claimed while the read is ahead.

    Claiming it afterwards would leave the card where it was, several messages below the link
    someone pasted, so the order is what this pins rather than the message itself.
    """
    cog, _ = _cog()
    message = _message()
    replies_when_the_read_began: list[int] = []
    build = cog.__dict__["downloader_factory"]

    def watched_factory(output_folder: str) -> _StubDownloader:
        """Wraps the stub's read so the test can see the channel as it starts."""
        stub = build(output_folder=output_folder)
        read = stub.parse_metadata

        def watched(url: str) -> DouyinMetadata:
            replies_when_the_read_began.append(len(message.replies))
            return read(url=url)

        stub.parse_metadata = watched
        return stub

    cog.__dict__["downloader_factory"] = watched_factory

    await cog.on_message(message=as_message(fake=message))

    assert replies_when_the_read_began == [1]
    assert message.reactions[-1] == _GREEN


async def test_a_channel_that_refuses_the_placeholder_is_never_read_from() -> None:
    """A channel that will not take the placeholder will not take the card either.

    Finding that out before the read is the point: Douyin bans on request volume, so a
    read-only channel must not cost one fetch per pasted link.
    """
    cog, made = _cog()
    message = _message()

    async def refuse(**kwargs: object) -> FakeDiscordMessage:
        """Refuses the reply the way a channel without Send Messages does."""
        del kwargs
        raise make_forbidden()

    message.reply = refuse  # ty: ignore[invalid-assignment]

    await cog.on_message(message=as_message(fake=message))

    assert made == {}  # no downloader was ever built, so Douyin was never contacted
    assert message.reactions[-1] == _RED


async def test_a_message_addressed_to_the_bot_is_left_alone() -> None:
    """A mention (or a DM) hands the link to gen_reply, so the cog must not fetch anything."""
    cog, made = _cog()

    mentioned = _message(content=f"<@999> what is this {_URL}")
    await cog.on_message(message=as_message(fake=mentioned))
    assert mentioned.reactions == []
    assert mentioned.replies == []

    direct_message = _message()
    direct_message.guild = None  # a DM always reaches gen_reply, mention or not
    await cog.on_message(message=as_message(fake=direct_message))
    assert direct_message.reactions == []
    assert direct_message.replies == []

    assert made == {}  # no downloader was ever built, so Douyin was never contacted


async def test_a_message_without_a_link_is_ignored() -> None:
    """The listener sees every message, so a non-Douyin one must cost nothing."""
    cog, made = _cog()
    message = _message(content="just chatting")

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions == []
    assert made == {}


async def test_a_bot_author_is_ignored() -> None:
    """Without this the cog would re-expand its own posts and the other bots' link cards."""
    cog, made = _cog()
    message = _message()
    message.author = FakeUser(bot=True)

    await cog.on_message(message=as_message(fake=message))

    assert made == {}


async def test_a_blocked_request_is_never_reported_as_a_missing_post() -> None:
    """A WAF block is retryable and the link is fine, so it gets its own reaction.

    The reaction is the only thing keeping the two apart now that a failure says nothing in
    the channel, which is what makes ⏱️ load-bearing rather than decorative: ⚠️ means the
    post could not be read, ⏱️ means the request was refused and the same link works later.
    All four expansion cogs answer with the same five marks, so the reader learns them once.
    """
    cog, _ = _cog(download_error=DouyinBlockedError("bot wall"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == EXPANSION_RETRY_LATER_EMOJI
    assert placeholder_withdrawn(message=message)


async def test_a_deleted_post_is_marked_failed_without_a_message() -> None:
    """A post Douyin refuses to serve leaves the same reaction and nothing else."""
    cog, _ = _cog(download_error=DouyinUnavailableError("filtered"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == "⚠️"
    assert placeholder_withdrawn(message=message)


async def test_a_parse_failure_is_marked_failed_without_a_message() -> None:
    """A failure before the download reaches the user as a reaction and nothing else."""
    cog, _ = _cog(parse_error=DouyinError("unreadable"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == "⚠️"
    assert placeholder_withdrawn(message=message)


async def test_an_unexpected_failure_marks_the_message() -> None:
    """A failure outside the fetch must not leave the source silently unmarked."""
    cog, _ = _cog()

    async def boom(*, message: Message, url: str, current_emoji: str) -> None:
        """Fails the way a Discord API error would."""
        del message, url, current_emoji
        raise RuntimeError("discord exploded")

    cog.__dict__["_expand"] = boom
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == _RED


async def test_an_oversize_clip_is_hosted_as_a_url(tmp_path: Path) -> None:
    """Too big to attach means a hosted link, exactly as `/download_video` behaves."""
    cog, _ = _cog()
    (tmp_path / "serve").mkdir()  # pre-existing host mount; the bot never creates the serve dir
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(
            config=make_media_hosting_config(
                enabled=True, base_url="https://media.test", serve_dir=str(tmp_path / "serve")
            )
        )
    )
    message = _message(filesize_limit=4)  # tiny ceiling -> the clip counts as oversize

    await cog.on_message(message=as_message(fake=message))

    content = _reply_body(message=message)
    assert any(line.startswith("https://media.test/") for line in content.splitlines())
    assert message.reactions[-1] == _GREEN


async def test_an_unhostable_oversize_clip_is_refused() -> None:
    """With hosting off there is nothing to link, so the post is refused with a reaction.

    The size the refusal used to quote is logged instead: an expansion that delivers nothing
    leaves nothing behind, the same as a post that could not be read.
    """
    cog, _ = _cog()
    message = _message(filesize_limit=4)

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == "⚠️"
    assert placeholder_withdrawn(message=message)
    assert not message.suppressed  # nothing was delivered, so the source keeps its own preview


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
    assert message.reactions[-1] == _GREEN


async def test_the_parsed_post_is_handed_to_the_download() -> None:
    """The parsed post rides into the download, so the post is never resolved a second time.

    Asserting the download ran once would not catch dropping `post=`; the stub records what it
    was actually given.
    """
    cog, made = _cog()
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    stub = made["stub"]
    assert stub.download_calls == 1
    assert stub.received_post is stub.post


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
        assert made == {}, content


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

    def never_returns(url: str) -> DouyinMetadata:
        """Blocks the worker thread the way a stalling CDN read does."""
        del url
        time.sleep(1.0)
        raise AssertionError("should have been abandoned")

    cog.__dict__["downloader_factory"] = lambda output_folder: SimpleNamespace(
        parse_metadata=never_returns, download=never_returns
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

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
    removed: list[str] = []

    class _RacedTemporaryDirectory(tempfile.TemporaryDirectory[str]):
        """Loses the race the way a file arriving after the scan makes the closing rmdir lose it."""

        def cleanup(self) -> None:
            """Removes the tree, then raises what an ENOTEMPTY on the last step raises."""
            removed.append(self.name)
            super().cleanup()
            raise OSError("directory not empty")

    monkeypatch.setattr(
        scratch_dir, "tempfile", SimpleNamespace(TemporaryDirectory=_RacedTemporaryDirectory)
    )
    cog, _ = _cog()

    def never_returns(url: str) -> DouyinMetadata:
        """Blocks the worker thread the way a stalling CDN read does."""
        del url
        time.sleep(1.0)
        raise AssertionError("should have been abandoned")

    cog.__dict__["downloader_factory"] = lambda output_folder: SimpleNamespace(
        parse_metadata=never_returns, download=never_returns
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert removed  # the teardown really ran and really failed
    assert message.reactions[-1] == EXPANSION_RETRY_LATER_EMOJI
    assert placeholder_withdrawn(message=message)
