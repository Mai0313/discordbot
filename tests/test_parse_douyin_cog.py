"""Tests for the cog that auto-expands Douyin links pasted into a channel.

`DouyinCogs.on_message` is a listener with no command in front of it, so what it does is decided
entirely by what a message happens to carry. Both halves of that are pinned here: which messages
it must leave untouched, and what it posts back for the ones it takes.

The gates are pinned one at a time because each one costs a Douyin request when it is wrong, and
Douyin's WAF bans a share path for tens of minutes once it is hit hard. A bot author, a message
with no link, a profile or live-room link, a message addressed to the bot (which `gen_reply`
answers about instead, so expanding as well would fetch the same media twice) and the
`auto_expand_enabled` kill-switch each have to end the listener with no reaction, no reply, and
(asserted through the `made` dict `_cog` hands back) no downloader ever built.

The outcomes are pinned as distinct failures rather than as one error path, because reporting a
retryable WAF block as a deleted post is the worst thing this feature can produce: ⏱️ plus retry
wording for a block, ⚠️ plus deletion wording only for a post Douyin actually filtered out, a
pointer at `/download_video` for an oversize one, a red cross with no wording for a failure
outside the fetch, and a stalled expansion that reads as retryable so it never accuses a working
link. The delivery half pins the clip plus caption card plus suppressed source preview, the
oversize-to-hosted-URL fallback, the unhostable case that states the size and leaves the source
preview in place because nothing replaced it, and the gallery that names how many images the
attachment cap left behind.

Every test builds its cog through `_cog`, which is what keeps the suite hermetic and honest: it
stubs the downloader so Douyin is never contacted, pins the kill-switch on so a dev box's `.env`
cannot turn every assertion below into a no-op that still passes, and hands the cog a hosting-off
delivery planner so nothing can write into a real serve dir.
"""

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Unpack, TypedDict, cast
import asyncio
from pathlib import Path

import pytest
from nextcord import Message

from discordbot.utils.douyin import (
    DouyinPost,
    DouyinError,
    DouyinDownload,
    DouyinBlockedError,
    DouyinTooLargeError,
    DouyinUnavailableError,
)
from discordbot.cogs.parse_douyin import cog as parse_douyin
from discordbot.utils.media_delivery import MediaHostingService, MediaDeliveryPlanner
from discordbot.cogs.parse_douyin.cog import DouyinCogs

from tests.helpers.casting import as_bot, as_message, make_media_hosting_config
from tests.helpers.discord_mocks import FakeUser, FakeDiscordMessage

if TYPE_CHECKING:
    from discordbot.typings.douyin import DouyinConfig

_URL = "https://v.douyin.com/abc123"
_GREEN = "<:greencheck:1517565102424068226>"
_RED = "<:redcross:1517565100838355016>"


class _StubDownloader:
    """Stands in for DouyinDownloader, serving canned metadata and files."""

    def __init__(  # noqa: PLR0913 -- one canned outcome per stage the cog can hit
        self,
        output_folder: str,
        post: DouyinPost | None = None,
        files: list[tuple[str, bytes]] | None = None,
        parse_error: Exception | None = None,
        download_error: Exception | None = None,
        total_images: int = 0,
    ) -> None:
        """Records the scratch dir and the canned outcome for each stage."""
        self.output_folder = output_folder
        self.post = post or DouyinPost(aweme_id="1", title="caption", author_name="somebody")
        self.files = files if files is not None else [("1.mp4", b"video-bytes")]
        self.parse_error = parse_error
        self.download_error = download_error
        self.total_images = total_images
        self.download_calls = 0
        self.received_post: DouyinPost | None = None

    def parse_metadata(self, url: str) -> DouyinPost:
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
        post: DouyinPost | None = None,
    ) -> DouyinDownload:
        """Writes the canned files into the scratch dir, or raises the canned failure.

        Returns:
            A `DouyinDownload` over the paths just written, described by the `post` the cog
            handed in rather than by the canned one, so dropping `post=` is visible.
        """
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

    post: DouyinPost | None
    files: list[tuple[str, bytes]] | None
    parse_error: Exception | None
    download_error: Exception | None
    total_images: int


def _cog(
    bot_id: int = 999, **downloader_kwargs: Unpack[_StubOptions]
) -> tuple[DouyinCogs, dict[str, _StubDownloader]]:
    """Builds a cog wired to a stub downloader and a hosting-off delivery planner.

    Returns:
        The cog, plus a dict the downloader factory fills under `"stub"` once the cog builds
        one, so an empty dict is the assertion that Douyin was never contacted at all.
    """
    cog = DouyinCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=bot_id))))
    # Pinned explicitly: DouyinConfig reads the real environment (typings/douyin.py loads .env
    # at import), so a dev box with DOUYIN_AUTO_EXPAND_ENABLED=false would silently turn every
    # test below into a no-op that still passes.
    cog.config = cast("DouyinConfig", SimpleNamespace(auto_expand_enabled=True))
    # Explicitly disabled planner — never the no-arg default, whose config is `available` on a
    # dev box where .env enables hosting (it would write into the live serve dir).
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(config=make_media_hosting_config(enabled=False))
    )
    made: dict[str, _StubDownloader] = {}

    def factory(output_folder: str) -> _StubDownloader:
        """Records the stub so a test can assert on what it was asked to do.

        Returns:
            The stub standing in for this expansion's downloader.
        """
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
    """Builds a guild message carrying a Douyin link.

    Returns:
        A message double whose guild reports `filesize_limit`, which is what the cog reads to
        decide whether a file attaches or has to be hosted.
    """
    return _DouyinMessage(
        author=FakeUser(bot=False),
        content=content,
        guild=SimpleNamespace(filesize_limit=filesize_limit),
    )


def _reply_body(*, message: FakeDiscordMessage) -> str:
    """Returns the first reply's text, failing loudly when the cog posted none."""
    content = message.replies[0]["content"]
    assert content is not None
    return content


async def test_a_pasted_link_is_expanded_with_its_caption() -> None:
    """A plain paste attaches the clip, adds a caption card, and suppresses the raw preview."""
    cog, made = _cog()
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.suppressed
    reply = message.replies[0]
    assert reply["files"]
    assert reply["embeds"][0].description == "caption"
    assert reply["embeds"][0].author.name == "somebody"
    assert message.reactions[-1] == _GREEN
    # The scratch dir is per invocation and removed with its files once delivery finishes.
    assert not await asyncio.to_thread(Path(made["stub"].output_folder).exists)


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


async def test_the_kill_switch_stops_every_request() -> None:
    """Auto-expansion is the one lever that stops the bot talking to Douyin during a WAF ban."""
    cog, made = _cog()
    cog.config = cast("DouyinConfig", SimpleNamespace(auto_expand_enabled=False))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions == []
    assert made == {}


async def test_a_blocked_request_is_never_reported_as_a_missing_post() -> None:
    """A WAF block is retryable and the link is fine, so it gets its own reaction and wording."""
    cog, _ = _cog(download_error=DouyinBlockedError("bot wall"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == DouyinCogs.blocked_emoji
    body = _reply_body(message=message)
    assert "稍後再試" in body
    assert "刪除" not in body  # never conflated with a deleted or private post


async def test_a_deleted_post_says_so() -> None:
    """A post Douyin refuses to serve is reported as deleted or private, not as a block."""
    cog, _ = _cog(download_error=DouyinUnavailableError("filtered"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == "⚠️"
    assert "刪除" in _reply_body(message=message)


async def test_an_oversize_post_points_at_the_command() -> None:
    """A refused download still leaves the user somewhere to go instead of a dead end."""
    cog, _ = _cog(download_error=DouyinTooLargeError("too big"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == "⚠️"
    assert "/download_video" in _reply_body(message=message)


async def test_a_parse_failure_still_answers() -> None:
    """Any other failure reports plainly rather than leaving the source message unmarked."""
    cog, _ = _cog(parse_error=DouyinError("unreadable"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == "⚠️"
    assert message.replies[0]["content"] == "-# 檔案無法下載"


async def test_an_unexpected_failure_marks_the_message() -> None:
    """A failure outside the fetch must not leave the source silently unmarked."""
    cog, _ = _cog()

    async def boom(*, message: Message, url: str, current_emoji: str) -> None:
        """Fails the way a Discord API error would.

        Raises:
            RuntimeError: Always, standing in for a failure outside the Douyin fetch.
        """
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


async def test_an_unhostable_oversize_clip_says_so() -> None:
    """With hosting off there is nothing to link, so the size is stated instead of dropped."""
    cog, _ = _cog()
    message = _message(filesize_limit=4)

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == "⚠️"
    assert "檔案大小超過" in _reply_body(message=message)
    assert not message.suppressed  # nothing was posted, so the source keeps its own preview


async def test_a_capped_gallery_reports_what_it_left_out() -> None:
    """A gallery trimmed by Discord's attachment cap says so rather than silently dropping."""
    cog, _ = _cog(
        post=DouyinPost(aweme_id="1", title="gallery", author_name="a", is_photo=True),
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

    def never_returns(url: str) -> DouyinPost:
        """Blocks the worker thread the way a stalling CDN read does.

        Raises:
            AssertionError: The sleep ran to completion, so the timeout never abandoned it.
        """
        del url
        time.sleep(1.0)
        raise AssertionError("should have been abandoned")

    cog.__dict__["downloader_factory"] = lambda output_folder: SimpleNamespace(
        parse_metadata=never_returns, download=never_returns
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == "⚠️"
    body = _reply_body(message=message)
    assert "稍後再試" in body
    assert "刪除" not in body
