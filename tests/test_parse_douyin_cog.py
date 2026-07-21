"""Tests for the cog that auto-expands Douyin links pasted into a channel."""

from types import SimpleNamespace
import asyncio
from pathlib import Path

import pytest

from discordbot.utils.douyin import (
    DouyinPost,
    DouyinError,
    DouyinDownload,
    DouyinBlockedError,
    DouyinTooLargeError,
    DouyinUnavailableError,
)
from discordbot.cogs.parse_douyin import DouyinCogs
from discordbot.utils.media_delivery import (
    MediaHostingConfig,
    MediaHostingService,
    MediaDeliveryPlanner,
)

from tests.helpers.discord_mocks import FakeUser, FakeDiscordMessage

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
        """Writes the canned files into the scratch dir, or raises the canned failure."""
        del url, quality, max_images, max_bytes
        self.download_calls += 1
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


def _cog(
    bot_id: int = 999, **downloader_kwargs: object
) -> tuple[DouyinCogs, dict[str, _StubDownloader]]:
    """Builds a cog wired to a stub downloader and a hosting-off delivery planner."""
    cog = DouyinCogs(bot=SimpleNamespace(user=SimpleNamespace(id=bot_id)))
    # Explicitly disabled planner — never the no-arg default, whose config is `available` on a
    # dev box where .env enables hosting (it would write into the live serve dir).
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(config=MediaHostingConfig(MEDIA_HOSTING_ENABLED=False))
    )
    made: dict[str, _StubDownloader] = {}

    def factory(output_folder: str) -> _StubDownloader:
        """Records the stub so a test can assert on what it was asked to do."""
        stub = _StubDownloader(output_folder=output_folder, **downloader_kwargs)
        made["stub"] = stub
        return stub

    cog.downloader_factory = factory
    return cog, made


def _message(content: str = _URL, filesize_limit: int = 25 * 1024 * 1024) -> FakeDiscordMessage:
    """Builds a guild message carrying a Douyin link."""
    message = FakeDiscordMessage()
    message.author = FakeUser(bot=False)
    message.content = content
    message.guild = SimpleNamespace(filesize_limit=filesize_limit)
    return message


async def test_a_pasted_link_is_expanded_with_its_caption() -> None:
    """A plain paste attaches the clip, adds a caption card, and suppresses the raw preview."""
    cog, made = _cog()
    message = _message()

    await cog.on_message(message=message)

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
    await cog.on_message(message=mentioned)
    assert mentioned.reactions == []
    assert mentioned.replies == []

    direct_message = _message()
    direct_message.guild = None  # a DM always reaches gen_reply, mention or not
    await cog.on_message(message=direct_message)
    assert direct_message.reactions == []
    assert direct_message.replies == []

    assert made == {}  # no downloader was ever built, so Douyin was never contacted


async def test_a_message_without_a_link_is_ignored() -> None:
    """The listener sees every message, so a non-Douyin one must cost nothing."""
    cog, made = _cog()
    message = _message(content="just chatting")

    await cog.on_message(message=message)

    assert message.reactions == []
    assert made == {}


async def test_a_bot_author_is_ignored() -> None:
    """Without this the cog would re-expand its own posts and the other bots' link cards."""
    cog, made = _cog()
    message = _message()
    message.author = FakeUser(bot=True)

    await cog.on_message(message=message)

    assert made == {}


async def test_the_kill_switch_stops_every_request() -> None:
    """Auto-expansion is the one lever that stops the bot talking to Douyin during a WAF ban."""
    cog, made = _cog()
    cog.config = SimpleNamespace(auto_expand_enabled=False)
    message = _message()

    await cog.on_message(message=message)

    assert message.reactions == []
    assert made == {}


async def test_a_blocked_request_is_never_reported_as_a_missing_post() -> None:
    """A WAF block is retryable and the link is fine, so it gets its own reaction and wording."""
    cog, _ = _cog(download_error=DouyinBlockedError("bot wall"))
    message = _message()

    await cog.on_message(message=message)

    assert message.reactions[-1] == DouyinCogs.blocked_emoji
    body = message.replies[0]["content"]
    assert "稍後再試" in body
    assert "刪除" not in body  # never conflated with a deleted or private post


async def test_a_deleted_post_says_so() -> None:
    """A post Douyin refuses to serve is reported as deleted or private, not as a block."""
    cog, _ = _cog(download_error=DouyinUnavailableError("filtered"))
    message = _message()

    await cog.on_message(message=message)

    assert message.reactions[-1] == "⚠️"
    assert "刪除" in message.replies[0]["content"]


async def test_an_oversize_post_points_at_the_command() -> None:
    """A refused download still leaves the user somewhere to go instead of a dead end."""
    cog, _ = _cog(download_error=DouyinTooLargeError("too big"))
    message = _message()

    await cog.on_message(message=message)

    assert message.reactions[-1] == "⚠️"
    assert "/download_video" in message.replies[0]["content"]


async def test_a_parse_failure_still_answers() -> None:
    """Any other failure reports plainly rather than leaving the source message unmarked."""
    cog, _ = _cog(parse_error=DouyinError("unreadable"))
    message = _message()

    await cog.on_message(message=message)

    assert message.reactions[-1] == "⚠️"
    assert message.replies[0]["content"] == "-# 檔案無法下載"


async def test_an_unexpected_failure_marks_the_message() -> None:
    """A failure outside the fetch must not leave the source silently unmarked."""
    cog, _ = _cog()

    async def boom(**_: object) -> None:
        """Fails the way a Discord API error would."""
        raise RuntimeError("discord exploded")

    cog._expand = boom
    message = _message()

    await cog.on_message(message=message)

    assert message.reactions[-1] == _RED


async def test_an_oversize_clip_is_hosted_as_a_url(tmp_path: Path) -> None:
    """Too big to attach means a hosted link, exactly as `/download_video` behaves."""
    cog, _ = _cog()
    (tmp_path / "serve").mkdir()  # pre-existing host mount; the bot never creates the serve dir
    cog.media_delivery = MediaDeliveryPlanner(
        media_hosting=MediaHostingService(
            config=MediaHostingConfig(
                MEDIA_HOSTING_ENABLED=True,
                MEDIA_HOSTING_BASE_URL="https://media.test",
                MEDIA_HOSTING_SERVE_DIR=str(tmp_path / "serve"),
            )
        )
    )
    message = _message(filesize_limit=4)  # tiny ceiling -> the clip counts as oversize

    await cog.on_message(message=message)

    content = message.replies[0]["content"]
    assert any(line.startswith("https://media.test/") for line in content.splitlines())
    assert message.reactions[-1] == _GREEN


async def test_an_unhostable_oversize_clip_says_so() -> None:
    """With hosting off there is nothing to link, so the size is stated instead of dropped."""
    cog, _ = _cog()
    message = _message(filesize_limit=4)

    await cog.on_message(message=message)

    assert message.reactions[-1] == "⚠️"
    assert "檔案大小超過" in message.replies[0]["content"]
    assert not message.suppressed  # nothing was posted, so the source keeps its own preview


async def test_a_capped_gallery_reports_what_it_left_out() -> None:
    """A gallery trimmed by Discord's attachment cap says so rather than silently dropping."""
    cog, _ = _cog(
        post=DouyinPost(aweme_id="1", title="gallery", author_name="a", is_photo=True),
        files=[(f"1_{index}.jpg", b"x" * (index + 1)) for index in range(3)],
        total_images=12,
    )
    message = _message()

    await cog.on_message(message=message)

    assert "已省略 9 張圖片" in message.replies[0]["content"]
    assert message.reactions[-1] == _GREEN


async def test_the_parsed_post_is_reused_for_the_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caption is parsed once and handed to the download, so nothing is fetched twice."""
    cog, made = _cog()
    message = _message()

    await cog.on_message(message=message)

    assert made["stub"].download_calls == 1
