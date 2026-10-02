"""Shared doubles for the link-expansion cogs and the link-source builders.

Both read the same posts through the same downloaders, so the canned posts, the stub readers
and the block accessors live here once rather than in each platform's two test files.
"""

import json
import time
from types import TracebackType, SimpleNamespace
from typing import Any, Unpack, TypedDict
from pathlib import Path
from datetime import UTC, datetime
import tempfile
import threading
import contextlib
from collections.abc import Callable, Sequence

import pytest
from nextcord import Embed
from nextcord.ext import commands

from discordbot.utils import scratch_dir
from discordbot.typings.media import LoadedMedia
from discordbot.utils.expansion_cog import ConversationExpansionCog
from discordbot.utils.media_delivery import MediaHostingService, MediaDeliveryPlanner
from discordbot.services.platforms.base import PlatformDownloader, PlatformConversation
from discordbot.services.platforms.douyin import DouyinDownload, DouyinMetadata
from discordbot.services.platforms.threads import ThreadsOutput, ThreadsConversation
from discordbot.services.platforms.twitter import TwitterOutput, TwitterConversation
from discordbot.cogs.gen_reply.link_sources import image_ingest
from discordbot.services.platforms.facebook import FacebookOutput, FacebookConversation
from discordbot.services.platforms.instagram import InstagramOutput, InstagramConversation
from discordbot.services.platforms.page_json import FetchedPage

from tests.helpers.casting import as_bot, as_message, make_media_hosting_config
from tests.helpers.discord_mocks import FakeUser, FakeGuild, FakeDiscordMessage, expansion_payload

TWITTER_URL = "https://x.com/Dbacks/status/1628549742539194368"
FACEBOOK_URL = "https://www.facebook.com/groups/1176671326743489/posts/1730774811333135/"
INSTAGRAM_URL = "https://www.instagram.com/p/Dc5eNjYkoZE/"

# One post URL per registered link source, keyed by registry name, each accepted by its source's
# URL pattern and post filter.
SAMPLE_POST_URLS: dict[str, str] = {
    "threads": "https://www.threads.com/@user/post/ABC123",
    "facebook": FACEBOOK_URL,
    "instagram": INSTAGRAM_URL,
    "twitter": TWITTER_URL,
    "douyin": "https://v.douyin.com/tLgj3lCAnds",
    # A real BV id: a short one matches no pattern, so an assertion about it would hold for the
    # wrong reason.
    "bilibili": "https://www.bilibili.com/video/BV1jpK86hEc8",
}

# The id every stub bot answers to, so a test mentions it as `<@999>`.
BOT_USER_ID = 999


def twitter_output(**overrides: object) -> TwitterOutput:
    """One Twitter post, with any field overridden per test."""
    fields: dict[str, object] = {
        "url": TWITTER_URL,
        "text": "post body",
        "author_name": "Dbacks",
        "author_icon_url": "https://pbs.twimg.com/profile_images/1/a.jpg",
        "image_urls": ["https://pbs.twimg.com/media/a.jpg?name=orig"],
        "like_count": 125,
        "comment_count": 540,
        "taken_at": datetime(2026, 9, 5, 8, 47, tzinfo=UTC),
    }
    fields.update(overrides)
    return TwitterOutput(**fields)  # ty: ignore[invalid-argument-type]


def twitter_post(**overrides: object) -> TwitterConversation:
    """A readable Twitter conversation carrying one post."""
    return TwitterConversation(chain=[twitter_output(**overrides)])


def facebook_post(**overrides: object) -> FacebookConversation:
    """A readable Facebook conversation, with any post or conversation field overridden."""
    raw_comments = overrides.pop("comments", [])
    comments = raw_comments if isinstance(raw_comments, list) else []
    selected = overrides.pop("selected_comment_id", "")
    fields: dict[str, object] = {
        "url": FACEBOOK_URL,
        "text": "post body",
        "author_name": "Somebody",
        "author_icon_url": "https://scontent.example/avatar.jpg",
        "group_name": "Some Group",
        "image_urls": ["https://scontent.example/a.jpg", "https://scontent.example/b.jpg"],
        "like_count": 1017,
        "comment_count": 40,
        "share_count": 37,
        "taken_at": datetime(2026, 9, 5, 8, 47, tzinfo=UTC),
    }
    fields.update(overrides)
    return FacebookConversation(
        chain=[FacebookOutput(**fields)],  # ty: ignore[invalid-argument-type]
        reply_branches=[[comment] for comment in comments],
        selected_comment_id=selected,  # ty: ignore[invalid-argument-type]
    )


def instagram_post(**overrides: object) -> InstagramConversation:
    """A readable Instagram conversation, with any post or conversation field overridden."""
    raw_comments = overrides.pop("comments", [])
    comments = raw_comments if isinstance(raw_comments, list) else []
    selected = overrides.pop("selected_comment_id", "")
    fields: dict[str, object] = {
        "url": INSTAGRAM_URL,
        "text": "post body",
        "author_name": "c_cylynn",
        "author_full_name": "晏凌",
        "author_icon_url": "https://instagram.example/avatar.jpg",
        "image_urls": ["https://instagram.example/a.jpg", "https://instagram.example/b.jpg"],
        "like_count": 8855,
        "comment_count": 11,
        "taken_at": datetime(2026, 9, 5, 8, 47, tzinfo=UTC),
    }
    fields.update(overrides)
    return InstagramConversation(
        chain=[InstagramOutput(**fields)],  # ty: ignore[invalid-argument-type]
        reply_branches=[[comment] for comment in comments],
        selected_comment_id=selected,  # ty: ignore[invalid-argument-type]
    )


class StubConversationDownloader:
    """Stands in for the Facebook, Instagram or Twitter downloader, serving one canned outcome."""

    def __init__(self, outcome: PlatformConversation[Any] | Exception) -> None:
        """Holds the conversation to answer with, or the error to raise."""
        self.outcome = outcome
        self.seen: list[str] = []

    def parse_metadata(self, url: str) -> PlatformConversation[Any]:
        """Records the URL asked for, then answers with the canned outcome."""
        self.seen.append(url)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def stub_bot() -> commands.Bot:
    """A bot whose user a test can mention as `<@BOT_USER_ID>`."""
    return as_bot(fake=SimpleNamespace(user=FakeUser(user_id=BOT_USER_ID, bot=True)))


def stub_conversation_cog[CogT: commands.Cog](
    cog_type: type[CogT], outcome: PlatformConversation[Any] | Exception
) -> tuple[CogT, StubConversationDownloader]:
    """Builds a Facebook, Instagram or Twitter expansion cog reading from a stub downloader."""
    cog = cog_type(bot=stub_bot())
    stub = StubConversationDownloader(outcome=outcome)
    cog.__dict__["downloader_factory"] = lambda: stub
    return cog, stub


def expansion_embeds(message: FakeDiscordMessage) -> list[Embed]:
    """The embeds an expansion cog delivered onto its placeholder."""
    return list(expansion_payload(message=message)["embeds"])


def guild_message(content: str, filesize_limit: int = 25 * 1024 * 1024) -> FakeDiscordMessage:
    """A guild message carrying `content`, posted where uploads are capped at `filesize_limit`."""
    return FakeDiscordMessage(content=content, guild=FakeGuild(filesize_limit=filesize_limit))


async def expand(
    cog_type: type[ConversationExpansionCog[Any, Any]],
    outcome: PlatformConversation[Any] | Exception,
    content: str | None = None,
) -> tuple[FakeDiscordMessage, StubConversationDownloader]:
    """Runs a Facebook, Instagram or Twitter expansion of one guild message over a stub reader.

    The message carries the platform's sample post link unless `content` is given.

    Returns:
        The message, holding whatever the expansion did to it, and the stub that served the read.
    """
    cog, stub = stub_conversation_cog(cog_type=cog_type, outcome=outcome)
    message = guild_message(content=content or SAMPLE_POST_URLS[cog_type.SOURCE])
    await cog.on_message(message=as_message(fake=message))
    return message, stub


# Body of the comment every readable `ThreadsDownloaderStub` conversation carries, so a test can
# assert the expansion never renders it.
THREADS_STUB_COMMENT_TEXT = "a stranger's comment the expansion must ignore"


class ParseResultStub:
    """Stands in for the context manager `ThreadsDownloader.parse` returns."""

    def __init__(
        self,
        results: list[ThreadsOutput] | BaseException,
        exit_error: Exception | None = None,
        enter_delay_seconds: float = 0.0,
        output_folder: str | None = None,
    ) -> None:
        """Stores parsed results, the entry and exit errors, and how long the entry blocks."""
        self.results = results
        self.exit_error = exit_error
        self.enter_delay_seconds = enter_delay_seconds
        self.output_folder = output_folder
        self.exited = False
        self.wrote: Path | None = None
        self.finished = threading.Event()

    def __enter__(self) -> ThreadsConversation:
        """Returns the parsed conversation or raises the configured parsing error.

        A readable post always comes back carrying a comment, because that is what production
        yields: the expansion is supposed to ignore them, and a stub with no comments in it
        cannot tell "ignores them" apart from "never saw any".

        `enter_delay_seconds` blocks the worker thread the way a slow-drip CDN does, so a test
        can reach the caller's give-up path with the walk still running. What happens after that
        delay is `download_media`'s shape: the media write is attempted against the folder the
        caller handed over, and never against one this rebuilds.
        """
        time.sleep(self.enter_delay_seconds)
        if self.output_folder is not None:
            # Named before the write, so an abandoned walk still says where it aimed once the
            # removal turned that write into a FileNotFoundError. Suppressed for the same
            # reason production discards it: nothing is awaiting this thread any more.
            self.wrote = Path(self.output_folder) / "clip.mp4"
            with contextlib.suppress(OSError):
                self.wrote.write_bytes(b"clip")
        self.finished.set()
        if isinstance(self.results, BaseException):
            raise self.results
        comment = ThreadsOutput(
            text=THREADS_STUB_COMMENT_TEXT, image_urls=["https://x.test/c.png"]
        )
        return ThreadsConversation(
            chain=self.results, reply_branches=[[comment]] if self.results else []
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Keeps fake parsed outputs available after context exit, or fails the cleanup."""
        self.exited = True
        if self.exit_error:
            raise self.exit_error


class ThreadsDownloaderStub:
    """Stands in for ThreadsDownloader, answering every walk with the configured parse."""

    def __init__(
        self,
        results: list[ThreadsOutput] | BaseException,
        exit_error: Exception | None = None,
        enter_delay_seconds: float = 0.0,
    ) -> None:
        """Stores parsed results, both failures, and how long each parse blocks on entry."""
        self.results = results
        self.exit_error = exit_error
        self.enter_delay_seconds = enter_delay_seconds
        self.parsed: list[ParseResultStub] = []
        self.output_folders: list[str] = []

    def factory(self, output_folder: str) -> "ThreadsDownloaderStub":
        """Stands in for the ThreadsDownloader class, recording the scratch dir it was handed.

        The expansion builds its downloader inside a scratch directory of its own, so the factory
        is the seam a test takes over; this one stub answers every invocation so a test can
        still read back what it was asked to do.
        """
        self.output_folders.append(output_folder)
        return self

    def parse(self, url: str) -> ParseResultStub:
        """Returns a fake parse context manager, recorded so a test can inspect its exit."""
        result = ParseResultStub(
            results=self.results,
            exit_error=self.exit_error,
            enter_delay_seconds=self.enter_delay_seconds,
            output_folder=self.output_folders[-1] if self.output_folders else None,
        )
        self.parsed.append(result)
        return result


def hosting_off_planner() -> MediaDeliveryPlanner:
    """A delivery planner with hosting off, for code under test that is handed a planner.

    A cog built inside a test needs none: the autouse `media_hosting_disabled` fixture already
    turns hosting off for the planner it builds from the environment.
    """
    return MediaDeliveryPlanner(
        media_hosting=MediaHostingService(config=make_media_hosting_config(enabled=False))
    )


def hosting_service(
    serve_dir: Path, max_bytes: int | None = None, retention_hours: float | None = None
) -> MediaHostingService:
    """A host writer publishing into `serve_dir`, which it serves as `https://media.test/`.

    A cap left as None keeps the config's own default.
    """
    return MediaHostingService(
        config=make_media_hosting_config(
            enabled=True,
            base_url="https://media.test",
            serve_dir=str(serve_dir),
            max_bytes=max_bytes,
            retention_hours=retention_hours,
        )
    )


def hosting_planner(serve_dir: Path) -> MediaDeliveryPlanner:
    """A delivery planner hosting into `serve_dir`, which it serves as `https://media.test/`.

    The serve dir must already exist: the bot never creates one, so a missing one is a planner
    that cannot host.
    """
    return MediaDeliveryPlanner(media_hosting=hosting_service(serve_dir=serve_dir))


class StubDouyinDownloader:
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
        self.download_calls: list[dict[str, Any]] = []

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
        """Records the request, then writes the canned files into the scratch dir or raises.

        At most `max_images` files are written, as the real gallery download caps itself.
        """
        self.download_calls.append({
            "url": url,
            "quality": quality,
            "max_images": max_images,
            "max_bytes": max_bytes,
            "post": post,
        })
        if self.download_error is not None:
            raise self.download_error
        written: list[Path] = []
        for name, payload in self.files[:max_images]:
            path = Path(self.output_folder) / name
            path.write_bytes(payload)
            written.append(path)
        source = post or self.post
        return DouyinDownload(
            is_photo=source.is_photo, filenames=written, total_images=self.total_images
        )


class StubDouyinOptions(TypedDict, total=False):
    """Canned per-stage outcomes for every `StubDouyinDownloader` a test stages."""

    post: DouyinMetadata | None
    files: list[tuple[str, bytes]] | None
    parse_error: Exception | None
    download_error: Exception | None
    total_images: int


def stub_douyin_downloads(
    made: list[StubDouyinDownloader], **canned: Unpack[StubDouyinOptions]
) -> Callable[..., StubDouyinDownloader]:
    """Stands in for the DouyinDownloader class, keeping every stub it builds in `made`.

    The Douyin link-source builder builds one to read the post and another to download it, so
    `made` keeps every stub; each serves the same canned outcomes.
    """

    def factory(output_folder: str) -> StubDouyinDownloader:
        """Builds the stub one read gets."""
        stub = StubDouyinDownloader(output_folder=output_folder, **canned)
        made.append(stub)
        return stub

    return factory


def serve_conversation(
    monkeypatch: pytest.MonkeyPatch,
    downloader: type[PlatformDownloader],
    post: PlatformConversation[Any] | None = None,
    error: Exception | None = None,
) -> None:
    """Points a link-source builder's reader at a canned outcome instead of the network."""
    outcome = error if error is not None else post
    assert outcome is not None, "stage a post or an error"

    def parse_metadata(self: PlatformDownloader, url: str) -> PlatformConversation[Any]:
        """Answers with the canned post, or raises the canned error."""
        del self, url
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(target=downloader, name="parse_metadata", value=parse_metadata)


def sjs_script(payload: object) -> str:
    """Wraps one JSON payload in the `data-sjs` script block a post page serialises it in."""
    return f'<script type="application/json" data-sjs>{json.dumps(obj=payload)}</script>'


def sjs_page(blocks: Sequence[object]) -> str:
    """A post page carrying `blocks` as script blocks, after one that does not parse.

    The unparsable block is what a reader must skip rather than fail on, as it does among the
    dozens a real page carries.
    """
    scripts = "".join(sjs_script(payload=block) for block in blocks)
    return f'<html><script type="application/json">{{"broken"</script>{scripts}</html>'


def serve_page(
    monkeypatch: pytest.MonkeyPatch,
    downloader: type[PlatformDownloader],
    html: str,
    final_url: str | None = None,
) -> list[str]:
    """Points a page reader's one network call at canned HTML.

    The fetch lands on `final_url` when one is given, the way a redirect leaves it, and otherwise
    where it was asked to go.

    Returns:
        Every URL the reader fetched, in order, so a test can tell which form of a link it asked
        for.
    """
    fetched: list[str] = []

    def fetch_page(self: PlatformDownloader, url: str) -> FetchedPage:
        """Records the URL asked for, then answers with the canned page."""
        del self
        fetched.append(url)
        return FetchedPage(html=html, final_url=final_url or url)

    monkeypatch.setattr(target=downloader, name="_fetch_page", value=fetch_page)
    return fetched


def block_separator(blocks: list[Any]) -> str:
    """The separator text a builder led with."""
    return blocks[0]["content"][0]["text"]


def block_body(blocks: list[Any]) -> str:
    """The rendered post text a builder injected."""
    return blocks[1]["content"][0]["text"]


def block_parts(blocks: list[Any]) -> list[Any]:
    """Every content part of the injected user block, text and uploads alike."""
    return list(blocks[1]["content"])


class FakeUploads:
    """Records every media upload a builder performs and hands back canned uris."""

    def __init__(self, fail: bool = False) -> None:
        """Initializes the upload record and whether every upload should fail."""
        self.calls: list[tuple[object, str, str]] = []
        self.fail = fail

    async def __call__(
        self, client: object, source: object, mime_type: str, filename: str, timeout_seconds: float
    ) -> dict[str, str] | None:
        """Stands in for `upload_as_input_file`, returning a Files-API-shaped part."""
        del client, timeout_seconds
        self.calls.append((source, mime_type, filename))
        if self.fail:
            return None
        return {
            "type": "input_file",
            "file_id": f"https://files.test/{filename}",
            "filename": filename,
        }


def accept_image_uploads(
    monkeypatch: pytest.MonkeyPatch,
    uploads: FakeUploads | None = None,
    refused: Callable[[str], bool] | None = None,
) -> list[str]:
    """Makes the shared image fetch answer and installs `uploads` as the Files API upload.

    A source `refused` accepts fails its fetch the way an expired CDN URL does.

    Returns:
        Every source fetched, refused ones included, in order.
    """
    fetched: list[str] = []

    async def load_image_bytes(source: str) -> LoadedMedia:
        """Pretends the CDN answered, unless this source is one it refuses."""
        fetched.append(source)
        if refused is not None and refused(source):
            raise RuntimeError(f"cdn url expired: {source}")
        return LoadedMedia(data=b"image-bytes", mime_type="image/jpeg")

    monkeypatch.setattr(target=image_ingest, name="load_image_bytes", value=load_image_bytes)
    monkeypatch.setattr(
        target=image_ingest, name="upload_as_input_file", value=uploads or FakeUploads()
    )
    return fetched


def race_every_scratch_teardown(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Makes every scratch removal fail the way one racing a live writer does.

    Returns the list each attempt records itself in, so a test can tell a teardown that ran
    and failed from one that was never reached.
    """
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
    return removed
