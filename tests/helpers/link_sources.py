"""Shared doubles for the link-expansion cogs and the link-source builders.

Both read the same posts through the same downloaders, so the canned posts, the stub readers
and the block accessors live here once rather than in each platform's two test files.
"""

from types import SimpleNamespace
from typing import Any, Unpack, TypedDict
from pathlib import Path
from datetime import UTC, datetime
import tempfile
from collections.abc import Callable

import pytest
from nextcord import Embed
from nextcord.ext import commands

from discordbot.utils import scratch_dir
from discordbot.typings.media import LoadedMedia
from discordbot.utils.media_delivery import MediaHostingService, MediaDeliveryPlanner
from discordbot.services.platforms.base import PlatformDownloader, PlatformConversation
from discordbot.services.platforms.douyin import DouyinDownload, DouyinMetadata
from discordbot.services.platforms.twitter import TwitterOutput, TwitterConversation
from discordbot.cogs.gen_reply.link_sources import image_ingest
from discordbot.services.platforms.facebook import FacebookOutput, FacebookConversation
from discordbot.services.platforms.instagram import InstagramOutput, InstagramConversation

from tests.helpers.casting import as_bot, make_media_hosting_config
from tests.helpers.discord_mocks import FakeUser, FakeDiscordMessage, expansion_payload

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

    def __init__(self, *, outcome: PlatformConversation[Any] | Exception) -> None:
        """Holds the conversation to answer with, or the error to raise."""
        self.outcome = outcome
        self.seen: list[str] = []

    def parse_metadata(self, *, url: str) -> PlatformConversation[Any]:
        """Records the URL asked for, then answers with the canned outcome."""
        self.seen.append(url)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def stub_bot() -> commands.Bot:
    """A bot whose user a test can mention as `<@BOT_USER_ID>`."""
    return as_bot(fake=SimpleNamespace(user=FakeUser(user_id=BOT_USER_ID, bot=True)))


def stub_conversation_cog[CogT: commands.Cog](
    *, cog_type: type[CogT], outcome: PlatformConversation[Any] | Exception
) -> tuple[CogT, StubConversationDownloader]:
    """Builds a Facebook, Instagram or Twitter expansion cog reading from a stub downloader."""
    cog = cog_type(bot=stub_bot())
    stub = StubConversationDownloader(outcome=outcome)
    cog.__dict__["downloader_factory"] = lambda: stub
    return cog, stub


def expansion_embeds(*, message: FakeDiscordMessage) -> list[Embed]:
    """The embeds an expansion cog delivered onto its placeholder."""
    return list(expansion_payload(message=message)["embeds"])


def hosting_off_planner() -> MediaDeliveryPlanner:
    """A delivery planner with hosting explicitly disabled.

    Never the no-arg default, whose config is `available` on a dev box where `.env` enables
    hosting: an oversize file would then be moved into the live serve dir.
    """
    return MediaDeliveryPlanner(
        media_hosting=MediaHostingService(config=make_media_hosting_config(enabled=False))
    )


def hosting_planner(serve_dir: Path) -> MediaDeliveryPlanner:
    """A delivery planner hosting into `serve_dir`, which it serves as `https://media.test/`.

    The serve dir must already exist: the bot never creates one, so a missing one is a planner
    that cannot host.
    """
    return MediaDeliveryPlanner(
        media_hosting=MediaHostingService(
            config=make_media_hosting_config(
                enabled=True, base_url="https://media.test", serve_dir=str(serve_dir)
            )
        )
    )


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

    def parse_metadata(self, *, url: str) -> DouyinMetadata:
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
    *,
    downloader: type[PlatformDownloader],
    post: PlatformConversation[Any] | None = None,
    error: Exception | None = None,
) -> None:
    """Points a link-source builder's reader at a canned outcome instead of the network."""
    outcome = error if error is not None else post
    assert outcome is not None, "stage a post or an error"

    def parse_metadata(self: PlatformDownloader, *, url: str) -> PlatformConversation[Any]:
        """Answers with the canned post, or raises the canned error."""
        del self, url
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(target=downloader, name="parse_metadata", value=parse_metadata)


def accept_image_uploads(monkeypatch: pytest.MonkeyPatch, *, uploaded: list[str]) -> None:
    """Makes the shared image fetch and upload succeed, recording what was fetched."""

    async def load_image_bytes(*, source: str) -> LoadedMedia:
        """Pretends the CDN answered."""
        uploaded.append(source)
        return LoadedMedia(data=b"bytes", mime_type="image/jpeg")

    async def upload_as_input_file(
        *, client: object, source: bytes, mime_type: str, filename: str, timeout_seconds: float
    ) -> dict[str, str]:
        """Stands in for the Files API upload."""
        del client, source, mime_type, timeout_seconds
        return {"type": "input_file", "file_id": filename}

    monkeypatch.setattr(target=image_ingest, name="load_image_bytes", value=load_image_bytes)
    monkeypatch.setattr(
        target=image_ingest, name="upload_as_input_file", value=upload_as_input_file
    )


def block_separator(*, blocks: list[Any]) -> str:
    """The separator text a builder led with."""
    return blocks[0]["content"][0]["text"]


def block_body(*, blocks: list[Any]) -> str:
    """The rendered post text a builder injected."""
    return blocks[1]["content"][0]["text"]


def block_parts(*, blocks: list[Any]) -> list[Any]:
    """Every content part of the injected user block, text and uploads alike."""
    return list(blocks[1]["content"])


class FakeUploads:
    """Records every media upload a builder performs and hands back canned uris."""

    def __init__(self, fail: bool = False) -> None:
        """Initializes the upload record and whether every upload should fail."""
        self.calls: list[tuple[object, str, str]] = []
        self.fail = fail

    async def __call__(
        self,
        *,
        client: object,
        source: object,
        mime_type: str,
        filename: str,
        timeout_seconds: float,
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
