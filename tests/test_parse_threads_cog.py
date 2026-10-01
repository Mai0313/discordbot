"""Tests for the cog that auto-expands Threads links pasted into a channel, and its link cleaner.

The downloader is stubbed at `downloader_factory`, so nothing here touches the network or the
parser: what these cover is the cog's own decisions — when it fires, what it renders, and what
a reader sees when the post cannot be read.
"""

from __future__ import annotations

import time
from types import TracebackType, SimpleNamespace
from typing import TYPE_CHECKING, cast
import asyncio
from pathlib import Path
from datetime import UTC, datetime
import threading
import contextlib

import nextcord
from nextcord import Embed

from discordbot.cogs.parse_threads import cog as parse_threads
from discordbot.utils.discord_embeds import embed_text_length
from discordbot.cogs.parse_threads.cog import ThreadsCogs
from discordbot.services.platforms.threads import ThreadsOutput, ThreadsConversation
from discordbot.utils.expansion_placeholder import (
    EXPANSION_DONE_EMOJI,
    EXPANSION_FAILED_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
)

from tests.helpers.casting import as_message, as_interaction
from tests.helpers.link_sources import stub_bot, hosting_planner, hosting_off_planner
from tests.helpers.discord_mocks import (
    FakeGuild,
    FakeInteraction,
    FakeDiscordMessage,
    expansion_payload,
    placeholder_withdrawn,
)
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    import pytest

_URL = "https://www.threads.net/@alice/post/abc"


def _cog() -> ThreadsCogs:
    """Builds the cog on a stub bot, with media hosting off."""
    cog = ThreadsCogs(bot=stub_bot())
    cog.media_delivery = hosting_off_planner()
    return cog


def _message(content: str = _URL, filesize_limit: int = 25 * 1024 * 1024) -> FakeDiscordMessage:
    """Builds a guild message carrying a Threads link."""
    return FakeDiscordMessage(content=content, guild=FakeGuild(filesize_limit=filesize_limit))


# Body of the comment every readable ParseResultStub conversation carries, so a test can assert
# the Discord expansion never renders it.
_STUB_COMMENT_TEXT = "a stranger's comment the expansion must ignore"


class ParseResultStub:
    """Context manager stub for Threads parse results."""

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
        return ThreadsConversation(
            chain=self.results,
            reply_branches=(
                [[_thread_output(text=_STUB_COMMENT_TEXT, image_urls=["https://x.test/c.png"])]]
                if self.results
                else []
            ),
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
    """Fake Threads downloader returning a configured parse context manager."""

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


def _wire_threads(*, cog: ThreadsCogs, downloader: ThreadsDownloaderStub) -> ThreadsDownloaderStub:
    """Points the cog's per-invocation factory at one stub, recording the dir it was handed.

    The expansion builds its downloader inside a scratch directory of its own, so the factory
    is the seam a test takes over; the same stub answers every invocation so a test can
    still read back what it was asked to do.
    """

    def factory(output_folder: str) -> ThreadsDownloaderStub:
        """Records the scratch dir this invocation was given, then serves the shared stub."""
        downloader.output_folders.append(output_folder)
        return downloader

    cog.__dict__["downloader_factory"] = factory
    return downloader


def _thread_output(  # noqa: PLR0913 -- one knob per ThreadsOutput field the embeds render
    text: str = "hello",
    image_urls: list[str] | None = None,
    video_paths: list[Path] | None = None,
    video_urls: list[str] | None = None,
    author_name: str = "alice",
    quoted: ThreadsOutput | None = None,
    quoted_unavailable: bool = False,
) -> ThreadsOutput:
    """Builds a parsed Threads output fixture."""
    return ThreadsOutput(
        text=text,
        url=f"https://www.threads.net/@{author_name}/post/abc",
        image_urls=image_urls or [],
        video_urls=video_urls or [],
        video_paths=video_paths or [],
        author_name=author_name,
        author_icon_url="https://example.test/avatar.png",
        like_count=1,
        comment_count=2,
        repost_count=3,
        quote_count=4,
        share_count=5,
        taken_at=datetime(2026, 1, 1, tzinfo=UTC),
        quoted=quoted,
        quoted_unavailable=quoted_unavailable,
    )


def _long_threads_chain() -> list[ThreadsOutput]:
    """Builds the measured worst-case chain that crosses the message-wide embed limit."""
    chain: list[ThreadsOutput] = []
    for index in range(10):
        prefix = f"post-{index}-"
        post = _thread_output(
            text=prefix + "x" * (500 - len(prefix)),
            author_name=(f"user-{index}-" + "a" * 30)[:30],
            video_urls=([] if index == 9 else [f"https://example.test/video-{index}.mp4"]),
        )
        post.like_count = 9_999_999
        post.comment_count = 9_999_999
        post.repost_count = 9_999_999
        post.quote_count = 9_999_999
        post.share_count = 9_999_999
        chain.append(post)
    return chain


async def test_clean_threads_url_answers_the_caller_alone() -> None:
    """The pasted link names whoever shared it, so the answer is theirs to copy, not the channel's.

    The share text is handed over whole here because that is what a share button copies, and
    pasting the blob straight in has to work exactly as pasting it into a channel does.
    """
    cog = _cog()
    asked: list[str] = []

    def resolve_clean_url(*, url: str) -> str:
        """Records what it was handed and answers with the post's own URL."""
        asked.append(url)
        return "https://www.threads.com/@alice/post/ABC123"

    def factory(output_folder: str) -> SimpleNamespace:
        """Serves a downloader that resolves without a network of any kind."""
        del output_folder
        return SimpleNamespace(resolve_clean_url=resolve_clean_url)

    cog.__dict__["downloader_factory"] = factory
    interaction = FakeInteraction()

    await ThreadsCogs.clean_threads_url.callback(
        cog,
        as_interaction(fake=interaction),
        url="看看這個 https://www.threads.com/share/D1CytHQmC/ 很好笑",
    )

    assert interaction.response.deferred_ephemeral is True
    assert interaction.followup.sent == [
        {"content": "https://www.threads.com/@alice/post/ABC123", "ephemeral": True}
    ]
    assert asked == ["https://www.threads.com/share/D1CytHQmC/"]


async def test_clean_threads_url_asks_threads_nothing_about_a_link_it_cannot_use() -> None:
    """A non-Threads link is refused from the string alone; nothing is fetched to find that out."""
    cog = _cog()

    def factory(output_folder: str) -> SimpleNamespace:
        """Fails the test if the command ever builds a downloader for this input."""
        del output_folder
        raise AssertionError("no downloader may be built for a link that is not a Threads post")

    cog.__dict__["downloader_factory"] = factory
    interaction = FakeInteraction()

    await ThreadsCogs.clean_threads_url.callback(
        cog, as_interaction(fake=interaction), url="https://example.test/whatever"
    )

    assert interaction.followup.sent == [
        {"content": "這裡面沒有 Threads 貼文連結。", "ephemeral": True}
    ]


async def test_clean_threads_url_tells_the_two_failures_apart() -> None:
    """Neither failure may be worded as temporary: only one of them is worth trying again.

    A link that resolved to something other than a post will resolve to it again, and a fetch
    `_fetch_page` refused is as likely to be a page Threads will not serve as a network blip.
    """
    cog = _cog()
    share_url = "https://www.threads.com/share/D1CytHQmC/"

    def stage(outcome: str | RuntimeError) -> FakeInteraction:
        """Points the cog at a downloader answering with `outcome`, on a fresh interaction."""

        def resolve_clean_url(*, url: str) -> str:
            """Returns the staged answer, or raises it when that is what was staged."""
            del url
            if isinstance(outcome, RuntimeError):
                raise outcome
            return outcome

        cog.__dict__["downloader_factory"] = lambda output_folder: SimpleNamespace(
            resolve_clean_url=resolve_clean_url
        )
        return FakeInteraction()

    landed_on_no_post = stage("")
    await ThreadsCogs.clean_threads_url.callback(
        cog, as_interaction(fake=landed_on_no_post), url=share_url
    )
    assert landed_on_no_post.followup.sent == [
        {"content": "這個分享連結沒有指向任何貼文。", "ephemeral": True}
    ]

    fetch_failed = stage(RuntimeError("boom"))
    await ThreadsCogs.clean_threads_url.callback(
        cog, as_interaction(fake=fetch_failed), url=share_url
    )
    assert fetch_failed.followup.sent == [{"content": "這個連結現在拿不到。", "ephemeral": True}]


async def test_threads_cog_builds_embeds_and_handles_messages(tmp_path: Path) -> None:
    """Verifies Threads embed building, and that a delivered expansion carries its files."""
    cog = _cog()
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(data=b"123")

    parent = _thread_output(text="parent", video_urls=["https://example.test/video.mp4"])
    target = _thread_output(
        image_urls=["https://example.test/1.png", "https://example.test/2.png"]
    )
    embeds = cog._build_embeds(results=[parent, target])
    assert len(embeds) == 3
    first_description = embeds[0].description
    assert first_description is not None
    assert "點此觀看影片" in first_description
    assert ThreadsCogs._gradient_color(index=0, total=1) == nextcord.Color.default()

    no_match = FakeDiscordMessage(content="hello")
    await cog.on_message(message=as_message(fake=no_match))
    assert no_match.reactions == []

    success_message = _message()
    _wire_threads(
        cog=cog,
        downloader=ThreadsDownloaderStub(
            results=[_thread_output(video_paths=[video_file], image_urls=[])]
        ),
    )
    await cog.on_message(message=as_message(fake=success_message))
    assert success_message.suppressed
    delivered = expansion_payload(message=success_message)
    assert delivered["files"]
    assert success_message.reactions[-1] == EXPANSION_DONE_EMOJI
    # The parse carries the comments too, but the expansion shows the chain only: the 10-embed
    # cap belongs to the linked post, and a comment would push its own images out.
    assert all(
        _STUB_COMMENT_TEXT not in (embed.description or "") for embed in delivered["embeds"]
    )


async def test_threads_cog_takes_the_scratch_dir_of_a_walk_it_gave_up_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out expansion leaves nothing behind, without waiting for the walk.

    The `requests` calls under `parse` are per-read only, so a slow-drip CDN can hold one paste
    open indefinitely; the bound is what stops it. `asyncio.to_thread` cannot cancel the walk,
    so it is the scratch directory going away that both deletes what it wrote and fails its next
    write. The exit is deliberately not called on this path: the walk is still driving that
    generator on its own thread.

    The mark is the retryable one rather than the cross, which is the shared vocabulary every
    expansion cog answers with: the post is fine and the same link works later.
    """
    monkeypatch.setattr(target=parse_threads, name="THREADS_EXPAND_TIMEOUT_SECONDS", value=0.05)
    cog = _cog()
    downloader = _wire_threads(
        cog=cog, downloader=ThreadsDownloaderStub(results=[], enter_delay_seconds=0.3)
    )

    message = _message()
    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == EXPANSION_RETRY_LATER_EMOJI
    scratch = Path(downloader.output_folders[0])
    assert not await asyncio.to_thread(scratch.exists)
    assert downloader.parsed[0].exited is False
    # The abandoned walk runs on past the give-up and writes where it was told to; what it
    # produces has to be gone with the directory rather than stranded in the system temp dir.
    assert await asyncio.to_thread(downloader.parsed[0].finished.wait, 5.0)
    wrote = downloader.parsed[0].wrote
    assert wrote is not None
    assert not await asyncio.to_thread(wrote.exists)
    assert not await asyncio.to_thread(scratch.exists)


async def test_threads_cog_trims_long_chain_to_the_message_wide_embed_limit() -> None:
    """Far ancestors are removed before a total over 6000 can make Discord reject the reply."""
    cog = _cog()

    embeds = cog._build_embeds(results=_long_threads_chain())

    assert sum(embed_text_length(embed=embed) for embed in embeds) <= 6000
    authors = [cast("str", embed.author.name) for embed in embeds if embed.author]
    assert authors[-1].startswith("user-9-")
    assert any(author.startswith("user-8-") for author in authors)
    assert not any(author.startswith("user-0-") for author in authors)
    # What was left behind is stated on the target's own card rather than in a follow-up reply.
    target_embed = next(embed for embed in embeds if authors[-1] == (embed.author.name or ""))
    assert "📝 另有 2 篇未展開" in cast("str", target_embed.footer.text)


async def test_threads_cog_keeps_the_target_quote_and_nearest_ancestor() -> None:
    """The quoted post remains second in priority while a farther ancestor is dropped."""
    cog = _cog()
    root = _thread_output(text="root-" + "r" * 1095, author_name="root")
    parent = _thread_output(text="parent-" + "p" * 1093, author_name="parent")
    target = _thread_output(text="target-" + "t" * 2193, author_name="target")
    target.quoted = _thread_output(
        text="quoted-" + "q" * 2193,
        author_name="quoted",
        image_urls=["https://example.test/quoted-1.png", "https://example.test/quoted-2.png"],
    )

    embeds = cog._build_embeds(results=[root, parent, target])

    assert sum(embed_text_length(embed=embed) for embed in embeds) <= 6000
    descriptions = [embed.description or "" for embed in embeds]
    assert descriptions[0].startswith("parent-")
    assert descriptions[1].startswith("target-")
    assert descriptions[2].startswith("🔗 **被引用的貼文**")
    assert all(not description.startswith("root-") for description in descriptions)
    assert sum(1 for embed in embeds if embed.image) == 2


async def test_threads_cog_drops_an_over_budget_post_with_its_gallery() -> None:
    """A quote that cannot fit does not leave its image-only embeds detached from their text."""
    cog = _cog()
    parent = _thread_output(text="nearby context", author_name="parent")
    target = _thread_output(text="t" * 3500, author_name="target")
    target.quoted = _thread_output(
        text="q" * 3000,
        author_name="quoted",
        image_urls=[f"https://example.test/quoted-{index}.png" for index in range(4)],
    )

    embeds = cog._build_embeds(results=[parent, target])

    assert sum(embed_text_length(embed=embed) for embed in embeds) <= 6000
    assert [embed.author.name for embed in embeds if embed.author] == ["parent", "target"]
    assert all(not embed.image for embed in embeds)
    assert all("被引用的貼文" not in (embed.description or "") for embed in embeds)
    assert "📝 另有 1 篇未展開" in cast("str", embeds[-1].footer.text)


async def test_threads_cog_counts_astral_emoji_as_utf16_units() -> None:
    """Emoji-heavy posts stay safe even if Discord interprets characters as UTF-16 units."""
    cog = _cog()
    chain = [_thread_output(text="😀" * 500, author_name=f"user-{index}") for index in range(10)]

    embeds = cog._build_embeds(results=chain)

    assert len(embeds) < len(chain)
    assert sum(embed_text_length(embed=embed) for embed in embeds) <= 6000
    assert [embed.author.name for embed in embeds if embed.author] == [
        "user-5",
        "user-6",
        "user-7",
        "user-8",
        "user-9",
    ]


async def test_threads_cog_delivers_a_trimmed_chain_in_one_message() -> None:
    """An overflow is one card that says what it left out, never a second reply.

    A follow-up reply lands wherever the channel has got to, several messages below the
    embeds it describes, which is exactly what the placeholder exists to prevent.
    """
    cog = _cog()
    _wire_threads(cog=cog, downloader=ThreadsDownloaderStub(results=_long_threads_chain()))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert len(message.replies) == 1  # the placeholder the card was edited onto, and nothing else
    embeds = expansion_payload(message=message)["embeds"]
    assert sum(embed_text_length(embed=embed) for embed in embeds) <= 6000
    assert "篇未展開" in cast("str", embeds[-1].footer.text)
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


async def test_threads_cog_states_the_images_the_embed_cap_left_behind() -> None:
    """Ten slots is the whole budget, so a bigger carousel is trimmed and counted."""
    cog = _cog()
    target = _thread_output(
        image_urls=[f"https://example.test/image-{index}.png" for index in range(15)]
    )

    embeds = cog._build_embeds(results=[target])

    assert len(embeds) == 10
    assert "🖼️ 另有 5 張" in cast("str", embeds[0].footer.text)


async def test_threads_cog_keeps_the_expansion_when_the_scratch_cleanup_fails() -> None:
    """A temp file the user cannot see must never repaint a delivered expansion as failed."""
    cog = _cog()
    downloader = ThreadsDownloaderStub(
        results=[_thread_output(text="貼文內容")], exit_error=OSError("read-only file system")
    )
    _wire_threads(cog=cog, downloader=downloader)
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    # The cleanup really ran and really failed, so the ✅ below is the guard's doing.
    assert downloader.parsed[0].exited
    assert len(message.replies) == 1
    assert expansion_payload(message=message)["embeds"]
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


async def test_threads_cog_logs_both_a_failed_step_and_the_cleanup_that_failed_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swallowing the cleanup must not swallow the failure the cleanup used to replace."""
    cog = _cog()
    downloader = ThreadsDownloaderStub(
        results=[_thread_output(text="貼文內容")], exit_error=OSError("read-only file system")
    )
    _wire_threads(cog=cog, downloader=downloader)
    message = _message()
    errors = capture_logs(monkeypatch, level="error")
    # A warning rather than an error: the scratch directory around the cleanup removes what a
    # failing unlink left, so it is a degraded step rather than a leak nobody clears.
    warnings = capture_logs(monkeypatch, level="warn")

    def exploding_plan(*, results: list[ThreadsOutput]) -> list[Embed]:
        del results
        raise RuntimeError("the embed plan blew up")

    cog._build_embeds = exploding_plan  # ty: ignore[invalid-assignment]
    await cog.on_message(message=as_message(fake=message))

    assert downloader.parsed[0].exited
    assert placeholder_withdrawn(message=message)
    assert message.reactions[-1] == EXPANSION_FAILED_EMOJI
    # The step that lost the expansion is logged with its own cause rather than with the
    # OSError the cleanup used to overwrite it with, and the cleanup gets its own line.
    assert ("Could not clean up the Threads scratch files", "OSError") in [
        (text, fields.get("error_type")) for text, fields in warnings
    ]
    assert ("Threads expansion failed outside the read and the send", "RuntimeError") in [
        (text, fields.get("error_type")) for text, fields in errors
    ]


async def test_threads_cog_shows_the_post_a_quote_post_quotes() -> None:
    """The quoted post is the subject of a quote post, so it earns its own marked embed."""
    cog = _cog()
    quoted = _thread_output(
        text="the original argument",
        author_name="bob",
        video_urls=["https://example.test/quoted.mp4"],
    )
    target = _thread_output(text="這根本是胡說", image_urls=["https://example.test/1.png"])
    target.quoted = quoted

    embeds = cog._build_embeds(results=[target])

    assert len(embeds) == 2
    # The target owns the message, so it stays first and the quoted post hangs off the end.
    assert embeds[0].description == "這根本是胡說"
    quoted_embed = embeds[1]
    assert quoted_embed.author.name == "bob"
    assert quoted_embed.description is not None
    assert quoted_embed.description.startswith("🔗 **被引用的貼文**")
    assert "the original argument" in quoted_embed.description
    # Its clip is never downloaded, so it is linked instead of showing as an empty embed.
    assert "點此觀看影片" in quoted_embed.description
    # Off the greyscale chain gradient on purpose: it is not a layer of the thread.
    assert quoted_embed.colour == nextcord.Color.blurple()


async def test_threads_cog_keeps_the_commentary_beside_a_quoted_gallery() -> None:
    """The shape that motivated this: one line over someone else's ten-image carousel.

    Letting the gallery compete freely for the 10-embed cap drops the commentary that owns the
    message, which would leave the reader the same fragment showing the quoted post exists to fix.
    """
    cog = _cog()
    quoted = _thread_output(
        text="the subject",
        author_name="bob",
        image_urls=[f"https://example.test/{index}.png" for index in range(10)],
    )
    target = _thread_output(text="一句話評論")
    target.quoted = quoted

    embeds = cog._build_embeds(results=[target])

    assert len(embeds) == 10
    assert embeds[0].description == "一句話評論"
    assert embeds[1].description is not None
    assert embeds[1].description.startswith("🔗 **被引用的貼文**")
    # Nine of the quoted post's ten images fit; the tenth loses to the commentary, not the reverse.
    assert sum(1 for embed in embeds if embed.image) == 9


async def test_threads_cog_notes_a_quoted_post_that_is_gone() -> None:
    """A gone quoted post has nothing to show, so it rides on the target instead of an embed."""
    cog = _cog()
    target = _thread_output(text="回應一下")
    target.quoted_unavailable = True

    embeds = cog._build_embeds(results=[target])

    assert len(embeds) == 1
    assert embeds[0].description is not None
    assert "引用的貼文目前無法瀏覽" in embeds[0].description


async def test_threads_cog_reserves_the_quoted_posts_slot_against_an_ancestors_gallery() -> None:
    """The quoted post's reservation only bites when something else wants the last slot.

    A text-only target quoting a text-only post leaves the whole budget to an image-heavy
    ancestor, so without the reservation the ancestor's tenth image takes the slot and the quoted
    post disappears from a message that is supposed to be about it.
    """
    cog = _cog()
    ancestor = _thread_output(
        text="ancestor",
        author_name="root",
        image_urls=[f"https://example.test/{index}.png" for index in range(10)],
    )
    target = _thread_output(text="commentary")
    target.quoted = _thread_output(text="the post being argued with", author_name="bob")

    embeds = cog._build_embeds(results=[ancestor, target])

    assert len(embeds) == 10
    descriptions = [embed.description or "" for embed in embeds]
    assert any(text.startswith("🔗 **被引用的貼文**") for text in descriptions)
    assert "commentary" in descriptions
    # The ancestor gives up two of its ten images, not the target or the quoted post.
    assert sum(1 for embed in embeds if embed.image) == 8


async def test_threads_cog_says_nothing_about_an_ancestors_quote() -> None:
    """The expansion shows the target's quote only, so an ancestor's must not be half-announced.

    `_build_output` fills `quoted_unavailable` on every parsed post, so an ungated hint told the
    reader about an ancestor's quote in exactly the case where there was nothing to show, while an
    ancestor quoting a live post said nothing at all.
    """
    cog = _cog()
    root = _thread_output(text="root commentary", author_name="root")
    root.quoted_unavailable = True

    embeds = cog._build_embeds(results=[root, _thread_output(text="target")])

    assert embeds[0].description == "root commentary"
    assert all("引用的貼文目前無法瀏覽" not in (embed.description or "") for embed in embeds)


async def test_threads_cog_refuses_an_oversize_quoted_post_with_a_warning() -> None:
    """The user-visible outcome of that overflow is the ⚠️ skip, never the ❌ a 400 would give."""
    cog = _cog()
    target = _thread_output(text="t")
    target.quoted = _thread_output(text="q" * 4096, author_name="bob")
    _wire_threads(cog=cog, downloader=ThreadsDownloaderStub(results=[target]))

    message = _message()
    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == EXPANSION_UNREADABLE_EMOJI
    assert placeholder_withdrawn(message=message)


async def test_threads_cog_hosts_oversized_video(tmp_path: Path) -> None:
    """A Threads video too big to attach is hosted as a URL instead of a ⚠️ refusal."""
    cog = _cog()
    (tmp_path / "serve").mkdir()  # pre-existing host mount; the bot never creates the serve dir
    cog.media_delivery = hosting_planner(serve_dir=tmp_path / "serve")
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(data=b"123")

    # The 1 MiB envelope margin alone overshoots this ceiling, so no combined body ever fits
    # and even a 3-byte video is peeled out to a hosted URL.
    message = _message(filesize_limit=4)
    _wire_threads(
        cog=cog,
        downloader=ThreadsDownloaderStub(
            results=[_thread_output(video_paths=[video_file], image_urls=[])]
        ),
    )

    await cog.on_message(message=as_message(fake=message))

    # The video was hosted (its URL rides the reply content) and moved out of the temp dir.
    content = expansion_payload(message=message).get("content") or ""
    assert any(line.startswith("https://media.test/") for line in content.splitlines())
    assert not video_file.exists()
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


async def test_threads_cog_mixes_native_and_hosted_videos(tmp_path: Path) -> None:
    """A post with one small and one oversize video attaches the small and links only the big one."""
    cog = _cog()
    (tmp_path / "serve").mkdir()  # pre-existing host mount; the bot never creates the serve dir
    cog.media_delivery = hosting_planner(serve_dir=tmp_path / "serve")
    small = tmp_path / "small.mp4"
    small.write_bytes(data=b"0" * 100)
    big = tmp_path / "big.mp4"
    big.write_bytes(data=b"0" * (2 * 1024 * 1024))

    # The ceiling clears the small clip plus the 1 MiB envelope margin but not the 2 MiB clip,
    # so only the big one is peeled to a hosted URL while the small one attaches natively.
    message = _message(filesize_limit=1024 * 1024 + 200)
    _wire_threads(
        cog=cog,
        downloader=ThreadsDownloaderStub(
            results=[_thread_output(video_paths=[small, big], image_urls=[])]
        ),
    )

    await cog.on_message(message=as_message(fake=message))

    content = expansion_payload(message=message).get("content") or ""
    hosted = [line for line in content.splitlines() if line.startswith("https://media.test/")]
    assert len(hosted) == 1  # only the oversize clip was linked
    assert big.exists() is False  # the big clip was moved into the serve dir
    assert small.exists() is True  # the small clip stayed on disk to attach natively
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


async def test_threads_cog_refuses_oversized_video_when_hosting_off(tmp_path: Path) -> None:
    """With hosting off, an oversize Threads video refuses the whole post with a ⚠️."""
    cog = _cog()
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(data=b"123")

    message = _message(filesize_limit=4)  # tiny ceiling -> video oversize
    _wire_threads(
        cog=cog,
        downloader=ThreadsDownloaderStub(
            results=[_thread_output(video_paths=[video_file], image_urls=[])]
        ),
    )

    await cog.on_message(message=as_message(fake=message))

    # No host available + oversize -> whole-post ⚠️ refusal, no reply, and the file is left in place.
    assert message.reactions[-1] == EXPANSION_UNREADABLE_EMOJI
    assert placeholder_withdrawn(message=message)
    assert video_file.exists() is True
