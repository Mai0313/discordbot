"""What every conversation expansion cog's card owes, checked against each cog.

The post's own embed and its gallery come from `post_card_embeds` on every one of them, and
`ConversationExpansionCog`'s default card adds the comment a link singled out. What those
decide is tested here once per cog; each cog's own file keeps its hooks: the footer, the author
line, and where its links point.
"""

from typing import Any, Self
from pathlib import Path
import sqlite3
from datetime import UTC, datetime
from collections.abc import Callable

import pytest
from nextcord import Embed
import requests

from discordbot.utils.expansion_cog import (
    VIDEO_HINT,
    TRUNCATION_NOTICE,
    POST_CARD_MAX_IMAGES,
    ExpansionDelivery,
    ConversationExpansionCog,
    fetch_card_image,
    omitted_images_note,
    omitted_videos_note,
)
from discordbot.utils.discord_embeds import (
    DISCORD_ATTACHMENT_LIMIT,
    DEFAULT_EMBED_SPACER_FILENAME,
    utf16_length,
    embed_text_length,
)
from discordbot.utils.media_delivery import MEDIA_ENVELOPE_MARGIN, MediaItem
from discordbot.cogs.parse_twitter.cog import TwitterCogs
from discordbot.cogs.parse_facebook.cog import FacebookCogs
from discordbot.services.platforms.base import PlatformOutput, PlatformConversation
from discordbot.cogs.parse_instagram.cog import InstagramCogs
from discordbot.services.platforms.facebook import FacebookOutput, FacebookConversation
from discordbot.utils.expansion_placeholder import EXPANSION_DONE_EMOJI
from discordbot.services.platforms.instagram import InstagramOutput

from tests.helpers.casting import as_message
from tests.helpers.source_tree import REPO_ROOT
from tests.helpers.link_sources import (
    SAMPLE_POST_URLS,
    expand,
    twitter_post,
    facebook_post,
    guild_message,
    instagram_post,
    hosting_planner,
    expansion_embeds,
    serve_card_images,
    serve_card_videos,
    stub_conversation_cog,
)
from tests.helpers.discord_mocks import (
    DiscordPayload,
    FakeDiscordMessage,
    stub_bot,
    expansion_payload,
)
from tests.helpers.logfire_capture import capture_logs

type _CogType = type[ConversationExpansionCog[Any, Any]]
type _PostFactory = Callable[..., PlatformConversation[Any]]

every_card = pytest.mark.parametrize(
    argnames=("cog_type", "post"),
    argvalues=[
        (FacebookCogs, facebook_post),
        (InstagramCogs, instagram_post),
        (TwitterCogs, twitter_post),
    ],
    ids=["facebook", "instagram", "twitter"],
)
# The cogs taking the default card whole.
default_cards = pytest.mark.parametrize(
    argnames=("cog_type", "post"),
    argvalues=[(FacebookCogs, facebook_post), (InstagramCogs, instagram_post)],
    ids=["facebook", "instagram"],
)
# The same cogs, with the comment model each one's posts carry.
commented_cards = pytest.mark.parametrize(
    argnames=("cog_type", "post", "comment"),
    argvalues=[
        (FacebookCogs, facebook_post, FacebookOutput),
        (InstagramCogs, instagram_post, InstagramOutput),
    ],
    ids=["facebook", "instagram"],
)
# The cogs downloading direct MP4 target videos via the base hook.
video_cards = pytest.mark.parametrize(
    argnames=("cog_type", "post"),
    argvalues=[(InstagramCogs, instagram_post), (TwitterCogs, twitter_post)],
    ids=["instagram", "twitter"],
)


async def _expand(cog_type: _CogType, outcome: PlatformConversation[Any]) -> list[Embed]:
    """Expands a guild message carrying the platform's post link, returning what it delivered."""
    message, _ = await expand(cog_type=cog_type, outcome=outcome)
    return expansion_embeds(message=message)


@every_card
async def test_extra_images_become_embeds_sharing_the_post_url(
    cog_type: _CogType, post: _PostFactory
) -> None:
    """Sharing the URL is what makes Discord merge them into one gallery under the post."""
    images = [f"https://cdn.test/{n}.jpg" for n in range(3)]
    outcome = post(image_urls=images)
    assert outcome.target is not None

    embeds = await _expand(cog_type=cog_type, outcome=outcome)

    assert len(embeds) == 3
    assert all(embed.url == outcome.target.url for embed in embeds)
    assert [embed.image.url for embed in embeds] == images


@every_card
async def test_a_post_full_of_emoji_is_clipped_by_the_units_discord_counts(
    cog_type: _CogType, post: _PostFactory
) -> None:
    """Discord prices an emoji at the two UTF-16 units it costs, where `len` sees one."""
    embeds = await _expand(cog_type=cog_type, outcome=post(text="🐈" * 4200))

    description = embeds[0].description
    assert description is not None
    assert utf16_length(value=description) <= 4096


@default_cards
async def test_images_past_the_cap_are_counted_in_the_footer(
    cog_type: _CogType, post: _PostFactory
) -> None:
    """A gallery post must not scroll the channel, and must say what it left behind."""
    images = [f"https://cdn.test/{n}.jpg" for n in range(POST_CARD_MAX_IMAGES + 3)]

    embeds = await _expand(cog_type=cog_type, outcome=post(image_urls=images))

    assert len(embeds) == POST_CARD_MAX_IMAGES
    footer = embeds[0].footer.text
    assert footer is not None
    assert omitted_images_note(count=3) in footer


@commented_cards
async def test_a_named_comment_gets_its_own_card_outside_the_gallery(
    cog_type: _CogType, post: _PostFactory, comment: Callable[..., PlatformOutput]
) -> None:
    """A different URL is what keeps the comment from being folded in with the pictures."""
    named = comment(
        comment_id="222",
        text="the one linked",
        author_name="Commenter",
        taken_at=datetime(2026, 9, 5, 9, 25, tzinfo=UTC),
    )
    outcome = post(comments=[named], selected_comment_id="222")

    embeds = await _expand(cog_type=cog_type, outcome=outcome)

    comment_embed = embeds[-1]
    assert comment_embed.description is not None
    assert "the one linked" in comment_embed.description
    assert "Commenter" in (comment_embed.author.name or "")
    assert comment_embed.url not in {None, embeds[0].url}


@commented_cards
async def test_no_comment_card_without_one_named(
    cog_type: _CogType, post: _PostFactory, comment: Callable[..., PlatformOutput]
) -> None:
    """A plain link shows the post alone; the comments the page carried are not the card's."""
    unnamed = comment(comment_id="999", text="some comment", author_name="a")

    embeds = await _expand(cog_type=cog_type, outcome=post(comments=[unnamed]))

    # A comment card carries the comment's own URL; the post and its gallery share the post's.
    assert all(embed.url == embeds[0].url for embed in embeds)


@default_cards
async def test_a_video_post_shows_a_link_instead_of_an_empty_card(
    cog_type: _CogType, post: _PostFactory
) -> None:
    """Nothing here attaches the clip, so the link is the whole of what can be shown."""
    embeds = await _expand(
        cog_type=cog_type, outcome=post(image_urls=[], video_urls=["https://cdn.test/clip.mp4"])
    )

    description = embeds[0].description
    assert description is not None
    assert VIDEO_HINT.partition("{url}")[0] in description


@default_cards
async def test_a_long_post_is_cut_with_a_notice(cog_type: _CogType, post: _PostFactory) -> None:
    """A truncated post must never read as a whole one."""
    embeds = await _expand(cog_type=cog_type, outcome=post(text="x" * 5000))

    description = embeds[0].description
    assert description is not None
    assert len(description) <= 4096
    assert description.endswith(TRUNCATION_NOTICE)


@default_cards
async def test_a_long_post_with_a_video_stays_inside_the_description_limit(
    cog_type: _CogType, post: _PostFactory
) -> None:
    """The hint is reserved before the clip, or Discord rejects the send and the card is lost."""
    embeds = await _expand(
        cog_type=cog_type, outcome=post(text="x" * 5000, video_urls=["https://cdn.test/clip.mp4"])
    )

    description = embeds[0].description
    assert description is not None
    assert len(description) <= 4096
    assert VIDEO_HINT.partition("{url}")[0] in description
    # Without this the same test passes on a clip that truncated silently.
    assert TRUNCATION_NOTICE in description


@commented_cards
async def test_a_long_post_and_a_long_comment_fit_one_message(
    cog_type: _CogType, post: _PostFactory, comment: Callable[..., PlatformOutput]
) -> None:
    """Discord counts every embed in a message together and rejects the whole send when over."""
    named = comment(comment_id="222", text="y" * 4000, author_name="Commenter")
    outcome = post(text="x" * 5000, comments=[named], selected_comment_id="222")

    embeds = await _expand(cog_type=cog_type, outcome=outcome)

    assert sum(embed_text_length(embed=embed) for embed in embeds) <= 6000


@commented_cards
async def test_a_long_comment_under_a_short_post_stays_inside_the_description_limit(
    cog_type: _CogType, post: _PostFactory, comment: Callable[..., PlatformOutput]
) -> None:
    """A short post leaves the comment more of the message than one description may carry."""
    named = comment(comment_id="222", text="y" * 5000, author_name="Commenter")
    outcome = post(text="hi", comments=[named], selected_comment_id="222")

    embeds = await _expand(cog_type=cog_type, outcome=outcome)

    description = embeds[-1].description
    assert description is not None
    assert utf16_length(value=description) <= 4096
    assert description.endswith(TRUNCATION_NOTICE)


@commented_cards
async def test_a_named_comment_takes_its_room_from_the_post(
    cog_type: _CogType, post: _PostFactory, comment: Callable[..., PlatformOutput]
) -> None:
    """A long post gives up part of the message before a named comment is budgeted beside it.

    Measured against the same post with no comment named, which is cut only at the description
    ceiling: without the reservation the post takes that whole ceiling either way and the comment
    card gets only what is left over.
    """
    named = comment(comment_id="222", text="y" * 4000, author_name="Commenter")

    alone = await _expand(cog_type=cog_type, outcome=post(text="x" * 5000))
    beside_comment = await _expand(
        cog_type=cog_type,
        outcome=post(text="x" * 5000, comments=[named], selected_comment_id="222"),
    )

    assert utf16_length(value=beside_comment[0].description or "") < utf16_length(
        value=alone[0].description or ""
    )


def _uploads(payload: DiscordPayload) -> dict[str, bytes]:
    """The non-spacer media files a delivered card attached, by name."""
    return {
        str(file.filename): file.fp.read()
        for file in payload.get("files", [])
        if str(file.filename) != DEFAULT_EMBED_SPACER_FILENAME
    }


@every_card
async def test_card_images_are_uploaded_rather_than_linked(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discord fetches a linked image once, at send, and keeps a failed fetch as an empty image.

    So each embed names an upload carrying its own image's bytes, and the gallery still shares
    the post's URL.
    """
    serve_card_images(monkeypatch=monkeypatch)
    images = [f"https://cdn.test/{n}.jpg" for n in range(3)]
    outcome = post(image_urls=images)
    assert outcome.target is not None

    message, _ = await expand(cog_type=cog_type, outcome=outcome)

    payload = expansion_payload(message=message)
    uploads = _uploads(payload=payload)
    shown = [embed.image.url or "" for embed in payload["embeds"]]
    assert all(url.startswith("attachment://") for url in shown)
    assert [uploads[url.removeprefix("attachment://")].rstrip(b"\0") for url in shown] == [
        image.encode() for image in images
    ]
    assert all(embed.url == outcome.target.url for embed in payload["embeds"])


@every_card
async def test_an_image_that_fails_to_download_keeps_its_link(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One image the CDN refuses costs the card that upload only, never the card."""
    images = [f"https://cdn.test/{n}.jpg" for n in range(3)]
    serve_card_images(monkeypatch=monkeypatch, refused=frozenset({images[1]}))

    embeds = await _expand(cog_type=cog_type, outcome=post(image_urls=images))

    shown = [embed.image.url or "" for embed in embeds]
    assert shown[1] == images[1]
    assert all(url.startswith("attachment://") for url in (shown[0], shown[2]))


@every_card
async def test_a_channel_refusing_attachments_keeps_every_link(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There a card carrying files is refused whole, so nothing is downloaded for it."""
    asked = serve_card_images(monkeypatch=monkeypatch)
    monkeypatch.setattr(
        "discordbot.utils.expansion_cog.target_allows_file_uploads", lambda target: False
    )
    images = [f"https://cdn.test/{n}.jpg" for n in range(3)]

    embeds = await _expand(cog_type=cog_type, outcome=post(image_urls=images))

    assert [embed.image.url for embed in embeds] == images
    assert asked == []


@pytest.mark.parametrize(
    argnames=("status", "content_type"),
    argvalues=[(200, "text/html; charset=utf-8"), (403, "image/jpeg")],
    ids=["not-an-image", "refused"],
)
def test_a_card_image_download_refuses_what_is_not_a_served_image(
    monkeypatch: pytest.MonkeyPatch, status: int, content_type: str
) -> None:
    """An error page is never uploaded under an image's name; the card keeps the link instead."""

    class _FakeResponse:
        status_code = status

        def __init__(self) -> None:
            self.headers = {"Content-Type": content_type}

        def raise_for_status(self) -> None:
            if status >= 400:
                raise requests.HTTPError(f"{status}")

        def iter_content(self, chunk_size: int) -> list[bytes]:
            del chunk_size
            return [b"<html></html>"]

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(requests, "get", lambda url, timeout, stream: _FakeResponse())

    with pytest.raises((requests.HTTPError, ValueError)):
        fetch_card_image(url="https://cdn.test/0.jpg")


@pytest.mark.parametrize(
    argnames=("headers", "chunks"),
    argvalues=[
        pytest.param(
            {"Content-Type": "image/jpeg", "Content-Length": "100"},
            [b"x" * 100],
            id="declared-content-length",
        ),
        pytest.param({"Content-Type": "image/jpeg"}, [b"x" * 6, b"y" * 6], id="streamed-body"),
    ],
)
def test_fetch_card_image_rejects_oversize_payloads(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], chunks: list[bytes]
) -> None:
    """Both declared `Content-Length` and chunked bodies past `max_bytes` raise `ValueError`."""
    streamed = False

    class _FakeResponse:
        status_code = 200

        def __init__(self) -> None:
            self.headers = headers

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int) -> list[bytes]:
            nonlocal streamed
            del chunk_size
            streamed = True
            return chunks

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(requests, "get", lambda url, timeout, stream: _FakeResponse())

    with pytest.raises(ValueError, match="10"):
        fetch_card_image(url="https://cdn.test/huge.jpg", max_bytes=10)

    if "Content-Length" in headers:
        assert streamed is False


async def test_attach_card_images_short_circuits_without_slots_or_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no attachment slots or upload headroom remain, card images are not even downloaded."""
    asked = serve_card_images(monkeypatch=monkeypatch)
    cog = TwitterCogs(bot=stub_bot())
    embed = Embed(url="https://x.com/a/status/1")
    embed.set_image(url="https://cdn.test/img.jpg")

    full_slots = ExpansionDelivery(
        embeds=[embed],
        media=[
            MediaItem(source=b"x", filename=f"clip_{index}.mp4")
            for index in range(DISCORD_ATTACHMENT_LIMIT)
        ],
    )
    message = guild_message(content="https://x.com/a/status/1")
    await cog._attach_card_images(
        message=as_message(fake=message), url="https://x.com/a/status/1", delivery=full_slots
    )
    assert asked == []
    assert embed.image.url == "https://cdn.test/img.jpg"

    no_headroom = ExpansionDelivery(
        embeds=[embed], media=[MediaItem(source=b"x" * 100, filename="clip.mp4")]
    )
    tight_message = guild_message(
        content="https://x.com/a/status/1", filesize_limit=MEDIA_ENVELOPE_MARGIN + 100
    )
    await cog._attach_card_images(
        message=as_message(fake=tight_message),
        url="https://x.com/a/status/1",
        delivery=no_headroom,
    )
    assert asked == []
    assert embed.image.url == "https://cdn.test/img.jpg"


async def _expand_with_cog(
    cog_type: _CogType,
    outcome: PlatformConversation[Any],
    filesize_limit: int = 25 * 1024 * 1024,
    serve_dir: Path | None = None,
) -> FakeDiscordMessage:
    """Runs `cog_type` with configurable upload ceiling and optional media hosting."""
    cog, _ = stub_conversation_cog(cog_type=cog_type, outcome=outcome)
    if serve_dir is not None:
        cog.media_delivery = hosting_planner(serve_dir=serve_dir)
    message = guild_message(
        content=SAMPLE_POST_URLS[cog_type.SOURCE], filesize_limit=filesize_limit
    )
    await cog.on_message(message=as_message(fake=message))
    return message


@video_cards
async def test_target_video_is_attached_omitting_video_hint_and_cleaned_up(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A downloaded target video attaches natively, drops `VIDEO_HINT`, and removes its scratch file."""
    clip = "https://cdn.test/clip.mp4"
    calls = serve_card_videos(monkeypatch=monkeypatch)

    message, _ = await expand(cog_type=cog_type, outcome=post(image_urls=[], video_urls=[clip]))

    payload = expansion_payload(message=message)
    uploads = _uploads(payload=payload)
    assert list(uploads.keys()) == ["video_0.mp4"]
    assert uploads["video_0.mp4"].rstrip(b"\0") == clip.encode()
    assert VIDEO_HINT.partition("{url}")[0] not in (payload["embeds"][0].description or "")
    assert "另有" not in (payload["embeds"][0].footer.text or "")
    assert not calls[0]["filepath"].exists()
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


@video_cards
async def test_multi_video_post_attaches_all_clips_and_omits_footer_video_count(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every delivered video attaches in post order with no redundant video count in the footer."""
    clips = [f"https://cdn.test/{index}.mp4" for index in range(3)]
    serve_card_videos(monkeypatch=monkeypatch)

    message, _ = await expand(cog_type=cog_type, outcome=post(image_urls=[], video_urls=clips))

    payload = expansion_payload(message=message)
    uploads = _uploads(payload=payload)
    assert [uploads[f"video_{index}.mp4"].rstrip(b"\0") for index in range(3)] == [
        clip.encode() for clip in clips
    ]
    assert VIDEO_HINT.partition("{url}")[0] not in (payload["embeds"][0].description or "")
    assert "部影片" not in (payload["embeds"][0].footer.text or "")


@video_cards
async def test_oversize_video_falls_back_to_link_and_logs_info(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversize clip degrades to `VIDEO_HINT` at `info` with no traceback instead of failing."""
    clip = "https://cdn.test/huge.mp4"
    serve_card_videos(monkeypatch=monkeypatch, oversize=frozenset({clip}))
    infos = capture_logs(monkeypatch=monkeypatch, level="info")

    message, _ = await expand(cog_type=cog_type, outcome=post(image_urls=[], video_urls=[clip]))

    payload = expansion_payload(message=message)
    assert _uploads(payload=payload) == {}
    assert VIDEO_HINT.partition("{url}")[0] in (payload["embeds"][0].description or "")
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI
    assert any("byte cap" in msg and "_exc_info" not in fields for msg, fields in infos)


@video_cards
async def test_partial_video_failure_delivers_succeeded_clip_and_counts_remaining(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One working clip still attaches while two failed clips yield `VIDEO_HINT` plus `另有 1 部影片`."""
    clips = [f"https://cdn.test/{index}.mp4" for index in range(3)]
    serve_card_videos(
        monkeypatch=monkeypatch, refused=frozenset({clips[1]}), oversize=frozenset({clips[2]})
    )

    message, _ = await expand(cog_type=cog_type, outcome=post(image_urls=[], video_urls=clips))

    payload = expansion_payload(message=message)
    uploads = _uploads(payload=payload)
    description = payload["embeds"][0].description or ""
    assert list(uploads.keys()) == ["video_0.mp4"]
    expected_link = clips[1] if cog_type is TwitterCogs else SAMPLE_POST_URLS[cog_type.SOURCE]
    assert VIDEO_HINT.format(url=expected_link) in description
    assert clips[0] not in description
    assert omitted_videos_note(count=1) in (payload["embeds"][0].footer.text or "")


@video_cards
async def test_oversize_video_is_hosted_when_media_hosting_is_available(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With media hosting active, download cap rises to 100 MiB and oversize clips publish URLs."""
    clip = "https://cdn.test/big.mp4"
    guild_limit = 2 * 1024 * 1024
    calls = serve_card_videos(monkeypatch=monkeypatch, size=guild_limit + 512)

    message = await _expand_with_cog(
        cog_type=cog_type,
        outcome=post(image_urls=[], video_urls=[clip]),
        filesize_limit=guild_limit,
        serve_dir=tmp_path,
    )

    payload = expansion_payload(message=message)
    assert calls[0]["max_bytes"] == cog_type.VIDEO_DOWNLOAD_MAX_BYTES
    assert _uploads(payload=payload) == {}
    content = payload.get("content") or ""
    assert content.startswith("https://media.test/")
    assert content.endswith(".mp4")
    assert VIDEO_HINT.partition("{url}")[0] not in (payload["embeds"][0].description or "")


@video_cards
async def test_without_media_hosting_download_cap_matches_native_upload_room(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When hosting is off, never download more than the guild's native attachment room."""
    clip = "https://cdn.test/fit.mp4"
    guild_limit = 8 * 1024 * 1024
    calls = serve_card_videos(monkeypatch=monkeypatch)

    await _expand_with_cog(
        cog_type=cog_type,
        outcome=post(image_urls=[], video_urls=[clip]),
        filesize_limit=guild_limit,
    )

    assert calls[0]["max_bytes"] == guild_limit - MEDIA_ENVELOPE_MARGIN


@video_cards
async def test_channel_refusing_uploads_without_hosting_skips_video_download(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without upload permission or hosting, nothing is downloaded and `VIDEO_HINT` remains."""
    calls = serve_card_videos(monkeypatch=monkeypatch)
    monkeypatch.setattr(
        "discordbot.utils.expansion_cog.target_allows_file_uploads", lambda target: False
    )
    monkeypatch.setattr(
        "discordbot.utils.discord_embeds.target_allows_file_uploads", lambda target: False
    )

    message, _ = await expand(
        cog_type=cog_type, outcome=post(image_urls=[], video_urls=["https://cdn.test/clip.mp4"])
    )

    payload = expansion_payload(message=message)
    assert calls == []
    assert payload.get("files", []) == []
    assert VIDEO_HINT.partition("{url}")[0] in (payload["embeds"][0].description or "")


@video_cards
async def test_channel_refusing_uploads_with_hosting_hosts_video_without_attaching(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without Attach Files permission, an enabled media host still delivers the clip by URL."""
    serve_card_videos(monkeypatch=monkeypatch)
    monkeypatch.setattr(
        "discordbot.utils.expansion_cog.target_allows_file_uploads", lambda target: False
    )
    monkeypatch.setattr(
        "discordbot.utils.discord_embeds.target_allows_file_uploads", lambda target: False
    )

    message = await _expand_with_cog(
        cog_type=cog_type,
        outcome=post(image_urls=[], video_urls=["https://cdn.test/clip.mp4"]),
        serve_dir=tmp_path,
    )

    payload = expansion_payload(message=message)
    assert payload.get("files", []) == []
    assert (payload.get("content") or "").startswith("https://media.test/")
    assert VIDEO_HINT.partition("{url}")[0] not in (payload["embeds"][0].description or "")


@video_cards
async def test_more_than_ten_videos_caps_downloads_and_shows_clamped_notice(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past Discord's ten-attachment ceiling, trailing videos are skipped with a clamped notice."""
    clips = [f"https://cdn.test/{index}.mp4" for index in range(DISCORD_ATTACHMENT_LIMIT + 2)]
    calls = serve_card_videos(monkeypatch=monkeypatch)

    message, _ = await expand(cog_type=cog_type, outcome=post(image_urls=[], video_urls=clips))

    payload = expansion_payload(message=message)
    assert len(calls) == DISCORD_ATTACHMENT_LIMIT
    assert len(payload.get("files", [])) == DISCORD_ATTACHMENT_LIMIT
    assert "已省略 2 部影片" in (payload.get("content") or "")
    assert VIDEO_HINT.partition("{url}")[0] not in (payload["embeds"][0].description or "")
    assert "部影片" not in (payload["embeds"][0].footer.text or "")


@video_cards
async def test_mixed_photo_and_video_post_uploads_both_within_shared_budget(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Card image upload shares attachment slots and byte room cleanly with target videos."""
    images = [f"https://cdn.test/{index}.jpg" for index in range(2)]
    clips = [f"https://cdn.test/{index}.mp4" for index in range(2)]
    serve_card_images(monkeypatch=monkeypatch)
    serve_card_videos(monkeypatch=monkeypatch)

    message, _ = await expand(cog_type=cog_type, outcome=post(image_urls=images, video_urls=clips))

    payload = expansion_payload(message=message)
    uploads = _uploads(payload=payload)
    assert {"video_0.mp4", "video_1.mp4"} < set(uploads)
    assert len(uploads) == 4
    assert all((embed.image.url or "").startswith("attachment://") for embed in payload["embeds"])
    assert VIDEO_HINT.partition("{url}")[0] not in (payload["embeds"][0].description or "")


@video_cards
async def test_card_images_stay_linked_when_videos_consume_remaining_upload_room(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_attach_card_images` subtracts delivered video bytes so images never bust the message cap."""
    image = "https://cdn.test/0.jpg"
    clip = "https://cdn.test/0.mp4"
    serve_card_images(monkeypatch=monkeypatch, size=100)
    serve_card_videos(monkeypatch=monkeypatch, size=150)

    message = await _expand_with_cog(
        cog_type=cog_type,
        outcome=post(image_urls=[image], video_urls=[clip]),
        filesize_limit=MEDIA_ENVELOPE_MARGIN + 200,
    )

    payload = expansion_payload(message=message)
    assert list(_uploads(payload=payload).keys()) == ["video_0.mp4"]
    assert payload["embeds"][0].image.url == image


@video_cards
async def test_combined_video_overflow_drops_excess_and_falls_back_to_link(
    cog_type: _CogType, post: _PostFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two clips that fit alone but overflow combined with hosting off keep one clip and link the dropped one."""
    clips = ["https://cdn.test/0.mp4", "https://cdn.test/1.mp4"]
    serve_card_videos(monkeypatch=monkeypatch, sizes={clips[0]: 100, clips[1]: 150})

    message = await _expand_with_cog(
        cog_type=cog_type,
        outcome=post(image_urls=[], video_urls=clips),
        filesize_limit=MEDIA_ENVELOPE_MARGIN + 200,
    )

    payload = expansion_payload(message=message)
    uploads = _uploads(payload=payload)
    description = payload["embeds"][0].description or ""
    assert list(uploads.keys()) == ["video_0.mp4"]
    expected_link = clips[1] if cog_type is TwitterCogs else SAMPLE_POST_URLS[cog_type.SOURCE]
    assert VIDEO_HINT.format(url=expected_link) in description
    assert "部影片" not in (payload["embeds"][0].footer.text or "")


async def test_video_download_timeout_degrades_to_link_without_failing_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled video download times out into the link fallback rather than failing the card."""
    cog, _ = stub_conversation_cog(
        cog_type=InstagramCogs,
        outcome=instagram_post(image_urls=[], video_urls=["https://cdn.test/slow.mp4"]),
    )

    async def stall(**kwargs: object) -> list[Path]:
        """Simulates a download exceeding `VIDEO_DOWNLOAD_TIMEOUT_SECONDS`."""
        del kwargs
        raise TimeoutError

    cog.__dict__["_download_videos"] = stall
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    message = guild_message(content=SAMPLE_POST_URLS["instagram"])

    await cog.on_message(message=as_message(fake=message))

    payload = expansion_payload(message=message)
    assert _uploads(payload=payload) == {}
    assert VIDEO_HINT.partition("{url}")[0] in (payload["embeds"][0].description or "")
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI
    assert any("timed out" in msg for msg, _ in warns)


async def test_download_videos_defaults_to_off_on_base_conversation_cog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`DOWNLOAD_VIDEOS` stays opt-in so a subclass that leaves it unset keeps link-only cards."""

    class _OptOutConversationCog(ConversationExpansionCog[FacebookOutput, FacebookConversation]):
        SOURCE = FacebookCogs.SOURCE
        PLATFORM = FacebookCogs.PLATFORM
        POST_LINK = FacebookCogs.POST_LINK
        PLACEHOLDER_TEXT = FacebookCogs.PLACEHOLDER_TEXT
        READ_TIMEOUT_SECONDS = FacebookCogs.READ_TIMEOUT_SECONDS
        EMBED_COLOR = FacebookCogs.EMBED_COLOR
        downloader_factory = FacebookCogs.downloader_factory

        def _footer_text(self, post: FacebookOutput, shown_images: int) -> str:
            del post, shown_images
            return ""

    calls = serve_card_videos(monkeypatch=monkeypatch)

    message, _ = await expand(
        cog_type=_OptOutConversationCog,
        outcome=facebook_post(image_urls=[], video_urls=["https://cdn.test/clip.mp4"]),
    )

    assert calls == []
    assert VIDEO_HINT.partition("{url}")[0] in (
        expansion_embeds(message=message)[0].description or ""
    )


def test_expansion_cog_init_upgrades_reply_db_for_pending_expansions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initializing any `ExpansionCog` migrates `reply.db` so `pending_expansion` exists."""
    (tmp_path / "alembic.ini").symlink_to(target=REPO_ROOT / "alembic.ini")
    (tmp_path / "migrations").symlink_to(target=REPO_ROOT / "migrations")
    monkeypatch.chdir(path=tmp_path)

    TwitterCogs(bot=stub_bot())

    db_path = tmp_path / "data" / "database" / "reply.db"
    assert db_path.is_file()
    with sqlite3.connect(database=db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert "pending_expansion" in tables
