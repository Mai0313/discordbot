"""Tests for the cog that auto-expands Instagram links pasted into a channel.

The downloader is stubbed at `downloader_factory`, so nothing here touches the network or the
parser: what these cover is the cog's own decisions — when it fires, what it renders, and what
a reader sees when the post cannot be read.
"""

from types import SimpleNamespace
from datetime import UTC, datetime

from nextcord import Embed

from discordbot.typings.emojis import INSTAGRAM_EMOJI
from discordbot.utils.instagram import InstagramOutput, InstagramConversation
from discordbot.utils.discord_embeds import utf16_length
from discordbot.cogs.parse_instagram.cog import InstagramCogs
from discordbot.utils.expansion_placeholder import EXPANSION_UNREADABLE_EMOJI

from tests.helpers.casting import as_bot, as_message
from tests.helpers.discord_mocks import (
    FakeUser,
    FakeDiscordMessage,
    expansion_payload,
    placeholder_withdrawn,
)

_CODE = "Dc5eNjYkoZE"
_URL = f"https://www.instagram.com/p/{_CODE}/"
_GREEN = "<:greencheck:1517565102424068226>"
_RED = "<:redcross:1517565100838355016>"


def _post(**overrides: object) -> InstagramConversation:
    """A readable conversation, with any post or conversation field overridden per test."""
    raw_comments = overrides.pop("comments", [])
    comments = raw_comments if isinstance(raw_comments, list) else []
    selected = overrides.pop("selected_comment_id", "")
    fields: dict[str, object] = {
        "url": _URL,
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
        reply_branches=[[comment] for comment in comments],  # ty: ignore[invalid-argument-type]
        selected_comment_id=selected,  # ty: ignore[invalid-argument-type]
    )


class _StubDownloader:
    """Stands in for InstagramDownloader, serving one canned outcome."""

    def __init__(self, *, post: InstagramConversation | None, error: Exception | None) -> None:
        """Records what the cog asked for and answers with the canned outcome."""
        self.post = post
        self.error = error
        self.seen: list[str] = []

    def parse_metadata(self, *, url: str) -> InstagramConversation:
        """Answers with the canned conversation, or raises the canned error."""
        self.seen.append(url)
        if self.error is not None:
            raise self.error
        return self.post if self.post is not None else InstagramConversation()


class _InstagramMessage(FakeDiscordMessage):
    """Adds the author/content/guild fields `InstagramCogs.on_message` reads."""

    def __init__(self, author: FakeUser, content: str, guild: object) -> None:
        """Builds a message double carrying the fields the cog inspects."""
        super().__init__()
        self.author = author
        self.content = content
        self.guild = guild


def _message(content: str = _URL) -> _InstagramMessage:
    """Builds a guild message carrying an Instagram link."""
    return _InstagramMessage(
        author=FakeUser(bot=False), content=content, guild=SimpleNamespace(id=100)
    )


def _cog(
    *, post: InstagramConversation | None = None, error: Exception | None = None, bot_id: int = 999
) -> tuple[InstagramCogs, dict[str, _StubDownloader]]:
    """Builds a cog wired to a stub downloader."""
    cog = InstagramCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=bot_id))))
    made: dict[str, _StubDownloader] = {}

    def factory() -> _StubDownloader:
        """Records the stub so a test can assert on what it was asked to do."""
        stub = _StubDownloader(post=post, error=error)
        made["stub"] = stub
        return stub

    cog.__dict__["downloader_factory"] = factory
    return cog, made


def _embeds(message: _InstagramMessage) -> list[Embed]:
    """The embeds the cog delivered onto its placeholder."""
    return list(expansion_payload(message=message)["embeds"])


async def test_a_pasted_link_is_expanded_into_a_card() -> None:
    """The ordinary case: one post embed carrying the caption, the author and the counters."""
    cog, made = _cog(post=_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert made["stub"].seen == [_URL]
    assert message.suppressed
    embeds = _embeds(message)
    assert embeds[0].description == "post body"
    assert embeds[0].url == _URL
    assert message.reactions[-1] == _GREEN


async def test_the_author_line_carries_both_names() -> None:
    """Instagram identifies people by handle, but the display name is what a reader knows."""
    cog, _ = _cog(post=_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert _embeds(message)[0].author.name == "晏凌 (@c_cylynn)"


async def test_a_post_without_a_display_name_falls_back_to_the_handle() -> None:
    """Not every account sets one, and a bare `@handle` is still a complete label."""
    cog, _ = _cog(post=_post(author_full_name=""))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert _embeds(message)[0].author.name == "@c_cylynn"


async def test_the_footer_carries_the_counters() -> None:
    """Likes and comments are what Instagram reports; there is no share figure."""
    cog, _ = _cog(post=_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = _embeds(message)[0].footer.text
    assert footer is not None
    assert "❤️ 8,855" in footer
    assert "💬 11" in footer


async def test_extra_images_become_embeds_sharing_the_post_url() -> None:
    """Sharing the URL is what makes Discord merge them into one gallery under the post."""
    cog, _ = _cog(post=_post(image_urls=[f"https://instagram.example/{n}.jpg" for n in range(3)]))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    embeds = _embeds(message)
    assert len(embeds) == 3
    assert all(embed.url == _URL for embed in embeds)


async def test_images_past_the_cap_are_counted_in_the_footer() -> None:
    """A nine-image carousel is the ordinary Instagram post, so this cap really binds."""
    cog, _ = _cog(post=_post(image_urls=[f"https://instagram.example/{n}.jpg" for n in range(9)]))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    embeds = _embeds(message)
    assert len(embeds) == 4
    footer = embeds[0].footer.text
    assert footer is not None
    assert "另有 5 張" in footer


async def test_a_named_comment_gets_its_own_card_outside_the_gallery() -> None:
    """A different URL is what keeps the comment from being folded in with the pictures."""
    comment = InstagramOutput(
        comment_id="17946527169275440",
        text="the one linked",
        author_name="xiao_pang0704",
        taken_at=datetime(2026, 9, 5, 9, 25, tzinfo=UTC),
    )
    cog, _ = _cog(post=_post(comments=[comment], selected_comment_id="17946527169275440"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    comment_embed = _embeds(message)[-1]
    assert comment_embed.description is not None
    assert "the one linked" in comment_embed.description
    assert comment_embed.author.name == "@xiao_pang0704"
    assert comment_embed.url == f"{_URL}c/17946527169275440/"
    assert comment_embed.url != _URL


async def test_no_comment_card_without_one_named() -> None:
    """A plain link shows the post alone, the same rule Threads and Facebook follow."""
    comment = InstagramOutput(comment_id="999", text="some comment", author_name="a")
    cog, _ = _cog(post=_post(comments=[comment]))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert all("指定的留言" not in (embed.description or "") for embed in _embeds(message))


async def test_a_video_post_shows_a_link_instead_of_an_empty_card() -> None:
    """The cog attaches nothing, so a Reel is its caption plus a link to watch it."""
    cog, _ = _cog(post=_post(image_urls=[], video_urls=["https://instagram.example/clip.mp4"]))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = _embeds(message)[0].description
    assert description is not None
    assert "點此觀看影片" in description


async def test_a_long_post_with_a_video_stays_inside_the_description_limit() -> None:
    """The hint is reserved before the clip, or Discord rejects the send and the card is lost."""
    cog, _ = _cog(post=_post(text="x" * 5000, video_urls=["https://instagram.example/clip.mp4"]))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = _embeds(message)[0].description
    assert description is not None
    assert len(description) <= 4096
    assert "點此觀看影片" in description
    # Without this the same test passes on a clip that truncated silently.
    assert "（全文請看原貼文）" in description


async def test_a_long_post_and_a_long_comment_fit_one_message() -> None:
    """Discord counts every embed in a message together and rejects the whole send when over."""
    comment = InstagramOutput(comment_id="222", text="y" * 4000, author_name="c")
    cog, _ = _cog(post=_post(text="x" * 5000, comments=[comment], selected_comment_id="222"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    total = sum(
        len(embed.description or "") + len(embed.footer.text or "") + len(embed.author.name or "")
        for embed in _embeds(message)
    )
    assert total <= 6000


async def test_the_read_marker_rides_beside_the_status_chain() -> None:
    """The platform marker says a post was read and is never taken back by the chain."""
    cog, _ = _cog(post=_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[0] == INSTAGRAM_EMOJI
    assert all(emoji != INSTAGRAM_EMOJI for emoji, _ in message.removed)


async def test_a_comment_permalink_is_expanded_too() -> None:
    """The URL the user pastes is the comment's, and the cog reads the post behind it."""
    cog, made = _cog(post=_post())
    url = f"{_URL}c/17946527169275440/?img_index=1"
    message = _message(content=f"看這個 {url}")

    await cog.on_message(message=as_message(fake=message))

    assert made["stub"].seen == [url]
    assert message.reactions[-1] == _GREEN


async def test_a_message_addressed_to_the_bot_is_left_alone() -> None:
    """A mention hands the link to gen_reply, so the cog must not fetch anything."""
    cog, made = _cog(post=_post())
    message = _message(content=f"<@999> what is this {_URL}")

    await cog.on_message(message=as_message(fake=message))

    assert made == {}
    assert message.reactions == []


async def test_a_profile_url_is_ignored_silently() -> None:
    """A profile link is not a failure, so it earns no reaction at all."""
    cog, made = _cog(post=_post())
    message = _message(content="look https://www.instagram.com/c_cylynn/")

    await cog.on_message(message=as_message(fake=message))

    assert made == {}
    assert message.reactions == []


async def test_a_bot_message_is_ignored() -> None:
    """The bot's own expansion carries the link, and must not expand it again."""
    cog, made = _cog(post=_post())
    message = _message()
    message.author = FakeUser(bot=True)

    await cog.on_message(message=as_message(fake=message))

    assert made == {}


async def test_an_unreadable_post_is_marked_read_but_unshowable() -> None:
    """A private account is the post's own state, so it earns the unreadable mark, not the cross.

    The cross says the bot broke. Nothing did: the page came back and carries nothing
    showable, which is what every other expansion cog answers ⚠️ for.
    """
    cog, _ = _cog(post=InstagramConversation())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert placeholder_withdrawn(message=message)
    assert message.reactions[-1] == EXPANSION_UNREADABLE_EMOJI


async def test_a_parse_failure_is_marked_failed_without_a_message() -> None:
    """A fetch failure reaches the user as a reaction and nothing else."""
    cog, _ = _cog(error=RuntimeError("boom"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert placeholder_withdrawn(message=message)
    assert message.reactions[-1] == _RED


async def test_the_video_link_points_at_the_post_not_the_expiring_cdn_url() -> None:
    """`video_versions[0].url` is signed and dies within days; the embed carrying it does not."""
    clip = "https://instagram.example/clip.mp4?oe=DEADBEEF"
    cog, _ = _cog(post=_post(image_urls=[], video_urls=[clip]))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = _embeds(message)[0].description
    assert description is not None
    assert clip not in description
    assert _URL in description


async def test_a_mixed_carousel_counts_the_videos_nothing_linked() -> None:
    """Counting images alone reported four clips as nothing at all."""
    cog, _ = _cog(
        post=_post(
            image_urls=[f"https://instagram.example/{n}.jpg" for n in range(5)],
            video_urls=[f"https://instagram.example/{n}.mp4" for n in range(5)],
        )
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = _embeds(message)[0].footer.text
    assert footer is not None
    assert "🖼️ 另有 1 張" in footer
    assert "🎬 另有 4 部影片" in footer


async def test_a_post_full_of_emoji_is_clipped_by_the_units_discord_counts() -> None:
    """Discord prices an emoji at the two UTF-16 units it costs, where `len` sees one."""
    cog, _ = _cog(post=_post(text="🐈" * 4200))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = _embeds(message)[0].description
    assert description is not None
    assert utf16_length(value=description) <= 4096


async def test_a_post_whose_author_hid_its_likes_shows_no_like_count() -> None:
    """Instagram serves `-1` there rather than omitting the field."""
    cog, _ = _cog(post=_post(like_count=-1))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = _embeds(message)[0].footer.text
    assert footer is not None
    assert "❤️" not in footer


async def test_an_author_with_only_a_display_name_still_gets_a_line() -> None:
    """Dropping the line entirely would read as an anonymous post."""
    cog, _ = _cog(post=_post(author_name="", author_full_name="晏凌"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert _embeds(message)[0].author.name == "晏凌"
