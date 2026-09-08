"""Tests for the cog that auto-expands Facebook links pasted into a channel.

The downloader is stubbed at `downloader_factory`, so nothing here touches the network or the
parser: what these cover is the cog's own decisions — when it fires, what it renders, and what
a reader sees when the post cannot be read.
"""

from types import SimpleNamespace
from typing import Any, cast
from datetime import UTC, datetime

from nextcord import Embed

from discordbot.typings.emojis import FACEBOOK_EMOJI
from discordbot.utils.facebook import FacebookOutput, FacebookConversation
from discordbot.utils.link_errors import LinkRetryableError
from discordbot.utils.discord_embeds import utf16_length
from discordbot.cogs.parse_facebook.cog import FacebookCogs
from discordbot.utils.expansion_placeholder import (
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
)

from tests.helpers.casting import as_bot, as_message, make_forbidden
from tests.helpers.discord_mocks import (
    FakeUser,
    FakeDiscordMessage,
    expansion_payload,
    placeholder_withdrawn,
)

_POST_ID = "1730774811333135"
_GROUP_ID = "1176671326743489"
_URL = f"https://www.facebook.com/groups/{_GROUP_ID}/posts/{_POST_ID}/"
_GREEN = "<:greencheck:1517565102424068226>"
_RED = "<:redcross:1517565100838355016>"


def _post(**overrides: object) -> FacebookConversation:
    """A readable conversation, with any post or conversation field overridden per test."""
    raw_comments = overrides.pop("comments", [])
    comments = raw_comments if isinstance(raw_comments, list) else []
    selected = overrides.pop("selected_comment_id", "")
    fields: dict[str, object] = {
        "url": _URL,
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
        reply_branches=[[comment] for comment in comments],  # ty: ignore[invalid-argument-type]
        selected_comment_id=selected,  # ty: ignore[invalid-argument-type]
    )


class _StubDownloader:
    """Stands in for FacebookDownloader, serving one canned outcome."""

    def __init__(self, *, post: FacebookConversation | None, error: Exception | None) -> None:
        """Records what the cog asked for and answers with the canned outcome."""
        self.post = post
        self.error = error
        self.seen: list[str] = []

    def parse_metadata(self, *, url: str) -> FacebookConversation:
        """Answers with the canned post, or raises the canned error."""
        self.seen.append(url)
        if self.error is not None:
            raise self.error
        return self.post if self.post is not None else FacebookConversation()


class _FacebookMessage(FakeDiscordMessage):
    """Adds the author/content/guild fields `FacebookCogs.on_message` reads."""

    def __init__(self, author: FakeUser, content: str, guild: object) -> None:
        """Builds a message double carrying the fields the cog inspects."""
        super().__init__()
        self.author = author
        self.content = content
        self.guild = guild


def _message(content: str = _URL) -> _FacebookMessage:
    """Builds a guild message carrying a Facebook link."""
    return _FacebookMessage(
        author=FakeUser(bot=False), content=content, guild=SimpleNamespace(id=100)
    )


def _cog(
    *, post: FacebookConversation | None = None, error: Exception | None = None, bot_id: int = 999
) -> tuple[FacebookCogs, dict[str, _StubDownloader]]:
    """Builds a cog wired to a stub downloader."""
    cog = FacebookCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=bot_id))))
    made: dict[str, _StubDownloader] = {}

    def factory() -> _StubDownloader:
        """Records the stub so a test can assert on what it was asked to do."""
        stub = _StubDownloader(post=post, error=error)
        made["stub"] = stub
        return stub

    cog.__dict__["downloader_factory"] = factory
    return cog, made


def _embeds(message: _FacebookMessage) -> list[Embed]:
    """The embeds the cog delivered onto its placeholder."""
    return list(expansion_payload(message=message)["embeds"])


async def test_a_pasted_link_is_expanded_into_a_card() -> None:
    """The ordinary case: one post embed carrying the body, the author and the counters."""
    cog, made = _cog(post=_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert made["stub"].seen == [_URL]
    assert message.suppressed
    embeds = _embeds(message)
    assert embeds[0].description == "post body"
    assert embeds[0].author.name == "Somebody"
    assert embeds[0].url == _URL
    assert message.reactions[-1] == _GREEN


async def test_the_footer_names_the_group_and_the_counters() -> None:
    """The group is the part a reader cannot get from the post itself, so it leads."""
    cog, _ = _cog(post=_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = _embeds(message)[0].footer.text
    assert footer is not None
    assert footer.startswith("Some Group")
    assert "👍 1,017" in footer
    assert "💬 40" in footer
    assert "↗️ 37" in footer


async def test_a_page_post_leaves_the_group_out_of_the_footer() -> None:
    """A page post has no group, and the line must not carry a hole where one would go."""
    cog, _ = _cog(post=_post(group_name=""))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = _embeds(message)[0].footer.text
    assert footer is not None
    assert footer.startswith("👍 1,017")


async def test_extra_images_become_embeds_sharing_the_post_url() -> None:
    """Sharing the URL is what makes Discord merge them into one gallery under the post."""
    cog, _ = _cog(post=_post(image_urls=[f"https://scontent.example/{n}.jpg" for n in range(3)]))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    embeds = _embeds(message)
    assert len(embeds) == 3
    assert all(embed.url == _URL for embed in embeds)
    assert [embed.image.url for embed in embeds] == [
        "https://scontent.example/0.jpg",
        "https://scontent.example/1.jpg",
        "https://scontent.example/2.jpg",
    ]


async def test_images_past_the_cap_are_counted_in_the_footer() -> None:
    """A gallery post must not scroll the channel, and must say what it left behind."""
    cog, _ = _cog(post=_post(image_urls=[f"https://scontent.example/{n}.jpg" for n in range(7)]))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    embeds = _embeds(message)
    assert len(embeds) == 4
    footer = embeds[0].footer.text
    assert footer is not None
    assert "另有 3 張" in footer


async def test_a_named_comment_gets_its_own_card_outside_the_gallery() -> None:
    """A different URL is what keeps the comment from being folded in with the pictures."""
    comment = FacebookOutput(
        comment_id="1730777104666239",
        text="the one linked",
        author_name="Commenter",
        taken_at=datetime(2026, 9, 5, 9, 25, tzinfo=UTC),
    )
    cog, _ = _cog(post=_post(comments=[comment], selected_comment_id="1730777104666239"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    embeds = _embeds(message)
    comment_embed = embeds[-1]
    assert comment_embed.description is not None
    assert "the one linked" in comment_embed.description
    assert comment_embed.author.name == "Commenter"
    assert comment_embed.url != _URL
    assert "comment_id=1730777104666239" in (comment_embed.url or "")


async def test_no_comment_card_without_one_named() -> None:
    """A plain link shows the post alone; the preloaded comments are not the whole section."""
    comment = FacebookOutput(comment_id="999", text="some comment")
    cog, _ = _cog(post=_post(comments=[comment]))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert all("指定的留言" not in (embed.description or "") for embed in _embeds(message))


async def test_a_video_post_shows_a_link_instead_of_an_empty_card() -> None:
    """There is no file to attach logged out, so the link is the whole of what can be shown."""
    cog, _ = _cog(post=_post(image_urls=[], video_urls=["https://www.facebook.com/watch/?v=1"]))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = _embeds(message)[0].description
    assert description is not None
    assert "點此觀看影片" in description


async def test_a_long_post_is_cut_with_a_notice() -> None:
    """A truncated post must never read as a whole one."""
    cog, _ = _cog(post=_post(text="x" * 5000))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = _embeds(message)[0].description
    assert description is not None
    assert len(description) <= 4096
    assert description.endswith("（全文請看原貼文）")


async def test_the_read_marker_rides_beside_the_status_chain() -> None:
    """The platform marker says a post was read and is never taken back by the chain."""
    cog, _ = _cog(post=_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[0] == FACEBOOK_EMOJI
    assert all(emoji != FACEBOOK_EMOJI for emoji, _ in message.removed)


async def test_a_message_addressed_to_the_bot_is_left_alone() -> None:
    """A mention hands the link to gen_reply, so the cog must not fetch anything."""
    cog, made = _cog(post=_post())
    message = _message(content=f"<@999> what is this {_URL}")

    await cog.on_message(message=as_message(fake=message))

    assert made == {}
    assert message.reactions == []


async def test_a_url_that_names_no_post_is_ignored_silently() -> None:
    """A profile link is not a failure, so it earns no reaction at all."""
    cog, made = _cog(post=_post())
    message = _message(content="look https://www.facebook.com/NASA")

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
    """A private or deleted post is the post's own state, so it earns the unreadable mark, not the cross.

    The cross says the bot broke. Nothing did: the page came back and carries nothing
    showable, which is what every other expansion cog answers ⚠️ for.
    """
    cog, _ = _cog(post=FacebookConversation())
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


async def test_a_long_post_with_a_video_stays_inside_the_description_limit() -> None:
    """The hint is reserved before the clip, or Discord rejects the send and the card is lost."""
    cog, _ = _cog(post=_post(text="x" * 5000, video_urls=["https://www.facebook.com/watch/?v=1"]))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = _embeds(message)[0].description
    assert description is not None
    assert len(description) <= 4096
    assert "點此觀看影片" in description


async def test_a_long_post_and_a_long_comment_fit_one_message() -> None:
    """Discord counts every embed in a message together and rejects the whole send when over."""
    comment = FacebookOutput(comment_id="222", text="y" * 4000, author_name="Commenter")
    cog, _ = _cog(post=_post(text="x" * 5000, comments=[comment], selected_comment_id="222"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    total = sum(
        len(embed.description or "") + len(embed.footer.text or "") + len(embed.author.name or "")
        for embed in _embeds(message)
    )
    assert total <= 6000


async def test_the_comment_link_joins_an_existing_query_correctly() -> None:
    """A `permalink.php` post URL already carries a query, so a second `?` breaks the link."""
    url = "https://www.facebook.com/permalink.php?story_fbid=1&id=2"
    comment = FacebookOutput(comment_id="222", text="linked", author_name="C")
    cog, _ = _cog(post=_post(url=url, comments=[comment], selected_comment_id="222"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    comment_url = _embeds(message)[-1].url
    assert comment_url is not None
    assert comment_url.count("?") == 1
    assert comment_url.endswith("&comment_id=222")


async def test_a_post_full_of_emoji_is_clipped_by_the_units_discord_counts() -> None:
    """A Facebook post runs to 63,206 characters, so an emoji-heavy one reaches this easily."""
    cog, _ = _cog(post=_post(text="🐈" * 4200))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = _embeds(message)[0].description
    assert description is not None
    assert utf16_length(value=description) <= 4096


async def test_an_album_counts_the_videos_nothing_linked() -> None:
    """Only the first video gets a hint, so the rest would go unmentioned."""
    cog, _ = _cog(
        post=_post(video_urls=[f"https://www.facebook.com/watch/?v={n}" for n in range(3)])
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = _embeds(message)[0].footer.text
    assert footer is not None
    assert "🎬 另有 2 部影片" in footer


async def test_the_reply_slot_is_claimed_before_any_reaction_goes_on() -> None:
    """The card is what the reader is waiting for, so nothing queues in front of it.

    Both reactions share one per-channel rate-limit bucket that nextcord serializes itself,
    while a message send waits on none of it, so reacting first only delays the placeholder.
    `tests/test_expansion_contract.py` holds the other three cogs to the same order; this is
    the one that proves the order is real rather than a coincidence of source layout.
    """
    cog, _ = _cog(post=_post())
    message = _message()
    reactions_when_claimed: list[str] = []
    claim = message.reply

    async def recording_reply(**kwargs: object) -> object:
        """Snapshots the reaction row at the moment the slot is claimed."""
        reactions_when_claimed.extend(message.reactions)
        return await claim(**cast("Any", kwargs))

    message.reply = recording_reply  # ty: ignore[invalid-assignment]

    await cog.on_message(message=as_message(fake=message))

    assert reactions_when_claimed == []
    assert message.reactions[0] == FACEBOOK_EMOJI


async def test_a_refused_slot_still_says_which_platform_was_detected() -> None:
    """The marker is the diagnostic, so the one channel that cannot show a card keeps it.

    A channel granting Add Reactions but not Send Messages is exactly where someone has to
    work out what went wrong, and the cross alone does not say a Facebook link was even seen.
    """
    cog, _ = _cog(post=_post())
    message = _message()

    async def refuse(**kwargs: object) -> object:
        """Answers the way a channel the bot cannot post in does."""
        del kwargs
        raise make_forbidden()

    message.reply = refuse  # ty: ignore[invalid-assignment]

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions == [FACEBOOK_EMOJI, _RED]


async def test_a_platform_under_load_is_marked_retryable_not_broken() -> None:
    """The whole chain: a refused fetch reaches the channel as ⏱️, never as the cross.

    `tests/test_link_errors.py` proves the reader raises it and
    `tests/test_expansion_contract.py` proves all four cogs map it the same way; this is the
    one test that walks both halves, because a 429 answered as ❌ tells the reader the bot is
    broken when the link is fine and works in a minute.
    """
    cog, _ = _cog(error=LinkRetryableError("429 from Facebook"))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert message.reactions[-1] == EXPANSION_RETRY_LATER_EMOJI
    assert placeholder_withdrawn(message=message)
