"""Tests for the cog that auto-expands Twitter links pasted into a channel.

The downloader is stubbed at `downloader_factory`, so nothing here touches the network or the
parser: what these cover is the cog's own decisions — when it fires, what it renders, and what a
reader sees when the post cannot be read.
"""

from types import SimpleNamespace
from typing import Any, cast
from datetime import UTC, datetime

from nextcord import Embed

from discordbot.typings.emojis import TWITTER_EMOJI
from discordbot.utils.link_errors import LinkRetryableError
from discordbot.utils.discord_embeds import utf16_length
from discordbot.cogs.parse_twitter.cog import TwitterCogs
from discordbot.services.platforms.twitter import TwitterOutput, TwitterConversation
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

_STATUS_ID = "1628549742539194368"
_URL = f"https://x.com/Dbacks/status/{_STATUS_ID}"
_GREEN = "<:greencheck:1517565102424068226>"
_RED = "<:redcross:1517565100838355016>"


def _output(**overrides: object) -> TwitterOutput:
    """One post, with any field overridden per test."""
    fields: dict[str, object] = {
        "url": _URL,
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


def _post(**overrides: object) -> TwitterConversation:
    """A readable conversation carrying one post."""
    return TwitterConversation(chain=[_output(**overrides)])


class _StubDownloader:
    """Stands in for TwitterDownloader, serving one canned outcome."""

    def __init__(self, *, post: TwitterConversation | None, error: Exception | None) -> None:
        """Records what the cog asked for and answers with the canned outcome."""
        self.post = post
        self.error = error
        self.seen: list[str] = []

    def parse_metadata(self, *, url: str) -> TwitterConversation:
        """Answers with the canned post, or raises the canned error."""
        self.seen.append(url)
        if self.error is not None:
            raise self.error
        return self.post if self.post is not None else TwitterConversation()


class _TwitterMessage(FakeDiscordMessage):
    """Adds the author/content/guild fields `TwitterCogs.on_message` reads."""

    def __init__(self, author: FakeUser, content: str, guild: object) -> None:
        """Builds a message double carrying the fields the cog inspects."""
        super().__init__()
        self.author = author
        self.content = content
        self.guild = guild


def _message(content: str = _URL) -> _TwitterMessage:
    """Builds a guild message carrying a Twitter link."""
    return _TwitterMessage(
        author=FakeUser(bot=False), content=content, guild=SimpleNamespace(id=100)
    )


def _cog(
    *, post: TwitterConversation | None = None, error: Exception | None = None, bot_id: int = 999
) -> tuple[TwitterCogs, dict[str, _StubDownloader]]:
    """Builds a cog wired to a stub downloader."""
    cog = TwitterCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=bot_id))))
    made: dict[str, _StubDownloader] = {}

    def factory() -> _StubDownloader:
        """Records the stub so a test can assert on what it was asked to do."""
        stub = _StubDownloader(post=post, error=error)
        made["stub"] = stub
        return stub

    cog.__dict__["downloader_factory"] = factory
    return cog, made


def _embeds(message: _TwitterMessage) -> list[Embed]:
    """The embeds the cog delivered onto its placeholder."""
    return list(expansion_payload(message=message)["embeds"])


async def test_a_pasted_link_is_expanded_into_a_card() -> None:
    """The floor: a link becomes the post, and the source message is marked done."""
    cog, _ = _cog(post=_post())
    message = _message()

    await cog.on_message(as_message(fake=message))
    embeds = _embeds(message)

    assert embeds[0].description is not None
    assert "post body" in embeds[0].description
    assert embeds[0].author.name == "@Dbacks"
    assert _GREEN in message.reactions


async def test_the_footer_carries_the_counters_and_says_the_replies_are_gone() -> None:
    """The reply figure is the only number the endpoint gives for a thread it will not show.

    So the number cannot stand alone: `💬 540` under a card with no replies in it reads as an
    expansion that declined to show what it had, rather than as a platform that served none.
    """
    cog, _ = _cog(post=_post(image_urls=[]))
    message = _message()

    await cog.on_message(as_message(fake=message))
    footer = _embeds(message)[0].footer.text

    assert footer is not None
    assert "125" in footer
    assert "540" in footer
    assert "不提供留言" in footer


async def test_extra_images_become_embeds_sharing_the_post_url() -> None:
    """Discord merges embeds by URL, which is what turns the extras into one gallery."""
    cog, _ = _cog(post=_post(image_urls=[f"https://pbs.twimg.com/media/{n}.jpg" for n in "abc"]))
    message = _message()

    await cog.on_message(as_message(fake=message))
    embeds = _embeds(message)

    assert len(embeds) == 3
    assert [embed.url for embed in embeds] == [_URL, _URL, _URL]
    assert [embed.image.url for embed in embeds] == [
        "https://pbs.twimg.com/media/a.jpg",
        "https://pbs.twimg.com/media/b.jpg",
        "https://pbs.twimg.com/media/c.jpg",
    ]


async def test_a_video_post_shows_its_poster_and_a_link() -> None:
    """Nothing is downloaded here, so the frame stands in for the clip and the clip is a link.

    Without the poster a video post would be a card with no picture at all, which is what makes
    this the difference between the linked-video shape and Facebook's, whose video has no frame.
    """
    cog, _ = _cog(
        post=_post(
            image_urls=[],
            video_urls=["https://video.twimg.com/a/1280.mp4"],
            video_poster_urls=["https://pbs.twimg.com/amplify_video_thumb/p.jpg"],
        )
    )
    message = _message()

    await cog.on_message(as_message(fake=message))
    embed = _embeds(message)[0]

    assert embed.image.url == "https://pbs.twimg.com/amplify_video_thumb/p.jpg"
    assert embed.description is not None
    assert "https://video.twimg.com/a/1280.mp4" in embed.description


async def test_a_still_wins_the_preview_over_a_video_poster() -> None:
    """A post carrying both is showing the picture it chose, not the frame we fell back to."""
    cog, _ = _cog(
        post=_post(
            image_urls=["https://pbs.twimg.com/media/a.jpg"],
            video_urls=["https://video.twimg.com/a/1280.mp4"],
            video_poster_urls=["https://pbs.twimg.com/amplify_video_thumb/p.jpg"],
        )
    )
    message = _message()

    await cog.on_message(as_message(fake=message))

    assert _embeds(message)[0].image.url == "https://pbs.twimg.com/media/a.jpg"


async def test_a_truncated_post_says_so_on_the_card() -> None:
    """Twitter marks the cut in no way at all, so a quiet card passes a fragment off as the post."""
    cog, _ = _cog(post=_post(image_urls=[], is_truncated=True))
    message = _message()

    await cog.on_message(as_message(fake=message))
    description = _embeds(message)[0].description

    assert description is not None
    assert "Twitter 只提供開頭" in description


async def test_an_ordinary_post_does_not_claim_to_be_cut() -> None:
    """The notice rides on a flag, so its absence has to be silent."""
    cog, _ = _cog(post=_post(image_urls=[]))
    message = _message()

    await cog.on_message(as_message(fake=message))
    description = _embeds(message)[0].description

    assert description is not None
    assert "Twitter 只提供開頭" not in description


async def test_the_post_it_replies_to_gets_its_own_card_before_it() -> None:
    """Reading order, and a colour that says which of the two the link actually named."""
    parent = _output(text="Feel good.", url="https://x.com/Dbacks/status/1", image_urls=[])
    cog, _ = _cog(post=TwitterConversation(chain=[parent, _output(image_urls=[])]))
    message = _message()

    await cog.on_message(as_message(fake=message))
    embeds = _embeds(message)

    assert len(embeds) == 2
    assert embeds[0].description is not None
    assert "Feel good." in embeds[0].description
    assert embeds[0].url == "https://x.com/Dbacks/status/1"
    assert embeds[0].colour != embeds[1].colour


async def test_a_quoted_post_gets_a_card_outside_the_gallery() -> None:
    """Its own URL is what keeps Discord from folding it in among the pictures."""
    quoted = _output(text="quoted body", url="https://x.com/OpenAI/status/2", image_urls=[])
    cog, _ = _cog(post=_post(quoted=quoted))
    message = _message()

    await cog.on_message(as_message(fake=message))
    embeds = _embeds(message)

    assert embeds[-1].url == "https://x.com/OpenAI/status/2"
    assert embeds[-1].description is not None
    assert "quoted body" in embeds[-1].description


async def test_a_long_post_and_its_two_context_cards_fit_one_message() -> None:
    """Discord counts every embed's text against ONE ceiling and rejects the whole send past it.

    A post at the description limit plus a parent plus a quote is the worst case, and losing the
    expansion entirely is a far worse outcome than trimming the body.
    """
    parent = _output(text="p" * 3000, url="https://x.com/Dbacks/status/1", image_urls=[])
    quoted = _output(text="q" * 3000, url="https://x.com/OpenAI/status/2", image_urls=[])
    cog, _ = _cog(
        post=TwitterConversation(
            chain=[parent, _output(text="t" * 5000, image_urls=[], quoted=quoted)]
        )
    )
    message = _message()

    await cog.on_message(as_message(fake=message))
    embeds = _embeds(message)
    spent = sum(
        utf16_length(value=text)
        for embed in embeds
        for text in (embed.description, embed.footer.text, embed.author.name)
        if isinstance(text, str)
    )

    assert spent <= 6000


async def test_a_post_full_of_emoji_is_clipped_by_the_units_discord_counts() -> None:
    """Discord counts UTF-16 units, so a `len()`-based clip overshoots on non-BMP characters."""
    cog, _ = _cog(post=_post(text="🎉" * 3000, image_urls=[]))
    message = _message()

    await cog.on_message(as_message(fake=message))
    description = _embeds(message)[0].description

    assert description is not None
    assert utf16_length(value=description) <= 4096


async def test_the_read_marker_rides_beside_the_status_chain() -> None:
    """The platform marker says WHICH link was read, which the status tick alone cannot."""
    cog, _ = _cog(post=_post())
    message = _message()

    await cog.on_message(as_message(fake=message))

    assert TWITTER_EMOJI in message.reactions
    assert _GREEN in message.reactions


async def test_the_reply_slot_is_claimed_before_any_reaction_goes_on() -> None:
    """Reactions share one per-channel rate-limit bucket that a message send does not.

    Claiming first is what stops the card queueing behind two reaction adds, so the order is a
    latency decision rather than a style one. `tests/test_expansion_contract.py` holds every cog
    to it by reading the source; this proves it on the real call order.
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

    await cog.on_message(as_message(fake=message))

    assert reactions_when_claimed == []
    assert message.reactions[0] == TWITTER_EMOJI


async def test_a_message_addressed_to_the_bot_is_left_alone() -> None:
    """`gen_reply` answers about that one, and expanding as well reads the post twice."""
    cog, made = _cog(post=_post(), bot_id=999)
    message = _message(content=f"<@999> {_URL}")

    await cog.on_message(as_message(fake=message))

    assert made == {}
    assert message.reactions == []


async def test_a_url_that_names_no_post_is_ignored_silently() -> None:
    """A profile link never matches the pattern, so it earns no reaction and costs no request."""
    cog, made = _cog(post=_post())
    message = _message(content="https://x.com/Dbacks")

    await cog.on_message(as_message(fake=message))

    assert made == {}
    assert message.reactions == []


async def test_a_bot_message_is_ignored() -> None:
    """Otherwise the bot's own expansion would trigger the next one."""
    cog, made = _cog(post=_post())
    message = _message()
    message.author = FakeUser(bot=True)

    await cog.on_message(as_message(fake=message))

    assert made == {}


async def test_an_unreadable_post_is_marked_read_but_unshowable() -> None:
    """Deleted, protected and suspended are ordinary outcomes, not the bot breaking."""
    cog, _ = _cog(post=TwitterConversation())
    message = _message()

    await cog.on_message(as_message(fake=message))

    assert EXPANSION_UNREADABLE_EMOJI in message.reactions
    assert _RED not in message.reactions
    assert placeholder_withdrawn(message=message)


async def test_a_platform_under_load_is_marked_retryable_not_broken() -> None:
    """Telling someone a working link is dead is the worst outcome this feature has."""
    cog, _ = _cog(error=LinkRetryableError("429"))
    message = _message()

    await cog.on_message(as_message(fake=message))

    assert EXPANSION_RETRY_LATER_EMOJI in message.reactions
    assert _RED not in message.reactions


async def test_a_parse_failure_is_marked_failed_without_a_message() -> None:
    """The reaction is the whole report: a failed expansion leaves nothing in the channel."""
    cog, _ = _cog(error=RuntimeError("boom"))
    message = _message()

    await cog.on_message(as_message(fake=message))

    assert _RED in message.reactions
    assert placeholder_withdrawn(message=message)


async def test_a_refused_slot_still_says_which_platform_was_detected() -> None:
    """A cross alone cannot say which of two links died, so the marker goes on first."""
    cog, _ = _cog(post=_post())
    message = _message()

    async def refuse(**kwargs: object) -> object:
        """Answers the way a channel the bot cannot post in does."""
        del kwargs
        raise make_forbidden()

    message.reply = refuse  # ty: ignore[invalid-assignment]

    await cog.on_message(as_message(fake=message))

    assert message.reactions == [TWITTER_EMOJI, _RED]


async def test_the_link_is_read_at_the_url_the_message_carried() -> None:
    """The tracking tail rides along rather than being stripped here: the parser owns that."""
    cog, made = _cog(post=_post())
    message = _message(content=f"看這個 {_URL}?s=46&t=abc")

    await cog.on_message(as_message(fake=message))

    assert made["stub"].seen == [f"{_URL}?s=46&t=abc"]


def test_the_expansion_never_reaches_for_replies() -> None:
    """The conversation carries none and the card must not imply otherwise.

    A card built from `comments` would render an empty comment section on every post rather than
    saying, once, that replies are not available at all.
    """
    cog, _ = _cog()
    embeds = cog._build_embeds(conversation=_post(image_urls=[]))

    assert len(embeds) == 1
