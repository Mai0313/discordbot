"""Tests for the cog that auto-expands Twitter links pasted into a channel.

The downloader is stubbed at `downloader_factory`, so nothing here touches the network or the
parser: what these cover is the cog's own decisions — when it fires, what it renders, and what a
reader sees when the post cannot be read.
"""

from discordbot.utils.discord_embeds import utf16_length
from discordbot.cogs.parse_twitter.cog import TwitterCogs
from discordbot.services.platforms.twitter import TwitterConversation
from discordbot.utils.expansion_placeholder import EXPANSION_DONE_EMOJI

from tests.helpers.casting import as_message
from tests.helpers.link_sources import (
    TWITTER_URL,
    twitter_post,
    twitter_output,
    expansion_embeds,
    stub_conversation_cog,
)
from tests.helpers.discord_mocks import FakeGuild, FakeDiscordMessage


def _message(content: str = TWITTER_URL) -> FakeDiscordMessage:
    """Builds a guild message carrying a Twitter link."""
    return FakeDiscordMessage(content=content, guild=FakeGuild())


async def test_a_pasted_link_is_expanded_into_a_card() -> None:
    """The floor: a link becomes the post, and the source message is marked done."""
    cog, _ = stub_conversation_cog(cog_type=TwitterCogs, outcome=twitter_post())
    message = _message()

    await cog.on_message(as_message(fake=message))
    embeds = expansion_embeds(message=message)

    assert embeds[0].description is not None
    assert "post body" in embeds[0].description
    assert embeds[0].author.name == "@Dbacks"
    assert EXPANSION_DONE_EMOJI in message.reactions


async def test_the_footer_carries_the_counters_and_says_the_replies_are_gone() -> None:
    """The reply figure is the only number the endpoint gives for a thread it will not show.

    So the number cannot stand alone: `💬 540` under a card with no replies in it reads as an
    expansion that declined to show what it had, rather than as a platform that served none.
    """
    cog, _ = stub_conversation_cog(cog_type=TwitterCogs, outcome=twitter_post(image_urls=[]))
    message = _message()

    await cog.on_message(as_message(fake=message))
    footer = expansion_embeds(message=message)[0].footer.text

    assert footer is not None
    assert "125" in footer
    assert "540" in footer
    assert "不提供留言" in footer


async def test_extra_images_become_embeds_sharing_the_post_url() -> None:
    """Discord merges embeds by URL, which is what turns the extras into one gallery."""
    cog, _ = stub_conversation_cog(
        cog_type=TwitterCogs,
        outcome=twitter_post(image_urls=[f"https://pbs.twimg.com/media/{n}.jpg" for n in "abc"]),
    )
    message = _message()

    await cog.on_message(as_message(fake=message))
    embeds = expansion_embeds(message=message)

    assert len(embeds) == 3
    assert [embed.url for embed in embeds] == [TWITTER_URL, TWITTER_URL, TWITTER_URL]
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
    cog, _ = stub_conversation_cog(
        cog_type=TwitterCogs,
        outcome=twitter_post(
            image_urls=[],
            video_urls=["https://video.twimg.com/a/1280.mp4"],
            video_poster_urls=["https://pbs.twimg.com/amplify_video_thumb/p.jpg"],
        ),
    )
    message = _message()

    await cog.on_message(as_message(fake=message))
    embed = expansion_embeds(message=message)[0]

    assert embed.image.url == "https://pbs.twimg.com/amplify_video_thumb/p.jpg"
    assert embed.description is not None
    assert "https://video.twimg.com/a/1280.mp4" in embed.description


async def test_a_still_wins_the_preview_over_a_video_poster() -> None:
    """A post carrying both is showing the picture it chose, not the frame we fell back to."""
    cog, _ = stub_conversation_cog(
        cog_type=TwitterCogs,
        outcome=twitter_post(
            image_urls=["https://pbs.twimg.com/media/a.jpg"],
            video_urls=["https://video.twimg.com/a/1280.mp4"],
            video_poster_urls=["https://pbs.twimg.com/amplify_video_thumb/p.jpg"],
        ),
    )
    message = _message()

    await cog.on_message(as_message(fake=message))

    assert expansion_embeds(message=message)[0].image.url == "https://pbs.twimg.com/media/a.jpg"


async def test_a_truncated_post_says_so_on_the_card() -> None:
    """Twitter marks the cut in no way at all, so a quiet card passes a fragment off as the post."""
    cog, _ = stub_conversation_cog(
        cog_type=TwitterCogs, outcome=twitter_post(image_urls=[], is_truncated=True)
    )
    message = _message()

    await cog.on_message(as_message(fake=message))
    description = expansion_embeds(message=message)[0].description

    assert description is not None
    assert "Twitter 只提供開頭" in description


async def test_an_ordinary_post_does_not_claim_to_be_cut() -> None:
    """The notice rides on a flag, so its absence has to be silent."""
    cog, _ = stub_conversation_cog(cog_type=TwitterCogs, outcome=twitter_post(image_urls=[]))
    message = _message()

    await cog.on_message(as_message(fake=message))
    description = expansion_embeds(message=message)[0].description

    assert description is not None
    assert "Twitter 只提供開頭" not in description


async def test_the_post_it_replies_to_gets_its_own_card_before_it() -> None:
    """Reading order, and a colour that says which of the two the link actually named."""
    parent = twitter_output(text="Feel good.", url="https://x.com/Dbacks/status/1", image_urls=[])
    cog, _ = stub_conversation_cog(
        cog_type=TwitterCogs,
        outcome=TwitterConversation(chain=[parent, twitter_output(image_urls=[])]),
    )
    message = _message()

    await cog.on_message(as_message(fake=message))
    embeds = expansion_embeds(message=message)

    assert len(embeds) == 2
    assert embeds[0].description is not None
    assert "Feel good." in embeds[0].description
    assert embeds[0].url == "https://x.com/Dbacks/status/1"
    assert embeds[0].colour != embeds[1].colour


async def test_a_quoted_post_gets_a_card_outside_the_gallery() -> None:
    """Its own URL is what keeps Discord from folding it in among the pictures."""
    quoted = twitter_output(text="quoted body", url="https://x.com/OpenAI/status/2", image_urls=[])
    cog, _ = stub_conversation_cog(cog_type=TwitterCogs, outcome=twitter_post(quoted=quoted))
    message = _message()

    await cog.on_message(as_message(fake=message))
    embeds = expansion_embeds(message=message)

    assert embeds[-1].url == "https://x.com/OpenAI/status/2"
    assert embeds[-1].description is not None
    assert "quoted body" in embeds[-1].description


async def test_a_long_post_and_its_two_context_cards_fit_one_message() -> None:
    """Discord counts every embed's text against ONE ceiling and rejects the whole send past it.

    A post at the description limit plus a parent plus a quote is the worst case, and losing the
    expansion entirely is a far worse outcome than trimming the body.
    """
    parent = twitter_output(text="p" * 3000, url="https://x.com/Dbacks/status/1", image_urls=[])
    quoted = twitter_output(text="q" * 3000, url="https://x.com/OpenAI/status/2", image_urls=[])
    cog, _ = stub_conversation_cog(
        cog_type=TwitterCogs,
        outcome=TwitterConversation(
            chain=[parent, twitter_output(text="t" * 5000, image_urls=[], quoted=quoted)]
        ),
    )
    message = _message()

    await cog.on_message(as_message(fake=message))
    embeds = expansion_embeds(message=message)
    spent = sum(
        utf16_length(value=text)
        for embed in embeds
        for text in (embed.description, embed.footer.text, embed.author.name)
        if isinstance(text, str)
    )

    assert spent <= 6000


async def test_a_post_full_of_emoji_is_clipped_by_the_units_discord_counts() -> None:
    """Discord counts UTF-16 units, so a `len()`-based clip overshoots on non-BMP characters."""
    cog, _ = stub_conversation_cog(
        cog_type=TwitterCogs, outcome=twitter_post(text="🎉" * 3000, image_urls=[])
    )
    message = _message()

    await cog.on_message(as_message(fake=message))
    description = expansion_embeds(message=message)[0].description

    assert description is not None
    assert utf16_length(value=description) <= 4096


async def test_a_url_that_names_no_post_is_ignored_silently() -> None:
    """A profile link never matches the pattern, so it earns no reaction and costs no request."""
    cog, stub = stub_conversation_cog(cog_type=TwitterCogs, outcome=twitter_post())
    message = _message(content="https://x.com/Dbacks")

    await cog.on_message(as_message(fake=message))

    assert stub.seen == []
    assert message.reactions == []


async def test_the_link_is_read_at_the_url_the_message_carried() -> None:
    """The tracking tail rides along rather than being stripped here: the parser owns that."""
    cog, stub = stub_conversation_cog(cog_type=TwitterCogs, outcome=twitter_post())
    message = _message(content=f"看這個 {TWITTER_URL}?s=46&t=abc")

    await cog.on_message(as_message(fake=message))

    assert stub.seen == [f"{TWITTER_URL}?s=46&t=abc"]
