"""Tests for the cog that auto-expands Facebook links pasted into a channel.

The downloader is stubbed at `downloader_factory`, so nothing here touches the network or the
parser: what these cover is the cog's own decisions — when it fires, what it renders, and what
a reader sees when the post cannot be read.
"""

import pytest

from discordbot.cogs.parse_facebook.cog import FacebookCogs
from discordbot.services.platforms.facebook import FacebookOutput
from discordbot.utils.expansion_placeholder import EXPANSION_DONE_EMOJI

from tests.helpers.casting import as_message
from tests.helpers.link_sources import (
    FACEBOOK_URL,
    facebook_post,
    expansion_embeds,
    stub_conversation_cog,
)
from tests.helpers.discord_mocks import FakeGuild, FakeDiscordMessage


def _message(content: str = FACEBOOK_URL) -> FakeDiscordMessage:
    """Builds a guild message carrying a Facebook link."""
    return FakeDiscordMessage(content=content, guild=FakeGuild())


async def test_a_pasted_link_is_expanded_into_a_card() -> None:
    """The ordinary case: one post embed carrying the body, the author and the counters."""
    cog, stub = stub_conversation_cog(cog_type=FacebookCogs, outcome=facebook_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert stub.seen == [FACEBOOK_URL]
    assert message.suppressed
    embeds = expansion_embeds(message=message)
    assert embeds[0].description == "post body"
    assert embeds[0].author.name == "Somebody"
    assert embeds[0].url == FACEBOOK_URL
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


async def test_the_footer_names_the_group_and_the_counters() -> None:
    """The group is the part a reader cannot get from the post itself, so it leads."""
    cog, _ = stub_conversation_cog(cog_type=FacebookCogs, outcome=facebook_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = expansion_embeds(message=message)[0].footer.text
    assert footer is not None
    assert footer.startswith("Some Group")
    assert "👍 1,017" in footer
    assert "💬 40" in footer
    assert "↗️ 37" in footer


async def test_a_page_post_leaves_the_group_out_of_the_footer() -> None:
    """A page post has no group, and the line must not carry a hole where one would go."""
    cog, _ = stub_conversation_cog(cog_type=FacebookCogs, outcome=facebook_post(group_name=""))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = expansion_embeds(message=message)[0].footer.text
    assert footer is not None
    assert footer.startswith("👍 1,017")


async def test_a_url_that_names_no_post_is_ignored_silently() -> None:
    """A profile link is not a failure, so it earns no reaction at all."""
    cog, stub = stub_conversation_cog(cog_type=FacebookCogs, outcome=facebook_post())
    message = _message(content="look https://www.facebook.com/NASA")

    await cog.on_message(message=as_message(fake=message))

    assert stub.seen == []
    assert message.reactions == []


@pytest.mark.parametrize(
    argnames=("post_url", "expected"),
    argvalues=[
        (FACEBOOK_URL, f"{FACEBOOK_URL}?comment_id=222"),
        # A `permalink.php` post URL already carries a query, so a second `?` breaks the link.
        (
            "https://www.facebook.com/permalink.php?story_fbid=1&id=2",
            "https://www.facebook.com/permalink.php?story_fbid=1&id=2&comment_id=222",
        ),
    ],
    ids=["plain", "permalink-php"],
)
async def test_the_comment_link_names_the_comment_on_the_post_url(
    post_url: str, expected: str
) -> None:
    """The comment card links to the post's URL naming the comment, whatever query it carries."""
    comment = FacebookOutput(comment_id="222", text="linked", author_name="C")
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs,
        outcome=facebook_post(url=post_url, comments=[comment], selected_comment_id="222"),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert expansion_embeds(message=message)[-1].url == expected


async def test_an_album_counts_the_videos_nothing_linked() -> None:
    """Only the first video gets a hint, so the rest would go unmentioned."""
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs,
        outcome=facebook_post(
            video_urls=[f"https://www.facebook.com/watch/?v={n}" for n in range(3)]
        ),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = expansion_embeds(message=message)[0].footer.text
    assert footer is not None
    assert "🎬 另有 2 部影片" in footer
