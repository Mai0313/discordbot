"""Tests for the cog that auto-expands Facebook links pasted into a channel.

The downloader is stubbed at `downloader_factory`, so nothing here touches the network or the
parser: what these cover is the cog's own decisions — when it fires, what it renders, and what
a reader sees when the post cannot be read.
"""

from datetime import UTC, datetime

from discordbot.utils.discord_embeds import utf16_length
from discordbot.cogs.parse_facebook.cog import FacebookCogs
from discordbot.services.platforms.facebook import FacebookOutput, FacebookConversation
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


async def test_extra_images_become_embeds_sharing_the_post_url() -> None:
    """Sharing the URL is what makes Discord merge them into one gallery under the post."""
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs,
        outcome=facebook_post(image_urls=[f"https://scontent.example/{n}.jpg" for n in range(3)]),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    embeds = expansion_embeds(message=message)
    assert len(embeds) == 3
    assert all(embed.url == FACEBOOK_URL for embed in embeds)
    assert [embed.image.url for embed in embeds] == [
        "https://scontent.example/0.jpg",
        "https://scontent.example/1.jpg",
        "https://scontent.example/2.jpg",
    ]


async def test_images_past_the_cap_are_counted_in_the_footer() -> None:
    """A gallery post must not scroll the channel, and must say what it left behind."""
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs,
        outcome=facebook_post(image_urls=[f"https://scontent.example/{n}.jpg" for n in range(7)]),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    embeds = expansion_embeds(message=message)
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
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs,
        outcome=facebook_post(comments=[comment], selected_comment_id="1730777104666239"),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    embeds = expansion_embeds(message=message)
    comment_embed = embeds[-1]
    assert comment_embed.description is not None
    assert "the one linked" in comment_embed.description
    assert comment_embed.author.name == "Commenter"
    assert comment_embed.url != FACEBOOK_URL
    assert "comment_id=1730777104666239" in (comment_embed.url or "")


async def test_no_comment_card_without_one_named() -> None:
    """A plain link shows the post alone; the preloaded comments are not the whole section."""
    comment = FacebookOutput(comment_id="999", text="some comment")
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs, outcome=facebook_post(comments=[comment])
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert all(
        "指定的留言" not in (embed.description or "")
        for embed in expansion_embeds(message=message)
    )


async def test_a_video_post_shows_a_link_instead_of_an_empty_card() -> None:
    """There is no file to attach logged out, so the link is the whole of what can be shown."""
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs,
        outcome=facebook_post(image_urls=[], video_urls=["https://www.facebook.com/watch/?v=1"]),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = expansion_embeds(message=message)[0].description
    assert description is not None
    assert "點此觀看影片" in description


async def test_a_long_post_is_cut_with_a_notice() -> None:
    """A truncated post must never read as a whole one."""
    cog, _ = stub_conversation_cog(cog_type=FacebookCogs, outcome=facebook_post(text="x" * 5000))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = expansion_embeds(message=message)[0].description
    assert description is not None
    assert len(description) <= 4096
    assert description.endswith("（全文請看原貼文）")


async def test_a_url_that_names_no_post_is_ignored_silently() -> None:
    """A profile link is not a failure, so it earns no reaction at all."""
    cog, stub = stub_conversation_cog(cog_type=FacebookCogs, outcome=facebook_post())
    message = _message(content="look https://www.facebook.com/NASA")

    await cog.on_message(message=as_message(fake=message))

    assert stub.seen == []
    assert message.reactions == []


async def test_a_long_post_with_a_video_stays_inside_the_description_limit() -> None:
    """The hint is reserved before the clip, or Discord rejects the send and the card is lost."""
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs,
        outcome=facebook_post(text="x" * 5000, video_urls=["https://www.facebook.com/watch/?v=1"]),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = expansion_embeds(message=message)[0].description
    assert description is not None
    assert len(description) <= 4096
    assert "點此觀看影片" in description


async def test_a_long_post_and_a_long_comment_fit_one_message() -> None:
    """Discord counts every embed in a message together and rejects the whole send when over."""
    comment = FacebookOutput(comment_id="222", text="y" * 4000, author_name="Commenter")
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs,
        outcome=facebook_post(text="x" * 5000, comments=[comment], selected_comment_id="222"),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    total = sum(
        len(embed.description or "") + len(embed.footer.text or "") + len(embed.author.name or "")
        for embed in expansion_embeds(message=message)
    )
    assert total <= 6000


async def test_a_long_comment_under_a_short_post_stays_inside_the_description_limit() -> None:
    """A short post leaves the comment more of the message than one description may carry."""
    comment = FacebookOutput(comment_id="222", text="y" * 5000, author_name="Commenter")
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs,
        outcome=facebook_post(text="hi", comments=[comment], selected_comment_id="222"),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = expansion_embeds(message=message)[-1].description
    assert description is not None
    assert utf16_length(value=description) <= 4096
    assert description.endswith("（全文請看原貼文）")


async def _post_body_length(*, outcome: FacebookConversation) -> int:
    """Expands `outcome` and measures the post's own description, in the units Discord counts."""
    cog, _ = stub_conversation_cog(cog_type=FacebookCogs, outcome=outcome)
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    return utf16_length(value=expansion_embeds(message=message)[0].description or "")


async def test_a_named_comment_takes_its_room_from_the_post() -> None:
    """A long post gives up part of the message before a named comment is budgeted beside it.

    Measured against the same post with no comment named, which is cut only at the description
    ceiling: without the reservation the post takes that whole ceiling either way and the comment
    card gets only what is left over.
    """
    comment = FacebookOutput(comment_id="222", text="y" * 4000, author_name="Commenter")

    alone = await _post_body_length(outcome=facebook_post(text="x" * 5000))
    beside_comment = await _post_body_length(
        outcome=facebook_post(text="x" * 5000, comments=[comment], selected_comment_id="222")
    )

    assert beside_comment < alone


async def test_the_comment_link_joins_an_existing_query_correctly() -> None:
    """A `permalink.php` post URL already carries a query, so a second `?` breaks the link."""
    url = "https://www.facebook.com/permalink.php?story_fbid=1&id=2"
    comment = FacebookOutput(comment_id="222", text="linked", author_name="C")
    cog, _ = stub_conversation_cog(
        cog_type=FacebookCogs,
        outcome=facebook_post(url=url, comments=[comment], selected_comment_id="222"),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    comment_url = expansion_embeds(message=message)[-1].url
    assert comment_url is not None
    assert comment_url.count("?") == 1
    assert comment_url.endswith("&comment_id=222")


async def test_a_post_full_of_emoji_is_clipped_by_the_units_discord_counts() -> None:
    """A Facebook post runs to 63,206 characters, so an emoji-heavy one reaches this easily."""
    cog, _ = stub_conversation_cog(cog_type=FacebookCogs, outcome=facebook_post(text="🐈" * 4200))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = expansion_embeds(message=message)[0].description
    assert description is not None
    assert utf16_length(value=description) <= 4096


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
