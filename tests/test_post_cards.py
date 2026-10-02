"""What every conversation expansion cog's card owes, checked against each cog.

The post's own embed and its gallery come from `post_card_embeds` on every one of them, and
`ConversationExpansionCog`'s default card adds the comment a link singled out. What those
decide is tested here once per cog; each cog's own file keeps its hooks: the footer, the author
line, and where its links point.
"""

from typing import Any
from datetime import UTC, datetime
from collections.abc import Callable

import pytest
from nextcord import Embed

from discordbot.utils.expansion_cog import POST_CARD_MAX_IMAGES, ConversationExpansionCog
from discordbot.utils.discord_embeds import utf16_length
from discordbot.cogs.parse_twitter.cog import TwitterCogs
from discordbot.cogs.parse_facebook.cog import FacebookCogs
from discordbot.services.platforms.base import PlatformOutput, PlatformConversation
from discordbot.cogs.parse_instagram.cog import InstagramCogs
from discordbot.services.platforms.facebook import FacebookOutput
from discordbot.services.platforms.instagram import InstagramOutput

from tests.helpers.link_sources import (
    expand,
    twitter_post,
    facebook_post,
    instagram_post,
    expansion_embeds,
)

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
    assert "🖼️ 另有 3 張" in footer


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

    assert all("指定的留言" not in (embed.description or "") for embed in embeds)


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
    assert "點此觀看影片" in description


@default_cards
async def test_a_long_post_is_cut_with_a_notice(cog_type: _CogType, post: _PostFactory) -> None:
    """A truncated post must never read as a whole one."""
    embeds = await _expand(cog_type=cog_type, outcome=post(text="x" * 5000))

    description = embeds[0].description
    assert description is not None
    assert len(description) <= 4096
    assert description.endswith("（全文請看原貼文）")


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
    assert "點此觀看影片" in description
    # Without this the same test passes on a clip that truncated silently.
    assert "（全文請看原貼文）" in description


@commented_cards
async def test_a_long_post_and_a_long_comment_fit_one_message(
    cog_type: _CogType, post: _PostFactory, comment: Callable[..., PlatformOutput]
) -> None:
    """Discord counts every embed in a message together and rejects the whole send when over."""
    named = comment(comment_id="222", text="y" * 4000, author_name="Commenter")
    outcome = post(text="x" * 5000, comments=[named], selected_comment_id="222")

    embeds = await _expand(cog_type=cog_type, outcome=outcome)

    total = sum(
        len(embed.description or "") + len(embed.footer.text or "") + len(embed.author.name or "")
        for embed in embeds
    )
    assert total <= 6000


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
    assert description.endswith("（全文請看原貼文）")


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
