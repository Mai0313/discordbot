"""Tests for the cog that auto-expands Instagram links pasted into a channel.

The downloader is stubbed at `downloader_factory`, so nothing here touches the network or the
parser: what these cover is the cog's own decisions — when it fires, what it renders, and what
a reader sees when the post cannot be read.
"""

from discordbot.cogs.parse_instagram.cog import InstagramCogs
from discordbot.utils.expansion_placeholder import EXPANSION_DONE_EMOJI
from discordbot.services.platforms.instagram import InstagramOutput

from tests.helpers.casting import as_message
from tests.helpers.link_sources import (
    INSTAGRAM_URL,
    instagram_post,
    expansion_embeds,
    stub_conversation_cog,
)
from tests.helpers.discord_mocks import FakeGuild, FakeDiscordMessage


def _message(content: str = INSTAGRAM_URL) -> FakeDiscordMessage:
    """Builds a guild message carrying an Instagram link."""
    return FakeDiscordMessage(content=content, guild=FakeGuild())


async def test_a_pasted_link_is_expanded_into_a_card() -> None:
    """The ordinary case: one post embed carrying the caption, the author and the counters."""
    cog, stub = stub_conversation_cog(cog_type=InstagramCogs, outcome=instagram_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert stub.seen == [INSTAGRAM_URL]
    assert message.suppressed
    embeds = expansion_embeds(message=message)
    assert embeds[0].description == "post body"
    assert embeds[0].url == INSTAGRAM_URL
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


async def test_the_author_line_carries_both_names() -> None:
    """Instagram identifies people by handle, but the display name is what a reader knows."""
    cog, _ = stub_conversation_cog(cog_type=InstagramCogs, outcome=instagram_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert expansion_embeds(message=message)[0].author.name == "晏凌 (@c_cylynn)"


async def test_a_post_without_a_display_name_falls_back_to_the_handle() -> None:
    """Not every account sets one, and a bare `@handle` is still a complete label."""
    cog, _ = stub_conversation_cog(
        cog_type=InstagramCogs, outcome=instagram_post(author_full_name="")
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert expansion_embeds(message=message)[0].author.name == "@c_cylynn"


async def test_the_footer_carries_the_counters() -> None:
    """Likes and comments are what Instagram reports; there is no share figure."""
    cog, _ = stub_conversation_cog(cog_type=InstagramCogs, outcome=instagram_post())
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = expansion_embeds(message=message)[0].footer.text
    assert footer is not None
    assert "❤️ 8,855" in footer
    assert "💬 11" in footer


async def test_a_named_comment_links_to_its_own_permalink_under_its_handle() -> None:
    """A comment carries a handle and no display name, and its card links to its own `/c/` URL."""
    comment = InstagramOutput(
        comment_id="17946527169275440", text="the one linked", author_name="xiao_pang0704"
    )
    cog, _ = stub_conversation_cog(
        cog_type=InstagramCogs,
        outcome=instagram_post(comments=[comment], selected_comment_id="17946527169275440"),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    comment_embed = expansion_embeds(message=message)[-1]
    assert comment_embed.author.name == "@xiao_pang0704"
    assert comment_embed.url == f"{INSTAGRAM_URL}c/17946527169275440/"


async def test_a_comment_permalink_is_expanded_too() -> None:
    """The URL the user pastes is the comment's, and the cog reads the post behind it."""
    cog, stub = stub_conversation_cog(cog_type=InstagramCogs, outcome=instagram_post())
    url = f"{INSTAGRAM_URL}c/17946527169275440/?img_index=1"
    message = _message(content=f"看這個 {url}")

    await cog.on_message(message=as_message(fake=message))

    assert stub.seen == [url]
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


async def test_a_profile_url_is_ignored_silently() -> None:
    """A profile link is not a failure, so it earns no reaction at all."""
    cog, stub = stub_conversation_cog(cog_type=InstagramCogs, outcome=instagram_post())
    message = _message(content="look https://www.instagram.com/c_cylynn/")

    await cog.on_message(message=as_message(fake=message))

    assert stub.seen == []
    assert message.reactions == []


async def test_a_profile_url_does_not_hide_a_post_after_it() -> None:
    """A refused link is skipped, so the post after it is still expanded (#854)."""
    profile = "https://www.instagram.com/c_cylynn/"
    assert InstagramCogs.URL_PATTERN.search(string=profile) is not None
    cog, stub = stub_conversation_cog(cog_type=InstagramCogs, outcome=instagram_post())
    message = _message(content=f"{profile} 跟這篇 {INSTAGRAM_URL}")

    await cog.on_message(message=as_message(fake=message))

    assert stub.seen == [INSTAGRAM_URL]
    assert message.reactions[-1] == EXPANSION_DONE_EMOJI


async def test_the_video_link_points_at_the_post_not_the_expiring_cdn_url() -> None:
    """`video_versions[0].url` is signed and dies within days; the embed carrying it does not."""
    clip = "https://instagram.example/clip.mp4?oe=DEADBEEF"
    cog, _ = stub_conversation_cog(
        cog_type=InstagramCogs, outcome=instagram_post(image_urls=[], video_urls=[clip])
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    description = expansion_embeds(message=message)[0].description
    assert description is not None
    assert clip not in description
    assert INSTAGRAM_URL in description


async def test_a_mixed_carousel_counts_the_videos_nothing_linked() -> None:
    """Counting images alone reported four clips as nothing at all."""
    cog, _ = stub_conversation_cog(
        cog_type=InstagramCogs,
        outcome=instagram_post(
            image_urls=[f"https://instagram.example/{n}.jpg" for n in range(5)],
            video_urls=[f"https://instagram.example/{n}.mp4" for n in range(5)],
        ),
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = expansion_embeds(message=message)[0].footer.text
    assert footer is not None
    assert "🖼️ 另有 1 張" in footer
    assert "🎬 另有 4 部影片" in footer


async def test_a_post_whose_author_hid_its_likes_shows_no_like_count() -> None:
    """Instagram serves `-1` there rather than omitting the field."""
    cog, _ = stub_conversation_cog(cog_type=InstagramCogs, outcome=instagram_post(like_count=-1))
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    footer = expansion_embeds(message=message)[0].footer.text
    assert footer is not None
    assert "❤️" not in footer


async def test_an_author_with_only_a_display_name_still_gets_a_line() -> None:
    """Dropping the line entirely would read as an anonymous post."""
    cog, _ = stub_conversation_cog(
        cog_type=InstagramCogs, outcome=instagram_post(author_name="", author_full_name="晏凌")
    )
    message = _message()

    await cog.on_message(message=as_message(fake=message))

    assert expansion_embeds(message=message)[0].author.name == "晏凌"
