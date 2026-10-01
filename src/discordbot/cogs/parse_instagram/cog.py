"""Expands a public Instagram post URL into Discord embeds.

Nothing is downloaded. Images ride into the embeds as URLs for Discord to fetch itself, so this
cog needs no scratch directory, no media-delivery planner and no size ceiling.

`utils/expansion_cog.py` owns everything around the card and the card's shape: one post, its
carousel, and the comment a permalink singled out. What is here is what is Instagram's own: its
colour, its author line, its counters and where its links point.
"""

from nextcord.ext import commands

from discordbot.typings.timeouts import INSTAGRAM_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.expansion_cog import ConversationExpansionCog
from discordbot.services.platforms.instagram import (
    INSTAGRAM_URL_RE,
    InstagramOutput,
    InstagramDownloader,
    InstagramConversation,
    is_instagram_post_url,
)


class InstagramCogs(ConversationExpansionCog[InstagramOutput, InstagramConversation]):
    """Expands Instagram post links into Discord embeds."""

    SOURCE = "instagram"
    PLATFORM = "Instagram"
    URL_PATTERN = INSTAGRAM_URL_RE
    PLACEHOLDER_TEXT = "-# 正在讀取 Instagram 貼文⋯"
    READ_TIMEOUT_SECONDS = INSTAGRAM_EXPAND_TIMEOUT_SECONDS
    EMBED_COLOR = 0xE4405F
    downloader_factory = InstagramDownloader

    @staticmethod
    def url_is_expandable(*, url: str) -> bool:
        """Whether the matched URL names a post.

        The pattern matches the host, not the path, so a profile or the home page would otherwise
        earn a failure reaction on a link that was never a post.

        Args:
            url: The URL the pattern matched.

        Returns:
            True when the URL names a post.
        """
        return is_instagram_post_url(url=url)

    def _author_label(self, *, post: InstagramOutput) -> str:
        """The author line: the display name when the page carried one, always with the handle.

        Whichever half the page served is used on its own rather than dropping the line, since an
        author line missing entirely reads as an anonymous post.
        """
        if post.author_full_name and post.author_name:
            return f"{post.author_full_name} (@{post.author_name})"
        if post.author_name:
            return f"@{post.author_name}"
        return post.author_full_name

    def _comment_author_label(self, *, comment: InstagramOutput) -> str:
        """The commenter's handle, the only name a comment carries here."""
        return f"@{comment.author_name}" if comment.author_name else ""

    def _video_link(self, *, post: InstagramOutput) -> str:
        """The POST rather than `video_urls[0]`, which is Instagram's own signed CDN URL.

        That URL expires within days and the embed does not, so the link in it has to outlive
        the fetch.
        """
        return post.url

    def _footer_text(self, *, post: InstagramOutput, shown_images: int) -> str:
        """The counter line, plus what the image cap and the single video hint left behind.

        The videos are counted separately from the images because a mixed carousel is ordinary
        here and only its first video gets a link: counting images alone would report a
        five-image, five-video post as having one picture left over and stay silent about four
        clips. A count is shown only when it is positive, since Instagram serves `-1` rather than
        a number for a post whose author hid its likes.
        """
        parts: list[str] = []
        if post.like_count > 0:
            parts.append(f"❤️ {post.like_count:,}")
        if post.comment_count > 0:
            parts.append(f"💬 {post.comment_count:,}")
        parts.extend(self._omitted_media_notes(post=post, shown_images=shown_images))
        return " · ".join(parts)

    def _comment_url(self, *, post_url: str, comment: InstagramOutput) -> str:
        """The comment's own `/c/<id>/` permalink.

        Instagram will not serve that URL's payload to this bot, but it is the right thing to
        hand a reader.
        """
        return f"{post_url.rstrip('/')}/c/{comment.comment_id}/"


def setup(bot: commands.Bot) -> None:
    """Adds the InstagramCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(InstagramCogs(bot), override=True)
