"""Expands a public Facebook post URL into Discord embeds.

Nothing is downloaded. Images ride into the embeds as URLs for Discord to fetch itself, so this
cog needs no scratch directory, no media-delivery planner and no size ceiling.

`utils/expansion_cog.py` owns everything around the card and the card's shape: one post, its
images, and the comment a URL singled out. What is here is what is Facebook's own: its colour,
its counters and how a comment is linked.
"""

from nextcord.ext import commands

from discordbot.typings.timeouts import FACEBOOK_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.expansion_cog import ConversationExpansionCog
from discordbot.services.platforms.facebook import (
    FACEBOOK_URL_RE,
    FacebookOutput,
    FacebookDownloader,
    FacebookConversation,
    is_facebook_post_url,
)


class FacebookCogs(ConversationExpansionCog[FacebookOutput, FacebookConversation]):
    """Expands Facebook post links into Discord embeds."""

    SOURCE = "facebook"
    PLATFORM = "Facebook"
    URL_PATTERN = FACEBOOK_URL_RE
    PLACEHOLDER_TEXT = "-# 正在讀取 Facebook 貼文⋯"
    READ_TIMEOUT_SECONDS = FACEBOOK_EXPAND_TIMEOUT_SECONDS
    EMBED_COLOR = 0x1877F2
    downloader_factory = FacebookDownloader

    @staticmethod
    def url_is_expandable(*, url: str) -> bool:
        """Whether the matched URL names a post.

        The pattern matches the host, not the path, so a profile or group home page would
        otherwise earn a failure reaction on a link that was never a post.

        Args:
            url: The URL the pattern matched.

        Returns:
            True when the URL names a post.
        """
        return is_facebook_post_url(url=url)

    def _footer_text(self, *, post: FacebookOutput, shown_images: int) -> str:
        """The counter line: where the post lives, how it did, and what was left out.

        The group name leads because it is the part a reader cannot get from the post itself, and
        it is simply absent for a page or profile post rather than being replaced by a
        placeholder, which is what keeps the line from having a hole in it.
        """
        parts = [post.group_name] if post.group_name else []
        if post.like_count > 0:
            parts.append(f"👍 {post.like_count:,}")
        if post.comment_count > 0:
            parts.append(f"💬 {post.comment_count:,}")
        if post.share_count > 0:
            parts.append(f"↗️ {post.share_count:,}")
        parts.extend(self._omitted_media_notes(post=post, shown_images=shown_images))
        return " · ".join(parts)

    def _comment_url(self, *, post_url: str, comment: FacebookOutput) -> str:
        """The post's URL naming the comment, joined with `&` when it already carries a query.

        `permalink.php` links always do.
        """
        joiner = "&" if "?" in post_url else "?"
        return f"{post_url}{joiner}comment_id={comment.comment_id}"


def setup(bot: commands.Bot) -> None:
    """Adds the FacebookCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(FacebookCogs(bot), override=True)
