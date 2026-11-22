"""Expands public Instagram post URLs into Discord embeds via `ConversationExpansionCog`."""

from nextcord.ext import commands

from discordbot.typings.timeouts import INSTAGRAM_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.expansion_cog import ConversationExpansionCog
from discordbot.services.platforms.instagram import (
    INSTAGRAM_POST_LINK,
    InstagramOutput,
    InstagramDownloader,
    InstagramConversation,
)


class InstagramCogs(ConversationExpansionCog[InstagramOutput, InstagramConversation]):
    """Expands Instagram post links into Discord embeds."""

    SOURCE = "instagram"
    PLATFORM = "Instagram"
    POST_LINK = INSTAGRAM_POST_LINK
    PLACEHOLDER_TEXT = "-# 正在讀取 Instagram 貼文⋯"
    READ_TIMEOUT_SECONDS = INSTAGRAM_EXPAND_TIMEOUT_SECONDS
    EMBED_COLOR = 0xE4405F
    DOWNLOAD_VIDEOS = True
    downloader_factory = InstagramDownloader

    def _author_label(self, post: InstagramOutput) -> str:
        """Returns the author display string, falling back to whichever of full name or handle the page served."""
        if post.author_full_name and post.author_name:
            return f"{post.author_full_name} (@{post.author_name})"
        if post.author_name:
            return f"@{post.author_name}"
        return post.author_full_name

    def _comment_author_label(self, comment: InstagramOutput) -> str:
        """Returns the commenter's handle, the only name Instagram comments carry."""
        return f"@{comment.author_name}" if comment.author_name else ""

    def _video_download_headers(self) -> dict[str, str]:
        """Returns HTTP headers including Instagram's referer for CDN video fetches."""
        return {**super()._video_download_headers(), "Referer": "https://www.instagram.com/"}

    def _video_link(self, post: InstagramOutput, index: int = 0) -> str:
        """Returns `post.url` instead of `video_urls`; signed CDN URLs expire within days."""
        del index
        return post.url

    def _footer_text(self, post: InstagramOutput, shown_images: int) -> str:
        """Returns the footer line with positive engagement counts and omitted-media notes."""
        parts: list[str] = []
        # Instagram serves `-1` when an author hides likes.
        if post.like_count > 0:
            parts.append(f"❤️ {post.like_count:,}")
        if post.comment_count > 0:
            parts.append(f"💬 {post.comment_count:,}")
        parts.extend(self._omitted_media_notes(post=post, shown_images=shown_images))
        return " · ".join(parts)

    def _comment_url(self, post_url: str, comment: InstagramOutput) -> str:
        """Returns the `/c/<id>/` comment permalink (unfetchable by the bot, valid for users)."""
        return f"{post_url.rstrip('/')}/c/{comment.comment_id}/"


def setup(bot: commands.Bot) -> None:
    """Adds the InstagramCogs to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(InstagramCogs(bot), override=True)
