"""Expands Twitter (x.com) post URLs into Discord embeds via `ConversationExpansionCog`.

Renders the ancestor chain, target post (stills and downloaded videos, falling back to video
poster and link on failure), and quoted post; skips replies (left to `gen_reply`) and flags
truncated note tweets when FxTwitter is unavailable.
"""

from functools import partial

from nextcord import Color, Embed
from nextcord.ext import commands

from discordbot.typings.timeouts import TWITTER_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.expansion_cog import (
    VIDEO_HINT,
    TRUNCATION_NOTICE,
    CONTEXT_CARD_COLOR,
    POST_CARD_MAX_IMAGES,
    ConversationExpansionCog,
    post_card_embeds,
    omitted_posts_note,
    context_card_budget,
    omitted_videos_note,
)
from discordbot.utils.discord_embeds import (
    DISCORD_EMBED_COUNT_LIMIT,
    DISCORD_EMBED_TOTAL_LIMIT,
    DISCORD_EMBED_DESCRIPTION_LIMIT,
    utf16_length,
    clip_to_utf16_limit,
)
from discordbot.services.platforms.twitter import (
    TWITTER_POST_LINK,
    TwitterOutput,
    TwitterDownloader,
    TwitterConversation,
)

# Per-context-card text reserve, capped so the target post keeps at least 1,200 chars.
_CONTEXT_RESERVE = 1200
_CONTEXT_RESERVE_CAP = DISCORD_EMBED_TOTAL_LIMIT - 1200

# Appended when `is_truncated` is set because syndication cuts bodies with no ellipsis.
_TRUNCATED_NOTICE = "\n\n-# ⋯這則貼文較長，Twitter 只提供開頭"
_PARENT_HEADER = "↩️ **回覆的貼文**"
_QUOTED_HEADER = "💬 **引用的貼文**"


class TwitterCogs(ConversationExpansionCog[TwitterOutput, TwitterConversation]):
    """Expands Twitter links into Discord embeds."""

    SOURCE = "twitter"
    PLATFORM = "Twitter"
    POST_LINK = TWITTER_POST_LINK
    PLACEHOLDER_TEXT = "-# 正在讀取 Twitter 貼文⋯"
    READ_TIMEOUT_SECONDS = TWITTER_EXPAND_TIMEOUT_SECONDS
    # Twitter blue rather than X black so the left embed bar stays visible on dark theme.
    EMBED_COLOR = 0x1DA1F2
    DOWNLOAD_VIDEOS = True
    # Skipping the replies spares a short first post that quotes nothing the FxTwitter request.
    # `staticmethod` because a `partial` on a class turns into a method from Python 3.14 on.
    downloader_factory = staticmethod(partial(TwitterDownloader, read_replies=False))

    def _author_label(self, post: TwitterOutput) -> str:
        """Returns the handle, which is the only name a post carries here."""
        return f"@{post.author_name}" if post.author_name else ""

    def _footer_text(self, post: TwitterOutput, shown_images: int) -> str:
        """Returns the counters and omitted-media notes."""
        parts = [f"♥ {post.like_count:,}", f"💬 {post.comment_count:,}"]
        parts.extend(self._omitted_media_notes(post=post, shown_images=shown_images))
        return " · ".join(parts)

    def _context_embed(
        self, post: TwitterOutput, header: str, budget: int, show_media: bool
    ) -> Embed:
        """Returns a grey context embed for an ancestor or quoted post, using its own URL."""
        image = (
            next((url for url in (*post.image_urls, *post.video_poster_urls) if url), "")
            if show_media
            else ""
        )
        hint = (
            VIDEO_HINT.format(url=self._video_link(post=post))
            if show_media and post.video_urls
            else ""
        )
        lead = f"{header}\n" if header else ""
        # Cap description to 4096 minus lead/hint (`context_card_budget` slack covers author only).
        spent = utf16_length(value=lead) + utf16_length(value=hint)
        limit = min(max(budget - spent, 0), DISCORD_EMBED_DESCRIPTION_LIMIT - spent)
        body = clip_to_utf16_limit(text=post.text, limit=limit, notice=TRUNCATION_NOTICE)
        embed = Embed(
            description=f"{lead}{body}{hint}".strip() or None,
            url=post.url,
            color=Color(value=CONTEXT_CARD_COLOR),
        )
        if image:
            embed.set_image(url=image)
        author = self._author_label(post=post)
        if author:
            embed.set_author(name=author, url=post.url, icon_url=post.author_icon_url or None)
        return embed

    def _build_embeds(
        self, conversation: TwitterConversation, undelivered_video_indices: list[int] | None = None
    ) -> list[Embed]:
        """Returns embeds for the ancestor thread, target post (with images), and quoted post."""
        post = conversation.target
        if post is None:
            return []

        missing_videos = (
            list(range(len(post.video_urls)))
            if undelivered_video_indices is None
            else undelivered_video_indices
        )
        first_missing = missing_videos[0] if missing_videos else None
        hint = (
            VIDEO_HINT.format(url=self._video_link(post=post, index=first_missing))
            if first_missing is not None
            else ""
        )
        truncated = _TRUNCATED_NOTICE if post.is_truncated else ""
        shown = post.image_urls[:POST_CARD_MAX_IMAGES]
        # Fall back to the first undelivered clip's poster frame when there are no stills.
        poster = (
            post.video_poster_urls[first_missing]
            if first_missing is not None and first_missing < len(post.video_poster_urls)
            else ""
        )
        images = shown or ([poster] if poster else [])
        room = DISCORD_EMBED_COUNT_LIMIT - max(len(images), 1) - bool(post.quoted)
        ancestors = conversation.chain[:-1][-room:]
        omitted = len(conversation.chain) - 1 - len(ancestors)
        context_count = len(ancestors) + bool(post.quoted)
        footer_parts = [
            part
            for part in (
                self._footer_text(post=post, shown_images=len(shown)),
                omitted_videos_note(count=len(missing_videos) - 1)
                if len(missing_videos) > 1
                else "",
                omitted_posts_note(count=omitted) if omitted else "",
            )
            if part
        ]
        embeds = post_card_embeds(
            post=post,
            color=self.EMBED_COLOR,
            author=self._author_label(post=post),
            footer=" · ".join(footer_parts),
            images=images,
            suffix=truncated + hint,
            reserve=min(_CONTEXT_RESERVE * context_count, _CONTEXT_RESERVE_CAP),
        )

        budget = context_card_budget(card=embeds[0]) // max(context_count, 1)
        thread = [
            self._context_embed(
                post=ancestor,
                header=_PARENT_HEADER if index == len(ancestors) - 1 else "",
                budget=budget,
                show_media=True,
            )
            for index, ancestor in enumerate(ancestors)
        ]
        if post.quoted is not None:
            embeds.append(
                self._context_embed(
                    post=post.quoted, header=_QUOTED_HEADER, budget=budget, show_media=False
                )
            )
        return [*thread, *embeds]


def setup(bot: commands.Bot) -> None:
    """Adds the TwitterCogs to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(TwitterCogs(bot), override=True)
