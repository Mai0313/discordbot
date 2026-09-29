"""Expands a Twitter (x.com) post URL into Discord embeds.

Discord renders nothing useful for an x.com link on its own — the page it fetches is the same
empty SPA shell `services/platforms/twitter.py` documents — so a pasted link sits in the channel
as a bare URL. This turns it into the post.

Nothing is downloaded: the stills and a video's poster frame ride to Discord as URLs for it to
fetch itself, and the clip is a link. That is a choice rather than a limit — Twitter's mp4 is
directly fetchable — so a deployment that wants the file has `/download_video`.

`utils/expansion_cog.py` owns everything around the card and the post's own embed. What is here
is the rest of the card — the post it replies to and the post it quotes — and two things it has
to say out loud because the platform will not: a post past roughly 275 characters is served
truncated with no marker of any kind, and the endpoint serves no replies at all, so the reply
figure in the footer counts a conversation nothing here can show.
"""

from nextcord import Embed, Colour
from nextcord.ext import commands

from discordbot.typings.timeouts import TWITTER_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.expansion_cog import (
    VIDEO_HINT,
    TRUNCATION_NOTICE,
    CONTEXT_CARD_COLOR,
    POST_CARD_MAX_IMAGES,
    ConversationExpansionCog,
    post_card_embeds,
    context_card_budget,
)
from discordbot.utils.discord_embeds import (
    DISCORD_EMBED_DESCRIPTION_LIMIT,
    utf16_length,
    clip_to_utf16_limit,
)
from discordbot.services.platforms.twitter import (
    TWITTER_URL_RE,
    TwitterOutput,
    TwitterDownloader,
    TwitterConversation,
)

# What the post gives up so the two context cards always fit beside it: a card carrying a parent
# and a quote spends three descriptions against the message-wide ceiling.
_CONTEXT_RESERVE = 1200

# Said on the card rather than left to the reader, because Twitter marks a cut body in no way at
# all: no ellipsis, no "show more", nothing but a key saying a longer version exists that the
# endpoint will not serve. A card that stays quiet passes a fragment off as the whole post.
_TRUNCATED_NOTICE = "\n\n-# ⋯這則貼文較長，Twitter 只提供開頭"
_PARENT_HEADER = "↩️ **回覆的貼文**"
_QUOTED_HEADER = "💬 **引用的貼文**"


class TwitterCogs(ConversationExpansionCog[TwitterOutput, TwitterConversation]):
    """Expands Twitter links into Discord embeds."""

    SOURCE = "twitter"
    PLATFORM = "Twitter"
    URL_PATTERN = TWITTER_URL_RE
    PLACEHOLDER_TEXT = "-# 正在讀取 Twitter 貼文⋯"
    READ_TIMEOUT_SECONDS = TWITTER_EXPAND_TIMEOUT_SECONDS
    # The old Twitter blue rather than X's black. The rail down an embed's left edge exists to say
    # which platform a card came from at a glance, and black does not: it reads as a hairline on
    # Discord's light theme and vanishes into the dark one.
    EMBED_COLOR = 0x1DA1F2
    downloader_factory = TwitterDownloader

    def _author_label(self, *, post: TwitterOutput) -> str:
        """The handle, which is the only name a post carries here."""
        return f"@{post.author_name}" if post.author_name else ""

    def _footer_text(self, *, post: TwitterOutput, shown_images: int) -> str:
        """The counters, and what the card could not show.

        The reply figure is the size of the whole thread under the post, which is the only reply
        number the endpoint publishes — and it publishes none of the replies themselves, so a
        non-zero one says as much beside it. A count with nothing under it otherwise reads as an
        expansion that declined to show what it had. At zero there is nothing to explain away, and
        the caveat would read as an apology for a thread that does not exist.
        """
        replies = f"💬 {post.comment_count:,}"
        if post.comment_count:
            replies += "（Twitter 不提供留言）"
        parts = [f"♥ {post.like_count:,}", replies]
        # Never positive today: the reader keeps at most four media per post and the card shows
        # four. It says what was left out if that reader slice is ever widened.
        omitted = len(post.image_urls) - shown_images
        if omitted > 0:
            parts.append(f"另有 {omitted} 張圖片")
        return " · ".join(parts)

    def _context_embed(self, *, post: TwitterOutput, header: str, budget: int) -> Embed:
        """One grey card for a post this expansion only quotes or replies to.

        Its own URL rather than the target's, which is what keeps Discord from folding it into the
        image gallery: embeds sharing a URL are merged, and that merging is exactly what turns the
        target's extra images into one gallery.
        """
        # Beside a short post, a lone context card's share of the message exceeds what one
        # description may carry.
        limit = min(
            max(budget, 0), DISCORD_EMBED_DESCRIPTION_LIMIT - utf16_length(value=f"{header}\n")
        )
        body = clip_to_utf16_limit(text=post.text, limit=limit, notice=TRUNCATION_NOTICE)
        embed = Embed(
            description=f"{header}\n{body}" if body else header,
            url=post.url,
            colour=Colour(value=CONTEXT_CARD_COLOR),
        )
        author = self._author_label(post=post)
        if author:
            embed.set_author(name=author, url=post.url, icon_url=post.author_icon_url or None)
        return embed

    def _build_embeds(self, *, conversation: TwitterConversation) -> list[Embed]:
        """Builds the whole expansion: the post it replies to, the post, its images, its quote."""
        post = conversation.target
        if post is None:
            return []

        context_count = bool(conversation.parent) + bool(post.quoted)
        hint = VIDEO_HINT.format(url=self._video_link(post=post)) if post.video_urls else ""
        truncated = _TRUNCATED_NOTICE if post.is_truncated else ""
        # The reader keeps at most four media per post, so the shared cap binds on nothing here;
        # it is what keeps the card bounded if that reader slice is ever widened.
        shown = post.image_urls[:POST_CARD_MAX_IMAGES]
        # A video's poster frame stands in for the image a clip has none of, so a video post is
        # not a card with nothing on it. Stills win when the post has both.
        poster = next(iter(post.video_poster_urls), "")
        embeds = post_card_embeds(
            post=post,
            color=self.EMBED_COLOR,
            author=self._author_label(post=post),
            footer=self._footer_text(post=post, shown_images=len(shown)),
            images=shown or ([poster] if poster else []),
            suffix=truncated + hint,
            reserve=_CONTEXT_RESERVE * context_count,
        )

        budget = context_card_budget(card=embeds[0]) // max(context_count, 1)
        if conversation.parent is not None:
            embeds.insert(
                0,
                self._context_embed(
                    post=conversation.parent, header=_PARENT_HEADER, budget=budget
                ),
            )
        if post.quoted is not None:
            embeds.append(
                self._context_embed(post=post.quoted, header=_QUOTED_HEADER, budget=budget)
            )
        return embeds


def setup(bot: commands.Bot) -> None:
    """Adds the TwitterCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(TwitterCogs(bot), override=True)
