"""Expands a Twitter (x.com) post URL into Discord embeds.

Discord renders nothing useful for an x.com link on its own — the page it fetches is the same
empty SPA shell `services/platforms/twitter.py` documents — so a pasted link sits in the channel
as a bare URL. This turns it into the post.

Nothing is downloaded: the stills and a video's poster frame ride to Discord as URLs for it to
fetch itself, and the clip is a link. That is a choice rather than a limit — Twitter's mp4 is
directly fetchable — so a deployment that wants the file has `/download_video`.

`utils/expansion_cog.py` owns everything around the card. What is here is the card, and two
things it has to say out loud because the platform will not: a post past roughly 275 characters
is served truncated with no marker of any kind, and the endpoint serves no replies at all, so the
reply figure in the footer counts a conversation nothing here can show.
"""

import asyncio
import contextlib

import logfire
from nextcord import Embed, Colour, Message
from nextcord.ext import commands

from discordbot.typings.timeouts import TWITTER_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.expansion_cog import ExpansionCog, ExpansionDelivery
from discordbot.utils.discord_embeds import utf16_length, clip_to_utf16_limit
from discordbot.services.platforms.twitter import (
    TWITTER_URL_RE,
    TwitterOutput,
    TwitterDownloader,
    TwitterConversation,
)

# The old Twitter blue rather than X's black. The rail down an embed's left edge exists to say
# which platform a card came from at a glance, and black does not: it reads as a hairline on
# Discord's light theme and vanishes into the dark one. The grey is for a post this card only
# quotes or replies to, so the one being linked is never confused with its context. Deliberately
# NOT in `typings/colors.py`: that palette is Discord's own semantic set, and a third party's
# brand colour belongs to the card wearing it.
_EMBED_COLOR = 0x1DA1F2
_CONTEXT_COLOR = 0x65686C

# Twitter's own cap is four, so this bounds nothing today; it is here so a platform that raises
# its limit does not silently start scrolling the channel. Not a `context_budgets` constant:
# nothing here reaches a model, this bounds a rendered message.
_MAX_IMAGES = 4

# Discord's own ceiling on `embed.description`, and on the text of every embed in one message
# summed. A card carrying a parent and a quote spends three descriptions against the second one,
# which is why the context cards are budgeted rather than clipped on their own: overshooting makes
# Discord reject the WHOLE send, losing the expansion instead of trimming it.
_EMBED_DESCRIPTION_LIMIT = 4096
_EMBED_TOTAL_LENGTH_LIMIT = 6000

# What the post gives up so the two context cards always fit beside it. Every measurement is in
# UTF-16 units (`utf16_length`), Discord's own; the slack covers what the budget does not measure
# at all — each context card's author line and header.
_CONTEXT_RESERVE = 1200
_BUDGET_SLACK = 400

_TRUNCATION_NOTICE = "\n\n⋯（全文請看原貼文）"
# Said on the card rather than left to the reader, because Twitter marks a cut body in no way at
# all: no ellipsis, no "show more", nothing but a key saying a longer version exists that the
# endpoint will not serve. A card that stays quiet passes a fragment off as the whole post.
_TRUNCATED_NOTICE = "\n\n-# ⋯這則貼文較長，Twitter 只提供開頭"
_VIDEO_HINT = "\n\n🎬 [點此觀看影片]({url})"
_PARENT_HEADER = "↩️ **回覆的貼文**"
_QUOTED_HEADER = "💬 **引用的貼文**"


class TwitterCogs(ExpansionCog[TwitterConversation]):
    """Expands Twitter links into Discord embeds.

    Attributes:
        downloader_factory: Builds the reader; the seam a test replaces to keep an expansion off
            the network.
    """

    SOURCE = "twitter"
    PLATFORM = "Twitter"
    URL_PATTERN = TWITTER_URL_RE
    PLACEHOLDER_TEXT = "-# 正在讀取 Twitter 貼文⋯"

    def __init__(self, bot: commands.Bot):
        """Initializes the TwitterCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        super().__init__(bot=bot)
        self.downloader_factory = TwitterDownloader

    async def read(
        self, *, message: Message, url: str, stack: contextlib.AsyncExitStack
    ) -> TwitterConversation:
        """Reads the post under a wall-clock bound.

        One blocking request, so it runs off the event loop.

        Args:
            message: Unused; nothing here logs.
            url: The post to read.
            stack: Unused; nothing here outlives the read.

        Returns:
            The parsed conversation.
        """
        del message, stack
        downloader = self.downloader_factory()
        async with asyncio.timeout(delay=TWITTER_EXPAND_TIMEOUT_SECONDS):
            return await asyncio.to_thread(downloader.parse_metadata, url=url)

    async def build_delivery(
        self, *, message: Message, url: str, parsed: TwitterConversation
    ) -> ExpansionDelivery | None:
        """Builds the card, refusing a post with nothing showable in it.

        A post that comes back unreadable is a deleted, protected or suspended one: an ordinary
        outcome for a link someone pasted rather than a defect.

        Args:
            message: The message carrying the link.
            url: The post that was read.
            parsed: The parsed conversation.

        Returns:
            The card, or None when there is nothing to show.
        """
        target = parsed.target
        if target is None or not target.is_readable:
            logfire.info(
                "Twitter post is not readable; nothing to expand", url=url, message_id=message.id
            )
            return None
        return ExpansionDelivery(embeds=self._build_embeds(conversation=parsed))

    @staticmethod
    def _footer_text(*, post: TwitterOutput, shown_images: int) -> str:
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
        omitted = len(post.image_urls) - shown_images
        if omitted > 0:
            parts.append(f"另有 {omitted} 張圖片")
        return " · ".join(parts)

    @staticmethod
    def _context_embed(*, post: TwitterOutput, header: str, budget: int) -> Embed:
        """One grey card for a post this expansion only quotes or replies to.

        Its own URL rather than the target's, which is what keeps Discord from folding it into the
        image gallery: embeds sharing a URL are merged, and that merging is exactly what turns the
        target's extra images into one gallery.
        """
        body = clip_to_utf16_limit(text=post.text, limit=max(budget, 0), notice=_TRUNCATION_NOTICE)
        embed = Embed(
            description=f"{header}\n{body}" if body else header,
            url=post.url,
            colour=Colour(value=_CONTEXT_COLOR),
        )
        if post.author_name:
            embed.set_author(
                name=f"@{post.author_name}", url=post.url, icon_url=post.author_icon_url or None
            )
        return embed

    def _build_embeds(self, *, conversation: TwitterConversation) -> list[Embed]:
        """Builds the whole expansion: the post it replies to, the post, its images, its quote.

        Images past the first each become a bare embed reusing the post's URL, which is what makes
        Discord merge them into one gallery under the post rather than stacking separate cards.
        """
        post = conversation.target
        if post is None:
            return []

        # The parent is the chain's earlier entry when there is one. Only ever one: the endpoint
        # embeds a single ancestor and walking further costs a request per hop.
        parent = conversation.chain[0] if len(conversation.chain) > 1 else None
        context_count = bool(parent) + bool(post.quoted)

        hint = _VIDEO_HINT.format(url=post.video_urls[0]) if post.video_urls else ""
        truncated = _TRUNCATED_NOTICE if post.is_truncated else ""
        # The two notices are reserved BEFORE the body rather than appended after, or a post
        # already at the ceiling carries them past it and Discord rejects the send.
        limit = _EMBED_DESCRIPTION_LIMIT - utf16_length(value=hint + truncated)
        if context_count:
            limit = min(limit, _EMBED_TOTAL_LENGTH_LIMIT - _CONTEXT_RESERVE * context_count)
        description = (
            clip_to_utf16_limit(text=post.text, limit=limit, notice=_TRUNCATION_NOTICE)
            + truncated
            + hint
        )

        main = Embed(
            description=description or None,
            url=post.url,
            colour=Colour(value=_EMBED_COLOR),
            timestamp=post.taken_at,
        )
        if post.author_name:
            main.set_author(
                name=f"@{post.author_name}", url=post.url, icon_url=post.author_icon_url or None
            )
        # A video's poster frame stands in for the image a clip has none of, so a video post is
        # not a card with nothing on it. Stills win when the post has both.
        shown = post.image_urls[:_MAX_IMAGES]
        preview = shown[0] if shown else next(iter(post.video_poster_urls), "")
        if preview:
            main.set_image(url=preview)
        main.set_footer(text=self._footer_text(post=post, shown_images=len(shown)))

        embeds = [main]
        for image_url in shown[1:]:
            extra = Embed(url=post.url)
            extra.set_image(url=image_url)
            embeds.append(extra)

        spent = sum(
            utf16_length(value=text)
            for text in (description, main.footer.text, main.author.name)
            if isinstance(text, str)
        )
        budget = (_EMBED_TOTAL_LENGTH_LIMIT - spent - _BUDGET_SLACK) // max(context_count, 1)
        if parent is not None:
            embeds.insert(
                0, self._context_embed(post=parent, header=_PARENT_HEADER, budget=budget)
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
