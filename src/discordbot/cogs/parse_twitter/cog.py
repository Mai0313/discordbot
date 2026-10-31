"""Expands a Twitter (x.com) post URL into Discord embeds.

Discord renders nothing useful for an x.com link on its own — the page it fetches is the same
empty SPA shell `services/platforms/twitter.py` documents — so a pasted link sits in the channel
as a bare URL. This turns it into the post.

Nothing is downloaded here: the card names the stills and a video's poster frame by URL for the
shell to upload, and the clip is a link. That is a choice rather than a limit — Twitter's mp4 is
directly fetchable — so a deployment that wants the file has `/download_video`.

`utils/expansion_cog.py` owns everything around the card and the post's own embed. What is here
is the rest of the card — every post above it in the thread, as `parse_threads` shows one, and the
post it quotes — and one thing it has to say out loud because the platform will not: a post past
roughly 275 characters is served truncated with no marker of any kind. The replies the reader
also carries are `gen_reply`'s to read, as on Threads.
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
    omitted_posts_note,
    context_card_budget,
)
from discordbot.utils.discord_embeds import (
    DISCORD_EMBED_COUNT_LIMIT,
    DISCORD_EMBED_TOTAL_LIMIT,
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

# What the post gives up per context card so they always fit beside it, up to a ceiling that
# leaves the post four times the ~275 characters syndication ever serves of it: a thread of nine
# cards would otherwise reserve more than the whole message may carry.
_CONTEXT_RESERVE = 1200
_CONTEXT_RESERVE_CAP = DISCORD_EMBED_TOTAL_LIMIT - 1200

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

    def _author_label(self, post: TwitterOutput) -> str:
        """The handle, which is the only name a post carries here."""
        return f"@{post.author_name}" if post.author_name else ""

    def _footer_text(self, post: TwitterOutput, shown_images: int) -> str:
        """The counters, and what the card could not show."""
        parts = [f"♥ {post.like_count:,}", f"💬 {post.comment_count:,}"]
        parts.extend(self._omitted_media_notes(post=post, shown_images=shown_images))
        return " · ".join(parts)

    def _context_embed(
        self, post: TwitterOutput, header: str, budget: int, show_media: bool
    ) -> Embed:
        """One grey card for a post this expansion only quotes or shows above the linked one.

        Its own URL rather than the target's, which is what keeps Discord from folding it into the
        image gallery: embeds sharing a URL are merged, and that merging is exactly what turns the
        target's extra images into one gallery. With `show_media` it carries its first still, or a
        clip's poster frame and a link, since a reply on Twitter is often a picture and little else.
        """
        image = next(iter([*post.image_urls, *post.video_poster_urls]), "") if show_media else ""
        hint = (
            VIDEO_HINT.format(url=self._video_link(post=post))
            if show_media and post.video_urls
            else ""
        )
        lead = f"{header}\n" if header else ""
        # Beside a short post, a lone context card's share of the message exceeds what one
        # description may carry. The header and the hint come off this card's own share, since
        # the slack `context_card_budget` keeps is sized for author lines alone.
        spent = utf16_length(value=lead) + utf16_length(value=hint)
        limit = min(max(budget - spent, 0), DISCORD_EMBED_DESCRIPTION_LIMIT - spent)
        body = clip_to_utf16_limit(text=post.text, limit=limit, notice=TRUNCATION_NOTICE)
        embed = Embed(
            description=f"{lead}{body}{hint}".strip() or None,
            url=post.url,
            colour=Colour(value=CONTEXT_CARD_COLOR),
        )
        if image:
            embed.set_image(url=image)
        author = self._author_label(post=post)
        if author:
            embed.set_author(name=author, url=post.url, icon_url=post.author_icon_url or None)
        return embed

    def _build_embeds(self, conversation: TwitterConversation) -> list[Embed]:
        """Builds the whole expansion: the thread above the post, the post, its images, its quote.

        Discord carries ten embeds a message. The post's own (one per image) and its quote come
        first, and the posts above it fill what is left nearest first, the rest counted in the
        post's footer.
        """
        post = conversation.target
        if post is None:
            return []

        hint = VIDEO_HINT.format(url=self._video_link(post=post)) if post.video_urls else ""
        truncated = _TRUNCATED_NOTICE if post.is_truncated else ""
        # The reader keeps no more media per post than the card shows, so the cap binds on nothing
        # today; it keeps the card bounded if the reader ever keeps more.
        shown = post.image_urls[:POST_CARD_MAX_IMAGES]
        # A video's poster frame stands in for the image a clip has none of, so a video post is
        # not a card with nothing on it. Stills win when the post has both.
        poster = next(iter(post.video_poster_urls), "")
        images = shown or ([poster] if poster else [])
        room = DISCORD_EMBED_COUNT_LIMIT - max(len(images), 1) - bool(post.quoted)
        ancestors = conversation.chain[:-1][-room:]
        omitted = len(conversation.chain) - 1 - len(ancestors)
        context_count = len(ancestors) + bool(post.quoted)
        footer = self._footer_text(post=post, shown_images=len(shown))
        if omitted:
            footer = f"{footer} · {omitted_posts_note(count=omitted)}"
        embeds = post_card_embeds(
            post=post,
            color=self.EMBED_COLOR,
            author=self._author_label(post=post),
            footer=footer,
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
