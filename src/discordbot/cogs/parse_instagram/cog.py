"""Expands a public Instagram post URL into Discord embeds.

Nothing is downloaded. Images ride into the embeds as URLs for Discord to fetch itself, so this
cog needs no scratch directory, no media-delivery planner and no size ceiling.

`utils/expansion_cog.py` owns everything around the card. What is here is the card: one post, its
carousel, and the comment a permalink singled out.
"""

import asyncio
import contextlib

import logfire
from nextcord import Color, Embed, Message
from nextcord.ext import commands

from discordbot.typings.timeouts import INSTAGRAM_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.expansion_cog import ExpansionCog, ExpansionDelivery
from discordbot.utils.discord_embeds import utf16_length, clip_to_utf16_limit
from discordbot.services.platforms.instagram import (
    INSTAGRAM_URL_RE,
    InstagramOutput,
    InstagramDownloader,
    InstagramConversation,
    is_instagram_post_url,
)

# Instagram's own accent, so the card reads as an Instagram post at a glance, and a neutral grey
# for a comment so the two never look like the same kind of thing. Deliberately NOT in
# `typings/colors.py`: that palette is Discord's own semantic set, and a third party's brand
# colour belongs to the card wearing it.
_EMBED_COLOR = 0xE4405F
_COMMENT_COLOR = 0x65686C

# What fits before the expansion starts scrolling the channel, against a carousel that routinely
# carries nine or ten. The footer says how many were left behind and the embed's own link leads
# to the rest. Not a `context_budgets` constant: nothing here reaches a model, this bounds a
# rendered message.
_MAX_IMAGES = 4

# Discord's own ceiling on `embed.description`, and its message-wide ceiling across every embed in
# one send. The second is why the comment is budgeted against what the post spent rather than
# clipped on its own: clipping the two independently lets their sum reject the whole send with a
# 400, losing the expansion instead of trimming it.
_EMBED_DESCRIPTION_LIMIT = 4096
_EMBED_TOTAL_LENGTH_LIMIT = 6000

# What the post gives up so a comment card always fits beside it. Every measurement here is in
# UTF-16 units (`utf16_length`), Discord's own; the slack on top covers what the budget does not
# measure at all — the comment card's own author line and the blank line under its header.
_COMMENT_RESERVE = 2000
_BUDGET_SLACK = 400

_TRUNCATION_NOTICE = "\n\n⋯（全文請看原貼文）"
_VIDEO_HINT = "\n\n🎬 [點此觀看影片]({url})"
_COMMENT_HEADER = "💬 **指定的留言**"


def _author_label(*, post: InstagramOutput) -> str:
    """The author line: the display name when the page carried one, always with the handle.

    Whichever half the page served is used on its own rather than dropping the line, since an
    author line missing entirely reads as an anonymous post.
    """
    if post.author_full_name and post.author_name:
        return f"{post.author_full_name} (@{post.author_name})"
    if post.author_name:
        return f"@{post.author_name}"
    return post.author_full_name


class InstagramCogs(ExpansionCog[InstagramConversation]):
    """Expands Instagram post links into Discord embeds.

    Attributes:
        downloader_factory: Builds the reader; the seam a test replaces to stay off the network.
    """

    SOURCE = "instagram"
    PLATFORM = "Instagram"
    URL_PATTERN = INSTAGRAM_URL_RE
    PLACEHOLDER_TEXT = "-# 正在讀取 Instagram 貼文⋯"

    def __init__(self, bot: commands.Bot):
        """Initializes the InstagramCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        super().__init__(bot=bot)
        self.downloader_factory = InstagramDownloader

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

    async def read(
        self, *, message: Message, url: str, stack: contextlib.AsyncExitStack
    ) -> InstagramConversation:
        """Reads the post under a wall-clock bound.

        One blocking page fetch plus a walk over ~800KB of JSON, so it runs off the event loop.

        Args:
            message: Unused; nothing here logs.
            url: The post to read.
            stack: Unused; nothing here outlives the read.

        Returns:
            The parsed conversation.
        """
        del message, stack
        downloader = self.downloader_factory()
        async with asyncio.timeout(delay=INSTAGRAM_EXPAND_TIMEOUT_SECONDS):
            return await asyncio.to_thread(downloader.parse_metadata, url=url)

    async def build_delivery(
        self, *, message: Message, url: str, parsed: InstagramConversation
    ) -> ExpansionDelivery | None:
        """Builds the card, refusing a post with nothing showable in it.

        A post that comes back unreadable is a private, deleted or login-walled one: an ordinary
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
                "Instagram post is not readable; nothing to expand", url=url, message_id=message.id
            )
            return None
        return ExpansionDelivery(embeds=self._build_embeds(conversation=parsed))

    @staticmethod
    def _footer_text(*, post: InstagramOutput, shown_images: int) -> str:
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
        remaining_images = len(post.image_urls) - shown_images
        if remaining_images > 0:
            parts.append(f"🖼️ 另有 {remaining_images} 張")
        remaining_videos = len(post.video_urls) - 1
        if remaining_videos > 0:
            parts.append(f"🎬 另有 {remaining_videos} 部影片")
        return " · ".join(parts)

    @staticmethod
    def _comment_embed(*, comment: InstagramOutput, post_url: str, budget: int) -> Embed:
        """The card for the one comment a `/c/<id>/` permalink singled out.

        Grey rather than the post's accent, and headed by a line saying what it is: without both,
        a second card under the post reads as a second post rather than as a reply to this one.

        Its URL is the comment's own permalink, which differs from the post's — that is what keeps
        it OUT of the image gallery, since Discord merges embeds by URL. Instagram will not serve
        that URL's payload to this bot, but it is the right thing to hand a reader.
        """
        body = clip_to_utf16_limit(
            text=comment.text,
            limit=budget - utf16_length(value=_COMMENT_HEADER),
            notice=_TRUNCATION_NOTICE,
        )
        embed = Embed(
            description=f"{_COMMENT_HEADER}\n\n{body}",
            url=f"{post_url.rstrip('/')}/c/{comment.comment_id}/",
            color=Color(value=_COMMENT_COLOR),
            timestamp=comment.taken_at,
        )
        if comment.author_name:
            embed.set_author(
                name=f"@{comment.author_name}", icon_url=comment.author_icon_url or None
            )
        return embed

    def _build_embeds(self, *, conversation: InstagramConversation) -> list[Embed]:
        """Builds the whole expansion: the post, its images, and the named comment if any.

        Images past the first each become a bare embed reusing the post's URL, which is what makes
        Discord merge them into one gallery under the post rather than stacking separate cards.
        """
        post = conversation.target
        if post is None:
            return []
        # A reel renders as a card with nothing in it otherwise: its media is the clip, and this
        # cog uploads nothing, so a link is the whole of what can be shown. It points at the POST
        # rather than at `video_urls[0]`, which is Instagram's own signed CDN URL and expires
        # within days — the embed does not, so the link in it has to outlive the fetch. The hint's
        # length is reserved BEFORE the clip rather than appended after, or a post already at the
        # ceiling carries the hint past it and Discord rejects the send.
        hint = _VIDEO_HINT.format(url=post.url) if post.video_urls else ""
        comment = conversation.selected_comment
        post_limit = _EMBED_DESCRIPTION_LIMIT - utf16_length(value=hint)
        if comment is not None:
            post_limit = min(post_limit, _EMBED_TOTAL_LENGTH_LIMIT - _COMMENT_RESERVE)
        description = (
            clip_to_utf16_limit(text=post.text, limit=post_limit, notice=_TRUNCATION_NOTICE) + hint
        )
        main = Embed(
            description=description or None,
            url=post.url,
            color=Color(value=_EMBED_COLOR),
            timestamp=post.taken_at,
        )
        author = _author_label(post=post)
        if author:
            main.set_author(name=author, url=post.url, icon_url=post.author_icon_url or None)
        shown = post.image_urls[:_MAX_IMAGES]
        if shown:
            main.set_image(url=shown[0])
        main.set_footer(text=self._footer_text(post=post, shown_images=len(shown)))
        embeds = [main]
        for image_url in shown[1:]:
            extra = Embed(url=post.url)
            extra.set_image(url=image_url)
            embeds.append(extra)
        if comment is not None:
            spent = sum(
                utf16_length(value=text)
                for text in (description, main.footer.text, main.author.name)
                if isinstance(text, str)
            )
            embeds.append(
                self._comment_embed(
                    comment=comment,
                    post_url=post.url,
                    budget=_EMBED_TOTAL_LENGTH_LIMIT - spent - _BUDGET_SLACK,
                )
            )
        return embeds


def setup(bot: commands.Bot) -> None:
    """Adds the InstagramCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(InstagramCogs(bot), override=True)
