"""Cog that expands public Instagram post URLs into Discord embeds.

Mirrors `parse_facebook` down to the Discord limits it learned about, and differs only where
Instagram does: the counters are likes and comments with no share figure, the author is a
handle rather than a display name, and a comment permalink is a path segment (`/c/<id>/`)
rather than a query parameter.

Expansion is skipped when the message is addressed to the bot (a DM, or an explicit mention):
`gen_reply` reads the linked post and answers about it, so expanding as well would fetch the
same page twice and post a card nobody asked for. `is_addressed_to_bot` is the single predicate
deciding which of the two runs.

Nothing is downloaded here. Images ride into the embeds as URLs for Discord to fetch itself,
which is also why this cog needs no scratch directory, no media-delivery planner and no size
ceiling: the only bytes it ever sends are the embed JSON.

There is deliberately no kill-switch, for the reason `parse_facebook` states: turning this off
means deleting the cog.

Threads, Facebook, Instagram and Douyin are one feature four times over, and what makes them
one is `utils/expansion_placeholder.py`: the reply slot claimed before the read starts, the
five reactions and what each of them means, the restart sweep, and the rule that a failed
expansion says nothing in the channel at all. Read that module before changing anything here
that a reader would notice, and `tests/test_expansion_contract.py` before adding a fifth
source. What stays per platform is the card: this one renders one post, its carousel and the
comment a permalink singled out.
"""

import asyncio

import logfire
from nextcord import Color, Embed, Message, NotFound
from nextcord.ext import commands

from discordbot.typings.emojis import INSTAGRAM_EMOJI
from discordbot.utils.mentions import is_addressed_to_bot
from discordbot.utils.reactions import update_reaction
from discordbot.typings.timeouts import INSTAGRAM_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.discord_embeds import utf16_length, clip_to_utf16_limit
from discordbot.utils.expansion_placeholder import (
    EXPANSION_DONE_EMOJI,
    EXPANSION_FAILED_EMOJI,
    EXPANSION_WORKING_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    ExpansionPlaceholder,
    expansion_failure_emoji,
    send_expansion_placeholder,
    report_expansion_read_failure,
    resume_expansion_placeholders,
    report_expansion_delivery_failure,
)
from discordbot.services.platforms.instagram import (
    INSTAGRAM_URL_RE,
    InstagramOutput,
    InstagramDownloader,
    InstagramConversation,
    is_instagram_post_url,
)

# Instagram's own accent, so the card reads as an Instagram post at a glance, and the same
# neutral grey `parse_facebook` gives a comment. Deliberately NOT in `typings/colors.py`: that
# palette is Discord's own semantic set, and a third party's brand colour belongs to the card
# wearing it.
_EMBED_COLOR = 0xE4405F
_COMMENT_COLOR = 0x65686C

# Four is what fits before the expansion starts scrolling the channel, and an Instagram carousel
# routinely carries nine or ten. The footer says how many were left behind and the embed's own
# link leads to the rest. Not a `context_budgets` constant: nothing here reaches a model, this
# bounds a rendered message.
_MAX_IMAGES = 4

# Discord's own ceiling on `embed.description`, and its message-wide ceiling across every embed
# in one send. The second is why the comment is budgeted against what the post spent rather than
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
_PLACEHOLDER_TEXT = "-# 正在讀取 Instagram 貼文⋯"
# Owns this cog's rows in the pending-expansion table. Keyed as in `LINK_SOURCE_EMOJIS`, which
# `tests/test_link_source_emojis.py` pins, so the four cogs and the reply path name a platform
# the same way rather than each inventing a spelling.
_SOURCE = "instagram"


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


class InstagramCogs(commands.Cog):
    """Expands Instagram post links into Discord embeds.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        downloader_factory: Builds the reader; the seam a test replaces to stay off the network.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the InstagramCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.downloader_factory = InstagramDownloader
        self._resume_started = False

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Runs again the expansions a restart interrupted (once per process).

        `on_ready` fires on every gateway reconnect, so the flag guards it to one sweep.
        """
        if self._resume_started:
            return
        self._resume_started = True
        await resume_expansion_placeholders(bot=self.bot, source=_SOURCE, expand=self._expand)

    @staticmethod
    def _footer_text(*, post: InstagramOutput, shown_images: int) -> str:
        """The counter line, plus what the image cap and the single video hint left behind.

        The videos are counted separately from the images because a mixed carousel is ordinary
        on Instagram and only its first video gets a link: counting images alone would report a
        five-image, five-video post as having one picture left over and stay silent about four
        clips. A count is shown only when it is positive, since Instagram serves `-1` rather
        than a number for a post whose author hid its likes.
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

        Its URL is the comment's own permalink, which differs from the post's — that is what
        keeps it OUT of the image gallery, since Discord merges embeds by URL. Instagram will not
        serve that URL's payload to this bot, but it is the right thing to hand a reader.
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

        Images past the first each become a bare embed reusing the post's URL, which is what
        makes Discord merge them into one gallery under the post rather than stacking separate
        cards.
        """
        post = conversation.target
        if post is None:
            return []
        # A reel renders as a card with nothing in it otherwise: its media is the clip, and this
        # cog uploads nothing, so a link is the whole of what can be shown. It points at the
        # POST rather than at `video_urls[0]`, which is Instagram's own signed CDN URL and
        # expires within days — the embed does not, so the link in it has to outlive the fetch.
        # The hint's length is reserved BEFORE the clip rather than appended after, or a post
        # already at the ceiling carries the hint past it and Discord rejects the send.
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

    async def _mark_failed(self, *, message: Message, current_emoji: str | None) -> None:
        """Paints the failure cross, naming the platform when nothing else on the message does.

        `current_emoji` is None only when claiming the reply slot failed, before either mark
        went on. The cross alone cannot say WHICH link died, and on a message carrying two
        that is the whole of what someone needs, so the platform marker goes on first.
        """
        if current_emoji is None:
            await update_reaction(message=message, bot_user=self.bot.user, emoji=INSTAGRAM_EMOJI)
        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji=EXPANSION_FAILED_EMOJI,
            previous=current_emoji,
        )

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and expands Instagram links.

        Args:
            message: The message that was sent.
        """
        if message.author.bot:
            return

        match = INSTAGRAM_URL_RE.search(string=message.content)
        if not match:
            return

        # The regex matches the host, not the path, so a profile or the home page would
        # otherwise earn a failure reaction on a link that was never a post.
        if not is_instagram_post_url(url=match.group(0)):
            return

        # A link addressed to the bot is gen_reply's to answer about, not ours to expand; see
        # the module docstring. Checked after the URL match so the common no-link message costs
        # one regex, not two.
        if is_addressed_to_bot(message=message, bot_user=self.bot.user):
            return

        url = match.group(0)
        # Nothing is on the message yet, and the outer handler below is reachable before
        # anything is: claiming the reply slot is itself a step that can fail. Left unset
        # rather than pre-filled so a failure mark removes a reaction only when one is there.
        current_emoji: str | None = None
        try:
            placeholder = await send_expansion_placeholder(
                message=message, text=_PLACEHOLDER_TEXT, source=_SOURCE, url=url
            )
            if placeholder is None:
                # A channel that refused the placeholder will refuse the card too, so the
                # read is never started.
                await self._mark_failed(message=message, current_emoji=current_emoji)
                return
            # Persistent marker (added directly, not through the status chain, which replaces
            # its own reaction) saying an Instagram post was read, and the working ring under it.
            # Both go on AFTER the reply slot is claimed: they share one per-channel rate-limit
            # bucket that a message send does not, so claiming first is what stops the card
            # queueing behind them. `gen_reply` adds the same marker on the path it takes
            # instead of this one, so every read is marked the same way whichever cog did it.
            await update_reaction(message=message, bot_user=self.bot.user, emoji=INSTAGRAM_EMOJI)
            current_emoji = await update_reaction(
                message=message, bot_user=self.bot.user, emoji=EXPANSION_WORKING_EMOJI
            )
            try:
                await self._expand(
                    message=message, url=url, current_emoji=current_emoji, placeholder=placeholder
                )
            finally:
                # Every failure below returns rather than raising, so this one line covers
                # all of them: an expansion that delivered nothing leaves nothing behind.
                # Once delivered it is a no-op.
                await placeholder.discard()
        # Broad on purpose: the listener's last line of defence, so nothing escapes into the
        # dispatcher and every failure still reaches the user as a reaction.
        except Exception as error:
            logfire.error(
                "Instagram expansion failed outside the parse and delivery steps",
                url=url,
                message_id=message.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            await self._mark_failed(message=message, current_emoji=current_emoji)

    async def _expand(
        self, *, message: Message, url: str, current_emoji: str, placeholder: ExpansionPlaceholder
    ) -> None:
        """Reads the post under a wall-clock bound and hands a readable one to `_deliver`.

        The read is one blocking page fetch plus a walk over ~800KB of JSON, so it runs off the
        event loop. A post that comes back unreadable is a private, deleted or login-walled one:
        that is an ordinary outcome for a link someone pasted rather than a defect, so it takes
        the unreadable mark and never the cross, which says the bot itself broke.
        """
        downloader = self.downloader_factory()
        try:
            async with asyncio.timeout(delay=INSTAGRAM_EXPAND_TIMEOUT_SECONDS):
                conversation = await asyncio.to_thread(downloader.parse_metadata, url=url)
        # Broad on purpose: a fetch or parse failure must not escape into the listener; the
        # reaction is the user-visible outcome, and which one it is comes off the error's class
        # rather than a check here, so all four cogs answer a refusal the same way.
        except Exception as error:
            report_expansion_read_failure(
                error=error, platform="Instagram", url=url, message_id=message.id
            )
            await update_reaction(
                message=message,
                bot_user=self.bot.user,
                emoji=expansion_failure_emoji(error=error),
                previous=current_emoji,
            )
            return

        target = conversation.target
        if target is None or not target.is_readable:
            logfire.info(
                "Instagram post is not readable; nothing to expand", url=url, message_id=message.id
            )
            # The page came back and there is nothing showable in it, which is the unreadable
            # mark rather than the failure cross: nothing about the bot went wrong and a retry
            # would find the same private or deleted post.
            await update_reaction(
                message=message,
                bot_user=self.bot.user,
                emoji=EXPANSION_UNREADABLE_EMOJI,
                previous=current_emoji,
            )
            return

        await self._deliver(
            message=message,
            conversation=conversation,
            current_emoji=current_emoji,
            placeholder=placeholder,
        )

    async def _deliver(
        self,
        *,
        message: Message,
        conversation: InstagramConversation,
        current_emoji: str,
        placeholder: ExpansionPlaceholder,
    ) -> None:
        """Edits the expansion onto the placeholder and marks the source message done."""
        embeds = self._build_embeds(conversation=conversation)
        post_url = conversation.target.url if conversation.target else ""
        # Broad on purpose: the delivery step must never escape into the listener, and its
        # failures split three ways — the placeholder went away, the bot lacks a permission,
        # or something unexpected lost the expansion.
        try:
            try:
                await message.edit(suppress=True)
            # Deleting the link while the post was being read is a withdrawal: before the
            # placeholder existed the late reply was simply refused, and this keeps that,
            # since a reply outlives the message it answers.
            except NotFound:
                logfire.info(
                    "Instagram expansion target is gone",
                    url=post_url,
                    message_id=message.id,
                    channel_id=message.channel.id,
                )
                return
            # Broad on purpose: hiding Discord's own preview is cosmetic and must not abort the
            # expansion. A persistent Forbidden means the guild lacks Manage Messages.
            except Exception as error:
                logfire.warn(
                    "Could not suppress the source message embed",
                    message_id=message.id,
                    guild_id=message.guild.id if message.guild else None,
                    error_type=type(error).__name__,
                    _exc_info=error,
                )

            await placeholder.deliver(embeds=embeds)
        except Exception as error:
            report_expansion_delivery_failure(
                error=error,
                platform="Instagram",
                url=post_url,
                message_id=message.id,
                channel_id=message.channel.id,
            )
            await self._mark_failed(message=message, current_emoji=current_emoji)
            return

        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji=EXPANSION_DONE_EMOJI,
            previous=current_emoji,
        )


def setup(bot: commands.Bot) -> None:
    """Adds the InstagramCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(InstagramCogs(bot), override=True)
