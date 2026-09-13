"""Cog that expands Twitter (x.com) post URLs into Discord embeds.

Discord renders nothing useful for an x.com link on its own — the page it fetches is the same
empty SPA shell `services/platforms/twitter.py` documents — so a pasted link sits in the channel
as a bare URL. This turns it into the post.

Expansion is skipped when the message is addressed to the bot (a DM, or an explicit mention):
`gen_reply` reads the linked post and answers about it, so expanding as well would read the same
post twice and post a card nobody asked for. `is_addressed_to_bot` is the single predicate
deciding which of the two runs.

Threads, Facebook, Instagram, Douyin and Twitter are one feature five times over, and what makes
them one is `utils/expansion_placeholder.py`: the reply slot claimed before the read starts, the
five reactions and what each of them means, the restart sweep, and the rule that a failed
expansion says nothing in the channel at all. Read that module before changing anything here that
a reader would notice, and `tests/test_expansion_contract.py` before adding a sixth source.

What stays per platform is the card. This one downloads NOTHING: the stills and a video's poster
frame ride to Discord as URLs for it to fetch itself, and the clip is a link. That puts this cog
in `parse_facebook`'s shape — no scratch directory, no delivery planner, no size ceiling — rather
than `parse_douyin`'s, which is a deliberate choice here and not a limitation: Twitter's mp4 is
directly fetchable, and Facebook's is not.

Two things the card has to say out loud, because the platform will not. A post past roughly 275
characters is served truncated with no marker of any kind, and the endpoint serves no replies at
all — so the reply figure in the footer is a count of a conversation nothing here can show.
"""

import asyncio

import logfire
from nextcord import Embed, Colour, Message, NotFound
from nextcord.ext import commands

from discordbot.typings.emojis import TWITTER_EMOJI
from discordbot.utils.mentions import is_addressed_to_bot
from discordbot.utils.reactions import update_reaction
from discordbot.typings.timeouts import TWITTER_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.discord_embeds import utf16_length, clip_to_utf16_limit
from discordbot.services.platforms.twitter import (
    TWITTER_URL_RE,
    TwitterOutput,
    TwitterDownloader,
    TwitterConversation,
)
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

# The old Twitter blue rather than X's black. The rail down an embed's left edge exists to say
# which platform a card came from at a glance, and black does not: it reads as a hairline on
# Discord's light theme and vanishes into the dark one. The grey is for a post this card only
# quotes or replies to, so the one being linked is never confused with its context. Deliberately
# NOT in `typings/colors.py`, for the reason `parse_douyin` states of its brand red.
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
_PLACEHOLDER_TEXT = "-# 正在讀取 Twitter 貼文⋯"
_PARENT_HEADER = "↩️ **回覆的貼文**"
_QUOTED_HEADER = "💬 **引用的貼文**"

# Owns this cog's rows in the pending-expansion table. Keyed as in `LINK_SOURCE_EMOJIS`, which
# `tests/test_link_source_emojis.py` pins, so the five cogs and the reply path name a platform the
# same way rather than each inventing a spelling.
_SOURCE = "twitter"


class TwitterCogs(commands.Cog):
    """Expands Twitter links into Discord embeds.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        downloader_factory: Builds the reader; the seam a test replaces to keep an expansion off
            the network.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the TwitterCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.downloader_factory = TwitterDownloader
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

    async def _mark_failed(self, *, message: Message, current_emoji: str | None) -> None:
        """Paints the failure cross, naming the platform when nothing else on the message does.

        `current_emoji` is None only when claiming the reply slot failed, before either mark went
        on. The cross alone cannot say WHICH link died, and on a message carrying two that is the
        whole of what someone needs, so the platform marker goes on first.
        """
        if current_emoji is None:
            await update_reaction(message=message, bot_user=self.bot.user, emoji=TWITTER_EMOJI)
        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji=EXPANSION_FAILED_EMOJI,
            previous=current_emoji,
        )

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and expands Twitter links.

        Args:
            message: The message that was sent.
        """
        if message.author.bot:
            return

        match = TWITTER_URL_RE.search(string=message.content)
        if not match:
            return

        # No post-match guard, unlike Facebook, Instagram and Douyin: the pattern is path-anchored
        # on `/status/<digits>`, so a profile or the home page never matches in the first place.

        # A link addressed to the bot is gen_reply's to answer about, not ours to expand; see the
        # module docstring. Checked after the URL match so the common no-link message costs one
        # regex, not two.
        if is_addressed_to_bot(message=message, bot_user=self.bot.user):
            return

        url = match.group(0)
        # Nothing is on the message yet, and the outer handler below is reachable before anything
        # is: claiming the reply slot is itself a step that can fail. Left unset rather than
        # pre-filled so a failure mark removes a reaction only when one is there.
        current_emoji: str | None = None
        try:
            placeholder = await send_expansion_placeholder(
                message=message, text=_PLACEHOLDER_TEXT, source=_SOURCE, url=url
            )
            if placeholder is None:
                # A channel that refused the placeholder will refuse the card too, so the read is
                # never started.
                await self._mark_failed(message=message, current_emoji=current_emoji)
                return
            # Persistent marker saying a Twitter post was read, and the working ring under it.
            # Both go on AFTER the reply slot is claimed: they share one per-channel rate-limit
            # bucket that a message send does not, so claiming first is what stops the card
            # queueing behind them. `gen_reply` adds the same marker on the path it takes instead
            # of this one, so every read is marked the same way whichever cog did it.
            await update_reaction(message=message, bot_user=self.bot.user, emoji=TWITTER_EMOJI)
            current_emoji = await update_reaction(
                message=message, bot_user=self.bot.user, emoji=EXPANSION_WORKING_EMOJI
            )
            try:
                await self._expand(
                    message=message, url=url, current_emoji=current_emoji, placeholder=placeholder
                )
            finally:
                # Every failure below returns rather than raising, so this one line covers all of
                # them: an expansion that delivered nothing leaves nothing behind. Once delivered
                # it is a no-op.
                await placeholder.discard()
        # Broad on purpose: the listener's last line of defence, so nothing escapes into the
        # dispatcher and every failure still reaches the user as a reaction.
        except Exception as error:
            logfire.error(
                "Twitter expansion failed outside the parse and delivery steps",
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

        The read is one blocking request, so it runs off the event loop. A post that comes back
        unreadable is a deleted, protected or suspended one: an ordinary outcome for a link
        someone pasted rather than a defect, so it takes the unreadable mark and never the cross,
        which says the bot itself broke.
        """
        downloader = self.downloader_factory()
        try:
            async with asyncio.timeout(delay=TWITTER_EXPAND_TIMEOUT_SECONDS):
                conversation = await asyncio.to_thread(downloader.parse_metadata, url=url)
        # Broad on purpose: a fetch or parse failure must not escape into the listener; the
        # reaction is the user-visible outcome, and which one it is comes off the error's class
        # rather than a check here, so all five cogs answer a refusal the same way.
        except Exception as error:
            report_expansion_read_failure(
                error=error, platform="Twitter", url=url, message_id=message.id
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
                "Twitter post is not readable; nothing to expand", url=url, message_id=message.id
            )
            # The endpoint answered and there is nothing showable in it, which is the unreadable
            # mark rather than the failure cross: nothing about the bot went wrong and a retry
            # would find the same deleted or protected post.
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
        conversation: TwitterConversation,
        current_emoji: str,
        placeholder: ExpansionPlaceholder,
    ) -> None:
        """Edits the expansion onto the placeholder and marks the source message done."""
        embeds = self._build_embeds(conversation=conversation)
        post_url = conversation.target.url if conversation.target else ""
        # Broad on purpose: the delivery step must never escape into the listener, and its
        # failures split three ways — the placeholder went away, the bot lacks a permission, or
        # something unexpected lost the expansion.
        try:
            try:
                await message.edit(suppress=True)
            # Deleting the link while the post was being read is a withdrawal: before the
            # placeholder existed the late reply was simply refused, and this keeps that, since a
            # reply outlives the message it answers.
            except NotFound:
                logfire.info(
                    "Twitter expansion target is gone",
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
                platform="Twitter",
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
    """Adds the TwitterCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(TwitterCogs(bot), override=True)
