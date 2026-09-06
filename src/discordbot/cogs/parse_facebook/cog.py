"""Cog that expands public Facebook post URLs into Discord embeds.

Mirrors `parse_threads`, minus everything that cog needs for a conversation. A Facebook post
has no ancestors, no quoted post and no reply branches, so there is no embed budget to
allocate and no omitted-post notice to page: one post embed, its images, and optionally the
one comment the URL singled out.

Expansion is skipped when the message is addressed to the bot (a DM, or an explicit mention):
`gen_reply` reads the linked post and answers about it, so expanding as well would fetch the
same page twice and post a card nobody asked for. `is_addressed_to_bot` is the single
predicate deciding which of the two runs.

Nothing is downloaded here. Images ride into the embeds as URLs for Discord to fetch itself,
which is also why this cog needs no scratch directory, no media-delivery planner and no
size ceiling: the only bytes it ever sends are the embed JSON.

There is deliberately no kill-switch. `DOUYIN_AUTO_EXPAND_ENABLED` was the one precedent and
it was deleted in #636 rather than copied here: turning this off means deleting the cog.
"""

import asyncio

import logfire
from nextcord import Color, Embed, Message, NotFound, Forbidden, HTTPException, AllowedMentions
from nextcord.ext import commands

from discordbot.typings.emojis import FACEBOOK_EMOJI
from discordbot.utils.facebook import (
    FACEBOOK_URL_RE,
    FacebookOutput,
    FacebookDownloader,
    FacebookConversation,
    is_facebook_post_url,
)
from discordbot.utils.mentions import is_addressed_to_bot
from discordbot.utils.reactions import update_reaction
from discordbot.typings.timeouts import FACEBOOK_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.discord_embeds import embed_spacer_payload

# Facebook's own blue, so the card reads as a Facebook post at a glance, and a neutral grey for
# a comment so the two never look like the same kind of thing. Deliberately NOT in
# `typings/colors.py` for the reason `parse_douyin` states of its brand red: that palette is
# Discord's own semantic set, and a third party's brand colour belongs to the card wearing it.
_EMBED_COLOR = 0x1877F2
_COMMENT_COLOR = 0x65686C

# Four is what fits before the expansion starts scrolling the channel. Discord allows ten
# embeds per message and a gallery post can carry far more images than that, so the footer
# says how many were left behind and the embed's own link leads to the rest. Not a
# `context_budgets` constant: nothing here reaches a model, this bounds a rendered message.
_MAX_IMAGES = 4

# Discord's own ceiling on `embed.description`. A long post is cut rather than split across a
# second embed: the whole card is one post, and a reader who wants the tail has the link.
_EMBED_DESCRIPTION_LIMIT = 4096

# Discord counts every embed's text in one message toward a single ceiling, so the comment card
# is budgeted against what the post spent rather than clipped on its own: a long post plus a long
# comment would otherwise sum past it and Discord rejects the WHOLE send, losing the expansion
# rather than trimming it. `parse_threads/cog.py` carries the same limit for the same reason.
_EMBED_TOTAL_LENGTH_LIMIT = 6000

# What the post gives up so a comment card always fits beside it. The slack on top covers the
# footer, the author line, and the fact that Discord counts UTF-16 units, so one emoji costs two
# where `len` counts one.
_COMMENT_RESERVE = 2000
_BUDGET_SLACK = 400
_TRUNCATION_NOTICE = "\n\n⋯（全文請看原貼文）"
_VIDEO_HINT = "\n\n🎬 [點此觀看影片]({url})"
_COMMENT_HEADER = "💬 **指定的留言**"


def _clipped(*, text: str, limit: int) -> str:
    """Returns `text` within `limit`, marking the cut so a truncated post never reads as whole."""
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_NOTICE)] + _TRUNCATION_NOTICE


class FacebookCogs(commands.Cog):
    """Expands Facebook post links into Discord embeds.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        downloader_factory: Builds the reader; the seam a test replaces to stay off the network.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the FacebookCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.downloader_factory = FacebookDownloader

    @staticmethod
    def _footer_text(*, post: FacebookOutput, shown_images: int) -> str:
        """The counter line: where the post lives, how it did, and what was left out.

        The group name leads because it is the part a reader cannot get from the post itself,
        and it is simply absent for a page or profile post rather than being replaced by a
        placeholder, which is what keeps the line from having a hole in it.
        """
        parts = [post.group_name] if post.group_name else []
        if post.like_count:
            parts.append(f"👍 {post.like_count:,}")
        if post.comment_count:
            parts.append(f"💬 {post.comment_count:,}")
        if post.share_count:
            parts.append(f"↗️ {post.share_count:,}")
        remaining = len(post.image_urls) - shown_images
        if remaining > 0:
            parts.append(f"🖼️ 另有 {remaining} 張")
        return " · ".join(parts)

    @staticmethod
    def _comment_embed(*, comment: FacebookOutput, post_url: str, budget: int) -> Embed:
        """The card for the one comment a `?comment_id=` link singled out.

        Grey rather than Facebook blue, and headed by a line saying what it is: without both, a
        second card under the post reads as a second post rather than as a reply to this one.

        Its URL deliberately differs from the post's — it points at the comment — because that
        is what keeps it OUT of the image gallery below. Discord merges embeds by URL, so
        reusing the post's here would fold the comment into the pictures.
        """
        body = _clipped(text=comment.text, limit=max(budget - len(_COMMENT_HEADER), 0))
        # `&` when the post URL already carries a query, which `permalink.php` links always do.
        joiner = "&" if "?" in post_url else "?"
        embed = Embed(
            description=f"{_COMMENT_HEADER}\n\n{body}",
            url=f"{post_url}{joiner}comment_id={comment.comment_id}",
            color=Color(value=_COMMENT_COLOR),
            timestamp=comment.taken_at,
        )
        if comment.author_name:
            embed.set_author(name=comment.author_name, icon_url=comment.author_icon_url or None)
        return embed

    def _build_embeds(self, *, conversation: FacebookConversation) -> list[Embed]:
        """Builds the whole expansion: the post, its images, and the named comment if any.

        Images past the first each become a bare embed reusing the post's URL, which is what
        makes Discord merge them into one gallery under the post rather than stacking separate
        cards. The comment embed carries its own URL for the same reason inverted: a different
        link is what keeps it OUT of that gallery.
        """
        # A video post would otherwise render as a card with nothing in it: there is no file to
        # attach (see `utils/facebook.py`), so the link is the whole of what can be shown. Its
        # length is reserved BEFORE the clip rather than appended after, or a post already at the
        # ceiling carries the hint past it and Discord rejects the send.
        post = conversation.target
        if post is None:
            return []
        hint = _VIDEO_HINT.format(url=post.video_urls[0]) if post.video_urls else ""
        comment = conversation.selected_comment
        post_limit = _EMBED_DESCRIPTION_LIMIT - len(hint)
        if comment is not None:
            post_limit = min(post_limit, _EMBED_TOTAL_LENGTH_LIMIT - _COMMENT_RESERVE)
        description = _clipped(text=post.text, limit=post_limit) + hint
        main = Embed(
            description=description or None,
            url=post.url,
            color=Color(value=_EMBED_COLOR),
            timestamp=post.taken_at,
        )
        if post.author_name:
            main.set_author(
                name=post.author_name, url=post.url, icon_url=post.author_icon_url or None
            )
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
                len(text)
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

    async def _mark_failed(self, *, message: Message, current_emoji: str) -> None:
        """Replaces the working reaction with the failure cross."""
        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji="<:redcross:1517565100838355016>",
            previous=current_emoji,
        )

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and expands Facebook links.

        Args:
            message: The message that was sent.
        """
        if message.author.bot:
            return

        match = FACEBOOK_URL_RE.search(string=message.content)
        if not match:
            return

        # The regex matches the host, not the path, so a profile or group home page would
        # otherwise earn a failure reaction on a link that was never a post.
        if not is_facebook_post_url(url=match.group(0)):
            return

        # A link addressed to the bot is gen_reply's to answer about, not ours to expand; see
        # the module docstring. Checked after the URL match so the common no-link message costs
        # one regex, not two.
        if is_addressed_to_bot(message=message, bot_user=self.bot.user):
            return

        url = match.group(0)
        # Persistent marker (added directly, not through the status chain, which replaces its own
        # reaction) saying a Facebook post was read. `gen_reply` adds the same one on the path it
        # takes instead of this one, so every read is marked the same way whichever cog did it.
        await update_reaction(message=message, bot_user=self.bot.user, emoji=FACEBOOK_EMOJI)
        current_emoji = await update_reaction(message=message, bot_user=self.bot.user, emoji="🔗")

        try:
            await self._expand(message=message, url=url, current_emoji=current_emoji)
        # Broad on purpose: the listener's last line of defence, so nothing escapes into the
        # dispatcher and every failure still reaches the user as a reaction.
        except Exception as error:
            logfire.error(
                "Facebook expansion failed outside the parse and delivery steps",
                url=url,
                message_id=message.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            await self._mark_failed(message=message, current_emoji=current_emoji)

    async def _expand(self, *, message: Message, url: str, current_emoji: str) -> None:
        """Reads the post under a wall-clock bound and hands a readable one to `_deliver`.

        The read is one blocking page fetch plus a walk over ~950KB of JSON, so it runs off the
        event loop. A post that comes back unreadable is a private, deleted or login-walled one:
        that is the ordinary outcome for a link someone pasted, and it is reported with the same
        cross as a failure because from the channel's side there is no difference worth drawing.
        """
        downloader = self.downloader_factory()
        try:
            async with asyncio.timeout(delay=FACEBOOK_EXPAND_TIMEOUT_SECONDS):
                conversation = await asyncio.to_thread(downloader.parse_metadata, url=url)
        # Broad on purpose: a fetch or parse failure must not escape into the listener; the
        # cross reaction is the user-visible outcome. A timeout lands here as a plain failure.
        except Exception as error:
            logfire.warn(
                "Facebook parse failed",
                url=url,
                message_id=message.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            await self._mark_failed(message=message, current_emoji=current_emoji)
            return

        target = conversation.target
        if target is None or not target.is_readable:
            logfire.info(
                "Facebook post is not readable; nothing to expand", url=url, message_id=message.id
            )
            await self._mark_failed(message=message, current_emoji=current_emoji)
            return

        await self._deliver(
            message=message, conversation=conversation, current_emoji=current_emoji
        )

    async def _deliver(
        self, *, message: Message, conversation: FacebookConversation, current_emoji: str
    ) -> None:
        """Posts the expansion and marks the source message done."""
        embeds = self._build_embeds(conversation=conversation)
        post_url = conversation.target.url if conversation.target else ""
        # Broad on purpose: the delivery step must never escape into the listener, and its
        # failures split three ways — the source message went away, the bot lacks a permission,
        # or something unexpected lost the expansion.
        try:
            try:
                await message.edit(suppress=True)
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

            await message.reply(
                embeds=embeds,
                mention_author=False,
                allowed_mentions=AllowedMentions.none(),
                **embed_spacer_payload(embeds=embeds, is_edit=False, target=message),
            )
        except Exception as error:
            # A reply to a deleted source comes back as HTTP 50035, not only as NotFound.
            gone = isinstance(error, NotFound) or (
                isinstance(error, HTTPException) and error.code == 50035
            )
            if gone:
                logfire.info(
                    "Facebook expansion target is gone",
                    url=post_url,
                    message_id=message.id,
                    channel_id=message.channel.id,
                )
            elif isinstance(error, Forbidden):
                logfire.warn(
                    "Missing permission to post the Facebook expansion",
                    url=post_url,
                    message_id=message.id,
                    channel_id=message.channel.id,
                    error_type=type(error).__name__,
                    _exc_info=error,
                )
            else:
                logfire.error(
                    "Failed to send Facebook expansion",
                    url=post_url,
                    message_id=message.id,
                    channel_id=message.channel.id,
                    error_type=type(error).__name__,
                    _exc_info=error,
                )
            await self._mark_failed(message=message, current_emoji=current_emoji)
            return

        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji="<:greencheck:1517565102424068226>",
            previous=current_emoji,
        )


def setup(bot: commands.Bot) -> None:
    """Adds the FacebookCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(FacebookCogs(bot), override=True)
