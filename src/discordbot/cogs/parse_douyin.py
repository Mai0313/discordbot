"""Cog that expands Douyin post URLs into Discord attachments.

Unlike `/download_video`, which serves many platforms where a pasted link often just means
"look at this page", a Douyin link has no such ambiguity: posting one means "watch this",
so it is converted without anyone typing a command.

Expansion is skipped when the message is addressed to the bot (a DM, or an explicit
mention): `gen_reply` reads the linked post and answers about it, so expanding as well would
fetch the same post twice and post an attachment nobody asked for. The two paths are
mutually exclusive, and `is_addressed_to_bot` is the single predicate deciding which runs.

That predicate is deliberately coarser than `gen_reply`'s own guards, so a few addressed
messages get neither treatment: one typed inside an active research thread (the reply
pipeline skips those), and one the router sends to IMAGE / VIDEO (those routes discard the
link context). Both are rare enough to accept rather than couple the cogs together.

Douyin's WAF bans a share path for tens of minutes once it is hit hard, and this listener
sees every message in every channel, so the request-volume bounds in `_parse_douyin/fetch.py`
are load-bearing rather than defensive. A blocked request must never be reported as a missing
post: telling someone their working link is dead is the worst failure this feature can produce.
"""

from typing import ClassVar
import asyncio
import tempfile
import contextlib

import logfire
from nextcord import Embed, Message, AllowedMentions
from nextcord.ext import commands

from discordbot.utils.douyin import (
    DOUYIN_URL_RE,
    DouyinPost,
    DouyinDownload,
    DouyinDownloader,
    DouyinBlockedError,
    is_douyin_post_url,
)
from discordbot.typings.douyin import DouyinConfig
from discordbot.utils.mentions import is_addressed_to_bot
from discordbot.utils.reactions import update_reaction
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    DISCORD_ATTACHMENT_LIMIT,
    MediaItem,
    MediaPlan,
    upload_limit_for,
    build_media_delivery_planner,
)
from discordbot.cogs._parse_douyin.fetch import (
    douyin_url_locks,
    douyin_failure_message,
    douyin_fetch_semaphore,
)

# Douyin's own palette, so the expansion reads as a Douyin card at a glance.
_EMBED_COLOR = 0xFE2C55

# Bound on one expansion's Douyin work. It exists to cap how long a single paste can hold the
# fetch slot it shares with the reply path, not to hurry the download along: a healthy post
# finishes in seconds, while a stalling CDN would otherwise retry its way into the tens of
# minutes. A timeout is reported as a plain failure, never as a missing post.
DOUYIN_EXPAND_TIMEOUT_SECONDS = 120.0


class DouyinCogs(commands.Cog):
    """Expands Douyin links into Discord attachments.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        config: Runtime configuration carrying the auto-expansion kill-switch.
        media_delivery: Planner deciding which files attach and which are hosted as a URL.
    """

    # A retryable block gets its own reaction so it never reads like the ⚠️ "could not read
    # this post" outcome; the two are different problems and one of them resolves itself.
    blocked_emoji: ClassVar[str] = "⏱️"

    def __init__(self, bot: commands.Bot):
        """Initializes the DouyinCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = DouyinConfig()
        self.media_delivery = build_media_delivery_planner()
        self.downloader_factory = DouyinDownloader

    @staticmethod
    def _build_embed(post: DouyinPost, url: str) -> Embed:
        """Builds the caption card that accompanies the expanded media."""
        embed = Embed(description=post.title, url=url, color=_EMBED_COLOR)
        if post.author_name:
            embed.set_author(name=post.author_name, url=url)
        return embed

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and expands Douyin links.

        Args:
            message: The message that was sent.
        """
        if message.author.bot:
            return

        match = DOUYIN_URL_RE.search(string=message.content)
        if not match:
            return

        # The regex matches the host, not the path, so a profile or live-room link would
        # otherwise earn a warning reaction and a failure reply nobody asked for.
        if not is_douyin_post_url(url=match.group(0)):
            return

        # A link addressed to the bot is gen_reply's to answer about, not ours to expand; see
        # the module docstring. Checked after the URL match so the common no-link message costs
        # one regex, not two.
        if is_addressed_to_bot(message=message, bot_user=self.bot.user):
            return

        if not self.config.auto_expand_enabled:
            return

        url = match.group(0)
        current_emoji = await update_reaction(message=message, bot_user=self.bot.user, emoji="🔗")
        try:
            await self._expand(message=message, url=url, current_emoji=current_emoji)
        except Exception:
            logfire.error("Failed to expand Douyin link", _exc_info=True)
            with contextlib.suppress(Exception):
                await update_reaction(
                    message=message,
                    bot_user=self.bot.user,
                    emoji="<:redcross:1517565100838355016>",
                    previous=current_emoji,
                )

    async def _expand(self, message: Message, url: str, current_emoji: str) -> None:
        """Fetches the post and posts it back, reporting every failure mode distinctly."""
        # A private directory per invocation, because the filenames are derived from the post id:
        # two expansions of the same post in one shared temp dir would write the same paths,
        # letting one truncate the other's file and letting either one's cleanup delete a file
        # the other is still uploading.
        with tempfile.TemporaryDirectory(prefix="douyin-") as download_dir:
            downloader = self.downloader_factory(output_folder=download_dir)
            try:
                # Both bounds cover only the Douyin-facing work, never the Discord upload that
                # follows: the per-URL lock collapses simultaneous pastes of one link into a
                # single fetch (the payload cache alone loses that race), and the semaphore
                # keeps a burst of distinct links from arriving at Douyin all at once.
                async with (
                    douyin_url_locks.hold(url),
                    douyin_fetch_semaphore.get(),
                    asyncio.timeout(delay=DOUYIN_EXPAND_TIMEOUT_SECONDS),
                ):
                    # Bounded because the slot is shared with the reply path: a stalling CDN
                    # costs `download_timeout` x `max_retries` per file, so an unbounded gallery
                    # could hold one of two slots for half an hour and stall every AI reply
                    # about a Douyin link behind it.
                    #
                    # Parsed before the download so the caption survives a refused download: an
                    # oversize post still gets its card instead of a bare warning reaction. The
                    # share payload is cached, so this costs no extra request.
                    post = await asyncio.to_thread(downloader.parse_metadata, url=url)
                    result = await asyncio.to_thread(
                        downloader.download,
                        url=url,
                        post=post,
                        max_images=DISCORD_ATTACHMENT_LIMIT,
                    )
            except Exception as error:
                await self._report_failure(
                    message=message, url=url, error=error, current_emoji=current_emoji
                )
                return

            with result:
                await self._deliver(
                    message=message, url=url, post=post, result=result, current_emoji=current_emoji
                )

    async def _report_failure(
        self, message: Message, url: str, error: Exception, current_emoji: str
    ) -> None:
        """Reacts with the outcome the failure actually represents, never a generic error.

        A bot wall is retryable and the post is fine, so it gets its own reaction rather than
        the ⚠️ that means "this post could not be read". The reason is stated in the reply, so
        a reader is never left guessing which of the two happened.
        """
        logfire.warn(
            "Douyin expansion failed", url=url, error_type=type(error).__name__, _exc_info=error
        )
        emoji = self.blocked_emoji if isinstance(error, DouyinBlockedError) else "⚠️"
        await update_reaction(
            message=message, bot_user=self.bot.user, emoji=emoji, previous=current_emoji
        )
        await message.reply(
            content=douyin_failure_message(error=error),
            mention_author=False,
            allowed_mentions=AllowedMentions.none(),
        )

    async def _deliver(
        self,
        message: Message,
        url: str,
        post: DouyinPost,
        result: DouyinDownload,
        current_emoji: str,
    ) -> None:
        """Posts the downloaded media plus its caption card, then marks the source done."""
        items = [MediaItem(source=path, filename=path.name) for path in result.filenames]
        # Read BEFORE planning, which caches it: a successful host moves the source out of the
        # temp dir, so measuring afterwards would stat a deleted path and break the very
        # oversize-to-URL fallback this number describes.
        total_mb = result.total_bytes / 1024 / 1024
        plan = await self.media_delivery.plan(
            items=items,
            upload_limit=upload_limit_for(guild=message.guild),
            # A gallery rides several attachments on one send, and Discord measures the whole
            # multipart body, so hold back the envelope; a lone video is a single-file send.
            envelope_margin=MEDIA_ENVELOPE_MARGIN if len(items) > 1 else 0,
        )

        if not plan.native and not plan.hosted_urls:
            await update_reaction(
                message=message, bot_user=self.bot.user, emoji="⚠️", previous=current_emoji
            )
            await message.reply(
                content=f"-# 檔案大小超過 {total_mb:.1f}MB,無法傳送",
                mention_author=False,
                allowed_mentions=AllowedMentions.none(),
            )
            return

        with contextlib.suppress(Exception):
            await message.edit(suppress=True)

        await self._send(message=message, url=url, post=post, result=result, plan=plan)
        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji="<:greencheck:1517565102424068226>",
            previous=current_emoji,
        )

    async def _send(
        self, message: Message, url: str, post: DouyinPost, result: DouyinDownload, plan: MediaPlan
    ) -> None:
        """Sends the expansion, stating anything that was left out rather than dropping it."""
        lines: list[str] = []
        if result.omitted_images:
            lines.append(
                f"-# 已省略 {result.omitted_images} 張圖片 (Discord 單則訊息最多 "
                f"{DISCORD_ATTACHMENT_LIMIT} 個附件)"
            )
        if plan.dropped_items:
            lines.append(f"-# 有 {len(plan.dropped_items)} 個檔案傳送失敗")
        # Hosted URLs must stay unwrapped and each on its own line to stay clickable and, under
        # ~100 MiB, to render Discord's inline player.
        lines.extend(plan.hosted_urls)

        files = [item.to_file() for item in plan.native]
        embeds = [self._build_embed(post=post, url=url)]
        await message.reply(
            content="\n".join(lines) if lines else None,
            embeds=embeds,
            mention_author=False,
            allowed_mentions=AllowedMentions.none(),
            **embed_spacer_payload(
                embeds=embeds, is_edit=False, target=message, extra_files=files
            ),
        )


def setup(bot: commands.Bot) -> None:
    """Adds the DouyinCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(DouyinCogs(bot), override=True)
