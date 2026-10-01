"""Expands a Douyin post URL into Discord attachments.

Douyin's WAF bans a share path for tens of minutes once it is hit hard, and this listener sees
every message in every channel, so the request-volume bounds in `services/platforms/douyin.py` are
load-bearing rather than defensive. A blocked request must never be reported as a missing post:
telling someone their working link is dead is the worst failure this feature can produce.

`utils/expansion_cog.py` owns everything around the card. What is here is the card: a clip or a
gallery, downloaded and attached.
"""

import asyncio
import contextlib

import logfire
from nextcord import Embed, Message
from pydantic import Field, BaseModel
from nextcord.ext import commands

from discordbot.typings.timeouts import DOUYIN_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.scratch_dir import scratch_directory
from discordbot.utils.expansion_cog import ExpansionCog, ExpansionDelivery
from discordbot.utils.discord_embeds import DISCORD_ATTACHMENT_LIMIT
from discordbot.utils.media_delivery import upload_limit_for, build_media_delivery_planner
from discordbot.utils.douyin_delivery import plan_douyin_delivery, douyin_delivery_lines
from discordbot.services.platforms.douyin import (
    DOUYIN_URL_RE,
    DouyinDownload,
    DouyinMetadata,
    DouyinDownloader,
    douyin_url_locks,
    is_douyin_post_url,
    douyin_fetch_semaphore,
)

# Douyin's own palette, so the expansion reads as a Douyin card at a glance. Deliberately NOT in
# `typings/colors.py`: that palette is Discord's own semantic set (success / failure / info), and
# a third party's brand red belongs to the one card that wears it.
_EMBED_COLOR = 0xFE2C55


class DouyinPost(BaseModel):
    """A parsed Douyin post together with the files downloaded for it."""

    metadata: DouyinMetadata = Field(
        ..., description="The post's caption and author, for the card."
    )
    download: DouyinDownload = Field(
        ...,
        description=(
            "The downloaded clip or gallery, in a scratch directory removed once the expansion "
            "is on screen."
        ),
    )


class DouyinCogs(ExpansionCog[DouyinPost]):
    """Expands Douyin links into Discord attachments.

    Attributes:
        media_delivery: Planner deciding which files attach and which are hosted as a URL.
        downloader_factory: Builds the per-invocation downloader, one per scratch directory; the
            seam a test replaces to keep an expansion off the network.
    """

    SOURCE = "douyin"
    PLATFORM = "Douyin"
    URL_PATTERN = DOUYIN_URL_RE
    PLACEHOLDER_TEXT = "-# 正在讀取抖音貼文⋯"

    def __init__(self, bot: commands.Bot):
        """Initializes the DouyinCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        super().__init__(bot=bot)
        self.media_delivery = build_media_delivery_planner()
        self.downloader_factory = DouyinDownloader

    @staticmethod
    def url_is_expandable(*, url: str) -> bool:
        """Whether the matched URL names a post.

        The pattern matches the host, not the path, so a profile or live-room link would
        otherwise spend a rate-limited request to establish there is nothing to show.

        Args:
            url: The URL the pattern matched.

        Returns:
            True when the URL names a post.
        """
        return is_douyin_post_url(url=url)

    async def read(
        self, *, message: Message, url: str, stack: contextlib.AsyncExitStack
    ) -> DouyinPost:
        """Parses the post and downloads its media into a scratch directory of its own.

        The per-URL lock and the fetch semaphore (`services/platforms/douyin.py` has why) cover
        only the Douyin-facing work, never the Discord upload that follows.

        The post is parsed first because the card needs its caption and author, and the download
        is handed it rather than parsing it again.

        Args:
            message: Unused; nothing here logs.
            url: The post to read.
            stack: Holds the scratch directory, and with it the downloaded files, until delivery
                is done.

        Returns:
            The caption and the downloaded media.
        """
        del message
        download_dir = stack.enter_context(scratch_directory(prefix="parse-douyin-"))
        downloader = self.downloader_factory(output_folder=download_dir)
        async with (
            douyin_url_locks.hold(url),
            douyin_fetch_semaphore.get(),
            asyncio.timeout(delay=DOUYIN_EXPAND_TIMEOUT_SECONDS),
        ):
            metadata = await asyncio.to_thread(downloader.parse_metadata, url=url)
            download = await asyncio.to_thread(
                downloader.download, url=url, post=metadata, max_images=DISCORD_ATTACHMENT_LIMIT
            )
        return DouyinPost(metadata=metadata, download=download)

    async def build_delivery(
        self, *, message: Message, url: str, parsed: DouyinPost
    ) -> ExpansionDelivery | None:
        """Plans the media and builds the card, refusing a post nothing can carry.

        Args:
            message: The message carrying the link.
            url: The post that was read.
            parsed: The caption and the downloaded media.

        Returns:
            The card, or None when the media can be neither attached nor hosted.
        """
        delivery = await plan_douyin_delivery(
            planner=self.media_delivery,
            result=parsed.download,
            upload_limit=upload_limit_for(guild=message.guild),
        )
        plan = delivery.plan
        if not plan.native and not plan.hosted_urls:
            # The size is stated here rather than in the channel, which is the only place it would
            # otherwise exist: the mark says the post could not be delivered and nothing else is
            # left behind, as with every other expansion refusal.
            logfire.warn(
                "Douyin media could not be attached or hosted; refusing the post",
                url=url,
                message_id=message.id,
                total_mb=delivery.total_mb,
            )
            return None

        lines = douyin_delivery_lines(
            result=parsed.download,
            plan=plan,
            hosting_available=self.media_delivery.media_hosting.config.available,
            url=url,
            dropped_event="Douyin expansion dropped some media",
        )
        return ExpansionDelivery(
            content="\n".join(lines) if lines else None,
            embeds=[self._build_embed(post=parsed.metadata, url=url)],
            files=[item.to_file() for item in plan.native],
        )

    @staticmethod
    def _build_embed(*, post: DouyinMetadata, url: str) -> Embed:
        """Builds the caption card that accompanies the expanded media."""
        embed = Embed(description=post.title, url=url, color=_EMBED_COLOR)
        if post.author_name:
            embed.set_author(name=post.author_name, url=url)
        return embed


def setup(bot: commands.Bot) -> None:
    """Adds the DouyinCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(DouyinCogs(bot), override=True)
