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

# Douyin's brand red; kept out of `typings/colors.py`, which holds Discord's semantic palette.
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
        downloader_factory: Per-invocation downloader constructor, one per scratch directory.
    """

    SOURCE = "douyin"
    PLATFORM = "Douyin"
    URL_PATTERN = DOUYIN_URL_RE
    PLACEHOLDER_TEXT = "-# 正在讀取抖音貼文⋯"

    def __init__(self, bot: commands.Bot):
        """Initializes the DouyinCogs instance.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        super().__init__(bot=bot)
        self.media_delivery = build_media_delivery_planner()
        self.downloader_factory = DouyinDownloader

    @staticmethod
    def url_is_expandable(url: str) -> bool:
        """Returns whether `url` names a Douyin post.

        `DOUYIN_URL_RE` matches the host rather than the path, so profile and live-room links are
        filtered here before spending a rate-limited request.

        Args:
            url (str): The URL matched by `URL_PATTERN`.
        """
        return is_douyin_post_url(url=url)

    async def read(
        self, message: Message, url: str, stack: contextlib.AsyncExitStack
    ) -> DouyinPost:
        """Returns the parsed post and its media downloaded into a scratch directory on `stack`.

        The per-URL lock and fetch semaphore (`services/platforms/douyin.py`) cover only the
        Douyin-facing requests, never the Discord upload that follows.

        Args:
            message (Message): Unused; required by the `ExpansionCog` hook signature.
            url (str): The post URL to read.
            stack (contextlib.AsyncExitStack): Holds the scratch directory until delivery finishes.
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
        self, message: Message, url: str, parsed: DouyinPost
    ) -> ExpansionDelivery | None:
        """Returns the expansion card and media files, or `None` when no media can be delivered.

        Args:
            message (Message): The message carrying the link.
            url (str): The post URL that was read.
            parsed (DouyinPost): The post metadata and downloaded media.
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
            media=plan.native,
        )

    @staticmethod
    def _build_embed(post: DouyinMetadata, url: str) -> Embed:
        """Builds the caption card that accompanies the expanded media."""
        embed = Embed(description=post.title, url=url, color=_EMBED_COLOR)
        if post.author_name:
            embed.set_author(name=post.author_name, url=url)
        return embed


def setup(bot: commands.Bot) -> None:
    """Adds the DouyinCogs to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(DouyinCogs(bot), override=True)
