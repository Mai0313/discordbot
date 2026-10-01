"""Periodically reaps hosted media so the serve dir stays bounded by size and age.

The size cap is enforced eagerly at publish, so this sweep is its backstop rather than the
enforcement point; it also applies the age cap and clears crash-left temp files, on a timer whose
first iteration fires at startup, and self-disables when `cleanup_enabled` is false. The sweep
only ever deletes the bot's own content-addressed files (see `MediaHostingService`), never a
foreign file parked in the serve dir.
"""

import time
import asyncio

import logfire
from nextcord.ext import tasks, commands

from discordbot.utils.media_delivery import (
    MEDIA_CLEANUP_INTERVAL_HOURS,
    MediaHostingConfig,
    MediaHostingService,
)


class MediaCleanupCogs(commands.Cog):
    """Runs the hosted-media size/age/temp sweep on a timer whose first run is at startup.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the cog with its own media-hosting service.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.media_hosting = MediaHostingService(config=MediaHostingConfig())
        self._started = False

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Starts the cleanup loop, whose first iteration is the startup sweep, once.

        `on_ready` fires on every reconnect, so `_started` guards a single start. When cleanup is
        disabled the loop never starts and nothing in the serve dir is touched.
        """
        if self._started:
            return
        self._started = True
        if not self.media_hosting.config.cleanup_enabled:
            return
        self.cleanup_loop.start()

    def cog_unload(self) -> None:
        """Stops the loop when the cog is torn down."""
        self.cleanup_loop.cancel()

    @tasks.loop(hours=MEDIA_CLEANUP_INTERVAL_HOURS)
    async def cleanup_loop(self) -> None:
        """The periodic backstop sweep."""
        await self._sweep()

    async def _sweep(self) -> None:
        """Runs one off-loop maintenance pass, best-effort (never raises into the loop)."""
        try:
            deleted, freed = await asyncio.to_thread(
                self.media_hosting.run_maintenance, now=time.time()
            )
        except Exception as error:
            # Broad on purpose: a raise escaping into `cleanup_loop` stops the tasks.loop for the
            # process lifetime, leaving the serve dir unbounded.
            logfire.warn(
                "Media cleanup sweep failed",
                serve_dir=self.media_hosting.config.serve_dir,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            return
        if deleted or freed:
            logfire.info("Media cleanup sweep", deleted_count=deleted, freed_bytes=freed)


def setup(bot: commands.Bot) -> None:
    """Adds the MediaCleanupCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(MediaCleanupCogs(bot), override=True)
