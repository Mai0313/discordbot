"""Periodic size, age, and temp-file sweep for hosted media in the serve directory.

Acts as a backstop behind publish-time size enforcement, applies the age cap, and removes
leftover temp files. Runs on a fixed timer starting at startup when `cleanup_enabled` is true,
and deletes only the bot's own content-addressed files (`MediaHostingService`).
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
        bot (commands.Bot): The Discord bot instance that owns this cog.
        media_hosting (MediaHostingService): Service that performs the maintenance sweep.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the cog with its own media-hosting service.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot
        self.media_hosting = MediaHostingService(config=MediaHostingConfig())
        self._started = False

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Starts the cleanup loop once on the first `on_ready` event if cleanup is enabled."""
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
        """Runs the periodic maintenance sweep."""
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
    """Adds `MediaCleanupCogs` to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(MediaCleanupCogs(bot), override=True)
