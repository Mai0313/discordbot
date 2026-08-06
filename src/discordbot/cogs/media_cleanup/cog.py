"""Periodically reaps hosted media so the serve dir stays bounded by size and age.

The whole Discord surface is one `on_ready` listener: no slash command, no message listener,
nothing a user ever sees. What it registers is a `@tasks.loop` driving
`MediaHostingService.run_maintenance` every `MEDIA_CLEANUP_INTERVAL_HOURS`, plus one immediate
sweep at startup, since a `tasks.loop` first fires a whole interval after `start()` and a restart
would otherwise leave an over-cap serve dir alone for hours.

Each publish enforces the size cap eagerly, so this loop is the backstop: it is the only thing
that applies the age cap (`MEDIA_HOSTING_RETENTION_HOURS`) and clears crash-left temp files.
`MediaHostingConfig.cleanup_enabled` is the gate (hosting enabled and configured, AND at least one
of the two caps set); when it is off the loop never starts and nothing in the serve dir is
touched. The sweep only ever deletes the bot's own content-addressed files (see
`MediaHostingService`), never a foreign file parked in the shared serve dir.
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
    """Runs the hosted-media size/age/temp sweep on a timer plus once at startup.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        media_hosting: The hosting service whose serve dir this cog sweeps.
    """

    def __init__(self, bot: commands.Bot):
        """Builds the cog with its own media-hosting service.

        The service is built here rather than shared with the delivering cogs, so the config is
        read and the resolved serve dir logged once at cog load. Several instances over one serve
        dir are safe: every scan and delete takes `media_delivery`'s module-level dir lock.

        Args:
            bot (commands.Bot): The bot instance whose readiness gates the loop.
        """
        self.bot = bot
        self.media_hosting = MediaHostingService(config=MediaHostingConfig())
        self._started = False
        self._startup_task: asyncio.Task[None] | None = None

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Starts the cleanup loop once, only when hosting and at least one cap are configured.

        `on_ready` fires on every reconnect, so `_started` guards a single start. The loop fires
        only after the first interval, so a startup sweep is spawned immediately to catch a
        restart. When cleanup is disabled the loop never starts and nothing in the serve dir is
        touched.
        """
        if self._started:
            return
        self._started = True
        if not self.media_hosting.config.cleanup_enabled:
            return
        self._startup_task = asyncio.create_task(self._sweep())
        self.cleanup_loop.start()

    def cog_unload(self) -> None:
        """Stops the loop when the cog is torn down.

        A startup sweep still in flight is left to finish: it holds no Discord state and swallows
        its own failures.
        """
        self.cleanup_loop.cancel()

    @tasks.loop(hours=MEDIA_CLEANUP_INTERVAL_HOURS)
    async def cleanup_loop(self) -> None:
        """Runs one backstop sweep per interval."""
        await self._sweep()

    @cleanup_loop.before_loop
    async def _before_cleanup_loop(self) -> None:
        """Waits until the gateway is ready before the first scheduled sweep."""
        await self.bot.wait_until_ready()

    async def _sweep(self) -> None:
        """Runs one maintenance pass, best-effort (never raises into its caller).

        `run_maintenance` blocks on directory scans and unlinks under a threading lock, so it goes
        through a worker thread. A pass that deleted nothing logs nothing.
        """
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
        bot (commands.Bot): The bot instance to register the cog on.
    """
    bot.add_cog(MediaCleanupCogs(bot), override=True)
