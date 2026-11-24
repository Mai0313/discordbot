"""Discord bot entry point and runtime event handlers."""

import os
import sys
from time import monotonic
from typing import TYPE_CHECKING, Any
import logging
from pathlib import Path
import platform

import logfire
from logfire import LogfireLoggingHandler
import nextcord
from nextcord import Game, Intents, Interaction, HTTPException
from nextcord.ext import commands
from nextcord.errors import ApplicationError

from discordbot import setup_logging
from discordbot.typings.config import DiscordConfig, load_bot_config
from discordbot.utils.timezone import database_now
from discordbot.utils.asyncio_locks import spawn_tracked
from discordbot.utils.sqlite_config import ensure_database_upgraded
from discordbot.utils.media_delivery import AttachmentLimitInteraction
from discordbot.utils.message_cleanup import (
    delete_tracked_public_messages,
    schedule_public_message_delete,
)

if TYPE_CHECKING:
    import asyncio

    from nextcord.types.interactions import Interaction as InteractionPayload


class DiscordBot(commands.Bot):
    """Discord bot configured with project-specific intents and cogs."""

    def __init__(self, activity_name: str | None = None) -> None:
        """Initialises the Discord bot with specific intents and configuration.

        Args:
            activity_name (str | None): Playing-activity label for the bot's presence; defaults
                to `[global.bot].activity` from `config.toml`.
        """
        intents = Intents.all()
        intents.members = False
        intents.presences = False
        resolved_activity = (
            activity_name
            if activity_name is not None
            else load_bot_config().global_config.bot.activity
        )
        super().__init__(
            intents=intents,
            help_command=None,
            description="A Discord bot made with Nextcord.",
            activity=Game(name=resolved_activity),
        )
        for logger_name, level in (("nextcord.state", logging.WARNING), ("alembic", logging.INFO)):
            logger = logging.getLogger(name=logger_name)
            logger.setLevel(level=level)
            if not any(isinstance(handler, LogfireLoggingHandler) for handler in logger.handlers):
                logger.addHandler(hdlr=LogfireLoggingHandler())
        # Core public-message cleanup lives in its own DB; optional cogs upgrade their own in setup().
        ensure_database_upgraded(section="cleanup")
        # Load cogs synchronously before gateway connect so application_commands is populated.
        # Every cog's setup() must be sync: no event loop is running yet for asyncio.create_task().
        self._load_cogs_sync()
        self._startup_sweeps_done = False
        self._commands_synced = False
        self._commands_syncing = False
        # Taken before the gateway connects: everything an earlier process recorded falls before
        # it, and nothing this one records can, the interactions run ahead of `on_ready` included.
        self._started_at = database_now()
        self._startup_tasks: set[asyncio.Task[None]] = set()

    def _load_cogs_sync(self) -> None:
        """Loads every cog directory under `cogs/`; raises on incomplete cog directories."""
        cog_dir = Path(__file__).parent / "cogs"
        cog_files: list[str] = []
        for entry in sorted(cog_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            if not (entry / "__init__.py").is_file() or not (entry / "cog.py").is_file():
                raise RuntimeError(
                    f"{entry} is under cogs/ but is not a cog: a cog directory must hold "
                    "both __init__.py and cog.py."
                )
            cog_files.append(f"discordbot.cogs.{entry.name}.cog")
        self.load_extensions(cog_files, stop_at_error=True)
        logfire.info("Cogs Loaded", cogs=cog_files)

    def get_interaction(
        self, data: "InteractionPayload", cls: type[Interaction[Any]] = AttachmentLimitInteraction
    ) -> Interaction[Any]:
        """Returns a gateway interaction that preserves Discord's attachment size limit.

        Without this override, `upload_limit_for` cannot read the invoking user's Nitro
        upload allowance off the interaction and falls back to the guild's boost tier.

        Args:
            data ("InteractionPayload"): Raw interaction payload from the Discord gateway.
            cls (type[Interaction[Any]]): Interaction class to instantiate.
        """
        return cls(data=data, state=self._connection)

    async def on_connect(self) -> None:
        """Restores the local application-command registry on connect.

        `ConnectionState.parse_ready` clears the local registry on every `READY`; restoring it
        here without an API call prevents nextcord's guildless-interaction lazy load from
        running `delete_unknown` and wiping global commands. `on_ready` owns command sync.
        """
        self.add_all_application_commands()
        bot_user = self.user
        if bot_user is None:
            return
        logfire.info("Bot Connected", bot_name=bot_user.name, bot_id=bot_user.id)

    async def _count_registered_commands(self) -> int | None:
        """Returns the count of global commands registered on Discord, or None on error."""
        application_id = self.application_id
        if application_id is None:
            return None
        try:
            registered = await self.http.get_global_commands(
                application_id=application_id, with_localizations=False
            )
        except Exception as exc:
            # Broad on purpose: this is a diagnostic, and a boot must not fail because one
            # could not be taken. Losing the count costs a log field, not a command.
            logfire.warn(
                "Could not read the registered command count",
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        return len(registered)

    async def on_error(self, event_method: str, *args: object, **kwargs: object) -> None:
        """Logs an unhandled event-handler exception from `sys.exc_info()` through logfire.

        Overrides nextcord's default `sys.stderr` handler; must run inside an active `except`
        block.

        Args:
            event_method (str): Name of the event whose handler raised.
            args (object): Positional arguments the handler was dispatched with.
            kwargs (object): Keyword arguments the handler was dispatched with.
        """
        logfire.error(
            "Unhandled exception in an event handler",
            event_method=event_method,
            _exc_info=sys.exc_info()[1],
        )

    async def on_ready(self) -> None:
        """Performs one-time startup sweeps and application-command sync on first ready.

        Idempotent across gateway reconnects and resumes; retries command sync on a later ready
        when an earlier sync failed.
        """
        if self._startup_sweeps_done and self._commands_synced:
            return
        bot_user = self.user
        if bot_user is None:
            # Never expected once on_ready fires; leave the latches unset so a reconnect retries.
            logfire.warn("on_ready fired without a logged-in user; skipping initial setup")
            return
        if not self._startup_sweeps_done:
            self._startup_sweeps_done = True
            # Spawn startup sweeps before command sync so a sync error cannot skip them.
            spawn_tracked(
                coro=delete_tracked_public_messages(bot=self, tracked_before=self._started_at),
                tasks=self._startup_tasks,
                name="delete-stale-public-responses",
            )

            logfire.info(
                "Logged in",
                bot_name=bot_user.name,
                discord_version=nextcord.__version__,
                python_version=platform.python_version(),
                system=f"{platform.system()} {platform.release()} ({os.name})",
            )

        if not self._commands_synced and not self._commands_syncing:
            self._commands_syncing = True
            try:
                registered_before = await self._count_registered_commands()
                sync_started_at = monotonic()
                await self.sync_all_application_commands()
                self._commands_synced = True
            finally:
                self._commands_syncing = False
            # `registered_before` distinguishes a no-op sync (~1s) from a full upsert (~1m).
            logfire.info(
                "Application commands synced",
                registered_before=registered_before,
                local_count=len(self.get_application_commands()),
                elapsed_seconds=round(monotonic() - sync_started_at, 3),
            )

            app_info = await self.application_info()
            # Omit `scope`/`permissions` so Discord uses Developer Portal Default Install Settings;
            # hardcoding `scope=bot` drops the user-install ("add to my apps") option.
            invite_url = f"https://discord.com/oauth2/authorize?client_id={app_info.id}"
            logfire.info("Bot Started", bot_name=bot_user.name, bot_id=bot_user.id)
            logfire.info("Invite Link", invite_url=invite_url)

    async def on_application_command_error(
        self, interaction: Interaction[commands.Bot], exception: ApplicationError
    ) -> None:
        """Logs an unhandled slash-command error and notifies the caller if already acknowledged.

        Sends a follow-up notice only when the interaction has already been answered or deferred;
        commands that report their own failures must catch them rather than re-raise.

        Args:
            interaction (Interaction[commands.Bot]): Interaction whose command raised.
            exception (ApplicationError): Exception raised by the application command.
        """
        # nextcord wraps a command-body failure in ApplicationInvokeError, so report the
        # unwrapped type to name the real defect.
        original = getattr(exception, "original", exception)
        command = interaction.application_command
        command_name = command.qualified_name if command is not None else None
        user_id = interaction.user.id if interaction.user is not None else None
        logfire.error(
            "Unhandled application command error",
            command=command_name,
            guild_id=interaction.guild_id,
            user_id=user_id,
            error_type=type(original).__name__,
            _exc_info=exception,
        )
        if not interaction.response.is_done():
            return
        try:
            notice = await interaction.followup.send(
                content="指令執行時出錯了，部分操作可能已經生效，重試前請先確認結果。",
                ephemeral=True,
                wait=True,
            )
        except HTTPException:
            logfire.warn(
                "Could not tell the caller a slash command failed",
                command=command_name,
                guild_id=interaction.guild_id,
                user_id=user_id,
                _exc_info=True,
            )
            return
        # A deferred public interaction ignores `ephemeral=True` on its first followup.
        if not notice.flags.ephemeral:
            schedule_public_message_delete(
                message=notice,
                user_name=interaction.user.name if interaction.user is not None else None,
            )


def main() -> None:
    """Initialises and runs the Discord bot."""
    setup_logging()
    discord_config = DiscordConfig()
    bot = DiscordBot()
    bot.run(token=discord_config.discord_bot_token)


if __name__ == "__main__":
    main()
