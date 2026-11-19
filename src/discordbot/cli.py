"""Discord bot entry point and runtime event handlers."""

import os
import sys
from time import monotonic
from typing import TYPE_CHECKING, Any
import asyncio
import logging
from pathlib import Path
import platform

from alembic import command
import logfire
from logfire import LogfireLoggingHandler
import nextcord
from nextcord import Game, Intents, Message, Interaction, HTTPException
from nextcord.ext import tasks, commands
from alembic.config import Config
from nextcord.errors import ApplicationError

from discordbot import setup_logging
from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.config import DiscordConfig
from discordbot.utils.timezone import database_now
from discordbot.typings.economy import BASE_MESSAGE_REWARD_AMOUNT, MESSAGE_REWARD_COOLDOWN_SECONDS
from discordbot.utils.asyncio_locks import spawn_tracked
from discordbot.utils.model_pricing import MODEL_INFO_REFRESH_MINUTES, refresh_model_info
from discordbot.utils.media_delivery import AttachmentLimitInteraction
from discordbot.utils.message_cleanup import (
    delete_tracked_public_messages,
    schedule_public_message_delete,
)
from discordbot.services.economy.database import (
    credit_with_repayment,
    record_guild_participant,
    reject_loan_proposals_created_before,
)

if TYPE_CHECKING:
    from nextcord.types.interactions import Interaction as InteractionPayload


class DiscordBot(commands.Bot):
    """Discord bot configured with project-specific intents and cogs."""

    def __init__(self) -> None:
        """Initialises the Discord bot with specific intents and configuration."""
        intents = Intents.all()
        intents.members = False
        intents.presences = False
        super().__init__(
            intents=intents,
            help_command=None,
            description="A Discord bot made with Nextcord.",
            activity=Game("your mama"),
        )
        state_logger = logging.getLogger("nextcord.state")
        state_logger.setLevel(logging.WARNING)
        state_logger.addHandler(LogfireLoggingHandler())
        alembic_logger = logging.getLogger("alembic")
        alembic_logger.setLevel(logging.INFO)
        alembic_logger.addHandler(LogfireLoggingHandler())
        Path("./data/database").mkdir(parents=True, exist_ok=True)
        Path("./data/memories").mkdir(parents=True, exist_ok=True)
        # `messages.db` is left to the log_msg cog's setup: too large to migrate at startup.
        for database in ("economy", "games", "reply"):
            command.upgrade(
                config=Config(file_="alembic.ini", ini_section=database), revision="head"
            )
        # Load cogs synchronously before gateway connect so application_commands is populated.
        # Every cog's setup() must be sync: no event loop is running yet for asyncio.create_task().
        self._load_cogs_sync()
        self._initial_setup_done = False
        # Taken before the gateway connects: everything an earlier process recorded falls before
        # it, and nothing this one records can, the interactions run ahead of `on_ready` included.
        self._started_at = database_now()
        self._startup_tasks: set[asyncio.Task[None]] = set()
        # Process-local per-user reward cooldown; resets on restart by design.
        self._message_reward_at: dict[int, float] = {}
        self._message_reward_pruned_at = 0.0

    def _prune_message_reward_cooldowns(self, now: float) -> None:
        """Drops expired message-reward cooldown entries."""
        if now - self._message_reward_pruned_at < MESSAGE_REWARD_COOLDOWN_SECONDS:
            return
        cutoff = now - MESSAGE_REWARD_COOLDOWN_SECONDS
        self._message_reward_at = {
            user_id: rewarded_at
            for user_id, rewarded_at in self._message_reward_at.items()
            if rewarded_at > cutoff
        }
        self._message_reward_pruned_at = now

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
        """Restores the local application-command registry and starts background tasks on connect.

        `ConnectionState.parse_ready` clears the local registry on every `READY`; restoring it
        here without an API call prevents nextcord's guildless-interaction lazy load from
        running `delete_unknown` and wiping global commands. `on_ready` owns command sync.
        """
        self.add_all_application_commands()
        # Started on connect so an error during `on_ready`'s command sync cannot prevent it.
        if not self.price_table_task.is_running():
            self.price_table_task.start()
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

        Idempotent across gateway reconnects and resumes.
        """
        if self._initial_setup_done:
            return
        bot_user = self.user
        if bot_user is None:
            # Never expected once on_ready fires; leave the latch unset so a reconnect retries.
            logfire.warn("on_ready fired without a logged-in user; skipping initial setup")
            return
        self._initial_setup_done = True
        # Spawn startup sweeps before command sync so a sync error cannot skip them.
        spawn_tracked(
            coro=delete_tracked_public_messages(bot=self, tracked_before=self._started_at),
            tasks=self._startup_tasks,
            name="delete-stale-public-responses",
        )
        # Reject stale loan proposals whose decision views died with the previous process.
        spawn_tracked(
            coro=reject_loan_proposals_created_before(cutoff=self._started_at),
            tasks=self._startup_tasks,
            name="reject-stale-loan-requests",
        )

        logfire.info(
            "Logged in",
            bot_name=bot_user.name,
            discord_version=nextcord.__version__,
            python_version=platform.python_version(),
            system=f"{platform.system()} {platform.release()} ({os.name})",
        )

        registered_before = await self._count_registered_commands()
        sync_started_at = monotonic()
        await self.sync_all_application_commands()
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

    @tasks.loop(minutes=MODEL_INFO_REFRESH_MINUTES)
    async def price_table_task(self) -> None:
        """Refreshes the LiteLLM model price table off the event loop.

        Runs immediately on start to warm the cache, retries while degraded, and refreshes daily
        once upstream has succeeded.
        """
        try:
            await asyncio.to_thread(refresh_model_info)
        except Exception as exc:
            # Broad on purpose: an uncaught exception stops the tasks.loop permanently, and
            # nextcord's bound `_error` crashes silently on class methods (`_injected` arg).
            logfire.error("model price table refresh failed; retrying next pass", _exc_info=exc)

    async def on_message(self, message: Message) -> None:
        """Awards the cooldown-gated message reward.

        This is the only faucet that pays an action reward, and it is best-effort: a failed
        credit or participation write is logged and swallowed.

        Args:
            message (Message): The message that was sent.
        """
        if message.author == self.user or message.author.bot:
            return

        now = monotonic()
        self._prune_message_reward_cooldowns(now=now)
        last_rewarded_at = self._message_reward_at.get(message.author.id)
        if last_rewarded_at is None or now - last_rewarded_at >= MESSAGE_REWARD_COOLDOWN_SECONDS:
            # Reserve the cooldown slot before awaiting to prevent double-credits on rapid
            # messages; restored in the except block if crediting fails.
            self._message_reward_at[message.author.id] = now
            guild = message.guild
            try:
                avatar_url = await guild_avatar_url(user=message.author, guild=guild)
                await credit_with_repayment(
                    user_id=message.author.id,
                    name=message.author.name,
                    avatar_url=avatar_url,
                    amount=BASE_MESSAGE_REWARD_AMOUNT,
                )
            except Exception as exc:
                if last_rewarded_at is None:
                    self._message_reward_at.pop(message.author.id, None)
                else:
                    self._message_reward_at[message.author.id] = last_rewarded_at
                # Broad on purpose: the reward is best-effort and must never raise out of
                # the listener, so every failure mode (DB, avatar fetch) is swallowed.
                logfire.warn(
                    "Failed to award base message points",
                    user_id=message.author.id,
                    guild_id=guild.id if guild else None,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
            if guild is not None:
                # Kept outside the reward try block so a failure here cannot roll back the
                # cooldown slot and double-pay the reward on the next message.
                try:
                    await record_guild_participant(guild_id=guild.id, user_id=message.author.id)
                except Exception as exc:
                    # Broad on purpose: participation tracking is best-effort and must never
                    # raise out of the listener.
                    logfire.warn(
                        "Failed to record economy participation",
                        user_id=message.author.id,
                        guild_id=guild.id,
                        error_type=type(exc).__name__,
                        _exc_info=exc,
                    )

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
