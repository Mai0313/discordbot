"""Discord bot entry point and runtime event handlers."""

import os
from time import monotonic
import asyncio
import logging
from pathlib import Path
import secrets
import platform

import logfire
from logfire import LogfireLoggingHandler
import nextcord
from nextcord import Game, Intents, Message, Interaction
from nextcord.ext import tasks, commands
from nextcord.errors import ApplicationError

from discordbot import setup_logging
from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.config import DiscordConfig
from discordbot.typings.economy import BASE_MESSAGE_REWARD_AMOUNT, MESSAGE_REWARD_COOLDOWN_SECONDS
from discordbot.utils.model_pricing import MODEL_INFO_REFRESH_MINUTES, refresh_model_info
from discordbot.services.economy.database import credit_with_repayment


class DiscordBot(commands.Bot):
    """Discord bot configured with project-specific intents and cogs.

    Attributes:
        logger: Logger used by Nextcord state events.
    """

    def __init__(self) -> None:
        """Initialises the Discord bot with specific intents and configuration."""
        intents = Intents.all()
        intents.members = False
        intents.presences = False
        super().__init__(
            intents=intents, help_command=None, description="A Discord bot made with Nextcord."
        )
        self.logger = logging.getLogger("nextcord.state")
        self.logger.setLevel(logging.WARNING)
        self.logger.addHandler(LogfireLoggingHandler())
        Path("./data/database").mkdir(parents=True, exist_ok=True)
        Path("./data/memories").mkdir(parents=True, exist_ok=True)
        # Cogs are loaded synchronously so application_commands is populated
        # before the gateway connects. Each cog's setup() must also be sync:
        # load_extension fires async setups via asyncio.create_task() without
        # awaiting, so an async setup would still be pending when on_ready
        # triggers sync_all_application_commands(), making the first sync see
        # zero commands and register nothing with Discord.
        self._load_cogs_sync()
        self._initial_setup_done = False
        # Process-local per-user cooldown for the flat message reward, so it
        # cannot be farmed by spamming. Resets on restart by design.
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
        """Loads every cog directory under `cogs/`.

        A cog is a directory holding both `__init__.py` and `cog.py`, the latter being
        the module nextcord reads `setup` off; everything else in the directory is that
        cog's own helpers. The scan is one level deep on purpose: a nested helper
        subpackage such as `gen_reply/link_sources/` carries an `__init__.py` too, and
        handing one to `load_extensions` raises `NoEntryPointError` and aborts boot
        under `stop_at_error=True`. A `_`-prefixed entry is skipped, which is what keeps
        `__pycache__` from raising; anything else that is not a cog raises here rather
        than being skipped, so a half-finished move cannot silently stop loading a cog.
        """
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

    async def on_connect(self) -> None:
        """Called when the bot has successfully connected to Discord."""
        bot_user = self.user
        if bot_user is None:
            return
        logfire.info("Bot Connected", bot_name=bot_user.name, bot_id=bot_user.id)

    async def on_ready(self) -> None:
        """Called when the bot is ready; performs first-time-only setup.

        `on_ready` re-fires on every gateway reconnect/resume, so the body
        is gated on `_initial_setup_done` to keep sync + the task starts
        idempotent.
        """
        if self._initial_setup_done:
            return
        bot_user = self.user
        if bot_user is None:
            # Never expected once on_ready fires; leave the latch unset so a reconnect retries.
            logfire.warn("on_ready fired without a logged-in user; skipping initial setup")
            return
        self._initial_setup_done = True

        logfire.info(
            "Logged in",
            bot_name=bot_user.name,
            discord_version=nextcord.__version__,
            python_version=platform.python_version(),
            system=f"{platform.system()} {platform.release()} ({os.name})",
        )

        await self.sync_all_application_commands()
        self.status_task.start()
        self.price_table_task.start()

        app_info = await self.application_info()
        invite_url = (
            f"https://discord.com/oauth2/authorize?client_id={app_info.id}&permissions=8&scope=bot"
        )
        logfire.info("Bot Started", bot_name=bot_user.name, bot_id=bot_user.id)
        logfire.info("Invite Link", invite_url=invite_url)

    @tasks.loop(minutes=1.0)
    async def status_task(self) -> None:
        """Periodically updates the bot's game status."""
        statuses = ["your mama"]
        random_status = secrets.choice(statuses)
        await self.change_presence(activity=Game(random_status))
        activity = self.activity
        if activity is not None and activity.name is not None:
            logfire.debug("Status Changed", new_status=activity.name)

    @status_task.before_loop
    async def before_status_task(self) -> None:
        """Ensures the bot is ready before starting the status task."""
        await self.wait_until_ready()

    @tasks.loop(minutes=MODEL_INFO_REFRESH_MINUTES)
    async def price_table_task(self) -> None:
        """Loads the LiteLLM price table, and keeps re-checking upstream while it is degraded.

        `tasks.Loop` runs its body immediately, so the first pass is the warm-up that keeps
        the first AI reply from stalling on a synchronous network call; every later one is
        a branch once upstream has served. Off the event loop for the same reason the
        warm-up always was.
        """
        try:
            await asyncio.to_thread(refresh_model_info)
        except Exception as exc:
            # Broad because what must not happen here is the loop stopping, whatever the
            # reason: `Loop._loop` re-raises after one failed iteration, and its default
            # error hook never reports it — `_call_loop_function` prepends `_injected`, so
            # the bound `_error` gets an argument too many and dies first. A raise would
            # take recovery down for the life of the process with no line anywhere.
            logfire.error("model price table refresh failed; retrying next pass", _exc_info=exc)

    async def on_message(self, message: Message) -> None:
        """Awards the cooldown-gated message reward, then dispatches commands.

        This is the only faucet that pays an action reward; it is best-effort, so
        command dispatch runs whether or not the credit lands.

        Args:
            message: The message that was sent.
        """
        if message.author == self.user or message.author.bot:
            return

        now = monotonic()
        self._prune_message_reward_cooldowns(now=now)
        last_rewarded_at = self._message_reward_at.get(message.author.id)
        if last_rewarded_at is None or now - last_rewarded_at >= MESSAGE_REWARD_COOLDOWN_SECONDS:
            # Reserve the cooldown slot before awaiting so two rapid messages cannot
            # both pass the check and double-credit; roll it back if the credit fails
            # so a transient error does not cost the user their reward window.
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
                # Broad on purpose: the reward is best-effort and must never stop
                # process_commands, so every failure mode (DB, avatar fetch) is swallowed.
                logfire.warn(
                    "Failed to award base message points",
                    user_id=message.author.id,
                    guild_id=guild.id if guild else None,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
        await self.process_commands(message)

    async def on_application_command_error(
        self, interaction: Interaction[commands.Bot], exception: ApplicationError
    ) -> None:
        """Records a slash command that raised, which nothing else in this process does.

        This bot has no prefix commands at all, so the pair of `on_command_*` handlers that
        used to sit here could never fire: `command_prefix` is never passed to
        `commands.Bot`, nextcord then defaults it to `()`, and `get_context`'s
        `content.startswith(())` is False for every message, so `invoke` never reaches a
        command to dispatch either event from.

        What DOES fire is this one, and until now nothing overrode it: nextcord's default
        prints the traceback to `sys.stderr`, while `_TeeStream` tees only `sys.stdout` into
        `./data/logs`, so a failing slash command left no line in the file this project is
        debugged from. Logging only, deliberately — a cog that wants to tell the user
        something answers its own interaction, and an unanswered one already shows Discord's
        own failure notice.
        """
        # nextcord wraps a command-body failure in ApplicationInvokeError, so report the
        # unwrapped type to name the real defect.
        original = getattr(exception, "original", exception)
        command = interaction.application_command
        logfire.error(
            "Unhandled application command error",
            command=command.qualified_name if command is not None else None,
            guild_id=interaction.guild_id,
            user_id=interaction.user.id if interaction.user is not None else None,
            error_type=type(original).__name__,
            _exc_info=exception,
        )


def main() -> None:
    """Initialises and runs the Discord bot."""
    setup_logging()
    discord_config = DiscordConfig()
    bot = DiscordBot()
    bot.run(token=discord_config.discord_bot_token)


if __name__ == "__main__":
    main()
