"""Process entry point: the `DiscordBot` gateway client plus the handlers no cog owns.

`main` is what the `cli` / `discordbot` console scripts declared in `pyproject.toml` run —
configure logging, read the token, build the bot, connect — so everything the bot can actually do
lives under `cogs/` and only the pieces that have to sit above them are here.

Construction order is load-bearing. `__init__` loads every cog synchronously BEFORE the gateway
connects, so the command tree is already populated when `on_ready` syncs it; that is also what
forces each cog's `setup` to be sync (`_load_cogs_sync` carries the failure mode). `on_ready`
itself re-fires on every reconnect and resume, so its body is latched and the sync, the status
loop and the price-table warm-up happen once per process.

Two globally shared behaviors live here rather than in a cog. `on_message` pays the flat
per-message economy reward — one of only two faucets in the whole bot, the other being
`/checkin` — behind a process-local per-user cooldown, and then calls `process_commands` itself,
because overriding `Bot.on_message` replaces the nextcord default that would have done it (the
same trap `cogs/usage/cog.py` documents for `on_interaction`); a cog's own
`@commands.Cog.listener()` message handler is dispatched separately and is unaffected.
`on_command_error` is the single place a common command failure becomes a user-facing embed, so a
new case is added here instead of in each cog. It and `on_command_completion` are the
prefix-command surface (`commands.Context`); a slash command's failure and its usage record go
elsewhere, to `on_application_command_error` and `cogs/usage/cog.py` respectively.

Bot presence is hardcoded in `status_task`; there is no DB-backed rotation.
"""

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
from nextcord import Game, Embed, Intents, Message
from nextcord.ext import tasks, commands

from discordbot import setup_logging
from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.config import DiscordConfig
from discordbot.typings.economy import BASE_MESSAGE_REWARD_AMOUNT, MESSAGE_REWARD_COOLDOWN_SECONDS
from discordbot.utils.model_pricing import load_model_info
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.services.economy.database import credit_with_repayment


class DiscordBot(commands.Bot):
    """The bot process itself: intents, cog loading, and the events no cog handles.

    Attributes:
        discord_config: Runtime Discord configuration. `main` reads its own copy for the token,
            and nothing in the tree reads this one.
        logger: nextcord's own `nextcord.state` logger, pinned to WARNING and routed into
            logfire so gateway chatter does not bury the bot's own records.
    """

    def __init__(self) -> None:
        """Builds the gateway client, creates the data directories, and loads every cog.

        `data/database` and `data/memories` are created up front because several cogs write into
        them and none of them owns the directory, so a first boot on an empty volume has
        somewhere to put its files.
        """
        intents = Intents.all()
        intents.members = False
        intents.presences = False
        super().__init__(
            intents=intents, help_command=None, description="A Discord bot made with Nextcord."
        )
        self.discord_config = DiscordConfig()
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
        """Drops expired message-reward cooldown entries, at most once per cooldown window.

        Every message reaches this, so the sweep is itself rate-limited rather than rebuilding
        the whole dict per message on a busy guild. `_message_reward_pruned_at` is read through
        `getattr` so a bot object that never ran `__init__` still prunes.

        Args:
            now (float): The `monotonic()` reading the caller also measures its own cooldown
                against, so both see one clock read.
        """
        if now - getattr(self, "_message_reward_pruned_at", 0.0) < MESSAGE_REWARD_COOLDOWN_SECONDS:
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

        A cog is a directory holding `cog.py`, which is the module nextcord reads
        `setup` off; everything else in the directory is that cog's own helpers.
        The scan is one level deep on purpose: a nested helper subpackage such as
        `gen_reply/link_sources/` carries an `__init__.py` too, and handing one to
        `load_extensions` raises `NoEntryPointError` and aborts boot under
        `stop_at_error=True`. A directory that is not a cog raises here rather than
        being skipped, so a half-finished move cannot silently stop loading a cog.

        Raises:
            RuntimeError: A directory under `cogs/` is missing `__init__.py` or `cog.py`.
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
        """Records the identity the gateway handed back.

        Fires on every reconnect and before the cache is ready, so it does no setup work; that
        belongs in `on_ready`. Nothing is logged while the client still has no user.
        """
        bot_user = self.user
        if bot_user is None:
            return
        logfire.info("Bot Connected", bot_name=bot_user.name, bot_id=bot_user.id)

    async def on_ready(self) -> None:
        """Runs the once-per-process startup: command sync, status loop, price-table warm-up.

        `on_ready` re-fires on every gateway reconnect/resume, so the body is gated on
        `_initial_setup_done` to keep sync + status_task.start idempotent.
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
        # Fetch the LiteLLM price table now, off the event loop, so the first
        # AI reply does not stall on a synchronous network call.
        await asyncio.to_thread(load_model_info)

        app_info = await self.application_info()
        invite_url = (
            f"https://discord.com/oauth2/authorize?client_id={app_info.id}&permissions=8&scope=bot"
        )
        logfire.info("Bot Started", bot_name=bot_user.name, bot_id=bot_user.id)
        logfire.info("Invite Link", invite_url=invite_url)

    @tasks.loop(minutes=1.0)
    async def status_task(self) -> None:
        """Sets the bot's game status, once a minute for as long as the loop runs.

        Started from `on_ready` and never stopped; the status list is hardcoded here, so there is
        nothing to configure at runtime.
        """
        statuses = ["your mama"]
        random_status = secrets.choice(statuses)
        await self.change_presence(activity=Game(random_status))
        activity = self.activity
        if activity is not None and activity.name is not None:
            logfire.debug("Status Changed", new_status=activity.name)

    @status_task.before_loop
    async def before_status_task(self) -> None:
        """Holds the status loop back until the gateway reports ready.

        `change_presence` needs a live connection, so the first iteration cannot run before one
        exists.
        """
        await self.wait_until_ready()

    async def on_message(self, message: Message) -> None:
        """Pays the flat per-message reward, then hands the message to the command processor.

        The reward is best-effort: a failure only warns, and the reserved cooldown slot is rolled
        back so a transient error does not cost the author their next window. Messages from any
        bot, this one included, are dropped before either step. `process_commands` is called here
        because this override replaces the nextcord default that normally makes that call, and
        without it no prefix command would ever run.

        Args:
            message (Message): The message that arrived in a channel the bot can see.
        """
        if message.author == self.user or message.author.bot:
            return

        now = monotonic()
        DiscordBot._prune_message_reward_cooldowns(self, now=now)
        last_rewarded_at = self._message_reward_at.get(message.author.id)
        if last_rewarded_at is None or now - last_rewarded_at >= MESSAGE_REWARD_COOLDOWN_SECONDS:
            # Reserve the cooldown slot before awaiting so two rapid messages cannot
            # both pass the check and double-credit; roll it back if the credit fails
            # so a transient error does not cost the user their reward window.
            self._message_reward_at[message.author.id] = now
            guild = getattr(message, "guild", None)
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

    async def on_command_completion(self, context: commands.Context[commands.Bot]) -> None:
        """Logs one line per prefix command that ran to completion.

        Only the first word of the qualified name is recorded, so a subcommand is counted under
        its group. Slash commands never reach this event; `cogs/usage/cog.py` records those.

        Args:
            context (commands.Context[commands.Bot]): The invocation that just finished.
        """
        command = context.command
        if command is None:
            return
        full_command_name = command.qualified_name
        split = full_command_name.split(" ")
        executed_command = str(split[0])
        logfire.info("Command Received", command=executed_command)
        if context.guild is not None:
            logfire.info(
                f"Executed {executed_command} command in {context.guild.name} (ID: {context.guild.id}) by {context.author} (ID: {context.author.id})"
            )
        else:
            logfire.info(
                f"Executed {executed_command} command by {context.author} (ID: {context.author.id}) in DMs"
            )

    async def on_command_error(
        self,
        context: commands.Context[commands.Bot],
        exception: commands.CommandOnCooldown
        | commands.NotOwner
        | commands.MissingPermissions
        | commands.BotMissingPermissions
        | commands.MissingRequiredArgument
        | commands.CommandNotFound
        | Exception,
    ) -> None:
        """Answers a failed prefix command with the shared error embed, or logs what has no branch.

        The one place a common command failure becomes user-facing text, so a cog adds a case
        here rather than growing a handler of its own. Overriding this drops nextcord's default,
        which stood down whenever the command or its cog carried a local error handler; this one
        does not, so a cog-local handler and this method both run. An unrecognised failure is
        logged rather than answered, since the user has no action to take on it.

        Args:
            context (commands.Context[commands.Bot]): The invocation that failed.
            exception (commands.CommandOnCooldown | commands.NotOwner | commands.MissingPermissions | commands.BotMissingPermissions | commands.MissingRequiredArgument | commands.CommandNotFound | Exception): The failure nextcord dispatched.
        """
        if isinstance(exception, commands.CommandOnCooldown):
            minutes, seconds = divmod(exception.retry_after, 60)
            hours, minutes = divmod(minutes, 60)
            hours = hours % 24
            embed = Embed(
                description=f"**Please slow down** - You can use this command again in {f'{round(hours)} hours' if round(hours) > 0 else ''} {f'{round(minutes)} minutes' if round(minutes) > 0 else ''} {f'{round(seconds)} seconds' if round(seconds) > 0 else ''}.",
                color=0xE02B2B,
            )
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
        elif isinstance(exception, commands.NotOwner):
            embed = Embed(description="You are not the owner of the bot!", color=0xE02B2B)
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
            logfire.info(
                "Owner-only command refused",
                command=context.command.qualified_name if context.command else None,
                author=str(context.author),
                author_id=context.author.id,
                guild_id=context.guild.id if context.guild else None,
            )
        elif isinstance(exception, commands.MissingPermissions):
            embed = Embed(
                description="You are missing the permission(s) `"
                + ", ".join(exception.missing_permissions)
                + "` to execute this command!",
                color=0xE02B2B,
            )
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
        elif isinstance(exception, commands.BotMissingPermissions):
            embed = Embed(
                description="I am missing the permission(s) `"
                + ", ".join(exception.missing_permissions)
                + "` to fully perform this command!",
                color=0xE02B2B,
            )
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
        elif isinstance(exception, commands.MissingRequiredArgument):
            embed = Embed(
                title="Error!",
                # Capitalized here: the argument name opens the message and is lowercase in code.
                description=str(exception).capitalize(),
                color=0xE02B2B,
            )
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
        elif isinstance(exception, commands.CommandNotFound):
            embed = Embed(
                title="Error!",
                description=f"Command {exception.command_name} not found",
                color=0xE02B2B,
            )
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
        else:
            # nextcord wraps a command-body failure in CommandInvokeError, so report
            # the unwrapped type to name the real defect.
            original = getattr(exception, "original", exception)
            logfire.error(
                "Unhandled command error",
                command=context.command.qualified_name if context.command else None,
                guild_id=context.guild.id if context.guild else None,
                author_id=context.author.id,
                error_type=type(original).__name__,
                _exc_info=exception,
            )


def main() -> None:
    """Configures logging, then runs the bot until the connection is closed.

    The console-script entry point for `cli` and `discordbot`. The token is read through its own
    `DiscordConfig`, so a deployment missing `DISCORD_BOT_TOKEN` fails with a pydantic
    `ValidationError` here, before the bot is built and before the gateway is contacted.
    `bot.run` owns the event loop and does not return while the bot is connected.
    """
    setup_logging()
    discord_config = DiscordConfig()
    bot = DiscordBot()
    bot.run(token=discord_config.discord_bot_token)


if __name__ == "__main__":
    main()
