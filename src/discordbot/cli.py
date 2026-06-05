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
from nextcord import Game, Embed, Intents, Message
from nextcord.ext import tasks, commands

from discordbot import setup_logging
from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.config import DiscordConfig
from discordbot.typings.economy import BASE_MESSAGE_REWARD_AMOUNT, MESSAGE_REWARD_COOLDOWN_SECONDS
from discordbot.utils.model_pricing import warm_pricing_cache
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.cogs._economy.database import get_bot_statuses, credit_with_repayment


class DiscordBot(commands.Bot):
    """Discord bot configured with project-specific intents and cogs.

    Attributes:
        discord_config: Runtime Discord configuration loaded from settings.
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
        """Drops expired message-reward cooldown entries."""
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
        """Loads all cogs found in the cogs directory."""
        cog_dir = Path(__file__).parent / "cogs"
        cog_files = [
            f"discordbot.cogs.{f.stem}"
            for f in cog_dir.glob("*.py")
            if not f.stem.startswith("__")
        ]
        self.load_extensions(cog_files, stop_at_error=True)
        logfire.info("Cogs Loaded", cogs=cog_files)

    async def on_connect(self) -> None:
        """Called when the bot has successfully connected to Discord."""
        logfire.info("Bot Connected", bot_name=self.user.name, bot_id=self.user.id)

    async def on_ready(self) -> None:
        """Called when the bot is ready; performs first-time-only setup.

        `on_ready` re-fires on every gateway reconnect/resume, so the body
        is gated on `_initial_setup_done` to keep sync + status_task.start
        idempotent.
        """
        if self._initial_setup_done:
            return
        self._initial_setup_done = True

        logfire.info(
            "Logged in",
            bot_name=self.user.name,
            discord_version=nextcord.__version__,
            python_version=platform.python_version(),
            system=f"{platform.system()} {platform.release()} ({os.name})",
        )

        await self.sync_all_application_commands()
        self.status_task.start()
        # Fetch the LiteLLM price table now, off the event loop, so the first
        # AI reply does not stall on a synchronous network call.
        await asyncio.to_thread(warm_pricing_cache)

        app_info = await self.application_info()
        invite_url = (
            f"https://discord.com/oauth2/authorize?client_id={app_info.id}&permissions=8&scope=bot"
        )
        logfire.info("Bot Started", bot_name=self.user.name, bot_id=self.user.id)
        logfire.info("Invite Link", invite_url=invite_url)

    @tasks.loop(minutes=1.0)
    async def status_task(self) -> None:
        """Periodically updates the bot's game status."""
        statuses = await get_bot_statuses() or ["your mama"]
        random_status = secrets.choice(statuses)
        await self.change_presence(activity=Game(random_status))
        logfire.info("Status Changed", new_status=self.activity.name)

    @status_task.before_loop
    async def before_status_task(self) -> None:
        """Ensures the bot is ready before starting the status task."""
        await self.wait_until_ready()

    async def on_message(self, message: Message) -> None:
        """Handles incoming messages.

        Args:
            message: The message that was sent.
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
            try:
                avatar_url = await guild_avatar_url(
                    user=message.author, guild=getattr(message, "guild", None)
                )
                await credit_with_repayment(
                    user_id=message.author.id,
                    name=message.author.name,
                    avatar_url=avatar_url,
                    amount=BASE_MESSAGE_REWARD_AMOUNT,
                )
            except Exception:
                if last_rewarded_at is None:
                    self._message_reward_at.pop(message.author.id, None)
                else:
                    self._message_reward_at[message.author.id] = last_rewarded_at
                logfire.warn("Failed to award base message points", _exc_info=True)
        await self.process_commands(message)

    async def on_command_completion(self, context: commands.Context) -> None:
        """Handles successful command execution.

        Args:
            context: The context of the command that was executed.
        """
        full_command_name = context.command.qualified_name
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
        context: commands.Context,
        error: commands.CommandOnCooldown
        | commands.NotOwner
        | commands.MissingPermissions
        | commands.BotMissingPermissions
        | commands.MissingRequiredArgument
        | commands.CommandNotFound
        | Exception,
    ) -> None:
        """Handles command errors.

        Args:
            context: The context of the command that failed.
            error: The exception that was raised.
        """
        if isinstance(error, commands.CommandOnCooldown):
            minutes, seconds = divmod(error.retry_after, 60)
            hours, minutes = divmod(minutes, 60)
            hours = hours % 24
            embed = Embed(
                description=f"**Please slow down** - You can use this command again in {f'{round(hours)} hours' if round(hours) > 0 else ''} {f'{round(minutes)} minutes' if round(minutes) > 0 else ''} {f'{round(seconds)} seconds' if round(seconds) > 0 else ''}.",
                color=0xE02B2B,
            )
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
        elif isinstance(error, commands.NotOwner):
            embed = Embed(description="You are not the owner of the bot!", color=0xE02B2B)
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
            if context.guild:
                logfire.warn(
                    f"{context.author} (ID: {context.author.id}) tried to execute an owner only command in the guild {context.guild.name} (ID: {context.guild.id}), but the user is not an owner of the bot."
                )
            else:
                logfire.warn(
                    f"{context.author} (ID: {context.author.id}) tried to execute an owner only command in the bot's DMs, but the user is not an owner of the bot."
                )
        elif isinstance(error, commands.MissingPermissions):
            embed = Embed(
                description="You are missing the permission(s) `"
                + ", ".join(error.missing_permissions)
                + "` to execute this command!",
                color=0xE02B2B,
            )
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
        elif isinstance(error, commands.BotMissingPermissions):
            embed = Embed(
                description="I am missing the permission(s) `"
                + ", ".join(error.missing_permissions)
                + "` to fully perform this command!",
                color=0xE02B2B,
            )
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
        elif isinstance(error, commands.MissingRequiredArgument):
            embed = Embed(
                title="Error!",
                # We need to capitalize because the command arguments have no capital letter in the code and they are the first word in the error message.
                description=str(error).capitalize(),
                color=0xE02B2B,
            )
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
        elif isinstance(error, commands.CommandNotFound):
            embed = Embed(
                title="Error!",
                description=f"Command {error.command_name} not found",
                color=0xE02B2B,
            )
            await context.send(
                embed=embed, **embed_spacer_payload(embeds=[embed], is_edit=False, target=context)
            )
        else:
            pass


def main() -> None:
    """Initialises and runs the Discord bot."""
    setup_logging()
    discord_config = DiscordConfig()
    bot = DiscordBot()
    bot.run(token=discord_config.discord_bot_token)


if __name__ == "__main__":
    main()
